"""
Gemini Client for parallel shard adaptation.

Uses Gemini 2.5 Flash with:
- Global Factsheet for consistency
- Statistics tracking
- Retry with exponential backoff
- LangSmith tracing (optional)

PROMPTS: All prompts are now in src/utils/prompts.py for easier tracking.
"""
import os
import json
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .llm_stats import get_stats, StatsTimer
from .retry_handler import retry_with_backoff, RetryConfig
from .prompts import (
    build_factsheet_prompt,
    build_shard_adaptation_prompt,
    build_regeneration_prompt,
    strict_verify_output,
    VerificationResult,
    extract_adapted_content,
)

logger = logging.getLogger(__name__)


def sanitize_for_json(obj: Any) -> Any:
    """
    Recursively sanitize an object for JSON encoding.
    Removes surrogate characters that cause UTF-8 encoding errors.
    """
    if isinstance(obj, str):
        # Remove surrogate characters (U+D800 to U+DFFF)
        try:
            # Try encoding to catch surrogates
            obj.encode('utf-8')
            return obj
        except UnicodeEncodeError:
            # Remove surrogates by encoding with surrogateescape then replacing
            return obj.encode('utf-16', errors='surrogatepass').decode('utf-16', errors='replace')
    elif isinstance(obj, dict):
        return {sanitize_for_json(k): sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    return obj


def safe_json_dumps(obj: Any, **kwargs) -> str:
    """Safely dump object to JSON, handling surrogate characters."""
    sanitized = sanitize_for_json(obj)
    return json.dumps(sanitized, ensure_ascii=False, **kwargs)


def safe_str(s: str) -> str:
    """Sanitize a string to remove surrogate characters."""
    if not isinstance(s, str):
        return str(s) if s else ""
    try:
        return s.encode('utf-8', errors='surrogatepass').decode('utf-8', errors='replace')
    except Exception:
        return ''.join(c if ord(c) < 128 else '?' for c in s)


# Try to import langsmith for tracing
try:
    from langsmith import traceable
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    def traceable(func):
        return func  # No-op decorator

_gemini_model = None
_executor = ThreadPoolExecutor(max_workers=10)

# =============================================================================
# POISON LIST FILTERING - Remove common English words to reduce false positives
# =============================================================================

# Common English words that should NOT be treated as poison terms
# These are too generic and appear in any business context
COMMON_ENGLISH_WORDS = {
    # Common business words
    "role", "roles", "team", "teams", "work", "working", "job", "jobs",
    "process", "processes", "project", "projects", "task", "tasks",
    "goal", "goals", "target", "targets", "result", "results",
    "consistent", "consistency", "consistently", "ensure", "ensuring",
    "equitable", "equity", "fair", "fairly", "accurate", "accurately",
    "efficient", "efficiently", "effective", "effectively",
    "quality", "improve", "improved", "improvement", "improving",
    "evaluate", "evaluation", "evaluations", "evaluating",
    "assess", "assessment", "assessments", "assessing",
    "analysis", "analyze", "analyzing", "analytical",
    "decision", "decisions", "decide", "deciding",
    "strategy", "strategic", "strategies",
    "plan", "planning", "plans", "planned",
    "manage", "management", "managing", "manager", "managers",
    "lead", "leader", "leadership", "leading",
    "support", "supporting", "supports",
    "develop", "development", "developing",
    "review", "reviewing", "reviews", "reviewed",
    # Common adjectives
    "best", "better", "good", "great", "important", "key", "main",
    "new", "first", "last", "next", "previous", "current",
    "high", "low", "top", "bottom",
    # Common verbs
    "make", "making", "take", "taking", "give", "giving",
    "use", "using", "used", "find", "finding", "found",
    "get", "getting", "see", "seeing", "look", "looking",
    "know", "knowing", "think", "thinking",
    # Time words
    "time", "day", "week", "month", "year", "today", "tomorrow",
    # Common nouns
    "way", "ways", "thing", "things", "part", "parts",
    "person", "people", "group", "groups",
    "data", "information", "report", "reports",
    "meeting", "meetings", "email", "emails",
    # Numbers/amounts
    "one", "two", "three", "many", "few", "all", "some", "most",
}


def filter_poison_list(poison_list: list) -> list:
    """
    Filter out common English words from the poison list.

    Only keeps terms that are truly scenario-specific:
    - Proper nouns (company names, person names)
    - Industry-specific jargon
    - Branded terms

    Returns filtered list with only scenario-specific terms.
    """
    if not isinstance(poison_list, list):
        return []

    filtered = []
    for term in poison_list:
        if not isinstance(term, str):
            continue
        term_lower = term.lower().strip()

        # Skip empty terms
        if not term_lower:
            continue

        # Skip single-character terms
        if len(term_lower) <= 2:
            continue

        # Skip common English words (case-insensitive)
        if term_lower in COMMON_ENGLISH_WORDS:
            logger.debug(f"Filtered common word from poison list: {term}")
            continue

        # Keep multi-word terms (likely proper nouns/brand names)
        # Keep terms with capital letters (likely proper nouns)
        # Keep terms not in common words list
        filtered.append(term)

    logger.info(f"Poison list filtered: {len(poison_list)} -> {len(filtered)} terms")
    return filtered


def extract_names_from_text(text: str) -> list[str]:
    """
    Extract person names from text using pattern matching.

    Looks for patterns like:
    - "Name Name" (two capitalized words)
    - Common name patterns in business contexts

    Returns list of extracted names.
    """
    import re

    names = set()

    # Pattern for "First Last" names (two capitalized words)
    name_pattern = r'\b([A-Z][a-z]+)\s+([A-Z][a-z]+)\b'
    matches = re.findall(name_pattern, text)
    for first, last in matches:
        # Skip common phrases that match pattern
        skip_phrases = {'The Company', 'Market Entry', 'Key Learning', 'Learning Outcome',
                       'Business Case', 'Project Manager', 'Team Lead', 'Data Analysis'}
        full_name = f"{first} {last}"
        if full_name not in skip_phrases:
            names.add(full_name)
            names.add(first)  # Also add first name alone

    return list(names)


def augment_poison_list_with_names(poison_list: list, source_text: str) -> list:
    """
    Augment poison list with any person names found in source scenario.

    This ensures names like "Elizabeth Carter" are always included even if
    the LLM misses them in the factsheet extraction.

    Only extracts PERSON NAMES using regex - no hardcoded domain terms.
    Domain-specific terms should come from the LLM factsheet extraction.
    """
    extracted_names = extract_names_from_text(source_text)

    existing_lower = {t.lower() for t in poison_list if isinstance(t, str)}

    added = []
    for name in extracted_names:
        if name.lower() not in existing_lower:
            poison_list.append(name)
            added.append(name)

    if added:
        logger.info(f"[POISON LIST] Auto-extracted {len(added)} names from source: {added[:10]}...")

    return poison_list


def _get_gemini():
    """Lazy initialize Gemini model."""
    global _gemini_model
    if _gemini_model is None:
        try:
            import google.generativeai as genai

            api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY not set")

            genai.configure(api_key=api_key)
            _gemini_model = genai.GenerativeModel("gemini-2.5-flash")
            logger.info("Gemini 2.5 Flash initialized")
        except ImportError:
            raise ImportError("google-generativeai required. Install: pip install google-generativeai")
    return _gemini_model


def _repair_json(text: str) -> dict:
    """Repair malformed JSON using json-repair library."""
    import re

    text = text.strip()

    # Remove markdown code blocks
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1])
        else:
            text = "\n".join(lines[1:])

    def fix_escapes(s):
        s = re.sub(r'\\([^"\\/bfnrtu])', r'\\\\\1', s)
        return s

    text = fix_escapes(text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    try:
        from json_repair import repair_json
        repaired = repair_json(text, return_objects=True)
        logger.info("JSON repaired successfully using json-repair")
        return repaired
    except Exception as e:
        logger.warning(f"json-repair failed: {e}, trying manual fix")

    text = re.sub(r',(\s*[}\]])', r'\1', text)

    open_braces = text.count('{') - text.count('}')
    open_brackets = text.count('[') - text.count(']')
    if open_braces > 0:
        text += '}' * open_braces
    if open_brackets > 0:
        text += ']' * open_brackets

    return json.loads(text)


def _call_gemini_sync(prompt: str, temperature: float = 0.3, max_tokens: int = 65536) -> dict:
    """Synchronous Gemini call."""
    import time
    start = time.time()

    model = _get_gemini()
    init_time = time.time() - start
    logger.info(f"[GEMINI] Model init: {init_time:.2f}s, prompt: {len(prompt)} chars, max_tokens: {max_tokens}")

    call_start = time.time()
    response = model.generate_content(
        prompt,
        generation_config={
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "response_mime_type": "application/json",
        }
    )
    call_time = time.time() - call_start
    logger.info(f"[GEMINI] API call took: {call_time:.2f}s")

    text = response.text.strip()
    logger.info(f"[GEMINI] Response: {len(text)} chars")

    try:
        return _repair_json(text)
    except Exception as e:
        logger.error(f"JSON parse error: {e}")
        logger.error(f"Raw response (first 500 chars): {text[:500]}")
        raise


async def call_gemini(prompt: str, temperature: float = 0.3, max_tokens: int = 65536) -> dict:
    """Async Gemini call with retry."""
    loop = asyncio.get_event_loop()

    async def _call():
        return await loop.run_in_executor(
            _executor,
            lambda: _call_gemini_sync(prompt, temperature, max_tokens)
        )

    config = RetryConfig(
        max_attempts=2,
        min_wait=1.0,
        max_wait=10.0,
        multiplier=1.5,
        jitter=True
    )

    stats = get_stats()

    def on_retry(attempt, exc, wait_time):
        stats.add_retry(wait_time, is_rate_limit="rate" in str(exc).lower())

    return await retry_with_backoff(_call, config=config, on_retry=on_retry)


# =============================================================================
# GLOBAL FACTSHEET EXTRACTION
# =============================================================================

@traceable
async def extract_global_factsheet(
    source_scenario: str,
    target_scenario: str,
) -> dict[str, Any]:
    """
    Extract global facts BEFORE parallel shard processing.

    Args:
        source_scenario: Original scenario text
        target_scenario: Target scenario text

    Returns:
        Global factsheet dict
    """
    # Truncate scenarios to reasonable size - full text isn't needed for factsheet
    # Keeps first 2000 chars which contains the key info
    safe_source = safe_str(source_scenario)[:2000]
    safe_target = safe_str(target_scenario)[:2000]

    # Build prompt from prompts.py
    prompt = build_factsheet_prompt(safe_source, safe_target)
    logger.info(f"Factsheet prompt size: {len(prompt)} chars")

    with StatsTimer("global_factsheet") as timer:
        try:
            # Increased max_tokens to ensure complete factsheet
            result = await call_gemini(prompt, temperature=0.2, max_tokens=8192)
            get_stats().add_call(
                success=True,
                shard_id="global_factsheet",
                elapsed_time=timer.elapsed
            )
            if isinstance(result, list) and len(result) > 0:
                result = result[0] if isinstance(result[0], dict) else {}
            if not isinstance(result, dict):
                logger.warning(f"Factsheet returned unexpected type: {type(result)}")
                result = {}

            # === VALIDATE FACTSHEET COMPLETENESS ===
            required_fields = ['company', 'reporting_manager', 'poison_list', 'industry_context', 'learner_role']
            missing = [f for f in required_fields if not result.get(f)]
            if missing:
                logger.warning(f"[FACTSHEET] Missing required fields: {missing}")
                # Fill in defaults for missing fields
                if 'reporting_manager' not in result or not result.get('reporting_manager'):
                    result['reporting_manager'] = {
                        'name': 'Unknown Manager',
                        'role': 'Director',
                        'email': 'manager@company.com',
                        'gender': 'Unknown'
                    }
                if 'poison_list' not in result or not result.get('poison_list'):
                    result['poison_list'] = []
                if 'industry_context' not in result or not result.get('industry_context'):
                    result['industry_context'] = {'kpis': [], 'terminology': [], 'wrong_terms': []}
                if 'learner_role' not in result or not result.get('learner_role'):
                    result['learner_role'] = {'role': 'Analyst', 'key_responsibilities': []}

            # Filter poison list to remove common English words (reduces false positives)
            raw_poison_list = result.get('poison_list', [])
            raw_count = len(raw_poison_list) if isinstance(raw_poison_list, list) else 0
            filtered_poison_list = filter_poison_list(raw_poison_list)

            # CRITICAL: Augment poison list with names extracted from source scenario
            # This ensures person names like "Elizabeth Carter" are always included
            augmented_poison_list = augment_poison_list_with_names(filtered_poison_list, source_scenario)
            result['poison_list'] = augmented_poison_list

            manager_obj = result.get('reporting_manager', {})
            manager_name = manager_obj.get('name', 'Unknown') if isinstance(manager_obj, dict) else 'Unknown'
            logger.info(f"[FACTSHEET] ✅ Extracted: {raw_count} poison terms -> {len(augmented_poison_list)} after filtering+augmenting, manager={manager_name}")
            return result
        except Exception as e:
            get_stats().add_call(
                success=False,
                shard_id="global_factsheet",
                elapsed_time=timer.elapsed
            )
            logger.error(f"Failed to extract factsheet: {e}")
            return {
                "company": {"name": "Unknown", "industry": "Unknown"},
                "financials": {},
                "products": {},
                "context": {},
                "poison_list": [],
                "replacement_hints": {}
            }


# =============================================================================
# SHARD ADAPTATION WITH FACTSHEET
# =============================================================================

@traceable
async def adapt_shard_content(
    shard_id: str,
    shard_name: str,
    content: dict,
    source_scenario: str,
    target_scenario: str,
    global_factsheet: dict = None,
    rag_context: str = "",
    max_verification_retries: int = 2,
) -> tuple[dict, dict]:
    """
    Adapt a single shard's content using the global factsheet.

    NOW WITH VERIFICATION LOOP:
    1. Generate adapted content
    2. Run strict_verify_output() to catch issues
    3. If verification fails, regenerate with specific feedback
    4. Repeat until pass or max retries

    Args:
        shard_id: Shard identifier
        shard_name: Human-readable name
        content: Shard content to transform
        source_scenario: Original scenario text
        target_scenario: Target scenario text
        global_factsheet: Pre-extracted global facts (CRITICAL for consistency)
        rag_context: Additional context from RAG
        max_verification_retries: Max regeneration attempts on verification failure

    Returns:
        (adapted_content, entity_mappings)
    """
    safe_source = safe_str(source_scenario)[:200]
    safe_target = safe_str(target_scenario)[:200]
    safe_rag = safe_str(rag_context) if rag_context else ""
    factsheet = global_factsheet if isinstance(global_factsheet, dict) else {}

    # Build prompt using prompts.py (centralized prompt management)
    prompt = build_shard_adaptation_prompt(
        shard_id=shard_id,
        shard_name=shard_name,
        content=content,
        source_scenario=safe_source,
        target_scenario=safe_target,
        global_factsheet=factsheet,
        rag_context=safe_rag,
    )

    with StatsTimer(shard_id) as timer:
        try:
            # === PASS 1: Initial generation ===
            result = await call_gemini(prompt, temperature=0.3)

            if isinstance(result, list) and len(result) > 0:
                result = result[0] if isinstance(result[0], dict) else {}
            if not isinstance(result, dict):
                logger.warning(f"Shard {shard_id} returned unexpected type: {type(result)}")
                return content, {}

            # === VERIFICATION GATE ===
            verification = strict_verify_output(result, factsheet, check_verification_fields=True)

            retry_count = 0
            while not verification.passed and retry_count < max_verification_retries:
                retry_count += 1
                logger.warning(f"[VERIFY] Shard {shard_id} failed verification (attempt {retry_count}): {verification.errors[:3]}")

                # Build feedback from verification errors
                feedback = {
                    "failed_rules": verification.errors,
                    "critical_issues": verification.errors,
                    "poison_terms_found": [e for e in verification.errors if "POISON" in e],
                    "placeholders_found": [e for e in verification.errors if "PLACEHOLDER" in e],
                    "sender_mismatches": [e for e in verification.errors if "SENDER" in e],
                }

                # Regenerate with specific feedback
                regen_prompt = build_regeneration_prompt(
                    shard_id=shard_id,
                    shard_name=shard_name,
                    content=result.get("adapted_content", result) if isinstance(result, dict) else result,
                    source_scenario=safe_source,
                    target_scenario=safe_target,
                    global_factsheet=factsheet,
                    feedback=feedback,
                )

                result = await call_gemini(regen_prompt, temperature=0.2)

                if isinstance(result, list) and len(result) > 0:
                    result = result[0] if isinstance(result[0], dict) else {}
                if not isinstance(result, dict):
                    break

                # Re-verify
                verification = strict_verify_output(result, factsheet, check_verification_fields=True)

            # Log final verification status
            if verification.passed:
                logger.info(f"[VERIFY] ✅ Shard {shard_id} passed verification" + (f" after {retry_count} retries" if retry_count > 0 else ""))
            else:
                logger.warning(f"[VERIFY] ⚠️ Shard {shard_id} failed after {retry_count} retries: {verification.errors[:2]}")

            # Extract content (handles both formats)
            adapted_content = extract_adapted_content(result)
            entity_mappings = result.get("entity_mappings", {}) if isinstance(result, dict) else {}
            if not isinstance(entity_mappings, dict):
                entity_mappings = {}

            get_stats().add_call(
                success=True,
                shard_id=shard_id,
                elapsed_time=timer.elapsed
            )

            logger.debug(f"Shard {shard_id} adapted with {len(entity_mappings)} mappings")
            return adapted_content, entity_mappings

        except json.JSONDecodeError as e:
            get_stats().add_call(success=False, shard_id=shard_id, elapsed_time=timer.elapsed)
            logger.error(f"Invalid JSON for shard {shard_id}: {e}")
            raise ValueError(f"Invalid JSON from Gemini: {e}")
        except Exception as e:
            get_stats().add_call(success=False, shard_id=shard_id, elapsed_time=timer.elapsed)
            logger.error(f"Gemini error for shard {shard_id}: {e}")
            raise


# =============================================================================
# REGENERATION WITH FEEDBACK
# =============================================================================

@traceable
async def regenerate_shard_with_feedback(
    shard_id: str,
    shard_name: str,
    content: dict,
    source_scenario: str,
    target_scenario: str,
    global_factsheet: dict,
    feedback: dict,
) -> tuple[dict, dict]:
    """
    Regenerate a shard using feedback from failed alignment rules.

    Args:
        shard_id: Shard identifier
        shard_name: Human-readable name
        content: Current shard content (to fix)
        source_scenario: Original scenario
        target_scenario: Target scenario
        global_factsheet: Global facts
        feedback: Dict with failed_rules, critical_issues, suggestions

    Returns:
        (regenerated_content, entity_mappings)
    """
    safe_source = safe_str(source_scenario)
    safe_target = safe_str(target_scenario)

    # Build prompt using prompts.py (centralized prompt management)
    prompt = build_regeneration_prompt(
        shard_id=shard_id,
        shard_name=shard_name,
        content=content,
        source_scenario=safe_source,
        target_scenario=safe_target,
        global_factsheet=global_factsheet,
        feedback=feedback,
    )

    with StatsTimer(f"regen_{shard_id}") as timer:
        try:
            result = await call_gemini(prompt, temperature=0.2)

            if isinstance(result, list) and len(result) > 0:
                result = result[0] if isinstance(result[0], dict) else {}
            if not isinstance(result, dict):
                return content, {}

            adapted_content = result.get("adapted_content", content)
            entity_mappings = result.get("entity_mappings", {})
            fixes_applied = result.get("fixes_applied", [])

            get_stats().add_call(
                success=True,
                shard_id=f"regen_{shard_id}",
                elapsed_time=timer.elapsed
            )

            logger.info(f"Regenerated {shard_id} with {len(fixes_applied)} fixes: {fixes_applied[:3]}")
            return adapted_content, entity_mappings

        except Exception as e:
            get_stats().add_call(success=False, shard_id=f"regen_{shard_id}", elapsed_time=timer.elapsed)
            logger.error(f"Regeneration failed for {shard_id}: {e}")
            return content, {}


async def regenerate_shards_with_feedback(
    shards: list,
    global_factsheet: dict,
    feedback: dict,
    focus_shards: list = None,
) -> list:
    """
    Regenerate multiple shards in parallel using feedback.

    Args:
        shards: List of shard objects
        global_factsheet: Global facts
        feedback: Alignment feedback
        focus_shards: Optional list of shard IDs to focus on

    Returns:
        List of regenerated shards
    """
    if focus_shards:
        target_shards = [s for s in shards if hasattr(s, 'id') and s.id in focus_shards]
    else:
        target_shards = shards

    if not target_shards:
        logger.info("No shards to regenerate")
        return shards

    logger.info(f"Regenerating {len(target_shards)} shards with feedback...")

    tasks = []
    for shard in target_shards:
        shard_id = shard.id if hasattr(shard, 'id') else str(shard)
        shard_name = shard.name if hasattr(shard, 'name') else shard_id
        shard_content = shard.content if hasattr(shard, 'content') else shard
        task = regenerate_shard_with_feedback(
            shard_id=shard_id,
            shard_name=shard_name,
            content=shard_content,
            source_scenario=global_factsheet.get("source_scenario", "") if isinstance(global_factsheet, dict) else "",
            target_scenario=global_factsheet.get("target_scenario", "") if isinstance(global_factsheet, dict) else "",
            global_factsheet=global_factsheet,
            feedback=feedback,
        )
        tasks.append((shard, task))

    results = await asyncio.gather(*[t[1] for t in tasks], return_exceptions=True)

    for i, (shard, _) in enumerate(tasks):
        result = results[i]
        shard_id = shard.id if hasattr(shard, 'id') else str(shard)
        if isinstance(result, Exception):
            logger.error(f"Regeneration failed for {shard_id}: {result}")
        else:
            adapted_content, entity_map = result
            shard.content = adapted_content
            shard.current_hash = ""

    logger.info(f"Regeneration complete for {len(target_shards)} shards")
    return shards


# =============================================================================
# POST-PROCESSING FIXES
# =============================================================================

def post_process_adapted_content(
    content: dict,
    company_name: str = None,
) -> dict:
    """
    Post-process adapted content to fix common LLM output issues.

    Args:
        content: The adapted content dict
        company_name: The correct company name to enforce

    Returns:
        Fixed content dict
    """
    if not isinstance(content, dict):
        return content

    def fix_company_name(text: str, correct_name: str) -> str:
        if not text or not correct_name:
            return text
        import re
        base_name = correct_name.rstrip('s')
        patterns = [
            (rf'\b{re.escape(base_name)}\b(?!s)', correct_name),
            (rf'\b{re.escape(base_name.lower())}\b', correct_name),
            (rf'\b{re.escape(base_name.upper())}\b', correct_name),
        ]
        for pattern, replacement in patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text

    def fix_truncated_text(text: str) -> str:
        if not text or len(text) < 10:
            return text
        if text[-1] not in '.!?"\'):':
            if text.endswith(' and') or text.endswith(' or'):
                text = text.rstrip(' and').rstrip(' or') + '.'
            elif text.endswith(' the') or text.endswith(' a') or text.endswith(' an'):
                text = text.rsplit(' ', 1)[0] + '.'
            elif not text[-1].isalnum():
                pass
            else:
                text += '.'
        return text

    def fix_trailing_punctuation_for_names(obj: Any, path: str = "") -> Any:
        if isinstance(obj, str):
            return obj
        elif isinstance(obj, dict):
            result = {}
            for key, value in obj.items():
                key_lower = key.lower()
                if key_lower in ('name', 'fullname', 'email', 'role', 'title', 'designation', 'jobtitle', 'senderemail', 'avatarurl'):
                    if isinstance(value, str) and value.endswith('.'):
                        result[key] = value.rstrip('.')
                    else:
                        result[key] = value
                elif key_lower == 'gender':
                    if isinstance(value, str):
                        result[key] = value.capitalize() if value.lower() in ('male', 'female', 'other') else value
                    else:
                        result[key] = value
                else:
                    result[key] = fix_trailing_punctuation_for_names(value, f"{path}.{key}")
            return result
        elif isinstance(obj, list):
            return [fix_trailing_punctuation_for_names(item, path) for item in obj]
        return obj

    def remove_duplicate_activities(activities: list) -> list:
        if not activities:
            return activities
        seen_names = set()
        unique_activities = []
        for activity in activities:
            if isinstance(activity, dict):
                name = activity.get('name', '')
                name_key = name.lower().strip() if name else ''
                if name_key and name_key not in seen_names:
                    seen_names.add(name_key)
                    unique_activities.append(activity)
                elif not name_key:
                    unique_activities.append(activity)
            else:
                unique_activities.append(activity)
        return unique_activities

    def process_value(value: Any, company_name: str) -> Any:
        if isinstance(value, str):
            if company_name:
                value = fix_company_name(value, company_name)
            value = fix_truncated_text(value)
            return value
        elif isinstance(value, dict):
            return {k: process_value(v, company_name) for k, v in value.items()}
        elif isinstance(value, list):
            if value and isinstance(value[0], dict) and 'name' in value[0]:
                value = remove_duplicate_activities(value)
            return [process_value(item, company_name) for item in value]
        return value

    processed = process_value(content, company_name)
    processed = fix_trailing_punctuation_for_names(processed)
    return processed


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

async def test_gemini_connection() -> bool:
    """Test if Gemini is properly configured."""
    try:
        result = await call_gemini('Return exactly: {"status": "ok"}')
        return isinstance(result, dict) and result.get("status") == "ok"
    except Exception as e:
        logger.error(f"Gemini connection test failed: {e}")
        return False


def get_langsmith_status() -> dict:
    """Get LangSmith configuration status."""
    tracing_enabled = (
        os.getenv("LANGSMITH_TRACING", "").lower() == "true" or
        os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true"
    )
    project = os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT", "default")
    api_key_set = bool(os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY"))

    return {
        "available": LANGSMITH_AVAILABLE,
        "tracing_enabled": tracing_enabled,
        "api_key_configured": api_key_set,
        "project": project,
        "endpoint": os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"),
    }
