"""
Simple Adapter - Phase 1 Implementation

Simplified approach: Scenario Prompt + JSON -> LLM -> Adapted JSON

No factsheet extraction, no RAG, no poison lists.
Just let the LLM figure out what to change based on the scenario prompt.

Usage:
    from src.stages.simple_adapter import adapt_simple

    result = await adapt_simple(
        input_json=my_json,
        scenario_prompt="learners will act as a junior consultant..."
    )
"""
import json
import logging
import time
import asyncio
from dataclasses import dataclass, field
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

# LangSmith tracing
try:
    from langsmith import traceable
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    def traceable(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

logger = logging.getLogger(__name__)

# Thread pool for running sync Gemini calls
# Higher workers = more parallel Gemini calls (API can handle ~20 concurrent)
_executor = ThreadPoolExecutor(max_workers=15)


@dataclass
class SimpleAdaptationResult:
    """Result from simple adaptation."""
    adapted_json: dict
    scenario_prompt: str
    time_ms: int
    input_chars: int
    output_chars: int
    mode: str = "monolithic"  # or "sharded"
    shards_processed: int = 0
    errors: list = field(default_factory=list)


def _get_gemini():
    """Get Gemini model instance."""
    import os
    try:
        import google.generativeai as genai

        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY not set")

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-3-flash-preview")
        logger.info("Gemini 3 Flash Preview initialized for simple adapter")
        return model
    except ImportError:
        raise ImportError("google-generativeai required. Install: pip install google-generativeai")


def _repair_json(text: str) -> dict:
    """Repair malformed JSON using json-repair library."""
    import re

    text = text.strip()

    # Remove markdown code blocks
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```)
        lines = lines[1:]
        # Remove last line if it's ```
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try json_repair library
    try:
        from json_repair import repair_json
        repaired = repair_json(text, return_objects=True)
        if isinstance(repaired, dict):
            return repaired
    except Exception as e:
        logger.warning(f"json_repair failed: {e}")

    # Last resort: find JSON object boundaries
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
    except Exception:
        pass

    raise ValueError(f"Could not parse JSON from response: {text[:200]}...")


def extract_klos_from_json(input_json: dict) -> list[dict]:
    """Extract Key Learning Outcomes from the JSON - these must be PRESERVED."""
    topic_data = input_json.get("topicWizardData", {})
    if not isinstance(topic_data, dict):
        return []

    klo_sources = [
        topic_data.get("assessmentCriterion", []),
        topic_data.get("selectedAssessmentCriterion", []),
    ]

    klos = []
    for source in klo_sources:
        if isinstance(source, list) and len(source) > 0:
            for item in source:
                if isinstance(item, dict):
                    klo = {
                        "id": item.get("id", ""),
                        "outcome": item.get("keyLearningOutcome", item.get("title", "")),
                    }
                    if klo["outcome"]:
                        klos.append(klo)
            break
    return klos


def format_klos_for_prompt(klos: list[dict]) -> str:
    """Format KLOs for inclusion in prompt."""
    if not klos:
        return ""

    lines = ["\n## KEY LEARNING OUTCOMES (PRESERVE THESE - DO NOT CHANGE):\n"]
    for i, klo in enumerate(klos, 1):
        lines.append(f"KLO {i}: {klo.get('outcome', '')[:500]}\n")
    return "\n".join(lines)


@dataclass
class CompanyContext:
    """Extracted company details from first shard - passed to all other shards."""
    company_name: str
    manager_name: str
    manager_email: str
    industry: str


def extract_company_context(adapted_shard: dict) -> CompanyContext:
    """Extract company name, manager, email from an adapted shard.

    This is called after the first shard is adapted to extract the
    company details that will be passed to all remaining shards.
    """
    import re

    # Convert to string for regex matching
    content_str = json.dumps(adapted_shard, indent=2)

    # First try to extract email - it often has the best company/manager info
    manager_email = ""
    company_name = ""
    manager_name = ""

    email_match = re.search(r'([a-z]+)\.([a-z]+)@([a-z]+)(?:threads|apparel|fashion|wear)?\.com', content_str, re.IGNORECASE)
    if email_match:
        first_name = email_match.group(1).capitalize()
        last_name = email_match.group(2).capitalize()
        domain = email_match.group(3)
        manager_email = email_match.group(0).lower()
        manager_name = f"{first_name} {last_name}"

        # Convert domain to company name (e.g., verdantthreads -> Verdant Threads)
        # Split camelCase or concatenated words
        domain_parts = re.findall(r'[A-Z]?[a-z]+', domain.title())
        if domain_parts:
            company_name = " ".join(domain_parts)

    # Fallback: look for company name patterns
    if not company_name or company_name.lower() == "the company":
        company_patterns = [
            r'"organizationName":\s*"([^"]+)"',
            r'"companyName":\s*"([^"]+)"',
            r'Welcome to (?:the )?([A-Z][A-Za-z0-9\s]+?)(?:\.|,|!|\s+team)',
        ]
        for pattern in company_patterns:
            match = re.search(pattern, content_str)
            if match:
                found = match.group(1).strip()
                if found.lower() not in ["the company", "company", "our company"]:
                    company_name = found
                    break

    # Fallback: extract manager name from other patterns
    if not manager_name or manager_name.lower() == "the manager":
        manager_patterns = [
            r'"(?:manager|reportingManager|from)":\s*"([A-Z][a-z]+\s+[A-Z][a-z]+)"',
            r'"name":\s*"([A-Z][a-z]+\s+[A-Z][a-z]+)"',
            r'([A-Z][a-z]+\s+[A-Z][a-z]+),?\s+(?:Director|Manager|VP|Head)',
        ]
        for pattern in manager_patterns:
            match = re.search(pattern, content_str)
            if match:
                manager_name = match.group(1).strip()
                break

    # Set defaults if still not found
    if not company_name:
        company_name = "the company"
    if not manager_name:
        manager_name = "the manager"

    # Extract industry
    industry = "the target industry"
    industry_keywords = ["fashion", "retail", "apparel", "clothing", "sustainable", "organic"]
    content_lower = content_str.lower()
    for keyword in industry_keywords:
        if keyword in content_lower:
            industry = "sustainable fashion / apparel retail"
            break

    logger.info(f"[CONTEXT] Extracted: company='{company_name}', manager='{manager_name}', email='{manager_email}'")

    return CompanyContext(
        company_name=company_name,
        manager_name=manager_name,
        manager_email=manager_email,
        industry=industry
    )


def build_simple_prompt(
    scenario_prompt: str,
    json_content: dict,
    shard_name: str = None,
    company_context: CompanyContext = None
) -> str:
    """
    Build adaptation prompt using scenario as SINGLE SOURCE OF TRUTH.

    From PROMPT_SIMPLIFICATION.md - the scenario prompt contains everything
    the LLM needs to derive: company, industry, KLOs, terminology.

    If company_context is provided (from first shard), use those names for consistency.
    """
    content_str = json.dumps(json_content, indent=2, ensure_ascii=False)

    shard_hint = ""
    if shard_name:
        shard_hint = f'\nThis is the "{shard_name}" section of the simulation.\n'

    # If we have company context from first shard, include it
    company_section = ""
    if company_context:
        company_section = f"""
## COMPANY DETAILS (USE THESE EXACTLY - DO NOT INVENT NEW NAMES):
- Company Name: {company_context.company_name}
- Manager Name: {company_context.manager_name}
- Manager Email: {company_context.manager_email}
- Industry: {company_context.industry}

CRITICAL: Use these EXACT names throughout. Do NOT create different variations.
"""

    derive_section = ""
    if not company_context:
        # Only ask to derive if we don't have context yet (first shard)
        derive_section = """
## FIRST: CREATE COMPANY IDENTITY

INVENT a fictional company name. Do NOT use generic terms like "the company" or "our company".

Rules:
- Company name must be 1-3 words (a real brand name)
- Do NOT use "the company", "our company", or the scenario description
- Use the actual company name throughout the content, never "the company"

CREATE and use consistently:
- COMPANY NAME: A catchy brand name (1-3 words, like a trademark)
- MANAGER NAME: A realistic full name
- MANAGER EMAIL: firstname.lastname@companyname.com
- INDUSTRY: The target industry
"""

    return f"""You are adapting a business simulation to a completely different domain.

## TARGET SCENARIO (YOUR SOURCE OF TRUTH):
{scenario_prompt}
{company_section}{derive_section}

---

## CRITICAL: COMPLETE DOMAIN TRANSFORMATION

**STEP 1: Identify the CURRENT domain**
Read the JSON below. Identify what industry/domain it currently represents.
Note ALL domain-specific terms: processes, roles, KPIs, activities, terminology.

**STEP 2: Replace EVERYTHING with TARGET domain**
Every single domain-specific term must be replaced with the TARGET equivalent.
After transformation, a reader should have NO IDEA what the original domain was.

Examples of what MUST be replaced:
- Job functions -> TARGET industry functions
- Processes -> TARGET industry processes
- KPIs/metrics -> TARGET industry metrics
- Activities -> TARGET industry activities
- Terminology -> TARGET industry terminology
- Department names -> TARGET industry equivalents
- Role titles -> TARGET industry roles

**STEP 3: Verify complete replacement**
Before outputting, scan your output. If ANY term from the original domain remains, replace it.

## CRITICAL: PERSON NAMES AND EMAILS
- ALL person names in the source JSON are from the ORIGINAL domain
- You MUST invent NEW names appropriate for the TARGET industry
- Update ALL email addresses to match new names (firstname.lastname@company.com)
- Do NOT keep any original person names - they belong to the old scenario

---

## WHAT YOU ARE ADAPTING:
{shard_hint}
```json
{content_str}
```

---

## RULES:

### Structure (DO NOT CHANGE)
- Keep ALL IDs exactly as they are
- Keep ALL object/array structures
- Keep ALL KEYS exactly as provided (including dotted keys like "data.taskEmail")
- Only change content VALUES (strings, descriptions, names)

### Domain Fidelity (CRITICAL)
- ZERO tolerance for source domain terms
- Every domain-specific word must become TARGET domain
- Use industry-appropriate KPIs, metrics, terminology
- If unsure whether a term is domain-specific, replace it

### Content Quality
- Resources provide DATA (statistics, facts) not answers
- Questions ask for ANALYSIS (justify, develop, explain)
- Resources: 500-1500 words with real citations (e.g., "Source: McKinsey 2024")

### Consistency
- ONE company name throughout (derive from scenario)
- ONE manager name throughout (create realistic name for TARGET industry)
- Manager email: firstname.lastname@company.com

### Completeness
- NO placeholders [like this]
- NO truncated content
- ALL content must be complete and realistic

---

## OUTPUT:
Return ONLY the adapted JSON. Same structure, completely new domain content.
No explanations. Just valid JSON."""


async def call_gemini_async(prompt: str) -> str:
    """Call Gemini asynchronously using thread pool."""
    loop = asyncio.get_event_loop()

    def _call():
        model = _get_gemini()
        # Use temperature=0 for deterministic outputs, high max_output_tokens to prevent truncation
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.0,
                "max_output_tokens": 65536  # Prevent truncation of large shards
            }
        )
        return response.text

    return await loop.run_in_executor(_executor, _call)


@traceable(name="adapt_simple", run_type="chain")
async def adapt_simple(
    input_json: dict,
    scenario_prompt: str,
) -> SimpleAdaptationResult:
    """
    Simple adaptation using PARALLEL SHARDING.

    Key insight: Scenario prompt is the SINGLE SOURCE OF TRUTH.
    All shards get the SAME scenario prompt, so they all derive
    the same company/KLOs/terminology = cross-connected.

    Args:
        input_json: The simulation JSON to adapt
        scenario_prompt: Description of the target scenario (source of truth)

    Returns:
        SimpleAdaptationResult with adapted JSON
    """
    start_time = time.time()
    input_chars = len(json.dumps(input_json))

    logger.info(f"[SIMPLE ADAPTER] Starting PARALLEL shard adaptation")
    logger.info(f"[SIMPLE ADAPTER] Input size: {input_chars} chars")
    logger.info(f"[SIMPLE ADAPTER] Scenario: {scenario_prompt[:100]}...")

    # Always use parallel sharding - this is the correct approach
    adapted_json, shards_count, errors = await _adapt_with_sharding(
        input_json, scenario_prompt
    )

    time_ms = int((time.time() - start_time) * 1000)
    output_chars = len(json.dumps(adapted_json))

    logger.info(f"[SIMPLE ADAPTER] Complete in {time_ms}ms")
    logger.info(f"[SIMPLE ADAPTER] Output size: {output_chars} chars")
    logger.info(f"[SIMPLE ADAPTER] Shards processed: {shards_count}")

    return SimpleAdaptationResult(
        adapted_json=adapted_json,
        scenario_prompt=scenario_prompt,
        time_ms=time_ms,
        input_chars=input_chars,
        output_chars=output_chars,
        mode="parallel_shards",
        shards_processed=shards_count,
        errors=errors,
    )


async def _adapt_monolithic(input_json: dict, scenario_prompt: str) -> dict:
    """
    Adapt entire JSON in a single LLM call.

    This works because Gemini 2.5 Flash has 1M token context window.
    """
    logger.info("[SIMPLE ADAPTER] Using monolithic approach (single LLM call)")

    # Extract KLOs from input JSON (these must be preserved)
    klos = extract_klos_from_json(input_json)
    klos_text = format_klos_for_prompt(klos)
    logger.info(f"[SIMPLE ADAPTER] Extracted {len(klos)} KLOs to preserve")

    # Build the prompt with KLOs included
    prompt = build_simple_prompt(scenario_prompt, input_json, klos_text=klos_text)
    prompt_chars = len(prompt)
    logger.info(f"[SIMPLE ADAPTER] Prompt size: {prompt_chars} chars (~{prompt_chars // 4} tokens)")

    # Call Gemini
    logger.info("[SIMPLE ADAPTER] Calling Gemini...")
    response_text = await call_gemini_async(prompt)
    logger.info(f"[SIMPLE ADAPTER] Response received: {len(response_text)} chars")

    # Parse response
    adapted_json = _repair_json(response_text)

    return adapted_json


async def _adapt_with_sharding(
    input_json: dict,
    scenario_prompt: str
) -> tuple[dict, int, list]:
    """
    Adapt JSON by sharding into smaller pieces.

    Uses SEQUENTIAL FIRST SHARD approach:
    1. Adapt "overview" shard first
    2. Extract company name, manager, email from it
    3. Pass those to all remaining shards in parallel

    This ensures consistent naming across all shards without extra LLM call.
    """
    from .sharder import Sharder, merge_shards
    from ..models.shard import LockState

    logger.info("[SIMPLE ADAPTER] Using sharded approach with sequential first shard")

    # Shard the JSON
    sharder = Sharder()
    collection = sharder.shard(input_json)

    # Separate locked vs unlocked
    locked_shards = [s for s in collection.shards if s.lock_state == LockState.FULLY_LOCKED]
    unlocked_shards = [s for s in collection.shards if s.lock_state != LockState.FULLY_LOCKED]

    logger.info(f"[SIMPLE ADAPTER] {len(locked_shards)} locked, {len(unlocked_shards)} to adapt")

    errors = []
    company_context = None

    # Find a good shard to adapt first (overview or lesson_information)
    first_shard = None
    remaining_shards = []
    priority_names = ["overview", "lesson_information", "workplace_scenario"]

    for shard in unlocked_shards:
        if first_shard is None and shard.name in priority_names:
            first_shard = shard
        else:
            remaining_shards.append(shard)

    # If no priority shard found, use first unlocked shard
    if first_shard is None and unlocked_shards:
        first_shard = unlocked_shards[0]
        remaining_shards = unlocked_shards[1:]

    # STEP 1: Adapt first shard (no company context yet)
    if first_shard:
        logger.info(f"[SIMPLE ADAPTER] STEP 1: Adapting first shard '{first_shard.name}' to establish names...")
        try:
            first_result = await _adapt_single_shard_simple(first_shard, scenario_prompt, company_context=None)
            first_shard.content = first_result
            first_shard.current_hash = ""
            collection.update_shard(first_shard)

            # Extract company context from first shard
            company_context = extract_company_context(first_result)
            logger.info(f"[SIMPLE ADAPTER] Extracted context: {company_context.company_name}, {company_context.manager_name}")

        except Exception as e:
            logger.error(f"[SIMPLE ADAPTER] First shard failed: {e}")
            errors.append(f"{first_shard.id}: {str(e)}")

    # STEP 2: Adapt remaining shards - split large ones first
    if remaining_shards:
        logger.info(f"[SIMPLE ADAPTER] STEP 2: Adapting {len(remaining_shards)} remaining shards...")

        # Process shards - split large ones to avoid LLM output truncation
        # LLM seems to have ~12K output limit despite max_output_tokens setting
        # LLM output limit seems to be ~10-12K despite max_output_tokens setting
        # Split aggressively at 8K to ensure no truncation
        MAX_SHARD_SIZE = 8000

        tasks = []
        task_info = []  # Track which shard/index each task corresponds to

        for shard in remaining_shards:
            shard_size = len(json.dumps(shard.content))
            split_done = False

            # Check if this shard needs splitting
            if shard_size > MAX_SHARD_SIZE:
                # Case 1: Content IS a list
                if isinstance(shard.content, list) and len(shard.content) > 1:
                    logger.info(f"[SIMPLE ADAPTER] SPLITTING list shard '{shard.name}' ({shard_size} chars, {len(shard.content)} items)")
                    for idx, item in enumerate(shard.content):
                        task = _adapt_single_item(item, scenario_prompt, company_context, f"{shard.name}[{idx}]")
                        tasks.append(task)
                        task_info.append({"shard": shard, "type": "list_item", "index": idx, "key": None})
                    split_done = True

                # Case 2: Content is a DICT - look for large arrays inside
                elif isinstance(shard.content, dict):
                    # First, look for large arrays
                    for key, val in shard.content.items():
                        if isinstance(val, list) and len(val) > 1:
                            array_size = len(json.dumps(val))
                            if array_size > MAX_SHARD_SIZE * 0.5:
                                logger.info(f"[SIMPLE ADAPTER] SPLITTING nested array '{shard.name}.{key}' ({array_size} chars, {len(val)} items)")
                                for idx, item in enumerate(val):
                                    item_size = len(json.dumps(item))
                                    # If individual item is still too large, split it by keys
                                    if item_size > MAX_SHARD_SIZE and isinstance(item, dict):
                                        logger.info(f"[SIMPLE ADAPTER]   -> Item {idx} too large ({item_size}), splitting by keys")

                                        # Collect fields to adapt together (reduces LLM calls)
                                        small_fields = {}  # Small strings
                                        medium_fields = {}  # Medium-sized dicts/arrays (< 10K)

                                        # Only split fields that are REALLY large (> 10K)
                                        SPLIT_THRESHOLD = 10000

                                        for subkey, subval in item.items():
                                            subval_size = len(json.dumps(subval))

                                            # Skip truly trivial fields (IDs, types, empty arrays, etc.)
                                            # But include name/title fields even if small - they often need adaptation
                                            is_name_field = subkey.lower() in ('name', 'title', 'label', 'heading')
                                            if subval_size <= 20 or (subval_size <= 50 and not is_name_field):
                                                continue

                                            # Small string fields (like name, title, descriptions)
                                            if isinstance(subval, str) and subval_size <= 500:
                                                small_fields[subkey] = subval
                                            # Large fields (> 10K) - split further if dict
                                            elif subval_size > SPLIT_THRESHOLD:
                                                if isinstance(subval, dict):
                                                    logger.info(f"[SIMPLE ADAPTER]     -> Subkey {subkey} large ({subval_size}), splitting deeper")
                                                    for deepkey, deepval in subval.items():
                                                        deepval_size = len(json.dumps(deepval))
                                                        if deepval_size > SPLIT_THRESHOLD:
                                                            task = _adapt_single_item(deepval, scenario_prompt, company_context, f"{shard.name}.{key}[{idx}].{subkey}.{deepkey}")
                                                            tasks.append(task)
                                                            task_info.append({"shard": shard, "type": "deeper_item", "index": idx, "key": key, "subkey": subkey, "deepkey": deepkey})
                                                        elif deepval_size > 50:
                                                            # Collect in medium_fields for batch adaptation
                                                            medium_fields[f"{subkey}.{deepkey}"] = deepval
                                                else:
                                                    # Large non-dict (like big array) - adapt individually
                                                    task = _adapt_single_item(subval, scenario_prompt, company_context, f"{shard.name}.{key}[{idx}].{subkey}")
                                                    tasks.append(task)
                                                    task_info.append({"shard": shard, "type": "deep_item", "index": idx, "key": key, "subkey": subkey})
                                            # Medium fields (50-10K) - collect for batch adaptation
                                            else:
                                                medium_fields[subkey] = subval

                                        # Combine small and medium fields - split into chunks if too large
                                        all_fields = {**small_fields, **medium_fields}
                                        if all_fields:
                                            # Split into chunks of ~10K each to avoid truncation
                                            MAX_BATCH_SIZE = 10000
                                            current_batch = {}
                                            current_size = 0
                                            batch_num = 0

                                            for field_name, field_val in all_fields.items():
                                                field_size = len(json.dumps(field_val))
                                                if current_size + field_size > MAX_BATCH_SIZE and current_batch:
                                                    # Flush current batch
                                                    logger.info(f"[SIMPLE ADAPTER]     -> Batch {batch_num} ({len(current_batch)} fields, {current_size} chars)")
                                                    task = _adapt_single_item(current_batch, scenario_prompt, company_context, f"{shard.name}.{key}[{idx}].batch_{batch_num}")
                                                    tasks.append(task)
                                                    task_info.append({"shard": shard, "type": "batch_fields", "index": idx, "key": key, "fields": list(current_batch.keys())})
                                                    current_batch = {}
                                                    current_size = 0
                                                    batch_num += 1
                                                current_batch[field_name] = field_val
                                                current_size += field_size

                                            # Flush remaining
                                            if current_batch:
                                                logger.info(f"[SIMPLE ADAPTER]     -> Batch {batch_num} ({len(current_batch)} fields, {current_size} chars)")
                                                task = _adapt_single_item(current_batch, scenario_prompt, company_context, f"{shard.name}.{key}[{idx}].batch_{batch_num}")
                                                tasks.append(task)
                                                task_info.append({"shard": shard, "type": "batch_fields", "index": idx, "key": key, "fields": list(current_batch.keys())})
                                    else:
                                        task = _adapt_single_item(item, scenario_prompt, company_context, f"{shard.name}.{key}[{idx}]")
                                        tasks.append(task)
                                        task_info.append({"shard": shard, "type": "nested_item", "index": idx, "key": key})
                                split_done = True
                                break

            if not split_done:
                task = _adapt_single_shard_simple(shard, scenario_prompt, company_context=company_context)
                tasks.append(task)
                task_info.append({"shard": shard, "type": "whole", "index": None, "key": None})

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results - reassemble split shards
        for i, result in enumerate(results):
            info = task_info[i]
            shard = info["shard"]

            if isinstance(result, Exception):
                logger.error(f"[SIMPLE ADAPTER] Task {i} failed: {result}")
                errors.append(f"{shard.id}: {str(result)}")
            elif info["type"] == "whole":
                shard.content = result
                shard.current_hash = ""
                collection.update_shard(shard)
            elif info["type"] == "list_item":
                # Update specific item in the array (shard content IS the list)
                if isinstance(shard.content, list) and info["index"] < len(shard.content):
                    shard.content[info["index"]] = result
                    collection.update_shard(shard)
            elif info["type"] == "nested_item":
                # Update item in nested array (shard content is dict with array inside)
                key = info["key"]
                idx = info["index"]
                if isinstance(shard.content, dict) and key in shard.content:
                    if isinstance(shard.content[key], list) and idx < len(shard.content[key]):
                        shard.content[key][idx] = result
                        collection.update_shard(shard)
            elif info["type"] == "deep_item":
                # Update a specific field within an item in a nested array
                key = info["key"]
                idx = info["index"]
                subkey = info["subkey"]
                if isinstance(shard.content, dict) and key in shard.content:
                    if isinstance(shard.content[key], list) and idx < len(shard.content[key]):
                        if isinstance(shard.content[key][idx], dict):
                            shard.content[key][idx][subkey] = result
                            collection.update_shard(shard)
            elif info["type"] == "deeper_item":
                # Update a field within a field within an item (3 levels deep)
                key = info["key"]
                idx = info["index"]
                subkey = info["subkey"]
                deepkey = info["deepkey"]
                if isinstance(shard.content, dict) and key in shard.content:
                    if isinstance(shard.content[key], list) and idx < len(shard.content[key]):
                        if isinstance(shard.content[key][idx], dict) and subkey in shard.content[key][idx]:
                            if isinstance(shard.content[key][idx][subkey], dict):
                                shard.content[key][idx][subkey][deepkey] = result
                                collection.update_shard(shard)
            elif info["type"] in ("small_fields", "batch_fields"):
                # Update multiple fields within an item (batched to reduce LLM calls)
                key = info["key"]
                idx = info["index"]
                fields = info["fields"]  # List of field names that were adapted
                logger.info(f"[DEBUG] batch_fields for [{idx}]: expected={fields}")
                logger.info(f"[DEBUG] result type={type(result).__name__}, keys={list(result.keys()) if isinstance(result, dict) else 'N/A'}")
                # DUMP the result to see what LLM actually returns
                with open(f'debug_batch_{idx}.json', 'w') as f:
                    json.dump(result, f, indent=2, default=str)
                logger.info(f"[DEBUG] Dumped result to debug_batch_{idx}.json")
                # Check if LLM wrapped the response - unwrap if needed
                # Handle cases like {"batch_0": {...actual content...}}
                if isinstance(result, dict):
                    # Check for wrapper keys like "batch_0", "batch_1", etc.
                    import re
                    for k in list(result.keys()):
                        if re.match(r'batch_\d+', k) and isinstance(result[k], dict):
                            logger.info(f"[DEBUG] Found batch wrapper key '{k}', unwrapping")
                            result = result[k]
                            break
                    # Also check for single-key wrapper where key not in expected fields
                    if len(result) == 1:
                        only_key = list(result.keys())[0]
                        if only_key not in fields and isinstance(result[only_key], dict):
                            logger.info(f"[DEBUG] Unwrapping single-key wrapper '{only_key}'")
                            result = result[only_key]
                logger.info(f"[DEBUG] Final result keys: {list(result.keys()) if isinstance(result, dict) else 'N/A'}")
                if isinstance(shard.content, dict) and key in shard.content:
                    if isinstance(shard.content[key], list) and idx < len(shard.content[key]):
                        item = shard.content[key][idx]
                        if isinstance(item, dict) and isinstance(result, dict):
                            matched = 0
                            for field_name, adapted_value in result.items():
                                # Handle nested keys like "data.taskEmail"
                                if "." in field_name:
                                    parts = field_name.split(".", 1)
                                    parent_key, child_key = parts[0], parts[1]
                                    if parent_key in item and isinstance(item[parent_key], dict):
                                        item[parent_key][child_key] = adapted_value
                                        matched += 1
                                    else:
                                        logger.warning(f"[DEBUG] FAILED to place {field_name}: parent_key={parent_key} not in item or not dict")
                                else:
                                    if field_name in item:
                                        item[field_name] = adapted_value
                                        matched += 1
                                    else:
                                        logger.warning(f"[DEBUG] FAILED to place {field_name}: not in item")
                            collection.update_shard(shard)
                            logger.info(f"[SIMPLE ADAPTER] Reassembled {info['type']} for [{idx}]: {matched}/{len(result)} fields matched")

    # Merge back
    adapted_json = merge_shards(collection, input_json)

    total_shards = 1 + len(remaining_shards) if first_shard else len(remaining_shards)
    return adapted_json, total_shards, errors


@traceable(name="adapt_small_fields", run_type="llm")
async def _adapt_small_fields(
    fields: dict,
    scenario_prompt: str,
    company_context: CompanyContext,
    item_name: str
) -> dict:
    """Adapt small string fields (like name, title) together in one call."""
    logger.info(f"[SIMPLE ADAPTER] Adapting small fields: {item_name}")

    company_section = ""
    if company_context:
        company_section = f"""Use these names consistently:
- Company: {company_context.company_name}
- Manager: {company_context.manager_name}
"""

    prompt = f"""You are adapting field names/titles for a business simulation.

TARGET SCENARIO:
{scenario_prompt}

{company_section}

FIELDS TO ADAPT (JSON object):
```json
{json.dumps(fields, indent=2)}
```

RULES:
1. Replace ALL domain-specific terms with TARGET scenario equivalents
2. Keep the same keys, only change string values
3. COMPLETELY remove any trace of the original domain
4. Use TARGET industry terminology

Return ONLY the adapted JSON object with same keys, new values.
"""

    response_text = await call_gemini_async(prompt)
    adapted = _repair_json(response_text)

    logger.info(f"[SIMPLE ADAPTER] Small fields adapted: {list(adapted.keys())}")
    return adapted


@traceable(name="adapt_item", run_type="llm")
async def _adapt_single_item(
    item: dict,
    scenario_prompt: str,
    company_context: CompanyContext,
    item_name: str
) -> dict:
    """Adapt a single array item (e.g., one stage from simulationFlow)."""
    logger.info(f"[SIMPLE ADAPTER] Adapting item: {item_name}")

    # Build prompt for single item
    prompt = build_simple_prompt(
        scenario_prompt,
        item,
        shard_name=item_name,
        company_context=company_context
    )

    # Call Gemini
    response_text = await call_gemini_async(prompt)
    raw_size = len(response_text)

    # Parse response
    adapted_content = _repair_json(response_text)

    # Check for truncation
    input_size = len(json.dumps(item))
    output_size = len(json.dumps(adapted_content))

    logger.info(f"[SIMPLE ADAPTER] Item {item_name}: input={input_size}, raw={raw_size}, parsed={output_size}")

    if output_size < input_size * 0.7:
        logger.warning(f"[SIMPLE ADAPTER] Item {item_name} TRUNCATED!")
        if raw_size < input_size * 0.8:
            logger.warning(f"  -> LLM truncated output (raw {raw_size} < input {input_size})")
        else:
            logger.warning(f"  -> JSON parsing lost data (raw {raw_size} -> parsed {output_size})")

    return adapted_content


@traceable(name="adapt_shard", run_type="llm")
async def _adapt_single_shard_simple(
    shard,
    scenario_prompt: str,
    company_context: CompanyContext = None
) -> dict:
    """Adapt a single shard using scenario prompt as source of truth.

    If company_context is provided, uses those exact names for consistency.
    """
    context_info = f" (with company context: {company_context.company_name})" if company_context else " (first shard - establishing names)"
    logger.info(f"[SIMPLE ADAPTER] Adapting shard: {shard.id} ({shard.name}){context_info}")

    # Build prompt - pass company_context if we have it
    prompt = build_simple_prompt(
        scenario_prompt,
        shard.content,
        shard_name=shard.name,
        company_context=company_context
    )

    # Call Gemini
    response_text = await call_gemini_async(prompt)

    # Parse response
    adapted_content = _repair_json(response_text)

    # Check for truncation - warn if output is significantly smaller than input
    input_size = len(json.dumps(shard.content))
    output_size = len(json.dumps(adapted_content))
    if output_size < input_size * 0.7:  # More than 30% shrinkage
        logger.warning(f"[SIMPLE ADAPTER] Shard {shard.id} may be TRUNCATED: {input_size} -> {output_size} chars ({output_size/input_size:.0%})")

    logger.info(f"[SIMPLE ADAPTER] Shard {shard.id} adapted successfully ({input_size} -> {output_size} chars)")
    return adapted_content


# =============================================================================
# Convenience function for testing
# =============================================================================

async def test_simple_adapter(json_path: str, scenario_prompt: str):
    """
    Quick test function for the simple adapter.

    Usage:
        import asyncio
        from src.stages.simple_adapter import test_simple_adapter

        asyncio.run(test_simple_adapter(
            "sample_input.json",
            "learners will act as a junior consultant for EcoChic Threads..."
        ))
    """
    import json

    # Load JSON
    with open(json_path, "r", encoding="utf-8") as f:
        input_json = json.load(f)

    print(f"Loaded {json_path}: {len(json.dumps(input_json))} chars")

    # Run adaptation
    result = await adapt_simple(input_json, scenario_prompt, use_sharding=False)

    print(f"\n=== RESULT ===")
    print(f"Mode: {result.mode}")
    print(f"Time: {result.time_ms}ms")
    print(f"Input: {result.input_chars} chars")
    print(f"Output: {result.output_chars} chars")

    # Save output
    output_path = "simple_adapted_output.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result.adapted_json, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to: {output_path}")

    return result
