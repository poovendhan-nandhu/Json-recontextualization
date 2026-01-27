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
import sys
import time
import asyncio
from dataclasses import dataclass, field
from typing import Optional

# Configure logging if not already configured
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

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

# RAG Fact Retriever for resource enrichment
_fact_retriever_initialized = False

async def _get_rag_facts(scenario: str, questions: list[str] = None, klos: list[dict] = None) -> str:
    """Get relevant facts from RAG for resource adaptation."""
    global _fact_retriever_initialized
    try:
        from src.rag.fact_retriever import get_fact_retriever, get_relevant_facts

        retriever = get_fact_retriever()

        # Index facts on first use
        if not _fact_retriever_initialized and retriever.count() == 0:
            from pathlib import Path
            facts_file = Path(__file__).parent.parent.parent / "data" / "business_facts.txt"
            if facts_file.exists():
                indexed = retriever.index_facts_from_file(str(facts_file))
                logger.info(f"[RAG] Initialized with {indexed} facts")
            _fact_retriever_initialized = True

        facts = await get_relevant_facts(scenario, questions, klos, max_facts=10)
        return facts
    except Exception as e:
        logger.warning(f"[RAG] Failed to get facts: {e}")
        return ""


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
    entity_map: dict = field(default_factory=dict)
    domain_profile: dict = field(default_factory=dict)


def _repair_json(text: str) -> dict | list:
    """Repair malformed JSON using json-repair library. Handles both objects and arrays."""
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
        if isinstance(repaired, (dict, list)):
            return repaired
    except Exception as e:
        logger.warning(f"json_repair failed: {e}")

    # Try to find JSON object boundaries
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
    except Exception:
        pass

    # Try to find JSON array boundaries
    try:
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
    except Exception:
        pass

    # Try to repair truncated JSON by adding closing brackets
    try:
        # Count open/close brackets
        open_braces = text.count("{") - text.count("}")
        open_brackets = text.count("[") - text.count("]")

        repaired_text = text
        # Add missing closing brackets/braces
        if open_brackets > 0:
            repaired_text += "]" * open_brackets
        if open_braces > 0:
            repaired_text += "}" * open_braces

        return json.loads(repaired_text)
    except Exception:
        pass

    raise ValueError(f"Could not parse JSON from response: {text[:200]}...")


def cleanup_forbidden_terms(adapted_json: dict, forbidden_terms: list[str], replacement_map: dict = None) -> dict:
    """Post-process adapted JSON to remove any leaked forbidden terms.

    Args:
        adapted_json: The adapted JSON that may still contain leaked terms
        forbidden_terms: List of terms that should NOT appear in output
        replacement_map: Optional dict mapping forbidden term -> replacement
                        If not provided, terms are removed or replaced with generic placeholder

    Returns:
        Cleaned JSON with forbidden terms replaced
    """
    if not forbidden_terms:
        return adapted_json

    # Convert to string for search/replace
    adapted_str = json.dumps(adapted_json, ensure_ascii=False)
    cleaned = False

    for term in forbidden_terms:
        if not term or len(term) < 2:  # Skip empty or single-char terms
            continue

        if term in adapted_str:
            # Get replacement from map, or use empty string
            replacement = ""
            if replacement_map and term in replacement_map:
                replacement = replacement_map[term]

            adapted_str = adapted_str.replace(term, replacement)
            cleaned = True
            logger.info(f"[CLEANUP] Replaced leaked term '{term}' with '{replacement or '(removed)'}'")

    if cleaned:
        try:
            return json.loads(adapted_str)
        except json.JSONDecodeError:
            logger.warning("[CLEANUP] JSON decode error after cleanup, returning original")
            return adapted_json

    return adapted_json


def extract_klos_from_json(input_json: dict) -> list[dict]:
    """Extract Key Learning Outcomes from the JSON - these will be adapted to target scenario.

    Supports both formats:
    - New format: Root-level keys with snake_case (assessment_criterion)
    - Old format: topicWizardData wrapper with camelCase (assessmentCriterion)
    """
    # Try new format first (root-level snake_case)
    klo_sources = [
        input_json.get("assessment_criterion", []),
        input_json.get("selected_assessment_criterion", []),
    ]

    # Fallback to old format (topicWizardData wrapper)
    if not any(klo_sources):
        topic_data = input_json.get("topicWizardData", {})
        if isinstance(topic_data, dict):
            klo_sources = [
                topic_data.get("assessmentCriterion", []),
                topic_data.get("selectedAssessmentCriterion", []),
            ]

    klos = []
    for source in klo_sources:
        if isinstance(source, list) and len(source) > 0:
            for item in source:
                if isinstance(item, dict):
                    # Support both snake_case and camelCase
                    klo = {
                        "id": item.get("id", ""),
                        "outcome": item.get("key_learning_outcome",
                                          item.get("keyLearningOutcome",
                                                  item.get("title", ""))),
                    }
                    if klo["outcome"]:
                        klos.append(klo)
            break
    return klos


def format_klos_for_prompt(klos: list[dict]) -> str:
    """Format derived KLOs for alignment - all shards must follow these."""
    if not klos:
        return ""

    lines = ["\n## KEY LEARNING OUTCOMES (ALL CONTENT MUST ALIGN TO THESE):\n"]
    lines.append("These KLOs are already adapted for the TARGET scenario. Your content MUST support these:\n\n")
    for i, klo in enumerate(klos, 1):
        lines.append(f"**KLO {i}:** {klo.get('outcome', '')[:500]}\n")
    lines.append("\n**ALIGNMENT RULES:**\n")
    lines.append("- Resources: Provide DATA that helps learners achieve these KLOs\n")
    lines.append("- Questions: Each question must assess one or more of these KLOs\n")
    lines.append("- Activities: Must build skills toward these KLOs\n")
    lines.append("- Rubrics: Must evaluate mastery of these KLOs\n")
    return "\n".join(lines)


def extract_klos_from_adapted_shard(adapted_content: dict) -> list[dict]:
    """Extract KLOs from an adapted assessment_criterion shard.

    Works with both formats:
    - New format: snake_case keys (key_learning_outcome, assessment_criterion)
    - Old format: camelCase keys (keyLearningOutcome, assessmentCriterion)
    """
    klos = []

    # Case 1: Content is a list of KLO objects directly
    if isinstance(adapted_content, list):
        for item in adapted_content:
            if isinstance(item, dict):
                # Support both snake_case and camelCase
                outcome = item.get("key_learning_outcome",
                                  item.get("keyLearningOutcome",
                                          item.get("title",
                                                  item.get("outcome", ""))))
                if outcome:
                    klos.append({"id": item.get("id", ""), "outcome": outcome})
        return klos

    # Case 2: Content is a dict - look for KLOs inside
    if isinstance(adapted_content, dict):
        # Check common KLO container keys (both snake_case and camelCase)
        klo_keys = [
            # New format (snake_case)
            "assessment_criterion",
            "selected_assessment_criterion",
            # Old format (camelCase)
            "assessmentCriterion",
            "selectedAssessmentCriterion",
            # Generic
            "klos",
            "outcomes",
        ]
        for key in klo_keys:
            if key in adapted_content and isinstance(adapted_content[key], list):
                return extract_klos_from_adapted_shard(adapted_content[key])

        # Check if it's a single KLO object (both formats)
        outcome = adapted_content.get("key_learning_outcome",
                                     adapted_content.get("keyLearningOutcome",
                                                        adapted_content.get("title", "")))
        if outcome:
            klos.append({"id": adapted_content.get("id", ""), "outcome": outcome})

    return klos


def extract_questions_from_input(input_json: dict) -> list[str]:
    """Extract submission questions from input JSON for inference map.

    These questions must be answerable from the resources.
    """
    questions = []

    def extract_recursive(obj):
        if isinstance(obj, dict):
            for key, val in obj.items():
                # Look for question-related keys
                if 'question' in key.lower() and isinstance(val, list):
                    for q in val:
                        if isinstance(q, dict):
                            qtext = q.get("question") or q.get("text") or q.get("content", "")
                            if qtext and len(qtext) > 20:  # Real questions are longer
                                questions.append(qtext)
                        elif isinstance(q, str) and len(q) > 20:
                            questions.append(q)
                elif 'question' in key.lower() and isinstance(val, str) and len(val) > 20:
                    questions.append(val)
                else:
                    extract_recursive(val)
        elif isinstance(obj, list):
            for item in obj:
                extract_recursive(item)

    # Search in simulation flow and submission questions
    sim_flow = input_json.get("simulation_flow") or input_json.get("topicWizardData", {}).get("simulationFlow", [])
    extract_recursive(sim_flow)

    # Also check direct submission_questions
    sub_q = input_json.get("submission_questions") or input_json.get("topicWizardData", {}).get("submissionQuestions", [])
    extract_recursive(sub_q)

    # Deduplicate
    seen = set()
    unique_questions = []
    for q in questions:
        if q not in seen:
            seen.add(q)
            unique_questions.append(q)

    return unique_questions


# Shards that should NEVER be split
NEVER_SPLIT_SHARDS = ["resources"]

# Shards to SKIP entirely - none currently (chat history needs adaptation too)
SKIP_SHARDS = [
    # "activities_chat_history",  # Removed - contains HR terms that need adaptation
    # "scenario_chat_history",    # Removed - contains HR terms that need adaptation
]

# Model selection - gemini-2.5-flash is faster and more reliable
STABLE_MODEL = "gemini-2.5-flash"
DEFAULT_MODEL = "gemini-2.5-flash"  # Use stable model everywhere (3-flash-preview hangs)
SHARDS_USING_STABLE_MODEL = set()  # Not needed - all use stable model now


def scan_json_for_domain_terms(input_json: dict) -> list[str]:
    """Fast regex-based scan to extract names, roles, and domain terms from JSON.

    This runs instantly and catches obvious terms like person names, company names,
    email domains, and role titles.
    """
    import re

    content_str = json.dumps(input_json, ensure_ascii=False)
    terms = set()

    # 1. Extract person names (First Last pattern in quotes)
    name_patterns = [
        r'"(?:name|from|to|manager|author|sender)":\s*"([A-Z][a-z]+\s+[A-Z][a-z]+)"',
        r'"reportingManager":\s*"([A-Z][a-z]+\s+[A-Z][a-z]+)"',
    ]
    for pattern in name_patterns:
        for match in re.findall(pattern, content_str):
            terms.add(match.lower())
            # Also add individual parts
            parts = match.split()
            for part in parts:
                if len(part) > 2:
                    terms.add(part.lower())

    # 2. Extract company/organization names
    company_patterns = [
        r'"(?:company|organization|organizationName|companyName)":\s*"([^"]+)"',
        r'Welcome to (?:the )?([A-Z][A-Za-z\s]+?)(?:\.|,|!)',
    ]
    for pattern in company_patterns:
        for match in re.findall(pattern, content_str):
            terms.add(match.lower().strip())

    # 3. Extract email domains (source company domain)
    email_pattern = r'@([a-z0-9]+)\.(com|org|net)'
    for match in re.findall(email_pattern, content_str.lower()):
        domain = match[0]
        if domain not in ['gmail', 'yahoo', 'hotmail', 'outlook', 'company']:
            terms.add(domain)

    # 4. Extract role titles from common fields
    role_patterns = [
        r'"(?:role|title|position|jobTitle)":\s*"([^"]+)"',
        r'"designation":\s*"([^"]+)"',
    ]
    for pattern in role_patterns:
        for match in re.findall(pattern, content_str, re.IGNORECASE):
            terms.add(match.lower())

    # 5. Domain keywords are NOT hardcoded - LLM infers from scenario prompt
    # The regex above already extracts: names, companies, emails, roles
    # The LLM should figure out domain-specific terms (HR, interview, etc.) from context

    logger.info(f"[DOMAIN] Regex scan found {len(terms)} terms")
    return list(terms)


def format_forbidden_terms_for_prompt(terms: list[str]) -> str:
    """Format forbidden terms for inclusion in the prompt."""
    if not terms:
        return ""

    # Limit to 30 terms for faster prompts (prioritize longer/more specific terms)
    sorted_terms = sorted(terms, key=lambda x: -len(x))[:30]
    terms_str = ", ".join(f'"{t}"' for t in sorted_terms)

    return f"""
## FORBIDDEN SOURCE DOMAIN TERMS (MUST BE REPLACED):

The following terms are from the SOURCE domain and MUST NOT appear in your output:
{terms_str}

**ZERO TOLERANCE:** If ANY of these terms appear in your output, you have FAILED.
Replace each with an appropriate TARGET domain equivalent.
"""


@dataclass
class CompanyContext:
    """Extracted company details from first shard - passed to all other shards."""
    company_name: str
    manager_name: str
    manager_email: str
    industry: str


@dataclass
class EntityMap:
    """Maps source entities to target entities - generated BEFORE adaptation."""
    company: dict  # {"source": "Velocity Dome", "target": "EcoChic Threads", "domain": "ecochicthreads.com"}
    competitor: dict  # {"source": "original competitor", "target": "MegaBurger"} - DISTINCT from protagonist
    people: dict   # {"Elizabeth Carter": {"name": "Sarah Chen", "role": "Director of Strategy", "email": "..."}}
    roles: dict    # {"HR Analyst": "Market Analyst", "Senior HR Manager": "Director of Strategy"}


@dataclass
class DomainProfile:
    """Source→Target domain terminology mapping - generated BEFORE adaptation."""
    source_domain: str  # "HR/Recruitment"
    target_domain: str  # "Market Entry Analysis"
    terminology_map: dict  # {"interview": "market assessment", "hiring": "market entry", ...}
    forbidden_terms: list  # ["HR", "hiring", "candidate", "interview", ...]
    target_kpis: list  # ["Market penetration rate", "Customer acquisition cost", ...]


async def generate_entity_and_domain_maps(
    input_json: dict,
    scenario_prompt: str
) -> tuple[EntityMap, DomainProfile, list[dict]]:
    """
    Generate entity_map, domain_profile, AND adapted KLOs BEFORE adaptation starts.

    This is ONE LLM call that:
    1. Analyzes source JSON to identify domain and entities
    2. Analyzes scenario prompt to understand target domain
    3. Generates complete mappings for consistent adaptation
    4. Adapts KLOs to target domain (so all shards can align to them)

    Returns:
        tuple of (EntityMap, DomainProfile, adapted_klos)
    """
    # Get a representative sample of the source JSON
    import re
    sample_parts = []

    # Support both formats: new (root-level snake_case) and old (topicWizardData wrapper)
    # New format: keys at root level with snake_case
    # Old format: keys under topicWizardData with camelCase

    # Try new format first
    lesson_info = input_json.get("lesson_information") or input_json.get("topicWizardData", {}).get("lessonInformation")
    workplace = input_json.get("workplace_scenario") or input_json.get("topicWizardData", {}).get("workplaceScenario")
    sim_flow = input_json.get("simulation_flow") or input_json.get("topicWizardData", {}).get("simulationFlow", [])

    # Lesson info (has company, scenario context)
    if lesson_info:
        sample_parts.append(("lesson_information", json.dumps(lesson_info)[:1500]))

    # Workplace scenario (reveals domain clearly + has manager name)
    if workplace:
        sample_parts.append(("workplace_scenario", json.dumps(workplace)[:2000]))

    # Characters (has names, roles) - check ALL stages
    for stage in sim_flow:  # Check ALL stages for users
        activity_data = stage.get("data", {}).get("activity_data") or stage.get("data", {}).get("activityData", {})
        selected_val = activity_data.get("selected_value") or activity_data.get("selectedValue", {})
        if selected_val.get("users"):
            sample_parts.append(("characters", json.dumps(selected_val["users"])[:1500]))
            break

    # Extract emails and names from the entire JSON for entity discovery
    content_str = json.dumps(input_json, ensure_ascii=False)
    found_emails = list(set(re.findall(r'[a-z]+\.[a-z]+@[a-z0-9]+\.com', content_str, re.IGNORECASE)))[:10]
    found_names = list(set(re.findall(r'"(?:manager|reportingManager|name|from|to|author)":\s*"([A-Z][a-z]+ [A-Z][a-z]+)"', content_str)))[:10]

    if found_emails or found_names:
        sample_parts.append(("entities_found", json.dumps({
            "emails": found_emails,
            "names": found_names
        })))
        logger.info(f"[MAPS] Found entities: {len(found_names)} names, {len(found_emails)} emails")

    # Combine sample
    sample_json = "\n\n".join([f"[{name}]:\n{content}" for name, content in sample_parts])
    if len(sample_json) < 1000:
        sample_json = json.dumps(input_json)[:5000]

    # Extract source KLOs to adapt
    source_klos = extract_klos_from_json(input_json)
    klos_json = json.dumps(source_klos, indent=2) if source_klos else "[]"

    prompt = f"""Analyze this source simulation JSON and target scenario to generate mapping tables AND adapt the KLOs.

## SOURCE JSON SAMPLE:
{sample_json}

## SOURCE KLOs (Key Learning Outcomes to ADAPT):
{klos_json}

## TARGET SCENARIO:
{scenario_prompt}

## YOUR TASK:
Generate THREE things in ONE response:

1. **entity_map** - Map ALL source entities to new target equivalents:
   - company: source company name → new target company name + email domain (this is the PROTAGONIST company the learner works for)
   - competitor: if there's a competitor mentioned, map it to a new competitor name (e.g., "MegaBurger", "ValueMeals Inc"). IMPORTANT: The word "competitor" itself should NOT be replaced - only specific competitor company names.
   - people: Extract REAL PERSON NAMES ONLY (like "Elizabeth Carter", "John Smith"). Look in emails, manager fields. DO NOT include stage names like "Manager Chat" or "Information Email". Map each person to a NEW invented name.
   - roles: ALL job titles → target domain equivalents

   CRITICAL: Do NOT map generic terms like "competitor", "competitor's", "the competition" to the protagonist company name. These should stay as generic terms OR be mapped to a specific competitor name.

2. **domain_profile** - Map source domain terminology to target:
   - source_domain: What domain/industry is the SOURCE? Identify it from the JSON.
   - target_domain: What domain/industry is the TARGET? Identify it from the scenario prompt.
   - terminology_map: Map key domain-specific terms to target equivalents. Include 15-20 important mappings.
   - forbidden_terms: List ALL company/brand names from the source JSON that must NOT appear in output. Scan the ENTIRE JSON for any company names, brand names, or organization names. Include ALL of them.
   - target_kpis: List of relevant KPIs/metrics for target domain

3. **adapted_klos** - The KLOs rewritten for the TARGET domain:
   - Keep same structure (id, outcome)
   - Transform the outcome text to fit TARGET domain
   - These will guide ALL content adaptation

## OUTPUT FORMAT (valid JSON only):
```json
{{
  "entity_map": {{
    "company": {{
      "source": "Original Company Name",
      "target": "New Company Name",
      "domain": "newcompany.com"
    }},
    "competitor": {{
      "source": "Original Competitor Name or null if generic",
      "target": "New Competitor Name (e.g., MegaBurger)"
    }},
    "people": {{
      "Original Person Name": {{
        "name": "New Person Name",
        "role": "New Role Title",
        "email": "firstname.lastname@newcompany.com"
      }}
    }},
    "roles": {{
      "Original Role": "New Role",
      "Another Role": "Another New Role"
    }}
  }},
  "domain_profile": {{
    "source_domain": "Source Industry/Domain",
    "target_domain": "Target Industry/Domain",
    "terminology_map": {{
      "source_term_1": "target_term_1",
      "source_term_2": "target_term_2"
    }},
    "forbidden_terms": ["term1", "term2", "term3"],
    "target_kpis": ["KPI 1", "KPI 2", "KPI 3"]
  }},
  "adapted_klos": [
    {{"id": "klo1", "outcome": "Adapted KLO text for target domain..."}},
    {{"id": "klo2", "outcome": "Another adapted KLO..."}}
  ]
}}
```

Return ONLY the JSON, no explanations."""

    logger.info("[MAPS] Generating entity_map, domain_profile, AND adapted KLOs (Option B)...")

    try:
        response_text = await call_gemini_async(prompt, expect_json=True, model=DEFAULT_MODEL)
        result = _repair_json(response_text)

        # Parse entity_map
        em = result.get("entity_map", {})
        entity_map = EntityMap(
            company=em.get("company", {}),
            competitor=em.get("competitor", {}),
            people=em.get("people", {}),
            roles=em.get("roles", {})
        )

        # Parse domain_profile
        dp = result.get("domain_profile", {})
        domain_profile = DomainProfile(
            source_domain=dp.get("source_domain", "Unknown"),
            target_domain=dp.get("target_domain", "Unknown"),
            terminology_map=dp.get("terminology_map", {}),
            forbidden_terms=dp.get("forbidden_terms", []),
            target_kpis=dp.get("target_kpis", [])
        )

        # Parse adapted KLOs
        adapted_klos = result.get("adapted_klos", [])

        logger.info(f"[MAPS] Generated: {len(entity_map.people)} people, {len(entity_map.roles)} roles, {len(domain_profile.terminology_map)} term mappings, {len(domain_profile.forbidden_terms)} forbidden terms, {len(adapted_klos)} KLOs")

        # DEBUG: Log actual contents to verify key terms are captured
        logger.info(f"[MAPS] People: {list(entity_map.people.keys())}")
        logger.info(f"[MAPS] Roles: {entity_map.roles}")
        logger.info(f"[MAPS] Terminology (first 20): {dict(list(domain_profile.terminology_map.items())[:20])}")
        logger.info(f"[MAPS] Forbidden terms: {domain_profile.forbidden_terms}")

        return entity_map, domain_profile, adapted_klos

    except Exception as e:
        logger.error(f"[MAPS] Generation failed: {e}")
        # Return empty - adaptation will still work but without strong guidance
        return EntityMap({}, {}, {}, {}), DomainProfile("Unknown", "Unknown", {}, [], []), []


async def extract_company_context_with_llm(adapted_shard: dict) -> CompanyContext:
    """Use LLM to extract company context when regex fails.

    This is a fallback - makes a small LLM call to find the company/manager names
    that were created during adaptation.
    """
    content_str = json.dumps(adapted_shard, indent=2, ensure_ascii=False)[:8000]  # Limit size

    prompt = f"""Look at this adapted JSON and extract the company details that were created.

JSON Content:
```json
{content_str}
```

Find and return:
1. The company/organization name used (NOT "the company" - the actual brand name)
2. The main manager/contact person's full name (NOT "the manager" - the actual name)
3. The manager's email address
4. The industry/domain this is about

Return ONLY valid JSON in this exact format:
{{"company_name": "...", "manager_name": "...", "manager_email": "...", "industry": "..."}}

If you can't find a value, make a reasonable inference from context. Never return "the company" or "the manager"."""

    logger.info("[CONTEXT] Regex extraction failed, using LLM fallback...")

    try:
        response_text = await call_gemini_async(prompt)
        result = _repair_json(response_text)

        company_name = result.get("company_name", "")
        manager_name = result.get("manager_name", "")
        manager_email = result.get("manager_email", "")
        industry = result.get("industry", "the target industry")

        # Validate we didn't get forbidden terms
        if company_name.lower() in ["the company", "company", ""]:
            company_name = "Unnamed Company"  # Last resort - still better than "the company"
        if manager_name.lower() in ["the manager", "manager", ""]:
            manager_name = "Team Lead"  # Last resort

        logger.info(f"[CONTEXT] LLM extracted: company='{company_name}', manager='{manager_name}'")

        return CompanyContext(
            company_name=company_name,
            manager_name=manager_name,
            manager_email=manager_email,
            industry=industry
        )
    except Exception as e:
        logger.error(f"[CONTEXT] LLM extraction failed: {e}")
        # Return placeholder that's still better than "the company"
        return CompanyContext(
            company_name="Target Company",
            manager_name="Project Manager",
            manager_email="contact@company.com",
            industry="the target industry"
        )


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

    # Generic email pattern - captures firstname.lastname@domain.com (agnostic)
    email_match = re.search(r'([a-z]+)\.([a-z]+)@([a-z0-9]+)\.com', content_str, re.IGNORECASE)
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

    # Don't use forbidden fallbacks - return None to trigger LLM fallback
    if not company_name or company_name.lower() in ["the company", "company"]:
        company_name = None
    if not manager_name or manager_name.lower() in ["the manager", "manager"]:
        manager_name = None

    # Industry is agnostic - derived from scenario, not hardcoded
    industry = "the target industry"

    logger.info(f"[CONTEXT] Regex extracted: company='{company_name}', manager='{manager_name}', email='{manager_email}'")

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
    company_context: CompanyContext = None,
    derived_klos: list[dict] = None,
    forbidden_terms: list[str] = None,
    entity_map: EntityMap = None,
    domain_profile: DomainProfile = None,
    questions_for_resources: list[str] = None,  # Questions that resources must support (inference map)
    rag_facts: str = None  # RAG-retrieved industry facts for resources
) -> str:
    """
    Build adaptation prompt using scenario as SINGLE SOURCE OF TRUTH.

    NEW (Option B): If entity_map and domain_profile are provided (from upfront LLM call),
    use those for consistent entity/terminology mapping across ALL shards.
    derived_klos can be adapted KLOs from the upfront call.

    questions_for_resources: For resources shard, include the submission questions that must be
    answerable from the resource data. Resources must contain DATA to derive answers,
    but must NOT give answers directly (inference map concept).

    Legacy: If company_context is provided (from first shard), use those names for consistency.
    Legacy: If forbidden_terms is provided, include explicit list of terms that must be replaced.
    """
    content_str = json.dumps(json_content, indent=2, ensure_ascii=False)

    shard_hint = ""
    if shard_name:
        shard_hint = f'\nThis is the "{shard_name}" section of the simulation.\n'

    # Build sections based on what's provided
    company_section = ""
    derive_section = ""
    forbidden_section = ""

    # NEW: Use entity_map and domain_profile if provided (Option B - preferred)
    if entity_map and domain_profile:
        # Entity mapping section
        # Get company and competitor info
        co_source = entity_map.company.get('source', 'Unknown')
        co_target = entity_map.company.get('target', 'Target Company')
        comp_source = entity_map.competitor.get('source', '') if entity_map.competitor else ''
        comp_target = entity_map.competitor.get('target', 'the competitor') if entity_map.competitor else 'the competitor'

        entity_section = f"""
## ⚠️ TWO DIFFERENT COMPANIES - DO NOT CONFUSE ⚠️

| SOURCE NAME | TARGET NAME | ROLE |
|-------------|-------------|------|
| {co_source} | {co_target} | PROTAGONIST (learner advises) |
| {comp_source} | {comp_target} | COMPETITOR (threat/rival) |

**CRITICAL:** "{co_source}" → "{co_target}" but "{comp_source}" → "{comp_target}"
These are TWO DIFFERENT companies. Never replace "{comp_source}" with "{co_target}"!

**Email domain:** @{entity_map.company.get('domain', 'company.com')}

**People (USE ONLY THESE - no other names allowed):**
"""
        target_names = []
        for source_name, target_info in entity_map.people.items():
            if isinstance(target_info, dict):
                tname = target_info.get('name', 'Unknown')
                target_names.append(tname)
                entity_section += f"- \"{source_name}\" → \"{tname}\" ({target_info.get('role', 'Employee')}, {target_info.get('email', '')})\n"
            else:
                target_names.append(str(target_info))
                entity_section += f"- \"{source_name}\" → \"{target_info}\"\n"

        if target_names:
            entity_section += f"\n⚠️ ONLY USE: {', '.join(target_names)}. Do NOT invent other names.\n"

        entity_section += f"""
**Role Titles (replace ALL occurrences):**
"""
        for source_role, target_role in entity_map.roles.items():
            entity_section += f"- \"{source_role}\" → \"{target_role}\"\n"

        # Domain terminology section
        terminology_section = f"""
## DOMAIN TRANSFORMATION:

**Source Domain:** {domain_profile.source_domain}
**Target Domain:** {domain_profile.target_domain}

**Terminology Mapping (MUST replace ALL occurrences):**
"""
        for source_term, target_term in list(domain_profile.terminology_map.items())[:25]:
            terminology_section += f"- \"{source_term}\" → \"{target_term}\"\n"

        # Forbidden terms - MAKE THIS VERY STRONG
        if domain_profile.forbidden_terms:
            # Show more terms and make it emphatic
            forbidden_list = ", ".join(f'"{t}"' for t in domain_profile.forbidden_terms[:50])
            terminology_section += f"""

## ⛔ FORBIDDEN TERMS - AUTOMATIC REJECTION ⛔

{forbidden_list}

^^^ NONE of these terms may appear ANYWHERE in your output. ^^^

For each forbidden term: replace it with a TARGET DOMAIN equivalent that makes sense in context.

BEFORE OUTPUTTING: Scan your JSON. If ANY forbidden term exists, replace it. Your output is auto-rejected otherwise.

"""

        # Target KPIs
        if domain_profile.target_kpis:
            kpis_list = ", ".join(domain_profile.target_kpis[:10])
            terminology_section += f"""
**Target Domain KPIs/Metrics to use:**
{kpis_list}
"""

        company_section = entity_section + terminology_section

    elif company_context:
        # Legacy: If we have company context from first shard
        company_section = f"""
## COMPANY DETAILS (USE THESE EXACTLY):
- Company Name: {company_context.company_name}
- Manager Name: {company_context.manager_name}
- Manager Email: {company_context.manager_email}
- Industry: {company_context.industry}

**CRITICAL RULES:**
1. Use ONLY these names - no variations, no "the company", no "the manager"
2. Replace ALL original person names with NEW names appropriate for this company
3. ALL emails must be firstname.lastname@{company_context.company_name.lower().replace(' ', '')}.com
4. NEVER keep names from the source JSON - they belong to the old scenario
"""

    if not company_context and not entity_map:
        # Only ask to derive if we don't have any context (shouldn't happen with Option B)
        derive_section = """
## FIRST: CREATE COMPANY IDENTITY (MANDATORY)

You MUST invent these NOW before proceeding:

1. **COMPANY NAME:** A real brand name like "EcoChic Threads" or "Verdant Apparel" (NOT "the company")
2. **MANAGER NAME:** A full name like "Sarah Chen" or "Marcus Rivera" (NOT "the manager")
3. **MANAGER EMAIL:** firstname.lastname@companyname.com (e.g., sarah.chen@ecochic.com)

**FAILURE CONDITIONS (your output will be rejected if):**
- You use "the company" anywhere
- You use "the manager" anywhere
- You keep any original person names (e.g., Elizabeth, Carter, or any name from the input)
- Any email doesn't match the pattern firstname.lastname@newcompany.com
"""

    # KLO alignment section - now can be adapted KLOs from upfront call
    klo_section = ""
    if derived_klos:
        klo_section = format_klos_for_prompt(derived_klos)

    # Legacy: Forbidden terms (only if not using domain_profile which has its own)
    if forbidden_terms and not domain_profile:
        forbidden_section = format_forbidden_terms_for_prompt(forbidden_terms)

    # INFERENCE MAP: For resources shard, include KLOs + questions + RAG facts
    inference_map_section = ""
    if derived_klos:
        klos_list = "\n".join([f"- KLO{i+1}: {klo.get('outcome', '')[:200]}" for i, klo in enumerate(derived_klos[:5])])

        # Build questions section if this is resources shard
        questions_section = ""
        if questions_for_resources:
            q_list = "\n".join([f"- Q{i+1}: {q[:250]}" for i, q in enumerate(questions_for_resources[:8])])
            questions_section = f"""
**QUESTIONS LEARNERS MUST ANSWER (Resources must provide DATA for these):**
{q_list}

**For EACH question above, ensure resources contain specific numerical data (costs, percentages, counts, timelines) that enables analysis.**
**DO NOT give answers directly - provide the raw data learners need to analyze.**
"""

        # Build RAG facts section if available
        rag_section = ""
        if rag_facts:
            rag_section = f"""
**INDUSTRY DATA TO USE (Embed these statistics in your resources):**
{rag_facts}

**Incorporate these industry facts naturally into resource content as market data, industry statistics, and competitor information.**
"""

        inference_map_section = f"""

## RESOURCE ADAPTATION: TRANSFORM DATA FOR TARGET DOMAIN

**KLOs learners must achieve:**
{klos_list}
{questions_section}{rag_section}
**CRITICAL: Transform DATA values, not just terminology**

The source data is for a DIFFERENT domain. You must:
1. Identify what industry/domain the source data represents
2. Transform ALL numbers, metrics, and statistics to be REALISTIC for the TARGET domain
3. Use metrics that are relevant to the TARGET industry
4. Make competitors/entities realistic for the TARGET domain

**Example transformation logic:**
- Source has market share % → Keep structure, adjust % to be realistic for TARGET industry
- Source has revenue figures → Transform to realistic scale for TARGET industry
- Source has customer segments → Replace with segments relevant to TARGET domain
- Source has competitor names → Replace with realistic TARGET domain competitors

**DO NOT keep source domain data literally - transform it to match TARGET reality.**
"""

    return f"""You are adapting a business simulation to a completely different domain.
{inference_map_section}
## TARGET SCENARIO (YOUR SOURCE OF TRUTH):
{scenario_prompt}
{company_section}{derive_section}{klo_section}{forbidden_section}

---

## CRITICAL: PRESERVE LEARNING CONTEXT (CONTEXT FIDELITY)

**THE SCENARIO DEFINES THE LEARNING JOURNEY - PRESERVE IT COMPLETELY**

From the scenario prompt above, identify and PRESERVE these elements:
1. **LEARNER'S GOAL** - What outcome must the learner achieve? (e.g., "develop a market entry strategy")
2. **LEARNER'S CHALLENGE** - What problem must they solve? (e.g., "analyze the sustainable fashion market")
3. **LEARNER'S ROLE** - What role do they play? (e.g., "junior consultant at a strategy firm")
4. **LEARNING OBJECTIVES** - What skills/knowledge should they gain?

**THESE MUST APPEAR IN YOUR OUTPUT:**
- The goal must be clearly stated in overview/description sections
- The challenge must drive the tasks and questions
- The role must be referenced in workplace scenario and emails
- All content must build toward the learning objectives

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

**IMPORTANT FOR CHAT HISTORY / CONVERSATION CONTENT:**
If this JSON contains chat messages, conversation history, or previous dialogue:
- Transform ALL content within messages - treat them as if they were written for the TARGET domain
- Replace ALL domain terminology inside message strings (e.g., "HR" → "market strategy", "interview" → "market analysis")
- Replace ALL person names mentioned in conversations
- The conversation should read as if it ALWAYS was about the TARGET domain

## CRITICAL: PERSON NAMES AND EMAILS (ZERO TOLERANCE)

**FORBIDDEN - NEVER USE THESE:**
- "the manager" / "the Manager" / "The Manager"
- "the company" / "the Company" / "The Company"
- "the team" / "the director" / "the supervisor"
- Any generic placeholder like "Manager", "Company", "Team Lead"

**REQUIRED - YOU MUST:**
1. INVENT a realistic full name for EVERY person (e.g., "Sarah Chen", "Marcus Rivera")
2. EVERY email MUST use format: firstname.lastname@companyname.com
3. Replace ALL original names - they belong to the OLD scenario
4. Characters, managers, senders - ALL need new invented names

**SCAN YOUR OUTPUT:** If you see "the manager" or "the company" anywhere, you have FAILED. Replace with actual names.

---

## WHAT YOU ARE ADAPTING:
{shard_hint}
```json
{content_str}
```

---

## RULES:


### Structure (DO NOT CHANGE - CRITICAL)
- Keep ALL IDs exactly as they are
- Keep ALL object/array structures
- **PRESERVE EXACT KEYS** - Return the SAME keys as in the input JSON
- If input has {{"name": "...", "body": "..."}}, output must have {{"name": "...", "body": "..."}}
- Do NOT invent new keys or use path notation as keys
- Only change content VALUES (strings, descriptions, names)

### Domain Fidelity (CRITICAL)
- ZERO tolerance for source domain terms
- Every domain-specific word must become TARGET domain
- Use industry-appropriate KPIs, metrics, terminology
- If unsure whether a term is domain-specific, replace it

### Content Quality - INFERENCE MAP RULES (CRITICAL)

**RESOURCES MUST PROVIDE "DOTS TO CONNECT" - NOT "CONNECTED DOTS"**

FORBIDDEN in resources (giving answers - AUTO REJECT):
- "should" / "we should" / "you should" / "the company should"
- "recommend" / "recommendation" / "recommended"
- "therefore" / "thus" / "hence" / "consequently"
- "suggests that" / "indicates that" / "implies that"
- "The best approach is..." / "The optimal strategy is..."
- "Based on this data, you should..." / "This means that..."
- "In conclusion..." / "To summarize..."
- Any phrase that TELLS the learner what to decide

REQUIRED in resources (providing data for inference):
- Raw statistics: "Market size: $X billion"
- Percentages: "Category grew 15% YoY"
- Competitor data: "Competitor A holds 23% market share"
- Cost figures: "Production costs: $X per unit"
- Consumer data: "Survey shows 67% prefer X"

**WORD LIMIT: 800-1400 words per resource (STRICT)**
- Under 800 words = TOO SHORT, will be rejected
- Over 1400 words = TOO LONG, will be rejected
- Count your words before outputting

**NO REAL CITATIONS** - Do NOT cite real reports, consulting firms, or studies:
- NEVER cite: McKinsey, BCG, Deloitte, Gartner, Forrester, IBISWorld, Statista, Nielsen, etc.
- ALL data should be presented as internal company research or market observations
- Use: "Internal analysis shows...", "Company research indicates...", "Market data reveals..."

- Questions ask for ANALYSIS (justify, develop, explain) - NOT copy/paste from resources

### Consistency
- ONE company name throughout (derive from scenario)
- ONE manager name throughout (create realistic name for TARGET industry)
- Manager email: firstname.lastname@company.com

### Completeness (CRITICAL - OUTPUT MUST BE COMPLETE)
- NO placeholders like [COMPETITOR_NAME], [COMPANY_NAME], [MANAGER_NAME], [TBD], [INSERT], TODO, XXX
- Use ACTUAL names from the entity mapping above - never use bracketed placeholders
- NO truncated content - every sentence must be complete
- NO "..." or ellipses indicating cut-off content
- ALL content must be complete and realistic
- **OUTPUT LENGTH RULE: Your output must have AT LEAST as many characters as the input**
  - If input has 5000 chars of content, output must have ~5000 chars
  - If you find yourself shortening content, STOP and expand instead
  - Short outputs = AUTOMATIC REJECTION

### Word Counts (STRICT ENFORCEMENT)
- Resources: 800-1400 words EACH (count before outputting)
- Descriptions: At least as long as original
- Emails: 150-300 words
- If original content was long, your adaptation MUST be equally long

---

## OUTPUT:
Return ONLY the adapted JSON. Same structure, completely new domain content.
No explanations. Just valid JSON.

**LENGTH CHECK: Before outputting, verify your JSON is NOT shorter than the input.**

**FINAL CHECK:**
- Keys in output MUST match keys in input exactly
- If input is {{"name": "...", "body": "..."}} then output MUST be {{"name": "...", "body": "..."}}
- If input is an array [...] then output MUST be an array [...]
- Do NOT wrap output in a path-like key"""


async def call_gemini_async(
    prompt: str,
    expect_json: bool = False,
    max_continuations: int = 5,
    model: str = "gemini-3-flash-preview"
) -> str:
    """Call Gemini asynchronously with iterative continuation for large outputs.

    Args:
        prompt: The prompt to send
        expect_json: If True, will continue until valid JSON is received
        max_continuations: Maximum number of continuation requests (default 3)
        model: Model to use (default "gemini-3-flash-preview", use "gemini-2.5-flash" for stability)

    Returns:
        Complete response text (concatenated if multiple calls needed)
    """
    import os
    from google import genai

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY not set")

    client = genai.Client(api_key=api_key)

    # First call
    prompt_size = len(prompt)
    logger.info(f"[GEMINI] Starting API call with {model}, prompt size: {prompt_size} chars (~{prompt_size // 4} tokens)")

    import time as _time
    start_time = _time.time()

    response = await client.aio.models.generate_content(
        model=model,
        contents=prompt,
        config={
            "temperature": 0.0,
            "max_output_tokens": 65536
        }
    )

    elapsed = _time.time() - start_time
    accumulated_text = response.text
    if accumulated_text is None:
        logger.error(f"[GEMINI] API returned None response after {elapsed:.1f}s")
        raise ValueError("Gemini API returned empty response")
    logger.info(f"[GEMINI] API call completed in {elapsed:.1f}s, response size: {len(accumulated_text)} chars")

    # Check if we need to continue (for JSON responses)
    if not expect_json:
        return accumulated_text

    # Check if JSON is complete
    continuation_count = 0
    while continuation_count < max_continuations:
        # Try to parse as JSON
        try:
            # Quick check - does it look complete?
            text = accumulated_text.strip()
            if text.startswith("```"):
                # Extract from code block
                lines = text.split("\n")[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                text = "\n".join(lines)

            json.loads(text)
            # Valid JSON - we're done
            logger.info(f"[GEMINI] Complete JSON received after {continuation_count} continuations")
            return accumulated_text
        except json.JSONDecodeError as e:
            # JSON incomplete - need to continue
            logger.info(f"[GEMINI] JSON incomplete (error at char {e.pos}), requesting continuation {continuation_count + 1}/{max_continuations}")

        # Check finish reason if available
        finish_reason = getattr(response, 'finish_reason', None)
        if finish_reason and finish_reason not in ['MAX_TOKENS', 'STOP', None]:
            # Some other stop reason - don't continue
            logger.warning(f"[GEMINI] Stopping due to finish_reason: {finish_reason}")
            break

        # Request continuation
        continuation_prompt = f"""Continue the JSON output exactly from where you left off.
Your previous output ended with:
...{accumulated_text[-500:]}

Continue from there. Output ONLY the remaining JSON, no explanation. Start exactly where you stopped."""

        continuation_response = await client.aio.models.generate_content(
            model=model,  # Use same model as original request
            contents=continuation_prompt,
            config={
                "temperature": 0.0,
                "max_output_tokens": 65536
            }
        )

        continuation_text = continuation_response.text
        if continuation_text is None:
            logger.warning(f"[GEMINI] Continuation returned None, stopping")
            break
        continuation_text = continuation_text.strip()

        # Remove any markdown wrapper from continuation
        if continuation_text.startswith("```"):
            lines = continuation_text.split("\n")[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            continuation_text = "\n".join(lines)

        # Append continuation
        accumulated_text = accumulated_text.rstrip() + continuation_text
        continuation_count += 1
        response = continuation_response

    if continuation_count >= max_continuations:
        logger.warning(f"[GEMINI] Reached max continuations ({max_continuations}), JSON may be incomplete")

    return accumulated_text


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
    adapted_json, shards_count, errors, entity_map_dict, domain_profile_dict = await _adapt_with_sharding(
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
        entity_map=entity_map_dict,
        domain_profile=domain_profile_dict,
    )


async def _adapt_with_sharding(
    input_json: dict,
    scenario_prompt: str
) -> tuple[dict, int, list, dict, dict]:
    """
    Adapt JSON by sharding into smaller pieces.

    Uses OPTION B: ONE UPFRONT CALL then ALL SHARDS IN PARALLEL
    1. Generate entity_map, domain_profile, AND adapted_klos in ONE LLM call
    2. ALL shards adapt in parallel with full context (no PASS 1/PASS 2)

    This ensures consistent naming AND faster execution.

    Returns:
        (adapted_json, shard_count, errors, entity_map_dict, domain_profile_dict)
    """
    from .sharder import Sharder, merge_shards
    from ..models.shard import LockState
    import time as _time

    logger.info("[SIMPLE ADAPTER] Using OPTION B: One upfront call, then ALL shards in parallel")

    # Shard the JSON
    sharder = Sharder()
    collection = sharder.shard(input_json)

    # Separate locked vs unlocked
    locked_shards = [s for s in collection.shards if s.lock_state == LockState.FULLY_LOCKED]
    unlocked_shards = [s for s in collection.shards if s.lock_state != LockState.FULLY_LOCKED]

    logger.info(f"[SIMPLE ADAPTER] {len(locked_shards)} locked, {len(unlocked_shards)} to adapt")

    errors = []

    # Extract questions for inference map (resources must contain data to answer these)
    questions_for_resources = extract_questions_from_input(input_json)
    logger.info(f"[SIMPLE ADAPTER] Extracted {len(questions_for_resources)} questions for inference map")

    # STEP 0: Generate entity_map, domain_profile, AND adapted KLOs in ONE upfront call
    logger.info("[SIMPLE ADAPTER] STEP 0: Generating entity_map + domain_profile + adapted KLOs (ONE LLM call)...")
    upfront_start = _time.time()

    try:
        entity_map, domain_profile, adapted_klos = await generate_entity_and_domain_maps(
            input_json, scenario_prompt
        )
        upfront_time = _time.time() - upfront_start
        logger.info(f"[SIMPLE ADAPTER] Upfront generation complete in {upfront_time:.1f}s")
        logger.info(f"[SIMPLE ADAPTER]   - Entity map: {len(entity_map.people)} people, {len(entity_map.roles)} roles")
        logger.info(f"[SIMPLE ADAPTER]   - Domain profile: {len(domain_profile.terminology_map)} term mappings, {len(domain_profile.forbidden_terms)} forbidden")
        logger.info(f"[SIMPLE ADAPTER]   - Adapted KLOs: {len(adapted_klos)}")

        # Log the adapted KLOs for debugging
        for i, klo in enumerate(adapted_klos):
            logger.info(f"[SIMPLE ADAPTER]   KLO {i+1}: {klo.get('outcome', '')[:80]}...")

    except Exception as e:
        logger.error(f"[SIMPLE ADAPTER] Upfront generation failed: {e}")
        errors.append(f"upfront_generation: {str(e)}")
        # Fallback to empty maps - shards will still adapt but less consistently
        entity_map = EntityMap({}, {}, {}, {})
        domain_profile = DomainProfile("Unknown", "Unknown", {}, [], [])
        adapted_klos = []

    # ALL SHARDS IN PARALLEL (including resources)
    # Resources gets questions from INPUT JSON (not adapted) - simpler and faster
    logger.info(f"[SIMPLE ADAPTER] Adapting ALL {len(unlocked_shards)} shards in PARALLEL...")

    if unlocked_shards:
        logger.info(f"[SIMPLE ADAPTER] Processing ALL {len(unlocked_shards)} unlocked shards in PARALLEL...")
        logger.info(f"[SIMPLE ADAPTER] Shard IDs: {[s.id for s in unlocked_shards]}")

        # Process shards - split large ones to avoid LLM output truncation
        # 20K threshold balances fewer API calls vs truncation risk
        MAX_SHARD_SIZE = 20000

        tasks = []
        task_info = []  # Track which shard/index each task corresponds to

        for shard in unlocked_shards:  # All shards in parallel
            # Skip shards that don't need adaptation (chat history, etc.)
            if shard.id in SKIP_SHARDS:
                logger.info(f"[SIMPLE ADAPTER] SKIPPING shard '{shard.id}' (in SKIP_SHARDS - just logs)")
                # Keep original content unchanged - update collection directly
                collection.update_shard(shard)
                continue

            logger.info(f"[SIMPLE ADAPTER] Processing shard '{shard.id}': content type={type(shard.content).__name__}, keys={list(shard.content.keys()) if isinstance(shard.content, dict) else 'N/A'}")
            shard_size = len(json.dumps(shard.content))
            split_done = False

            # Select model - use stable model for problematic shards (e.g., resources)
            shard_model = STABLE_MODEL if shard.id in SHARDS_USING_STABLE_MODEL else DEFAULT_MODEL
            if shard.id in SHARDS_USING_STABLE_MODEL:
                logger.info(f"[SIMPLE ADAPTER] Using STABLE model ({shard_model}) for shard '{shard.id}'")

            # Check if this shard should NEVER be split (e.g., resources need full context)
            if shard.id in NEVER_SPLIT_SHARDS:
                # Use ADAPTATION for all shards (preserves structure better than generation)
                logger.info(f"[SIMPLE ADAPTER] Shard '{shard.id}' in NEVER_SPLIT_SHARDS - ADAPTING as WHOLE ({shard_size} chars)")
                task = _adapt_single_shard_simple(shard, scenario_prompt, derived_klos=adapted_klos, entity_map=entity_map, domain_profile=domain_profile, questions_for_resources=questions_for_resources)
                tasks.append(task)
                task_info.append({"shard": shard, "type": "whole", "index": None, "key": None})
                continue  # Skip splitting logic entirely

            # Check if this shard needs splitting
            if shard_size > MAX_SHARD_SIZE:
                # Case 1: Content IS a list
                if isinstance(shard.content, list) and len(shard.content) > 1:
                    logger.info(f"[SIMPLE ADAPTER] SPLITTING list shard '{shard.name}' ({shard_size} chars, {len(shard.content)} items)")
                    for idx, item in enumerate(shard.content):
                        task = _adapt_single_item(item, scenario_prompt, f"{shard.name}[{idx}]", adapted_klos, entity_map, domain_profile, model=shard_model)
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
                                                            task = _adapt_single_item(deepval, scenario_prompt, f"{shard.name}.{key}[{idx}].{subkey}.{deepkey}", adapted_klos, entity_map, domain_profile, model=shard_model)
                                                            tasks.append(task)
                                                            task_info.append({"shard": shard, "type": "deeper_item", "index": idx, "key": key, "subkey": subkey, "deepkey": deepkey})
                                                        elif deepval_size > 50:
                                                            # Collect in medium_fields for batch adaptation
                                                            medium_fields[f"{subkey}.{deepkey}"] = deepval
                                                else:
                                                    # Large non-dict (like big array) - adapt individually
                                                    task = _adapt_single_item(subval, scenario_prompt, f"{shard.name}.{key}[{idx}].{subkey}", adapted_klos, entity_map, domain_profile, model=shard_model)
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
                                                    task = _adapt_single_item(current_batch, scenario_prompt, f"{shard.name}.{key}[{idx}].batch_{batch_num}", adapted_klos, entity_map, domain_profile, model=shard_model)
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
                                                task = _adapt_single_item(current_batch, scenario_prompt, f"{shard.name}.{key}[{idx}].batch_{batch_num}", adapted_klos, entity_map, domain_profile, model=shard_model)
                                                tasks.append(task)
                                                task_info.append({"shard": shard, "type": "batch_fields", "index": idx, "key": key, "fields": list(current_batch.keys())})
                                    else:
                                        task = _adapt_single_item(item, scenario_prompt, f"{shard.name}.{key}[{idx}]", adapted_klos, entity_map, domain_profile, model=shard_model)
                                        tasks.append(task)
                                        task_info.append({"shard": shard, "type": "nested_item", "index": idx, "key": key})
                                split_done = True
                                break

            if not split_done:
                logger.info(f"[SIMPLE ADAPTER] Shard '{shard.id}' NOT split ({shard_size} chars <= {MAX_SHARD_SIZE}), adapting as WHOLE")
                task = _adapt_single_shard_simple(shard, scenario_prompt, derived_klos=adapted_klos, entity_map=entity_map, domain_profile=domain_profile, questions_for_resources=questions_for_resources)
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
                logger.debug(f"[SIMPLE ADAPTER] batch_fields for [{idx}]: expected={fields}")

                # Check if LLM wrapped the response - unwrap if needed
                # Handle cases like {"batch_0": {...actual content...}}
                if isinstance(result, dict):
                    # Check for wrapper keys like "batch_0", "batch_1", etc.
                    import re
                    for k in list(result.keys()):
                        if re.match(r'batch_\d+', k) and isinstance(result[k], dict):
                            logger.debug(f"[SIMPLE ADAPTER] Unwrapping batch wrapper key '{k}'")
                            result = result[k]
                            break
                    # Also check for single-key wrapper where key not in expected fields
                    if len(result) == 1:
                        only_key = list(result.keys())[0]
                        if only_key not in fields and isinstance(result[only_key], dict):
                            logger.debug(f"[SIMPLE ADAPTER] Unwrapping single-key wrapper '{only_key}'")
                            result = result[only_key]

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
                                        logger.warning(f"[SIMPLE ADAPTER] FAILED to place {field_name}: parent_key={parent_key} not in item or not dict")
                                else:
                                    if field_name in item:
                                        item[field_name] = adapted_value
                                        matched += 1
                                    else:
                                        logger.warning(f"[SIMPLE ADAPTER] FAILED to place {field_name}: not in item")
                            collection.update_shard(shard)
                            logger.info(f"[SIMPLE ADAPTER] Reassembled {info['type']} for [{idx}]: {matched}/{len(result)} fields matched")

    # Merge back
    adapted_json = merge_shards(collection, input_json)

    # Count total shards processed
    total_shards = len(unlocked_shards)

    # Convert entity_map and domain_profile to dicts for result
    entity_map_dict = {
        "company": entity_map.company if entity_map else {},
        "competitor": entity_map.competitor if entity_map else {},
        "people": entity_map.people if entity_map else {},
        "roles": entity_map.roles if entity_map else {}
    }
    domain_profile_dict = {
        "source_domain": domain_profile.source_domain if domain_profile else "",
        "target_domain": domain_profile.target_domain if domain_profile else "",
        "terminology_map": domain_profile.terminology_map if domain_profile else {},
        "forbidden_terms": domain_profile.forbidden_terms if domain_profile else []
    }

    # POST-PROCESSING: Final cleanup of any remaining forbidden terms
    if domain_profile and domain_profile.forbidden_terms:
        # Build replacement map using terminology_map (has correct mappings for both company AND competitor)
        replacement_map = {}
        term_map = domain_profile.terminology_map or {}
        target_company = entity_map.company.get('target', '') if entity_map else ''
        target_competitor = entity_map.competitor.get('target', '') if entity_map else ''

        for term in domain_profile.forbidden_terms:
            if term:
                # First check terminology_map for the correct replacement
                if term in term_map:
                    replacement_map[term] = term_map[term]
                # Check if it's the competitor source name
                elif entity_map and entity_map.competitor.get('source') == term:
                    replacement_map[term] = target_competitor
                # Default to target company
                else:
                    replacement_map[term] = target_company if target_company else ""

        adapted_json = cleanup_forbidden_terms(adapted_json, domain_profile.forbidden_terms, replacement_map)
        logger.info(f"[POST-PROCESS] Cleaned {len(domain_profile.forbidden_terms)} forbidden terms")

    return adapted_json, total_shards, errors, entity_map_dict, domain_profile_dict


@traceable(name="adapt_item", run_type="llm")
async def _adapt_single_item(
    item: dict,
    scenario_prompt: str,
    item_name: str,
    derived_klos: list[dict] = None,
    entity_map: EntityMap = None,
    domain_profile: DomainProfile = None,
    model: str = DEFAULT_MODEL
) -> dict:
    """Adapt a single array item (e.g., one stage from simulationFlow).

    Uses entity_map and domain_profile from upfront generation (Option B).
    """
    input_size = len(json.dumps(item))
    logger.info(f"[ADAPT ITEM] >>> START: {item_name} ({input_size} chars)")

    # Build prompt for single item with Option B maps
    prompt = build_simple_prompt(
        scenario_prompt,
        item,
        shard_name=item_name,
        derived_klos=derived_klos,
        entity_map=entity_map,
        domain_profile=domain_profile
    )

    # Call Gemini with JSON continuation support
    logger.info(f"[ADAPT ITEM] Calling Gemini ({model}) for: {item_name}")
    response_text = await call_gemini_async(prompt, expect_json=True, model=model)
    logger.info(f"[ADAPT ITEM] Gemini returned for: {item_name}")
    raw_size = len(response_text)

    # Parse response
    adapted_content = _repair_json(response_text)

    # Check for truncation
    input_size = len(json.dumps(item))
    output_size = len(json.dumps(adapted_content))

    logger.info(f"[ADAPT ITEM] <<< DONE: {item_name} (input={input_size}, output={output_size})")

    if output_size < input_size * 0.7:
        logger.warning(f"[ADAPT ITEM] {item_name} may be truncated (raw {raw_size} -> parsed {output_size})")

    return adapted_content


@traceable(name="generate_resources", run_type="llm")
async def _generate_resources_shard(
    shard,
    scenario_prompt: str,
    derived_klos: list[dict] = None,
    entity_map: EntityMap = None,
    domain_profile: DomainProfile = None,
    questions_for_resources: list[str] = None
) -> dict:
    """GENERATE resources content (not just adapt) with data for questions/KLOs.

    KEY INSIGHT: Instead of passing original questions (which won't match adapted content),
    we pass ADAPTED KLOs and ask the LLM to:
    1. Infer what questions would assess those KLOs
    2. Generate data that supports answering those questions

    This keeps single-phase execution (fast) while aligning resources to KLOs.
    """
    input_size = len(json.dumps(shard.content))
    logger.info(f"[GENERATE RESOURCES] >>> START: Generating resources with KLO-aligned data ({input_size} chars structure)")

    # Get structure from original - we'll generate new content keeping this structure
    structure_str = json.dumps(shard.content, indent=2)

    # Build KLOs section with full detail
    klos_section = ""
    if derived_klos:
        klos_items = []
        for i, klo in enumerate(derived_klos):
            outcome = klo.get('outcome', '')
            if outcome:
                klos_items.append(f"**KLO {i+1}:** {outcome}")
        klos_section = "\n".join(klos_items)
        logger.info(f"[GENERATE RESOURCES] Using {len(derived_klos)} adapted KLOs for data alignment")

    # Build QUESTIONS section - THIS IS CRITICAL FOR SOLVABILITY
    questions_section = ""
    if questions_for_resources:
        questions_items = []
        for i, q in enumerate(questions_for_resources[:10]):
            questions_items.append(f"**Q{i+1}:** {q[:300]}")
        questions_section = "\n".join(questions_items)
        logger.info(f"[GENERATE RESOURCES] Including {len(questions_for_resources)} questions for solvability")

    # Company info from entity map or domain profile
    company_name = "the company"
    target_domain = "the target domain"

    # Try entity_map first
    if entity_map and entity_map.company:
        company_name = entity_map.company.get('target', 'the company')

    # Fallback: check terminology_map for company name (often has Source->Target mapping)
    if company_name == "the company" and domain_profile and domain_profile.terminology_map:
        # Look for company name patterns in terminology map values
        for source_term, target_term in domain_profile.terminology_map.items():
            # Company names are usually capitalized multi-word terms
            if isinstance(target_term, str) and len(target_term) > 3:
                # Check if it looks like a company name (capitalized, short)
                words = target_term.split()
                if 1 <= len(words) <= 3 and all(w[0].isupper() for w in words if w):
                    company_name = target_term
                    logger.info(f"[GENERATE RESOURCES] Found company name in terminology_map: {company_name}")
                    break

    # Final fallback: extract from scenario prompt
    if company_name == "the company" and scenario_prompt:
        import re
        # Look for patterns like "brand called 'X'" or "company X"
        match = re.search(r'(?:brand|company)\s+(?:called\s+)?["\']?([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?)["\']?', scenario_prompt)
        if match:
            company_name = match.group(1)
            logger.info(f"[GENERATE RESOURCES] Extracted company name from scenario: {company_name}")

    if domain_profile:
        target_domain = domain_profile.target_domain

    logger.info(f"[GENERATE RESOURCES] Using company name: {company_name}")

    prompt = f"""## ⚠️ CRITICAL: USE "{company_name}" AS THE COMPANY NAME THROUGHOUT ⚠️

NEVER write "The company" or "the company" - ALWAYS use "{company_name}" by name.
Every reference to the company MUST say "{company_name}".

---

You are GENERATING a business simulation resource with DATA for students to analyze.

## SCENARIO:
{scenario_prompt}

## COMPANY: {company_name} (USE THIS NAME IN EVERY PARAGRAPH)

## QUESTIONS TO ANSWER (resource must contain data for each):
{questions_section if questions_section else "Generate comprehensive market/financial data"}

## KLOs TO SUPPORT:
{klos_section if klos_section else "Support general business analysis skills"}

## WHAT TO INCLUDE:

For each question, include SPECIFIC DATA:
- Market questions → market size ($), growth %, competitor shares
- Digital questions → app downloads, user engagement, feature stats
- Operations questions → costs, capacity, efficiency metrics
- Financial questions → revenue, margins, investment, ROI
- Risk questions → risk factors with probability and impact

## DATA GENERATION RULES:

### MUST include (raw data for analysis):
- Specific numbers with units (dollar amounts, counts, durations)
- Percentages and ratios
- Comparisons between entities/options/time periods
- Trends showing change over time
- Breakdowns and segmentation

### MUST NOT include (conclusions/answers):
- "Therefore..." / "Thus..." / "Hence..."
- "The company should..." / "We recommend..."
- "This suggests that..." / "This indicates..."
- Any statement that TELLS the learner what to decide

## STRUCTURE TO MAINTAIN:
Keep the EXACT same JSON structure as below. Replace content values with your generated data:

```json
{structure_str[:12000]}
```

## OUTPUT REQUIREMENTS:
1. Same JSON structure as input
2. 800-1400 words of substantive content
3. High data density (many specific numbers/percentages)
4. Data must support analysis for EACH KLO listed above
5. Facts only - no recommendations or conclusions

## OUTPUT:
Return ONLY valid JSON. Same structure, new data-rich content."""

    try:
        response_text = await call_gemini_async(prompt, expect_json=True, model=STABLE_MODEL)
        generated_content = _repair_json(response_text)

        output_size = len(json.dumps(generated_content))
        logger.info(f"[GENERATE RESOURCES] <<< DONE: Generated {output_size} chars (was {input_size} chars)")

        # Log data density check
        content_str = json.dumps(generated_content)
        import re
        numbers_found = len(re.findall(r'\d+(?:\.\d+)?%|\$\d+(?:,\d{3})*(?:\.\d+)?|\d+(?:,\d{3})+', content_str))
        logger.info(f"[GENERATE RESOURCES] Data density: {numbers_found} numbers/percentages found")

        return generated_content
    except Exception as e:
        logger.error(f"[GENERATE RESOURCES] Failed: {e}, falling back to adaptation")
        # Fallback to regular adaptation
        return await _adapt_single_shard_simple(
            shard, scenario_prompt, derived_klos, entity_map, domain_profile, questions_for_resources
        )


@traceable(name="adapt_shard", run_type="llm")
async def _adapt_single_shard_simple(
    shard,
    scenario_prompt: str,
    derived_klos: list[dict] = None,
    entity_map: EntityMap = None,
    domain_profile: DomainProfile = None,
    questions_for_resources: list[str] = None  # For resources shard - inference map
) -> dict:
    """Adapt a single shard using scenario prompt as source of truth.

    Uses entity_map and domain_profile from upfront generation (Option B).
    derived_klos are already adapted KLOs from the same upfront call.
    questions_for_resources: For resources shard only - questions that must be answerable.
    """
    input_size = len(json.dumps(shard.content))
    logger.info(f"[ADAPT SHARD] >>> START: {shard.id} ({input_size} chars)")

    # Select model - use stable model for problematic shards
    model = STABLE_MODEL if shard.id in SHARDS_USING_STABLE_MODEL else DEFAULT_MODEL

    # Build prompt with Option B maps
    # Only pass questions_for_resources if this is the resources shard
    questions_param = questions_for_resources if shard.id == "resources" else None
    rag_facts = None

    if shard.id == "resources":
        if questions_param:
            logger.info(f"[ADAPT SHARD] >>> INFERENCE MAP: Passing {len(questions_param)} questions to resources shard")

        # Get RAG facts for resources shard
        try:
            rag_facts = await _get_rag_facts(
                scenario=scenario_prompt,
                questions=questions_param,
                klos=derived_klos
            )
            if rag_facts:
                logger.info(f"[ADAPT SHARD] >>> RAG: Retrieved {len(rag_facts.split(chr(10)))} facts for resources")
            else:
                logger.info("[ADAPT SHARD] >>> RAG: No relevant facts found")
        except Exception as e:
            logger.warning(f"[ADAPT SHARD] >>> RAG failed: {e}")

    prompt = build_simple_prompt(
        scenario_prompt,
        shard.content,
        shard_name=shard.name,
        derived_klos=derived_klos,
        entity_map=entity_map,
        domain_profile=domain_profile,
        questions_for_resources=questions_param,
        rag_facts=rag_facts
    )

    # Call Gemini
    logger.info(f"[ADAPT SHARD] Calling Gemini ({model}) for: {shard.id}")
    response_text = await call_gemini_async(prompt, expect_json=True, model=model)
    logger.info(f"[ADAPT SHARD] Gemini returned for: {shard.id}")

    # Parse response
    adapted_content = _repair_json(response_text)

    output_size = len(json.dumps(adapted_content))
    if output_size < input_size * 0.7:
        logger.warning(f"[ADAPT SHARD] {shard.id} may be truncated: {input_size} -> {output_size}")

    logger.info(f"[ADAPT SHARD] <<< DONE: {shard.id} (input={input_size}, output={output_size})")
    return adapted_content


# =============================================================================
# WebSocket-enabled adaptation with progress streaming
# =============================================================================

async def adapt_simple_with_progress(
    input_json: dict,
    scenario_prompt: str,
    progress_callback: callable = None,
) -> SimpleAdaptationResult:
    """
    Simple adaptation with progress callbacks for WebSocket streaming.

    Same as adapt_simple but emits progress events via callback.

    Args:
        input_json: The simulation JSON to adapt
        scenario_prompt: Description of the target scenario (source of truth)
        progress_callback: Async callback function(event_type, data) for progress updates

    Returns:
        SimpleAdaptationResult with adapted JSON
    """
    start_time = time.time()
    input_chars = len(json.dumps(input_json))

    async def emit(event_type: str, data: dict):
        if progress_callback:
            await progress_callback(event_type, data)

    logger.info(f"[SIMPLE ADAPTER] Starting PARALLEL shard adaptation (with progress)")
    logger.info(f"[SIMPLE ADAPTER] Input size: {input_chars} chars")
    logger.info(f"[SIMPLE ADAPTER] Scenario: {scenario_prompt[:100]}...")

    await emit("progress", {
        "stage": "sharding",
        "message": "Splitting JSON into shards",
        "data": {"input_chars": input_chars}
    })

    # Run adaptation with progress tracking
    adapted_json, shards_count, errors, entity_map_dict, domain_profile_dict = await _adapt_with_sharding_progress(
        input_json, scenario_prompt, emit
    )

    time_ms = int((time.time() - start_time) * 1000)
    output_chars = len(json.dumps(adapted_json))

    await emit("progress", {
        "stage": "complete",
        "message": "Adaptation complete",
        "data": {
            "time_ms": time_ms,
            "output_chars": output_chars,
            "shards_processed": shards_count
        }
    })

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
        entity_map=entity_map_dict,
        domain_profile=domain_profile_dict,
    )


async def _adapt_with_sharding_progress(
    input_json: dict,
    scenario_prompt: str,
    emit: callable
) -> tuple[dict, int, list, dict, dict]:
    """
    Adapt JSON by sharding with progress callbacks.

    Same as _adapt_with_sharding but emits progress events.
    """
    from .sharder import Sharder, merge_shards
    from ..models.shard import LockState
    import time as _time

    logger.info("[SIMPLE ADAPTER] Using OPTION B with progress: One upfront call, then ALL shards in parallel")

    # Shard the JSON
    sharder = Sharder()
    collection = sharder.shard(input_json)

    # Separate locked vs unlocked
    locked_shards = [s for s in collection.shards if s.lock_state == LockState.FULLY_LOCKED]
    unlocked_shards = [s for s in collection.shards if s.lock_state != LockState.FULLY_LOCKED]

    await emit("progress", {
        "stage": "sharded",
        "message": f"JSON split into {len(collection.shards)} shards",
        "data": {
            "total_shards": len(collection.shards),
            "locked_shards": len(locked_shards),
            "unlocked_shards": len(unlocked_shards),
            "shard_ids": [s.id for s in unlocked_shards]
        }
    })

    logger.info(f"[SIMPLE ADAPTER] {len(locked_shards)} locked, {len(unlocked_shards)} to adapt")

    errors = []

    # Extract questions for inference map
    questions_for_resources = extract_questions_from_input(input_json)
    logger.info(f"[SIMPLE ADAPTER] Extracted {len(questions_for_resources)} questions for inference map")

    # STEP 0: Generate entity_map, domain_profile, AND adapted KLOs
    await emit("progress", {
        "stage": "entity_mapping",
        "message": "Generating entity map and domain profile",
        "data": {}
    })

    upfront_start = _time.time()

    try:
        entity_map, domain_profile, adapted_klos = await generate_entity_and_domain_maps(
            input_json, scenario_prompt
        )
        upfront_time = _time.time() - upfront_start

        await emit("progress", {
            "stage": "entity_mapping_complete",
            "message": "Entity and domain mapping complete",
            "data": {
                "time_ms": int(upfront_time * 1000),
                "people_mapped": len(entity_map.people),
                "roles_mapped": len(entity_map.roles),
                "term_mappings": len(domain_profile.terminology_map),
                "forbidden_terms": len(domain_profile.forbidden_terms),
                "klos_adapted": len(adapted_klos)
            }
        })

        logger.info(f"[SIMPLE ADAPTER] Upfront generation complete in {upfront_time:.1f}s")

    except Exception as e:
        logger.error(f"[SIMPLE ADAPTER] Upfront generation failed: {e}")
        errors.append(f"upfront_generation: {str(e)}")
        entity_map = EntityMap({}, {}, {}, {})
        domain_profile = DomainProfile("Unknown", "Unknown", {}, [], [])
        adapted_klos = []

        await emit("progress", {
            "stage": "entity_mapping_failed",
            "message": f"Entity mapping failed: {str(e)}",
            "data": {"error": str(e)}
        })

    # ALL SHARDS IN PARALLEL
    await emit("progress", {
        "stage": "adapting_shards",
        "message": f"Adapting {len(unlocked_shards)} shards in parallel",
        "data": {"shard_count": len(unlocked_shards)}
    })

    if unlocked_shards:
        logger.info(f"[SIMPLE ADAPTER] Processing ALL {len(unlocked_shards)} unlocked shards in PARALLEL...")

        MAX_SHARD_SIZE = 20000
        tasks = []
        task_info = []
        shard_start_times = {}

        for shard in unlocked_shards:
            if shard.id in SKIP_SHARDS:
                logger.info(f"[SIMPLE ADAPTER] SKIPPING shard '{shard.id}'")
                collection.update_shard(shard)
                continue

            shard_size = len(json.dumps(shard.content))
            shard_model = STABLE_MODEL if shard.id in SHARDS_USING_STABLE_MODEL else DEFAULT_MODEL

            # Emit shard start
            await emit("shard_start", {
                "shard_id": shard.id,
                "shard_name": shard.name,
                "size_chars": shard_size,
                "model": shard_model
            })

            shard_start_times[shard.id] = _time.time()

            # Check if shard should NEVER be split
            if shard.id in NEVER_SPLIT_SHARDS:
                task = _adapt_single_shard_simple(shard, scenario_prompt, derived_klos=adapted_klos, entity_map=entity_map, domain_profile=domain_profile, questions_for_resources=questions_for_resources)
                tasks.append(task)
                task_info.append({"shard": shard, "type": "whole", "index": None, "key": None})
                continue

            # Check if shard needs splitting
            split_done = False
            if shard_size > MAX_SHARD_SIZE:
                if isinstance(shard.content, list) and len(shard.content) > 1:
                    logger.info(f"[SIMPLE ADAPTER] SPLITTING list shard '{shard.name}'")
                    for idx, item in enumerate(shard.content):
                        task = _adapt_single_item(item, scenario_prompt, f"{shard.name}[{idx}]", adapted_klos, entity_map, domain_profile, model=shard_model)
                        tasks.append(task)
                        task_info.append({"shard": shard, "type": "list_item", "index": idx, "key": None})
                    split_done = True
                elif isinstance(shard.content, dict):
                    for key, val in shard.content.items():
                        if isinstance(val, list) and len(val) > 1:
                            array_size = len(json.dumps(val))
                            if array_size > MAX_SHARD_SIZE * 0.5:
                                logger.info(f"[SIMPLE ADAPTER] SPLITTING nested array '{shard.name}.{key}'")
                                for idx, item in enumerate(val):
                                    task = _adapt_single_item(item, scenario_prompt, f"{shard.name}.{key}[{idx}]", adapted_klos, entity_map, domain_profile, model=shard_model)
                                    tasks.append(task)
                                    task_info.append({"shard": shard, "type": "nested_item", "index": idx, "key": key})
                                split_done = True
                                break

            if not split_done:
                task = _adapt_single_shard_simple(shard, scenario_prompt, derived_klos=adapted_klos, entity_map=entity_map, domain_profile=domain_profile, questions_for_resources=questions_for_resources)
                tasks.append(task)
                task_info.append({"shard": shard, "type": "whole", "index": None, "key": None})

        # Execute all tasks in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Track completed shards for progress
        completed_shards = set()

        # Process results
        for i, result in enumerate(results):
            info = task_info[i]
            shard = info["shard"]

            if isinstance(result, Exception):
                logger.error(f"[SIMPLE ADAPTER] Task {i} failed: {result}")
                errors.append(f"{shard.id}: {str(result)}")

                await emit("shard_error", {
                    "shard_id": shard.id,
                    "error": str(result)
                })
            else:
                # Update shard content based on type
                if info["type"] == "whole":
                    shard.content = result
                    shard.current_hash = ""
                    collection.update_shard(shard)
                elif info["type"] == "list_item":
                    if isinstance(shard.content, list) and info["index"] < len(shard.content):
                        shard.content[info["index"]] = result
                        collection.update_shard(shard)
                elif info["type"] == "nested_item":
                    key = info["key"]
                    idx = info["index"]
                    if isinstance(shard.content, dict) and key in shard.content:
                        if isinstance(shard.content[key], list) and idx < len(shard.content[key]):
                            shard.content[key][idx] = result
                            collection.update_shard(shard)

                # Emit shard complete (only once per shard)
                if shard.id not in completed_shards:
                    completed_shards.add(shard.id)
                    elapsed = int((_time.time() - shard_start_times.get(shard.id, _time.time())) * 1000)
                    await emit("shard_complete", {
                        "shard_id": shard.id,
                        "shard_name": shard.name,
                        "time_ms": elapsed,
                        "completed": len(completed_shards),
                        "total": len(unlocked_shards)
                    })

    # Merge back
    await emit("progress", {
        "stage": "merging",
        "message": "Merging adapted shards back into JSON",
        "data": {}
    })

    adapted_json = merge_shards(collection, input_json)

    total_shards = len(unlocked_shards)

    # Convert entity_map and domain_profile to dicts for result
    entity_map_dict = {
        "company": entity_map.company if entity_map else {},
        "competitor": entity_map.competitor if entity_map else {},
        "people": entity_map.people if entity_map else {},
        "roles": entity_map.roles if entity_map else {}
    }
    domain_profile_dict = {
        "source_domain": domain_profile.source_domain if domain_profile else "",
        "target_domain": domain_profile.target_domain if domain_profile else "",
        "terminology_map": domain_profile.terminology_map if domain_profile else {},
        "forbidden_terms": domain_profile.forbidden_terms if domain_profile else []
    }

    # POST-PROCESSING: Final cleanup of any remaining forbidden terms
    if domain_profile and domain_profile.forbidden_terms:
        # Build replacement map using terminology_map (has correct mappings for both company AND competitor)
        replacement_map = {}
        term_map = domain_profile.terminology_map or {}
        target_company = entity_map.company.get('target', '') if entity_map else ''
        target_competitor = entity_map.competitor.get('target', '') if entity_map else ''

        for term in domain_profile.forbidden_terms:
            if term:
                # First check terminology_map for the correct replacement
                if term in term_map:
                    replacement_map[term] = term_map[term]
                # Check if it's the competitor source name
                elif entity_map and entity_map.competitor.get('source') == term:
                    replacement_map[term] = target_competitor
                # Default to target company
                else:
                    replacement_map[term] = target_company if target_company else ""

        adapted_json = cleanup_forbidden_terms(adapted_json, domain_profile.forbidden_terms, replacement_map)
        logger.info(f"[POST-PROCESS] Cleaned {len(domain_profile.forbidden_terms)} forbidden terms")

    return adapted_json, total_shards, errors, entity_map_dict, domain_profile_dict


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
    result = await adapt_simple(input_json, scenario_prompt)

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