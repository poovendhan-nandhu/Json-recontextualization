"""
Simple Adapter - V2 Implementation with Skeleton-Based Generation

NEW APPROACH (V2):
- Stage 0: Generate entity_map, domain_profile, adapted_klos, alignment_map, canonical_numbers, resource_sections
- Programmatic: Extract skeleton (structure only, no content) and word_targets from source
- Stage 1: Generate content to FILL skeleton (not adapt source content)
- Post-process: Enforce entity_map and canonical_numbers consistency

OLD APPROACH (V1 - still available):
- Scenario Prompt + JSON -> LLM -> Adapted JSON
- Shows source content to LLM, asks to "adapt"

Usage:
    from src.stages.simple_adapter import adapt_simple

    # V2 (skeleton-based generation) - default
    result = await adapt_simple(
        input_json=my_json,
        scenario_prompt="learners will act as a junior consultant...",
        use_v2=True  # default
    )

    # V1 (legacy adaptation)
    result = await adapt_simple(
        input_json=my_json,
        scenario_prompt="...",
        use_v2=False
    )
"""
import json
import logging
import sys
import time
import asyncio
from dataclasses import dataclass, field
from typing import Optional

# V2 imports - skeleton-based generation
from ..extractors.skeleton_extractor import extract_skeleton, extract_structure_summary, get_shard_skeleton
from ..extractors.word_target_extractor import measure_word_targets
from ..generators.stage0_generator import generate_stage0_content, validate_stage0_output, get_alignment_for_shard, Stage0Result
from ..prompts.shard_prompts import build_shard_prompt, CONTENT_RULES
from ..enforcers.post_processor import post_process_adapted_json

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


@dataclass
class SimpleAdaptationResult:
    """Result from simple adaptation."""
    adapted_json: dict
    scenario_prompt: str
    time_ms: int
    input_chars: int
    output_chars: int
    mode: str = "monolithic"  # or "sharded" or "skeleton_v2"
    shards_processed: int = 0
    errors: list = field(default_factory=list)
    # V2 additions
    entity_map: dict = field(default_factory=dict)
    domain_profile: dict = field(default_factory=dict)
    alignment_map: dict = field(default_factory=dict)
    canonical_numbers: dict = field(default_factory=dict)


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


def extract_klos_from_json(input_json: dict) -> list[dict]:
    """Extract Key Learning Outcomes from the JSON - these will be adapted to target scenario."""
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

    Works with both the shard content directly (list of KLOs) or
    wrapped in topicWizardData structure.
    """
    klos = []

    # Case 1: Content is a list of KLO objects directly
    if isinstance(adapted_content, list):
        for item in adapted_content:
            if isinstance(item, dict):
                outcome = item.get("keyLearningOutcome", item.get("title", item.get("outcome", "")))
                if outcome:
                    klos.append({"id": item.get("id", ""), "outcome": outcome})
        return klos

    # Case 2: Content is a dict - look for KLOs inside
    if isinstance(adapted_content, dict):
        # Check common KLO container keys (including dot-notation from sharder)
        klo_keys = [
            "assessmentCriterion",
            "selectedAssessmentCriterion",
            "klos",
            "outcomes",
            "topicWizardData.assessmentCriterion",  # Dot-notation from sharder
            "topicWizardData.selectedAssessmentCriterion",
        ]
        for key in klo_keys:
            if key in adapted_content and isinstance(adapted_content[key], list):
                return extract_klos_from_adapted_shard(adapted_content[key])

        # Check if it's a single KLO object
        outcome = adapted_content.get("keyLearningOutcome", adapted_content.get("title", ""))
        if outcome:
            klos.append({"id": adapted_content.get("id", ""), "outcome": outcome})

    return klos


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
    topic_data = input_json.get("topicWizardData", {})

    # Lesson info (has company, scenario context)
    if topic_data.get("lessonInformation"):
        sample_parts.append(("lessonInformation", json.dumps(topic_data["lessonInformation"])[:1500]))

    # Workplace scenario (reveals domain clearly + has manager name)
    if topic_data.get("workplaceScenario"):
        sample_parts.append(("workplaceScenario", json.dumps(topic_data["workplaceScenario"])[:2000]))

    # Characters (has names, roles) - check ALL stages
    sim_flow = topic_data.get("simulationFlow", [])
    for stage in sim_flow:  # Check ALL stages for users
        if stage.get("data", {}).get("activityData", {}).get("selectedValue", {}).get("users"):
            users = stage["data"]["activityData"]["selectedValue"]["users"]
            sample_parts.append(("characters", json.dumps(users)[:1500]))
            break

    # Extract emails and names from the entire JSON for entity discovery
    content_str = json.dumps(topic_data, ensure_ascii=False)
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
   - company: source company name → new target company name + email domain
   - people: Extract ALL person names from the source (check emails, manager fields, etc.) and map each to a NEW invented name with role and email. MUST have at least 3-5 people mappings.
   - roles: ALL job titles → target domain equivalents (e.g., "HR Manager" → "Market Strategy Director")

2. **domain_profile** - Map source domain terminology to target:
   - source_domain: What domain/industry is the SOURCE? Identify it from the JSON.
   - target_domain: What domain/industry is the TARGET? Identify it from the scenario prompt.
   - terminology_map: Scan the source JSON and extract EVERY domain-specific term. Map each to target equivalent. MUST have at least 30 mappings including all variations (singular/plural, verb forms).
   - forbidden_terms: List EVERY term from the source domain that must NEVER appear in adapted output. Scan the JSON thoroughly - include department names, process names, role titles, activities, metrics, acronyms. MUST have at least 50 terms.
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

        return entity_map, domain_profile, adapted_klos

    except Exception as e:
        logger.error(f"[MAPS] Generation failed: {e}")
        # Return empty - adaptation will still work but without strong guidance
        return EntityMap({}, {}, {}), DomainProfile("Unknown", "Unknown", {}, [], []), []


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
    domain_profile: DomainProfile = None
) -> str:
    """
    Build adaptation prompt using scenario as SINGLE SOURCE OF TRUTH.

    NEW (Option B): If entity_map and domain_profile are provided (from upfront LLM call),
    use those for consistent entity/terminology mapping across ALL shards.
    derived_klos can be adapted KLOs from the upfront call.

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
        entity_section = f"""
## ENTITY MAPPING (USE THESE EXACT REPLACEMENTS):

**Company:**
- Source: "{entity_map.company.get('source', 'Unknown')}" → Target: "{entity_map.company.get('target', 'Target Company')}"
- Email domain: @{entity_map.company.get('domain', 'company.com')}

**People (replace ALL occurrences):**
"""
        for source_name, target_info in entity_map.people.items():
            if isinstance(target_info, dict):
                entity_section += f"- \"{source_name}\" → \"{target_info.get('name', 'Unknown')}\" ({target_info.get('role', 'Employee')}, {target_info.get('email', '')})\n"
            else:
                entity_section += f"- \"{source_name}\" → \"{target_info}\"\n"

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

        # Forbidden terms
        if domain_profile.forbidden_terms:
            forbidden_list = ", ".join(f'"{t}"' for t in domain_profile.forbidden_terms[:30])
            terminology_section += f"""
**FORBIDDEN TERMS (ZERO TOLERANCE - must NOT appear in output):**
{forbidden_list}

If ANY of these terms appear in your output, you have FAILED.
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

    return f"""You are adapting a business simulation to a completely different domain.

## TARGET SCENARIO (YOUR SOURCE OF TRUTH):
{scenario_prompt}
{company_section}{derive_section}{klo_section}{forbidden_section}

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

### Content Quality
- Resources provide DATA (statistics, facts) NOT answers or conclusions
  * NEVER include: "should", "recommend", "therefore", "thus", "conclusion", "suggests that"
  * NEVER include: "The best approach is...", "Based on this data, you should..."
  * ONLY include: Raw data, statistics, facts, market figures, percentages
  * Let learners draw their own conclusions from the data
- Questions ask for ANALYSIS (justify, develop, explain)
- Resources: MUST be 500-1500 words of substantive DATA (statistics, market data, figures) - TOO SHORT WILL BE REJECTED

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
No explanations. Just valid JSON.

**FINAL CHECK:**
- Keys in output MUST match keys in input exactly
- If input is {{"name": "...", "body": "..."}} then output MUST be {{"name": "...", "body": "..."}}
- If input is an array [...] then output MUST be an array [...]
- Do NOT wrap output in a path-like key"""


def _find_json_overlap(end_text: str, start_text: str, max_check: int = 500) -> int:
    """
    Find overlap between end of accumulated text and start of continuation.

    When Gemini continues output, it sometimes repeats context from where it left off.
    This function detects that overlap so we can trim it before concatenation.

    Args:
        end_text: The end of the accumulated text
        start_text: The start of the continuation text
        max_check: Maximum characters to check for overlap

    Returns:
        Number of overlapping characters (0 if no overlap)
    """
    end_sample = end_text[-max_check:] if len(end_text) > max_check else end_text
    start_sample = start_text[:max_check] if len(start_text) > max_check else start_text

    # Look for overlap - start with longest possible and work down
    for i in range(min(len(end_sample), len(start_sample)), 10, -1):
        if end_sample[-i:] == start_sample[:i]:
            return i

    return 0


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

    # Add timeout to prevent indefinite hanging (120 seconds)
    try:
        response = await asyncio.wait_for(
            client.aio.models.generate_content(
                model=model,
                contents=prompt,
                config={
                    "temperature": 0.0,
                    "max_output_tokens": 65536
                }
            ),
            timeout=120.0
        )
    except asyncio.TimeoutError:
        elapsed = _time.time() - start_time
        logger.error(f"[GEMINI] API call timed out after {elapsed:.1f}s")
        raise ValueError(f"Gemini API call timed out after {elapsed:.1f}s")

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

        # Detect and trim overlap before concatenation
        overlap = _find_json_overlap(accumulated_text, continuation_text)
        if overlap > 10:  # Significant overlap detected
            logger.info(f"[GEMINI] Detected {overlap} char overlap in continuation, trimming")
            continuation_text = continuation_text[overlap:]

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
    use_v2: bool = True,  # NEW: Use skeleton-based generation by default
) -> SimpleAdaptationResult:
    """
    Simple adaptation using PARALLEL SHARDING.

    V2 (default): Skeleton-based GENERATION
    - Extracts skeleton (structure only, no content)
    - Stage 0 generates alignment_map, canonical_numbers, resource_sections
    - Each shard GENERATES content to fill skeleton
    - Post-processes to enforce consistency

    V1 (legacy): Content-based ADAPTATION
    - Shows source content to LLM
    - Asks LLM to "adapt" content
    - Tends to copy source phrasing

    Args:
        input_json: The simulation JSON to adapt
        scenario_prompt: Description of the target scenario
        use_v2: If True, use skeleton-based generation (default). If False, use legacy adaptation.

    Returns:
        SimpleAdaptationResult with adapted JSON
    """
    start_time = time.time()
    input_chars = len(json.dumps(input_json))

    if use_v2:
        logger.info(f"[SIMPLE ADAPTER] Starting V2 SKELETON-BASED GENERATION")
    else:
        logger.info(f"[SIMPLE ADAPTER] Starting V1 PARALLEL shard adaptation (legacy)")

    logger.info(f"[SIMPLE ADAPTER] Input size: {input_chars} chars")
    logger.info(f"[SIMPLE ADAPTER] Scenario: {scenario_prompt[:100]}...")

    if use_v2:
        # NEW: Skeleton-based generation
        result = await _adapt_with_sharding_v2(input_json, scenario_prompt)
        adapted_json = result["adapted_json"]
        shards_count = result["shards_count"]
        errors = result["errors"]
        entity_map = result.get("entity_map", {})
        domain_profile = result.get("domain_profile", {})
        alignment_map = result.get("alignment_map", {})
        canonical_numbers = result.get("canonical_numbers", {})
        mode = "skeleton_v2"
    else:
        # Legacy: Content-based adaptation
        adapted_json, shards_count, errors = await _adapt_with_sharding(
            input_json, scenario_prompt
        )
        entity_map = {}
        domain_profile = {}
        alignment_map = {}
        canonical_numbers = {}
        mode = "parallel_shards"

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
        mode=mode,
        shards_processed=shards_count,
        errors=errors,
        entity_map=entity_map,
        domain_profile=domain_profile,
        alignment_map=alignment_map,
        canonical_numbers=canonical_numbers,
    )


async def _adapt_with_sharding(
    input_json: dict,
    scenario_prompt: str
) -> tuple[dict, int, list]:
    """
    Adapt JSON by sharding into smaller pieces.

    Uses OPTION B: ONE UPFRONT CALL then ALL SHARDS IN PARALLEL
    1. Generate entity_map, domain_profile, AND adapted_klos in ONE LLM call
    2. ALL shards adapt in parallel with full context (no PASS 1/PASS 2)

    This ensures consistent naming AND faster execution.
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
        entity_map = EntityMap({}, {}, {})
        domain_profile = DomainProfile("Unknown", "Unknown", {}, [], [])
        adapted_klos = []

    # ALL SHARDS IN PARALLEL - no PASS 1/PASS 2 needed!
    logger.info(f"[SIMPLE ADAPTER] Adapting ALL {len(unlocked_shards)} shards in PARALLEL...")

    # ALL SHARDS IN PARALLEL with entity_map, domain_profile, and adapted_klos
    if unlocked_shards:
        logger.info(f"[SIMPLE ADAPTER] Processing ALL {len(unlocked_shards)} unlocked shards in PARALLEL...")
        logger.info(f"[SIMPLE ADAPTER] Shard IDs: {[s.id for s in unlocked_shards]}")

        # Process shards - split large ones to avoid LLM output truncation
        # 20K threshold balances fewer API calls vs truncation risk
        MAX_SHARD_SIZE = 20000

        tasks = []
        task_info = []  # Track which shard/index each task corresponds to

        for shard in unlocked_shards:
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
                logger.info(f"[SIMPLE ADAPTER] Shard '{shard.id}' in NEVER_SPLIT_SHARDS - adapting as WHOLE ({shard_size} chars)")
                task = _adapt_single_shard_simple(shard, scenario_prompt, derived_klos=adapted_klos, entity_map=entity_map, domain_profile=domain_profile)
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
                            # More aggressive splitting (30% threshold) to prevent output truncation
                            if array_size > MAX_SHARD_SIZE * 0.3:
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
                task = _adapt_single_shard_simple(shard, scenario_prompt, derived_klos=adapted_klos, entity_map=entity_map, domain_profile=domain_profile)
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

    # Count total shards processed (all unlocked shards)
    total_shards = len(unlocked_shards)
    return adapted_json, total_shards, errors


# =============================================================================
# V2: SKELETON-BASED GENERATION (NEW)
# =============================================================================

async def _adapt_with_sharding_v2(
    input_json: dict,
    scenario_prompt: str
) -> dict:
    """
    V2: Skeleton-based GENERATION (not adaptation).

    Flow:
    1. PROGRAMMATIC: Extract skeleton, word_targets, structure_summary from source
    2. STAGE 0: Generate entity_map, domain_profile, adapted_klos, alignment_map, canonical_numbers, resource_sections
    3. STAGE 1: Parallel shard GENERATION (each shard fills skeleton, doesn't see source content)
    4. POST-PROCESS: Enforce entity_map and canonical_numbers consistency

    Returns:
        dict with keys: adapted_json, shards_count, errors, entity_map, domain_profile, alignment_map, canonical_numbers
    """
    from .sharder import Sharder, merge_shards
    from ..models.shard import LockState
    import time as _time

    logger.info("[V2] ========== SKELETON-BASED GENERATION ==========")

    errors = []

    # ═══════════════════════════════════════════════════════════════════
    # STEP 1: PROGRAMMATIC - Extract from source (instant)
    # ═══════════════════════════════════════════════════════════════════
    logger.info("[V2] STEP 1: Extracting skeleton and word targets...")
    step1_start = _time.time()

    # Extract skeleton (structure with IDs, no content)
    skeleton = extract_skeleton(input_json)
    logger.info(f"[V2]   - Skeleton extracted")

    # Extract structure summary (counts, IDs)
    structure_summary = extract_structure_summary(input_json)
    logger.info(f"[V2]   - Structure: {structure_summary['klo_count']} KLOs, "
                f"{structure_summary['question_count']} questions, "
                f"{structure_summary['rubric_count']} rubrics")

    # Measure word targets from source
    word_targets = measure_word_targets(input_json)
    logger.info(f"[V2]   - Word targets measured")

    step1_time = _time.time() - step1_start
    logger.info(f"[V2] STEP 1 complete in {step1_time:.2f}s")

    # ═══════════════════════════════════════════════════════════════════
    # STEP 2: STAGE 0 - Generate domain content (one LLM call)
    # ═══════════════════════════════════════════════════════════════════
    logger.info("[V2] STEP 2: Stage 0 generation (alignment_map, canonical_numbers, etc.)...")
    step2_start = _time.time()

    try:
        stage0_result = await generate_stage0_content(
            scenario_prompt=scenario_prompt,
            structure_summary=structure_summary,
            call_llm_func=call_gemini_async,
            model=DEFAULT_MODEL
        )

        # Validate Stage 0 output
        validation_errors = validate_stage0_output(stage0_result, structure_summary)
        if validation_errors:
            logger.warning(f"[V2] Stage 0 validation found {len(validation_errors)} issues")
            errors.extend([f"stage0: {e}" for e in validation_errors[:3]])
            # Continue anyway - partial results are better than nothing

        step2_time = _time.time() - step2_start
        logger.info(f"[V2] STEP 2 complete in {step2_time:.2f}s")
        logger.info(f"[V2]   - entity_map: company={stage0_result.entity_map.get('company', {}).get('name', 'N/A')}")
        logger.info(f"[V2]   - alignment_map: {len(stage0_result.alignment_map)} entries")
        logger.info(f"[V2]   - canonical_numbers: {len(stage0_result.canonical_numbers)} metrics")
        logger.info(f"[V2]   - resource_sections: {len(stage0_result.resource_sections)} sections")

    except Exception as e:
        logger.error(f"[V2] Stage 0 failed: {e}")
        errors.append(f"stage0_generation: {str(e)}")
        # Create empty Stage 0 result
        stage0_result = Stage0Result(errors=[str(e)])
        step2_time = _time.time() - step2_start

    # ═══════════════════════════════════════════════════════════════════
    # STEP 3: PARALLEL SHARD GENERATION
    # ═══════════════════════════════════════════════════════════════════
    logger.info("[V2] STEP 3: Parallel shard generation...")
    step3_start = _time.time()

    # Shard the SKELETON (not source JSON)
    sharder = Sharder()
    collection = sharder.shard(skeleton)

    # Separate locked vs unlocked
    locked_shards = [s for s in collection.shards if s.lock_state == LockState.FULLY_LOCKED]
    unlocked_shards = [s for s in collection.shards if s.lock_state != LockState.FULLY_LOCKED]

    logger.info(f"[V2]   - {len(locked_shards)} locked, {len(unlocked_shards)} to generate")

    # Build tasks for parallel generation
    tasks = []
    task_info = []

    for shard in unlocked_shards:
        if shard.id in SKIP_SHARDS:
            logger.info(f"[V2]   - Skipping shard '{shard.id}'")
            continue

        # Get alignment requirements for this shard
        alignment_req = get_alignment_for_shard(
            shard.id,
            stage0_result.alignment_map,
            stage0_result.adapted_klos
        )

        # Shards that contain questions and need KLO alignment
        # These shards contain questions that must assess KLOs, so they need the KLO-Question mapping
        question_containing_shards = ["simulation_flow", "rubrics", "resources", "assessment_criteria"]

        # Build the generation prompt
        prompt = build_shard_prompt(
            shard_name=shard.id,
            skeleton=shard.content,
            word_targets=word_targets,
            entity_map=stage0_result.entity_map,
            domain_profile=stage0_result.domain_profile,
            canonical_numbers=stage0_result.canonical_numbers,
            scenario_prompt=scenario_prompt,
            alignment_requirements=alignment_req if alignment_req else None,
            resource_sections=stage0_result.resource_sections if shard.id == "resources" else None,
            adapted_klos=stage0_result.adapted_klos if shard.id in question_containing_shards else None
        )

        # Create task
        task = _generate_shard_content(shard.id, prompt, shard.content)
        tasks.append(task)
        task_info.append({"shard": shard})

    # Run all tasks in parallel
    if tasks:
        logger.info(f"[V2]   - Running {len(tasks)} generation tasks in parallel...")
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for i, result in enumerate(results):
            info = task_info[i]
            shard = info["shard"]

            if isinstance(result, Exception):
                logger.error(f"[V2]   - Shard '{shard.id}' failed: {result}")
                errors.append(f"{shard.id}: {str(result)}")
            else:
                shard.content = result
                shard.current_hash = ""
                collection.update_shard(shard)
                logger.info(f"[V2]   - Shard '{shard.id}' generated successfully")

    step3_time = _time.time() - step3_start
    logger.info(f"[V2] STEP 3 complete in {step3_time:.2f}s")

    # ═══════════════════════════════════════════════════════════════════
    # STEP 4: MERGE SHARDS
    # ═══════════════════════════════════════════════════════════════════
    logger.info("[V2] STEP 4: Merging shards...")

    # Merge back using ORIGINAL input_json structure (not skeleton)
    # The skeleton was just for sharding - we merge results back to original structure
    adapted_json = merge_shards(collection, input_json)

    # ═══════════════════════════════════════════════════════════════════
    # STEP 5: POST-PROCESSING
    # ═══════════════════════════════════════════════════════════════════
    logger.info("[V2] STEP 5: Post-processing enforcement...")
    step5_start = _time.time()

    adapted_json = post_process_adapted_json(
        adapted_json=adapted_json,
        entity_map=stage0_result.entity_map,
        canonical_numbers=stage0_result.canonical_numbers,
        domain_profile=stage0_result.domain_profile
    )

    step5_time = _time.time() - step5_start
    logger.info(f"[V2] STEP 5 complete in {step5_time:.2f}s")

    # ═══════════════════════════════════════════════════════════════════
    # DONE
    # ═══════════════════════════════════════════════════════════════════
    total_time = step1_time + step2_time + step3_time + step5_time
    logger.info(f"[V2] ========== COMPLETE in {total_time:.2f}s ==========")
    logger.info(f"[V2] Timings: Step1={step1_time:.1f}s, Step2={step2_time:.1f}s, "
                f"Step3={step3_time:.1f}s, Step5={step5_time:.1f}s")

    return {
        "adapted_json": adapted_json,
        "shards_count": len(unlocked_shards),
        "errors": errors,
        "entity_map": stage0_result.entity_map,
        "domain_profile": stage0_result.domain_profile,
        "alignment_map": stage0_result.alignment_map,
        "canonical_numbers": stage0_result.canonical_numbers,
    }


@traceable(name="generate_shard_v2", run_type="llm")
async def _generate_shard_content(
    shard_id: str,
    prompt: str,
    skeleton: dict
) -> dict:
    """
    Generate content for a single shard using the skeleton-based prompt.

    Args:
        shard_id: The shard identifier
        prompt: The complete generation prompt
        skeleton: The skeleton structure (for reference)

    Returns:
        Generated content as dict
    """
    logger.info(f"[GENERATE V2] >>> START: {shard_id}")

    try:
        response_text = await call_gemini_async(prompt, expect_json=True, model=DEFAULT_MODEL)
        generated_content = _repair_json(response_text)

        output_size = len(json.dumps(generated_content))
        logger.info(f"[GENERATE V2] <<< DONE: {shard_id} ({output_size} chars)")

        return generated_content

    except Exception as e:
        logger.error(f"[GENERATE V2] Failed for {shard_id}: {e}")
        # Return the skeleton as fallback (better than nothing)
        return skeleton


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


@traceable(name="adapt_shard", run_type="llm")
async def _adapt_single_shard_simple(
    shard,
    scenario_prompt: str,
    derived_klos: list[dict] = None,
    entity_map: EntityMap = None,
    domain_profile: DomainProfile = None
) -> dict:
    """Adapt a single shard using scenario prompt as source of truth.

    Uses entity_map and domain_profile from upfront generation (Option B).
    derived_klos are already adapted KLOs from the same upfront call.
    """
    input_size = len(json.dumps(shard.content))
    logger.info(f"[ADAPT SHARD] >>> START: {shard.id} ({input_size} chars)")

    # Select model - use stable model for problematic shards
    model = STABLE_MODEL if shard.id in SHARDS_USING_STABLE_MODEL else DEFAULT_MODEL

    # Build prompt with Option B maps
    prompt = build_simple_prompt(
        scenario_prompt,
        shard.content,
        shard_name=shard.name,
        derived_klos=derived_klos,
        entity_map=entity_map,
        domain_profile=domain_profile
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
        