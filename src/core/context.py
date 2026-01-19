"""
Context Extraction for Leaf-Based Adaptation.

Extracts context in ONE LLM call - NO HARDCODED TERM MAPPINGS.

What we extract:
- Company name (generate if not in scenario)
- Old company names (for poison list)
- Industry identification (source vs target)
- KLO terms (for alignment validation)
- Poison terms (what to REMOVE, not what to replace with)

What we DON'T do:
- Create term-to-term mappings (that causes literal replacement!)
- The LLM in smart_prompts handles semantic transformation
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import logging
import json
import re

from langsmith import traceable

logger = logging.getLogger(__name__)


@dataclass
class AdaptationContext:
    """Full context for smart adaptation."""

    # Company/Industry
    new_company_name: str = ""
    old_company_names: List[str] = field(default_factory=list)
    target_industry: str = ""
    source_industry: str = ""

    # Entity mappings (company names only, NOT industry terms)
    entity_map: Dict[str, str] = field(default_factory=dict)  # old company -> new company

    # REMOVED: industry_term_map - This caused literal replacement!
    # The LLM handles semantic transformation, not term-for-term replacement

    # Poison terms (things to REMOVE from source)
    poison_terms: List[str] = field(default_factory=list)

    # KLO data (for question/rubric alignment)
    klo_terms: Dict[str, str] = field(default_factory=dict)  # klo1 -> "key phrase"
    klo_details: List[Dict] = field(default_factory=list)

    # Resource data (for question answerability)
    resource_data: Dict[str, str] = field(default_factory=dict)
    resource_summary: str = ""

    # Questions summary (for resource adaptation)
    questions_summary: str = ""

    # Industry context
    valid_kpis: List[str] = field(default_factory=list)
    invalid_kpis: List[str] = field(default_factory=list)

    # Scenario
    target_scenario: str = ""
    source_scenario: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "company": {
                "new_name": self.new_company_name,
                "old_names": self.old_company_names,
                "target_industry": self.target_industry,
                "source_industry": self.source_industry,
            },
            "entity_map": self.entity_map,
            "poison_terms": self.poison_terms,
            "klo_terms": self.klo_terms,
            "resource_data": self.resource_data,
            "valid_kpis": self.valid_kpis,
            "invalid_kpis": self.invalid_kpis,
        }

    def apply_factsheet(self, factsheet: Dict[str, Any]) -> None:
        """
        Merge factsheet data into context.

        This ensures canonical entity names, invalid KPIs, and additional
        poison terms from the factsheet are applied to the context.

        Args:
            factsheet: Global factsheet with entity mappings and constraints
        """
        if not factsheet:
            return

        # Extract company info
        company = factsheet.get("company", {})
        if isinstance(company, dict):
            # Override company name if specified in factsheet
            new_name = company.get("name") or company.get("new_name")
            if new_name:
                self.new_company_name = new_name

            # Add old company names to poison terms
            old_name = company.get("old_name") or company.get("source_name")
            if old_name and old_name not in self.old_company_names:
                self.old_company_names.append(old_name)
                if old_name not in self.poison_terms:
                    self.poison_terms.append(old_name)

            # Set industry if not already set
            industry = company.get("industry") or company.get("target_industry")
            if industry and not self.target_industry:
                self.target_industry = industry

        # Extract manager mapping
        manager = factsheet.get("manager", {})
        if isinstance(manager, dict):
            old = manager.get("old_name") or manager.get("source_name")
            new = manager.get("name") or manager.get("new_name")
            if old and new:
                self.entity_map[old] = new
                # Old manager name is also poison
                if old not in self.poison_terms:
                    self.poison_terms.append(old)

        # Extract additional entity mappings
        extra_mappings = factsheet.get("entity_mappings", {})
        if isinstance(extra_mappings, dict):
            for old, new in extra_mappings.items():
                self.entity_map[old] = new
                if old not in self.poison_terms:
                    self.poison_terms.append(old)

        # Extract invalid KPIs (metrics that shouldn't appear in target)
        invalid_kpis = factsheet.get("invalid_kpis", [])
        if isinstance(invalid_kpis, list):
            for kpi in invalid_kpis:
                if kpi and kpi not in self.invalid_kpis:
                    self.invalid_kpis.append(kpi)

        # Extract additional poison terms
        extra_poison = factsheet.get("poison_terms", [])
        if isinstance(extra_poison, list):
            for term in extra_poison:
                if term and term not in self.poison_terms:
                    self.poison_terms.append(term)

        # Extract valid KPIs for target industry
        valid_kpis = factsheet.get("valid_kpis", [])
        if isinstance(valid_kpis, list):
            for kpi in valid_kpis:
                if kpi and kpi not in self.valid_kpis:
                    self.valid_kpis.append(kpi)

        logger.info(f"[FACTSHEET] Applied factsheet: "
                   f"company='{self.new_company_name}', "
                   f"{len(self.entity_map)} entity mappings, "
                   f"{len(self.invalid_kpis)} invalid KPIs")


# NEW PROMPT - No term mappings, generates company name
CONTEXT_EXTRACTION_PROMPT = """
You are analyzing a simulation to prepare for context adaptation.

**SOURCE SCENARIO (current simulation):**
{source_scenario}

**TARGET SCENARIO (what we're adapting to):**
{target_scenario}

**KEY LEARNING OUTCOMES (KLOs) from source:**
{klos_text}

═══════════════════════════════════════════════════════════════════════
                         EXTRACT THE FOLLOWING
═══════════════════════════════════════════════════════════════════════

1. **IDENTIFY SOURCE COMPANY:**
   - What company/organization is in the source simulation?
   - List ALL variations of the name (full name, abbreviations, etc.)

2. **GENERATE TARGET COMPANY NAME:**
   - If the target scenario mentions a specific company name, use it
   - If NOT, GENERATE an appropriate fictional company name for the target industry
   - The name should sound professional and fit the industry

3. **IDENTIFY INDUSTRIES:**
   - What industry is the SOURCE simulation about?
   - What industry is the TARGET scenario about?

4. **KLO KEY PHRASES:**
   - Extract the CORE assessment concept from each KLO
   - These are the skills/knowledge students should demonstrate

5. **POISON TERMS (VERY IMPORTANT!):**
   - List PROPER NOUNS - specific company names, product names, person names
   - ALSO list DOMAIN-SPECIFIC TERMS from the source that would break immersion in the target
   - Think: "What terms are central to the SOURCE scenario that have no place in the TARGET?"
   - DO NOT include truly generic terms that apply to any business context
   - DO NOT include educational framework terms used across all simulations

6. **INVALID KPIs:**
   - What metrics/KPIs are specific to the SOURCE industry?
   - These should NOT appear in content adapted for the TARGET industry

═══════════════════════════════════════════════════════════════════════
                    CRITICAL: WHAT IS A POISON TERM?
═══════════════════════════════════════════════════════════════════════

POISON = Terms that would BREAK IMMERSION in the target scenario
  ✅ YES: Specific company/product/person names from the source
  ✅ YES: Domain-specific terminology central to the SOURCE but alien to TARGET
  ❌ NO: Truly generic business terms that work in any context
  ❌ NO: Educational framework terms (validity, reliability, assessment)

Ask yourself: "Would a learner in the TARGET scenario be confused by this term?"

═══════════════════════════════════════════════════════════════════════
                         RESPOND AS JSON
═══════════════════════════════════════════════════════════════════════

{{
  "source_company": {{
    "names": ["Velocity Dome", "VD", "Velocity Dome Entertainment"],
    "industry": "entertainment/events"
  }},
  "target_company": {{
    "name": "EcoThread Co.",
    "industry": "retail/apparel",
    "generated": true
  }},
  "klo_terms": {{
    "klo1": "core skill or knowledge being assessed",
    "klo2": "core skill or knowledge being assessed"
  }},
  "poison_terms": [
    "Velocity Dome", "VD", "Velocity Dome Entertainment"
  ],
  "invalid_kpis": [
    "ticket sales", "event attendance", "venue capacity"
  ]
}}

NOTE: If the target scenario says "Gen Z organic T-shirts brand" without a company name,
GENERATE one like "ThreadGen", "EcoThread Co.", "GreenWear Inc.", etc.
"""


@traceable(name="extract_adaptation_context")
async def extract_adaptation_context(
    input_json: Dict[str, Any],
    target_scenario: str,
    source_scenario: str = "",
) -> AdaptationContext:
    """
    Extract adaptation context in ONE LLM call.

    Key changes from before:
    - GENERATES company name if not specified
    - Does NOT create industry term mappings (prevents literal replacement)
    - Only identifies WHAT TO REMOVE (poison terms)
    """
    from ..utils.gemini_client import call_gemini

    # Extract KLOs from JSON
    klos = _extract_klos(input_json)
    klos_text = _format_klos(klos)

    # Extract questions from JSON
    questions = _extract_questions(input_json)
    questions_text = _format_questions(questions)

    # Extract resource data from JSON
    resources = _extract_resources(input_json)
    resources_text = _format_resources(resources)

    # If no source scenario provided, try to extract from JSON
    if not source_scenario:
        source_scenario = _extract_source_scenario(input_json)

    # Build prompt
    prompt = CONTEXT_EXTRACTION_PROMPT.format(
        source_scenario=source_scenario[:2000] if source_scenario else "Not specified",
        target_scenario=target_scenario[:2000],
        klos_text=klos_text,
        questions_text=questions_text,
        resources_text=resources_text,
    )

    logger.info("Extracting adaptation context...")

    try:
        response = await call_gemini(prompt, temperature=0.3)

        # Parse response
        context = _parse_context_response(response, target_scenario)

        # Add scenario info
        context.target_scenario = target_scenario
        context.source_scenario = source_scenario

        # Store KLO details
        context.klo_details = klos

        # Build resource summary
        context.resource_summary = resources_text
        context.questions_summary = questions_text

        logger.info(f"Context extracted: company='{context.new_company_name}', "
                   f"{len(context.poison_terms)} poison terms, "
                   f"source={context.source_industry}, target={context.target_industry}")

        return context

    except Exception as e:
        logger.error(f"Context extraction failed: {e}")
        return _build_fallback_context(input_json, target_scenario, source_scenario)


def _parse_context_response(response: Dict[str, Any], target_scenario: str) -> AdaptationContext:
    """Parse LLM response into AdaptationContext."""
    context = AdaptationContext()

    # Source company info
    source_company = response.get("source_company", {})
    context.old_company_names = source_company.get("names", [])
    context.source_industry = source_company.get("industry", "")

    # Target company info
    target_company = response.get("target_company", {})
    context.new_company_name = target_company.get("name", "")
    context.target_industry = target_company.get("industry", "")

    # If no company name was extracted/generated, create one
    if not context.new_company_name:
        context.new_company_name = _generate_company_name(target_scenario, context.target_industry)
        logger.info(f"Generated company name: {context.new_company_name}")

    # Build entity map (ONLY company names, not industry terms)
    for old_name in context.old_company_names:
        if old_name:
            context.entity_map[old_name] = context.new_company_name

    # KLO terms
    context.klo_terms = response.get("klo_terms", {})

    # Poison terms
    context.poison_terms = response.get("poison_terms", [])

    # Add old company names to poison terms
    for name in context.old_company_names:
        if name and name not in context.poison_terms:
            context.poison_terms.append(name)

    # Invalid KPIs (treated as poison for the target industry)
    context.invalid_kpis = response.get("invalid_kpis", [])

    return context


def _generate_company_name(target_scenario: str, target_industry: str) -> str:
    """Generate a fictional company name based on scenario and industry."""
    scenario_lower = target_scenario.lower()

    # Try to extract hints from scenario
    if "gen z" in scenario_lower:
        prefix = "Gen"
    elif "organic" in scenario_lower or "eco" in scenario_lower:
        prefix = "Eco"
    elif "sustainable" in scenario_lower:
        prefix = "Green"
    else:
        prefix = ""

    # Industry-specific name generation
    industry_lower = (target_industry or "").lower()

    if "apparel" in industry_lower or "fashion" in industry_lower or "t-shirt" in scenario_lower or "clothing" in scenario_lower:
        options = ["Thread", "Wear", "Fabric", "Style", "Attire"]
    elif "beverage" in industry_lower or "drink" in scenario_lower:
        options = ["Brew", "Sip", "Refresh", "Flow", "Pour"]
    elif "food" in industry_lower:
        options = ["Taste", "Bite", "Flavor", "Fresh", "Harvest"]
    elif "tech" in industry_lower or "software" in scenario_lower:
        options = ["Logic", "Byte", "Code", "Tech", "Digital"]
    elif "retail" in industry_lower:
        options = ["Mart", "Shop", "Store", "Market", "Trade"]
    elif "hospitality" in industry_lower or "hotel" in scenario_lower:
        options = ["Stay", "Rest", "Haven", "Lodge", "Inn"]
    else:
        options = ["Corp", "Co", "Inc", "Group", "Holdings"]

    import random
    suffix = random.choice(options)

    if prefix:
        name = f"{prefix}{suffix}"
    else:
        # Generate a two-part name
        first_parts = ["Nova", "Prime", "Apex", "Vertex", "Peak", "Core", "Pulse", "Spark"]
        name = f"{random.choice(first_parts)} {suffix}"

    # Add company suffix
    company_suffixes = ["Co.", "Inc.", "Corp", ""]
    final_suffix = random.choice(company_suffixes)

    if final_suffix:
        return f"{name} {final_suffix}"
    return name


def _extract_klos(json_data: Dict[str, Any]) -> List[Dict]:
    """Extract KLOs from JSON."""
    klos = []

    topic_data = json_data.get("topicWizardData", {})

    # Try assessmentCriterion
    criteria = topic_data.get("assessmentCriterion", [])
    for item in criteria:
        if isinstance(item, dict):
            klo = item.get("keyLearningOutcome") or item.get("klo") or item.get("outcome")
            if klo:
                klos.append({
                    "id": item.get("id", ""),
                    "text": klo,
                })

    # Try selectedAssessmentCriterion
    selected = topic_data.get("selectedAssessmentCriterion", [])
    for item in selected:
        if isinstance(item, dict):
            klo = item.get("keyLearningOutcome") or item.get("klo")
            if klo and not any(k["text"] == klo for k in klos):
                klos.append({
                    "id": item.get("id", ""),
                    "text": klo,
                })

    return klos


def _extract_questions(json_data: Dict[str, Any]) -> List[Dict]:
    """Extract submission questions from JSON."""
    questions = []

    topic_data = json_data.get("topicWizardData", {})
    sim_flow = topic_data.get("simulationFlow", [])

    for stage in sim_flow:
        if isinstance(stage, dict):
            data = stage.get("data", {})

            # Check for submission questions
            submission = data.get("submission", {})
            if isinstance(submission, dict):
                q_list = submission.get("questions", [])
                for q in q_list:
                    if isinstance(q, dict):
                        text = q.get("text") or q.get("question")
                        if text:
                            questions.append({
                                "id": q.get("id", ""),
                                "text": text,
                            })

    return questions


def _extract_resources(json_data: Dict[str, Any]) -> List[Dict]:
    """Extract resource data from JSON."""
    resources = []

    topic_data = json_data.get("topicWizardData", {})
    sim_flow = topic_data.get("simulationFlow", [])

    for stage in sim_flow:
        if isinstance(stage, dict):
            data = stage.get("data", {})

            # Check for resources
            resource = data.get("resource", {})
            if isinstance(resource, dict):
                content = resource.get("content") or resource.get("text") or resource.get("body")
                if content:
                    resources.append({
                        "id": resource.get("id", ""),
                        "title": resource.get("title", ""),
                        "content": content[:500] if isinstance(content, str) else str(content)[:500],
                    })

            # Check resourceOptions
            options = data.get("resourceOptions", [])
            for opt in options if isinstance(options, list) else []:
                if isinstance(opt, dict):
                    content = opt.get("content") or opt.get("text")
                    if content:
                        resources.append({
                            "id": opt.get("id", ""),
                            "title": opt.get("title", ""),
                            "content": content[:500] if isinstance(content, str) else str(content)[:500],
                        })

    return resources


def _extract_source_scenario(json_data: Dict[str, Any]) -> str:
    """Extract source scenario from JSON."""
    topic_data = json_data.get("topicWizardData", {})

    # Try workplace scenario
    workplace = topic_data.get("workplaceScenario", {})
    if isinstance(workplace, dict):
        scenario = workplace.get("scenario") or workplace.get("description")
        if scenario:
            return scenario

    # Try lesson information
    lesson = topic_data.get("lessonInformation", {})
    if isinstance(lesson, dict):
        lesson_text = lesson.get("lesson") or lesson.get("description")
        if lesson_text:
            return lesson_text

    return ""


def _format_klos(klos: List[Dict]) -> str:
    """Format KLOs for prompt."""
    if not klos:
        return "No KLOs found"

    lines = []
    for i, klo in enumerate(klos, 1):
        lines.append(f"KLO {i}: {klo.get('text', '')}")
    return "\n".join(lines)


def _format_questions(questions: List[Dict]) -> str:
    """Format questions for prompt."""
    if not questions:
        return "No questions found"

    lines = []
    for i, q in enumerate(questions, 1):
        lines.append(f"Q{i}: {q.get('text', '')[:200]}")
    return "\n".join(lines)


def _format_resources(resources: List[Dict]) -> str:
    """Format resources for prompt."""
    if not resources:
        return "No resources found"

    lines = []
    for r in resources[:5]:
        title = r.get("title", "Untitled")
        content = r.get("content", "")[:200]
        lines.append(f"- {title}: {content}...")
    return "\n".join(lines)


def _build_fallback_context(
    json_data: Dict[str, Any],
    target_scenario: str,
    source_scenario: str,
) -> AdaptationContext:
    """Build minimal context when LLM extraction fails."""
    context = AdaptationContext()
    context.target_scenario = target_scenario
    context.source_scenario = source_scenario

    # Generate company name
    context.new_company_name = _generate_company_name(target_scenario, "")

    # Extract KLOs
    klos = _extract_klos(json_data)
    context.klo_details = klos
    for i, klo in enumerate(klos, 1):
        context.klo_terms[f"klo{i}"] = klo.get("text", "")[:100]

    # Basic poison terms from source scenario
    if source_scenario:
        words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', source_scenario)
        context.poison_terms = list(set(words))[:15]

    return context
