"""
Stage 0 Generator - Generate all domain-specific content BEFORE shard adaptation.

This is the SINGLE upfront LLM call that generates:
1. entity_map - Company, people, roles, terminology
2. domain_profile - Industry, scenario type, challenge
3. adapted_klos - KLOs rewritten for target scenario
4. alignment_map - Links KLO ↔ Question ↔ Resource ↔ Rubric
5. canonical_numbers - Exact figures for consistency across shards
6. resource_sections - Scenario-specific sections for the resource

All shards then use these outputs for CONSISTENT generation.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Stage0Result:
    """Result from Stage 0 generation."""
    entity_map: dict = field(default_factory=dict)
    domain_profile: dict = field(default_factory=dict)
    adapted_klos: list = field(default_factory=list)
    alignment_map: dict = field(default_factory=dict)
    canonical_numbers: dict = field(default_factory=dict)
    resource_sections: list = field(default_factory=list)
    errors: list = field(default_factory=list)


def build_stage0_prompt(scenario_prompt: str, structure_summary: dict) -> str:
    """
    Build the Stage 0 prompt.

    Args:
        scenario_prompt: Target scenario description
        structure_summary: Counts and IDs from source (no content)

    Returns:
        Complete prompt for Stage 0 generation
    """
    klo_count = structure_summary.get("klo_count", 3)
    klo_ids = structure_summary.get("klo_ids", [])
    criteria_per_klo = structure_summary.get("criteria_per_klo", [3, 3, 3])
    question_count = structure_summary.get("question_count", 5)
    question_ids = structure_summary.get("question_ids", [])
    rubric_count = structure_summary.get("rubric_count", 4)

    # Format KLO IDs for the prompt
    klo_ids_str = json.dumps(klo_ids) if klo_ids else '["klo_1", "klo_2", "klo_3"]'
    question_ids_str = json.dumps(question_ids) if question_ids else '["q_1", "q_2", "q_3", "q_4", "q_5"]'

    prompt = f"""## YOUR TASK
Generate all domain-specific content for a business simulation about the following scenario.

## TARGET SCENARIO
{scenario_prompt}

## STRUCTURE TO FILL (generate content for ALL of these)
- KLO count: {klo_count} (IDs: {klo_ids_str})
- Criteria per KLO: {criteria_per_klo}
- Question count: {question_count} (IDs: {question_ids_str})
- Rubric count: {rubric_count}

## GENERATE THE FOLLOWING (all in one JSON response):

### 1. entity_map
Map of entities specific to this scenario:
```json
{{
  "company": {{
    "name": "Company name from scenario",
    "domain": "companydomain.com"
  }},
  "people": [
    {{
      "name": "Full Name",
      "role": "Job Title appropriate for this industry",
      "email": "firstname.lastname@companydomain.com",
      "gender": "male or female"
    }},
    {{
      "name": "Another Person",
      "role": "Another Role",
      "email": "another.person@companydomain.com",
      "gender": "female or male"
    }}
  ],
  "competitor": {{
    "name": "Competitor company name"
  }},
  "products": ["Product 1", "Product 2", "Product 3"],
  "terminology": {{
    "key_term_1": "industry-specific term",
    "key_term_2": "another domain term"
  }}
}}
```

### 2. domain_profile
```json
{{
  "industry": "e.g., pharmaceutical, QSR, gaming, retail",
  "scenario_type": "e.g., competitive response, market entry, pricing strategy",
  "key_challenge": "One sentence describing the core business decision",
  "stakeholders": ["Who cares about this decision"],
  "success_metrics": ["How success is measured in this domain"]
}}
```

### 3. adapted_klos
Generate exactly {klo_count} KLOs. Each must:
- Be SPECIFIC to THIS scenario (not generic business skills)
- Follow format: "Learners will be able to [verb] [object] by [method]"
- Have exactly the number of criteria specified

```json
[
  {{
    "id": "use_first_id_from_list",
    "keyLearningOutcome": "Learners will be able to [specific skill for this scenario]",
    "criterion": [
      {{"id": "c1", "criteria": "First criterion"}},
      {{"id": "c2", "criteria": "Second criterion"}},
      {{"id": "c3", "criteria": "Third criterion"}}
    ]
  }}
]
```

### 4. resource_sections
Generate sections appropriate for THIS scenario's resource document.
Each section provides DATA (not recommendations) that learners will analyze.

```json
[
  {{
    "title": "Section title relevant to the scenario challenge",
    "purpose": "What data this section provides",
    "must_include": ["specific data point 1", "specific data point 2", "specific data point 3"]
  }},
  {{
    "title": "Another relevant section",
    "purpose": "What this section provides",
    "must_include": ["data point", "data point", "data point"]
  }}
]
```
Generate 4-6 sections that cover the key aspects of the scenario.

### 5. alignment_map
For EACH question (use the actual question IDs), specify:
- Which KLO it assesses
- The exact question to generate
- What data the resource must contain to answer it
- What the rubric should check

```json
{{
  "question_id_1": {{
    "assesses_klo": "Summary of which KLO this assesses",
    "question_must_ask": "The specific question text to generate",
    "resource_must_contain": ["Data point 1 needed to answer", "Data point 2 needed"],
    "rubric_must_check": "What a 5-star answer demonstrates"
  }},
  "question_id_2": {{
    "assesses_klo": "...",
    "question_must_ask": "...",
    "resource_must_contain": ["...", "..."],
    "rubric_must_check": "..."
  }}
}}
```
IMPORTANT: Generate an entry for EACH of the {question_count} questions using the provided IDs.

### 6. canonical_numbers
Exact figures to use EVERYWHERE in the simulation. All shards will use these exact values.

```json
{{
  "market_size": "$X.XB",
  "company_revenue": "$XXM",
  "market_share": "XX%",
  "growth_rate": "X.X%",
  "key_metric_1_name": "value with units",
  "key_metric_2_name": "value with units",
  "competitor_metric": "value",
  "time_period": "specific timeframe"
}}
```
Generate 8-12 realistic metrics specific to this industry and scenario.

## CRITICAL REQUIREMENTS
1. ALL content must be specific to: {scenario_prompt}
2. Do NOT use generic business terms - use industry-specific terminology
3. Numbers must be realistic for this industry
4. alignment_map must have an entry for EVERY question ID
5. resource_sections must provide data to answer ALL questions
6. KLOs must be assessable through the questions

## OUTPUT FORMAT
Return a single valid JSON object with all 6 sections:
```json
{{
  "entity_map": {{ ... }},
  "domain_profile": {{ ... }},
  "adapted_klos": [ ... ],
  "resource_sections": [ ... ],
  "alignment_map": {{ ... }},
  "canonical_numbers": {{ ... }}
}}
```

Return ONLY the JSON, no explanations."""

    return prompt


async def generate_stage0_content(
    scenario_prompt: str,
    structure_summary: dict,
    call_llm_func,
    model: str = "gemini-2.5-flash"
) -> Stage0Result:
    """
    Generate all Stage 0 content in one LLM call.

    Args:
        scenario_prompt: Target scenario description
        structure_summary: Counts and IDs from source
        call_llm_func: Async function to call the LLM
        model: Model to use

    Returns:
        Stage0Result with all generated content
    """
    logger.info("[STAGE 0] Generating entity_map, domain_profile, KLOs, alignment_map, canonical_numbers, resource_sections...")

    prompt = build_stage0_prompt(scenario_prompt, structure_summary)

    try:
        response_text = await call_llm_func(prompt, expect_json=True, model=model)

        # Parse response
        result_json = _parse_stage0_response(response_text)

        # Build result object
        result = Stage0Result(
            entity_map=result_json.get("entity_map", {}),
            domain_profile=result_json.get("domain_profile", {}),
            adapted_klos=result_json.get("adapted_klos", []),
            alignment_map=result_json.get("alignment_map", {}),
            canonical_numbers=result_json.get("canonical_numbers", {}),
            resource_sections=result_json.get("resource_sections", []),
        )

        # Log summary
        logger.info(f"[STAGE 0] Generated:")
        logger.info(f"  - entity_map: company={result.entity_map.get('company', {}).get('name', 'N/A')}, "
                   f"{len(result.entity_map.get('people', []))} people")
        logger.info(f"  - domain_profile: {result.domain_profile.get('industry', 'N/A')} / "
                   f"{result.domain_profile.get('scenario_type', 'N/A')}")
        logger.info(f"  - adapted_klos: {len(result.adapted_klos)} KLOs")
        logger.info(f"  - alignment_map: {len(result.alignment_map)} question mappings")
        logger.info(f"  - canonical_numbers: {len(result.canonical_numbers)} metrics")
        logger.info(f"  - resource_sections: {len(result.resource_sections)} sections")

        return result

    except Exception as e:
        logger.error(f"[STAGE 0] Generation failed: {e}")
        return Stage0Result(errors=[str(e)])


def _parse_stage0_response(response_text: str) -> dict:
    """Parse Stage 0 LLM response into dict."""
    import json

    text = response_text.strip()

    # Remove markdown code blocks
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]  # Remove first line (```json)
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try json_repair
    try:
        from json_repair import repair_json
        repaired = repair_json(text, return_objects=True)
        if isinstance(repaired, dict):
            return repaired
    except Exception:
        pass

    # Try to find JSON object
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
    except Exception:
        pass

    raise ValueError(f"Could not parse Stage 0 response: {text[:500]}...")


def validate_stage0_output(result: Stage0Result, structure_summary: dict) -> list[str]:
    """
    Validate Stage 0 output BEFORE running shard generation.

    Args:
        result: The Stage 0 result to validate
        structure_summary: Expected counts from source

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    # 1. Check entity_map
    if not result.entity_map:
        errors.append("entity_map is empty")
    elif not result.entity_map.get("company", {}).get("name"):
        errors.append("entity_map.company.name is missing")
    elif not result.entity_map.get("company", {}).get("domain"):
        errors.append("entity_map.company.domain is missing")

    # 2. Check domain_profile
    if not result.domain_profile:
        errors.append("domain_profile is empty")
    elif not result.domain_profile.get("industry"):
        errors.append("domain_profile.industry is missing")

    # 3. Check KLO count
    expected_klos = structure_summary.get("klo_count", 3)
    if len(result.adapted_klos) != expected_klos:
        errors.append(f"KLO count mismatch: expected {expected_klos}, got {len(result.adapted_klos)}")

    # 4. Check alignment_map covers all questions
    expected_questions = set(structure_summary.get("question_ids", []))
    aligned_questions = set(result.alignment_map.keys())

    missing = expected_questions - aligned_questions
    if missing:
        errors.append(f"alignment_map missing questions: {list(missing)[:5]}...")

    # 5. Check each alignment entry has required fields
    for qid, data in result.alignment_map.items():
        if not isinstance(data, dict):
            errors.append(f"alignment_map[{qid}] is not a dict")
            continue
        if "question_must_ask" not in data:
            errors.append(f"alignment_map[{qid}] missing question_must_ask")
        if "resource_must_contain" not in data:
            errors.append(f"alignment_map[{qid}] missing resource_must_contain")
        if "assesses_klo" not in data:
            errors.append(f"alignment_map[{qid}] missing assesses_klo")

    # 6. Check canonical_numbers
    if not result.canonical_numbers:
        errors.append("canonical_numbers is empty")
    elif len(result.canonical_numbers) < 5:
        errors.append(f"canonical_numbers has only {len(result.canonical_numbers)} entries (need at least 5)")

    # 7. Check resource_sections
    if not result.resource_sections:
        errors.append("resource_sections is empty")
    elif len(result.resource_sections) < 3:
        errors.append(f"resource_sections has only {len(result.resource_sections)} sections (need at least 3)")

    if errors:
        logger.warning(f"[STAGE 0] Validation found {len(errors)} issues:")
        for err in errors[:5]:
            logger.warning(f"  - {err}")

    return errors


async def repair_stage0(
    result: Stage0Result,
    errors: list[str],
    scenario_prompt: str,
    structure_summary: dict,
    call_llm_func,
    model: str = "gemini-2.5-flash"
) -> Stage0Result:
    """
    Repair Stage 0 output by regenerating missing/invalid parts.

    Args:
        result: The original Stage 0 result
        errors: List of validation errors
        scenario_prompt: Target scenario description
        structure_summary: Expected counts from source
        call_llm_func: Async function to call the LLM
        model: Model to use

    Returns:
        Repaired Stage0Result
    """
    logger.info(f"[STAGE 0] Repairing {len(errors)} issues...")

    # For now, regenerate everything if there are errors
    # Future optimization: only regenerate the broken parts

    repaired = await generate_stage0_content(
        scenario_prompt,
        structure_summary,
        call_llm_func,
        model
    )

    # Validate again
    new_errors = validate_stage0_output(repaired, structure_summary)

    if new_errors:
        logger.warning(f"[STAGE 0] Repair still has {len(new_errors)} issues")
        repaired.errors = new_errors
    else:
        logger.info("[STAGE 0] Repair successful")

    return repaired


# =============================================================================
# Helper: Get alignment requirements for a specific shard
# =============================================================================

def get_alignment_for_shard(shard_name: str, alignment_map: dict, adapted_klos: list = None) -> dict:
    """
    Extract alignment requirements relevant to a specific shard.

    Args:
        shard_name: The shard identifier
        alignment_map: The full alignment map from Stage 0
        adapted_klos: The adapted KLOs (for questions/rubrics shards)

    Returns:
        Dict of alignment requirements for this shard
    """
    if shard_name in ("simulation_flow", "questions"):
        # Questions need to know what to ask and which KLO they assess
        return {
            qid: {
                "must_ask": data.get("question_must_ask", ""),
                "assesses_klo": data.get("assesses_klo", "")
            }
            for qid, data in alignment_map.items()
        }

    elif shard_name == "resources":
        # Resources need the FULL KLO → Question → Data chain
        # This ensures resources contain data that supports both KLOs and questions

        klo_question_data_chain = []

        # Group questions by which KLO they assess
        klo_groups = {}
        for qid, data in alignment_map.items():
            klo_assessed = data.get("assesses_klo", "N/A")
            question = data.get("question_must_ask", "N/A")
            required_data = data.get("resource_must_contain", [])

            # Create a KLO key (first 50 chars for grouping)
            klo_key = klo_assessed[:50] if klo_assessed else "Unknown"

            if klo_key not in klo_groups:
                klo_groups[klo_key] = {
                    "klo": klo_assessed,
                    "questions": [],
                    "required_data": []
                }

            klo_groups[klo_key]["questions"].append({
                "id": qid,
                "text": question[:150] if question else ""
            })

            if isinstance(required_data, list):
                klo_groups[klo_key]["required_data"].extend(required_data)

        # Format for prompt
        for klo_key, group in klo_groups.items():
            klo_question_data_chain.append({
                "klo_being_assessed": group["klo"],
                "questions_for_this_klo": group["questions"][:3],  # Limit to 3 questions per KLO
                "data_resource_must_contain": list(set(group["required_data"]))
            })

        # Also collect all required data as flat list
        all_required_data = []
        for qid, data in alignment_map.items():
            required = data.get("resource_must_contain", [])
            if isinstance(required, list):
                all_required_data.extend(required)

        return {
            "klo_question_resource_chain": klo_question_data_chain,
            "all_required_data": list(set(all_required_data)),
            "instruction": "Resources must contain ALL data points listed below. Each KLO is assessed by specific questions, and the resource MUST provide data to answer those questions."
        }

    elif shard_name == "rubrics":
        # Rubrics need to know what to evaluate for each question
        return {
            qid: {
                "rubric_must_check": data.get("rubric_must_check", ""),
                "assesses_klo": data.get("assesses_klo", "")
            }
            for qid, data in alignment_map.items()
        }

    # Other shards don't have specific alignment requirements
    return {}
