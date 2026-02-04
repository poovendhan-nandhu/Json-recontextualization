"""
Shard Prompt Builder - Build 4-part prompts for shard generation.

Each shard prompt has 4 sections:
1. CANONICAL NUMBERS - Exact figures to use everywhere
2. STRUCTURE TO FILL - Skeleton with IDs, values = __GENERATE__
3. WORD TARGETS - Derived from source ±20%
4. ALIGNMENT REQUIREMENTS - Cross-shard consistency rules

The LLM GENERATES content to fill the skeleton, rather than ADAPTING source content.
"""

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# =============================================================================
# CONTENT RULES - Universal quality rules (only these are hardcoded)
# =============================================================================

CONTENT_RULES = {
    "resource": {
        "purpose": "Provide DATA for learner analysis, NOT conclusions",
        "must_NOT_contain": [
            "should", "recommend", "therefore", "in conclusion",
            "clearly", "obviously", "the answer is", "best option",
            "suggests that", "based on this analysis", "it is clear"
        ],
        "must_contain": [
            "specific numbers/statistics",
            "data tables or structured data",
            "multiple perspectives/options"
        ],
        "format": "Markdown with H2 sections"
    },

    "questions": {
        "must_be_answerable_from_resource": True,
        "must_align_with_klos": True,
        "must_end_with_question_mark": True,
        "must_NOT_contain": ["what do you think", "in your opinion", "do you believe"]
    },

    "rubric": {
        "star_order": "5-star = best performance, 1-star = worst",
        "must_be_scenario_specific": True,
        "must_reference_resource_data": True,
        "5_star_must_include": "specific data points from resource",
        "1_star_describes": "generic/missing/no evidence"
    },

    "emails": {
        "must_use_entity_map_names": True,
        "domain_must_match_company": True,
        "task_email_tone": "professional, clear expectations",
        "secondary_email_tone": "business casual, peer support"
    },

    "klos": {
        "format": "Learners will be able to [verb] [object] by [method]",
        "must_be_assessable": True,
        "must_be_scenario_specific": True
    },

    "workplace_scenario": {
        "about_organization_must_NOT_contain": [
            "should", "will need to", "must", "recommendation"
        ],
        "challenge_must_NOT_contain": [
            "solution", "answer", "should", "recommendation"
        ]
    }
}


# =============================================================================
# SHARD-SPECIFIC INSTRUCTIONS
# =============================================================================

SHARD_SPECIFIC_INSTRUCTIONS = {
    "lesson_information": """
## ADDITIONAL INSTRUCTIONS
- lesson: Describe what the learner will do in this simulation
- Include the company name and deliverable type
- Keep it concise but informative
- level and duration are constants - keep them unchanged
""",

    "assessment_criteria": """
## ADDITIONAL INSTRUCTIONS
- Each KLO must be specific to THIS scenario
- Format: "Learners will be able to [verb] [object] by [method]"
- Criteria should be measurable and observable
- All criteria must be assessable through the simulation activities
""",

    "workplace_scenario": """
## ADDITIONAL INSTRUCTIONS
- about_organization: Describe the company factually (NO recommendations)
- industry_context: Market situation and trends (data only)
- current_issue: Present the challenge as a decision to be made (NO solutions)
- role_description: Explain what the learner will do
- manager message: Professional background, sets expectations
- Use ONLY names from entity_map
- Use numbers from canonical_numbers
""",

    "selected_scenario": """
## ADDITIONAL INSTRUCTIONS
- This is the specific scenario option selected
- Should match the scenario_prompt context
- Describe the learner's specific role and task
""",

    "simulation_flow": """
## ADDITIONAL INSTRUCTIONS
- Each question must be answerable using ONLY the resource data
- Questions must directly assess the aligned KLO (see alignment_requirements)
- Use specific terminology from entity_map
- Do NOT ask for opinions - ask for ANALYSIS
- Do NOT ask "what do you think" - ask "what does the data show"
- Emails should use names from entity_map with proper email format
""",

    "resources": """
## ADDITIONAL INSTRUCTIONS
- Generate each section with FACTUAL DATA only
- Include specific numbers, tables, statistics
- NEVER recommend or conclude - let learner analyze
- All numbers MUST match canonical_numbers exactly

**CRITICAL: RESOURCE-KLO-QUESTION ALIGNMENT**
The alignment_requirements show the FULL CHAIN:
1. KLOs = what students should LEARN
2. Questions = what students must ANSWER to demonstrate learning
3. Data = what YOUR RESOURCE must CONTAIN to answer those questions

For EACH KLO in the chain:
- Identify what questions assess that KLO
- Ensure your resource contains SPECIFIC DATA to answer those questions
- Use canonical_numbers for all metrics

Resources provide "dots to connect" - NOT "connected dots".
The learner must analyze your data to reach conclusions.

- Use markdown formatting with headers, tables, bullet points
""",

    "rubrics": """
## ADDITIONAL INSTRUCTIONS
- 5-star: Cites specific data points from resource, fully addresses KLO
- 4-star: Mostly specific, minor gaps in analysis
- 3-star: Mix of specific and generic statements
- 2-star: Mostly generic, few specifics from resource
- 1-star: No evidence from resource, completely generic
- Reference actual data from canonical_numbers in 5-star descriptions
- Each rubric must assess the aligned KLO (see alignment_requirements)
""",

    "emails": """
## ADDITIONAL INSTRUCTIONS
- task_email: From the reporting manager to the learner
  - Professional tone, clear expectations
  - Reference the specific deliverable
  - Use manager name and email from entity_map
- secondary_task_email: From a peer colleague offering tips
  - Business casual, supportive tone
  - Practical structuring advice
  - Personal anecdote about similar work
  - Use a different person from entity_map
""",

    "chat_history": """
## ADDITIONAL INSTRUCTIONS
- Transform all conversation content to match the target domain
- Use names from entity_map
- Use terminology appropriate for the industry
- Keep the conversational structure but change all domain content
""",

    "launch_settings": """
## ADDITIONAL INSTRUCTIONS
- simulation_name: Descriptive title for the simulation
- overview: What learners will do, the company, and the challenge
- Keep it engaging but professional
"""
}


def _build_klo_question_section(
    adapted_klos: list,
    shard_name: str,
    alignment_requirements: Optional[dict] = None
) -> str:
    """
    Build a detailed KLO-Question alignment section for question-containing shards.

    This ensures questions directly assess KLOs using correct terminology.

    Args:
        adapted_klos: List of adapted KLOs with outcomes
        shard_name: The shard being generated
        alignment_requirements: Cross-shard alignment (may contain question mappings)

    Returns:
        Formatted prompt section
    """
    if not adapted_klos:
        return ""

    # Format KLOs with indices for reference
    klos_formatted = []
    for i, klo in enumerate(adapted_klos[:5]):  # Limit to 5 KLOs
        outcome = klo.get('outcome', klo.get('keyLearningOutcome', ''))
        if isinstance(outcome, str):
            outcome = outcome[:200]  # Truncate long outcomes
        klos_formatted.append(f"  KLO {i+1}: {outcome}")

    klos_str = "\n".join(klos_formatted)

    section = f"""
---

## CRITICAL: KLO-QUESTION ALIGNMENT

Every question in this content MUST directly assess one of these adapted KLOs:

{klos_str}

"""

    if shard_name == "simulation_flow":
        section += """**FOR EACH QUESTION YOU GENERATE:**
1. Identify which KLO it should assess
2. Use EXACT terminology from the TARGET scenario
3. Ensure the question tests the SKILL described in the KLO
4. The question must be answerable from the resource data
5. Do NOT use generic phrasing - be specific to this scenario

**BAD:** "What factors should be considered?"
**GOOD:** "Based on CardioFlow's 65% market share data, what pricing strategy would best address Apex Generics' 45% price cut?"

"""
    elif shard_name == "rubrics":
        section += """**FOR EACH RUBRIC:**
1. The rubric criteria MUST align with the KLO being assessed
2. 5-star description should cite SPECIFIC data points from canonical numbers
3. Each star level should reflect how well the learner demonstrates the KLO skill
4. Reference actual metrics (e.g., "$7.2B revenue", "45% price reduction")

**5-STAR EXAMPLE:** "Response cites CardioFlow's 65% market share and Apex's 45% price cut, proposing a tiered rebate strategy with projected impact on the 15-20% revenue erosion..."
**1-STAR EXAMPLE:** "Response is generic, lacks data from the scenario, and doesn't address the specific pricing challenge..."

"""
    elif shard_name == "resources":
        section += """**CRITICAL: KLO-RESOURCE ALIGNMENT**

Your resources MUST contain DATA that allows learners to:
1. LEARN the skills described in each KLO above
2. ANSWER questions that assess those KLOs

For EACH KLO:
- Identify what DATA/FACTS a learner needs to demonstrate that skill
- Include that specific data in your resource sections
- Use canonical_numbers for all metrics

**Resources provide "dots to connect" - NOT "connected dots":**
- Include raw data, statistics, market figures, comparisons
- NEVER include recommendations, conclusions, or "the answer"
- Let learners analyze your data to reach their own conclusions

**EXAMPLE:** If KLO is "analyze competitive pricing strategies":
- Include competitor pricing data, market share trends, price elasticity figures
- Do NOT include "therefore you should lower prices by 20%"

"""

    if alignment_requirements:
        section += """**ALIGNMENT REQUIREMENTS (from Stage 0):**
"""
        # Handle different alignment structures for different shards
        if "klo_question_resource_chain" in alignment_requirements:
            # Resources shard - has KLO → Question → Data chain
            section += "The resource must contain data to support these KLO-Question chains:\n\n"
            chain = alignment_requirements.get("klo_question_resource_chain", [])
            for item in chain[:3]:
                if isinstance(item, dict):
                    klo = item.get("klo_being_assessed", "N/A")[:100]
                    data_needed = item.get("data_resource_must_contain", [])
                    section += f"- KLO: {klo}...\n"
                    if data_needed:
                        section += f"  → Data needed: {', '.join(data_needed[:3])}...\n\n"
        else:
            # Questions/Rubrics shard - has QID → {must_ask, assesses_klo}
            section += "These specific questions must assess these specific KLOs:\n\n"
            for qid, reqs in list(alignment_requirements.items())[:3]:
                if isinstance(reqs, dict):
                    assesses = reqs.get('assesses_klo', reqs.get('assesses', 'N/A'))
                    if assesses:
                        assesses = str(assesses)[:100]
                    question = reqs.get('question_must_ask', reqs.get('must_ask', 'N/A'))
                    if question:
                        question = str(question)[:150]
                    section += f"- Question {qid}: {question}...\n  → Assesses: {assesses}...\n\n"

    return section


def build_shard_prompt(
    shard_name: str,
    skeleton: dict,
    word_targets: dict,
    entity_map: dict,
    domain_profile: dict,
    canonical_numbers: dict,
    scenario_prompt: str,
    alignment_requirements: Optional[dict] = None,
    resource_sections: Optional[list] = None,
    adapted_klos: Optional[list] = None
) -> str:
    """
    Build a 4-part prompt for shard generation.

    Args:
        shard_name: The shard identifier
        skeleton: Structure with IDs, values = __GENERATE__
        word_targets: Dict of field_name -> [min, max] word counts
        entity_map: Company, people, roles, terminology
        domain_profile: Industry, scenario type, challenge
        canonical_numbers: Exact figures for consistency
        scenario_prompt: The target scenario description
        alignment_requirements: Cross-shard alignment rules (optional)
        resource_sections: Sections for resource shard (optional)
        adapted_klos: KLOs for questions/rubrics shards (optional)

    Returns:
        Complete prompt string
    """
    # Get rules for this shard
    rules = CONTENT_RULES.get(shard_name, CONTENT_RULES.get("workplace_scenario", {}))

    # Get word targets for this shard type
    shard_word_targets = _get_shard_word_targets(shard_name, word_targets)

    # Build the 4-part prompt
    prompt = f"""## YOUR TASK
Generate content for the "{shard_name}" section of a business simulation.

## TARGET SCENARIO
{scenario_prompt}

## DOMAIN CONTEXT
Industry: {domain_profile.get('industry', 'business')}
Scenario Type: {domain_profile.get('scenario_type', 'business challenge')}
Key Challenge: {domain_profile.get('key_challenge', 'strategic decision')}

---

## PART 1: CANONICAL NUMBERS (use EXACTLY - no variations)
These are the official figures. Use them EXACTLY as written in your content.

{json.dumps(canonical_numbers, indent=2)}

---

## PART 2: ENTITY MAP (use these names/terms EXACTLY)

**Company:**
- Name: {entity_map.get('company', {}).get('name', 'Company')}
- Email Domain: @{entity_map.get('company', {}).get('domain', 'company.com')}

**People (use these exact names and roles):**
"""

    # Add people
    people = entity_map.get('people', [])
    if isinstance(people, list):
        for person in people:
            if isinstance(person, dict):
                prompt += f"- {person.get('name', 'Person')}: {person.get('role', 'Role')} ({person.get('email', '')})\n"
    elif isinstance(people, dict):
        for name, info in people.items():
            if isinstance(info, dict):
                prompt += f"- {info.get('name', name)}: {info.get('role', 'Role')} ({info.get('email', '')})\n"

    # Add competitor and products
    prompt += f"""
**Competitor:** {entity_map.get('competitor', {}).get('name', 'Competitor')}

**Products:** {', '.join(entity_map.get('products', ['Product']))}

**Industry Terminology:**
"""
    terminology = entity_map.get('terminology', {})
    for term, meaning in list(terminology.items())[:10]:
        prompt += f"- {term}: {meaning}\n"

    prompt += f"""
---

## PART 3: STRUCTURE TO FILL
Replace all "__GENERATE__" placeholders with appropriate content.
Keep all IDs, types, and structural metadata unchanged.

```json
{json.dumps(skeleton, indent=2)}
```

---

## PART 4: WORD TARGETS
Generate content within these word count ranges:

"""
    for field, target in shard_word_targets.items():
        prompt += f"- {field}: {target[0]}-{target[1]} words\n"

    prompt += f"""
---

## CONTENT RULES (must follow)

{json.dumps(rules, indent=2)}

"""

    # Add alignment requirements if provided
    if alignment_requirements:
        prompt += f"""
---

## ALIGNMENT REQUIREMENTS (CRITICAL - your content must satisfy these)

{json.dumps(alignment_requirements, indent=2)}

"""

    # Add resource sections if this is the resource shard
    if resource_sections and shard_name == "resources":
        prompt += f"""
---

## SECTIONS TO GENERATE
Generate these specific sections in your resource content:

{json.dumps(resource_sections, indent=2)}

Each section must:
1. Provide FACTUAL DATA only (no recommendations)
2. Include specific numbers from canonical_numbers
3. Support answering the aligned questions
"""

    # Add KLOs if this shard needs KLO alignment (questions, rubrics, resources)
    # Resources need KLOs to ensure they provide data that supports answering KLO-aligned questions
    if adapted_klos and shard_name in ("simulation_flow", "questions", "rubrics", "resources"):
        prompt += _build_klo_question_section(adapted_klos, shard_name, alignment_requirements)

    # Add shard-specific instructions
    specific_instructions = SHARD_SPECIFIC_INSTRUCTIONS.get(shard_name, "")
    if specific_instructions:
        prompt += f"""
{specific_instructions}
"""

    prompt += """
---

## OUTPUT FORMAT
Return valid JSON matching the structure above.
- Replace all __GENERATE__ placeholders with generated content
- Keep all IDs unchanged
- Keep all structural metadata (type, is_default, etc.) unchanged
- Use entity_map names and canonical_numbers exactly

**CRITICAL: NO PLACEHOLDERS**
Do NOT use any placeholders in your output. Generate COMPLETE, SPECIFIC content:
- Instead of "[Learner Name]" or "[Your Name]" → use a generic greeting like "Hi there," or "Hello,"
- Instead of "[Current Date]" or "[Date]" → use a realistic date like "October 15, 2024"
- Instead of "[Company Name]" → use the actual company name from entity_map
- Instead of "[Insert X]" or "[TBD]" → generate actual content
- Instead of "XX" or "###" → use real numbers from canonical_numbers

Every field must contain COMPLETE, READY-TO-USE content. No brackets, no blanks, no fill-in-the-blank patterns.

**CRITICAL: OUTPUT RULES**
1. Start your response with { (the opening brace of JSON)
2. End your response with } (the closing brace of JSON)
3. Do NOT include any text before or after the JSON
4. Do NOT wrap JSON in markdown code blocks (no ```)
5. Do NOT explain what you're doing
6. ONLY output the raw JSON object

Your response must be parseable by json.loads() directly.
"""

    return prompt


def _get_shard_word_targets(shard_name: str, word_targets: dict) -> dict:
    """Get relevant word targets for a specific shard."""

    shard_field_mapping = {
        "lesson_information": ["lesson"],
        "assessment_criteria": ["klo", "criteria"],
        "workplace_scenario": [
            "scenario", "about_organization", "industry_context",
            "current_issue", "role_description", "manager_message"
        ],
        "selected_scenario": ["scenario"],
        "simulation_flow": [
            "task_email_subject", "task_email_body",
            "secondary_email_subject", "secondary_email_body",
            "question", "guidelines", "guidelines_purpose"
        ],
        "resources": ["resource_markdown", "resource_title"],
        "rubrics": [
            "rubric_question", "star_5_description", "star_4_description",
            "star_3_description", "star_2_description", "star_1_description"
        ],
        "emails": [
            "task_email_subject", "task_email_body",
            "secondary_email_subject", "secondary_email_body"
        ],
        "launch_settings": ["lesson"],  # Similar length
    }

    fields = shard_field_mapping.get(shard_name, [])
    result = {}

    for field in fields:
        if field in word_targets:
            result[field] = word_targets[field]

    # Add default if empty
    if not result:
        result["content"] = [50, 200]

    return result


def build_generation_prompt_simple(
    shard_name: str,
    skeleton: dict,
    scenario_prompt: str,
    entity_map: dict,
    domain_profile: dict,
    canonical_numbers: dict,
    word_targets: dict,
    alignment_requirements: Optional[dict] = None,
) -> str:
    """
    Simplified prompt builder for smaller shards.

    Uses a more compact format when full 4-part structure isn't needed.
    """
    prompt = f"""Generate content for "{shard_name}" section.

SCENARIO: {scenario_prompt}

COMPANY: {entity_map.get('company', {}).get('name', 'Company')}
INDUSTRY: {domain_profile.get('industry', 'business')}

KEY NUMBERS (use exactly):
{json.dumps(canonical_numbers, indent=2)}

STRUCTURE TO FILL:
{json.dumps(skeleton, indent=2)}

Replace __GENERATE__ placeholders with scenario-appropriate content.
Keep all IDs unchanged.
Use the company name and numbers exactly as provided.

Return ONLY valid JSON."""

    return prompt
