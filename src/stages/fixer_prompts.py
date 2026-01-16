"""
Specialized prompt templates for the Semantic Fixer agent.
Each prompt is targeted at a specific type of issue for better fix quality.
"""

# Metrics/KPIs Fix Prompt
METRICS_KPI_PROMPT = """You are a business analyst fixing metrics and KPIs.

CONTEXT:
- Industry: {industry}
- Company: {company_name}

CONTENT TO FIX:
```json
{content}
```

ISSUES:
{issues}

REQUIREMENTS:
1. All percentages must be between 0% and 100%
2. All financial figures must be realistic for the industry
3. KPIs must be measurable and relevant
4. No placeholders like "X-Y" or "TBD"

Return JSON with fixes array. Each fix needs: path, current_value, new_value, reason, fix_type="semantic"
"""

# Personas & Communications Fix Prompt
PERSONAS_COMMS_PROMPT = """You are a communication specialist adapting personas for a new domain.

CONTEXT:
- Industry: {industry}
- Company: {company_name}
- Poison List (MUST REMOVE): {poison_list}
- Replacements: {replacements}

CONTENT TO FIX:
```json
{content}
```

ISSUES:
{issues}

REQUIREMENTS:
1. Remove ALL poison list terms
2. Apply replacement mappings consistently
3. Maintain professional communication style
4. Update industry-specific terminology

Return JSON with fixes array. Each fix needs: path, current_value, new_value, reason, fix_type="semantic"
Also return replacements dict mapping old terms to new.
"""

# Resource Alignment Prompt - KEY for alignment score
RESOURCE_ALIGNMENT_PROMPT = """You are a learning resource specialist creating COMPREHENSIVE learner resources.

CONTEXT:
- Industry: {industry}
- Company: {company_name}
- KLOs: {klos}

CONTENT TO FIX:
```json
{content}
```

ALIGNMENT ISSUES:
{issues}

## ⚠️ CRITICAL REQUIREMENTS - Resources must be LEARNER-READY:

### 1. MINIMUM CONTENT LENGTH
- Each markdownText MUST be 500+ words (at least 2-3 paragraphs)
- NO truncated content or "..." endings
- Complete, actionable information learners can use

### 2. REQUIRED STRUCTURE (use markdown):
```markdown
## Overview
Introduction paragraph with context...

## Key Data
- Market Size: $X.X billion (Source: Statista 2024)
- Growth Rate: X.X% CAGR (Source: IBISWorld 2024)
- Target demographic insights with citations

## Competitor Analysis
- **Brand A** - XX% market share, known for...
- **Brand B** - Key differentiator is...
- **Brand C** - Competes on price/quality...

## Framework Application
### SWOT Analysis for {company_name}
**Strengths:** ...
**Weaknesses:** ...
**Opportunities:** ...
**Threats:** ...

## Actionable Steps
1. Step one with explanation...
2. Step two with formula: ROI = (Gain - Cost) / Cost × 100
```

### 3. DATA REQUIREMENTS
- Every statistic needs source: "Value (Source: Org Year)"
- Include realistic industry benchmarks
- Worked calculation examples with formulas

### 4. COMPANY-SPECIFIC
- Use "{company_name}" explicitly (not "the company")
- Reference scenario-specific challenges and opportunities

### 5. REJECTION TRIGGERS (your fix will be rejected if):
- markdownText under 300 words
- Statistics without sources
- Empty or placeholder content like "[TBD]"
- Truncated sentences ending with "..."

Return JSON with fixes array. Each fix needs: path, current_value, new_value, reason, fix_type="semantic"
The new_value for markdownText fields MUST be comprehensive (500+ words).
"""

# KLO-Question Alignment Prompt - KEY for alignment score
KLO_QUESTION_ALIGNMENT_PROMPT = """You are an instructional designer fixing KLO-Question alignment.

CONTEXT:
- Industry: {industry}
- Company: {company_name}
- KLOs: {klos}

CONTENT TO FIX:
```json
{content}
```

ALIGNMENT ISSUES:
{issues}

REQUIREMENTS:
1. Each KLO MUST have at least one question that assesses it
2. Each question must clearly map to a specific KLO
3. Questions should use action verbs matching KLO cognitive levels
4. Activities must let learners demonstrate KLO mastery

Return JSON with fixes array. Each fix needs: path, current_value, new_value, reason, fix_type="semantic"
"""

# Scenario Coherence Prompt
SCENARIO_COHERENCE_PROMPT = """You are a scenario designer ensuring internal consistency.

CONTEXT:
- Industry: {industry}
- Company: {company_name}
- Role: {learner_role}
- Challenge: {challenge}

CONTENT TO FIX:
```json
{content}
```

COHERENCE ISSUES:
{issues}

REQUIREMENTS:
1. Role responsibilities must match assigned tasks
2. Tasks must lead to the stated challenge/goal
3. Resources must support task completion
4. All names, dates, facts must be internally consistent

Return JSON with fixes array. Each fix needs: path, current_value, new_value, reason, fix_type="semantic"
"""

# Rubric Contextualization Prompt
RUBRIC_PROMPT = """You are an assessment specialist contextualizing rubrics.

CONTEXT:
- Industry: {industry}
- Company: {company_name}
- KLOs: {klos}

CONTENT TO FIX:
```json
{content}
```

ISSUES:
{issues}

REQUIREMENTS:
1. Make criteria specific to the industry
2. Use industry-appropriate terminology
3. Ensure criteria align with KLOs
4. Keep star rating structure (1-5)

Return JSON with fixes array. Each fix needs: path, current_value, new_value, reason, fix_type="semantic"
"""


# Mapping from issue rule_id to prompt
ISSUE_TO_PROMPT = {
    # Metrics issues
    "data_consistency": "metrics",
    "realism": "metrics",
    "inference_integrity": "metrics",

    # Persona/entity issues
    "entity_removal": "personas",
    "domain_fidelity": "personas",
    "tone": "personas",
    "reporting_manager_consistency": "personas",
    "company_consistency": "personas",
    "poison_term_avoidance": "personas",

    # Resource alignment issues
    "contentcompleteness": "resources",
    "resource_self_contained": "resources",
    "klo_to_resources": "resources",
    "scenario_to_resources": "resources",

    # KLO alignment issues
    "context_fidelity": "klo_alignment",
    "klo_to_questions": "klo_alignment",
    "klo_task_alignment": "klo_alignment",

    # Coherence issues
    "role_to_tasks": "coherence",
    "scenario_coherence": "coherence",
}

# Mapping from shard type to prompt
SHARD_TO_PROMPT = {
    "resources": "resources",
    "rubrics": "rubrics",
    "emails": "personas",
    "workplace_scenario": "coherence",
    "simulation_flow": "klo_alignment",
    "assessment_criteria": "klo_alignment",
    "industry_activities": "klo_alignment",
    "selected_scenario": "coherence",
    "scenario_chat_history": "personas",
    "lesson_information": "personas",
    "videos": "resources",
    "launch_settings": "metrics",
}

PROMPTS = {
    "metrics": METRICS_KPI_PROMPT,
    "personas": PERSONAS_COMMS_PROMPT,
    "resources": RESOURCE_ALIGNMENT_PROMPT,
    "klo_alignment": KLO_QUESTION_ALIGNMENT_PROMPT,
    "coherence": SCENARIO_COHERENCE_PROMPT,
    "rubrics": RUBRIC_PROMPT,
}


def get_prompt_for_shard(shard_id: str) -> str:
    """Get the appropriate prompt for a shard type."""
    prompt_type = SHARD_TO_PROMPT.get(shard_id, "personas")
    return PROMPTS.get(prompt_type, PERSONAS_COMMS_PROMPT)


def get_prompt_type_for_shard(shard_id: str) -> str:
    """Get the prompt type name for a shard."""
    return SHARD_TO_PROMPT.get(shard_id, "personas")


def categorize_issues(issues: list) -> dict:
    """Categorize issues by their prompt type."""
    categories = {k: [] for k in PROMPTS.keys()}

    for issue in issues:
        if isinstance(issue, dict):
            rule_id = issue.get("rule_id", "")
        else:
            rule_id = getattr(issue, "rule_id", "")

        prompt_type = ISSUE_TO_PROMPT.get(rule_id, "personas")
        categories[prompt_type].append(issue)

    return categories
