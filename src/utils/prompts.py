"""
Centralized Prompt Templates for Simulation Adaptation.

All LLM prompts are stored here for:
- Easy tracking and version control
- Consistent prompt engineering
- A/B testing different prompts
- Separation of concerns (prompts vs code logic)

USAGE:
    from .prompts import (
        build_factsheet_prompt,
        build_shard_adaptation_prompt,
        build_regeneration_prompt,
    )
"""

# =============================================================================
# GLOBAL FACTSHEET EXTRACTION PROMPT
# =============================================================================

FACTSHEET_SYSTEM_CONTEXT = """Analyze these two business scenarios and extract key facts for the TARGET scenario.

## SOURCE SCENARIO (current):
{source_scenario}

## TARGET SCENARIO (new):
{target_scenario}

## TASK:
Extract a comprehensive factsheet for the TARGET scenario. This will ensure CONSISTENCY and ALIGNMENT across all simulation content."""

FACTSHEET_OUTPUT_FORMAT = """
## OUTPUT FORMAT (JSON):
{{
  "company": {{
    "name": "The company name from target scenario (EXACT spelling)",
    "industry": "Industry type (hospitality, retail, beverage, tech, etc.)",
    "size": "Company size descriptor if mentioned",
    "founding_year": "Year founded if mentioned, or reasonable estimate",
    "headquarters": "Location if mentioned"
  }},
  "financials": {{
    "revenue": "Revenue figure if mentioned, or realistic estimate for company size",
    "growth_target": "Growth target if mentioned",
    "key_metric": "Primary business metric (e.g., market share, revenue growth)",
    "secondary_metrics": ["List 3-5 secondary metrics relevant to the scenario"]
  }},
  "products": {{
    "main_product": "Primary product or service",
    "secondary": "Secondary offerings if any",
    "product_details": ["List 3-5 specific product attributes that can be cited"]
  }},
  "context": {{
    "challenge": "Main business challenge",
    "market": "Market context with specific details (size, growth rate, trends)",
    "simulation_type": "Type of simulation (e.g., Strategic Analysis, Product Launch, etc.)",
    "strategic_focus": "What the learner should focus on analyzing"
  }},
  "learner_role": {{
    "role": "Learner's job title (e.g., Strategic Analyst, Product Manager)",
    "description": "Brief description of what the learner does",
    "key_responsibilities": ["List 3-4 key tasks the learner performs"]
  }},
  "reporting_manager": {{
    "name": "Manager's full name (generate appropriate name for the scenario)",
    "role": "Manager's job title appropriate for the industry",
    "email": "Generate appropriate email based on name and company (NO trailing period)",
    "gender": "Infer from the name you generate (Male/Female - capitalized)"
  }},
  "industry_context": {{
    "kpis": ["List 10-15 KEY PERFORMANCE INDICATORS specific to this industry"],
    "terminology": ["List 15-20 industry-specific TERMS to use"],
    "wrong_terms": ["List terms that should NOT be used for this industry"],
    "data_types": ["List 5-10 types of DATA/REPORTS that would be realistic"]
  }},
  "alignment_guidance": {{
    "klo_themes": ["List 5-7 KEY LEARNING OUTCOMES themes for this scenario"],
    "question_types": ["List 5-7 types of QUESTIONS that test these KLOs"],
    "resource_requirements": ["List 5-7 types of DATA resources need to contain"]
  }},
  "poison_list": [
    "List ALL terms from SOURCE scenario that should NOT appear in adapted content",
    "Include: company name, ALL product names, industry-specific terms, role names, named individuals"
  ],
  "replacement_hints": {{
    "source_company": "target_company",
    "source_product": "target_product",
    "source_role": "target_role"
  }},
  "citable_facts": [
    "List 10-15 SPECIFIC facts/numbers that can be cited in resources"
  ],
  "shard_rules": {{
    "resources": [
      "MUST: Every statistic needs a source citation",
      "MUST: Include a SCORING MODEL TEMPLATE for KLO3",
      "MUST: Include a QUESTION DESIGN GUIDE for KLO2",
      "MUST: Include industry-specific metrics with realistic values",
      "MUST: Include competitor data table with at least 3 named competitors"
    ],
    "rubrics": [
      "MUST: Each KLO must be directly testable by at least one question",
      "MUST: KLO2 should explicitly require learners to CREATE analysis questions",
      "MUST: KLO3 should explicitly require learners to BUILD a scoring model",
      "MUST: Criteria must reference specific data that EXISTS in resources"
    ],
    "simulation_flow": [
      "MUST: NO DUPLICATE questions or activities",
      "MUST: Each question must explicitly map to a specific KLO",
      "MUST: Questions must be answerable using ONLY the data in resources",
      "MUST: Include activity for KLO2 where learner writes their own analysis questions",
      "MUST: Include activity for KLO3 where learner builds a scoring/weighting model"
    ],
    "emails": [
      "MUST: Use exact manager name and email from factsheet",
      "MUST: Reference specific deliverables that match the KLOs",
      "MUST: Mention the company name explicitly"
    ],
    "workplace_scenario": [
      "MUST: Use exact company name from factsheet throughout",
      "MUST: Include specific industry metrics with sources",
      "MUST: Define learner role clearly with 3-5 specific responsibilities"
    ]
  }},
  "required_templates": {{
    "scoring_model_template": "Generate a COMPLETE scoring model template with: criteria names, weight percentages, 1-5 scale definitions, and a worked calculation example",
    "question_design_guide": "Generate a COMPLETE question design guide with: how to write leading questions, probing questions, bias checks, validity/reliability criteria",
    "industry_metrics": [
      "List 10+ industry-specific metrics with REALISTIC benchmark values for this industry"
    ]
  }}
}}

Return ONLY valid JSON:"""


def build_factsheet_prompt(source_scenario: str, target_scenario: str) -> str:
    """
    Build the global factsheet extraction prompt.

    Args:
        source_scenario: The original/source scenario text
        target_scenario: The new/target scenario text

    Returns:
        Complete prompt string for factsheet extraction
    """
    return FACTSHEET_SYSTEM_CONTEXT.format(
        source_scenario=source_scenario,
        target_scenario=target_scenario
    ) + FACTSHEET_OUTPUT_FORMAT


# =============================================================================
# SHARD ADAPTATION PROMPT
# =============================================================================

SHARD_ADAPTATION_SYSTEM = """You are a High-Fidelity Simulation Adapter for UG (undergraduate) business education.
Your job is to rewrite a specific component (Shard) of a business simulation to fit a NEW context."""

SHARD_ADAPTATION_SCENARIO_SECTION = """
## 1. SCENARIO TRANSITION:
**FROM:** {source_scenario}...
**TO:** {target_scenario}...
{factsheet_text}
{poison_list_text}
"""

SHARD_ADAPTATION_INFO_SECTION = """
## 2. SHARD INFO:
- Shard ID: {shard_id}
- Shard Name: {shard_name}

{shard_rules}
"""

SHARD_ADAPTATION_CONTEXT_SECTION = """
## 3. ADDITIONAL CONTEXT FROM RAG:
{rag_context}

## 4. INDUSTRY-SPECIFIC TERMS (USE THESE):
- KPIs for this industry: {industry_kpis}
- Terminology: {industry_terms}

## 5. CONTENT TO ADAPT:
```json
{content_json}
```
"""

# Blocker rules that cause instant failure
BLOCKER_RULES = """
## 6. CRITICAL ALIGNMENT RULES - VIOLATIONS = AUTOMATIC FAILURE

### BLOCKER A: NO DUPLICATES (INSTANT FAIL)
**BEFORE OUTPUTTING, CHECK:** Are there ANY duplicate activity names or questions?
- SCAN all activities - EACH must have UNIQUE name
- SCAN all questions - NO verbatim duplicates
- **IF YOU CREATE DUPLICATES, THE ENTIRE OUTPUT FAILS VALIDATION**

### BLOCKER B: NO FAKE/GENERIC SOURCES (INSTANT FAIL)
**EVERY statistic MUST have a REAL, VERIFIABLE source:**
- WRONG: "15% CAGR" (no source)
- WRONG: "Retail Analytics Inc." (fake company)
- RIGHT: "Market size: $45B (Source: McKinsey State of Fashion 2024)"
**ACCEPTABLE SOURCES:** McKinsey, Bain, BCG, Deloitte, NielsenIQ, Euromonitor, Statista, IBISWorld, Company Internal Reports, SEC Filings, Trade Associations

### BLOCKER C: NO TRUNCATION (INSTANT FAIL)
**ALL text MUST be COMPLETE - never cut off mid-sentence**
- WRONG: "such as targ..." or "previously analyze..."
- RIGHT: Complete sentences with proper endings

### BLOCKER D: COMPETITOR DATA MUST BE SPECIFIC (INSTANT FAIL)
**When mentioning competitors, include ALL of:**
1. Company name (real or realistic)
2. Market share % with source
3. Price range
4. Distribution channels
5. Target demographic

### BLOCKER E: FINANCIAL MODEL REQUIRED (FOR RESOURCES SHARD)
**If this is a RESOURCES shard, MUST include:**
- TAM/SAM/SOM breakdown
- Unit economics: CAC, LTV, Gross Margin, AOV
- Break-even analysis framework

### BLOCKER F: COMPANY NAME EXPLICIT
- ALWAYS use EXACT company name: "{company_name}"
- WRONG: "our brand", "the company", "this organization"
- RIGHT: "{company_name}'s market analysis..."

### BLOCKER G: WRONG INDUSTRY TERMS
{forbidden_terms}
- Replace with industry-appropriate equivalents

### KLO-RESOURCE ALIGNMENT:
- EVERY KLO must have supporting data in resources WITH SOURCES

### KLO-QUESTION ALIGNMENT:
- EVERY question must map to specific KLOs
"""

VALIDATION_CHECKLIST = """
## 7. PRE-OUTPUT VALIDATION CHECKLIST:
- Are there ANY duplicate activity names? FIX THEM
- Are there ANY truncated sentences? COMPLETE THEM
- Does EVERY statistic have a real source? ADD SOURCES
- Is competitor data specific? ADD DETAILS
- Is company name used explicitly? FIX IT
- Are resources complete? ADD CONTENT
"""

STANDARD_RULES = """
## 8. STANDARD RULES:
1. **JSON Integrity:** Keep same keys and nesting structure. For CONTENT fields, you CAN expand/enrich.
2. **ID Preservation:** ANY field containing "id" or "Id" must NOT change
3. **Data Consistency:** Use values from GLOBAL FACTSHEET. Do NOT invent conflicting numbers.
4. **Poison Avoidance:** Do NOT use any term from the POISON LIST.
5. **Pedagogical Logic:** Questions should NOT reveal answers; Resources provide raw data, not conclusions
6. **Professional Tone:** Maintain business simulation quality
7. **ADAPT ALL NESTED CONTENT:** ALL names, emails, organizations must be updated
8. **NO TRUNCATION:** COMPLETE ALL CONTENT
9. **NO DUPLICATES:** Each activity must have unique name
10. **NO TRAILING PERIODS** on names/emails/roles
"""

SHARD_OUTPUT_FORMAT = """
## OUTPUT FORMAT:
Return valid JSON with exactly these keys:
{{
  "adapted_content": {{ ... the transformed content with same structure ... }},
  "entity_mappings": {{
    "old_term": "new_term"
  }},
  "changes_summary": ["Brief list of key changes made"],
  "alignment_notes": ["How KLOs map to resources", "How questions test KLOs"]
}}

Return ONLY valid JSON, no explanations:"""


def build_factsheet_section(global_factsheet: dict) -> tuple[str, str]:
    """
    Build the factsheet and poison list sections for the prompt.

    Args:
        global_factsheet: The global factsheet dictionary

    Returns:
        Tuple of (factsheet_text, poison_list_text)
    """
    if not global_factsheet or not isinstance(global_factsheet, dict):
        return "", ""

    # Extract nested values safely
    company = global_factsheet.get('company', {})
    company = company if isinstance(company, dict) else {}
    products = global_factsheet.get('products', {})
    products = products if isinstance(products, dict) else {}
    financials = global_factsheet.get('financials', {})
    financials = financials if isinstance(financials, dict) else {}
    context = global_factsheet.get('context', {})
    context = context if isinstance(context, dict) else {}
    learner_role = global_factsheet.get('learner_role', {})
    learner_role = learner_role if isinstance(learner_role, dict) else {}
    reporting_manager = global_factsheet.get('reporting_manager', {})
    reporting_manager = reporting_manager if isinstance(reporting_manager, dict) else {}

    factsheet_text = f"""
## GLOBAL FACTSHEET (Use these EXACT values for consistency across ALL shards):

### Company & Context:
- Company Name: {company.get('name', 'Unknown')}
- Industry: {company.get('industry', 'Unknown')}
- Main Product/Service: {products.get('main_product', 'Unknown')}
- Business Challenge: {context.get('challenge', 'Not specified')}
- Simulation Type: {context.get('simulation_type', 'Strategic Analysis')}

### Financials (if applicable):
- Revenue: {financials.get('revenue', 'Not specified')}
- Key Metric: {financials.get('key_metric', 'Not specified')}

### Learner Role (USE THIS EXACT ROLE):
- Role Title: {learner_role.get('role', 'Analyst')}
- Role Description: {learner_role.get('description', 'Not specified')}

### Reporting Manager (USE THIS EXACT PERSON - CRITICAL FOR CONSISTENCY):
- Name: {reporting_manager.get('name', 'Unknown Manager')}
- Title: {reporting_manager.get('role', 'Director')}
- Email: {reporting_manager.get('email', 'manager@company.com')}
- Gender: {reporting_manager.get('gender', 'female')}

**IMPORTANT:** The reporting manager must be IDENTICAL in all emails, video descriptions, and scenario text.
"""

    poison_list = global_factsheet.get('poison_list', [])
    poison_list = poison_list if isinstance(poison_list, list) else []
    poison_list_text = ""
    if poison_list:
        import json
        poison_list_text = f"""
## POISON LIST (DO NOT use these terms - they are from the OLD scenario):
{json.dumps(poison_list, indent=2)}
Replace any of these terms with appropriate TARGET scenario equivalents.
"""

    return factsheet_text, poison_list_text


def get_shard_specific_rules(shard_id: str, factsheet: dict, company_name: str, industry: str) -> str:
    """
    Get shard-specific rules from factsheet (DYNAMIC, not hardcoded).

    Args:
        shard_id: The shard identifier
        factsheet: The global factsheet
        company_name: Company name for the scenario
        industry: Industry type

    Returns:
        Formatted rules string for the shard
    """
    shard_lower = shard_id.lower()
    shard_rules_dict = factsheet.get('shard_rules', {}) if factsheet else {}

    # Determine which shard type this is
    if "resource" in shard_lower:
        rules = shard_rules_dict.get('resources', [])
        shard_type = "RESOURCES"
    elif "rubric" in shard_lower:
        rules = shard_rules_dict.get('rubrics', [])
        shard_type = "RUBRICS"
    elif "simulation_flow" in shard_lower or "sim_flow" in shard_lower:
        rules = shard_rules_dict.get('simulation_flow', [])
        shard_type = "SIMULATION FLOW"
    elif "email" in shard_lower:
        rules = shard_rules_dict.get('emails', [])
        shard_type = "EMAILS"
    elif "workplace" in shard_lower or "scenario" in shard_lower:
        rules = shard_rules_dict.get('workplace_scenario', [])
        shard_type = "WORKPLACE SCENARIO"
    else:
        rules = []
        shard_type = "GENERAL"

    # Build rules text from dynamic list
    rules_text = ""
    if rules and isinstance(rules, list):
        actual_rules = [r for r in rules if isinstance(r, str) and r.startswith("MUST:")]
        if actual_rules:
            rules_text = "\n".join([f"{i+1}. {rule}" for i, rule in enumerate(actual_rules)])

    # Add templates for resources shard
    templates_text = ""
    if "resource" in shard_lower:
        templates_text = f"""

## MANDATORY RESOURCE REQUIREMENTS:

### 1. COMPETITOR TABLE (REQUIRED):
Include a competitor analysis table with REAL data - names, market share, price range, channels, demographics.

### 2. FINANCIAL MODEL TEMPLATE (REQUIRED):
Include unit economics with example numbers - TAM/SAM/SOM, AOV, CAC, LTV, margins.

### 3. MARKET DATA WITH SOURCES (REQUIRED):
Every statistic needs format: "Value (Source: Organization Year)"

### 4. SCORING RUBRIC TEMPLATE (if KLO mentions evaluation):
Include evaluation criteria with weights and scale definitions.
"""

    # Add anti-duplicate rules for simulation_flow
    duplicate_rules = ""
    if "simulation_flow" in shard_lower or "sim_flow" in shard_lower:
        duplicate_rules = """

## ANTI-DUPLICATE RULES:
1. List all activity names you're outputting
2. Check: Are ANY names identical or near-identical?
3. If YES - RENAME one to be distinct
4. EACH ACTIVITY MUST BE UNIQUE
"""

    if rules_text:
        return f"""
## SHARD-SPECIFIC RULES FOR {shard_type}:
{rules_text}
- Use company name "{company_name}" throughout
- Use {industry}-appropriate terminology
{templates_text}
{duplicate_rules}
"""

    return f"""
## SHARD RULES FOR {shard_type}:
- Use company name "{company_name}" explicitly throughout
- Use {industry}-appropriate terminology and metrics
- Ensure content is specific to the scenario (not generic)
- All data should be citable with sources
{templates_text}
{duplicate_rules}
"""


def build_shard_adaptation_prompt(
    shard_id: str,
    shard_name: str,
    content: dict,
    source_scenario: str,
    target_scenario: str,
    global_factsheet: dict = None,
    rag_context: str = "",
) -> str:
    """
    Build the complete shard adaptation prompt.

    Args:
        shard_id: Shard identifier
        shard_name: Human-readable shard name
        content: Shard content to transform (dict)
        source_scenario: Original scenario text
        target_scenario: Target scenario text
        global_factsheet: Pre-extracted global facts
        rag_context: Additional context from RAG

    Returns:
        Complete prompt string for shard adaptation
    """
    import json

    # Build factsheet sections
    factsheet_text, poison_list_text = build_factsheet_section(global_factsheet)

    # Extract company info for rules
    company = {}
    industry_context = {}
    if global_factsheet:
        company = global_factsheet.get('company', {})
        company = company if isinstance(company, dict) else {}
        industry_context = global_factsheet.get('industry_context', {})
        industry_context = industry_context if isinstance(industry_context, dict) else {}

    company_name = company.get('name', 'the company')
    industry = company.get('industry', 'business')

    # Get shard-specific rules
    shard_rules = get_shard_specific_rules(shard_id, global_factsheet, company_name, industry)

    # Build industry terms
    industry_kpis = industry_context.get('kpis', []) if isinstance(industry_context, dict) else []
    industry_terms = industry_context.get('terminology', []) if isinstance(industry_context, dict) else []

    kpis_str = ', '.join(industry_kpis[:10]) if industry_kpis else 'market share, revenue growth, customer satisfaction'
    terms_str = ', '.join(industry_terms[:10]) if industry_terms else 'Use industry-appropriate terms'

    # Build forbidden terms
    wrong_terms = industry_context.get('wrong_terms', []) if industry_context else []
    forbidden_terms = f"FORBIDDEN TERMS for {industry}: {', '.join(wrong_terms[:15])}" if wrong_terms else "Avoid tech/SaaS terms in non-tech industries"

    # Build blocker rules with company name
    blocker_section = BLOCKER_RULES.format(
        company_name=company_name,
        forbidden_terms=forbidden_terms
    )

    # Assemble the prompt
    prompt_parts = [
        SHARD_ADAPTATION_SYSTEM,
        SHARD_ADAPTATION_SCENARIO_SECTION.format(
            source_scenario=source_scenario[:200],
            target_scenario=target_scenario[:200],
            factsheet_text=factsheet_text,
            poison_list_text=poison_list_text
        ),
        SHARD_ADAPTATION_INFO_SECTION.format(
            shard_id=shard_id,
            shard_name=shard_name,
            shard_rules=shard_rules
        ),
        SHARD_ADAPTATION_CONTEXT_SECTION.format(
            rag_context=rag_context or "No additional context available.",
            industry_kpis=kpis_str,
            industry_terms=terms_str,
            content_json=json.dumps(content, indent=2, default=str)
        ),
        blocker_section,
        VALIDATION_CHECKLIST,
        STANDARD_RULES,
        SHARD_OUTPUT_FORMAT
    ]

    return "".join(prompt_parts)


# =============================================================================
# REGENERATION PROMPT
# =============================================================================

REGENERATION_SYSTEM = """You are REGENERATING simulation content that FAILED validation."""

REGENERATION_FEEDBACK_SECTION = """
## REGENERATION MODE - FIX THESE ISSUES:

### Failed Validation Rules:
{failed_rules}

### Critical Issues to Address:
{critical_issues}

### Suggestions for Improvement:
{suggestions}

YOU MUST FIX ALL ISSUES ABOVE. This is a regeneration attempt after the first pass failed validation.
"""

REGENERATION_FIX_REQUIREMENTS = """
## FIX REQUIREMENTS:
1. Address ALL failed rules listed above
2. Fix ALL critical issues
3. Apply suggestions for improvement
4. Use EXACT company name: {company_name}
5. Use industry-appropriate terminology for {industry}
6. Remove any wrong terms: {wrong_terms}
7. Ensure KLOs are supported by resources
8. Ensure questions align to KLOs
9. NO duplicate activities
10. NO trailing periods on names/emails
"""

REGENERATION_OUTPUT_FORMAT = """
## OUTPUT FORMAT:
Return valid JSON:
{{
  "adapted_content": {{ ... the FIXED content ... }},
  "entity_mappings": {{ "old": "new" }},
  "fixes_applied": ["List of fixes you applied"]
}}

Return ONLY valid JSON:"""


def build_regeneration_prompt(
    shard_id: str,
    shard_name: str,
    content: dict,
    source_scenario: str,
    target_scenario: str,
    global_factsheet: dict,
    feedback: dict,
) -> str:
    """
    Build the regeneration prompt for failed shards.

    Args:
        shard_id: Shard identifier
        shard_name: Human-readable shard name
        content: Current shard content to fix
        source_scenario: Original scenario text
        target_scenario: Target scenario text
        global_factsheet: Global facts
        feedback: Dict with failed_rules, critical_issues, suggestions

    Returns:
        Complete prompt string for regeneration
    """
    import json

    # Extract feedback components
    failed_rules = feedback.get("failed_rules", [])
    critical_issues = feedback.get("critical_issues", [])
    suggestions = feedback.get("suggestions", [])

    feedback_text = REGENERATION_FEEDBACK_SECTION.format(
        failed_rules=json.dumps(failed_rules[:5], indent=2) if failed_rules else "None",
        critical_issues=json.dumps(critical_issues[:5], indent=2) if critical_issues else "None",
        suggestions=json.dumps(suggestions[:5], indent=2) if suggestions else "None"
    )

    # Extract company info
    company = global_factsheet.get('company', {}) if global_factsheet else {}
    company = company if isinstance(company, dict) else {}
    industry_context = global_factsheet.get('industry_context', {}) if global_factsheet else {}
    industry_context = industry_context if isinstance(industry_context, dict) else {}

    company_name = company.get('name', 'Unknown')
    industry = company.get('industry', 'Unknown')
    wrong_terms = ', '.join(industry_context.get('wrong_terms', [])[:10]) if industry_context.get('wrong_terms') else 'N/A'

    fix_requirements = REGENERATION_FIX_REQUIREMENTS.format(
        company_name=company_name,
        industry=industry,
        wrong_terms=wrong_terms
    )

    prompt = f"""{REGENERATION_SYSTEM}

## SCENARIO TRANSITION:
**FROM:** {source_scenario[:200]}...
**TO:** {target_scenario[:200]}...

## COMPANY (use EXACT name):
{company_name} - {industry}

{feedback_text}

## SHARD TO FIX:
- Shard ID: {shard_id}
- Shard Name: {shard_name}

## CURRENT CONTENT (needs fixing):
```json
{json.dumps(content, indent=2, default=str)}
```

{fix_requirements}

{REGENERATION_OUTPUT_FORMAT}"""

    return prompt


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_acceptable_sources() -> list[str]:
    """
    Get list of acceptable data sources for citations.

    Returns:
        List of acceptable source names
    """
    return [
        "McKinsey",
        "Bain",
        "BCG",
        "Deloitte",
        "NielsenIQ",
        "Euromonitor",
        "Statista",
        "IBISWorld",
        "Company Internal Reports",
        "SEC Filings",
        "Trade Associations",
        "Harvard Business Review",
        "Gartner",
        "Forrester",
        "PwC",
        "EY",
        "KPMG",
    ]


def get_blocker_rule_names() -> list[str]:
    """
    Get list of blocker rule names for reference.

    Returns:
        List of blocker rule identifiers
    """
    return [
        "NO_DUPLICATES",
        "NO_FAKE_SOURCES",
        "NO_TRUNCATION",
        "COMPETITOR_DATA_SPECIFIC",
        "FINANCIAL_MODEL_REQUIRED",
        "COMPANY_NAME_EXPLICIT",
        "NO_WRONG_INDUSTRY_TERMS",
    ]
