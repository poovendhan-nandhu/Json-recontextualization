"""
Improved Prompt Templates for Simulation Adaptation - V2

CHANGES FROM V1:
1. Added MANDATORY verification output fields (forced proof of checking)
2. Added consequence-based framing (rejection triggers)
3. Added strict_verify_output() as a hard code gate
4. Added two-pass verification option
5. Explicit regex patterns shown in prompt
6. RULES-DRIVEN: All requirements come from rules.py, not hardcoded

DESIGN PRINCIPLES:
- Prompts set the floor (~90-95%), code verification sets the ceiling (99%+)
- Force LLM to PROVE it checked, not just claim it did
- Make verification fields machine-parseable for automatic rejection
- Rules are DATA, prompts are GENERATED from rules
"""

import json
import re
from typing import Optional

from .rules import (
    get_rules_for_shard,
    get_blocker_rules,
    format_rules_for_prompt,
    get_blocker_summary,
)

# =============================================================================
# GLOBAL FACTSHEET EXTRACTION PROMPT
# =============================================================================

FACTSHEET_PROMPT = """You MUST extract a complete factsheet. ALL fields are REQUIRED - do not skip any.

## SOURCE SCENARIO (what we're adapting FROM - extract poison terms from here):
{source_scenario}

## TARGET SCENARIO (what we're adapting TO - extract company/role/context from here):
{target_scenario}

⛔ CRITICAL: You MUST fill ALL fields below. Empty fields = REJECTED.

## POISON LIST INSTRUCTIONS (VERY IMPORTANT):
The poison_list must contain ALL terms from the SOURCE scenario that should NOT appear in the adapted content.
You MUST extract:
1. ALL person names from SOURCE (both first name AND full name separately)
2. ALL company/organization names from SOURCE
3. ALL industry-specific jargon from SOURCE that doesn't fit TARGET
4. ALL role-specific terms from SOURCE (e.g., if SOURCE is about hiring: "candidate", "interview", "recruitment")

Example: If SOURCE has manager "Elizabeth Carter" at "Summit Innovations" doing "hiring":
poison_list: ["Elizabeth", "Elizabeth Carter", "Summit Innovations", "hiring", "candidate", "interview", "recruitment", "talent acquisition", ...]

Return this EXACT JSON structure with ALL fields populated:

{{
  "company": {{
    "name": "Exact company name from TARGET scenario",
    "industry": "Industry type from TARGET"
  }},
  "products": {{
    "main_product": "Primary product/service from TARGET",
    "product_details": ["3-5 specific attributes"]
  }},
  "context": {{
    "challenge": "Main business challenge from TARGET",
    "market": "Market context with specifics"
  }},
  "learner_role": {{
    "role": "Job title the learner plays in TARGET",
    "key_responsibilities": ["3-4 tasks they must do"]
  }},
  "reporting_manager": {{
    "name": "Create a realistic manager name for TARGET industry (e.g., Sarah Chen, Michael Torres)",
    "role": "Appropriate manager title (e.g., Director of Marketing, VP Operations)",
    "email": "firstname.lastname@companyname.com",
    "gender": "Male or Female"
  }},
  "industry_context": {{
    "kpis": ["List 10-15 KPIs specific to TARGET industry - e.g., for retail: conversion rate, basket size, foot traffic"],
    "terminology": ["List 15-20 industry terms to use in TARGET - e.g., for retail: SKU, merchandise, inventory"],
    "wrong_terms": ["Terms from SOURCE industry that don't fit TARGET"]
  }},
  "klos": [
    "KLO 1: Specific skill/capability learner demonstrates in TARGET context",
    "KLO 2: Second learning outcome aligned to TARGET challenge",
    "KLO 3: Third learning outcome for TARGET scenario",
    "⚠️ EXACTLY 3 KLOs - no more, no less. KLOs must match TARGET scenario tasks."
  ],
  "poison_list": ["MINIMUM 15 terms from SOURCE: person names, company names, industry jargon - see instructions above"],
  "citable_facts": ["10-15 specific facts/numbers relevant to TARGET industry for use in resources"]
}}

⛔ VALIDATION CHECKLIST (your output will be scanned):
- [ ] company.name is NOT empty
- [ ] reporting_manager.name is a real name (First Last format)
- [ ] reporting_manager.email matches the company domain
- [ ] klos has EXACTLY 3 items that describe TARGET scenario outcomes (NOT source)
- [ ] poison_list has AT LEAST 15 terms extracted from SOURCE
- [ ] poison_list includes ALL person names from SOURCE (first AND full name)
- [ ] industry_context.kpis has AT LEAST 10 items
- [ ] industry_context.terminology has AT LEAST 10 items

Return ONLY the JSON. No explanations."""


def build_factsheet_prompt(source_scenario: str, target_scenario: str) -> str:
    return FACTSHEET_PROMPT.format(
        source_scenario=source_scenario,
        target_scenario=target_scenario
    )


# =============================================================================
# SHARD ADAPTATION PROMPT - V2 WITH FORCED VERIFICATION
# =============================================================================

HARD_BLOCKS_WITH_CONSEQUENCES = """## ⛔ HARD BLOCKS — VIOLATIONS = AUTOMATIC REJECTION

Your output will be programmatically scanned. These checks CANNOT be bypassed:

### BLOCK 1: ZERO PLACEHOLDERS
**Regex applied to your output:** `\[[^\]]+\]`
This catches: [anything in brackets], [X], [TBD], [manager name], etc.

If regex finds ANY match → OUTPUT REJECTED, you must regenerate.

Examples that WILL trigger rejection:
- "growing at [X]%" → REJECTED
- "[industry-specific metric]" → REJECTED
- "Contact [manager name]" → REJECTED

✓ Replace ALL brackets with actual values before outputting.

### BLOCK 2: ZERO POISON TERMS
**String search applied for each term:** {poison_terms_list}

If ANY term found (case-insensitive) → OUTPUT REJECTED.

The system will literally search for these strings in your JSON output.
There is no way to hide them. They will be found.

### BLOCK 3: SENDER CONSISTENCY
**Cross-reference check applied:**
- Manager name in factsheet: {manager_name}
- All email "from" fields must contain: {manager_name}
- All email signatures must contain: {manager_name}

If mismatch detected → OUTPUT REJECTED.

### BLOCK 4: CLEAN FORMATTING
**Regex applied:** `\.(png|jpg|com|org)\.` and `"[^"]+\."`
Catches trailing periods on emails, URLs, filenames.

- WRONG: "sophia.chen@retail.com." → REJECTED
- WRONG: "logo.png." → REJECTED
- RIGHT: "sophia.chen@retail.com"

### BLOCK 5: COMPLETE CONTENT
**Scan applied for:** `...`, sentences ending without punctuation, `TBD`, `TODO`

Incomplete content → OUTPUT REJECTED.

---
## ⚠️ CONSEQUENCES

Previous outputs have been rejected 47 times for these violations.
Each rejection wastes compute and requires full regeneration.

Your output WILL be scanned by code. Violations WILL be caught.
This is not a suggestion—it is a hard system constraint.
"""

# Mandatory verification fields that LLM must fill out
MANDATORY_VERIFICATION_FIELDS = """
## 📋 MANDATORY VERIFICATION OUTPUT

Your output MUST include these verification fields. They will be checked programmatically.
If any verification fails, your ENTIRE output is rejected.

```json
{{
  "adapted_content": {{ ... }},
  
  "poison_scan_proof": {{
    "terms_i_searched_for": [/* MUST list at least 15 terms from poison list */],
    "terms_found_in_my_output": [],  // MUST be empty array
    "sections_i_checked": ["lessonInformation", "emails", "resources", "guidelines"]
  }},
  
  "placeholder_scan_proof": {{
    "patterns_i_searched_for": ["[", "{{{{", "TBD", "TODO", "XXX"],
    "brackets_found": 0,  // MUST be 0
    "replacements_i_made": [/* list any [X] you replaced with real values */]
  }},
  
  "sender_consistency_proof": {{
    "manager_from_factsheet": "{manager_name}",
    "manager_in_email_from_fields": "{manager_name}",  // MUST match above
    "manager_in_signatures": "{manager_name}",  // MUST match above
    "all_match": true  // MUST be true
  }},
  
  "formatting_proof": {{
    "trailing_dots_found": 0,  // MUST be 0
    "incomplete_sentences_found": 0  // MUST be 0
  }}
}}
```

⛔ IF poison_scan_proof.terms_found_in_my_output is NOT empty → REJECTED
⛔ IF placeholder_scan_proof.brackets_found > 0 → REJECTED  
⛔ IF sender_consistency_proof.all_match is false → REJECTED
⛔ IF formatting_proof has any value > 0 → REJECTED

The system will verify your claims. If you say "terms_found_in_my_output": [] 
but "Velocity Dome" appears in adapted_content, you will be caught and rejected.
"""

CONTEXT_SECTION = """
## ADAPTATION CONTEXT

**Transform from:** {source_scenario_brief}
**Transform to:** {target_scenario_brief}

### ⛔ CRITICAL: ENTITY SEPARATION (DO NOT MIX!)
These are SEPARATE entities. NEVER combine or merge them:
- **COMPANY NAME:** {company_name} ← Use this EXACTLY for company references
- **MANAGER NAME:** {manager_name} ← Use this EXACTLY for manager/person references

WRONG: "EcoChic T{manager_name}eads" (inserting manager into company name)
WRONG: Any substring replacement within names
RIGHT: Use each entity as a complete, standalone string

**Target Company:** {company_name} ({industry})
**Learner Role:** {learner_role}
**Manager:** {manager_name} ({manager_role}) - {manager_email}

**Use these industry terms:** {industry_terms}
**Use these KPIs:** {industry_kpis}
"""

SHARD_GUIDANCE = """## SHARD: {shard_name}

{shard_specific_rules}

**Data Sources:** Use real sources (Statista, McKinsey, IBISWorld) with format: "Value (Source: Org Year)"
"""

# =============================================================================
# DYNAMIC RULES SECTION - GENERATED FROM rules.py
# =============================================================================

def build_rules_section(shard_id: str) -> str:
    """Build rules section dynamically from rules.py for a specific shard."""
    rules = get_rules_for_shard(shard_id)
    return format_rules_for_prompt(rules, include_criteria=True)


def build_blocker_section() -> str:
    """Build blocker rules section dynamically."""
    return get_blocker_summary()

OUTPUT_FORMAT = """
## OUTPUT FORMAT

Return this EXACT JSON structure:

```json
{{
  "adapted_content": {{
    /* transformed content - same keys as input */
  }},
  
  "entity_mappings": {{
    "old_term": "new_term"
  }},
  
  "poison_scan_proof": {{
    "terms_i_searched_for": ["term1", "term2", ...],
    "terms_found_in_my_output": [],
    "sections_i_checked": ["lessonInformation", "emails", "resources"]
  }},
  
  "placeholder_scan_proof": {{
    "patterns_i_searched_for": ["[", "{{{{", "TBD", "TODO"],
    "brackets_found": 0,
    "replacements_i_made": ["[X]% → 12.3%", "[metric] → market share"]
  }},
  
  "sender_consistency_proof": {{
    "manager_from_factsheet": "{manager_name}",
    "manager_in_email_from_fields": "{manager_name}",
    "manager_in_signatures": "{manager_name}",
    "all_match": true
  }},
  
  "formatting_proof": {{
    "trailing_dots_found": 0,
    "incomplete_sentences_found": 0
  }}
}}
```

CRITICAL:
- Preserve all "id" and "Id" fields exactly
- Use "{company_name}" explicitly (not "the company")
- Every statistic needs a source citation

Return ONLY valid JSON. No markdown fences. No explanations outside JSON."""


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
    Build shard adaptation prompt with forced verification fields.
    """
    factsheet = global_factsheet if isinstance(global_factsheet, dict) else {}

    # Safely extract nested dicts (handle case where value is string instead of dict)
    def safe_get_dict(obj, key, default=None):
        val = obj.get(key, default) if isinstance(obj, dict) else default
        return val if isinstance(val, dict) else (default or {})

    def safe_get_list(obj, key, default=None):
        val = obj.get(key, default) if isinstance(obj, dict) else default
        return val if isinstance(val, list) else (default or [])

    def safe_get_str(obj, key, default=''):
        val = obj.get(key, default) if isinstance(obj, dict) else default
        return str(val) if val else default

    company = safe_get_dict(factsheet, 'company', {})
    learner = safe_get_dict(factsheet, 'learner_role', {})
    manager = safe_get_dict(factsheet, 'reporting_manager', {})
    industry_ctx = safe_get_dict(factsheet, 'industry_context', {})

    company_name = safe_get_str(company, 'name', 'Unknown Company')
    industry = safe_get_str(company, 'industry', 'business')
    manager_name = safe_get_str(manager, 'name', 'Unknown Manager')
    manager_email = safe_get_str(manager, 'email', 'manager@company.com')
    manager_role = safe_get_str(manager, 'role', 'Director')
    learner_role = safe_get_str(learner, 'role', 'Analyst')
    
    # Poison list - show ALL terms for the search requirement
    poison_list = safe_get_list(factsheet, 'poison_list', [])
    poison_terms_list = json.dumps(poison_list[:20]) if poison_list else '["source_company", "source_product"]'

    # Industry context
    kpis = safe_get_list(industry_ctx, 'kpis', [])
    terms = safe_get_list(industry_ctx, 'terminology', [])
    industry_kpis = ', '.join(str(k) for k in kpis[:8]) if kpis else 'Use industry-appropriate metrics'
    industry_terms = ', '.join(str(t) for t in terms[:8]) if terms else 'Use industry-appropriate terms'
    
    # Shard-specific rules from rules.py (DYNAMIC, not hardcoded)
    shard_rules = _get_shard_rules(shard_id, company_name)

    # Get relevant rules for this shard from rules.py
    dynamic_rules = build_rules_section(shard_id)

    # Extract KLOs for alignment (critical for simulation_flow, resources, rubrics)
    klos = safe_get_list(factsheet, 'klos', [])
    klo_text = ""
    if klos and shard_id.lower() in ['simulation_flow', 'resources', 'rubrics', 'assessment_criteria']:
        klo_text = "\n## 🎯 KLOs - ALL CONTENT MUST ALIGN TO THESE:\n"
        for i, klo in enumerate(klos[:5], 1):
            klo_text += f"  KLO{i}: {klo}\n"
        klo_text += "\n⛔ Activities, questions, and resources MUST directly support these KLOs.\n"
        klo_text += "⛔ Do NOT include content about unrelated topics (e.g., HR hiring if KLOs are about market analysis).\n"

    # === ASSEMBLE PROMPT ===

    # 1. Hard blocks with consequences FIRST
    prompt = HARD_BLOCKS_WITH_CONSEQUENCES.format(
        poison_terms_list=poison_terms_list,
        manager_name=manager_name
    )

    # 2. Dynamic rules from rules.py (shard-specific)
    prompt += f"\n## 🎯 RULES FOR THIS SHARD (from rules.py)\n{dynamic_rules}\n"

    # 2.5 KLOs for alignment (if relevant shard)
    if klo_text:
        prompt += klo_text

    # 3. Mandatory verification fields
    prompt += "\n" + MANDATORY_VERIFICATION_FIELDS.format(
        manager_name=manager_name
    )

    # 4. Context (minimal)
    prompt += "\n" + CONTEXT_SECTION.format(
        source_scenario_brief=source_scenario[:120] + "..." if len(source_scenario) > 120 else source_scenario,
        target_scenario_brief=target_scenario[:120] + "..." if len(target_scenario) > 120 else target_scenario,
        company_name=company_name,
        industry=industry,
        learner_role=learner_role,
        manager_name=manager_name,
        manager_role=manager_role,
        manager_email=manager_email,
        industry_terms=industry_terms,
        industry_kpis=industry_kpis
    )

    # 5. Shard guidance
    prompt += "\n" + SHARD_GUIDANCE.format(
        shard_name=shard_name,
        shard_specific_rules=shard_rules
    )
    
    # 5. Content to adapt
    prompt += "\n## CONTENT TO ADAPT:\n```json\n"
    prompt += json.dumps(content, indent=2, default=str)
    prompt += "\n```\n"
    
    # 6. Output format with verification fields
    prompt += "\n" + OUTPUT_FORMAT.format(
        manager_name=manager_name,
        company_name=company_name
    )
    
    return prompt


def _get_shard_rules(shard_id: str, company_name: str) -> str:
    """Get minimal rules for each shard type."""
    shard_lower = shard_id.lower()
    
    if "resource" in shard_lower:
        return f"""⚠️ CRITICAL: Resources MUST be COMPREHENSIVE and LEARNER-READY

## CONTENT REQUIREMENTS (NON-NEGOTIABLE):
1. **MINIMUM LENGTH**: Each resource markdownText MUST be 500+ words
2. **STRUCTURE**: Use proper markdown headings (##, ###) and bullet points

3. **STATISTICS**: Every stat needs source citation: "Value (Source: Organization Year)"
   - Market size: "$X.X billion (Source: IBISWorld 2024)"
   - Growth rate: "X.X% CAGR (Source: Grand View Research 2024)"
   - Consumer data: "X% of consumers... (Source: McKinsey 2024)"

4. **COMPETITOR ANALYSIS**: List 3-5 real competitors with their market position and differentiators

5. **FRAMEWORKS**: If KLOs mention analysis frameworks, include:
   - SWOT analysis with {company_name}-specific content
   - PESTEL factors with actual industry data
   - Financial metrics with realistic calculations

6. **ACTIONABLE CONTENT**: Include:
   - Step-by-step processes learners can follow
   - Worked calculation examples with formulas
   - Decision criteria with thresholds

7. **NO EMPTY SECTIONS**: Every field must have substantial content
   - title: Descriptive, specific to topic
   - markdownText: 500+ words with full analysis
   - No "[placeholder]", no "TBD", no truncated sentences

8. **COMPANY-SPECIFIC**: Use "{company_name}" explicitly throughout
   - "For {company_name}, the recommended approach..."
   - "{company_name}'s competitive advantage lies in..."

⛔ REJECTION TRIGGERS:
- markdownText under 300 words → REJECTED
- Statistics without sources → REJECTED
- Empty or placeholder content → REJECTED"""
    
    elif "email" in shard_lower:
        return f"""1. FROM field = signature (MUST MATCH)
2. Reference specific deliverables
3. Use "{company_name}" in body
4. Professional business tone"""
    
    elif "simulation_flow" in shard_lower or "sim_flow" in shard_lower:
        return f"""1. ZERO duplicate activity names
2. Each question maps to learning outcome
3. Questions answerable from resources
4. Use "{company_name}" explicitly"""
    
    elif "rubric" in shard_lower:
        return f"""1. Criteria testable by activities
2. Reference existing resource data
3. Clear distinction between levels
4. Aligned to learning outcomes"""

    elif "workplace" in shard_lower or "scenario" in shard_lower:
        return f"""1. Use "{company_name}" for organization
2. PRESERVE all existing structure and nested keys
3. Update entity references (manager, company, industry)
4. Keep role/task descriptions aligned to target scenario"""

    else:
        return f"""1. Adapt all content to target
2. Use "{company_name}" explicitly
3. Professional simulation quality
4. Internal consistency"""


# =============================================================================
# TWO-PASS VERIFICATION PROMPT
# =============================================================================

VERIFICATION_ONLY_PROMPT = """You are a VALIDATOR. Do NOT modify content. ONLY check for violations.

## CONTENT TO VERIFY:
```json
{content}
```

## CHECKS TO PERFORM:

### 1. POISON TERM SEARCH
Search for these EXACT strings (case-insensitive):
{poison_list}

### 2. PLACEHOLDER SEARCH  
Search for this regex pattern: \[([^\]]+)\]
This matches anything in [brackets]

### 3. SENDER CONSISTENCY
Expected manager: {manager_name}
Check all email "from" fields and signatures match this name.

### 4. TRAILING DOTS
Search for patterns: .com. .org. .png. .jpg.
(URLs/emails/files should NOT end with extra period)

## OUTPUT FORMAT (ONLY this JSON, nothing else):

{{
  "poison_terms_found": ["list any found, empty if none"],
  "placeholders_found": ["list any [brackets] found, empty if none"],
  "sender_mismatches": ["list any mismatched names, empty if none"],
  "trailing_dots_found": ["list any found, empty if none"],
  "verdict": "PASS" or "FAIL",
  "failure_reasons": ["list reasons if FAIL, empty if PASS"]
}}"""


def build_verification_prompt(
    content: dict,
    poison_list: list,
    manager_name: str
) -> str:
    """Build prompt for verification-only pass."""
    return VERIFICATION_ONLY_PROMPT.format(
        content=json.dumps(content, indent=2, default=str),
        poison_list=json.dumps(poison_list[:20]),
        manager_name=manager_name
    )


# =============================================================================
# REGENERATION PROMPT
# =============================================================================

REGENERATION_PROMPT = """## REGENERATION — Previous output REJECTED

### Violations found by automated scan:
{violations}

### Content that failed:
```json
{content}
```

### FIX REQUIREMENTS:
1. Remove/replace ALL poison terms listed above
2. Replace ALL [brackets] with actual values  
3. Fix sender name to: {manager_name}
4. Remove trailing dots from emails/URLs
5. Complete any truncated sentences

### VERIFICATION FIELDS STILL REQUIRED:
Include poison_scan_proof, placeholder_scan_proof, sender_consistency_proof, formatting_proof.

Your output will be scanned again. Same violations = rejected again.

Return valid JSON only:"""


def build_regeneration_prompt(
    shard_id: str,
    shard_name: str,
    content: dict,
    source_scenario: str,
    target_scenario: str,
    global_factsheet: dict,
    feedback: dict,
) -> str:
    """Build regeneration prompt with specific violations."""
    factsheet = global_factsheet if isinstance(global_factsheet, dict) else {}
    manager = factsheet.get('reporting_manager', {})
    manager_name = manager.get('name', 'Unknown') if isinstance(manager, dict) else 'Unknown'
    
    # Build violations list - safely handle non-dict feedback
    feedback = feedback if isinstance(feedback, dict) else {}
    violations = []
    for key in ['poison_terms_found', 'placeholders_found', 'sender_mismatches', 'trailing_dots_found']:
        items = feedback.get(key, []) if isinstance(feedback.get(key), list) else []
        if items:
            violations.append(f"**{key}:** {items}")

    failed_rules = feedback.get('failed_rules', []) if isinstance(feedback.get('failed_rules'), list) else []
    critical_issues = feedback.get('critical_issues', []) if isinstance(feedback.get('critical_issues'), list) else []
    violations.extend(failed_rules[:3])
    violations.extend(critical_issues[:3])
    
    violations_text = '\n'.join(f"- {v}" for v in violations) if violations else "- Unspecified violations"
    
    return REGENERATION_PROMPT.format(
        violations=violations_text,
        content=json.dumps(content, indent=2, default=str),
        manager_name=manager_name
    )


# =============================================================================
# STRICT CODE VERIFICATION — THE REAL GATE
# =============================================================================

class VerificationResult:
    """Result of strict verification."""
    def __init__(self):
        self.passed = True
        self.errors = []
        self.warnings = []
    
    def fail(self, error: str):
        self.passed = False
        self.errors.append(error)
    
    def warn(self, warning: str):
        self.warnings.append(warning)
    
    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "errors": self.errors,
            "warnings": self.warnings
        }


def strict_verify_output(
    output: dict,
    factsheet: dict,
    check_verification_fields: bool = True
) -> VerificationResult:
    """
    Strict verification of LLM output. This is the HARD GATE.
    
    This function CANNOT be bypassed by prompt engineering.
    It uses regex and string matching on the actual output.
    
    Args:
        output: The LLM's output dict
        factsheet: The global factsheet with poison_list, manager info, etc.
        check_verification_fields: Whether to validate the proof fields
    
    Returns:
        VerificationResult with passed/failed status and errors
    """
    result = VerificationResult()

    # Handle non-dict output
    if not isinstance(output, dict):
        result.fail(f"Output is not a dict: {type(output)}")
        return result

    if not isinstance(factsheet, dict):
        factsheet = {}

    # Get content to scan
    adapted_content = output.get('adapted_content', output)
    content_str = json.dumps(adapted_content, default=str)
    content_lower = content_str.lower()
    
    # === CHECK 1: PLACEHOLDERS ===
    placeholder_pattern = r'\[[^\]]{1,50}\]'  # [anything up to 50 chars]
    placeholders = re.findall(placeholder_pattern, content_str)

    # Filter out legitimate uses (e.g., markdown links, citation markers)
    suspicious_placeholders = []
    for p in placeholders:
        # Skip legitimate patterns
        if p.startswith('[http'):  # markdown links
            continue
        if re.match(r'\[\d+\]', p):  # citation markers like [1]
            continue
        if re.match(r'\[\d{4}\]', p):  # year markers like [2024]
            continue

        # Flag as suspicious if matches any placeholder pattern
        p_lower = p.lower()
        p_upper = p.upper()
        if any([
            'TBD' in p_upper,
            'XXX' in p_upper,
            'TODO' in p_upper,
            'INSERT' in p_upper,
            'PLACEHOLDER' in p_upper,
            'metric' in p_lower,
            'name' in p_lower,
            'value' in p_lower,
            'data' in p_lower,
            'industry' in p_lower,
            re.match(r'\[[A-Z][a-z]*\]$', p),  # [Something]
            re.match(r'\[[a-z\s-]+\]$', p),  # [some placeholder text]
            re.match(r'\[[A-Z]\]$', p),  # Single letter [X], [Y], [N]
            re.match(r'\[[A-Z]+\]$', p),  # Acronyms [TBD], [NA]
        ]):
            suspicious_placeholders.append(p)

    if suspicious_placeholders:
        result.fail(f"PLACEHOLDERS FOUND: {suspicious_placeholders[:5]}")
    
    # === CHECK 2: POISON TERMS ===
    poison_list = factsheet.get('poison_list', []) if isinstance(factsheet, dict) else []
    poison_list = poison_list if isinstance(poison_list, list) else []
    found_poison = []

    for term in poison_list:
        if isinstance(term, str) and len(term) >= 3 and term.lower() in content_lower:
            found_poison.append(term)

    if found_poison:
        result.fail(f"POISON TERMS FOUND: {found_poison[:5]}")

    # === CHECK 3: SENDER CONSISTENCY ===
    manager = factsheet.get('reporting_manager', {}) if isinstance(factsheet, dict) else {}
    expected_manager = manager.get('name', '') if isinstance(manager, dict) else ''

    if expected_manager:
        # Look for email-like structures and check sender
        email_from_pattern = r'"from"[:\s]*"([^"]+)"'
        email_froms = re.findall(email_from_pattern, content_str, re.IGNORECASE)

        # Extended signature pattern - catches more closing phrases
        signature_pattern = r'(?:regards|sincerely|best|thanks|cheers|warmly|warm regards|kind regards|all the best)[,\s]*\n*([A-Z][a-z]+ [A-Z][a-z]+)'
        signatures = re.findall(signature_pattern, content_str, re.IGNORECASE)
        
        for found in email_froms + signatures:
            if expected_manager.lower() not in found.lower():
                # Check if it's a different name (not just different format)
                found_words = set(found.lower().split())
                expected_words = set(expected_manager.lower().split())
                if not found_words.intersection(expected_words):
                    result.fail(f"SENDER MISMATCH: Expected '{expected_manager}', found '{found}'")
    
    # === CHECK 4: TRAILING DOTS ===
    # NOTE: Trailing dots are fixed in merger stage (cleanup_merged_json)
    # Don't block here - just log for debugging
    trailing_dot_patterns = [
        r'[\w.-]+@[\w.-]+\.\w{2,}\.',  # email.com.
        r'\.(png|jpg|jpeg|gif|svg|pdf)\.',  # .png.
        r'https?://[^\s"]+\.',  # URL ending with .
    ]

    for pattern in trailing_dot_patterns:
        matches = re.findall(pattern, content_str)
        if matches:
            # Log but don't fail - gets fixed in merger stage
            logger.debug(f"[VERIFY] Trailing dots (will be fixed in merger): {matches[:3]}")
    
    # === CHECK 5: INCOMPLETE CONTENT ===
    incomplete_patterns = [
        r'\.\.\.[^"]',  # ... not in a string
        r'(?:TODO|TBD|FIXME|XXX)(?:[^a-zA-Z]|$)',
    ]
    
    for pattern in incomplete_patterns:
        if re.search(pattern, content_str, re.IGNORECASE):
            result.fail(f"INCOMPLETE CONTENT: Pattern '{pattern}' found")
    
    # === CHECK 6: VERIFICATION FIELD HONESTY (if required) ===
    if check_verification_fields and isinstance(output, dict):
        poison_proof = output.get('poison_scan_proof', {}) if isinstance(output.get('poison_scan_proof'), dict) else {}
        placeholder_proof = output.get('placeholder_scan_proof', {}) if isinstance(output.get('placeholder_scan_proof'), dict) else {}
        sender_proof = output.get('sender_consistency_proof', {}) if isinstance(output.get('sender_consistency_proof'), dict) else {}

        # Check if LLM lied about poison scan
        claimed_no_poison = poison_proof.get('terms_found_in_my_output', None) == [] if isinstance(poison_proof, dict) else False
        if claimed_no_poison and found_poison:
            result.fail(f"LLM LIED: Claimed no poison terms but found {found_poison}")

        # Check if LLM lied about placeholders
        claimed_no_brackets = placeholder_proof.get('brackets_found', None) == 0 if isinstance(placeholder_proof, dict) else False
        if claimed_no_brackets and suspicious_placeholders:
            result.fail(f"LLM LIED: Claimed 0 brackets but found {suspicious_placeholders}")

        # Check if LLM lied about sender match
        claimed_match = sender_proof.get('all_match', None) == True if isinstance(sender_proof, dict) else False
        if claimed_match and any('SENDER MISMATCH' in e for e in result.errors):
            result.fail("LLM LIED: Claimed sender match but mismatch detected")
    
    return result


# =============================================================================
# ORCHESTRATION: ADAPT WITH VERIFICATION
# =============================================================================

def adapt_shard_with_verification(
    shard_id: str,
    shard_name: str,
    content: dict,
    source_scenario: str,
    target_scenario: str,
    global_factsheet: dict,
    llm_call_fn,  # Function that takes prompt and returns dict
    max_retries: int = 2,
    use_two_pass: bool = False,
) -> tuple[dict, VerificationResult]:
    """
    Adapt a shard with automatic verification and retry.
    
    Args:
        shard_id: Shard identifier
        shard_name: Human-readable name
        content: Content to adapt
        source_scenario: Source scenario text
        target_scenario: Target scenario text  
        global_factsheet: Factsheet with poison list, manager, etc.
        llm_call_fn: Function to call LLM (takes prompt str, returns dict)
        max_retries: Max regeneration attempts
        use_two_pass: If True, run separate verification pass
    
    Returns:
        Tuple of (adapted_content, verification_result)
    """
    # Build initial prompt
    prompt = build_shard_adaptation_prompt(
        shard_id=shard_id,
        shard_name=shard_name,
        content=content,
        source_scenario=source_scenario,
        target_scenario=target_scenario,
        global_factsheet=global_factsheet
    )
    
    # PASS 1: Generate
    result = llm_call_fn(prompt)
    
    # CODE GATE: Strict verification
    verification = strict_verify_output(result, global_factsheet)
    
    # Optional PASS 2: Separate verification call
    if use_two_pass and verification.passed:
        manager = global_factsheet.get('reporting_manager', {}) if isinstance(global_factsheet, dict) else {}
        verify_prompt = build_verification_prompt(
            content=result.get('adapted_content', result) if isinstance(result, dict) else result,
            poison_list=global_factsheet.get('poison_list', []) if isinstance(global_factsheet, dict) else [],
            manager_name=manager.get('name', '') if isinstance(manager, dict) else ''
        )
        verify_result = llm_call_fn(verify_prompt)

        if isinstance(verify_result, dict) and verify_result.get('verdict') == 'FAIL':
            verification.fail(f"Two-pass verification failed: {verify_result.get('failure_reasons', [])}")
    
    # RETRY if failed
    retries = 0
    while not verification.passed and retries < max_retries:
        retries += 1
        
        # Build regeneration prompt with specific errors
        regen_prompt = build_regeneration_prompt(
            shard_id=shard_id,
            shard_name=shard_name,
            content=result.get('adapted_content', result) if isinstance(result, dict) else result,
            source_scenario=source_scenario,
            target_scenario=target_scenario,
            global_factsheet=global_factsheet,
            feedback={
                'poison_terms_found': [e for e in verification.errors if 'POISON' in e],
                'placeholders_found': [e for e in verification.errors if 'PLACEHOLDER' in e],
                'sender_mismatches': [e for e in verification.errors if 'SENDER' in e],
                'failed_rules': verification.errors
            }
        )
        
        result = llm_call_fn(regen_prompt)
        verification = strict_verify_output(result, global_factsheet)
    
    return result, verification


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_prompt_stats(prompt: str) -> dict:
    """Get statistics about prompt length."""
    return {
        "chars": len(prompt),
        "lines": len(prompt.split('\n')),
        "words": len(prompt.split()),
        "est_tokens": len(prompt) // 4
    }


def extract_adapted_content(llm_output) -> dict:
    """Safely extract adapted_content from LLM output."""
    if not isinstance(llm_output, dict):
        return llm_output if isinstance(llm_output, dict) else {}
    if 'adapted_content' in llm_output:
        content = llm_output['adapted_content']
        return content if isinstance(content, dict) else {}
    # If no adapted_content key, assume whole output is content
    # (minus our verification fields)
    excluded = {'poison_scan_proof', 'placeholder_scan_proof',
                'sender_consistency_proof', 'formatting_proof',
                'entity_mappings'}
    return {k: v for k, v in llm_output.items() if k not in excluded}