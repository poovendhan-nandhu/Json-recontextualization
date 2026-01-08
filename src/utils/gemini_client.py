"""
Gemini Client for parallel shard adaptation.

Uses Gemini 2.5 Flash with:
- Global Factsheet for consistency
- Statistics tracking
- Retry with exponential backoff
- LangSmith tracing (optional)

⚠️ PROMPT ENGINEERING: Review adapt_shard_content() for tuning.
"""
import os
import json
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .llm_stats import get_stats, StatsTimer
from .retry_handler import retry_with_backoff, RetryConfig

logger = logging.getLogger(__name__)


def sanitize_for_json(obj: Any) -> Any:
    """
    Recursively sanitize an object for JSON encoding.
    Removes surrogate characters that cause UTF-8 encoding errors.
    """
    if isinstance(obj, str):
        # Remove surrogate characters (\uD800-\uDFFF)
        return obj.encode('utf-8', errors='surrogateescape').decode('utf-8', errors='replace')
    elif isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
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

    # Remove any leading/trailing whitespace
    text = text.strip()

    # Remove markdown code blocks
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1])
        else:
            text = "\n".join(lines[1:])

    # Fix invalid escape sequences (e.g., \n in middle of string that should be \\n)
    # Replace invalid escapes with their escaped versions
    def fix_escapes(s):
        # Fix common invalid escapes
        s = re.sub(r'\\([^"\\/bfnrtu])', r'\\\\\1', s)
        return s

    text = fix_escapes(text)

    # Try standard parsing first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Use json-repair library for robust fixing
    try:
        from json_repair import repair_json
        repaired = repair_json(text, return_objects=True)
        logger.info("JSON repaired successfully using json-repair")
        return repaired
    except Exception as e:
        logger.warning(f"json-repair failed: {e}, trying manual fix")

    # Manual fallback: Remove trailing commas
    text = re.sub(r',(\s*[}\]])', r'\1', text)

    # Ensure proper closing brackets
    open_braces = text.count('{') - text.count('}')
    open_brackets = text.count('[') - text.count(']')
    if open_braces > 0:
        text += '}' * open_braces
    if open_brackets > 0:
        text += ']' * open_brackets

    return json.loads(text)


def _call_gemini_sync(prompt: str, temperature: float = 0.3) -> dict:
    """Synchronous Gemini call."""
    model = _get_gemini()

    response = model.generate_content(
        prompt,
        generation_config={
            "temperature": temperature,  # Use passed temperature (was hardcoded to 0.1!)
            "max_output_tokens": 65536,  # Max tokens to prevent truncation
            "response_mime_type": "application/json",
        }
    )

    text = response.text.strip()

    try:
        # Use robust JSON repair
        return _repair_json(text)
    except Exception as e:
        logger.error(f"JSON parse error: {e}")
        logger.error(f"Raw response (first 500 chars): {text[:500]}")
        raise


async def call_gemini(prompt: str, temperature: float = 0.3) -> dict:
    """Async Gemini call with retry."""
    loop = asyncio.get_event_loop()

    async def _call():
        return await loop.run_in_executor(_executor, _call_gemini_sync, prompt, temperature)

    # Retry config - fewer attempts to fail fast
    config = RetryConfig(
        max_attempts=2,  # Reduced from 5 - fail fast
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

    This ensures ALL shards use consistent values for:
    - Company name
    - Revenue/financials
    - Products/services
    - Key metrics
    - Poison list (terms to avoid)

    Args:
        source_scenario: Original scenario text
        target_scenario: Target scenario text

    Returns:
        Global factsheet dict
    """
    # Sanitize inputs to prevent UTF-8 encoding errors
    safe_source = safe_str(source_scenario)
    safe_target = safe_str(target_scenario)

    prompt = f"""Analyze these two business scenarios and extract key facts for the TARGET scenario.

## SOURCE SCENARIO (current):
{safe_source}

## TARGET SCENARIO (new):
{safe_target}

## TASK:
Extract a comprehensive factsheet for the TARGET scenario. This will ensure CONSISTENCY and ALIGNMENT across all simulation content.

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
    "kpis": ["List 10-15 KEY PERFORMANCE INDICATORS specific to this industry", "e.g. for beverage: market share, distribution coverage, trial rate, repeat purchase rate, shelf space, brand awareness"],
    "terminology": ["List 15-20 industry-specific TERMS to use", "e.g. for beverage: consumer, retailer, shelf space, SKU, formulation, distribution channel"],
    "wrong_terms": ["List terms that should NOT be used for this industry", "e.g. for beverage: CAC, MRR, churn, subscription, activation rate, SaaS"],
    "data_types": ["List 5-10 types of DATA/REPORTS that would be realistic", "e.g. for beverage: market research report, consumer survey, retail audit, brand tracking study"]
  }},
  "alignment_guidance": {{
    "klo_themes": ["List 5-7 KEY LEARNING OUTCOMES themes for this scenario", "e.g. market analysis, competitive positioning, distribution strategy, pricing analysis"],
    "question_types": ["List 5-7 types of QUESTIONS that test these KLOs", "e.g. analyze market share data, evaluate distribution options, recommend pricing strategy"],
    "resource_requirements": ["List 5-7 types of DATA resources need to contain", "e.g. market size figures, competitor data, consumer demographics, cost structure"]
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
    "List 10-15 SPECIFIC facts/numbers that can be cited in resources",
    "e.g. 'Market size: $45 billion', 'Growth rate: 12% CAGR', 'Target demographic: 25-45 year olds'"
  ],
  "shard_rules": {{
    "resources": [
      "MUST: Every statistic needs a source citation (e.g., 'Market size: $X billion - Source: McKinsey 2024')",
      "MUST: Include a SCORING MODEL TEMPLATE for KLO3 (weighted criteria, 1-5 scales, calculation example)",
      "MUST: Include a QUESTION DESIGN GUIDE for KLO2 (how to write unbiased questions, validity checks)",
      "MUST: Include industry-specific metrics with realistic values (gross margin %, CAC, return rates, AOV)",
      "MUST: Include competitor data table with at least 3 named competitors and their metrics",
      "Generate 3-5 additional SPECIFIC data requirements for THIS scenario"
    ],
    "rubrics": [
      "MUST: Each KLO must be directly testable by at least one question in simulation_flow",
      "MUST: KLO2 should explicitly require learners to CREATE analysis questions and defend their validity",
      "MUST: KLO3 should explicitly require learners to BUILD a scoring model with weights",
      "MUST: Criteria must reference specific data that EXISTS in resources",
      "Generate 2-3 additional SPECIFIC rubric rules for THIS scenario"
    ],
    "simulation_flow": [
      "MUST: NO DUPLICATE questions or activities - each must have a UNIQUE name",
      "MUST: Each question must explicitly map to a specific KLO",
      "MUST: Questions must be answerable using ONLY the data in resources",
      "MUST: Include activity for KLO2 where learner writes their own analysis questions",
      "MUST: Include activity for KLO3 where learner builds a scoring/weighting model",
      "Generate 2-3 additional SPECIFIC flow rules for THIS scenario"
    ],
    "emails": [
      "MUST: Use exact manager name and email from factsheet",
      "MUST: Reference specific deliverables that match the KLOs",
      "MUST: Mention the company name explicitly",
      "Generate 2-3 additional SPECIFIC email rules for THIS scenario"
    ],
    "workplace_scenario": [
      "MUST: Use exact company name from factsheet throughout",
      "MUST: Include specific industry metrics (market size, growth rate) with sources",
      "MUST: Define learner role clearly with 3-5 specific responsibilities",
      "Generate 2-3 additional SPECIFIC scenario rules for THIS scenario"
    ]
  }},
  "required_templates": {{
    "scoring_model_template": "Generate a COMPLETE scoring model template with: criteria names, weight percentages, 1-5 scale definitions, and a worked calculation example",
    "question_design_guide": "Generate a COMPLETE question design guide with: how to write leading questions, probing questions, bias checks, validity/reliability criteria",
    "industry_metrics": [
      "List 10+ industry-specific metrics with REALISTIC benchmark values for this industry",
      "e.g., 'Gross margin: 50-60% for DTC apparel'",
      "e.g., 'Customer acquisition cost: $25-50 for sustainable fashion brands'",
      "e.g., 'Return rate: 15-25% for online apparel'"
    ]
  }}
}}

Return ONLY valid JSON:"""

    with StatsTimer("global_factsheet") as timer:
        try:
            result = await call_gemini(prompt, temperature=0.2)
            get_stats().add_call(
                success=True,
                shard_id="global_factsheet",
                elapsed_time=timer.elapsed
            )
            # Ensure result is a dict (LLM might return list)
            if isinstance(result, list) and len(result) > 0:
                result = result[0] if isinstance(result[0], dict) else {}
            if not isinstance(result, dict):
                logger.warning(f"Factsheet returned unexpected type: {type(result)}")
                result = {}
            poison_list = result.get('poison_list', [])
            poison_count = len(poison_list) if isinstance(poison_list, list) else 0
            logger.info(f"Extracted global factsheet with {poison_count} poison terms")
            return result
        except Exception as e:
            get_stats().add_call(
                success=False,
                shard_id="global_factsheet",
                elapsed_time=timer.elapsed
            )
            logger.error(f"Failed to extract factsheet: {e}")
            # Return minimal factsheet on failure
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
) -> tuple[dict, dict]:
    """
    Adapt a single shard's content using the global factsheet.

    Args:
        shard_id: Shard identifier
        shard_name: Human-readable name
        content: Shard content to transform
        source_scenario: Original scenario text
        target_scenario: Target scenario text
        global_factsheet: Pre-extracted global facts (CRITICAL for consistency)
        rag_context: Additional context from RAG

    Returns:
        (adapted_content, entity_mappings)
    """
    # Build factsheet section for prompt
    factsheet_text = ""
    poison_list_text = ""

    if global_factsheet and isinstance(global_factsheet, dict):
        # Extract nested values safely - ensure each is a dict
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
- Revenue: {financials.get('revenue', 'Not specified') if isinstance(financials, dict) else 'Not specified'}
- Key Metric: {financials.get('key_metric', 'Not specified') if isinstance(financials, dict) else 'Not specified'}

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
        if poison_list:
            poison_list_text = f"""
## POISON LIST (DO NOT use these terms - they are from the OLD scenario):
{safe_json_dumps(poison_list, indent=2)}
Replace any of these terms with appropriate TARGET scenario equivalents.
"""

    # ==========================================================================
    # SHARD-SPECIFIC RULES (DYNAMIC - from factsheet, not hardcoded!)
    # ==========================================================================
    def get_dynamic_shard_rules(shard_id: str, factsheet: dict, company_name: str, industry: str) -> str:
        """
        Get shard-specific rules from factsheet (DYNAMIC, not hardcoded).
        Falls back to minimal rules if factsheet doesn't have them.
        """
        shard_lower = shard_id.lower()
        shard_rules_dict = factsheet.get('shard_rules', {}) if factsheet else {}
        required_templates = factsheet.get('required_templates', {}) if factsheet else {}
        industry_metrics = factsheet.get('industry_metrics', []) if factsheet else []

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
            # Keep MUST rules, filter out example/placeholder text
            actual_rules = [r for r in rules if isinstance(r, str) and (r.startswith("MUST:") or (not r.startswith("e.g.") and not r.startswith("Generate") and not r.startswith("Focus on")))]
            if actual_rules:
                rules_text = "\n".join([f"{i+1}. {rule}" for i, rule in enumerate(actual_rules)])

        # Add templates for resources shard - MUCH STRONGER
        templates_text = ""
        if "resource" in shard_lower:
            templates_text = f"""

## ⛔ MANDATORY RESOURCE REQUIREMENTS (FAILURE IF MISSING):

### 1. COMPETITOR TABLE (REQUIRED - NOT OPTIONAL):
You MUST include a competitor analysis table with REAL data:
```
| Competitor | Market Share | Price Range | Channels | Target Demo | Source |
|------------|--------------|-------------|----------|-------------|--------|
| [Name 1]   | X%           | $XX-$XX     | DTC/Retail| Demographics| [Real Source] |
| [Name 2]   | X%           | $XX-$XX     | Channels  | Demographics| [Real Source] |
| [Name 3]   | X%           | $XX-$XX     | Channels  | Demographics| [Real Source] |
```
❌ DO NOT just mention "3-5 competitors" - GIVE THE TABLE

### 2. FINANCIAL MODEL TEMPLATE (REQUIRED):
You MUST include unit economics with example numbers:
```
FINANCIAL FRAMEWORK FOR {company_name}:
- TAM: $XXB (Total Addressable Market) - Source: [Real Source]
- SAM: $XXB (Serviceable Available Market)
- SOM: $XXM (Serviceable Obtainable Market - Year 1 target)

UNIT ECONOMICS:
- Average Order Value (AOV): $XX
- Customer Acquisition Cost (CAC): $XX
- Customer Lifetime Value (LTV): $XXX
- LTV:CAC Ratio: X:1
- Gross Margin: XX%
- Return Rate: XX%
```
❌ DO NOT just say "consider financial viability" - GIVE THE TEMPLATE

### 3. MARKET DATA WITH SOURCES (REQUIRED):
Every statistic needs format: "Value (Source: Organization Year)"
- ✅ "Market size: $45B (Source: McKinsey State of Fashion 2024)"
- ❌ "The market is growing at 15% CAGR" (NO SOURCE = BLOCKER)

### 4. SCORING RUBRIC TEMPLATE (if KLO mentions evaluation/scoring):
```
EVALUATION CRITERIA:
| Criterion | Weight | 1 (Poor) | 3 (Average) | 5 (Excellent) |
|-----------|--------|----------|-------------|---------------|
| [Crit 1]  | 30%    | [Desc]   | [Desc]      | [Desc]        |
| [Crit 2]  | 25%    | [Desc]   | [Desc]      | [Desc]        |
```
"""

        # Add anti-duplicate rules for simulation_flow
        duplicate_rules = ""
        if "simulation_flow" in shard_lower or "sim_flow" in shard_lower:
            duplicate_rules = f"""

## ⛔ ANTI-DUPLICATE RULES (INSTANT FAILURE):
**BEFORE RETURNING, SCAN ALL ACTIVITIES AND QUESTIONS:**

1. List all activity names you're outputting
2. Check: Are ANY names identical or near-identical?
3. If YES → RENAME one to be distinct

❌ DUPLICATES TO AVOID:
- "Develop Market Entry Framework" appearing twice
- "Analyze Competitive Landscape" appearing twice
- Any activity name appearing more than once

✅ EACH ACTIVITY MUST BE UNIQUE:
- Activity 1: "Identify Critical Market Factors"
- Activity 2: "Analyze Competitive Landscape"
- Activity 3: "Develop Evaluation Framework"
- Activity 4: "Formulate Go/No-Go Recommendation"

**IF YOU CREATE DUPLICATES, THE ENTIRE OUTPUT FAILS**
"""

        if rules_text:
            return f"""
## 🎯 SHARD-SPECIFIC RULES FOR {shard_type} (Generated for this scenario):
{rules_text}
- Use company name "{company_name}" throughout
- Use {industry}-appropriate terminology
{templates_text}
{duplicate_rules}
"""

        # Fallback: minimal dynamic rules if factsheet doesn't have specific ones
        return f"""
## 🎯 SHARD RULES FOR {shard_type}:
- Use company name "{company_name}" explicitly throughout
- Use {industry}-appropriate terminology and metrics
- Ensure content is specific to the scenario (not generic)
- All data should be citable with sources
{templates_text}
{duplicate_rules}
"""

    # Sanitize all string inputs to prevent UTF-8 encoding errors
    safe_source = safe_str(source_scenario)[:200]
    safe_target = safe_str(target_scenario)[:200]
    safe_rag = safe_str(rag_context) if rag_context else "No additional context available."

    # Extract KLO and industry context for alignment prompts
    industry_context = global_factsheet.get('industry_context', {}) if global_factsheet else {}
    industry_kpis = industry_context.get('kpis', []) if isinstance(industry_context, dict) else []
    industry_terms = industry_context.get('terminology', []) if isinstance(industry_context, dict) else []

    # Get shard-specific rules from FACTSHEET (dynamic!)
    shard_rules = get_dynamic_shard_rules(
        shard_id,
        global_factsheet,
        company.get('name', 'the company'),
        company.get('industry', 'business')
    )

    prompt = f"""You are a High-Fidelity Simulation Adapter for UG (undergraduate) business education.
Your job is to rewrite a specific component (Shard) of a business simulation to fit a NEW context.

## 1. SCENARIO TRANSITION:
**FROM:** {safe_source}...
**TO:** {safe_target}...
{factsheet_text}
{poison_list_text}

## 2. SHARD INFO:
- Shard ID: {shard_id}
- Shard Name: {shard_name}

{shard_rules}

## 3. ADDITIONAL CONTEXT FROM RAG:
{safe_rag}

## 4. INDUSTRY-SPECIFIC TERMS (USE THESE):
- KPIs for this industry: {', '.join(industry_kpis[:10]) if industry_kpis else 'market share, revenue growth, customer satisfaction'}
- Terminology: {', '.join(industry_terms[:10]) if industry_terms else 'Use industry-appropriate terms'}

## 5. CONTENT TO ADAPT:
```json
{safe_json_dumps(content, indent=2)}
```

## 6. ⛔ CRITICAL ALIGNMENT RULES - VIOLATIONS = AUTOMATIC FAILURE ⛔

### ❌ BLOCKER A: NO DUPLICATES (INSTANT FAIL)
**BEFORE OUTPUTTING, CHECK:** Are there ANY duplicate activity names or questions?
- SCAN all activities - EACH must have UNIQUE name
- SCAN all questions - NO verbatim duplicates
- ❌ WRONG: Two activities named "Develop Market Entry Framework"
- ✅ RIGHT: "Develop Market Entry Framework" + "Create Evaluation Scoring Model"
- **IF YOU CREATE DUPLICATES, THE ENTIRE OUTPUT FAILS VALIDATION**

### ❌ BLOCKER B: NO FAKE/GENERIC SOURCES (INSTANT FAIL)
**EVERY statistic MUST have a REAL, VERIFIABLE source:**
- ❌ WRONG: "15% CAGR" (no source)
- ❌ WRONG: "Retail Analytics Inc." (fake company)
- ❌ WRONG: "Industry reports show..." (vague)
- ✅ RIGHT: "Market size: $45B (Source: McKinsey State of Fashion 2024)"
- ✅ RIGHT: "Gen Z spending: 40% of apparel purchases (Source: NielsenIQ 2024)"
- ✅ RIGHT: "CAC: $35 (Source: Company Internal Q3 2024 Report)"
**ACCEPTABLE SOURCES ONLY:** McKinsey, Bain, BCG, Deloitte, NielsenIQ, Euromonitor, Statista, IBISWorld, Company Internal Reports, SEC Filings, Trade Associations

### ❌ BLOCKER C: NO TRUNCATION (INSTANT FAIL)
**ALL text MUST be COMPLETE - never cut off mid-sentence:**
- ❌ WRONG: "such as targ..." or "previously analyze..."
- ❌ WRONG: "Includes labeled sc..."
- ✅ RIGHT: Complete sentences with proper endings
- **SCAN YOUR OUTPUT** - if ANY text is cut off, FIX IT before returning

### ❌ BLOCKER D: COMPETITOR DATA MUST BE SPECIFIC (INSTANT FAIL)
**When mentioning competitors, include ALL of:**
1. Company name (real or realistic for scenario)
2. Market share % with source
3. Price range
4. Distribution channels
5. Target demographic
- ❌ WRONG: "3-5 competitors exist in the market"
- ❌ WRONG: "High-level competitive landscape"
- ✅ RIGHT: "Competitor Analysis:
   | Company | Market Share | Price Range | Channels | Target |
   | Patagonia | 12% | $50-120 | DTC, Retail | Eco-conscious 25-45 |
   | Everlane | 8% | $40-90 | DTC Only | Urban millennials |
   | Reformation | 6% | $60-150 | DTC, Select Retail | Fashion-forward Gen Z |
   (Source: Euromonitor Sustainable Apparel 2024)"

### ❌ BLOCKER E: FINANCIAL MODEL REQUIRED (INSTANT FAIL FOR RESOURCES SHARD)
**If this is a RESOURCES shard, MUST include financial model template:**
- TAM/SAM/SOM breakdown with calculations
- Unit economics: CAC, LTV, Gross Margin, AOV
- Break-even analysis framework
- ROI projection template
- ❌ WRONG: "Consider financial viability"
- ✅ RIGHT: Include actual template with example numbers

### ⚠️ BLOCKER F: COMPANY NAME EXPLICIT (BLOCKER IF MISSING)
- ALWAYS use EXACT company name: "{company.get('name', 'the company') if global_factsheet else 'the company'}"
- ❌ WRONG: "our brand", "the company", "this organization"
- ✅ RIGHT: "{company.get('name', 'the company') if global_factsheet else 'the company'}'s market analysis..."

### ⚠️ BLOCKER G: WRONG INDUSTRY TERMS
{f"FORBIDDEN TERMS for {company.get('industry', 'this industry') if global_factsheet else 'this industry'}: {', '.join(industry_context.get('wrong_terms', [])[:15]) if industry_context.get('wrong_terms') else 'CAC, MRR, churn, ARR, SaaS (unless applicable)'}" if global_factsheet else "Avoid tech/SaaS terms in non-tech industries"}
- Replace with industry-appropriate equivalents

### ⚠️ KLO ↔ RESOURCE ALIGNMENT:
- EVERY KLO must have supporting data in resources WITH SOURCES
- If KLO says "develop scoring model" → Resources MUST have: template, weights, example
- If KLO says "analyze competitors" → Resources MUST have: competitor table with specifics

### ⚠️ KLO ↔ QUESTION ALIGNMENT:
- EVERY question must map to specific KLOs
- Use SAME terminology in questions as in KLOs

## 7. 🔍 PRE-OUTPUT VALIDATION CHECKLIST (RUN THIS MENTALLY):
□ Are there ANY duplicate activity names? → FIX THEM
□ Are there ANY truncated sentences? → COMPLETE THEM
□ Does EVERY statistic have a real source? → ADD SOURCES
□ Is competitor data specific (name, share, price, channels)? → ADD DETAILS
□ Is company name used explicitly (not "the company")? → FIX IT
□ Are resources complete (not just titles/links)? → ADD CONTENT

## 8. STANDARD RULES:
1. **JSON Integrity:** Keep same keys and nesting structure. For CONTENT fields (text, descriptions, markdownText), you CAN and SHOULD expand/enrich the content - add competitor tables, financial models, detailed data. Only preserve the JSON KEYS, not content length.
2. **ID Preservation:** ANY field containing "id" or "Id" must NOT change
3. **Data Consistency:** Use values from GLOBAL FACTSHEET above. Do NOT invent conflicting numbers.
4. **Poison Avoidance:** Do NOT use any term from the POISON LIST. Replace with target equivalents.
5. **Pedagogical Logic:**
   - Questions should NOT reveal answers
   - Resources provide raw data (inputs), not calculated conclusions
6. **Professional Tone:** Maintain business simulation quality
7. **⚠️ ADAPT ALL NESTED CONTENT - DO NOT SKIP ANY:**
   - ALL person names (users, managers, senders, recipients) → use factsheet names
   - ALL email bodies (taskEmail, secondaryTaskEmail, introEmail) → rewrite completely
   - ALL chat activities (activityData.selectedValue.users) → use factsheet manager
   - ALL organization names → use target company name
   - ALL job titles/designations → use target-appropriate titles
   - SCAN EVERY LEVEL of nesting - children, data, activityData, selectedValue, etc.
8. **⚠️ NO TRUNCATION - COMPLETE ALL CONTENT:**
   - NEVER cut off text mid-sentence
   - ALL descriptions, scenarios, challenges must be COMPLETE sentences
   - If original content is long, keep it long - do not shorten
   - Every field must end properly (with period, closing quote, etc.)
9. **NO DUPLICATES:**
   - Do not create duplicate activities or questions
   - Each activity should have a unique name and description
10. **NO TRAILING PERIODS on names/emails/roles:**
   - Names like "Sarah Johnson" NOT "Sarah Johnson."
   - Emails like "sarah@company.com" NOT "sarah@company.com."
   - Roles like "Director of Marketing" NOT "Director of Marketing."

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

    with StatsTimer(shard_id) as timer:
        try:
            result = await call_gemini(prompt, temperature=0.3)

            # Ensure result is a dict (LLM might return list)
            if isinstance(result, list) and len(result) > 0:
                result = result[0] if isinstance(result[0], dict) else {}
            if not isinstance(result, dict):
                logger.warning(f"Shard {shard_id} returned unexpected type: {type(result)}")
                return content, {}  # Return original content on type error

            adapted_content = result.get("adapted_content", content)
            entity_mappings = result.get("entity_mappings", {})
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
# REGENERATION WITH FEEDBACK (from planner_generator.py pattern)
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

    This is the KEY FEATURE from planner_generator.py:
    - Takes failed rule feedback
    - Passes it to LLM for smarter regeneration
    - Focuses on fixing specific issues

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
    # Build feedback section for prompt
    failed_rules = feedback.get("failed_rules", [])
    critical_issues = feedback.get("critical_issues", [])
    suggestions = feedback.get("suggestions", [])

    feedback_text = f"""
## ⚠️ REGENERATION MODE - FIX THESE ISSUES:

### Failed Validation Rules:
{json.dumps(failed_rules[:5], indent=2) if failed_rules else "None"}

### Critical Issues to Address:
{json.dumps(critical_issues[:5], indent=2) if critical_issues else "None"}

### Suggestions for Improvement:
{json.dumps(suggestions[:5], indent=2) if suggestions else "None"}

YOU MUST FIX ALL ISSUES ABOVE. This is a regeneration attempt after the first pass failed validation.
"""

    # Extract factsheet values
    company = global_factsheet.get('company', {}) if global_factsheet else {}
    company = company if isinstance(company, dict) else {}
    industry_context = global_factsheet.get('industry_context', {}) if global_factsheet else {}
    industry_context = industry_context if isinstance(industry_context, dict) else {}

    prompt = f"""You are REGENERATING simulation content that FAILED validation.

## SCENARIO TRANSITION:
**FROM:** {safe_str(source_scenario)[:200]}...
**TO:** {safe_str(target_scenario)[:200]}...

## COMPANY (use EXACT name):
{company.get('name', 'Unknown')} - {company.get('industry', 'Unknown')}

{feedback_text}

## SHARD TO FIX:
- Shard ID: {shard_id}
- Shard Name: {shard_name}

## CURRENT CONTENT (needs fixing):
```json
{safe_json_dumps(content, indent=2)}
```

## FIX REQUIREMENTS:
1. Address ALL failed rules listed above
2. Fix ALL critical issues
3. Apply suggestions for improvement
4. Use EXACT company name: {company.get('name', 'Unknown')}
5. Use industry-appropriate terminology for {company.get('industry', 'Unknown')}
6. Remove any wrong terms: {', '.join(industry_context.get('wrong_terms', [])[:10]) if industry_context.get('wrong_terms') else 'N/A'}
7. Ensure KLOs are supported by resources
8. Ensure questions align to KLOs
9. NO duplicate activities
10. NO trailing periods on names/emails

## OUTPUT FORMAT:
Return valid JSON:
{{
  "adapted_content": {{ ... the FIXED content ... }},
  "entity_mappings": {{ "old": "new" }},
  "fixes_applied": ["List of fixes you applied"]
}}

Return ONLY valid JSON:"""

    with StatsTimer(f"regen_{shard_id}") as timer:
        try:
            result = await call_gemini(prompt, temperature=0.2)  # Lower temp for fixes

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
            return content, {}  # Return original on failure


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
    import asyncio

    # Filter to focus shards if specified
    if focus_shards:
        target_shards = [s for s in shards if s.id in focus_shards]
    else:
        target_shards = shards

    if not target_shards:
        logger.info("No shards to regenerate")
        return shards

    logger.info(f"Regenerating {len(target_shards)} shards with feedback...")

    # Create regeneration tasks
    tasks = []
    for shard in target_shards:
        task = regenerate_shard_with_feedback(
            shard_id=shard.id,
            shard_name=shard.name,
            content=shard.content,
            source_scenario=global_factsheet.get("source_scenario", ""),
            target_scenario=global_factsheet.get("target_scenario", ""),
            global_factsheet=global_factsheet,
            feedback=feedback,
        )
        tasks.append((shard, task))

    # Run in parallel
    results = await asyncio.gather(*[t[1] for t in tasks], return_exceptions=True)

    # Update shards with results
    for i, (shard, _) in enumerate(tasks):
        result = results[i]
        if isinstance(result, Exception):
            logger.error(f"Regeneration failed for {shard.id}: {result}")
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
    Post-process adapted content to fix common LLM output issues:
    1. Fix company name typos/inconsistencies
    2. Remove duplicate activities
    3. Ensure descriptions end with proper punctuation
    4. Fix truncated content

    Args:
        content: The adapted content dict
        company_name: The correct company name to enforce

    Returns:
        Fixed content dict
    """
    if not isinstance(content, dict):
        return content

    def fix_company_name(text: str, correct_name: str) -> str:
        """Fix company name typos and variations."""
        if not text or not correct_name:
            return text

        import re
        # Common typo patterns - missing 's', extra spaces, capitalization
        # E.g., "Verde Thread" → "Verde Threads"
        base_name = correct_name.rstrip('s')  # Remove trailing 's' if present

        # Fix variations
        patterns = [
            (rf'\b{re.escape(base_name)}\b(?!s)', correct_name),  # Missing 's'
            (rf'\b{re.escape(base_name.lower())}\b', correct_name),  # Lowercase
            (rf'\b{re.escape(base_name.upper())}\b', correct_name),  # Uppercase
        ]

        for pattern, replacement in patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        return text

    def fix_truncated_text(text: str) -> str:
        """Ensure text ends with proper punctuation."""
        if not text or len(text) < 10:
            return text

        # Check if text ends abruptly (no punctuation)
        if text[-1] not in '.!?"\'):':
            # Try to complete common truncations
            if text.endswith(' and') or text.endswith(' or'):
                text = text.rstrip(' and').rstrip(' or') + '.'
            elif text.endswith(' the') or text.endswith(' a') or text.endswith(' an'):
                text = text.rsplit(' ', 1)[0] + '.'
            elif not text[-1].isalnum():
                pass  # Has some punctuation, leave it
            else:
                text += '.'  # Add period for cleanly truncated text

        return text

    def fix_trailing_punctuation_for_names(obj: Any, path: str = "") -> Any:
        """Remove trailing punctuation from names, roles, and emails. Also normalize gender casing."""
        if isinstance(obj, str):
            return obj
        elif isinstance(obj, dict):
            result = {}
            for key, value in obj.items():
                current_path = f"{path}.{key}" if path else key
                # Check if this is a name/role/email field that shouldn't have trailing periods
                key_lower = key.lower()
                if key_lower in ('name', 'fullname', 'email', 'role', 'title', 'designation', 'jobtitle', 'senderemail', 'avatarurl'):
                    if isinstance(value, str) and value.endswith('.'):
                        result[key] = value.rstrip('.')
                    else:
                        result[key] = value
                # Normalize gender casing: "female" -> "Female", "male" -> "Male"
                elif key_lower == 'gender':
                    if isinstance(value, str):
                        result[key] = value.capitalize() if value.lower() in ('male', 'female', 'other') else value
                    else:
                        result[key] = value
                else:
                    result[key] = fix_trailing_punctuation_for_names(value, current_path)
            return result
        elif isinstance(obj, list):
            return [fix_trailing_punctuation_for_names(item, path) for item in obj]
        return obj

    def remove_duplicate_activities(activities: list) -> list:
        """Remove duplicate activities based on name."""
        if not activities:
            return activities

        seen_names = set()
        unique_activities = []

        for activity in activities:
            if isinstance(activity, dict):
                name = activity.get('name', '')
                # Normalize name for comparison
                name_key = name.lower().strip() if name else ''

                if name_key and name_key not in seen_names:
                    seen_names.add(name_key)
                    unique_activities.append(activity)
                elif not name_key:
                    # Keep activities without names
                    unique_activities.append(activity)
            else:
                unique_activities.append(activity)

        return unique_activities

    def process_value(value: Any, company_name: str) -> Any:
        """Recursively process values."""
        if isinstance(value, str):
            if company_name:
                value = fix_company_name(value, company_name)
            value = fix_truncated_text(value)
            return value
        elif isinstance(value, dict):
            return {k: process_value(v, company_name) for k, v in value.items()}
        elif isinstance(value, list):
            # Check if this is an activities list
            if value and isinstance(value[0], dict) and 'name' in value[0]:
                value = remove_duplicate_activities(value)
            return [process_value(item, company_name) for item in value]
        return value

    # First apply recursive processing
    processed = process_value(content, company_name)
    # Then fix trailing punctuation for names/roles/emails
    processed = fix_trailing_punctuation_for_names(processed)
    return processed


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

async def test_gemini_connection() -> bool:
    """Test if Gemini is properly configured."""
    try:
        result = await call_gemini('Return exactly: {"status": "ok"}')
        return result.get("status") == "ok"
    except Exception as e:
        logger.error(f"Gemini connection test failed: {e}")
        return False


def get_langsmith_status() -> dict:
    """Get LangSmith configuration status."""
    # Support both old (LANGCHAIN_*) and new (LANGSMITH_*) env var names
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
