# Leaf-Based Adaptation Plan

## Goal
- **One-shot adaptation** that's so good, validation/fixers have minimal work
- **Latency target**: < 150 seconds (not 720s)
- **Strategy**: Smart prompts + Parallelization + Batching

---

## Core Insight

> "The adaptation prompt should already KNOW all the validation rules"

Instead of: Adapt → Validate → Fix → Re-validate (multiple loops)

We want: **Adapt (with full context)** → Validate (mostly passes) → Minor fixes if any

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SMART ONE-SHOT LEAF ADAPTATION                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  INPUT JSON                                                                  │
│      │                                                                       │
│      ▼                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  PHASE 1: CONTEXT EXTRACTION (1 LLM call)            ~5s            │    │
│  │  - Extract factsheet (company, industry, entities)                  │    │
│  │  - Build entity map (old → new)                                     │    │
│  │  - Get RAG industry context (KPIs, terminology)                     │    │
│  │  - Extract KLO terms, resource data, rubric structure               │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│      │                                                                       │
│      ▼                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  PHASE 2: INDEX + GROUP                              ~1s            │    │
│  │  - Extract all leaf paths (907 leaves)                              │    │
│  │  - Pre-filter: Skip IDs, URLs, locked (480 skip)                    │    │
│  │  - Group by semantic type (6 groups)                                │    │
│  │  - Create batches (50 leaves/batch → ~9 batches)                    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│      │                                                                       │
│      ▼                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  PHASE 3: SMART PARALLEL ADAPTATION                  ~60-90s        │    │
│  │                                                                      │    │
│  │  ┌─────────────────────────────────────────────────────────────┐    │    │
│  │  │  9 BATCHES IN PARALLEL (semaphore: 6 concurrent)            │    │    │
│  │  │                                                              │    │    │
│  │  │  Each batch gets FULL CONTEXT in prompt:                    │    │    │
│  │  │  - Target scenario + Company + Industry                     │    │    │
│  │  │  - Entity mappings (old → new)                              │    │    │
│  │  │  - KLO terms to use (for questions)                         │    │    │
│  │  │  - Resource data summary (for questions)                    │    │    │
│  │  │  - Industry KPIs (from RAG)                                 │    │    │
│  │  │  - Validation rules embedded in prompt                      │    │    │
│  │  └─────────────────────────────────────────────────────────────┘    │    │
│  │                                                                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│      │                                                                       │
│      ▼                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  PHASE 4: PATCH APPLICATION                          ~1s            │    │
│  │  - Apply all "replace" decisions                                    │    │
│  │  - JSON Pointer surgical patches                                    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│      │                                                                       │
│      ▼                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  PHASE 5: VALIDATION (parallel)                      ~20-30s        │    │
│  │  - Run all validators in parallel                                   │    │
│  │  - Most should PASS because adaptation was smart                    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│      │                                                                       │
│      ▼                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  PHASE 6: TARGETED FIXES (if any)                    ~10-20s        │    │
│  │  - Only fix specific failures (not whole sections)                  │    │
│  │  - Surgical leaf-level repairs                                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│      │                                                                       │
│      ▼                                                                       │
│  OUTPUT JSON                                                                 │
│                                                                              │
│  TOTAL LATENCY: ~100-140s                                                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## The Secret: Smart Prompts

### Current (Dumb) Prompt:
```
Adapt this leaf to the target scenario.
Path: /questions/0/text
Value: "What are the critical success factors?"
```

### Smart Prompt (with validation rules built-in):
```
You are adapting simulation content. Your output must pass ALL validation rules.

TARGET CONTEXT:
- Company: Global Beverages Inc
- Industry: Beverage
- Valid KPIs: market share, revenue growth, distribution reach (NOT: CAC, churn, MRR)

KLO ALIGNMENT REQUIREMENT:
The questions MUST use these EXACT terms from KLOs:
- KLO1: "critical success factors"
- KLO2: "strategic analysis questions"
- KLO3: "feasibility scoring matrix"

RESOURCE ANSWERABILITY:
Questions must be answerable from this resource data:
- Revenue: $2.4M quarterly
- Market share: 12%
- Growth rate: 8% YoY

ENTITY REMOVAL:
These terms must NOT appear in output: "Accor", "HarvestBowls", "hotel"

---
LEAF TO ADAPT:
Path: /questions/0/text
Value: "What are the critical success factors for Accor Hotel?"

RESPOND WITH:
{
  "action": "replace",
  "new_value": "What are the critical success factors for Global Beverages Inc?",
  "reason": "Replaced entity, kept exact KLO term 'critical success factors'"
}
```

---

## Latency Breakdown

| Phase | What | Time | Strategy |
|-------|------|------|----------|
| 1 | Context extraction | ~5s | 1 LLM call |
| 2 | Index + Group | ~1s | No LLM, pure code |
| 3 | Adaptation | ~60-90s | 9 batches, 6 parallel |
| 4 | Patch | ~1s | No LLM, pure code |
| 5 | Validation | ~20-30s | Parallel validators |
| 6 | Fixes | ~10-20s | Only if needed |
| **TOTAL** | | **~100-140s** | Target: <150s |

---

## Parallelization Strategy

### Batch Configuration:
```python
BATCH_SIZE = 50              # leaves per batch
MAX_CONCURRENT = 6           # parallel LLM calls (avoid rate limit)
TOTAL_CANDIDATES = ~427      # after pre-filter
TOTAL_BATCHES = 9            # ceil(427/50)
```

### Execution:
```
Time 0s:   Start batches 1-6 in parallel
Time ~10s: Batch 1,2,3 complete → Start batch 7,8,9
Time ~20s: All batches complete

Total: ~20s for all batches (not 9 × 10s = 90s sequential)
```

### Semaphore Control:
```python
semaphore = asyncio.Semaphore(6)

async def process_batch(batch):
    async with semaphore:
        return await call_llm(batch)

# All batches start together, semaphore controls concurrency
tasks = [process_batch(b) for b in batches]
results = await asyncio.gather(*tasks)
```

---

## Upgraded Prompt System

### Flow:
```
1. CONTEXT EXTRACTION (once)     → Get company info, KLOs, resources
2. SMART DECISION (per batch)    → Decide + validate in ONE prompt
3. REFERENCE CHECK (if replace)  → Quick check for leaked old names
4. TARGETED RETRY (if failed)    → Re-adapt specific leaves with feedback
```

---

### PROMPT 1: Context Extraction (1 LLM call)

```python
CONTEXT_EXTRACTION_PROMPT = """
You are analyzing a simulation JSON to extract adaptation context.

**SOURCE SIMULATION:**
{source_json_summary}

**TARGET SCENARIO:**
{target_scenario}

**EXTRACT:**

1. **COMPANY MAPPING:**
   - Old company name (all variations: full, short, abbreviations)
   - New company name from target scenario
   - Old industry terms (e.g., "hotel", "hospitality", "guest")
   - New industry terms (e.g., "beverage", "drink", "consumer")

2. **KEY LEARNING OUTCOMES (KLOs):**
   Extract the EXACT key terms from each KLO that questions MUST reference:
   - KLO1: [exact key phrase]
   - KLO2: [exact key phrase]
   - KLO3: [exact key phrase]

3. **RESOURCE DATA POINTS:**
   Extract specific numbers/data that questions should reference:
   - Revenue/financial figures
   - Growth rates/percentages
   - Market metrics

4. **INDUSTRY CONTEXT:**
   - Valid KPIs for target industry
   - Invalid KPIs (from source industry, must avoid)

**RESPOND AS JSON:**
{
  "company": {
    "old_names": ["Accor Hotel", "Accor", "AccorHotels"],
    "new_name": "Global Beverages Inc",
    "old_industry_terms": ["hotel", "hospitality", "guest", "room", "booking"],
    "new_industry_terms": ["beverage", "drink", "consumer", "distribution"]
  },
  "klo_terms": {
    "klo1": "critical success factors",
    "klo2": "strategic analysis questions",
    "klo3": "feasibility scoring matrix"
  },
  "resource_data": {
    "revenue": "$2.4M quarterly",
    "growth": "8% YoY",
    "market_share": "12%"
  },
  "industry": {
    "target": "beverage",
    "valid_kpis": ["market share", "distribution reach", "brand awareness"],
    "invalid_kpis": ["occupancy rate", "ADR", "RevPAR", "guest satisfaction"]
  },
  "poison_terms": ["Accor", "hotel", "hospitality", "guest", "room", "booking"]
}
"""
```

---

### PROMPT 2: Smart Decision (per batch) - THE KEY PROMPT

```python
SMART_DECISION_PROMPT = """
You are a JSON contextualization expert adapting simulation content.

═══════════════════════════════════════════════════════════════════════
                         TARGET CONTEXT
═══════════════════════════════════════════════════════════════════════

**Company:** {new_company_name}
**Industry:** {target_industry}
**Scenario:** {target_scenario_summary}

═══════════════════════════════════════════════════════════════════════
                    MANDATORY REPLACEMENTS
═══════════════════════════════════════════════════════════════════════

OLD → NEW Company Names:
{entity_mapping}

OLD → NEW Industry Terms:
{industry_term_mapping}

**POISON TERMS (must NOT appear in output):**
{poison_terms}

═══════════════════════════════════════════════════════════════════════
                    VALIDATION RULES (built-in)
═══════════════════════════════════════════════════════════════════════

Your output will be validated against these rules. Get it RIGHT first time:

**RULE 1: Entity Removal**
- NO old company names or variations (case-insensitive)
- NO old industry terms
- Check: "Accor", "accor", "ACCOR", "hotel", "Hotel" → ALL must be replaced

**RULE 2: KLO-Question Alignment** (for question leaves)
Questions MUST use these EXACT terms from KLOs:
{klo_terms_list}

Example:
  ❌ "What are the core assessment criteria?"
  ✅ "What are the critical success factors?"  ← Uses exact KLO term

**RULE 3: Resource Answerability** (for question leaves)
Questions must be answerable from this data:
{resource_data_summary}

Example:
  ❌ "What was the revenue last year?" (if not in resources)
  ✅ "Analyze the $2.4M quarterly revenue" (uses actual data)

**RULE 4: Domain Fidelity**
Valid KPIs for {target_industry}: {valid_kpis}
INVALID KPIs (do not use): {invalid_kpis}

Example:
  ❌ "Improve occupancy rate" (hotel KPI)
  ✅ "Improve market share" (beverage KPI)

**RULE 5: Data Consistency**
Numbers must match across sections:
{data_consistency_points}

═══════════════════════════════════════════════════════════════════════
                         LEAVES TO ADAPT
═══════════════════════════════════════════════════════════════════════

{leaves_to_adapt}

═══════════════════════════════════════════════════════════════════════
                         RESPONSE FORMAT
═══════════════════════════════════════════════════════════════════════

For each leaf, decide: KEEP (already appropriate) or REPLACE (needs change)

{
  "decisions": [
    {
      "index": 0,
      "action": "keep",
      "new_value": null,
      "reason": "Generic text, no changes needed"
    },
    {
      "index": 1,
      "action": "replace",
      "new_value": "What are the critical success factors for Global Beverages Inc?",
      "reason": "Replaced 'Accor Hotel' with company name, kept exact KLO term 'critical success factors'"
    }
  ]
}

**CRITICAL:**
- Maintain same tone, length, and style as original
- Do NOT change IDs, URLs, or technical identifiers
- When in doubt, KEEP the original
- For REPLACE: ensure ALL validation rules are satisfied
"""
```

---

### PROMPT 3: Reference Check (quick validation)

```python
REFERENCE_CHECK_PROMPT = """
Check if this text contains ANY reference to old context:

**OLD NAMES TO DETECT:**
{old_names_list}

**OLD INDUSTRY TERMS:**
{old_industry_terms}

**TEXT TO CHECK:**
"{text_to_check}"

**CHECK FOR:**
- Case variations: "Accor", "accor", "ACCOR"
- Partial matches: "Accor" in "AccorHotels"
- Industry terms: "hotel", "hospitality", "guest"

**RESPOND:**
{
  "has_old_references": true/false,
  "found_references": ["Accor", "hotel"],
  "locations": ["word 5", "word 12"]
}
"""
```

---

### PROMPT 4: Targeted Retry (only for failed leaves)

```python
TARGETED_RETRY_PROMPT = """
**VALIDATION FAILED** - Previous adaptation still contains old references.

═══════════════════════════════════════════════════════════════════════
                      WHAT WENT WRONG
═══════════════════════════════════════════════════════════════════════

**Leaf Path:** {path}
**Your Previous Output:** "{previous_output}"
**Problems Found:**
{validation_failures}

═══════════════════════════════════════════════════════════════════════
                      MANDATORY FIXES
═══════════════════════════════════════════════════════════════════════

{specific_fixes_required}

**EXAMPLES:**
{fix_examples}

═══════════════════════════════════════════════════════════════════════
                      TRY AGAIN
═══════════════════════════════════════════════════════════════════════

Provide a corrected value that passes ALL validation rules:

{
  "new_value": "...",
  "fixes_applied": ["Replaced 'Accor' with 'Global Beverages'", "Used KLO term 'critical success factors'"]
}

**THIS IS YOUR LAST ATTEMPT - GET IT RIGHT**
"""
```

---

### Group-Specific Context Additions

For **QUESTIONS** batch, add to SMART_DECISION_PROMPT:
```
**QUESTION-SPECIFIC RULES:**
- Each question MUST reference at least one KLO term
- Questions must be answerable from the provided resource data
- Use {target_industry} terminology, not {source_industry}
```

For **RESOURCES** batch, add:
```
**RESOURCE-SPECIFIC RULES:**
- Data must support answering the submission questions:
  {questions_summary}
- Numbers must be realistic for {target_industry} industry
- Maintain data consistency with scenario: {scenario_numbers}
```

For **RUBRICS** batch, add:
```
**RUBRIC-SPECIFIC RULES:**
- Criteria must align with KLOs:
  {klo_to_rubric_mapping}
- Scoring levels (1-5) must have clear progression
- Assessment language must match {target_industry}
```

---

## Context Extraction (Phase 1)

Extract ONCE, use everywhere:

```python
@dataclass
class AdaptationContext:
    # Company/Industry
    company_name: str
    old_company_name: str
    industry: str

    # Entity mappings
    entity_map: Dict[str, str]  # old → new
    poison_terms: List[str]      # terms to remove

    # KLO data (for question/rubric alignment)
    klo_terms: List[str]         # exact terms from KLOs
    klo_details: List[Dict]      # full KLO objects

    # Resource data (for question answerability)
    resource_summary: str        # key data points
    resource_numbers: Dict       # revenue, growth, etc.

    # Industry context (from RAG)
    valid_kpis: List[str]        # KPIs for this industry
    invalid_kpis: List[str]      # KPIs to avoid
    terminology: List[str]       # industry-specific terms
```

---

## Validation Checks (Phase 5)

Run in parallel, most should pass:

| Check | What | Expected After Smart Adapt |
|-------|------|---------------------------|
| Entity Removal | No old names | 99% pass |
| KLO-Question Alignment | Terms match | 95% pass |
| Resource Answerability | Data available | 90% pass |
| Rubric-KLO Alignment | Criteria map | 95% pass |
| Domain Fidelity | Valid KPIs | 98% pass |
| Data Consistency | Numbers match | 90% pass |

---

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `src/core/context.py` | CREATE | AdaptationContext extraction |
| `src/core/smart_prompts.py` | CREATE | Group-specific prompt templates |
| `src/core/decider.py` | MODIFY | Use smart prompts with full context |
| `src/core/leaf_adapter.py` | MODIFY | Pass context to decider |
| `src/core/validators.py` | CREATE | Leaf-level validators |

---

## Implementation Order

1. **Create AdaptationContext** - Extract all context upfront
2. **Create Smart Prompts** - Group-specific templates
3. **Update Decider** - Use smart prompts
4. **Add Leaf Validators** - Lightweight checks
5. **Integration Test** - Measure latency + quality

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Total latency | < 150s |
| Validation pass rate (first attempt) | > 90% |
| Fixes needed | < 10% of leaves |
| Entity removal | 100% |
| KLO alignment | > 95% |

---

## Key Principle

> "Give the LLM ALL the context it needs to get it right the FIRST time"

- Don't adapt blindly, then validate, then fix
- Adapt WITH validation rules built into the prompt
- Validation becomes a safety check, not a primary quality gate
