# Prompt Simplification - Direct Scenario Adaptation

**Date:** 2026-01-17
**Goal:** Scenario prompt is the SINGLE SOURCE OF TRUTH for everything

---

## THE KEY INSIGHT

The **scenario prompt** contains EVERYTHING:

```
"learners will act as a junior consultant for an exciting Gen Z organic T-shirts
brand, tasked with analyzing the U.S. market and providing a go/no-go market entry
recommendation. Using only the simulation's provided data, they will apply structured
frameworks to assess market potential, competition, capabilities, finances, and risks
before developing their final strategy."
```

From this ONE prompt, LLM understands:

| Element | Extracted From Scenario |
|---------|------------------------|
| **Company** | "Gen Z organic T-shirts brand" → EcoChic Threads |
| **Industry** | Organic fashion / Sustainable retail |
| **Role** | Junior consultant |
| **Challenge** | Go/no-go market entry recommendation |
| **KLO1** | Assess market potential (TAM/SAM/SOM) |
| **KLO2** | Analyze competition, capabilities, finances |
| **KLO3** | Develop go/no-go recommendation with risk assessment |
| **Frameworks** | SWOT, PESTEL, financial analysis |

**No separate factsheet extraction needed!**

---

## CROSS-CONNECTED PARALLEL GENERATION

All shards run in PARALLEL but are CROSS-CONNECTED via the same scenario understanding:

```
                 ┌──────────────────────────────────────┐
                 │         SCENARIO PROMPT              │
                 │  (Single Source of Truth)            │
                 │                                      │
                 │  Company: EcoChic Threads            │
                 │  Industry: Organic T-shirts          │
                 │  KLOs: Market analysis, go/no-go     │
                 │  Role: Junior consultant             │
                 └──────────────────────────────────────┘
                                  │
                                  │ SAME prompt given to ALL shards
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        │                         │                         │
        ▼                         ▼                         ▼
 ┌─────────────┐          ┌─────────────┐          ┌─────────────┐
 │    KLOs     │          │  Questions  │          │  Resources  │
 │   Shard     │          │   Shard     │          │   Shard     │
 └─────────────┘          └─────────────┘          └─────────────┘
        │                         │                         │
        ▼                         ▼                         ▼
  KLOs about               Questions that            Resources with
  market analysis,         ASSESS market             market data,
  go/no-go decision        analysis skills           competitor info,
                                                     financial models

        └─────────────────────────┴─────────────────────────┘
                                  │
                                  ▼
                    ALL ALIGNED because ALL understand
                    the SAME scenario prompt
```

**Why this works:**
- Each shard READS the scenario prompt
- Each shard UNDERSTANDS what KLOs should be (from scenario)
- Each shard generates content that ALIGNS to those KLOs
- No need to pass KLOs between shards - they all derive from same source!

---

## THE PROMPT

```python
SCENARIO_ADAPTATION_PROMPT = """You are adapting a business simulation.

## TARGET SCENARIO (READ THIS CAREFULLY - THIS IS YOUR SOURCE OF TRUTH):
{scenario_prompt}

From this scenario, understand:
- What COMPANY is this about?
- What INDUSTRY is this?
- What ROLE does the learner play?
- What CHALLENGE must they solve?
- What KEY LEARNING OUTCOMES (KLOs) are implied?

---

## WHAT YOU ARE ADAPTING:
This is the "{shard_name}" section of the simulation.

```json
{content}
```

---

## QUALITY RULES:

### 1. Alignment to Scenario KLOs
- The scenario implies specific learning outcomes (what learner must DO)
- ALL content must support these learning outcomes
- Questions must ASSESS these outcomes
- Resources must PROVIDE data to achieve these outcomes

### 2. Domain Fidelity
- Use terminology specific to the TARGET industry
- Use KPIs appropriate for the TARGET industry
- If scenario is "organic T-shirts" → use retail/fashion terms
- NOT HR terms, NOT old scenario terms

### 3. Consistency (CRITICAL)
- Use ONE company name throughout (derive from scenario)
- Use ONE manager name throughout (create realistic name for industry)
- Manager email: firstname.lastname@companyname.com
- These must be CONSISTENT across the entire simulation

### 4. Resource Quality
- Resources: 500-1500 words (complete, not truncated)
- Self-contained: all info needed to answer questions
- Don't give answers directly: student must analyze and infer
- Include REAL statistics: "X% (Source: McKinsey 2024)"
- Include competitor analysis with real companies

### 5. Question-KLO Alignment
- Each submission question must assess a specific KLO from the scenario
- Questions must use terminology that matches the scenario
- Questions must be answerable using the resources

### 6. No Placeholders
- NO [brackets] with placeholder text
- NO "TBD", "TODO", "XXX"
- NO truncated sentences
- ALL content must be complete and real

---

## OUTPUT:
Return ONLY the adapted JSON for this shard.
Same structure as input, new content aligned to TARGET SCENARIO.
No explanations. Just valid JSON.
"""
```

---

## IMPLEMENTATION

### File: `src/stages/adaptation_engine.py`

```python
class AdaptationEngine:
    """Simplified: Scenario prompt is the only input needed."""

    async def adapt(
        self,
        input_json: dict,
        scenario_prompt: str,
    ) -> AdaptationResult:
        """
        Adapt simulation using scenario prompt as single source of truth.

        All shards run in PARALLEL but are CROSS-CONNECTED because
        they all receive the SAME scenario prompt and derive the same
        understanding (company, industry, KLOs) from it.
        """
        total_start = time.time()

        # 1. Shard the JSON (structure already known, IDs locked)
        sharder = Sharder()
        collection = sharder.shard(input_json)

        locked = [s for s in collection.shards if s.lock_state == LockState.FULLY_LOCKED]
        unlocked = [s for s in collection.shards if s.lock_state != LockState.FULLY_LOCKED]

        logger.info(f"Shards: {len(locked)} locked, {len(unlocked)} unlocked")

        # 2. Adapt ALL unlocked shards in PARALLEL
        #    Each shard gets the SAME scenario_prompt
        #    Cross-connection happens because all derive from same source
        tasks = [
            self._adapt_shard(
                shard=s,
                scenario_prompt=scenario_prompt,
            )
            for s in unlocked
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 3. Update shards with results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Shard {unlocked[i].id} failed: {result}")
            else:
                unlocked[i].content = result

        # 4. Merge back to full JSON
        adapted_json = merge_shards(collection, input_json)

        total_time_ms = int((time.time() - total_start) * 1000)

        return AdaptationResult(
            adapted_json=adapted_json,
            scenario_prompt=scenario_prompt,
            shards_adapted=len(unlocked),
            shards_locked=len(locked),
            total_time_ms=total_time_ms,
        )

    async def _adapt_shard(
        self,
        shard,
        scenario_prompt: str,
    ) -> dict:
        """Adapt single shard using scenario prompt."""
        from ..utils.gemini_client import call_gemini

        prompt = SCENARIO_ADAPTATION_PROMPT.format(
            scenario_prompt=scenario_prompt,
            shard_name=shard.name,
            content=json.dumps(shard.content, indent=2),
        )

        return await call_gemini(prompt)
```

---

### File: `src/utils/prompts.py`

**REMOVE (not needed anymore):**
- `FACTSHEET_PROMPT` - scenario prompt has everything
- `HARD_BLOCKS_WITH_CONSEQUENCES` - simplify
- `MANDATORY_VERIFICATION_FIELDS` - trust LLM more
- `build_factsheet_prompt()` - not needed
- Complex `build_shard_adaptation_prompt()` - replace with simple one

**ADD:**
```python
SCENARIO_ADAPTATION_PROMPT = """..."""  # The prompt above

def build_adaptation_prompt(scenario_prompt: str, shard_name: str, content: dict) -> str:
    return SCENARIO_ADAPTATION_PROMPT.format(
        scenario_prompt=scenario_prompt,
        shard_name=shard_name,
        content=json.dumps(content, indent=2),
    )
```

---

### File: `src/utils/gemini_client.py`

**REMOVE:**
- `extract_global_factsheet()` - not needed
- `filter_poison_list()` - LLM infers from scenario
- Poison list handling - not needed

**SIMPLIFY `adapt_shard_content()`:**
```python
async def adapt_shard_content(
    shard_name: str,
    content: dict,
    scenario_prompt: str,
) -> dict:
    """Simple: just scenario + content → adapted content."""
    prompt = build_adaptation_prompt(scenario_prompt, shard_name, content)
    return await call_gemini(prompt)
```

---

## HOW RESOURCES AND QUESTIONS WORK TOGETHER

**Critical insight from sample_main.json:**

Resources provide **DATA/EVIDENCE**, not answers:
```
"teams led by coordinators rated high in communication resolved guest issues 43% faster"
"68% of events rated 'Excellent' were managed by coordinators scoring high in teamwork"
"27% reduction in first-year turnover compared to unstructured interviews"
```

Questions ask for **ANALYSIS**, not recall:
```
"Justify why Communication and Conflict Resolution is essential..."
"Develop one structured leading interview question..."
"Explain how your structured rating scale supports validity..."
```

### The Pattern

| Resources Provide | Questions Ask | Learner Must Do |
|-------------------|---------------|-----------------|
| Raw statistics (43% faster) | "Justify why..." | Connect stat to conclusion |
| Facts about the role | "Develop questions..." | Create questions using context |
| Industry benchmarks | "Explain validity..." | Apply data to defend approach |

### Resources Rules (CRITICAL)
1. **Provide DATA, not answers** - Give statistics, facts, benchmarks
2. **Enable analysis** - Data must be usable to answer questions
3. **Don't solve the problem** - Learner must do the analytical work
4. **Be self-contained** - All needed data is in the resources
5. **Match the domain** - Data must be about TARGET scenario (e.g., organic T-shirts market data)

### Example for Market Entry Scenario

**Resource should contain:**
```
- "US organic apparel market: $12.8B (Source: Grand View Research 2024)"
- "Gen Z consumers: 73% prefer sustainable brands (Source: McKinsey 2024)"
- "Competitor analysis: Pact ($45M revenue), Organic Basics ($28M revenue)"
- "Customer acquisition cost in DTC apparel: $35-50 per customer"
```

**Question should ask:**
```
"What is your TAM/SAM/SOM estimate for EcoChic Threads in the US market?"
```

**Learner must:**
- Use the $12.8B market size
- Apply Gen Z percentage to narrow to SAM
- Consider competitor data for realistic SOM
- Show their calculation (not copy-paste the number)

---

## WHY CROSS-CONNECTION WORKS

All shards derive from the SAME scenario prompt, so they all understand:
- What DATA is needed (resources)
- What ANALYSIS is required (questions)
- What OUTCOMES matter (KLOs)

```
                    SCENARIO PROMPT
                    "go/no-go market entry for organic T-shirts"
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
         ▼                 ▼                 ▼
    KLOs shard        Questions shard   Resources shard
         │                 │                 │
         ▼                 ▼                 ▼
  "Analyze market     "What is your     "Market size is
   opportunity"        TAM estimate?"    $12.8B, Gen Z is
  "Evaluate fit"      "Go or no-go?"     73% sustainable..."
  "Recommend"
         │                 │                 │
         └─────────────────┴─────────────────┘
                           │
                           ▼
            Questions ASK about what
            Resources PROVIDE data for
            (but don't give direct answers)
```

---

## SUMMARY

| Before | After |
|--------|-------|
| Factsheet extraction (separate LLM call) | No factsheet needed |
| Poison list (incomplete, misses terms) | LLM infers from scenario |
| Complex 900-line prompts | Simple 50-line prompt |
| Shards don't see each other | All shards see same scenario |
| KLOs passed between shards | KLOs derived from scenario |
| 14+ LLM calls | ~13 LLM calls (one per shard) |

---

## TEST SCENARIOS (from Shweta)

```
PROMPT 1:
learners will act as a junior consultant for an exciting Gen Z organic T-shirts
brand, tasked with analyzing the U.S. market and providing a go/no-go market
entry recommendation. Using only the simulation's provided data, they will
apply structured frameworks to assess market potential, competition,
capabilities, finances, and risks before developing their final strategy.

PROMPT 2:
students will propose how a fast food brand should respond to its competitor's
$1 menu by analyzing the competitor's move, market impact, strengths, and four
strategic options. Their goal is to propose a clear, realistic, and sustainable
plan to protect or grow market share via an executive summary.

PROMPT 3:
Acting as a strategic analyst at ThriveBite Nutrition, learners will assess
the viability of launching a new adaptogen-infused functional beverage
targeting health-conscious consumers seeking stress relief. They will analyze
product-market fit, estimate the market opportunity, benchmark competitors,
evaluate internal capabilities, assess financial feasibility, and weigh
potential risks using the resources provided.
```

---

## NEXT STEPS

1. **Create `simple_adapt.py`** - New simplified adaptation function
2. **Update `prompts.py`** - Add SCENARIO_ADAPTATION_PROMPT
3. **Test with PROMPT 1** - Run on sample_main.json
4. **Compare alignment scores** - Should be better because cross-connected

---

## DEEP ANALYSIS: COMPLETE FILE CHANGES

### CODEBASE OVERVIEW

**Total**: 61 Python files, ~31,175 lines of code

```
src/
├── api/              (2 files, ~1,400 LOC) - FastAPI routes
├── core/             (15 files, ~7,800 LOC) - Leaf-based adaptation
├── stages/           (11 files, ~7,500 LOC) - Pipeline stages
├── graph/            (3 files, ~2,000 LOC) - LangGraph workflow
├── models/           (3 files, ~400 LOC) - Pydantic schemas
├── rag/              (4 files, ~1,200 LOC) - RAG/embedding system
├── utils/            (10 files, ~2,000 LOC) - Config, prompts, helpers
├── validation/       (7 files, ~900 LOC) - Validation checking
└── validators/       (2 files, ~400 LOC) - Base validators
```

---

### FILE-BY-FILE CHANGES

#### 1. API LAYER (`src/api/`)

| File | LOC | Action | Changes |
|------|-----|--------|---------|
| `routes.py` | 1,387 | **SIMPLIFY** | Keep only `/adapt` endpoint, remove 20+ others |

**Current Endpoints (REMOVE):**
- `/shard`, `/shard/{id}` - Shard management
- `/adapt` (shard-based) - Old adaptation
- `/align/check`, `/adapt-and-check` - Alignment
- `/pipeline`, `/pipeline/stream` - Full pipeline
- `/rag/*` - RAG endpoints
- `/validate` - Validation only

**New Endpoint (KEEP):**
```python
POST /api/v1/adapt
Body: {
  "input_json": {...},
  "scenario_prompt": "learners will act as..."
}
```

**Result**: 1,387 LOC → ~200 LOC (86% reduction)

---

#### 2. STAGES LAYER (`src/stages/`)

| File | LOC | Action | Changes |
|------|-----|--------|---------|
| `sharder.py` | 407 | **KEEP** | Still needed for structure preservation |
| `adaptation_engine.py` | 846 | **REMOVE** | Replace with simple scenario-based |
| `alignment_checker.py` | 1,472 | **REMOVE** | Not needed - LLM handles alignment |
| `alignment_fixer.py` | 1,782 | **REMOVE** | Not needed |
| `alignment_fixer_prompts.py` | 200 | **REMOVE** | Not needed |
| `fixers.py` | 1,612 | **SIMPLIFY** | Keep basic structure fixer only |
| `fixer_prompts.py` | 200 | **SIMPLIFY** | Reduce prompts |
| `finisher.py` | 392 | **REMOVE** | Not needed for simple flow |
| `human_approval.py` | 200 | **REMOVE** | Not needed |

**Result**: 7,500 LOC → ~1,000 LOC (87% reduction)

---

#### 3. UTILS LAYER (`src/utils/`)

| File | LOC | Action | Changes |
|------|-----|--------|---------|
| `config.py` | 300 | **KEEP** | SHARD_DEFINITIONS still needed |
| `prompts.py` | 600 | **REWRITE** | Replace with SCENARIO_ADAPTATION_PROMPT |
| `gemini_client.py` | 250 | **SIMPLIFY** | Remove factsheet extraction |
| `openai_client.py` | 150 | **KEEP** | Still needed for validation |
| `rules.py` | 400 | **REMOVE** | Not needed - LLM validates |
| `helpers.py` | 200 | **KEEP** | General utilities |
| `patcher.py` | 200 | **KEEP** | JSON patching |

**Key Change in `prompts.py`:**

```python
# REMOVE:
- FACTSHEET_PROMPT (600+ lines)
- HARD_BLOCKS_WITH_CONSEQUENCES
- MANDATORY_VERIFICATION_FIELDS
- build_factsheet_prompt()
- build_shard_adaptation_prompt()

# ADD:
SCENARIO_ADAPTATION_PROMPT = """..."""  # ~50 lines
```

**Result**: 2,000 LOC → ~800 LOC (60% reduction)

---

#### 4. GRAPH LAYER (`src/graph/`)

| File | LOC | Action | Changes |
|------|-----|--------|---------|
| `state.py` | 229 | **SIMPLIFY** | Reduce to essential state fields |
| `nodes.py` | 1,220 | **REMOVE** | Not needed for simple flow |
| `workflow.py` | 48 | **REMOVE** | Not needed |

**Result**: 2,000 LOC → ~100 LOC (95% reduction)

---

#### 5. CORE LAYER (`src/core/`)

| File | LOC | Action | Changes |
|------|-----|--------|---------|
| `leaf_adapter.py` | 500 | **SIMPLIFY** | Keep but remove RAG calls |
| `leaf_graph.py` | 743 | **SIMPLIFY** | Reduce to 4 stages |
| `context.py` | 400 | **SIMPLIFY** | Remove factsheet, use scenario |
| `indexer.py` | 200 | **KEEP** | Still needed |
| `decider.py` | 1,244 | **KEEP** | Core decision making |
| `smart_prompts.py` | 559 | **REWRITE** | Use SCENARIO_ADAPTATION_PROMPT |
| `leaf_validators.py` | 400 | **SIMPLIFY** | Keep 2/5 validators |
| `leaf_fixers.py` | 300 | **KEEP** | Basic fixing |
| `leaf_repair_loop.py` | 400 | **SIMPLIFY** | 1 iteration max |
| `feedback_agent.py` | 200 | **REMOVE** | Not needed |
| `leaf_rag.py` | 300 | **REMOVE** | RAG not needed |
| `classifier.py` | 200 | **KEEP** | Pre-filter leaves |
| `grouper.py` | 300 | **KEEP** | Semantic grouping |

**Result**: 7,800 LOC → ~3,500 LOC (55% reduction)

---

#### 6. RAG LAYER (`src/rag/`)

| File | LOC | Action | Changes |
|------|-----|--------|---------|
| `vector_store.py` | 300 | **REMOVE** | Not needed |
| `embeddings.py` | 150 | **REMOVE** | Not needed |
| `retriever.py` | 651 | **REMOVE** | Not needed |
| `industry_knowledge.py` | 100 | **REMOVE** | Not needed |

**Result**: 1,200 LOC → 0 LOC (100% removal)

---

#### 7. VALIDATION LAYER (`src/validation/`)

| File | LOC | Action | Changes |
|------|-----|--------|---------|
| `check_definitions.py` | 200 | **REMOVE** | LLM validates |
| `check_runner.py` | 300 | **REMOVE** | Not needed |
| `report_formatter.py` | 150 | **REMOVE** | Not needed |
| `report_generator.py` | 200 | **REMOVE** | Not needed |
| `validation_agent.py` | 250 | **REMOVE** | Not needed |

**Result**: 900 LOC → 0 LOC (100% removal)

---

### SUMMARY OF CHANGES

| Component | Current LOC | After Simplification | Reduction |
|-----------|-------------|---------------------|-----------|
| API Routes | 1,387 | 200 | 86% |
| Stages | 7,500 | 1,000 | 87% |
| Utils | 2,000 | 800 | 60% |
| Graph | 2,000 | 100 | 95% |
| Core | 7,800 | 3,500 | 55% |
| RAG | 1,200 | 0 | 100% |
| Validation | 900 | 0 | 100% |
| Models | 400 | 300 | 25% |
| Validators | 400 | 200 | 50% |
| **TOTAL** | **31,175** | **~6,100** | **~80%** |

---

### SIMPLIFIED ADAPTATION FLOW

```
BEFORE (Complex - 14+ LLM calls):
────────────────────────────────────
Request
  ↓
Factsheet Extraction (Gemini) ─────────┐
  ↓                                    │
Sharder ───────────────────────────────┤
  ↓                                    │
Parallel Shard Adaptation (Gemini x11) │
  ↓                                    │
Alignment Checker (GPT x5) ────────────┤
  ↓                                    │
Alignment Fixer (GPT) ─────────────────┤
  ↓                                    │
Validation (GPT x13) ──────────────────┤
  ↓                                    │
Fixers (GPT) ──────────────────────────┤
  ↓                                    │
Merger ────────────────────────────────┤
  ↓                                    │
Finisher ──────────────────────────────┤
  ↓                                    │
Human Approval ────────────────────────┘
  ↓
Response


AFTER (Simple - with 6 Validation Agents + Repair):
────────────────────────────────────
Request (input_json + scenario_prompt)
  ↓
┌─────────────────────────────────────────┐
│ STAGE 1: SHARDER                        │
│ Split JSON into 13 shards (2 locked)    │
└─────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────┐
│ STAGE 2: ADAPTATION (Gemini 2.5 Flash)  │
│ Parallel shard adaptation x13           │
│ Each shard gets SAME scenario_prompt    │
│ LLM derives: company, KLOs, terminology │
└─────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────┐
│ STAGE 3: MERGER                         │
│ Reassemble shards into full JSON        │
└─────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 4: VALIDATION (GPT 5.2) - 6 AGENTS IN PARALLEL               │
│                                                                     │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐                   │
│  │1. Domain    │ │2. Context   │ │3. Resource  │                   │
│  │   Fidelity  │ │   Fidelity  │ │   Quality   │                   │
│  │   (98%)     │ │   (98%)     │ │   (95%)     │                   │
│  └─────────────┘ └─────────────┘ └─────────────┘                   │
│                                                                     │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐                   │
│  │4. KLO-Q     │ │5. Consistency│ │6. Complete- │                   │
│  │   Alignment │ │   (98%)     │ │   ness (98%)│                   │
│  │   (98%)     │ │             │ │             │                   │
│  └─────────────┘ └─────────────┘ └─────────────┘                   │
│                                                                     │
│  Output: Dashboard report with scores + issues                      │
└─────────────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────┐
│ STAGE 5: REPAIR (GPT 5.2) - if needed   │
│ Fix issues from any agent < threshold   │
│ Max 3 iterations, then accept ≥95%      │
└─────────────────────────────────────────┘
  ↓
Response (adapted JSON + validation report)
```

---

### AGENTS OVERVIEW

| Agent | Model | Purpose |
|-------|-------|---------|
| **Sharder** | None (code) | Split JSON into shards, preserve structure |
| **Adaptation Agent** | Gemini 2.5 Flash | Transform content to match scenario prompt |
| **Merger** | None (code) | Reassemble shards into full JSON |
| **1. Domain Fidelity Agent** | GPT 5.2 | Check industry terminology, KPIs match target |
| **2. Context Fidelity Agent** | GPT 5.2 | Verify goal/challenge preserved from scenario |
| **3. Resource Quality Agent** | GPT 5.2 | Ensure self-contained, <1500 words, inference map (no direct answers) |
| **4. KLO-Question Alignment Agent** | GPT 5.2 | Confirm questions assess KLOs from scenario |
| **5. Consistency Agent** | GPT 5.2 | Validate same company/manager names throughout |
| **6. Completeness Agent** | GPT 5.2 | Detect placeholders, truncation, missing data |
| **Repair Agent** | GPT 5.2 | Fix issues found by validation agents |

---

### VALIDATION AGENTS (Based on Shweta's Requirements)

**One-line summary:**
> 6 specialized validation agents check domain fidelity, context preservation, resource quality (inference map), KLO-question alignment, consistency, and completeness - targeting 98% threshold.

---

#### VALIDATION DASHBOARD

```
┌────────────────────────────────────────────────────────────────────────────┐
│                        ADAPTATION VALIDATION REPORT                        │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  Agent                    Score     Threshold    Status                    │
│  ────────────────────────────────────────────────────────────────────────  │
│  1. Domain Fidelity       97.5%     98%          ⚠️  NEEDS REPAIR          │
│  2. Context Fidelity      100%      98%          ✅ PASS                   │
│  3. Resource Quality      94.2%     95%          ⚠️  NEEDS REPAIR          │
│  4. KLO-Question Align    98.5%     98%          ✅ PASS                   │
│  5. Consistency           100%      98%          ✅ PASS                   │
│  6. Completeness          100%      98%          ✅ PASS                   │
│                                                                            │
│  ────────────────────────────────────────────────────────────────────────  │
│  OVERALL SCORE: 98.4%     THRESHOLD: 98%         STATUS: ⚠️  PARTIAL       │
│                                                                            │
│  Issues Found: 3                                                           │  
| 3 | **Resource Quality** | Ensures resources are self-contained, <1500 words, and DON'T give direct answers | 95% | GPT 5.2 |
| 4 | **KLO-Question Alignment** | Confirms each question assesses a KLO implied by the scenario | 98% | GPT 5.2 |
| 5 | **Consistency** | Validates same company/manager/email used throughout simulation | 98% | GPT 5.2 |
| 6 | **Completeness** | Detects placeholders, truncated content, or missing data | 98% | GPT 5.2 |

---

#### AGENT DETAILS

##### Agent 1: Domain Fidelity Agent

**What it checks:**
- All terminology matches TARGET industry (not source)
- KPIs are appropriate for TARGET industry
- Examples and data references are industry-relevant

**Shweta's Example:**
> "If scenario is fast food → use '$1.01 menu', 'drive-thru efficiency', 'quick-service KPIs'"
> "If scenario is airline → use 'loyalty points', 'load factor', 'yield management'"

**Prompt:**
```
You are validating domain fidelity for a business simulation.

SCENARIO: {scenario_prompt}

Check if ALL content uses terminology appropriate for this industry.

FLAG any terms that:
1. Belong to a DIFFERENT industry (e.g., "HR interview" in a market entry sim)
2. Are generic when industry-specific terms exist
3. Use wrong KPIs for the industry

CONTENT TO CHECK:
{content}

OUTPUT:
{
  "score": 0.0-1.0,
  "issues": [
    {"location": "...", "term": "...", "should_be": "...", "reason": "..."}
  ],
  "pass": true/false
}
```

---

##### Agent 2: Context Fidelity Agent

**What it checks:**
- Original learning goal is preserved (go/no-go, recommendation, analysis)
- Challenge type matches (market entry, competitor response, product launch)
- Educational purpose is maintained

**Shweta's Example:**
> "If original was 'go/no-go decision' → output must still require a go/no-go decision"
> "If original was 'executive summary' → output must still ask for executive summary"

**Prompt:**
```
You are validating context fidelity for a business simulation.

SCENARIO: {scenario_prompt}

From scenario, the GOAL is: {inferred_goal}
The learner should: {inferred_challenge}

Check if the adapted content PRESERVES this goal and challenge.

FLAG if:
1. The main deliverable has changed (go/no-go → strategy plan)
2. The decision type has changed (entry → expansion)
3. The learning purpose is different

CONTENT TO CHECK:
{content}

OUTPUT:
{
  "score": 0.0-1.0,
  "goal_preserved": true/false,
  "challenge_preserved": true/false,
  "issues": [...]
}
```

---

##### Agent 3: Resource Quality Agent (CRITICAL - Inference Map)

**What it checks:**
- Resources are self-contained (have ALL info needed)
- Word count is under 1500 words
- Resources provide DATA not answers (dots to connect, not connected dots)

**Shweta's Key Insight:**
> "Resources should be an INFERENCE MAP - give the learner dots to connect, not the connected dots"
> "Don't say 'The TAM is $12.8B so you should enter the market'"
> "DO say 'US organic apparel market: $12.8B (Source)... Gen Z preference: 73%...'"

**Prompt:**
```
You are validating resource quality for a business simulation.

RULES:
1. SELF-CONTAINED: Resource must have all data needed to answer its questions
2. WORD LIMIT: Resource must be 500-1500 words (complete but not bloated)
3. INFERENCE MAP: Resource provides DATA/FACTS, NOT conclusions

CRITICAL - Check for DIRECT ANSWERS:
❌ BAD: "The TAM is $12.8B, which makes this an attractive market"
❌ BAD: "Based on competitor analysis, EcoChic should enter the market"
❌ BAD: "The recommended strategy is to focus on Gen Z consumers"

✅ GOOD: "US organic apparel market: $12.8B (Source: Grand View 2024)"
✅ GOOD: "Competitor A: $45M revenue, 12% market share"
✅ GOOD: "Gen Z consumers: 73% prefer sustainable brands"

The learner must CONNECT these dots to reach conclusions.

RESOURCE TO CHECK:
{resource_content}

QUESTIONS THIS RESOURCE SUPPORTS:
{related_questions}

OUTPUT:
{
  "score": 0.0-1.0,
  "word_count": N,
  "is_self_contained": true/false,
  "has_direct_answers": true/false,
  "direct_answer_examples": [...],
  "missing_data_for_questions": [...],
  "issues": [...]
}
```

---

##### Agent 4: KLO-Question Alignment Agent

**What it checks:**
- Each submission question assesses a KLO from the scenario
- Question terminology matches scenario
- Questions are answerable from resources

**Prompt:**
```
You are validating KLO-question alignment.

SCENARIO: {scenario_prompt}

IMPLIED KLOs from scenario:
{inferred_klos}

For each question, check:
1. Does it assess one of the implied KLOs?
2. Does it use terminology matching the scenario?
3. Can it be answered from the resources?

QUESTIONS TO CHECK:
{questions}

RESOURCES AVAILABLE:
{resource_summary}

OUTPUT:
{
  "score": 0.0-1.0,
  "question_alignments": [
    {"question": "...", "maps_to_klo": "...", "answerable": true/false}
  ],
  "issues": [...]
}
```

---

##### Agent 5: Consistency Agent

**What it checks:**
- ONE company name used throughout
- ONE manager name used throughout
- Email format: firstname.lastname@company.com
- No mixed naming conventions

**Prompt:**
```
You are validating naming consistency.

SCENARIO: {scenario_prompt}

Extract from content:
- All company name variations
- All manager/person names
- All email addresses

FLAG inconsistencies:
❌ "EcoChic" vs "EcoChic Threads" vs "Eco-Chic"
❌ "Sarah Chen" vs "Sarah" vs "S. Chen"
❌ "sarah@ecochic.com" vs "s.chen@ecochicthreads.com"

CONTENT TO CHECK:
{full_json}

OUTPUT:
{
  "score": 0.0-1.0,
  "company_names_found": [...],
  "manager_names_found": [...],
  "emails_found": [...],
  "is_consistent": true/false,
  "issues": [...]
}
```

---

##### Agent 6: Completeness Agent

**What it checks:**
- No placeholder text [TBD], [INSERT], [Your Name]
- No truncated sentences ending mid-thought
- No empty or null fields that should have content
- No TODO comments left in content

**Prompt:**
```
You are validating content completeness.

SCAN for:
1. PLACEHOLDERS: [TBD], [INSERT], [Your Name], [Company], XXX, TODO
2. TRUNCATION: Sentences that end mid-word or mid-thought "The market is..."
3. EMPTY FIELDS: Fields that should have content but are empty/null
4. INCOMPLETE LISTS: "1. First item 2. Second item 3. [more to come]"

CONTENT TO CHECK:
{full_json}

OUTPUT:
{
  "score": 0.0-1.0,
  "placeholders_found": [...],
  "truncated_content": [...],
  "empty_fields": [...],
  "issues": [...]
}
```

---

#### VALIDATION FLOW

```
┌─────────────────────────────────────────────────────────────────┐
│                     ADAPTED JSON (from Gemini)                  │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│              PARALLEL VALIDATION (GPT 5.2)                      │
│                                                                 │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │
│  │ Domain   │ │ Context  │ │ Resource │ │ KLO-Q    │           │
│  │ Fidelity │ │ Fidelity │ │ Quality  │ │ Align    │           │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘           │
│       │            │            │            │                  │
│  ┌──────────┐ ┌──────────┐                                      │
│  │Consistency│ │Complete- │                                     │
│  │          │ │ness      │                                      │
│  └────┬─────┘ └────┬─────┘                                      │
│       │            │                                            │
└───────┴────────────┴────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    AGGREGATE SCORES                             │
│                                                                 │
│  IF all agents ≥ threshold (98%):                               │
│    → PASS: Return adapted JSON                                  │
│                                                                 │
│  ELSE:                                                          │
│    → Collect all issues                                         │
│    → Send to REPAIR AGENT                                       │
│    → Re-validate (max 3 iterations)                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

#### THRESHOLD STRATEGY (From Shweta)

**Initial Target: 98%**
- All agents must score ≥ 98%
- If ANY agent fails, trigger repair

**Acceptable: 95%**
- After 3 repair iterations, accept ≥ 95%
- Flag for human review if between 95-98%
- Reject if < 95%

```
┌─────────────────────────────────────────────┐
│           THRESHOLD DECISION TREE           │
├─────────────────────────────────────────────┤
│                                             │
│  Score ≥ 98% ───────────→ ✅ PASS           │
│                                             │
│  95% ≤ Score < 98% ─────→ ⚠️  PASS          │
│                            (flag for review)│
│                                             │
│  Score < 95% ───────────→ ❌ FAIL           │
│                            (needs repair)   │
│                                             │
└─────────────────────────────────────────────┘
```

---

### REPAIR LOGIC

```
IF validation finds issues:
  ↓
  FOR each issue:
    ↓
    Repair Agent (GPT 5.2) rewrites the problematic section
    ↓
  Re-validate
  ↓
  IF still issues AND iterations < 3:
    ↓
    Retry repair
  ↓
  ELSE:
    ↓
    Return result with validation report
```

---

### THE NEW SIMPLIFIED PROMPT

```python
SCENARIO_ADAPTATION_PROMPT = """You are adapting a business simulation.

## TARGET SCENARIO (YOUR SOURCE OF TRUTH):
{scenario_prompt}

From this scenario, understand:
- COMPANY: What company/brand is this about?
- INDUSTRY: What industry/domain?
- ROLE: What role does the learner play?
- CHALLENGE: What problem must they solve?
- KLOs: What must the learner demonstrate?
- DATA REQUIREMENTS: What data/analysis is needed?

---

## WHAT YOU ARE ADAPTING:
This is the "{shard_name}" section of the simulation.

{content}

---

## RULES:

### Structure
- Keep ALL IDs exactly as they are
- Keep ALL object/array structures
- Only change content VALUES

### Content Quality
- Resources provide DATA (statistics, facts) not answers
- Questions ask for ANALYSIS (justify, develop, explain)
- Learner must connect data to conclusions

### Domain Fidelity
- Use terminology for TARGET industry
- Replace ALL terms from source scenario
- Use appropriate KPIs for TARGET

### Consistency
- ONE company name throughout
- ONE manager name throughout
- Manager email: firstname.lastname@company.com

### Completeness
- NO placeholders [like this]
- NO truncated content
- Resources: 500-1500 words with citations

---

## OUTPUT:
Return ONLY the adapted JSON. Same structure, new content.
No explanations. Just valid JSON.
"""
```

---

### IMPLEMENTATION PRIORITY

| Priority | Task | Files | Impact |
|----------|------|-------|--------|
| **P0** | Create new `simple_adapt.py` | New file | Core functionality |
| **P0** | Add SCENARIO_ADAPTATION_PROMPT | `prompts.py` | Core prompt |
| **P0** | Simplify `/adapt` endpoint | `routes.py` | API entry point |
| **P1** | Remove factsheet extraction | `gemini_client.py` | Cleanup |
| **P1** | Remove RAG layer | `src/rag/` | Cleanup |
| **P2** | Remove alignment checker | `alignment_checker.py` | Cleanup |
| **P2** | Remove validation layer | `src/validation/` | Cleanup |
| **P3** | Remove unused graph code | `src/graph/` | Cleanup |

---

### QUICK START IMPLEMENTATION

**Step 1: Create `src/stages/simple_adapt.py`**
```python
async def adapt_simple(input_json: dict, scenario_prompt: str) -> dict:
    """Simplified adaptation using scenario prompt only."""

    # 1. Shard (for structure)
    sharder = Sharder()
    collection = sharder.shard(input_json)

    # 2. Adapt each unlocked shard in parallel
    tasks = [
        adapt_shard_simple(shard, scenario_prompt)
        for shard in collection.unlocked_shards
    ]
    results = await asyncio.gather(*tasks)

    # 3. Merge and return
    return sharder.merge(collection, results)
```

**Step 2: Add to `routes.py`**
```python
@router.post("/adapt")
async def adapt(request: AdaptRequest):
    adapted = await adapt_simple(
        input_json=request.input_json,
        scenario_prompt=request.scenario_prompt
    )
    return {"adapted_json": adapted, "status": "OK"}
```

**Step 3: Test with Shweta's prompt**
```python
scenario_prompt = """learners will act as a junior consultant for an exciting
Gen Z organic T-shirts brand, tasked with analyzing the U.S. market and
providing a go/no-go market entry recommendation..."""

result = await adapt_simple(sample_main_json, scenario_prompt)
```

---

*Updated: 2026-01-17*
*Deep analysis complete - 6 validation agents defined based on Shweta's requirements*
*Ready for implementation*
