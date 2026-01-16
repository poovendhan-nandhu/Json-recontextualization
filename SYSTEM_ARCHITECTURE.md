# Cartedo Simulation Adaptation System - Complete Architecture

## Overview

A LangGraph-based pipeline that adapts business simulations from one industry/company to another while maintaining structural integrity, KLO alignment, and content quality.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CARTEDO ADAPTATION PIPELINE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────────┐ │
│  │   INPUT      │───▶│  ADAPTATION  │───▶│  VALIDATION  │───▶│  FIXERS   │ │
│  │   (JSON)     │    │  (Gemini)    │    │  (GPT 5.2)   │    │ (GPT 5.2) │ │
│  └──────────────┘    └──────────────┘    └──────────────┘    └───────────┘ │
│                                                                     │       │
│                                                                     ▼       │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────────┐ │
│  │   OUTPUT     │◀───│  FEEDBACK    │◀───│   MERGER     │◀───│ FINISHER  │ │
│  │   (JSON)     │    │   AGENT      │    │              │    │           │ │
│  └──────────────┘    └──────────────┘    └──────────────┘    └───────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Model Split

| Stage | Model | Purpose |
|-------|-------|---------|
| **Adaptation** | Gemini 2.5 Flash | Content transformation, context extraction |
| **Validation** | GPT 5.2 | Quality checks, compliance verification |
| **Fixers** | GPT 5.2 | Structural & semantic repairs |
| **Feedback Agent** | GPT 5.2 | Final validation report generation |

---

## Pipeline Stages

### Stage 1: Input Processing
```
src/models/shard.py          → Shard definitions
src/utils/config.py          → SHARD_DEFINITIONS
```

**Input**: Raw simulation JSON + Target scenario
**Output**: Sharded JSON ready for parallel processing

### Stage 2: Context Extraction (Gemini)
```
src/core/context.py          → AdaptationContext extraction
src/utils/gemini_client.py   → extract_global_factsheet()
```

**Extracts**:
- Company mappings (old → new)
- KLO terms (for question alignment)
- Resource data (for answerability)
- Industry context (valid/invalid KPIs)
- Poison terms (must remove)

### Stage 3: Adaptation (Gemini)
```
src/stages/adaptation_engine.py  → Main adaptation orchestrator
src/core/leaf_adapter.py         → Leaf-based adaptation
src/core/decider.py              → LLM decision making
src/core/smart_prompts.py        → Smart prompts with validation rules
```

**Two Modes**:
1. **Shard-based**: Parallel shard adaptation
2. **Leaf-based**: Individual leaf decisions (new)

### Stage 4A: Validation (GPT 5.2)
```
src/validators/scoped_validators.py  → 8 validators
src/validation/check_definitions.py  → Check definitions
src/validation/check_runner.py       → Check execution
```

**8 Validators**:
| # | Validator | Checks | Blocker |
|---|-----------|--------|---------|
| 1 | Domain Fidelity | Industry terms, KPIs match target | YES |
| 2 | Context Fidelity | KLO/criteria counts match base | YES |
| 3 | Structure Integrity | JSON structure matches schema | NO |
| 4 | Resource Self-Contained | Questions answerable from resources | YES |
| 5 | Inference Integrity | No ranges/placeholders in resources | YES |
| 6 | Word Count | Section lengths within bounds | NO |
| 7 | Data Consistency | Numbers match across sections | YES |
| 8 | Entity Removal | No old company/industry references | YES |

### Stage 4B: Fixers (GPT 5.2)
```
src/stages/fixers.py             → All fixer classes
src/stages/fixer_prompts.py      → Fixer-specific prompts
src/stages/alignment_fixer.py    → KLO alignment fixes
```

**Fixer Classes**:
- `StructuralFixer` - Fix JSON shape (missing keys, types)
- `SemanticFixer` - Fix meaning (entities, KPIs, terminology)
- `KLOAlignmentFixer` - Fix KLO-question mapping
- `BatchedSemanticFixer` - Apply batched fixes
- `ScopedFixer` - Orchestrate all fixes

### Stage 5: Alignment Check
```
src/stages/alignment_checker.py  → Cross-shard alignment
```

**Checks**:
- KLO-Question alignment
- Resource-Question answerability
- Cross-reference consistency

### Stage 6: Finisher & Merger
```
src/stages/finisher.py           → Final processing
src/graph/nodes.py               → merge_shards()
```

**Actions**:
- Merge adapted shards back
- Final validation pass
- Prepare output JSON

### Stage 7: Feedback Agent (GPT 5.2) - NEW
```
src/validation/feedback_agent.py  → Canonical validation report
src/validation/report_generator.py → Report data generation
src/validation/report_formatter.py → Markdown formatting
```

**Generates**:
- Executive decision gate
- Critical checks dashboard
- Failure summaries
- Recommended fixes
- Binary ship decision

---

## Directory Structure

```
src/
├── api/
│   └── routes.py                 # FastAPI endpoints
├── core/                         # NEW: Leaf-based adaptation
│   ├── __init__.py
│   ├── context.py               # AdaptationContext extraction
│   ├── decider.py               # LLM decision maker
│   ├── grouper.py               # Semantic grouping
│   ├── indexer.py               # Leaf path indexing
│   ├── classifier.py            # Leaf classification
│   ├── leaf_adapter.py          # Main leaf orchestrator
│   └── smart_prompts.py         # Smart prompts with validation rules
├── graph/
│   ├── nodes.py                 # LangGraph nodes
│   └── state.py                 # Pipeline state
├── models/
│   ├── shard.py                 # Shard model
│   └── schemas.py               # Pydantic schemas
├── stages/
│   ├── adaptation_engine.py     # Main adaptation
│   ├── alignment_checker.py     # KLO alignment
│   ├── alignment_fixer.py       # Alignment fixes
│   ├── finisher.py              # Final processing
│   ├── fixers.py                # All fixers (GPT 5.2)
│   └── fixer_prompts.py         # Fixer prompts
├── utils/
│   ├── config.py                # Configuration
│   ├── gemini_client.py         # Gemini API (adaptation)
│   ├── openai_client.py         # OpenAI API (validation/fix)
│   ├── patcher.py               # JSON Pointer patching
│   ├── prompts.py               # Prompt templates
│   └── llm_stats.py             # Token/call tracking
├── validators/
│   └── scoped_validators.py     # 8 validators (GPT 5.2)
├── validation/
│   ├── check_definitions.py     # Check definitions
│   ├── check_runner.py          # Check execution
│   ├── report_generator.py      # Report data
│   ├── report_formatter.py      # Markdown formatting
│   └── feedback_agent.py        # NEW: Feedback Agent
└── rag/
    └── embeddings.py            # RAG embeddings
```

---

## Data Flow

```
INPUT JSON
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. SHARDING                                                  │
│    Split JSON into 12 shards for parallel processing         │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. CONTEXT EXTRACTION (Gemini)                               │
│    - Extract global factsheet                                │
│    - Identify poison terms                                   │
│    - Map entities (old → new)                                │
│    - Extract KLO terms                                       │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. ADAPTATION (Gemini) - PARALLEL                            │
│    ┌─────────┬─────────┬─────────┬─────────┐                │
│    │ Shard 1 │ Shard 2 │ Shard 3 │ Shard N │                │
│    │ rubrics │resources│sim_flow │workplace│                │
│    └─────────┴─────────┴─────────┴─────────┘                │
│    OR                                                        │
│    Leaf-based: Index → Pre-filter → Decide → Patch          │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 4A. VALIDATION (GPT 5.2) - PARALLEL                          │
│    ┌─────────────┬─────────────┬─────────────┐              │
│    │ Structural  │  Semantic   │  Alignment  │              │
│    │  Checks     │   Checks    │   Checks    │              │
│    └─────────────┴─────────────┴─────────────┘              │
└─────────────────────────────────────────────────────────────┘
    │
    ▼ (if issues found)
┌─────────────────────────────────────────────────────────────┐
│ 4B. FIXERS (GPT 5.2) - PARALLEL                              │
│    ┌─────────────┬─────────────┬─────────────┐              │
│    │ Structural  │  Semantic   │    KLO      │              │
│    │   Fixer     │   Fixer     │   Fixer     │              │
│    └─────────────┴─────────────┴─────────────┘              │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. ALIGNMENT CHECK                                           │
│    - KLO-Question alignment                                  │
│    - Resource answerability                                  │
│    - Cross-reference consistency                             │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. MERGE & FINISH                                            │
│    - Merge shards back to single JSON                        │
│    - Final validation pass                                   │
│    - Prepare output                                          │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 7. FEEDBACK AGENT (GPT 5.2) - FINAL                          │
│    - Generate canonical validation report                    │
│    - Executive decision gate                                 │
│    - Ship/Fix recommendation                                 │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
OUTPUT JSON + VALIDATION REPORT
```

---

## Observability

### LangSmith Integration
```python
# Enabled via environment variables
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_key
LANGSMITH_PROJECT=cartedo-adaptation

# @traceable decorator on all key functions
@traceable(name="adaptation_engine")
async def adapt_shard(...):
    ...
```

### Metrics Tracking
```python
# src/utils/llm_stats.py
stats = get_stats()
stats.add_call(success=True, shard_id="rubrics", elapsed_time=3.2)
stats.add_retry(wait_time=2.0, is_rate_limit=True)

# Get summary
summary = stats.get_summary()
# {
#     "total_calls": 45,
#     "successful_calls": 43,
#     "failed_calls": 2,
#     "total_retries": 5,
#     "avg_latency_ms": 2100,
#     ...
# }
```

---

## Configuration

### Environment Variables
```bash
# Models
GOOGLE_API_KEY=your_gemini_key
OPENAI_API_KEY=your_openai_key
OPENAI_MODEL=gpt-5.2-2025-12-11
FIXER_MODEL=gpt-5.2-2025-12-11
VALIDATION_MODEL=gpt-5.2-2025-12-11

# LangSmith
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_PROJECT=cartedo-adaptation

# Parallelism
MAX_CONCURRENT_LLM_CALLS=10

# Thresholds
BLOCKER_PASS_RATE_REQUIRED=1.0
OVERALL_SCORE_REQUIRED=0.95
```

---

## API Endpoints

```
POST /adapt              → Full adaptation pipeline
POST /adapt/stream       → Streaming adaptation with progress
POST /validate           → Validation only
GET  /health             → Health check
GET  /langsmith/status   → LangSmith configuration status
```

---

## Latency Targets

| Stage | Target | Strategy |
|-------|--------|----------|
| Context Extraction | <5s | Single Gemini call |
| Adaptation | <100s | Parallel shards/leaves |
| Validation | <20s | Parallel checks |
| Fixers | <30s | Parallel fixes |
| Feedback Agent | <10s | Single GPT 5.2 call |
| **Total** | **<150s** | Parallelization + batching |

---

## Key Design Decisions

1. **Gemini for Adaptation**: Faster, cheaper for bulk content transformation
2. **GPT 5.2 for Validation**: Better reasoning for quality checks
3. **Leaf-based Adaptation**: More precise, fewer unnecessary changes
4. **Smart Prompts**: Validation rules built into prompts = fewer retries
5. **JSON Pointer Patching**: Surgical fixes, not regeneration
6. **Parallel Everything**: Semaphore-controlled concurrency
7. **Feedback Agent at End**: Single source of truth for ship decision
