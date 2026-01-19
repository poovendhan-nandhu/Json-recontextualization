# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

JSON Recontextualization is a FastAPI + LangGraph pipeline that transforms business simulation JSON from one industry/scenario to another while preserving structure and educational alignment. The system follows the principle: **"Same bones, new skin"** - only recontextualize content values, never change structure.

**Current Example Transformation:** HR Hiring/Selection → Market Entry Analysis (EcoChic Threads)

## Current System Status

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Alignment Score | 92.33% | ≥95% | Gap: 2.67% |
| Validation Score | 78.96% | ≥95% | Gap: 16% |
| Compliance Score | 100% | 100% | PASS |
| Blockers | 48 | 0 | HIGH |

### Critical Issues Blocking Release

1. **HR Terminology Leakage** - 185 occurrences of "HR", "hiring", "candidate", "interview"
2. **KLO-to-Resources Gap** - 88% (resources truncated, missing financial model)
3. **KLO-to-Questions Gap** - 90% (questions duplicated, not KLO-specific)

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run development server
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# Run production server
uvicorn src.main:app --host 0.0.0.0 --port 8000

# Run tests
pytest

# Run streaming test (quick validation)
python test_api.py

# Run full sample test
python test_with_sample.py

# Health check
curl http://localhost:8000/api/v1/health
```

## Environment Setup

Copy `.env.example` to `.env` and configure:
- `OPENAI_API_KEY` - Required for validation/fixing (GPT-5.2)
- `GEMINI_API_KEY` / `GOOGLE_API_KEY` - Required for adaptation (Gemini 2.5 Flash)
- `LANGSMITH_API_KEY` - Optional for observability
- `LANGSMITH_TRACING=true` - Enable LangSmith tracing
- `LANGSMITH_PROJECT=cartedo-adaptation` - Project name

## Architecture

### 7-Stage LangGraph Pipeline

```
INPUT → SHARDER → ADAPTATION → ALIGNMENT → [ALIGNMENT FIXER] → VALIDATION → FIXERS → MERGER → FINISHER → HUMAN APPROVAL
```

1. **Sharder** (4ms) - Splits JSON into 15 shards; 2 locked (workspace_ids, scenario_options), 13 unlocked
2. **Adaptation** (142s) - Gemini 2.5 Flash transforms each shard in parallel with RAG context + factsheet
3. **Alignment** (27s) - 9 parallel GPT checks for cross-shard consistency (target: 95%)
4. **Alignment Fixer** - Fixes KLO-to-questions, KLO-to-resources, scenario coherence if score < 95%
5. **Validation** (85s) - 8 critical (blocking) + 6 flagged (non-blocking) checks
6. **Fixers** (30s) - Structural fixer (shape) then Semantic fixer (meaning), parallel per shard
7. **Merger** (48ms) - Reassembles shards into complete JSON
8. **Finisher** (56s) - Compliance checking with optional retry (max 3)
9. **Human Approval** - Creates approval package for review

### Key Directories

- `src/api/routes.py` - All REST API endpoints (~1400 lines)
- `src/graph/nodes.py` - LangGraph pipeline stages with @traceable decorators
- `src/graph/state.py` - PipelineState TypedDict
- `src/stages/` - Stage implementations:
  - `sharder.py` (407 LOC) - JSON splitting, merge, cleanup
  - `adaptation_engine.py` (846 LOC) - Parallel shard adaptation with Gemini
  - `alignment_checker.py` (1,472 LOC) - 9 cross-shard alignment checks
  - `alignment_fixer.py` (1,782 LOC) - KLO alignment repairs
  - `fixers.py` (1,612 LOC) - Structural + Semantic fixers
  - `finisher.py` (392 LOC) - Compliance scoring and decision
- `src/core/` - Leaf-based adaptation (leaf_adapter, decider, context, smart_prompts)
- `src/rag/` - ChromaDB RAG system (retriever, vector_store, industry_knowledge)
- `src/validators/scoped_validators.py` - 8 validators (Domain, Context, Structure, Resource, Inference, WordCount, Data, Entity)
- `src/validation/` - Human-readable reports (feedback_agent, report_generator, report_formatter)
- `src/utils/`:
  - `gemini_client.py` - Gemini API wrapper with factsheet extraction
  - `openai_client.py` - OpenAI/GPT wrapper
  - `prompts.py` - All prompt templates
  - `config.py` - Shard definitions and thresholds
  - `patcher.py` - JSON Pointer patching

### LLM Model Usage

| Purpose | Model | File |
|---------|-------|------|
| Adaptation | Gemini 2.5 Flash | `gemini_client.py` |
| Factsheet Extraction | Gemini 2.5 Flash | `gemini_client.py:352-439` |
| Validation | GPT-5.2 | `scoped_validators.py` |
| Alignment Checks | GPT-5.2 | `alignment_checker.py` |
| Alignment Fixes | GPT-5.2 | `alignment_fixer.py` |
| Structural/Semantic Fixes | GPT-5.2 | `fixers.py` |

### The 15 Shards

**Locked (2):** workspace_ids, scenario_options

**Unlocked (13):** lesson_information, overview, guidelines, simulation_flow, submission_questions, rubric, assessment_criterion (KLOs), emails, characters, resources, workplace_scenario, activities, stage_definitions

## Alignment Rules (9 Rules)

| Rule | What It Checks | Threshold |
|------|----------------|-----------|
| R1 | Reporting Manager Consistency | ≥95% |
| R2 | Company Name Consistency | ≥95% |
| R3 | Poison Term Avoidance | 100% |
| R4 | KLO ↔ Questions Alignment | ≥95% |
| R5 | KLO ↔ Resources Alignment | ≥95% |
| R6 | Scenario ↔ Resources Alignment | ≥95% |
| R7 | Role ↔ Tasks Alignment | ≥90% |
| R8 | KLO ↔ Task Alignment | ≥90% |
| R9 | Scenario Coherence | ≥90% |

## Critical Validation Checks (8 Blocking)

| Check | What It Ensures | Threshold |
|-------|-----------------|-----------|
| C1 | Entity Removal - No original scenario references | 100% |
| C2 | KPI Alignment - Industry KPIs correctly updated | 100% |
| C3 | Schema Validity - Output JSON conforms to schema | 100% |
| C4 | Rubric Integrity - Rubric levels, scoring preserved | 100% |
| C5 | End-to-End Executability - No missing references | 100% |
| C6 | Barrier Compliance - Locked elements never modified | 100% |
| C7 | KLO Preservation - Key Learning Outcomes preserved | ≥95% |
| C8 | Resource Completeness - All resources exist with valid content | 100% |

## Key API Endpoints

- `POST /api/v1/pipeline` - Full 7-stage pipeline
- `POST /api/v1/pipeline/stream` - Streaming pipeline with SSE
- `POST /api/v1/transform/stream` - Streaming with node progress
- `POST /api/v1/transform/stream-openai` - Full OpenAI streaming
- `POST /api/v1/adapt` - Adaptation stage only
- `POST /api/v1/align/check` - Alignment checking only
- `POST /api/v1/leaves/adapt` - Leaf-based LangGraph adaptation
- `POST /api/v1/rag/index` - Index simulation for RAG

## Core Design Principles (Founder's Rules)

1. **"Same bones, new skin"** - Recontextualize content values, NEVER change JSON structure
2. **Shards enable scoped validation/repair** - Never rewrite whole JSON; always work on isolated shards
3. **Checkers diagnose, fixers repair** - Unified Checker produces scorecard; fixers apply targeted changes
4. **Structural fixers before semantic fixers** - Fix shape first (keys, ordering), then meaning (KPIs, tone)
5. **Locked shards are immutable** - After structural fixing, shards are locked and cannot be structurally modified
6. **Alignment threshold: 95%** - Pipeline retries alignment fixes until 95% or max 2 retries
7. **Human confirms realism, not correctness** - System handles correctness; humans approve learning quality
8. **Invisible safety systems** - Frozen structure, shard locks, hash comparisons protect integrity

## Known Issues & Debugging

### Common Problems

1. **Alignment Fixer Finding Empty Questions**
   - **Cause:** Questions are in `simulationFlow[].data.submissionQuestions`, not top-level
   - **File:** `alignment_fixer.py:391-399`
   - **Fix:** Search all locations including simulationFlow stages

2. **Poison List Missing Domain Terms**
   - **Cause:** Factsheet extraction misses domain-implied terms like "HR", "hiring"
   - **File:** `gemini_client.py:129-168`
   - **Fix:** Add HR-specific terms to static poison list

3. **Resource Truncation**
   - **Cause:** No word count validation, LLM stops early
   - **File:** `prompts.py:437-473`
   - **Fix:** Add minimum word count (500+) enforcement

4. **Cross-Shard KLO Blindness**
   - **Cause:** KLOs in `assessment_criterion` shard, questions in `simulation_flow` - adapted independently
   - **File:** `adaptation_engine.py`
   - **Fix:** Pass full KLO context to adaptation prompts

### Logs to Check

```
[ALIGNMENT FIXER] Found N questions total from all locations
Indexed input for RAG: {'klos': 1, 'activities': 4, ...}
RAG retrieval: 5/8 shards got examples
```

## Documentation Map

| Document | Purpose |
|----------|---------|
| `ARCHITECTURE.md` | Pipeline overview |
| `DEEP_ANALYSIS.md` | 23,300 LOC analysis, agent clashes |
| `ADAPTATION_FLOW.md` | Complete adaptation prompts and data flow |
| `VALIDATION_DASHBOARD.md` | Agent glossary, KPI dashboard |
| `VALIDATION_REPORT.md` | Canonical validation output format |
| `CURRENT_ISSUES.md` | Remaining issues and status |
| `LEAF_ADAPTER_PLAN.md` | Smart one-shot adaptation plan |
| `KLO_FIXER_PLAN.md` | KLO-Question alignment fix plan |
| `src/CHECKRULES.md` | Founder's architecture rules |
| `AUDIT.md` | Codebase audit with critical blockers |

## Testing Files

| File | Purpose |
|------|---------|
| `test_api.py` | Quick streaming test (~10s) |
| `test_with_sample.py` | Full sample_input.json test (~30-60s) |
| `test_openai_stream.py` | Raw OpenAI streaming output |
| `test_stream.py` | Alternative streaming test |

## Output Files

| File | From | Contains |
|------|------|----------|
| `test_output.json` | test_api.py | Minimal test output |
| `transformed_output.json` | test_with_sample.py | Full response with validation |
| `transformed_data_only.json` | test_with_sample.py | Just the transformed JSON |
| `openai_streamed_text.txt` | test_openai_stream.py | Raw streamed text |
