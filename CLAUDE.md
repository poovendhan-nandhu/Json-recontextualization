# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

JSON Recontextualization is a FastAPI + LangGraph pipeline that transforms business simulation JSON from one industry/scenario to another while preserving structure and educational alignment. The system follows the principle: **"Same bones, new skin"** - only recontextualize content values, never change structure.

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

# Health check
curl http://localhost:8000/api/v1/health
```

## Environment Setup

Copy `.env.example` to `.env` and configure:
- `OPENAI_API_KEY` - Required for validation/fixing (GPT-5.2)
- `GEMINI_API_KEY` - Required for adaptation (Gemini 2.5 Flash)
- `LANGSMITH_API_KEY` - Optional for observability

## Architecture

### 7-Stage LangGraph Pipeline

```
INPUT → SHARDER → ADAPTATION → ALIGNMENT → [ALIGNMENT FIXER] → VALIDATION → FIXERS → MERGER → FINISHER → HUMAN APPROVAL
```

1. **Sharder** (4ms) - Splits JSON into 14 independent shards; 2 locked (metadata, scenario_options), 12 unlocked
2. **Adaptation** (142s) - Gemini 2.5 Flash transforms each shard in parallel with RAG context
3. **Alignment** (27s) - 9 parallel GPT checks for cross-shard consistency (target: 98%)
4. **Alignment Fixer** - Fixes KLO-to-questions, KLO-to-resources, scenario coherence if score < 98%
5. **Validation** (85s) - Fast validators (structure, IDs, completeness) + batched LLM checks per shard
6. **Fixers** (30s) - Specialized prompts by shard type, parallel processing
7. **Merger** (48ms) - Reassembles shards into complete JSON
8. **Finisher** (56s) - Compliance checking with optional retry (max 3)
9. **Human Approval** - Creates approval package for review

### Key Directories

- `src/api/routes.py` - All REST API endpoints (~1400 lines)
- `src/graph/nodes.py` - LangGraph pipeline stages with @traceable decorators
- `src/graph/state.py` - PipelineState TypedDict
- `src/stages/` - Stage implementations (sharder, adaptation_engine, alignment_checker, alignment_fixer, fixers, finisher)
- `src/core/` - Leaf-based adaptation (leaf_adapter, leaf_validators, leaf_graph)
- `src/rag/` - ChromaDB RAG system (retriever, vector_store, industry_knowledge)
- `src/validators/` - Fast + LLM validators (scoped_validators.py)
- `src/utils/config.py` - Shard definitions and thresholds

### LLM Model Usage

| Purpose | Model |
|---------|-------|
| Adaptation | Gemini 2.5 Flash |
| Validation | GPT-5.2 |
| Alignment | GPT-5.2 |
| Fixers | GPT-5.2 |

### The 14 Shards

**Locked (2):** metadata, scenario_options

**Unlocked (12):** lesson_information, assessment_criteria (KLOs), industry_activities, activities_chat_history, selected_scenario, workplace_scenario, scenario_chat_history, simulation_flow (stages), emails, rubrics, resources, launch_settings

## Key API Endpoints

- `POST /api/v1/pipeline` - Full 7-stage pipeline
- `POST /api/v1/pipeline/stream` - Streaming pipeline with SSE
- `POST /api/v1/adapt` - Adaptation stage only
- `POST /api/v1/align/check` - Alignment checking only
- `POST /api/v1/leaves/adapt` - Leaf-based LangGraph adaptation
- `POST /api/v1/rag/index` - Index simulation for RAG

## Core Design Principles

1. **Shards enable scoped validation/repair** - Never rewrite whole JSON; always work on isolated shards
2. **Checkers diagnose, fixers repair** - Unified Checker produces scorecard; fixers apply targeted changes
3. **Structural fixers before semantic fixers** - Fix shape first (keys, ordering), then meaning (KPIs, tone)
4. **Locked shards are immutable** - After structural fixing, shards are locked and cannot be structurally modified
5. **Alignment threshold: 98%** - Pipeline retries alignment fixes until 98% or max 2 retries
6. **Human confirms realism, not correctness** - System handles correctness; humans approve learning quality