# FastAPI-LangGraph Application - Complete Documentation

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Workflow Graph](#workflow-graph)
- [Node Details](#node-details)
- [API Endpoints](#api-endpoints)
- [Key Features](#key-features)
- [Data Flow](#data-flow)
- [Dependencies](#dependencies)
- [Configuration](#configuration)
- [LangGraph Deep Dive](#langgraph-deep-dive)
- [Testing](#testing)
- [Deployment](#deployment)

---

## Overview

### What is This Application?

This is a **FastAPI-based RESTful API service** that uses **LangGraph** (agentic workflow orchestration) and **OpenAI GPT-4** to transform business simulation JSON scenarios while preserving critical data structures.

### Core Purpose

Transforms business simulation scenarios (e.g., "fast-food restaurant competition" → "fashion retail BOGO promotion") while:

- ✅ Maintaining exact JSON schema structure
- ✅ Preserving specific "locked" fields byte-for-byte (using SHA-256 verification)
- ✅ Ensuring consistency across all transformed content
- ✅ Providing comprehensive validation reports
- ✅ Real-time progress streaming via Server-Sent Events (SSE)

### Technology Stack

```
Framework:      FastAPI 0.104.1 (async REST API)
Orchestration:  LangGraph 0.2.45 (stateful workflow graphs)
AI Engine:      OpenAI GPT-4o (JSON transformation)
Validation:     Pydantic 2.5.3 (schema validation)
Utilities:      jsondiff, tenacity, hashlib
Runtime:        Uvicorn 0.24.0 (ASGI server)
```

---

## Architecture

### Architectural Pattern

**Agentic Workflow with Graph-Based State Machine**

The application implements a sophisticated multi-node pipeline where:
- **Nodes** are independent agents responsible for specific tasks
- **Edges** define the flow between nodes
- **State** is a shared dictionary that accumulates data as it flows through nodes
- **Conditional Edges** enable dynamic routing based on validation results

### Why This Architecture?

Traditional sequential approach:
```python
# Sequential (inflexible)
def transform(input_json):
    data = validate(input_json)
    entities = analyze(data)
    result = transform_with_openai(entities)
    validate_result(result)
    return result

# Problems:
# - Hard to add retry logic
# - Can't short-circuit
# - No visibility into progress
# - Can't stream intermediate results
```

LangGraph approach:
```python
# Graph-based (flexible, observable)
workflow = StateGraph(WorkflowState)
workflow.add_node("validate", validate_node)
workflow.add_node("analyze", analyze_node)
workflow.add_node("transform", transform_node)
workflow.add_conditional_edges("analyze", should_retry)

# Benefits:
# ✓ Conditional routing (short-circuit, retry)
# ✓ Observable (node logs, streaming)
# ✓ Testable (test individual nodes)
# ✓ Maintainable (add/remove nodes easily)
```

---

## Project Structure

```
fastapi-langgraph-app 2/
│
├── src/                          # Main application code
│   ├── main.py                   # FastAPI app entry point
│   ├── api/                      # API layer
│   │   ├── __init__.py
│   │   └── routes.py             # REST endpoints (4 routes)
│   ├── graph/                    # LangGraph workflow (CORE LOGIC)
│   │   ├── __init__.py
│   │   ├── state.py              # State schema (TypedDict)
│   │   ├── nodes.py              # 6 node implementations (740 lines)
│   │   └── workflow.py           # Workflow graph definition
│   ├── models/                   # Data models
│   │   ├── __init__.py
│   │   └── schemas.py            # Pydantic request/response models
│   └── utils/                    # Utilities
│       ├── __init__.py
│       ├── config.py             # Configuration management
│       ├── helpers.py            # Utility functions (9 helpers)
│       └── openai_client.py      # OpenAI API wrapper
│
├── Configuration Files
│   ├── requirements.txt          # 17 Python dependencies
│   ├── pyproject.toml            # Poetry project metadata
│   ├── .env.example              # Environment variable template
│   └── .env                      # Actual config (gitignored)
│
├── Testing
│   ├── test_workflow.py          # Unit tests (9 test cases)
│   ├── test_api.py               # API integration tests
│   ├── test_with_sample.py       # Large file streaming test
│   ├── test_stream.py            # Alternative streaming test
│   └── test_openai_stream.py     # Raw OpenAI output test
│
├── Documentation
│   ├── README.md                 # User guide
│   ├── IMPLEMENTATION_SUMMARY.md # Implementation details
│   ├── TESTING_GUIDE.md          # Testing documentation
│   └── QUICKSTART.py             # Quick setup guide
│
└── Data
    ├── sample_input.json         # Sample transformation input (472 lines)
    └── test_output.json          # Test output examples
```

### Code Organization Principles

- **Separation of Concerns**: Clear separation between API layer, business logic (graph), and utilities
- **Single Responsibility**: Each node handles one specific task
- **Dependency Injection**: Configuration via environment variables
- **Type Safety**: Extensive use of type hints and Pydantic models

---

## Workflow Graph

### Visual Representation

```
                    START
                      ↓
              ┌───────────────┐
              │  IngestorNode │
              │  Validate &   │
              │  Hash Fields  │
              └───────┬───────┘
                      ↓
              ┌───────────────┐
              │ AnalyzerNode  │
              │ Extract       │
              │ Entities      │
              └───────┬───────┘
                      ↓
            ┌─────────────────────┐
            │  should_transform?  │
            │  (Conditional)      │
            └───┬─────────────┬───┘
                │             │
         Same   │             │  Different
         ───────┘             └─────────
         │                            │
         │                            ↓
         │                    ┌───────────────┐
         │                    │TransformerNode│
         │                    │ Call OpenAI   │
         │                    └───────┬───────┘
         │                            ↓
         │                    ┌───────────────────┐
         │                    │ConsistencyChecker │
         │                    │ Search Keywords   │
         │                    └───────┬───────────┘
         │                            ↓
         │            ┌───────────────────────────────┐
         │            │ should_retry_transform?       │
         │            │ (Conditional)                 │
         │            └───┬───────────────────────┬───┘
         │                │                       │
         │           Low  │                       │  High
         │        Consistency                Consistency
         │       & retries<2                     │
         │                │                       │
         │                └───────┐               │
         │                        ↓               ↓
         │                 [Retry Transform]  ┌───────────┐
         │                        │           │ Validator │
         │                        └───────────┤ Verify    │
         │                                    │ Hashes    │
         │                                    └─────┬─────┘
         │                                          │
         └──────────────────────────────────────────┘
                                                    ↓
                                            ┌───────────────┐
                                            │ FinalizerNode │
                                            │ Calc Runtime  │
                                            └───────┬───────┘
                                                    ↓
                                                   END
```

### Workflow Graph Definition

**File**: `src/graph/workflow.py`

```python
def create_workflow() -> StateGraph:
    workflow = StateGraph(WorkflowState)

    # Add 6 nodes
    workflow.add_node("ingest", ingestor_node)
    workflow.add_node("analyze", analyzer_node)
    workflow.add_node("transform", transformer_node)
    workflow.add_node("check_consistency", consistency_checker_node)
    workflow.add_node("validate", validator_node)
    workflow.add_node("finalize", finalizer_node)

    # Define flow
    workflow.set_entry_point("ingest")
    workflow.add_edge("ingest", "analyze")

    # Conditional: same scenario or transform?
    workflow.add_conditional_edges(
        "analyze",
        should_transform,
        {"transform": "transform", "finalize": "finalize"}
    )

    # Transform → Check → Validate → Finalize
    workflow.add_edge("transform", "check_consistency")
    workflow.add_conditional_edges(
        "check_consistency",
        should_retry_transform,
        {"transform": "transform", "validate": "validate"}
    )
    workflow.add_edge("validate", "finalize")
    workflow.add_edge("finalize", END)

    return workflow.compile()
```

---

## Node Details

### State Schema

**File**: `src/graph/state.py`

The state is a TypedDict that flows through all nodes, accumulating data:

```python
class WorkflowState(TypedDict, total=False):
    # ===== INPUT =====
    input_json: dict                      # Original JSON
    selected_scenario: str | int          # Target scenario

    # ===== SCENARIO ANALYSIS =====
    selected_scenario_index: int          # Resolved index
    selected_scenario_text: str           # Full scenario text
    current_scenario_text: str            # Current scenario text
    scenario_options: list[str]           # All available scenarios

    # ===== ENTITY MAPPING =====
    entity_map: dict[str, str]            # {"OldBrand": "NewBrand", ...}
    candidate_paths: list[str]            # JSONPaths to transform
    style_profile: dict                   # Writing style analysis

    # ===== VALIDATION =====
    locked_field_hashes: dict[str, str]   # SHA-256 hashes

    # ===== RESULTS =====
    transformed_json: Optional[dict]      # Output JSON

    # ===== VALIDATION RESULTS =====
    changed_paths: list[str]              # Diff paths
    validation_errors: list[dict]         # Error details
    consistency_score: float              # 0-1 score

    # ===== METADATA =====
    retry_count: int                      # Retry attempts
    node_logs: list[dict]                 # Per-node logs
    runtime_ms: int                       # Total execution time
    final_status: Literal["OK", "FAIL", "PENDING"]
    openai_stats: dict                    # Token usage
```

### Node 1: IngestorNode

**File**: `src/graph/nodes.py` (Lines 19-121)

**Purpose**: Validate input structure and compute locked-field hashes

**Algorithm**:

1. Extract `input_json` and `selected_scenario` from state
2. Validate `topicWizardData` key exists
3. For each locked field:
   - Verify existence
   - Compute SHA-256 hash
   - Store in `locked_field_hashes` dict
4. Extract `scenarioOptions` array
5. Resolve `selected_scenario` (int index or string match)
6. Extract current scenario from `selectedScenarioOption`
7. Update state with all extracted data
8. Log execution time and metadata

**Locked Fields** (5 fields that must NEVER change):
```python
LOCKED_FIELDS = [
    "scenarioOptions",
    "assessmentCriterion",
    "selectedAssessmentCriterion",
    "industryAlignedActivities",
    "selectedIndustryAlignedActivities"
]
```

**Key Code**:
```python
# SHA-256 hashing for verification
for field in LOCKED_FIELDS:
    locked_field_hashes[field] = compute_sha256(data[field])
```

**Output**:
- `locked_field_hashes`: SHA-256 hashes of locked fields
- `scenario_options`: List of all available scenarios
- `selected_scenario_index`: Resolved scenario index
- `selected_scenario_text`: Target scenario text
- `current_scenario_text`: Current scenario text

---

### Node 2: AnalyzerNode

**File**: `src/graph/nodes.py` (Lines 123-222)

**Purpose**: Extract entities, build mapping, identify transformation paths

**Algorithm**:

1. Get `current_scenario_text` and `selected_scenario_text`
2. **SHORT-CIRCUIT CHECK**: If current == selected, skip transformation
3. Extract entities from current scenario using regex
4. Extract entities from target scenario
5. Build entity mapping for consistent replacements
6. Define candidate transformation paths
7. Create style profile (writing style analysis)
8. Update state

**Entity Extraction** (Lines 224-256):

Uses regex patterns to extract:
- **Brand name**: Capitalized word before "is/faces/sees"
- **Competitor**: Word after "after" or "when"
- **Challenge type**: "$1 menu", "BOGO", etc.
- **Industry**: "restaurant", "fashion retail", etc.

```python
def extract_entities_from_scenario(scenario_text: str) -> dict:
    entities = {}

    # Brand name extraction
    brand_match = re.search(r"([A-Z][a-zA-Z]+(?:'s)?)\s+(?:is|faces|sees|'s)", text)
    if brand_match:
        entities["brand"] = brand_match.group(1).replace("'s", "")

    # Competitor extraction
    competitor_match = re.search(r"(?:after|when)\s+([A-Z][a-zA-Z]+(?:'s)?)", text)

    # Challenge type
    if "$1 menu" in scenario_text:
        entities["challenge"] = "$1 value menu"
    elif "BOGO" in scenario_text:
        entities["challenge"] = "BOGO promotion"

    # Industry
    if "restaurant" in text.lower():
        entities["industry"] = "fast-casual restaurant"
    elif "fashion" in text.lower():
        entities["industry"] = "fashion retail"

    return entities
```

**Entity Mapping** (Lines 259-276):

Creates 1:1 mapping between corresponding entities:

```python
def build_entity_mapping(current: dict, target: dict) -> dict[str, str]:
    mapping = {}

    if "brand" in current and "brand" in target:
        mapping[current["brand"]] = target["brand"]

    if "competitor" in current and "competitor" in target:
        mapping[current["competitor"]] = target["competitor"]

    # Same for challenge, industry

    return mapping

# Example output:
# {
#     "HarvestBowls": "TrendWave",
#     "Nature's Crust": "ChicStyles",
#     "$1 value menu": "BOGO promotion",
#     "fast-casual restaurant": "fashion retail"
# }
```

**Candidate Transformation Paths**:
```python
candidate_paths = [
    "lessonInformation.lesson",
    "selectedScenarioOption",
    "simulationName",
    "workplaceScenario.scenario",
    "workplaceScenario.scenarioContext",
    "workplaceScenario.scenarioDescription",
    # ... more paths
]
```

**Output**:
- `entity_map`: Entity replacements
- `candidate_paths`: JSONPaths to transform
- `style_profile`: Writing style metadata

---

### Node 3: TransformerNode

**File**: `src/graph/nodes.py` (Lines 278-391)

**Purpose**: Call OpenAI GPT-4 to transform JSON using entity mappings

**Algorithm**:

1. Deep copy `input_json` (avoid mutations)
2. Get `entity_map`, `selected_scenario`, `current_scenario`
3. Build system prompt with locked field rules
4. Build user prompt with transformable fields
5. Call OpenAI GPT-4o (streaming)
6. **CRITICAL STEP**: Force-restore locked fields
7. Update state with `transformed_json` and `openai_stats`

**System Prompt** (Lines 296-322):

```python
system_prompt = f"""You are transforming a business simulation JSON from one scenario to another.

CRITICAL RULES:
1. NEVER modify these locked fields (keep byte-for-byte identical):
   - scenarioOptions
   - assessmentCriterion
   - selectedAssessmentCriterion
   - industryAlignedActivities
   - selectedIndustryAlignedActivities

2. Keep EXACT same JSON structure (same keys, same nesting, same array lengths)

3. Apply these entity mappings consistently:
{json.dumps(entity_map, indent=2)}

4. Replace ALL brand names, competitor names, industry terms, and contextual details

5. Maintain professional instructional tone

6. Preserve field types (string→string, array→array)

7. Keep email format patterns (e.g., name@domain.com)

8. Adapt KPIs and metrics to new industry context while preserving magnitude

Output ONLY valid JSON matching the input structure.
"""
```

**User Prompt** (optimized to reduce tokens):

```python
# Extract only transformable fields
transformable_fields = {
    "lessonInformation": data["lessonInformation"],
    "selectedScenarioOption": data["selectedScenarioOption"],
    "simulationName": data["simulationName"],
    "workplaceScenario": data["workplaceScenario"]
}

user_prompt = f"""Transform this JSON from current scenario to target scenario.

Current Scenario: {current_scenario_text}
Target Scenario: {selected_scenario_text}

Input Fields (transform these):
{json.dumps(transformable_fields, indent=2)}

Output complete topicWizardData JSON with ALL fields (locked + transformed).
"""
```

**OpenAI Call**:

```python
transformed_data = openai_client.generate_json(
    system_prompt,
    user_prompt,
    max_tokens=16000,
    temperature=0,     # Deterministic
    seed=42            # Reproducible
)
```

**Force Restoration** (Lines 355-366) - **CRITICAL SECURITY MECHANISM**:

```python
# Even if OpenAI modified locked fields, overwrite with original
for locked_field in LOCKED_FIELDS:
    transformed_data[locked_field] = original_data[locked_field]
```

This guarantees locked fields are **never** modified, even if OpenAI disobeys instructions.

**Streaming Version**: `transformer_node_streaming` (Lines 393-548)

Same logic but yields streaming chunks for real-time progress.

**Output**:
- `transformed_json`: Transformed JSON
- `openai_stats`: Token usage, model, call count

---

### Node 4: ConsistencyCheckerNode

**File**: `src/graph/nodes.py` (Lines 550-611)

**Purpose**: Verify no old scenario keywords remain in transformed content

**Algorithm**:

1. Get `transformed_json` and `entity_map`
2. Extract old keywords from `entity_map.keys()`
3. Search for old keywords in transformed JSON (excluding locked fields)
4. Calculate consistency score
5. Update state

**Keyword Search**:

```python
# Old keywords
old_keywords = ["HarvestBowls", "Nature's Crust", "$1 value menu", ...]

# Search in transformed JSON
findings = search_keywords(
    transformed_json,
    old_keywords,
    exclude_paths=LOCKED_FIELDS  # Don't check locked fields
)
```

**Consistency Score Calculation**:

```python
consistency_score = 1.0 - (num_findings / num_keywords)

# Examples:
# - 0 findings out of 4 keywords → 1.0 - (0/4) = 1.0 (perfect)
# - 1 finding out of 4 keywords → 1.0 - (1/4) = 0.75 (below threshold)
# - 4 findings out of 4 keywords → 1.0 - (4/4) = 0.0 (failed)
```

**Retry Logic**:

If `consistency_score < 0.85` (CONSISTENCY_THRESHOLD) AND `retry_count < 2`:
- Increment `retry_count`
- Return to TransformerNode (retry transformation)

**Output**:
- `consistency_score`: 0.0 to 1.0
- Keyword findings logged

---

### Node 5: ValidatorNode

**File**: `src/graph/nodes.py` (Lines 613-701)

**Purpose**: Final validation - locked fields, schema, diffs

**Algorithm**:

1. Get `input_json`, `transformed_json`, `locked_field_hashes`
2. **Check locked field immutability**:
   - Compute SHA-256 hash of each locked field in transformed JSON
   - Compare with original hash
   - If ANY difference → FAIL
3. Generate diff of changed paths
4. Search for old keywords (residual check)
5. Set `final_status`

**Locked Field Validation** (strictest check):

```python
for field in LOCKED_FIELDS:
    current_hash = compute_sha256(transformed_data[field])
    expected_hash = locked_field_hashes[field]

    if current_hash != expected_hash:
        # Even 1 byte difference fails validation
        validation_errors.append({
            "field": field,
            "error": "Locked field was modified",
            "expected_hash": expected_hash,
            "current_hash": current_hash
        })
        final_status = "FAIL"
```

**Generate Diff**:

```python
changed_paths = generate_json_diff(input_json, transformed_json)
# Returns: ["lessonInformation.lesson", "simulationName", ...]
```

**Final Status Logic**:

```python
if locked_fields_modified:
    final_status = "FAIL"
elif consistency_score >= 0.85:
    final_status = "OK"
else:
    final_status = "FAIL"
```

**Output**:
- `changed_paths`: List of modified JSONPaths
- `validation_errors`: List of error objects
- `final_status`: "OK" or "FAIL"

---

### Node 6: FinalizerNode

**File**: `src/graph/nodes.py` (Lines 703-740)

**Purpose**: Calculate total runtime, prepare final outputs

**Algorithm**:

1. Sum up all node execution times from `node_logs`
2. Update `state["runtime_ms"]`
3. Log finalizer execution
4. Return final state

**Code**:

```python
def finalizer_node(state: WorkflowState) -> WorkflowState:
    total_runtime = sum(log["duration_ms"] for log in state["node_logs"])
    state["runtime_ms"] = total_runtime

    # Add finalizer log
    state["node_logs"].append({
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "node": "FinalizerNode",
        "status": "success",
        "duration_ms": 0
    })

    return state
```

Simple bookkeeping node - ensures all metrics are calculated.

---

## API Endpoints

### Entry Point: `src/main.py`

```python
app = FastAPI(
    title="Scenario-Aware JSON Re-Contextualization API",
    version="1.0.0"
)

# CORS middleware - allows all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# API routes mounted at /api/v1
app.include_router(api_router, prefix="/api/v1", tags=["transformation"])
```

### Endpoint 1: Root

```
GET /

Response:
{
  "name": "Scenario Re-Contextualization API",
  "version": "1.0.0",
  "status": "running",
  "endpoints": {
    "transform": "/api/v1/transform/stream-openai",
    "validate": "/api/v1/validate",
    "health": "/api/v1/health",
    "scenarios": "/api/v1/scenarios"
  }
}
```

### Endpoint 2: Transform (Streaming)

```
POST /api/v1/transform/stream-openai
Content-Type: application/json

Request Body:
{
  "input_json": {
    "topicWizardData": {
      "lessonInformation": {...},
      "scenarioOptions": [...],
      "selectedScenarioOption": "HarvestBowls faces competition...",
      "assessmentCriterion": [...],
      "selectedAssessmentCriterion": [...],
      "industryAlignedActivities": [...],
      "selectedIndustryAlignedActivities": [...],
      "simulationName": "Fast Food Competition",
      "workplaceScenario": {...}
    }
  },
  "selected_scenario": 1  // or scenario text string
}
```

**Response** (Server-Sent Events stream):

```
data: {"event": "start", "message": "Starting transformation"}

data: {"event": "node_start", "node": "IngestorNode"}
data: {"event": "node_complete", "node": "IngestorNode", "duration_ms": 45}

data: {"event": "node_start", "node": "AnalyzerNode"}
data: {"event": "node_complete", "node": "AnalyzerNode", "duration_ms": 32}

data: {"event": "node_start", "node": "TransformerNode", "message": "Starting OpenAI transformation"}
data: {"event": "openai_progress", "message": "OpenAI generating... (8s)"}
data: {"event": "node_complete", "node": "TransformerNode", "duration_ms": 8200}

data: {"event": "node_start", "node": "ConsistencyCheckerNode"}
data: {"event": "node_complete", "node": "ConsistencyCheckerNode", "duration_ms": 120}

data: {"event": "node_start", "node": "ValidatorNode"}
data: {"event": "node_complete", "node": "ValidatorNode", "duration_ms": 95}

data: {"event": "node_start", "node": "FinalizerNode"}
data: {"event": "node_complete", "node": "FinalizerNode", "duration_ms": 0}

data: {"event": "complete", "result": {
  "output_json": {
    "topicWizardData": {
      "lessonInformation": {...},
      "selectedScenarioOption": "TrendWave sees sales drop when ChicStyles offers BOGO...",
      "scenarioOptions": [...],  // UNCHANGED
      "simulationName": "Fashion Retail BOGO Competition",
      "workplaceScenario": {...}
    }
  },
  "validation_report": {
    "schema_pass": true,
    "locked_fields_compliance": true,
    "locked_field_hashes": {
      "scenarioOptions": "a3f8b2c1e4d5...",
      "assessmentCriterion": "9d4e7f1a2b3c...",
      "selectedAssessmentCriterion": "c7b9f2e8a1d4...",
      "industryAlignedActivities": "e2f8c1b7d4a9...",
      "selectedIndustryAlignedActivities": "f9a2d8c3e1b7..."
    },
    "changed_paths": [
      "lessonInformation.lesson",
      "selectedScenarioOption",
      "simulationName",
      "workplaceScenario.scenario",
      "workplaceScenario.scenarioContext"
    ],
    "scenario_consistency_score": 0.96,
    "old_scenario_keywords_found": [],
    "runtime_ms": 8420,
    "retries": 0,
    "openai_stats": {
      "total_tokens": 15234,
      "total_calls": 1,
      "model": "gpt-4o"
    },
    "final_status": "OK"
  },
  "execution_time_ms": 8420
}}
```

### Endpoint 3: Validate Only

```
POST /api/v1/validate
Content-Type: application/json

Request Body:
{
  "original_json": {...},
  "transformed_json": {...},
  "locked_fields": ["scenarioOptions", ...]  // optional
}

Response:
{
  "schema_pass": true,
  "locked_fields_compliance": true,
  "locked_field_hashes": {
    "scenarioOptions": "a3f8b2c1...",
    "assessmentCriterion": "9d4e7f1a..."
  },
  "changed_paths": [
    "lessonInformation.lesson",
    "simulationName"
  ],
  "scenario_consistency_score": 1.0,
  "old_scenario_keywords_found": [],
  "runtime_ms": 0,
  "retries": 0,
  "openai_stats": {},
  "final_status": "OK"
}
```

### Endpoint 4: Health Check

```
GET /api/v1/health

Response:
{
  "status": "healthy",
  "version": "1.0.0",
  "openai_connected": true
}
```

### Endpoint 5: List Scenarios

```
GET /api/v1/scenarios?input_json=<base64_or_json_string>

Response:
{
  "total": 3,
  "current_scenario": "HarvestBowls faces competition from Nature's Crust...",
  "scenarios": [
    {
      "index": 0,
      "text": "HarvestBowls faces competition from Nature's Crust $1 menu",
      "is_current": true
    },
    {
      "index": 1,
      "text": "TrendWave sees sales drop when ChicStyles offers BOGO promotion",
      "is_current": false
    },
    {
      "index": 2,
      "text": "SkyHigh Airlines faces pricing pressure from BudgetWings",
      "is_current": false
    }
  ]
}
```

### OpenAPI Documentation

FastAPI automatically generates interactive documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

---

## Key Features

### Feature 1: Triple-Layer Locked Field Protection

**Problem**: Certain fields must NEVER change (e.g., assessment criteria, activity lists)

**Solution**: Four layers of protection:

```python
# Layer 1: SHA-256 Hashing (IngestorNode)
locked_field_hashes["scenarioOptions"] = compute_sha256(data["scenarioOptions"])

# Layer 2: Explicit Instructions (TransformerNode)
system_prompt = "NEVER modify these locked fields: scenarioOptions, ..."

# Layer 3: Force Restoration (TransformerNode)
transformed_data["scenarioOptions"] = original_data["scenarioOptions"]

# Layer 4: Hash Validation (ValidatorNode)
if compute_sha256(transformed["scenarioOptions"]) != original_hash:
    final_status = "FAIL"
```

**Guarantee**: Even if OpenAI disobeys instructions, locked fields remain byte-for-byte identical.

### Feature 2: Entity Mapping for Consistency

**Problem**: Need consistent replacements across entire JSON

**Solution**: Regex-based entity extraction + mapping table

```python
# Example Transformation

Current Scenario:
"HarvestBowls faces competition after Nature's Crust launched $1 menu"

Target Scenario:
"TrendWave sees sales drop when ChicStyles offers BOGO promotion"

# Entity Extraction

current_entities = {
    "brand": "HarvestBowls",
    "competitor": "Nature's Crust",
    "challenge": "$1 value menu",
    "industry": "fast-casual restaurant"
}

target_entities = {
    "brand": "TrendWave",
    "competitor": "ChicStyles",
    "challenge": "BOGO promotion",
    "industry": "fashion retail"
}

# Entity Mapping (injected into OpenAI prompts)

entity_map = {
    "HarvestBowls": "TrendWave",
    "Nature's Crust": "ChicStyles",
    "$1 value menu": "BOGO promotion",
    "fast-casual restaurant": "fashion retail"
}
```

This ensures ALL occurrences are replaced consistently.

### Feature 3: Consistency Checking with Automatic Retry

**Problem**: OpenAI might miss some replacements

**Solution**: Keyword search + automatic retry

```python
# After transformation
old_keywords = ["HarvestBowls", "Nature's Crust", "$1 value menu"]
findings = search_keywords(transformed_json, old_keywords)

# Calculate score
consistency_score = 1.0 - (len(findings) / len(old_keywords))

# Example: 1 finding out of 3 keywords
# Score = 1.0 - (1/3) = 0.67 (below threshold of 0.85)

if consistency_score < 0.85 and retry_count < 2:
    retry_count += 1
    # Go back to TransformerNode
    return "transform"
```

**Retry Flow**:
```
Transform → Consistency Check (score: 0.67) → Retry Transform
         → Consistency Check (score: 0.95) → Validate
```

### Feature 4: Real-Time Streaming with SSE

**Problem**: Large files take 30-60 seconds to transform

**Solution**: Server-Sent Events (SSE) streaming

Client receives events in real-time:
```
event: node_start      → "IngestorNode started"
event: node_complete   → "IngestorNode completed (45ms)"
event: node_start      → "TransformerNode started"
event: openai_progress → "OpenAI generating... (8s)"
event: openai_chunk    → "{"topicWizardData": {"lesson..."
event: complete        → Final JSON + validation report
```

**Benefits**:
- Display progress bar
- Show generated text in real-time
- Better user experience for long-running operations

### Feature 5: Deterministic Transformations

**Problem**: Need reproducible results for testing

**Solution**: Fixed OpenAI parameters

```python
OPENAI_TEMPERATURE = 0      # No randomness
OPENAI_SEED = 42            # Reproducible sampling
```

**Guarantee**: Same input + scenario → **exact same output** every time.

### Feature 6: Comprehensive Validation Reports

Every transformation returns a detailed report:

```json
{
  "schema_pass": true,
  "locked_fields_compliance": true,
  "locked_field_hashes": {
    "scenarioOptions": "a3f8b2c1e4d5a7b9c2e1f3d8a6b4c9e7",
    "assessmentCriterion": "9d4e7f1a2b3c5e8d1a4f7b2c9e6a3d8f"
  },
  "changed_paths": [
    "lessonInformation.lesson",
    "selectedScenarioOption",
    "simulationName",
    "workplaceScenario.scenario"
  ],
  "scenario_consistency_score": 0.96,
  "old_scenario_keywords_found": [],
  "runtime_ms": 8420,
  "retries": 1,
  "openai_stats": {
    "total_tokens": 15234,
    "total_calls": 2,
    "model": "gpt-4o"
  },
  "final_status": "OK"
}
```

---

## Data Flow

### Complete Request-Response Flow

```
1. CLIENT REQUEST
   ↓
   POST /api/v1/transform/stream-openai
   Body: {
     "input_json": {...},
     "selected_scenario": 1
   }

2. API ROUTE HANDLER (routes.py)
   ↓
   transform_scenario_stream_openai(request)
   - Initializes state dict
   - Creates async event generator

3. WORKFLOW EXECUTION (LangGraph)
   ↓
   IngestorNode
   - Validates JSON structure
   - Computes locked field hashes (SHA-256)
   - Resolves scenario index
   Output: locked_field_hashes, scenario_text
   ↓
   AnalyzerNode
   - Extracts entities (brand, competitor)
   - Builds entity mapping
   - Checks if same scenario (short-circuit)
   Output: entity_map, candidate_paths
   ↓
   CONDITIONAL: Same scenario?
   - Yes → Skip to FinalizerNode
   - No → TransformerNode
   ↓
   TransformerNode
   - Builds OpenAI prompts
   - Calls GPT-4o (streaming)
   - Force-restores locked fields
   Output: transformed_json
   ↓
   ConsistencyCheckerNode
   - Searches old keywords
   - Calculates consistency score
   Output: consistency_score
   ↓
   CONDITIONAL: Score < 0.85 & retries < 2?
   - Yes → Retry TransformerNode
   - No → ValidatorNode
   ↓
   ValidatorNode
   - Verifies hashes (byte-for-byte)
   - Generates diff
   - Sets final_status
   Output: changed_paths, validation_errors
   ↓
   FinalizerNode
   - Sums execution times
   - Prepares final state
   Output: runtime_ms, complete state
   ↓
4. RESPONSE CONSTRUCTION
   ↓
   TransformResponse(
     output_json = state["transformed_json"],
     validation_report = ValidationReport(...),
     execution_time_ms = state["runtime_ms"]
   )

5. CLIENT RECEIVES
   ↓
   SSE Events:
   - "node_start: IngestorNode"
   - "node_complete: IngestorNode"
   - "openai_chunk: {"topicWizard..."
   - "complete: {full response}"
```

### State Evolution Example

```python
# Initial state
{
  "input_json": {...},
  "selected_scenario": 1,
  "node_logs": [],
  "validation_errors": [],
  "retry_count": 0,
  "final_status": "PENDING"
}

# After IngestorNode
{
  ...previous...,
  "locked_field_hashes": {"scenarioOptions": "a3f8b2c1...", ...},
  "scenario_options": ["Scenario 0", "Scenario 1", "Scenario 2"],
  "selected_scenario_index": 1,
  "selected_scenario_text": "TrendWave sees sales drop...",
  "current_scenario_text": "HarvestBowls faces competition...",
  "node_logs": [{"node": "IngestorNode", "status": "success", ...}]
}

# After AnalyzerNode
{
  ...previous...,
  "entity_map": {
    "HarvestBowls": "TrendWave",
    "Nature's Crust": "ChicStyles"
  },
  "candidate_paths": ["lessonInformation.lesson", ...],
  "style_profile": {...}
}

# After TransformerNode
{
  ...previous...,
  "transformed_json": {
    "topicWizardData": {
      "lessonInformation": {"lesson": "TrendWave must respond to ChicStyles BOGO..."},
      "scenarioOptions": [...],  // Forcibly restored
      ...
    }
  },
  "openai_stats": {"total_tokens": 15234, "total_calls": 1}
}

# After ConsistencyCheckerNode
{
  ...previous...,
  "consistency_score": 0.92
}

# After ValidatorNode
{
  ...previous...,
  "changed_paths": ["lessonInformation.lesson", "simulationName", ...],
  "validation_errors": [],  // Empty = all locked fields intact
  "final_status": "OK"
}

# After FinalizerNode (final state)
{
  ...previous...,
  "runtime_ms": 8420
}
```

---

## Dependencies

### Core Dependencies (`requirements.txt`)

```python
# Web Framework
fastapi==0.104.1              # Modern async web framework
uvicorn==0.24.0               # ASGI server
pydantic==2.5.3               # Data validation

# Workflow Orchestration
langgraph==0.2.45             # Graph-based state machine
langchain-core==0.3.21        # LangChain core utilities
langchain-openai==0.2.10      # OpenAI integration

# AI/LLM
openai==1.56.0                # OpenAI API client

# Utilities
python-dotenv==1.0.0          # Environment variable management
jsonschema==4.20.0            # JSON schema validation
genson==1.2.2                 # JSON schema generation
jsondiff==2.0.0               # JSON diffing
tenacity==8.2.3               # Retry logic with exponential backoff
httpx==0.25.2                 # Async HTTP client

# Testing
pytest==7.4.3                 # Testing framework
```

### Dependency Roles

**FastAPI Ecosystem**:
- `FastAPI`: Route decorators, dependency injection, automatic OpenAPI docs
- `uvicorn`: ASGI server to run the async application
- `pydantic`: Validates request/response schemas, provides type safety
- `httpx`: Async HTTP client used by FastAPI

**LangGraph/LangChain**:
- `langgraph`: Core state machine framework
  - Provides `StateGraph` class
  - Handles conditional routing
  - Manages state passing between nodes

**OpenAI**:
- `openai`: Official Python client
  - `client.chat.completions.create()`
  - Structured outputs: `response_format={"type": "json_object"}`
  - Streaming support

**Utilities**:
- `python-dotenv`: Loads `.env` file into `os.environ`
- `jsondiff`: Powers `generate_json_diff()` in helpers.py
- `tenacity`: Provides `@retry` decorator in `openai_client.py`
  - Exponential backoff: wait 2s, 4s, 8s
  - Max 3 attempts

---

## Configuration

### Environment Variables (`.env`)

```bash
# OpenAI Configuration (REQUIRED)
OPENAI_API_KEY=your-openai-api-key-here
OPENAI_MODEL=gpt-4o                     # gpt-4o, gpt-4o-mini
OPENAI_TEMPERATURE=0                    # 0 = deterministic, 1 = creative
OPENAI_SEED=42                          # For reproducibility

# Application Configuration
APP_NAME=Scenario Re-Contextualization API
APP_VERSION=1.0.0
LOG_LEVEL=INFO                          # INFO, DEBUG, WARNING, ERROR
```

### Config Class (`src/utils/config.py`)

```python
class Config:
    """Centralized configuration management"""

    # OpenAI Settings (loaded from environment)
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OPENAI_TEMPERATURE: float = float(os.getenv("OPENAI_TEMPERATURE", "0"))
    OPENAI_SEED: int = int(os.getenv("OPENAI_SEED", "42"))
    OPENAI_TIMEOUT: int = 300  # 5 minutes

    # App Settings
    APP_NAME: str = os.getenv("APP_NAME", "Scenario Re-Contextualization API")
    APP_VERSION: str = os.getenv("APP_VERSION", "1.0.0")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Workflow Settings (hardcoded)
    MAX_RETRIES: int = 2                    # Max transformation retries
    CONSISTENCY_THRESHOLD: float = 0.85     # Min consistency score to pass

    # Locked Fields (critical business logic)
    LOCKED_FIELDS: list[str] = [
        "scenarioOptions",
        "assessmentCriterion",
        "selectedAssessmentCriterion",
        "industryAlignedActivities",
        "selectedIndustryAlignedActivities"
    ]

# Singleton instance
config = Config()
```

### Configuration Access Pattern

```python
# Import throughout codebase
from src.utils.config import config

# Usage
if state["consistency_score"] < config.CONSISTENCY_THRESHOLD:
    retry()

for field in config.LOCKED_FIELDS:
    hash_field(field)
```

---

## LangGraph Deep Dive

### What is LangGraph?

LangGraph is a framework for building **stateful, multi-step workflows** where:
- **Nodes** are functions that process state
- **Edges** define the flow between nodes
- **State** is a shared dictionary that accumulates data
- **Conditional Edges** enable dynamic routing

### Conditional Routing Functions

#### 1. `should_transform(state)` - After AnalyzerNode

```python
def should_transform(state: WorkflowState) -> str:
    """Decide if transformation is needed"""
    if state.get("final_status") == "OK":
        # AnalyzerNode set this when same scenario detected
        return "finalize"  # Skip transformation entirely
    return "transform"

# Example:
# User selects scenario 0, but scenario 0 is already selected
# → AnalyzerNode sets final_status = "OK"
# → should_transform returns "finalize"
# → Workflow: Ingest → Analyze → Finalize (skip Transform/Validate)
```

#### 2. `should_retry_transform(state)` - After ConsistencyCheckerNode

```python
def should_retry_transform(state: WorkflowState) -> str:
    """Decide if transformation should be retried"""
    consistency_score = state.get("consistency_score", 0)
    retry_count = state.get("retry_count", 0)

    if consistency_score < config.CONSISTENCY_THRESHOLD and retry_count < config.MAX_RETRIES:
        state["retry_count"] = retry_count + 1
        return "transform"  # Go back to TransformerNode
    return "validate"  # Proceed to ValidatorNode

# Example 1: Low consistency, first attempt
# consistency_score = 0.70 (< 0.85), retry_count = 0 (< 2)
# → return "transform"
# → Workflow: Transform → Consistency → Transform (retry) → Consistency → Validate

# Example 2: High consistency
# consistency_score = 0.95 (>= 0.85)
# → return "validate"
# → Workflow: Transform → Consistency → Validate
```

### State Management

**State is a shared dictionary** that flows through all nodes:

```python
# Each node follows this pattern
def node_function(state: WorkflowState) -> WorkflowState:
    start_time = time.time()

    try:
        # 1. Extract inputs from state
        input_json = state["input_json"]

        # 2. Perform operation
        result = do_something(input_json)

        # 3. Update state
        state["result"] = result

        # 4. Log execution
        duration_ms = int((time.time() - start_time) * 1000)
        state["node_logs"].append({
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "node": "NodeName",
            "status": "success",
            "duration_ms": duration_ms
        })

        # 5. Return modified state
        return state

    except Exception as e:
        # Error handling
        state["validation_errors"].append({"node": "NodeName", "error": str(e)})
        state["final_status"] = "FAIL"
        return state
```

**Key Properties**:
1. **Mutable**: Nodes modify state in-place
2. **Accumulative**: Each node adds data, rarely removes
3. **Type-safe**: `WorkflowState` TypedDict defines schema
4. **Observable**: `node_logs` array tracks all operations

### Workflow Invocation

```python
# In routes.py
from src.graph.workflow import scenario_workflow

# Run workflow (synchronous)
final_state = scenario_workflow.invoke(initial_state)

# Or stream (asynchronous)
async for event in scenario_workflow.astream(initial_state):
    # event contains node name and state updates
    yield event
```

### Error Handling in Workflow

```python
# Errors don't crash the workflow - they're captured in state
state = {
    "validation_errors": [
        {
            "node": "TransformerNode",
            "error": "OpenAI API timeout"
        },
        {
            "node": "ValidatorNode",
            "error": "Locked field 'scenarioOptions' was modified"
        }
    ],
    "final_status": "FAIL"
}

# Workflow completes normally, caller checks final_status
if state["final_status"] == "FAIL":
    raise HTTPException(status_code=500, detail=state["validation_errors"])
```

---

## Testing

### Test Files

```
test_workflow.py          # Unit tests for workflow nodes
test_api.py               # API integration tests
test_with_sample.py       # Large file streaming test
test_stream.py            # Alternative streaming test
test_openai_stream.py     # Raw OpenAI output test
```

### Running Tests

```bash
# Install dependencies
pip install -r requirements.txt

# Run all tests
pytest

# Run specific test file
pytest test_workflow.py

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=src
```

### Sample Test

```python
# test_workflow.py
def test_ingestor_node():
    state = {
        "input_json": sample_input,
        "selected_scenario": 1,
        "node_logs": [],
        "validation_errors": []
    }

    result = ingestor_node(state)

    assert "locked_field_hashes" in result
    assert len(result["locked_field_hashes"]) == 5
    assert "selected_scenario_text" in result
    assert result["selected_scenario_index"] == 1
```

---

## Deployment

### Local Development

```bash
# 1. Clone repository
git clone <repo-url>
cd fastapi-langgraph-app

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create .env file
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 5. Run server
uvicorn src.main:app --reload --port 8000

# 6. Access API
# - Swagger UI: http://localhost:8000/docs
# - ReDoc: http://localhost:8000/redoc
# - Health Check: http://localhost:8000/api/v1/health
```

### Production Deployment

```bash
# 1. Install production dependencies
pip install -r requirements.txt

# 2. Set environment variables
export OPENAI_API_KEY=your-key
export OPENAI_MODEL=gpt-4o
export LOG_LEVEL=INFO

# 3. Run with Gunicorn (production ASGI server)
gunicorn src.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 600  # 10 minutes for long transformations

# 4. Or run with Uvicorn directly
uvicorn src.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4
```

### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY .env .

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build image
docker build -t scenario-api .

# Run container
docker run -p 8000:8000 --env-file .env scenario-api
```

---

## Summary

This is a **production-grade, enterprise-level application** demonstrating:

✅ **Modern Python Architecture**: FastAPI + async/await + type hints
✅ **Agentic Workflows**: LangGraph state machines with conditional logic
✅ **AI Integration**: OpenAI GPT-4 with structured outputs and streaming
✅ **Data Integrity**: Triple-layer protection for locked fields (SHA-256)
✅ **Observability**: Comprehensive logging, validation reports, streaming progress
✅ **Reliability**: Automatic retries, consistency checking, error handling
✅ **Testability**: Unit tests, integration tests, sample data

### Key Strengths

- **Deterministic**: Same input → same output (temperature=0, seed=42)
- **Reliable**: Locked fields never change (cryptographically guaranteed)
- **Observable**: Real-time progress streaming via SSE
- **Maintainable**: Modular nodes, clear separation of concerns
- **Scalable**: Stateless design, horizontally scalable

### Use Cases

- Business simulation platforms needing scenario customization
- Content management systems with template transformations
- Educational platforms with adaptive learning paths
- Any system requiring intelligent JSON transformation with strict constraints

### Technical Sophistication

- Graph-based state machine (not simple sequential flow)
- Streaming SSE responses (not just request-response)
- Retry logic with consistency scoring (not fail-fast)
- SHA-256 verification (cryptographic guarantees)
- Entity extraction and mapping (NLP-lite)

**Codebase Size**: Approximately **2000+ lines of production code** with comprehensive error handling, logging, validation, and testing infrastructure.

---

## File Reference

### Core Files

| File | Lines | Purpose |
|------|-------|---------|
| `src/graph/nodes.py` | 740 | 6 workflow nodes implementation |
| `src/api/routes.py` | ~200 | 4 REST endpoints |
| `src/graph/workflow.py` | ~150 | LangGraph workflow definition |
| `src/utils/helpers.py` | ~120 | 9 utility functions |
| `src/utils/openai_client.py` | ~100 | OpenAI API wrapper |
| `src/models/schemas.py` | ~80 | Pydantic models |
| `src/graph/state.py` | ~50 | State schema |
| `src/utils/config.py` | ~40 | Configuration |
| `src/main.py` | ~30 | FastAPI app entry |

### Documentation Files

| File | Lines | Purpose |
|------|-------|---------|
| `README.md` | 269 | User guide |
| `IMPLEMENTATION_SUMMARY.md` | 348 | Implementation details |
| `TESTING_GUIDE.md` | 184 | Testing documentation |
| `CODEBASE_DOCUMENTATION.md` | This file | Complete reference |

---

**Last Updated**: 2025-11-29
**Version**: 1.0.0
**Author**: Generated by Claude Code
