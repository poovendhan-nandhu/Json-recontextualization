# JSON Recontextualization

A FastAPI + LangGraph pipeline for transforming simulation JSON between different business scenarios while preserving structure and alignment.

## Pipeline Architecture

```
ADAPT → VALIDATE → REPAIR (loop) → FINALIZE
```

1. **ADAPT**: Gemini 2.5 Flash transforms JSON to target scenario (parallel shards)
2. **VALIDATE**: 8 GPT validators check quality in parallel
3. **REPAIR**: Fix issues and loop back to validate (max 2 iterations)
4. **FINALIZE**: Return final result with scores

### Validators

| Validator | What it checks |
|-----------|----------------|
| Domain Fidelity | Correct industry terminology |
| Context Fidelity | Goal/challenge preserved |
| Resource Quality | Data not answers, word count |
| KLO-Question Alignment | Questions map to KLOs |
| Consistency | Names/companies consistent |
| Completeness | No missing content |
| KLO-Resource Alignment | Resources support KLOs |
| Solvability | Questions answerable from resources |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Add your API keys to .env

# Run server
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

## Environment Variables

```env
GEMINI_API_KEY=your-gemini-key
OPENAI_API_KEY=your-openai-key
LANGSMITH_API_KEY=your-langsmith-key (optional)
```

## API Endpoints

### WebSocket `/api/v1/pipeline/ws`
Full pipeline with real-time streaming (recommended for production).

```javascript
const ws = new WebSocket('ws://localhost:8000/api/v1/pipeline/ws');
ws.onopen = () => {
    ws.send(JSON.stringify({
        input_json: myJson,
        scenario_prompt: "learners will act as consultants..."
    }));
};
ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    console.log(msg.type, msg);
};
```

**Message Types:**
- `connected` - Connection established
- `stage_complete` - Stage finished (adapt/validate/repair)
- `validation_result` - Score, issues, agent_scores
- `result` - Final JSON and metrics
- `error` - Error with traceback

### POST `/api/v1/adapt/simple`
Run adaptation only (no validation/repair).

```json
{
  "input_json": { ... },
  "scenario_prompt": "Fashion retail BOGO promotion"
}
```

### WebSocket `/api/v1/adapt/simple/ws`
Adaptation only with shard-level progress streaming.

### GET `/api/v1/health`
Health check endpoint.

## Key Features

- **Parallel shard adaptation** with Gemini 2.5 Flash
- **8 parallel validators** with GPT
- **Automatic repair loop** (max 2 iterations)
- **RAG fact retrieval** for resource enrichment
- **Entity mapping** (protagonist vs competitor distinction)
- **Forbidden term cleanup** (no source scenario leakage)
- **WebSocket streaming** for real-time progress

## Validation Targets

| Metric | Target |
|--------|--------|
| Overall Score | ≥95% |
| Solvability | 100% |
| Source Term Leakage | 0 |

## Project Structure

```
src/
├── api/routes.py              # API endpoints + WebSocket
├── graph/
│   ├── nodes.py               # LangGraph pipeline nodes
│   ├── state.py               # Pipeline state schema
│   └── workflow.py            # Workflow exports
├── stages/
│   ├── simple_adapter.py      # Gemini adaptation engine
│   ├── simple_validators.py   # 8 validators + repair
│   └── sharder.py             # JSON sharding/merging
├── rag/
│   └── fact_retriever.py      # RAG for resource facts
├── utils/
│   └── config.py              # Configuration
└── main.py                    # FastAPI app
```

## Testing

```bash
# Run full pipeline test
python test_simple_adapter.py

# Test via API (server must be running)
python test_simple_adapter.py --api
```

## Deployment

Works with Railway, Render, Fly.io, or any platform supporting Python.

```bash
# Start production server
uvicorn src.main:app --host 0.0.0.0 --port 8000
```
