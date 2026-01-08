# JSON Recontextualization

A FastAPI + LangGraph pipeline for transforming simulation JSON between different business scenarios while preserving structure and alignment.

## 7-Stage Pipeline

```
Sharder → Adaptation → Alignment → Validation → Fixers → Merger → Finisher
```

1. **Sharder**: Splits JSON into manageable shards
2. **Adaptation**: Gemini 2.5 Flash transforms each shard to target scenario
3. **Alignment**: GPT validates cross-shard consistency (9 alignment rules)
4. **Validation**: Schema and content validation
5. **Fixers**: Auto-fixes validation issues
6. **Merger**: Reassembles shards into final JSON
7. **Finisher**: Generates output and reports

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Add your API keys to .env

# Run server
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

## Environment Variables

```env
GEMINI_API_KEY=your-gemini-key
OPENAI_API_KEY=your-openai-key
LANGSMITH_API_KEY=your-langsmith-key (optional)
```

## API Endpoints

### POST `/api/v1/pipeline`
Run full transformation pipeline.

```json
{
  "input_json": { ... },
  "scenario_prompt": "Fashion retail BOGO promotion"
}
```

### POST `/api/v1/pipeline/stream`
Run pipeline with streaming progress updates (SSE).

### GET `/api/v1/health`
Health check endpoint.

## Key Features

- Parallel shard adaptation with Gemini 2.5 Flash
- Cross-shard alignment checking with GPT
- Auto-regeneration of failed shards
- Duplicate detection and removal
- Empty entry cleanup
- Industry-aware RAG context

## Project Structure

```
src/
├── api/routes.py          # API endpoints
├── graph/
│   ├── nodes.py           # Pipeline stages
│   ├── state.py           # State schema
│   └── workflow.py        # LangGraph workflow
├── stages/
│   ├── sharder.py         # JSON sharding
│   ├── adaptation_engine.py
│   ├── alignment_checker.py
│   └── fixers.py
├── utils/
│   ├── gemini_client.py   # Gemini API wrapper
│   ├── openai_client.py   # OpenAI API wrapper
│   └── content_processor.py
└── main.py
```

## Deployment

Works with Railway, Render, Fly.io, or any platform supporting Python.

```bash
# Railway
railway up

# Or push to GitHub and connect to Railway
git push origin main
```
