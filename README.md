# Scenario-Aware JSON Re-Contextualization POC

A FastAPI + LangGraph + OpenAI solution for transforming business simulation JSON between different scenarios while preserving structure and locked fields.

## 🎯 Guarantees

✅ **Locked fields remain byte-for-byte identical**  
   - `scenarioOptions`, `assessmentCriterion`, `selectedAssessmentCriterion`, `industryAlignedActivities`, `selectedIndustryAlignedActivities`

✅ **JSON schema/structure preserved exactly**  
   - Same keys, same nesting depth, same array lengths

✅ **Deterministic transformations**  
   - Same input + scenario → same output (reproducible)

✅ **Validation report included**  
   - Schema validation, locked-field compliance, consistency scoring

✅ **Fast execution**  
   - Target: < 10 seconds for typical inputs

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- OpenAI API key

### Installation

```bash
# Clone and navigate to directory
cd fastapi-langgraph-app

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### Run the API

```bash
# Start the server
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

# Server will run at http://localhost:8000
```

## 📚 API Endpoints

### POST `/api/v1/transform`
Transform JSON to a different scenario.

**Request:**
```json
{
  "input_json": { /* full topicWizardData JSON */ },
  "selected_scenario": 3,  // or scenario text string
  "locked_fields": ["scenarioOptions", ...]  // optional
}
```

**Response:**
```json
{
  "output_json": { /* transformed JSON */ },
  "validation_report": {
    "schema_pass": true,
    "locked_fields_compliance": true,
    "changed_paths": ["lessonInformation.lesson", ...],
    "scenario_consistency_score": 0.96,
    "runtime_ms": 8420,
    "retries": 1,
    "final_status": "OK"
  },
  "execution_time_ms": 8420
}
```

### POST `/api/v1/validate`
Validate an already-transformed JSON.

**Request:**
```json
{
  "original_json": { /* original */ },
  "transformed_json": { /* transformed */ }
}
```

### GET `/api/v1/health`
Check API and OpenAI connectivity.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "openai_connected": true
}
```



### GET `/api/v1/scenarios`
List available scenarios from input JSON.

## 🧠 How It Works

### Architecture
- **FastAPI**: RESTful API with async endpoints
- **LangGraph**: Stateful workflow with conditional routing
- **OpenAI GPT-4o**: JSON transformation with structured outputs

### Workflow

```
START → Ingestor → Analyzer → Transformer → ConsistencyChecker → Validator → Finalizer → END
                       ↓                           ↓
                  [same scenario?]          [retry if needed]
```

### Nodes

1. **IngestorNode**: Parse JSON, validate structure, compute locked-field hashes
2. **AnalyzerNode**: Extract scenarios, build entity mapping, identify transformation paths
3. **TransformerNode**: Use OpenAI to transform content with entity mapping
4. **ConsistencyCheckerNode**: Verify cross-field consistency, search for old keywords
5. **ValidatorNode**: Schema validation, locked-field checks, diff generation
6. **FinalizerNode**: Generate reports and outputs

### Locked Fields
These fields are **never modified**:
- `scenarioOptions`
- `assessmentCriterion`
- `selectedAssessmentCriterion`
- `industryAlignedActivities`
- `selectedIndustryAlignedActivities`

## 📊 Validation Report

Each transformation includes a comprehensive validation report:

- **schema_pass**: Boolean indicating schema validity
- **locked_fields_compliance**: Boolean for locked-field immutability
- **locked_field_hashes**: SHA-256 hashes of all locked fields
- **changed_paths**: List of JSONPaths that were modified
- **scenario_consistency_score**: 0-1 score for cross-field consistency
- **old_scenario_keywords_found**: List of residual old scenario terms
- **runtime_ms**: Total execution time
- **retries**: Number of retry attempts
- **openai_stats**: Token usage and API call statistics
- **final_status**: "OK" or "FAIL"

## 🧪 Example Usage

### Python
```python
import requests

url = "http://localhost:8000/api/v1/transform"
payload = {
    "input_json": {...},  # Your topicWizardData JSON
    "selected_scenario": 3  # Target scenario index
}

response = requests.post(url, json=payload)
result = response.json()

print(f"Status: {result['validation_report']['final_status']}")
print(f"Changed {len(result['validation_report']['changed_paths'])} fields")
```

### cURL
```bash
curl -X POST "http://localhost:8000/api/v1/transform" \
  -H "Content-Type: application/json" \
  -d @request.json
```

## 🔧 Configuration

Edit `.env` file:

```env
OPENAI_API_KEY=your-key-here
OPENAI_MODEL=gpt-4o
OPENAI_TEMPERATURE=0
OPENAI_SEED=42
```

## 📝 Sample Transformation

**From** (Scenario 0): HarvestBowls vs Nature's Crust ($1 menu - Fast Casual Food)  
**To** (Scenario 3): TrendWave vs ChicStyles (BOGO Promotion - Fashion Retail)

**Changed Fields**: ~37 paths including:
- `simulationName`
- `workplaceScenario.*`
- All email bodies
- Resource content
- Rubrics and guidelines

**Preserved**: All 5 locked fields remain byte-identical

## 🛠️ Development

### Run Tests
```bash
pytest test_workflow.py -v
```

### Project Structure
```
fastapi-langgraph-app/
├── src/
│   ├── main.py              # FastAPI app
│   ├── api/
│   │   └── routes.py        # API endpoints
│   ├── graph/
│   │   ├── state.py         # State schema
│   │   ├── nodes.py         # Node implementations
│   │   └── workflow.py      # LangGraph workflow
│   ├── models/
│   │   └── schemas.py       # Pydantic models
│   └── utils/
│       ├── config.py        # Configuration
│       ├── helpers.py       # Utility functions
│       └── openai_client.py # OpenAI wrapper
├── requirements.txt
├── .env.example
└── README.md
```

## ⚙️ Advanced Features

### Retry Logic
- Automatic retry for consistency issues (max 2 retries)
- Exponential backoff for OpenAI API calls

### Determinism
- Fixed temperature (0) and seed (42)
- Reproducible outputs for same inputs

### Observability
- Structured logging for each node
- Token usage tracking
- Execution time metrics

## 🔒 Security

- API keys stored in environment variables
- CORS middleware configured
- Input validation with Pydantic

## 📄 License

MIT License - Use as needed for your projects.

## 🤝 Support

For issues or questions, refer to the code documentation or raise an issue.

---

**Built with FastAPI, LangGraph, and OpenAI GPT-4o**
