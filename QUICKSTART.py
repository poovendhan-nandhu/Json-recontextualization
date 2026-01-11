"""
Quick Start Guide for Scenario Re-Contextualization POC
"""

# Installation Steps
print("=" * 60)
print("SCENARIO RE-CONTEXTUALIZATION POC - QUICK START")
print("=" * 60)
print()

print("📦 STEP 1: Install Dependencies")
print("-" * 60)
print("Run: pip install -r requirements.txt")
print()

print("🔑 STEP 2: Configure OpenAI API Key")
print("-" * 60)
print("1. Copy .env.example to .env")
print("2. Edit .env and add your OPENAI_API_KEY")
print()

print("🚀 STEP 3: Start the API Server")
print("-" * 60)
print("Run: uvicorn src.main:app --reload")
print("Server will be at: http://localhost:8000")
print()

print("🧪 STEP 4: Test the API")
print("-" * 60)
print("Visit: http://localhost:8000/docs")
print("Or use the example below:")
print()

print("""
Example Python Usage:
---------------------
import requests
import json

# Load sample input
with open('sample_input.json', 'r') as f:
    input_data = json.load(f)

# Transform to scenario 3 (Fashion Retail)
response = requests.post(
    'http://localhost:8000/api/v1/transform',
    json={
        'input_json': input_data,
        'selected_scenario': 3
    }
)

result = response.json()
print(f"Status: {result['validation_report']['final_status']}")
print(f"Changed Fields: {len(result['validation_report']['changed_paths'])}")
print(f"Consistency Score: {result['validation_report']['scenario_consistency_score']}")

# Save output
with open('output.json', 'w') as f:
    json.dump(result['output_json'], f, indent=2)
""")

print()
print("=" * 60)
print("✅ IMPLEMENTATION COMPLETE!")
print("=" * 60)
print()

print("📁 Project Structure:")
print("""
fastapi-langgraph-app/
├── src/
│   ├── main.py                   # FastAPI app entry point
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py             # API endpoints
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── state.py              # LangGraph state schema
│   │   ├── nodes.py              # 6 workflow nodes
│   │   └── workflow.py           # LangGraph workflow
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py            # Pydantic models
│   └── utils/
│       ├── __init__.py
│       ├── config.py             # Configuration
│       ├── helpers.py            # Helper functions
│       └── openai_client.py      # OpenAI wrapper
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment template
├── sample_input.json             # Sample data
└── README.md                     # Full documentation
""")

print()
print("🎯 Key Features:")
print("  ✅ Locked fields preserved (byte-for-byte)")
print("  ✅ JSON structure maintained")
print("  ✅ Deterministic transformations")
print("  ✅ Comprehensive validation reports")
print("  ✅ Fast execution (< 10s typical)")
print()

print("📚 API Endpoints:")
print("  POST /api/v1/transform  - Transform JSON to new scenario")
print("  POST /api/v1/validate   - Validate transformed JSON")
print("  GET  /api/v1/health     - Health check")
print("  GET  /api/v1/scenarios  - List available scenarios")
print()

print("🔧 Configuration (.env):")
print("  OPENAI_API_KEY      - Your OpenAI API key")
print("  OPENAI_MODEL        - gpt-4o (default)")
print("  OPENAI_TEMPERATURE  - 0 (deterministic)")
print("  OPENAI_SEED         - 42 (reproducible)")
print()

print("=" * 60)
print("Ready to transform scenarios! 🚀")
print("=" * 60)
