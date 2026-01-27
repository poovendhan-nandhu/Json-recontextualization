"""
Test script for the simplified adapter.

Run:
    python test_simple_adapter.py

This tests the Phase 1 simplified approach:
- Scenario Prompt + JSON -> LLM -> Adapted JSON
- No factsheet, no RAG, no poison lists

Input format: Sim2/3/4/5_topic_wizard_data.json (root-level snake_case keys)
"""
import asyncio
import json
import sys
import time
import logging

# Configure logging to see adapter progress
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Available input files (use Sim5 - HarvestBowls QSR $1 menu response - better match for BurgerBlitz)
INPUT_FILE = "Sim5_topic_wizard_data.json"

# The target scenario prompt - recontextualize MoodSip to a NEW scenario
# Example: Change from juice bar to fast food response simulation
SCENARIO_PROMPT = """
Acting as a consultant, students will develop a short executive summary recommending
how a fast food brand called "BurgerBlitz" should respond to its competitor's $1 menu.
They'll analyze the competitor's move, market impact, BurgerBlitz's strengths, and
four strategic options. Their goal is to propose a clear, realistic, and sustainable
plan to protect or grow market share via an executive summary.
"""


async def test_simple_adapter():
    """Test the simple adapter with validation + repair."""
    from src.stages.simple_adapter import adapt_simple
    from src.stages.simple_validators import validate_and_repair, run_all_validators

    # Load input file
    print(f"Loading {INPUT_FILE}...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        input_json = json.load(f)

    input_size = len(json.dumps(input_json))
    print(f"Input size: {input_size:,} chars")
    print(f"\nScenario prompt:\n{SCENARIO_PROMPT[:200]}...")

    # STAGE 1: Adaptation
    print("\n" + "=" * 60)
    print("STAGE 1: PARALLEL SHARD ADAPTATION (Gemini 3.0 Flash)")
    print("=" * 60)

    start = time.time()
    result = await adapt_simple(
        input_json=input_json,
        scenario_prompt=SCENARIO_PROMPT,
    )
    adapt_time = time.time() - start

    print(f"Mode: {result.mode}")
    print(f"Time: {result.time_ms}ms ({adapt_time:.1f}s)")
    print(f"Shards processed: {result.shards_processed}")

    # STAGE 2: Validation + Repair
    print("\n" + "=" * 60)
    print("STAGE 2: VALIDATION + REPAIR (GPT-4o)")
    print("=" * 60)

    start = time.time()
    final_json, validation_report = await validate_and_repair(
        result.adapted_json,
        SCENARIO_PROMPT,
        max_iter=2
    )
    validate_time = time.time() - start

    print(f"Validation time: {validate_time:.1f}s")
    print(f"Overall score: {validation_report.overall_score:.2%}")
    print(f"Passed: {validation_report.passed}")
    print(f"Total issues: {validation_report.total_issues}")

    print("\nAgent Results:")
    for ar in validation_report.agent_results:
        status = "PASS" if ar.passed else "FAIL"
        print(f"  {ar.agent_name}: {ar.score:.2%} [{status}] ({len(ar.issues)} issues)")

    # Print results
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Total time: {adapt_time + validate_time:.1f}s")
    print(f"Input: {result.input_chars:,} chars")
    print(f"Output: {len(json.dumps(final_json)):,} chars")

    # Save output
    output_path = "simple_adapted_output.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_json, f, indent=2, ensure_ascii=False)
    print(f"\nOutput saved to: {output_path}")

    # Quick validation - check if key terms were replaced
    # Strip locked shards for accurate validation (they contain original HR content)
    import copy
    import re
    unlocked_json = copy.deepcopy(final_json)
    topic_data = unlocked_json.get("topicWizardData", {})
    topic_data.pop("scenarioOptions", None)
    topic_data.pop("workspaceIds", None)
    output_str = json.dumps(unlocked_json)

    print("\n" + "=" * 60)
    print("QUICK VALIDATION (unlocked content only)")
    print("=" * 60)

    # Check for target terms (should be present)
    target_terms = ["organic", "T-shirt", "sustainable", "Gen Z", "fashion"]
    for term in target_terms:
        count = output_str.lower().count(term.lower())
        status = "OK" if count > 0 else "MISSING"
        print(f"  {term}: {count} occurrences [{status}]")

    # Check for source terms (should NOT be present) - use word boundaries for HR
    print("\nSource terms (should be 0):")
    source_checks = [
        ("Summit Innovations", output_str.count("Summit Innovations")),
        ("Elizabeth Carter", output_str.count("Elizabeth Carter")),
        ("HR (standalone)", len(re.findall(r'\bHR\b', output_str))),
        ("hiring", output_str.lower().count("hiring")),
        ("candidate", output_str.lower().count("candidate")),
        ("interview", output_str.lower().count("interview")),
    ]
    issues = []
    for term, count in source_checks:
        status = "OK" if count == 0 else f"LEAKED ({count})"
        print(f"  {term}: {count} occurrences [{status}]")
        if count > 0:
            issues.append(f"{term}: {count}")

    if issues:
        print(f"\nWARNING: {len(issues)} source terms still present!")
    else:
        print("\nSUCCESS: No source terms found!")

    return final_json, validation_report


async def test_via_api():
    """Test via the API endpoint (server must be running)."""
    import httpx

    print("Testing via API endpoint...")
    print("Make sure server is running: uvicorn src.main:app --reload")

    # Load sample input
    with open("sample_main.json", "r", encoding="utf-8") as f:
        input_json = json.load(f)

    # Call API
    async with httpx.AsyncClient(timeout=600.0) as client:
        response = await client.post(
            "http://localhost:8000/api/v1/adapt/simple",
            json={
                "input_json": input_json,
                "scenario_prompt": SCENARIO_PROMPT,
            }
        )

    if response.status_code == 200:
        result = response.json()
        print(f"Status: {result['status']}")
        print(f"Mode: {result['mode']}")
        print(f"Time: {result['timing']['time_ms']}ms")
        print(f"Input: {result['size']['input_chars']:,} chars")
        print(f"Output: {result['size']['output_chars']:,} chars")

        # Save output
        with open("simple_api_output.json", "w", encoding="utf-8") as f:
            json.dump(result["adapted_json"], f, indent=2, ensure_ascii=False)
        print("Output saved to: simple_api_output.json")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--api":
        asyncio.run(test_via_api())
    else:
        asyncio.run(test_simple_adapter())
