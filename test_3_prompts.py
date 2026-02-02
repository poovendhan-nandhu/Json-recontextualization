"""
Test script to run 3 different scenario prompts.

Uses Sim2 (MoodSip juice bar) as BASE input and transforms to:
- PROMPT 1: Gen Z organic T-shirts brand
- PROMPT 2: Fast food $1 menu response
- PROMPT 3: ThriveBite Nutrition adaptogen beverage

Run:
    python test_3_prompts.py [1|2|3]

    python test_3_prompts.py 1   # Run prompt 1 only
    python test_3_prompts.py 2   # Run prompt 2 only
    python test_3_prompts.py 3   # Run prompt 3 only
    python test_3_prompts.py     # Run all 3
"""
import asyncio
import json
import sys
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

from dotenv import load_dotenv
load_dotenv()

# Base input file - MoodSip juice bar (will be transformed)
INPUT_FILE = "Sim2_topic_wizard_data.json"

# The 3 target prompts
PROMPTS = {
    1: {
        "name": "Gen Z Organic T-Shirts",
        "prompt": """learners will act as a junior consultant for an exciting Gen Z organic T-shirts brand
called "EcoThread Collective", tasked with analyzing the U.S. market and providing a go/no-go
market entry recommendation. Using only the simulation's provided data, they will apply
structured frameworks to assess market potential, competition, capabilities, finances, and
risks before developing their final strategy.""",
        "expected_terms": ["organic", "t-shirt", "sustainable", "Gen Z", "fashion", "apparel"],
        "forbidden_terms": ["juice", "beverage", "MoodSip", "wellness drink", "cold-pressed"],
    },
    2: {
        "name": "Fast Food $1 Menu Response",
        "prompt": """students will propose how a fast food brand called "BurgerBlitz" should respond
to its competitor's $1 menu by analyzing the competitor's move, market impact, BurgerBlitz's
strengths, and four strategic options. Their goal is to propose a clear, realistic, and
sustainable plan to protect or grow market share via an executive summary.""",
        "expected_terms": ["burger", "fast food", "$1 menu", "competitor", "market share", "restaurant"],
        "forbidden_terms": ["juice", "beverage", "MoodSip", "wellness", "organic t-shirt"],
    },
    3: {
        "name": "ThriveBite Nutrition Beverage",
        "prompt": """Acting as a strategic analyst at ThriveBite Nutrition, learners will assess
the viability of launching a new adaptogen-infused functional beverage targeting health-conscious
consumers seeking stress relief. They will analyze product-market fit, estimate the market
opportunity, benchmark competitors, evaluate internal capabilities, assess financial feasibility,
and weigh potential risks using the resources provided. Students must deliver a concise
executive summary recommending a go/no-go decision.""",
        "expected_terms": ["ThriveBite", "adaptogen", "functional beverage", "stress relief", "wellness"],
        "forbidden_terms": ["MoodSip", "juice bar", "t-shirt", "burger", "fast food"],
    }
}


def count_terms(json_data: dict, terms: list[str]) -> dict:
    """Count occurrences of terms in JSON."""
    content = json.dumps(json_data, ensure_ascii=False).lower()
    counts = {}
    for term in terms:
        count = content.count(term.lower())
        if count > 0:
            counts[term] = count
    return counts


async def run_prompt(prompt_num: int, input_json: dict):
    """Run a single prompt test."""
    from src.stages.simple_adapter import adapt_simple
    from src.stages.simple_validators import validate_and_repair

    config = PROMPTS[prompt_num]

    print("\n" + "=" * 70)
    print(f"PROMPT {prompt_num}: {config['name']}")
    print("=" * 70)
    print(f"Scenario: {config['prompt'][:100]}...")

    # STAGE 1: Adaptation
    print("\n[STAGE 1] Adapting with Gemini...")
    start = time.time()

    result = await adapt_simple(
        input_json=input_json,
        scenario_prompt=config['prompt'],
    )

    adapt_time = time.time() - start
    print(f"Adaptation time: {adapt_time:.1f}s")
    print(f"Shards processed: {result.shards_processed}")

    # STAGE 2: Validation
    print("\n[STAGE 2] Validating with GPT...")
    start = time.time()

    final_json, validation_report = await validate_and_repair(
        result.adapted_json,
        config['prompt'],
        max_iter=1  # Quick validation
    )

    validate_time = time.time() - start
    print(f"Validation time: {validate_time:.1f}s")
    print(f"Overall score: {validation_report.overall_score:.2%}")

    # Check terms
    print("\n[TERM CHECK]")
    expected = count_terms(final_json, config['expected_terms'])
    forbidden = count_terms(final_json, config['forbidden_terms'])

    print("Expected terms found:")
    for term, count in expected.items():
        print(f"  [OK] {term}: {count}")

    if forbidden:
        print("\n[WARNING] FORBIDDEN terms still present:")
        for term, count in forbidden.items():
            print(f"  [X] {term}: {count}")
    else:
        print("\n[OK] No forbidden terms found!")

    # Check for real citations
    citation_terms = ["McKinsey", "BCG", "Deloitte", "Gartner", "Forrester", "IBISWorld", "Statista", "Nielsen"]
    citations = count_terms(final_json, citation_terms)
    if citations:
        print("\n[WARNING] Real citations found (should be removed):")
        for term, count in citations.items():
            print(f"  [X] {term}: {count}")
    else:
        print("[OK] No real citations found!")

    # Save output to outputs folder
    import os
    os.makedirs("outputs", exist_ok=True)
    output_file = f"outputs/output_prompt{prompt_num}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_json, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to: {output_file}")

    return {
        "prompt": prompt_num,
        "name": config['name'],
        "adapt_time": adapt_time,
        "validate_time": validate_time,
        "score": validation_report.overall_score,
        "expected_found": len(expected),
        "forbidden_found": len(forbidden),
        "citations_found": len(citations),
    }


async def main():
    # Load input
    print(f"Loading {INPUT_FILE}...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        input_json = json.load(f)

    input_size = len(json.dumps(input_json))
    print(f"Input size: {input_size:,} chars")

    # Determine which prompts to run
    prompts_to_run = []
    if len(sys.argv) > 1:
        try:
            num = int(sys.argv[1])
            if num in [1, 2, 3]:
                prompts_to_run = [num]
            else:
                print("Usage: python test_3_prompts.py [1|2|3]")
                return
        except ValueError:
            print("Usage: python test_3_prompts.py [1|2|3]")
            return
    else:
        prompts_to_run = [1, 2, 3]

    # Run tests
    results = []
    for num in prompts_to_run:
        result = await run_prompt(num, input_json)
        results.append(result)

    # Summary
    if len(results) > 1:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        for r in results:
            status = "[PASS]" if r['forbidden_found'] == 0 and r['citations_found'] == 0 else "[ISSUES]"
            print(f"Prompt {r['prompt']} ({r['name']}): {r['score']:.2%} {status}")


if __name__ == "__main__":
    asyncio.run(main())
