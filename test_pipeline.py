"""
Test the new LangGraph pipeline.

Usage:
    python test_pipeline.py [1|2|3]

    python test_pipeline.py 1   # Gen Z Organic T-Shirts
    python test_pipeline.py 2   # Fast Food $1 Menu
    python test_pipeline.py 3   # ThriveBite Nutrition
"""
import asyncio
import json
import sys
import time
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

from dotenv import load_dotenv
load_dotenv()

# Input file
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
    },
    2: {
        "name": "Fast Food $1 Menu Response",
        "prompt": """students will propose how a fast food brand called "BurgerBlitz" should respond
to its competitor's $1 menu by analyzing the competitor's move, market impact, BurgerBlitz's
strengths, and four strategic options. Their goal is to propose a clear, realistic, and
sustainable plan to protect or grow market share via an executive summary.""",
    },
    3: {
        "name": "ThriveBite Nutrition Beverage",
        "prompt": """Acting as a strategic analyst at ThriveBite Nutrition, learners will assess
the viability of launching a new adaptogen-infused functional beverage targeting health-conscious
consumers seeking stress relief. They will analyze product-market fit, estimate the market
opportunity, benchmark competitors, evaluate internal capabilities, assess financial feasibility,
and weigh potential risks using the resources provided. Students must deliver a concise
executive summary recommending a go/no-go decision.""",
    }
}


async def run_test(prompt_num: int):
    """Run the pipeline with a specific prompt."""
    from src.graph.nodes import run_pipeline

    config = PROMPTS[prompt_num]

    print("\n" + "=" * 70)
    print(f"PIPELINE TEST: Prompt {prompt_num} - {config['name']}")
    print("=" * 70)

    # Load input
    print(f"\nLoading {INPUT_FILE}...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        input_json = json.load(f)
    print(f"Input size: {len(json.dumps(input_json)):,} chars")

    # Run pipeline
    print(f"\nScenario: {config['prompt'][:100]}...")
    print("\n" + "-" * 70)

    start = time.time()
    result = await run_pipeline(
        input_json=input_json,
        scenario_prompt=config['prompt']
    )
    elapsed = time.time() - start

    # Print results
    print("\n" + "-" * 70)
    print("RESULTS")
    print("-" * 70)
    print(f"Status: {result.status.upper()}")
    print(f"Final Score: {result.final_score:.2%}")
    print(f"Total Time: {elapsed:.1f}s")
    print(f"Adaptation Time: {result.adaptation_time_ms/1000:.1f}s")
    print(f"Shards Processed: {result.shards_processed}")
    print(f"Repair Iterations: {result.repair_iterations}")
    print(f"Issues Remaining: {result.issues_remaining}")

    print("\nAgent Scores:")
    for name, score in result.agent_scores.items():
        status = "[OK]" if score >= 0.95 else "[!!]"
        print(f"  {status} {name}: {score:.2%}")

    if result.errors:
        print(f"\nErrors: {result.errors}")

    # Save output
    os.makedirs("outputs", exist_ok=True)
    output_file = f"outputs/pipeline_output_prompt{prompt_num}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result.final_json, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to: {output_file}")

    return result


async def main():
    # Determine which prompt to run
    if len(sys.argv) > 1:
        try:
            num = int(sys.argv[1])
            if num in [1, 2, 3]:
                await run_test(num)
            else:
                print("Usage: python test_pipeline.py [1|2|3]")
        except ValueError:
            print("Usage: python test_pipeline.py [1|2|3]")
    else:
        # Run all 3
        for num in [1, 2, 3]:
            await run_test(num)


if __name__ == "__main__":
    asyncio.run(main())
