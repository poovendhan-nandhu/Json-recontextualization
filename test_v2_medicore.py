"""
Test V2 Skeleton-Based Adapter with MediCore Scenario.

Source: Sim2_topic_wizard_data.json
Target: MediCore pharmaceutical competitive response scenario
"""

import asyncio
import json
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

SCENARIO_PROMPT = """MediCore's flagship drug sales are at risk after a competitor slashes prices on a generic equivalent. Learners must recommend strategies balancing regulatory compliance, pricing flexibility, and long-term R&D investment.

As a junior strategy consultant at MediCore Pharmaceuticals, learners will analyze market data, competitive intelligence, and internal capabilities to develop a comprehensive response strategy. They will present their recommendations to the VP of Commercial Strategy."""


async def main():
    print("\n" + "="*70)
    print("V2 SKELETON-BASED ADAPTER TEST")
    print("="*70)
    print(f"\nScenario: {SCENARIO_PROMPT[:100]}...")

    # Load Sim2
    source_file = "Sim2_topic_wizard_data.json"
    print(f"\nLoading source: {source_file}")

    with open(source_file, 'r', encoding='utf-8') as f:
        source_json = json.load(f)

    print(f"Source size: {len(json.dumps(source_json)):,} chars")

    # Import adapter
    from src.stages.simple_adapter import adapt_simple

    # Run V2 adaptation
    print("\n" + "-"*70)
    print("RUNNING V2 ADAPTATION...")
    print("-"*70)

    start_time = time.time()

    result = await adapt_simple(
        input_json=source_json,
        scenario_prompt=SCENARIO_PROMPT,
        use_v2=True
    )

    elapsed = time.time() - start_time

    # Print results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)

    print(f"\nMode: {result.mode}")
    print(f"Time: {elapsed:.1f}s ({result.time_ms}ms)")
    print(f"Input: {result.input_chars:,} chars")
    print(f"Output: {result.output_chars:,} chars")
    print(f"Shards processed: {result.shards_processed}")
    print(f"Errors: {len(result.errors)}")

    if result.errors:
        print("\nErrors:")
        for err in result.errors[:5]:
            print(f"  - {err}")

    # Print Stage 0 outputs
    print("\n" + "-"*70)
    print("STAGE 0 OUTPUTS")
    print("-"*70)

    print(f"\n## Entity Map")
    if result.entity_map:
        company = result.entity_map.get('company', {})
        print(f"  Company: {company.get('name', 'N/A')} (@{company.get('domain', 'N/A')})")
        people = result.entity_map.get('people', [])
        print(f"  People: {len(people)}")
        for p in people[:3]:
            if isinstance(p, dict):
                print(f"    - {p.get('name', 'N/A')}: {p.get('role', 'N/A')}")
        products = result.entity_map.get('products', [])
        print(f"  Products: {products[:3]}")

    print(f"\n## Canonical Numbers")
    if result.canonical_numbers:
        for key, val in list(result.canonical_numbers.items())[:8]:
            print(f"  {key}: {val}")

    print(f"\n## Alignment Map ({len(result.alignment_map)} entries)")
    if result.alignment_map:
        for qid, data in list(result.alignment_map.items())[:2]:
            print(f"\n  [{qid}]")
            print(f"    Assesses KLO: {data.get('assesses_klo', 'N/A')[:60]}...")
            print(f"    Question: {data.get('question_must_ask', 'N/A')[:60]}...")
            print(f"    Resource needs: {data.get('resource_must_contain', [])[:2]}...")

    # Save output
    output_file = "v2_medicore_output.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result.adapted_json, f, indent=2, ensure_ascii=False)
    print(f"\n\nOutput saved to: {output_file}")

    # Save Stage 0 data for inspection
    stage0_file = "v2_medicore_stage0.json"
    with open(stage0_file, 'w', encoding='utf-8') as f:
        json.dump({
            "entity_map": result.entity_map,
            "domain_profile": result.domain_profile,
            "alignment_map": result.alignment_map,
            "canonical_numbers": result.canonical_numbers,
        }, f, indent=2, ensure_ascii=False)
    print(f"Stage 0 data saved to: {stage0_file}")

    # Quick content check
    print("\n" + "-"*70)
    print("QUICK CONTENT CHECK")
    print("-"*70)

    output_str = json.dumps(result.adapted_json)

    # Check for MediCore mentions
    medicore_count = output_str.lower().count('medicore')
    print(f"'MediCore' mentions: {medicore_count}")

    # Check for forbidden terms
    forbidden = ['velocity dome', 'hr ', 'hiring', 'candidate', 'interview']
    for term in forbidden:
        count = output_str.lower().count(term)
        if count > 0:
            print(f"WARNING: '{term}' found {count} times!")

    # Check for "the company" / "the manager"
    the_company = output_str.lower().count('the company')
    the_manager = output_str.lower().count('the manager')
    print(f"'the company': {the_company} (should be 0)")
    print(f"'the manager': {the_manager} (should be 0)")

    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)

    return result


if __name__ == "__main__":
    result = asyncio.run(main())
