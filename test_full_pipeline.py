"""
Test Full Pipeline: V2 Adaptation + Validation

This script runs the complete pipeline:
1. V2 Skeleton-Based Adaptation
2. Validation with 8 agents
3. Reports scores and issues
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
    print("FULL PIPELINE TEST: V2 Adaptation + Validation")
    print("="*70)
    print(f"\nScenario: {SCENARIO_PROMPT[:100]}...")

    # Load source
    source_file = "Sim2_topic_wizard_data.json"
    print(f"\nLoading source: {source_file}")

    with open(source_file, 'r', encoding='utf-8') as f:
        source_json = json.load(f)

    print(f"Source size: {len(json.dumps(source_json)):,} chars")

    # =========================================================================
    # PHASE 1: ADAPTATION
    # =========================================================================
    print("\n" + "="*70)
    print("PHASE 1: V2 ADAPTATION")
    print("="*70)

    from src.stages.simple_adapter import adapt_simple

    adapt_start = time.time()

    result = await adapt_simple(
        input_json=source_json,
        scenario_prompt=SCENARIO_PROMPT,
        use_v2=True
    )

    adapt_time = time.time() - adapt_start

    print(f"\nAdaptation complete in {adapt_time:.1f}s")
    print(f"Mode: {result.mode}")
    print(f"Output: {result.output_chars:,} chars")
    print(f"Shards processed: {result.shards_processed}")
    print(f"Errors: {len(result.errors)}")

    if result.errors:
        print("\nErrors:")
        for err in result.errors[:3]:
            print(f"  - {err}")

    # Save adapted output
    output_file = "v2_medicore_output.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result.adapted_json, f, indent=2, ensure_ascii=False)
    print(f"\nSaved adapted JSON to: {output_file}")

    # =========================================================================
    # PHASE 2: VALIDATION
    # =========================================================================
    print("\n" + "="*70)
    print("PHASE 2: VALIDATION")
    print("="*70)

    from src.stages.simple_validators import run_all_validators

    validate_start = time.time()

    validation_report = await run_all_validators(
        adapted_json=result.adapted_json,
        scenario_prompt=SCENARIO_PROMPT
    )

    validate_time = time.time() - validate_start

    print(f"\nValidation complete in {validate_time:.1f}s")

    # =========================================================================
    # RESULTS
    # =========================================================================
    print("\n" + "="*70)
    print("VALIDATION RESULTS")
    print("="*70)

    print(f"\n{'Agent':<30} {'Score':<10} {'Status':<10}")
    print("-"*50)

    for agent_result in validation_report.agent_results:
        status = "PASS" if agent_result.passed else "FAIL"
        print(f"{agent_result.agent_name:<30} {agent_result.score:.1%}     {status}")

    print("-"*50)
    print(f"{'OVERALL':<30} {validation_report.overall_score:.1%}     "
          f"{'PASS' if validation_report.passed else 'FAIL'}")

    # Show issues
    if validation_report.total_issues > 0:
        print(f"\n\nTOTAL ISSUES: {validation_report.total_issues}")

        # Collect all issues
        all_issues = []
        for agent_result in validation_report.agent_results:
            for issue in agent_result.issues:
                all_issues.append(issue)

        print("\nTop Issues (first 10):")
        for i, issue in enumerate(all_issues[:10], 1):
            print(f"\n  {i}. [{issue.agent}] {issue.severity.upper()}")
            print(f"     Location: {issue.location[:80]}...")
            print(f"     Issue: {issue.issue[:100]}...")
            if issue.suggestion:
                print(f"     Fix: {issue.suggestion[:80]}...")

    # =========================================================================
    # CONTENT CHECK
    # =========================================================================
    print("\n" + "="*70)
    print("CONTENT CHECK")
    print("="*70)

    output_str = json.dumps(result.adapted_json)

    # Check for target domain mentions
    medicore_count = output_str.lower().count('medicore')
    cardioflow_count = output_str.lower().count('cardioflow')
    print(f"'MediCore' mentions: {medicore_count}")
    print(f"'CardioFlow' mentions: {cardioflow_count}")

    # Check for forbidden terms (source domain leakage)
    forbidden = ['velocity dome', 'greenbite', 'moodsip', 'juice', 'snacks', 'beverages', 'hr ', 'hiring', 'candidate', 'interview']
    leaks_found = []
    for term in forbidden:
        count = output_str.lower().count(term)
        if count > 0:
            leaks_found.append(f"'{term}': {count}")
            print(f"LEAK: '{term}' found {count} times!")

    if not leaks_found:
        print("No source domain leaks detected!")

    # Check for "the company" / "the manager"
    the_company = output_str.lower().count('the company')
    the_manager = output_str.lower().count('the manager')
    print(f"'the company': {the_company} (should be 0)")
    print(f"'the manager': {the_manager} (should be 0)")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    total_time = adapt_time + validate_time
    print(f"\nTotal time: {total_time:.1f}s (Adaptation: {adapt_time:.1f}s, Validation: {validate_time:.1f}s)")
    print(f"Overall score: {validation_report.overall_score:.1%}")
    print(f"Passed: {validation_report.passed}")
    print(f"Issues: {validation_report.total_issues}")

    if validation_report.passed:
        print("\nPIPELINE PASSED - Output is ready for review")
    else:
        print("\nPIPELINE FAILED - Review issues above")
        failed_agents = [a.agent_name for a in validation_report.agent_results if not a.passed]
        print(f"Failed agents: {', '.join(failed_agents)}")

    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)

    return validation_report


if __name__ == "__main__":
    report = asyncio.run(main())
