#!/usr/bin/env python3
"""
Simple script to generate a validation report from existing pipeline output.
Usage: python generate_report.py 43.json
"""

import json
import sys
from datetime import datetime

def generate_simple_report(data: dict) -> str:
    """Generate a human-readable report from pipeline output."""

    # Extract key data
    alignment_report = data.get("alignment_report", {})
    validation_report = data.get("validation_report", {})
    stage_timings = data.get("stage_timings", {})
    errors = data.get("errors", [])

    # Get scores
    alignment_score = alignment_report.get("overall_score", 0)
    alignment_passed = alignment_report.get("passed", False)

    # Build report
    lines = []
    lines.append("=" * 70)
    lines.append("ADAPTATION PIPELINE REPORT")
    lines.append("=" * 70)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Executive Summary
    lines.append("## EXECUTIVE SUMMARY")
    lines.append("-" * 40)
    if alignment_passed:
        lines.append("✅ PASSED - Adaptation meets quality bar")
    else:
        lines.append(f"⚠️  NEEDS REVIEW - Alignment score: {alignment_score:.1%}")
    lines.append("")

    # Alignment Results
    lines.append("## ALIGNMENT CHECK RESULTS")
    lines.append("-" * 40)
    lines.append(f"Overall Score: {alignment_score:.1%}")
    lines.append(f"Threshold: {alignment_report.get('threshold', 0.98):.0%}")
    lines.append(f"Status: {'PASSED' if alignment_passed else 'FAILED'}")
    lines.append("")

    # Individual checks
    results = alignment_report.get("results", [])
    if results:
        lines.append("Individual Checks:")
        for r in results:
            status = "✅" if r.get("passed", False) else "❌"
            score = r.get("score", 0)
            name = r.get("rule_name", r.get("rule_id", "Unknown"))
            lines.append(f"  {status} {name}: {score:.1%}")

            # Show issues for failed checks
            if not r.get("passed", True) and r.get("issues"):
                for issue in r.get("issues", [])[:3]:
                    desc = issue.get("description", str(issue))[:80]
                    lines.append(f"      → {desc}")
    lines.append("")

    # Validation Results
    if validation_report:
        lines.append("## VALIDATION RESULTS")
        lines.append("-" * 40)
        val_passed = validation_report.get("passed", False)
        val_score = validation_report.get("overall_score", 0)
        lines.append(f"Overall Score: {val_score:.1%}")
        lines.append(f"Status: {'PASSED' if val_passed else 'FAILED'}")

        # Show shard results
        shard_results = validation_report.get("shard_results", {})
        if shard_results:
            lines.append("\nShard Results:")
            for shard_id, result in list(shard_results.items())[:10]:
                shard_score = result.get("score", 0)
                shard_passed = result.get("passed", False)
                status = "✅" if shard_passed else "❌"
                lines.append(f"  {status} {shard_id}: {shard_score:.1%}")
        lines.append("")

    # Timing
    if stage_timings:
        lines.append("## STAGE TIMINGS")
        lines.append("-" * 40)
        total_ms = 0
        for stage, ms in stage_timings.items():
            seconds = ms / 1000
            total_ms += ms
            lines.append(f"  {stage}: {seconds:.1f}s")
        lines.append(f"  TOTAL: {total_ms/1000:.1f}s")
        lines.append("")

    # Errors
    if errors:
        lines.append("## ERRORS")
        lines.append("-" * 40)
        for err in errors:
            stage = err.get("stage", "unknown")
            msg = err.get("message", str(err))[:100]
            lines.append(f"  [{stage}] {msg}")
        lines.append("")

    # Recommendations
    lines.append("## RECOMMENDATIONS")
    lines.append("-" * 40)

    # Check for common issues
    for r in results:
        if not r.get("passed", True):
            rule_id = r.get("rule_id", "")
            if "manager" in rule_id:
                lines.append("• Fix reporting manager name inconsistencies")
            elif "company" in rule_id:
                lines.append("• Fix company name inconsistencies")
            elif "klo" in rule_id:
                lines.append("• Improve KLO alignment with questions/resources")
            elif "scenario" in rule_id:
                lines.append("• Ensure scenario coherence across sections")

    if alignment_passed:
        lines.append("• No critical issues - ready for review")

    lines.append("")
    lines.append("=" * 70)
    lines.append("END OF REPORT")
    lines.append("=" * 70)

    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_report.py <json_file>")
        print("Example: python generate_report.py 43.json")
        sys.exit(1)

    json_file = sys.argv[1]

    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{json_file}' not found")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in '{json_file}': {e}")
        sys.exit(1)

    report = generate_simple_report(data)
    print(report)

    # Also save to file
    output_file = json_file.replace('.json', '_report.txt')
    with open(output_file, 'w') as f:
        f.write(report)
    print(f"\nReport saved to: {output_file}")


if __name__ == "__main__":
    main()
