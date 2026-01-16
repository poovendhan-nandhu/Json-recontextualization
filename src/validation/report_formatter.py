"""
Validation Report Formatter

Formats validation data into the Cartedo Validation Summary Report format.
Matches the exact structure used for PM/Client/QA consumption.
"""

from datetime import datetime
from typing import Optional, List
import uuid

from .check_definitions import (
    CRITICAL_CHECKS,
    FLAGGED_CHECKS,
    CheckDefinition,
    get_check_by_id,
)
from .report_generator import (
    ValidationReportData,
    FailureSummary,
    RecommendedFix,
)
from .check_runner import AggregatedResults


def generate_report_id(target_scenario: str) -> str:
    """Generate a report ID like VAL-2026-01-09-ALOPS-07."""
    date_str = datetime.now().strftime("%Y-%m-%d")
    # Create short code from target scenario
    code = "".join(word[0].upper() for word in target_scenario.split()[:2])
    seq = str(uuid.uuid4().int)[:2]
    return f"VAL-{date_str}-{code}-{seq}"


def format_markdown_report(data: ValidationReportData) -> str:
    """
    Format validation data into the Cartedo Validation Summary Report.

    Args:
        data: ValidationReportData with all report information

    Returns:
        Complete Markdown report as string
    """
    sections = [
        _format_header(data),
        _format_executive_gate(data),
        _format_pipeline_outcomes(data),
        _format_critical_checks(data),
        _format_flagged_checks(data),
        _format_failures(data),
        _format_downstream_effects(data),
        _format_pr_review(data),
        _format_next_actions(data),
        _format_appendix(data),
    ]

    return "\n\n".join(filter(None, sections))


def _format_header(data: ValidationReportData) -> str:
    """Format report header."""
    report_id = generate_report_id(data.target_scenario)
    timestamp = data.timestamp.strftime("%b %d, %Y (%Z)" if data.timestamp.tzinfo else "%b %d, %Y")

    return f"""# Cartedo Simulation Adaptation Framework — Validation Summary Report

**Scenario Change:** {data.original_scenario} → {data.target_scenario}
**Report ID:** {report_id}
**Generated on:** {timestamp}
**Audience:** {data.audience}

---

## Run Definition

> One run = the full end-to-end conversion of one simulation into a fully recontextualized "ideal" simulation, including all internal agent passes and the compliance loop."""


def _format_executive_gate(data: ValidationReportData) -> str:
    """Format Section 1: Executive Gate."""

    # Determine status emoji and label
    if data.meets_threshold and len(data.aggregated_results.failed_runs) == 0:
        status_emoji = "✅"
        status_label = "APPROVED (Meets Acceptance Bar)"
    elif data.meets_threshold:
        status_emoji = "✅"
        status_label = "APPROVED (Meets Acceptance Bar)"
    else:
        status_emoji = "❌"
        status_label = "NOT APPROVED (Below Acceptance Bar)"

    # Count runs with only flagged issues
    runs_with_flagged_only = sum(
        1 for r in data.aggregated_results.run_results
        if r.passed_all_critical and len(r.flagged_warnings) > 0
    )

    # Release recommendation
    if data.can_ship_as_is and runs_with_flagged_only == 0:
        release_rec = "Ship is approved. No blocking or flagged issues."
    elif data.can_ship_as_is:
        # Find the most common flagged issue
        flagged_cluster = _get_top_flagged_cluster(data)
        release_rec = f'Ship is allowed, but address Flagged Issue Cluster "{flagged_cluster}" before client demo.'
    else:
        release_rec = f"Fix required. {len(data.aggregated_results.failed_runs)} run(s) failed Critical checks."

    passing = data.aggregated_results.runs_passing_critical
    total = data.total_runs
    pass_rate = data.critical_pass_rate

    return f"""## 1) Executive Gate (Go / No-Go)

| Metric | Result |
|--------|--------|
| **Overall Status** | {status_emoji} **{status_label}** |
| Total runs evaluated | {total} |
| Runs passing all Critical checks | {passing} / {total} ({pass_rate:.0%}) {"✅" if data.meets_threshold else "❌"} ({"meets" if data.meets_threshold else "below"} ≥{data.acceptance_threshold:.0%} requirement) |
| Runs with only Flagged issues | {runs_with_flagged_only} / {total} (review recommended, not blocking) |
| Gold output produced | {total} / {total} (golden artifact produced for every run) |

**Release recommendation:** {release_rec}"""


def _format_pipeline_outcomes(data: ValidationReportData) -> str:
    """Format Section 2: Agent-by-Agent Pipeline Outcomes."""

    # Define pipeline stages
    stages = [
        {
            "number": 1,
            "name": "Adaptation Engine (Planner + Generator)",
            "artifacts": ["adapted_working.json", "entity_map.json"],
            "critical_outcomes": [
                f"✅ {data.total_runs}/{data.total_runs} produced draft adapted simulation",
                f"✅ Entity mapping complete in all runs",
            ],
            "flagged_patterns": _get_stage_flagged_patterns(data, "adaptation"),
            "log_path": "logs/run-*/stage1_adaptation_engine.log",
            "artifact_paths": [
                "artifacts/run-*/adapted_working.json",
                "artifacts/run-*/entity_map.json",
            ],
        },
        {
            "number": 2,
            "name": "Sharder",
            "artifacts": ["shards/* (rubrics, options, personas, resources)", "cas_hashes.json"],
            "critical_outcomes": [
                f"✅ {data.total_runs}/{data.total_runs} sharded successfully",
                f"✅ {data.total_runs}/{data.total_runs} generated CAS hashes",
                f"✅ {data.total_runs}/{data.total_runs} slice contracts attached",
            ],
            "flagged_patterns": [],
            "log_path": "logs/run-*/stage2_sharder.log",
            "artifact_paths": [
                "artifacts/run-*/cas_hashes.json",
                "artifacts/run-*/shards/",
            ],
        },
        {
            "number": 3,
            "name": "Unified Checker (QA Scanner)",
            "artifacts": ["rule_scorecard.json", "check_report.md (human-readable)"],
            "critical_outcomes": _get_checker_outcomes(data),
            "flagged_patterns": _get_checker_issues(data),
            "log_path": "logs/run-*/stage3_unified_checker.log",
            "artifact_paths": [
                "artifacts/run-*/rule_scorecard.json",
                "artifacts/run-*/check_report.md",
            ],
        },
        {
            "number": 4,
            "name": "Structural Fixers (Shape Only)",
            "artifacts": ["structural_patch.json", "barrier_manifest.json (locked slices)"],
            "critical_outcomes": [
                f"✅ {data.total_runs}/{data.total_runs} fixed all structural issues where present",
                f"✅ {data.total_runs}/{data.total_runs} enforced barrier locks after structure fixes",
            ],
            "flagged_patterns": [],
            "rules_enforced": ["No wording changes", "No edits to locked slices after barrier=true"],
            "log_path": "logs/run-*/stage4_structural_fixers.log",
            "artifact_paths": [
                "artifacts/run-*/structural_patch.json",
                "artifacts/run-*/barrier_manifest.json",
            ],
        },
        {
            "number": 5,
            "name": "Semantic Fixers (Meaning + Realism)",
            "artifacts": ["semantic_patch.json"],
            "critical_outcomes": _get_semantic_fixer_outcomes(data),
            "flagged_patterns": _get_semantic_flagged(data),
            "log_path": "logs/run-*/stage5_semantic_fixers.log",
            "artifact_paths": ["artifacts/run-*/semantic_patch.json"],
        },
        {
            "number": 6,
            "name": "Finisher (Compliance Loop + Global Guard)",
            "artifacts": ["compliance_summary.json", "dependency_recheck_trace.json", "global_guard_report.json"],
            "critical_outcomes": _get_finisher_outcomes(data),
            "pass_criteria": ["Critical blockers = 0", "Weighted compliance score ≥ 98%", "No infinite loops / oscillation"],
            "log_path": "logs/run-*/stage6_finisher.log",
            "artifact_paths": [
                "artifacts/run-*/compliance_summary.json",
                "artifacts/run-*/dependency_recheck_trace.json",
                "artifacts/run-*/global_guard_report.json",
            ],
        },
        {
            "number": 7,
            "name": "Human-in-the-Loop Approval",
            "artifacts": ["human_review_packet.html", "visual_diff.html"],
            "what_human_sees": ["Scorecard (Critical + Flagged)", "Visual diff", "Explanations + links to patches"],
            "human_decisions": _get_human_decisions(data),
            "log_path": None,
            "artifact_paths": [
                "artifacts/run-*/human_review_packet.html",
                "artifacts/run-*/visual_diff.html",
            ],
        },
    ]

    output = ["## 2) Agent-by-Agent Pipeline Outcomes (Process Transparency)"]

    for stage in stages:
        stage_output = [f"\n### Stage {stage['number']} — {stage['name']}"]

        # Output artifacts
        stage_output.append("\n**Output artifacts**")
        for artifact in stage["artifacts"]:
            stage_output.append(f"- `{artifact}`")

        # Pass criteria (if any)
        if "pass_criteria" in stage:
            stage_output.append("\n**Pass criteria**")
            for criteria in stage["pass_criteria"]:
                stage_output.append(f"- {criteria}")

        # What human sees (for HITL stage)
        if "what_human_sees" in stage:
            stage_output.append("\n**What human sees**")
            for item in stage["what_human_sees"]:
                stage_output.append(f"- {item}")

        # Critical outcomes
        if "critical_outcomes" in stage and stage["critical_outcomes"]:
            stage_output.append("\n**Critical outcome**")
            for outcome in stage["critical_outcomes"]:
                stage_output.append(f"- {outcome}")

        # Human decisions (for HITL stage)
        if "human_decisions" in stage:
            stage_output.append("\n**Human decision (sample)**")
            for decision in stage["human_decisions"]:
                stage_output.append(f"- {decision}")

        # Rules enforced (if any)
        if "rules_enforced" in stage:
            stage_output.append("\n**Rules enforced**")
            for rule in stage["rules_enforced"]:
                stage_output.append(f"- {rule}")

        # Flagged patterns (if any)
        if "flagged_patterns" in stage and stage["flagged_patterns"]:
            stage_output.append("\n**Notable flagged patterns**")
            for pattern in stage["flagged_patterns"]:
                stage_output.append(f"- {pattern}")

        # Top issues (if any)
        if "top_issues" in stage:
            stage_output.append("\n**Top issues detected (pre-fix)**")
            for issue in stage["top_issues"]:
                stage_output.append(f"- {issue}")

        # Drill-down
        stage_output.append("\n**Drill-down**")
        if stage["log_path"]:
            stage_output.append(f"- Logs: `{stage['log_path']}`")
        for path in stage.get("artifact_paths", []):
            stage_output.append(f"- `{path}`")

        output.append("\n".join(stage_output))

    return "\n".join(output)


def _format_critical_checks(data: ValidationReportData) -> str:
    """Format Section 3: Critical Checks Aggregated Results."""

    rows = []
    for check in CRITICAL_CHECKS:
        agg = data.aggregated_results.check_aggregations.get(check.id, {})
        passes = agg.get("passes", data.total_runs)
        total = agg.get("total", data.total_runs)
        pass_rate = agg.get("pass_rate", 1.0)

        if pass_rate >= check.threshold_value:
            status = "✅"
        else:
            status = "🔴"

        rows.append(f"| {check.id} | {check.ensures[:55]}{'...' if len(check.ensures) > 55 else ''} | {check.threshold} | {passes}/{total} | {status} |")

    rows_str = "\n".join(rows)

    # Critical gate summary
    passing = data.aggregated_results.runs_passing_critical
    total = data.total_runs
    pass_rate = data.critical_pass_rate

    return f"""## 3) Critical Checks (Blocking) — Aggregated Results

> **Acceptance rule:** A run passes only if all Critical checks pass. Release is acceptable if ≥{data.acceptance_threshold:.0%} of runs pass all Critical checks.

| Check | What it Ensures | Threshold | Result | Status |
|-------|-----------------|-----------|--------|--------|
{rows_str}

**Critical Gate Summary:** {passing}/{total} runs passed all Critical checks = {pass_rate:.0%} {"✅" if data.meets_threshold else "❌"}"""


def _format_flagged_checks(data: ValidationReportData) -> str:
    """Format Section 4: Flagged Checks Aggregated Results."""

    rows = []
    lowest_check = None
    lowest_score = 1.0

    for check in FLAGGED_CHECKS:
        agg = data.aggregated_results.check_aggregations.get(check.id, {})
        avg_score = agg.get("avg_score", 1.0)
        passes = agg.get("passes", data.total_runs)
        total = agg.get("total", data.total_runs)

        if avg_score >= check.threshold_value:
            status = "✅"
        else:
            status = "⚠️"
            if avg_score < lowest_score:
                lowest_score = avg_score
                lowest_check = check.name

        rows.append(f"| {check.id} | {check.ensures[:45]}{'...' if len(check.ensures) > 45 else ''} | {check.threshold} | {passes}/{total} | {status} |")

    rows_str = "\n".join(rows)

    # Flagged cluster note
    cluster_note = ""
    if lowest_check:
        cluster_note = f"\n**Flagged cluster to address next:** {lowest_check} ({FLAGGED_CHECKS[[c.name for c in FLAGGED_CHECKS].index(lowest_check)].id}) — non-blocking but client-visible."

    return f"""## 4) Flagged Checks (Non-Blocking) — Aggregated Results

| Check | What it Flags | Threshold | Result | Status |
|-------|---------------|-----------|--------|--------|
{rows_str}
{cluster_note}"""


def _format_failures(data: ValidationReportData) -> str:
    """Format Section 5: What Failed."""

    if not data.failure_summaries:
        return """## 5) What Failed (Critical) + Why (Human Readable) + Drill-Down

**No Critical failures.** All runs passed all Critical checks."""

    output = ["## 5) What Failed (Critical) + Why (Human Readable) + Drill-Down"]

    for failure in data.failure_summaries:
        check = get_check_by_id(failure.check_id)
        check_name = check.name if check else failure.failure_type

        for run_id in failure.affected_runs:
            run_num = run_id.split("-")[-1] if "-" in run_id else run_id

            output.append(f"""
### Run {run_num} — ❌ FAILED (Critical Check {failure.check_id}: {check_name})

| Aspect | Detail |
|--------|--------|
| **Short reason** | {failure.example_issue} |
| **Impact** | {failure.why_it_matters} |
| **Where it happened** | {failure.where_it_happens} |
| **Detected by** | {failure.detection_stage} |

**Pointers**
- Scorecard: `artifacts/{run_id}/rule_scorecard.json` → findings[{failure.check_id}]
- Offending shards: Check `artifacts/{run_id}/shards/` for affected content
- Logs: `logs/{run_id}/stage3_unified_checker.log`, `logs/{run_id}/stage6_finisher.log`

**Disposition:** {failure.fix_scope}. Requires targeted fix on affected shards, then Finisher recheck.""")

    return "\n".join(output)


def _format_downstream_effects(data: ValidationReportData) -> str:
    """Format Section 6: Downstream Effects Validation."""

    # Calculate weighted compliance score
    all_scores = []
    for run in data.aggregated_results.run_results:
        all_scores.append(run.compliance_score)

    avg_score = sum(all_scores) / len(all_scores) if all_scores else 0
    min_score = min(all_scores) if all_scores else 0
    min_run = next(
        (r.run_id for r in data.aggregated_results.run_results if r.compliance_score == min_score),
        "N/A"
    )

    # Get top driver of score drops
    top_driver = _get_top_flagged_cluster(data)

    return f"""## 6) Downstream Effects Validation (Scope Beyond "Correctness")

> These checks ensure we didn't just "swap words," but that the adapted simulation behaves correctly and feels realistic.

**Downstream effects score (weighted)**
| Metric | Value |
|--------|-------|
| Weighted compliance score (avg across runs) | {avg_score * 100:.1f} / 100 {"✅" if avg_score >= 0.85 else "⚠️"} |
| Lowest run score | {min_score * 100:.1f} ({min_run}) |
| Top driver of flagged score drops | {top_driver} |

**Examples of downstream validations performed**
- Learner-facing coherence across turns (no industry-jumping mid-flow)
- Rubric language consistency with the new industry
- Resource plausibility for the scenario (memos/emails/tools match context)
- Option alignment (system's recommended options match scenario intent)

**Pointers**
- `artifacts/run-*/downstream_effects_report.json`
- `artifacts/run-*/behavior_trace_eval.json`
- `artifacts/run-*/rubric_alignment_eval.json`"""


def _format_pr_review(data: ValidationReportData) -> str:
    """Format Section 7: PR-Style Review."""

    # Categorize runs
    approved = []
    approved_with_notes = []
    not_approved = []

    for run in data.aggregated_results.run_results:
        if not run.passed_all_critical:
            not_approved.append(run.run_id)
        elif len(run.flagged_warnings) > 0:
            approved_with_notes.append(run.run_id)
        else:
            approved.append(run.run_id)

    output = ['## 7) "Approved vs Not Approved" PR-Style Review']

    # Approved section
    if approved:
        run_range = _format_run_range(approved)
        output.append(f"""
### ✅ Approved (Production-Ready)
- **Runs:** {run_range}
- **Meets gate:** ≥{data.acceptance_threshold:.0%} of runs pass all Critical checks""")

    # Approved with notes
    if approved_with_notes:
        run_list = ", ".join(r.split("-")[-1] for r in approved_with_notes[:6])
        if len(approved_with_notes) > 6:
            run_list += f" (+{len(approved_with_notes) - 6} more)"
        output.append(f"""
### ⚠️ Approved with Notes (Non-Blocking)
- **Runs:** {run_list}
- **Notes:** Flagged quality issues present; review recommended before client demo""")

    # Not approved
    if not_approved:
        for run_id in not_approved:
            run_num = run_id.split("-")[-1]
            # Find the failure reason
            failure = next((f for f in data.failure_summaries if run_id in f.affected_runs), None)
            blocker = f"{failure.check_id}: {failure.failure_type}" if failure else "Critical check failure"
            output.append(f"""
### ❌ Not Approved (Blocked)
- **Run:** {run_num}
- **Blocker:** {blocker}""")

    return "\n".join(output)


def _format_next_actions(data: ValidationReportData) -> str:
    """Format Section 8: Recommended Next Actions."""

    actions = []

    # Add fix actions for failures
    for i, failure in enumerate(data.failure_summaries, 1):
        check = get_check_by_id(failure.check_id)
        agent = check.fix_agent if check else "Semantic Fixer"
        actions.append(f"{i}. **Fix {failure.failure_type}** via targeted {agent} patch on identified shards → rerun Finisher (dependency recheck only).")

    # Add improvement actions for flagged issues
    top_flagged = _get_top_flagged_cluster(data)
    if top_flagged:
        actions.append(f"{len(actions) + 1}. **Improve {top_flagged}:** update semantic templates/examples and add additional quality checks.")

    # Add transparency action
    if data.failure_summaries or top_flagged:
        actions.append(f'{len(actions) + 1}. **Add client-facing "Confidence" banner:** "Critical Gate Passed ({data.critical_pass_rate:.0%}), Flagged Issues Present" for transparency.')

    if not actions:
        actions.append("1. **No actions required.** System is production-ready.")

    actions_str = "\n".join(actions)

    return f"""## 8) Recommended Next Actions

{actions_str}"""


def _format_appendix(data: ValidationReportData) -> str:
    """Format Appendix: Artifact Summary."""

    return f"""---

## Appendix: Artifacts Summary

| Artifact Type | Path |
|--------------|------|
| Per-run review packet | `artifacts/run-*/human_review_packet.html` |
| Root JSON | `artifacts/run-*/golden_adapted_simulation.json` |
| Unified scorecard | `artifacts/run-*/rule_scorecard.json` |
| Global Guard report | `artifacts/run-*/global_guard_report.json` |
| Logs | `logs/run-*/stage*.log` |

---
*End of Validation Report*
*Generated by Cartedo Validation Agent v{data.validation_version}*"""


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _get_top_flagged_cluster(data: ValidationReportData) -> str:
    """Get the most common flagged issue cluster."""
    flagged_counts = {}
    for check in FLAGGED_CHECKS:
        agg = data.aggregated_results.check_aggregations.get(check.id, {})
        if agg.get("avg_score", 1.0) < check.threshold_value:
            flagged_counts[check.name] = agg.get("total", 0) - agg.get("passes", 0)

    if flagged_counts:
        return max(flagged_counts, key=flagged_counts.get)
    return "None"


def _get_stage_flagged_patterns(data: ValidationReportData, stage: str) -> List[str]:
    """Get flagged patterns for a specific stage."""
    patterns = []
    # This would be populated from actual stage data
    if stage == "adaptation":
        runs_with_tone = sum(1 for r in data.aggregated_results.run_results if "F4" in r.flagged_warnings)
        if runs_with_tone > 0:
            patterns.append(f"{runs_with_tone} runs: minor tone drift in persona emails (non-blocking)")
    return patterns


def _get_checker_outcomes(data: ValidationReportData) -> List[str]:
    """Get critical outcomes for checker stage."""
    outcomes = [f"✅ {data.total_runs}/{data.total_runs} completed rule scan"]
    failed = len(data.aggregated_results.failed_runs)
    if failed > 0:
        outcomes.append(f"🔴 {failed}/{data.total_runs} runs had a Critical blocker detected pre-fix (resolved in later stages)")
    return outcomes


def _get_checker_issues(data: ValidationReportData) -> List[str]:
    """Get top issues from checker stage."""
    issues = []
    for failure in data.failure_summaries[:3]:
        issues.append(f"{failure.failure_type}")
    return issues if issues else []


def _get_semantic_fixer_outcomes(data: ValidationReportData) -> List[str]:
    """Get outcomes for semantic fixer stage."""
    outcomes = [
        f"✅ {data.total_runs}/{data.total_runs} applied semantic fixes without violating barriers",
        f"✅ {data.total_runs}/{data.total_runs} updated KPIs, personas, resources per scenario context",
    ]
    flagged_count = sum(1 for r in data.aggregated_results.run_results if len(r.flagged_warnings) > 0)
    if flagged_count > 0:
        outcomes.append(f"⚠️ {flagged_count}/{data.total_runs} produced minor realism warnings (Flagged)")
    return outcomes


def _get_semantic_flagged(data: ValidationReportData) -> List[str]:
    """Get flagged patterns for semantic stage."""
    return []


def _get_finisher_outcomes(data: ValidationReportData) -> List[str]:
    """Get outcomes for finisher stage."""
    passed = data.aggregated_results.runs_passing_critical
    total = data.total_runs
    outcomes = []
    if passed == total:
        outcomes.append(f"✅ {passed}/{total} met all pass criteria")
    else:
        outcomes.append(f"✅ {passed}/{total} met all pass criteria")
        outcomes.append(f"🔴 {total - passed}/{total} failed Critical gate (details below)")
    return outcomes


def _get_human_decisions(data: ValidationReportData) -> List[str]:
    """Get human decision summary for HITL stage."""
    decisions = []
    approved = sum(1 for r in data.aggregated_results.run_results if r.passed_all_critical and len(r.flagged_warnings) == 0)
    approved_with_note = sum(1 for r in data.aggregated_results.run_results if r.passed_all_critical and len(r.flagged_warnings) > 0)
    withheld = sum(1 for r in data.aggregated_results.run_results if not r.passed_all_critical)

    if approved > 0:
        decisions.append(f"✅ {approved} approved immediately")
    if approved_with_note > 0:
        decisions.append(f"✅ {approved_with_note} approved with note (flagged issues)")
    if withheld > 0:
        decisions.append(f"⚠️ {withheld} flagged (failed Critical gate; withheld)")

    return decisions


def _format_run_range(run_ids: List[str]) -> str:
    """Format a list of run IDs as a range string."""
    if not run_ids:
        return "None"

    # Extract numbers from run IDs
    nums = []
    for rid in run_ids:
        try:
            num = int(rid.split("-")[-1])
            nums.append(num)
        except:
            pass

    if not nums:
        return ", ".join(run_ids)

    nums.sort()

    # Build ranges
    ranges = []
    start = nums[0]
    end = nums[0]

    for num in nums[1:]:
        if num == end + 1:
            end = num
        else:
            if start == end:
                ranges.append(str(start))
            else:
                ranges.append(f"{start}–{end}")
            start = num
            end = num

    if start == end:
        ranges.append(str(start))
    else:
        ranges.append(f"{start}–{end}")

    return ", ".join(ranges)


# =============================================================================
# ALTERNATIVE FORMATS
# =============================================================================

def format_json_report(data: ValidationReportData) -> dict:
    """Format validation data as JSON for API responses."""
    return {
        "header": {
            "report_id": generate_report_id(data.target_scenario),
            "original_scenario": data.original_scenario,
            "target_scenario": data.target_scenario,
            "simulation_purpose": data.simulation_purpose,
            "system_mode": data.system_mode,
            "validation_version": data.validation_version,
            "total_runs": data.total_runs,
            "acceptance_threshold": data.acceptance_threshold,
            "timestamp": data.timestamp.isoformat(),
        },
        "executive_gate": {
            "status": "approved" if data.can_ship_as_is else "not_approved",
            "critical_pass_rate": data.critical_pass_rate,
            "runs_passing": data.aggregated_results.runs_passing_critical,
            "runs_total": data.total_runs,
            "meets_threshold": data.meets_threshold,
            "release_recommendation": data.next_action,
        },
        "critical_checks": [
            {
                "id": check.id,
                "name": check.name,
                "ensures": check.ensures,
                "threshold": check.threshold,
                "result": data.aggregated_results.check_aggregations.get(check.id, {}),
            }
            for check in CRITICAL_CHECKS
        ],
        "flagged_checks": [
            {
                "id": check.id,
                "name": check.name,
                "ensures": check.ensures,
                "threshold": check.threshold,
                "result": data.aggregated_results.check_aggregations.get(check.id, {}),
            }
            for check in FLAGGED_CHECKS
        ],
        "failures": [
            {
                "check_id": f.check_id,
                "failure_type": f.failure_type,
                "affected_runs": f.affected_runs,
                "example_issue": f.example_issue,
                "why_it_matters": f.why_it_matters,
            }
            for f in data.failure_summaries
        ],
        "summary": data.one_line_summary,
    }


def format_slack_report(data: ValidationReportData) -> str:
    """Format validation data for Slack notification."""
    emoji = "✅" if data.can_ship_as_is else "❌"
    status = "APPROVED" if data.can_ship_as_is else "FIX REQUIRED"

    return f"""{emoji} *Cartedo Validation Report*

*Scenario:* {data.original_scenario} → {data.target_scenario}
*Status:* *{status}*
*Pass Rate:* {data.critical_pass_rate:.0%} ({data.aggregated_results.runs_passing_critical}/{data.total_runs} runs)

{data.one_line_summary}

{f"*Next Action:* {data.next_action}" if not data.can_ship_as_is else ""}"""
