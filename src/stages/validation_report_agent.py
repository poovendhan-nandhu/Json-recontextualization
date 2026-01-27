"""
Validation Report Agent - Generates human-readable canonical validation reports.

This agent produces a NON-TECHNICAL, DECISION-FIRST validation report that allows
PMs, clients, QA, and prompt engineers to quickly and confidently decide whether
a simulation adaptation is ready to ship.

Based on the founder's requirements:
- Audience: PMs, clients, QA - human-readable first
- Must summarize results across multiple runs
- Tiered validation: Critical (blocking) vs Flagged (non-blocking)
- Clear "approved vs not approved" roll-up
- Links to underlying data for drill-down

Uses GPT-5.2 for report generation.
"""
import os
import json
import logging
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import asyncio

# LangSmith tracing
try:
    from langsmith import traceable
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    def traceable(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=4)

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class CheckResult:
    """Result of a single validation check."""
    check_id: str
    check_name: str
    what_it_ensures: str
    threshold: str
    passed: bool
    score: float
    result_summary: str  # e.g., "18/20 runs passed"
    status: str  # "Pass" or "Fail"
    action_needed: str
    is_critical: bool  # True = Critical, False = Flagged
    issues: list[str] = field(default_factory=list)
    locations: list[str] = field(default_factory=list)


@dataclass
class RunResult:
    """Result of a single run (end-to-end adaptation)."""
    run_id: str
    run_number: int
    passed_all_critical: bool
    critical_pass_count: int
    critical_total: int
    flagged_pass_count: int
    flagged_total: int
    check_results: dict  # check_id -> CheckResult


@dataclass
class FailureSummary:
    """Summary of a failure for actionable reporting."""
    failure_type: str
    affected_runs: str
    example_issue: str
    why_it_matters: str
    where_it_happens: str
    detection_stage: str
    fix_scope: str
    structural_risk: str


@dataclass
class RecommendedFix:
    """A recommended fix with priority."""
    priority: str  # P0, P1, P2
    recommendation: str
    target_agent: str
    expected_impact: str


@dataclass
class ValidationReportData:
    """Complete validation report data structure."""
    # Header metadata
    original_scenario: str
    target_scenario: str
    simulation_purpose: str
    system_mode: str
    validation_version: str
    total_runs: int
    acceptance_threshold: float
    timestamp: datetime

    # Executive decision
    critical_pass_rate: float
    meets_threshold: bool
    has_blocking_issues: bool
    release_decision: str  # "Approved" / "Fix Required" / "Blocked"
    system_verdict: str

    # Check results
    critical_checks: list[CheckResult]
    flagged_checks: list[CheckResult]

    # Failure analysis
    failure_summaries: list[FailureSummary]
    key_insight: str

    # Recommendations
    recommendations: list[RecommendedFix]

    # Binary decision
    can_ship_as_is: bool
    is_failure_scoped: bool
    is_fix_isolated: bool
    can_automation_rerun: bool
    system_instruction: str

    # Final summary
    one_line_summary: str


# ============================================================================
# CRITICAL AND FLAGGED CHECK DEFINITIONS
# ============================================================================

CRITICAL_CHECKS = [
    {
        "id": "C1",
        "name": "Entity Removal",
        "what_it_ensures": "No original scenario references remain in adapted simulation",
        "threshold": "100%",
        "maps_to_agent": "Domain Fidelity",
    },
    {
        "id": "C2",
        "name": "KPI Alignment",
        "what_it_ensures": "Industry KPIs correctly updated to match target scenario",
        "threshold": "100%",
        "maps_to_agent": "Domain Fidelity",
    },
    {
        "id": "C3",
        "name": "Schema Validity",
        "what_it_ensures": "Output JSON conforms to expected schema structure",
        "threshold": "100%",
        "maps_to_agent": "Completeness",
    },
    {
        "id": "C4",
        "name": "Rubric Integrity",
        "what_it_ensures": "Rubric levels and scoring preserved correctly",
        "threshold": "100%",
        "maps_to_agent": "Completeness",
    },
    {
        "id": "C5",
        "name": "End-to-End Executability",
        "what_it_ensures": "No missing references that would break simulation execution",
        "threshold": "100%",
        "maps_to_agent": "Completeness",
    },
    {
        "id": "C6",
        "name": "Barrier Compliance",
        "what_it_ensures": "Locked elements (workspace_ids, scenario_options) never modified",
        "threshold": "100%",
        "maps_to_agent": "Consistency",
    },
    {
        "id": "C7",
        "name": "KLO Preservation",
        "what_it_ensures": "Key Learning Outcomes preserved and aligned with questions",
        "threshold": ">=95%",
        "maps_to_agent": "KLO-Question Alignment",
    },
    {
        "id": "C8",
        "name": "Resource Completeness",
        "what_it_ensures": "All resources exist with valid content (no truncation)",
        "threshold": "100%",
        "maps_to_agent": "Resource Quality",
    },
]

FLAGGED_CHECKS = [
    {
        "id": "F1",
        "name": "Persona Realism",
        "what_it_measures": "Characters and personas feel realistic for target industry",
        "threshold": ">=85%",
        "maps_to_agent": "Context Fidelity",
    },
    {
        "id": "F2",
        "name": "Resource Authenticity",
        "what_it_measures": "Resources provide data (not answers) with realistic figures",
        "threshold": ">=85%",
        "maps_to_agent": "Resource Quality",
    },
    {
        "id": "F3",
        "name": "Narrative Coherence",
        "what_it_measures": "Story elements connect logically across simulation",
        "threshold": ">=90%",
        "maps_to_agent": "Consistency",
    },
    {
        "id": "F4",
        "name": "Tone Consistency",
        "what_it_measures": "Professional tone maintained throughout",
        "threshold": ">=90%",
        "maps_to_agent": "Context Fidelity",
    },
    {
        "id": "F5",
        "name": "Data Realism",
        "what_it_measures": "Numbers and statistics are realistic for the industry",
        "threshold": ">=85%",
        "maps_to_agent": "Resource Quality",
    },
    {
        "id": "F6",
        "name": "Industry Terminology",
        "what_it_measures": "Correct domain terminology used throughout",
        "threshold": ">=90%",
        "maps_to_agent": "Domain Fidelity",
    },
]


# ============================================================================
# OPENAI CLIENT
# ============================================================================

def _get_openai_client():
    from openai import OpenAI
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


async def _call_gpt_async(prompt: str, system: str = "You are a validation report generator.") -> str:
    """Call GPT-5.2 asynchronously."""
    loop = asyncio.get_event_loop()

    def _call():
        client = _get_openai_client()
        response = client.chat.completions.create(
            model="gpt-5.2-2025-12-11",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
        )
        return response.choices[0].message.content

    return await loop.run_in_executor(_executor, _call)


# ============================================================================
# REPORT GENERATION FROM VALIDATION DATA
# ============================================================================

def map_agent_results_to_checks(
    agent_results: list,  # List of AgentResult from simple_validators
    run_count: int = 1
) -> tuple[list[CheckResult], list[CheckResult]]:
    """
    Map validation agent results to Critical and Flagged checks.

    Args:
        agent_results: List of AgentResult objects from simple_validators
        run_count: Number of runs evaluated (for aggregated reporting)

    Returns:
        Tuple of (critical_checks, flagged_checks)
    """
    # Build agent name -> result lookup
    agent_lookup = {}
    for ar in agent_results:
        agent_lookup[ar.agent_name] = ar

    critical_checks = []
    for check_def in CRITICAL_CHECKS:
        agent_name = check_def["maps_to_agent"]
        agent_result = agent_lookup.get(agent_name)

        if agent_result:
            passed = agent_result.passed
            score = agent_result.score
            issues = [i.issue for i in agent_result.issues] if agent_result.issues else []
            locations = [i.location for i in agent_result.issues] if agent_result.issues else []
        else:
            passed = True
            score = 1.0
            issues = []
            locations = []

        # Calculate result summary
        passed_count = run_count if passed else 0
        result_summary = f"{passed_count}/{run_count}" if run_count > 1 else ("Pass" if passed else "Fail")

        critical_checks.append(CheckResult(
            check_id=check_def["id"],
            check_name=check_def["name"],
            what_it_ensures=check_def["what_it_ensures"],
            threshold=check_def["threshold"],
            passed=passed,
            score=score,
            result_summary=result_summary,
            status="Pass" if passed else "Fail",
            action_needed="None" if passed else f"Fix {len(issues)} issues",
            is_critical=True,
            issues=issues[:5],  # Top 5 issues
            locations=locations[:5],
        ))

    flagged_checks = []
    for check_def in FLAGGED_CHECKS:
        agent_name = check_def["maps_to_agent"]
        agent_result = agent_lookup.get(agent_name)

        if agent_result:
            passed = agent_result.score >= 0.85  # Flagged threshold
            score = agent_result.score
            issues = [i.issue for i in agent_result.issues] if agent_result.issues else []
            locations = [i.location for i in agent_result.issues] if agent_result.issues else []
        else:
            passed = True
            score = 1.0
            issues = []
            locations = []

        flagged_checks.append(CheckResult(
            check_id=check_def["id"],
            check_name=check_def["name"],
            what_it_ensures=check_def["what_it_measures"],
            threshold=check_def["threshold"],
            passed=passed,
            score=score,
            result_summary=f"{score:.0%}",
            status="Pass" if passed else "Warning",
            action_needed="None" if passed else "Review recommended",
            is_critical=False,
            issues=issues[:3],
            locations=locations[:3],
        ))

    return critical_checks, flagged_checks


def calculate_release_decision(
    critical_checks: list[CheckResult],
    total_runs: int = 1,
    acceptance_threshold: float = 0.95
) -> tuple[float, bool, str]:
    """
    Calculate release decision based on critical checks.

    Returns:
        Tuple of (critical_pass_rate, meets_threshold, release_decision)
    """
    passed_critical = sum(1 for c in critical_checks if c.passed)
    total_critical = len(critical_checks)

    # For single run, use check pass rate
    # For multi-run, this would be (runs with all critical passing) / total_runs
    critical_pass_rate = passed_critical / total_critical if total_critical > 0 else 1.0

    meets_threshold = critical_pass_rate >= acceptance_threshold

    if critical_pass_rate == 1.0:
        release_decision = "Approved"
    elif critical_pass_rate >= acceptance_threshold:
        release_decision = "Approved with Notes"
    elif critical_pass_rate >= 0.8:
        release_decision = "Fix Required"
    else:
        release_decision = "Blocked"

    return critical_pass_rate, meets_threshold, release_decision


def generate_key_insight(critical_checks: list[CheckResult]) -> str:
    """Generate key insight about dominant failure pattern."""
    failing_checks = [c for c in critical_checks if not c.passed]

    if not failing_checks:
        return "All critical checks passed. System is operating within acceptable parameters."

    # Find the most common issue pattern
    all_issues = []
    for check in failing_checks:
        all_issues.extend(check.issues)

    if not all_issues:
        check_names = [c.check_name for c in failing_checks]
        return f"Failures detected in: {', '.join(check_names)}. Review the specific checks for details."

    # Categorize issues
    domain_issues = sum(1 for i in all_issues if any(x in i.lower() for x in ['term', 'industry', 'domain', 'hr', 'hiring']))
    content_issues = sum(1 for i in all_issues if any(x in i.lower() for x in ['truncat', 'empty', 'missing', 'placeholder']))
    alignment_issues = sum(1 for i in all_issues if any(x in i.lower() for x in ['klo', 'question', 'align']))

    if domain_issues > content_issues and domain_issues > alignment_issues:
        return f"Dominant pattern: SOURCE TERMINOLOGY LEAKAGE. Found {domain_issues} instances of incorrect domain terms that need replacement."
    elif content_issues > alignment_issues:
        return f"Dominant pattern: CONTENT COMPLETENESS. Found {content_issues} instances of incomplete or truncated content."
    elif alignment_issues > 0:
        return f"Dominant pattern: KLO-QUESTION MISALIGNMENT. Found {alignment_issues} alignment issues between learning outcomes and assessment questions."
    else:
        return f"Multiple issue patterns detected across {len(failing_checks)} checks. Review each failing check for specific remediation."


def generate_failure_summaries(critical_checks: list[CheckResult]) -> list[FailureSummary]:
    """Generate actionable failure summaries for failing checks."""
    summaries = []

    for check in critical_checks:
        if check.passed:
            continue

        # Map check to failure details
        if check.check_id == "C1":
            summaries.append(FailureSummary(
                failure_type="Source Entity Leakage",
                affected_runs=check.result_summary,
                example_issue=check.issues[0] if check.issues else "Original scenario terms found in output",
                why_it_matters="Users will see references to the wrong industry, breaking immersion",
                where_it_happens=check.locations[0] if check.locations else "Multiple locations in adapted JSON",
                detection_stage="Domain Fidelity Validator",
                fix_scope="String replacement across affected shards",
                structural_risk="Low - text changes only",
            ))
        elif check.check_id == "C7":
            summaries.append(FailureSummary(
                failure_type="KLO-Question Misalignment",
                affected_runs=check.result_summary,
                example_issue=check.issues[0] if check.issues else "Questions don't assess the stated KLOs",
                why_it_matters="Students won't be assessed on what they're supposed to learn",
                where_it_happens="simulation_flow questions and assessment_criterion",
                detection_stage="KLO-Question Alignment Validator",
                fix_scope="Question rewriting to match KLO intent",
                structural_risk="Medium - content changes required",
            ))
        elif check.check_id == "C8":
            summaries.append(FailureSummary(
                failure_type="Resource Truncation/Missing Content",
                affected_runs=check.result_summary,
                example_issue=check.issues[0] if check.issues else "Resource content incomplete or missing",
                why_it_matters="Students won't have enough data to complete activities",
                where_it_happens=check.locations[0] if check.locations else "simulation_flow resources",
                detection_stage="Resource Quality Validator",
                fix_scope="Content regeneration for affected resources",
                structural_risk="Medium - content regeneration required",
            ))
        else:
            summaries.append(FailureSummary(
                failure_type=check.check_name,
                affected_runs=check.result_summary,
                example_issue=check.issues[0] if check.issues else f"{check.check_name} validation failed",
                why_it_matters=check.what_it_ensures,
                where_it_happens=check.locations[0] if check.locations else "See validation details",
                detection_stage=f"{check.check_name} Validator",
                fix_scope="Review and fix identified issues",
                structural_risk="Varies",
            ))

    return summaries


def generate_recommendations(failure_summaries: list[FailureSummary]) -> list[RecommendedFix]:
    """Generate prioritized fix recommendations."""
    recommendations = []

    for i, failure in enumerate(failure_summaries):
        priority = "P0" if i == 0 else ("P1" if i < 3 else "P2")

        # Map failure type to target agent
        if "Entity" in failure.failure_type or "Terminology" in failure.failure_type:
            target_agent = "Semantic Fixer"
            expected_impact = "Remove all source scenario references"
        elif "KLO" in failure.failure_type:
            target_agent = "Alignment Fixer"
            expected_impact = "Align questions with learning outcomes"
        elif "Resource" in failure.failure_type or "Truncation" in failure.failure_type:
            target_agent = "Resource Quality Fixer"
            expected_impact = "Regenerate complete resource content"
        elif "Schema" in failure.failure_type or "Structure" in failure.failure_type:
            target_agent = "Structural Fixer"
            expected_impact = "Restore valid JSON structure"
        else:
            target_agent = "General Repair Agent"
            expected_impact = "Address identified issues"

        recommendations.append(RecommendedFix(
            priority=priority,
            recommendation=f"Fix {failure.failure_type}: {failure.example_issue[:80]}...",
            target_agent=target_agent,
            expected_impact=expected_impact,
        ))

    return recommendations


# ============================================================================
# MARKDOWN REPORT FORMATTER
# ============================================================================

@traceable(name="generate_validation_report_markdown", run_type="chain")
async def generate_validation_report_markdown(
    agent_results: list,  # List of AgentResult from simple_validators
    original_scenario: str,
    target_scenario: str,
    simulation_purpose: str = "Business Simulation Training",
    total_runs: int = 1,
    acceptance_threshold: float = 0.95,
) -> str:
    """
    Generate the canonical validation report in Markdown format.

    This produces a human-readable report following the founder's template:
    1. Canonical Header
    2. Executive Decision Gate
    3. Critical Checks Dashboard
    4. Flagged Quality Checks
    5. What Failed
    6. Recommended Fixes
    7. Binary System Decision
    8. Final One-Line Summary

    Args:
        agent_results: List of AgentResult from simple_validators
        original_scenario: Source scenario description
        target_scenario: Target scenario description
        simulation_purpose: Purpose of the simulation
        total_runs: Number of runs evaluated
        acceptance_threshold: Required pass rate (default 95%)

    Returns:
        Markdown formatted validation report
    """
    logger.info("[REPORT] Generating canonical validation report...")

    # Map agent results to checks
    critical_checks, flagged_checks = map_agent_results_to_checks(agent_results, total_runs)

    # Calculate release decision
    critical_pass_rate, meets_threshold, release_decision = calculate_release_decision(
        critical_checks, total_runs, acceptance_threshold
    )

    # Generate insights and recommendations
    key_insight = generate_key_insight(critical_checks)
    failure_summaries = generate_failure_summaries(critical_checks)
    recommendations = generate_recommendations(failure_summaries)

    # Determine binary decisions
    has_blocking_issues = any(not c.passed for c in critical_checks)
    can_ship = release_decision == "Approved"
    is_failure_scoped = len(failure_summaries) <= 2
    is_fix_isolated = all(f.structural_risk in ["Low", "Medium"] for f in failure_summaries)
    can_automation_rerun = is_failure_scoped and is_fix_isolated

    # Generate system verdict
    if can_ship:
        system_verdict = f"System is READY FOR RELEASE. All critical checks passed with {critical_pass_rate:.0%} compliance."
    elif release_decision == "Fix Required":
        system_verdict = f"System requires FIXES before release. {len(failure_summaries)} issue(s) identified. {critical_pass_rate:.0%} critical pass rate."
    else:
        system_verdict = f"System is BLOCKED. Multiple critical failures detected. {critical_pass_rate:.0%} critical pass rate."

    # Generate system instruction
    if can_ship:
        system_instruction = "Proceed to deployment. No manual intervention required."
    elif can_automation_rerun:
        system_instruction = f"Run automated fix cycle. Target agents: {', '.join(set(r.target_agent for r in recommendations[:3]))}."
    else:
        system_instruction = "Escalate to human review. Issues require manual inspection before proceeding."

    # Generate one-line summary
    if can_ship:
        one_line_summary = f"Simulation adaptation from '{original_scenario[:30]}...' to '{target_scenario[:30]}...' PASSED all critical validation checks and is ready for deployment."
    else:
        failing_count = sum(1 for c in critical_checks if not c.passed)
        one_line_summary = f"Simulation adaptation REQUIRES FIXES: {failing_count} critical check(s) failed. Primary issue: {failure_summaries[0].failure_type if failure_summaries else 'See details'}."

    # Build the Markdown report
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

    report = f"""# Cartedo Simulation Adaptation
## Canonical Validation Output (Standard Contract)

---

### 1. Canonical Header (Contract Metadata)

| Field | Value |
|-------|-------|
| **Original Scenario** | {original_scenario[:80]}{'...' if len(original_scenario) > 80 else ''} |
| **Target Scenario** | {target_scenario[:80]}{'...' if len(target_scenario) > 80 else ''} |
| **Simulation Purpose** | {simulation_purpose} |
| **System Mode** | Fully Automated |
| **Validation Version** | v2.0 |
| **Total Runs Evaluated** | {total_runs} |
| **Acceptance Threshold** | {acceptance_threshold:.0%} |
| **Validation Timestamp** | {timestamp} |

---

### 2. Executive Decision Gate (Single-Glance)

| Metric | Value |
|--------|-------|
| **Critical Pass Rate** | {critical_pass_rate:.0%} |
| **Acceptance Threshold Met** | {'Yes' if meets_threshold else 'No'} |
| **Blocking Issues Present** | {'Yes' if has_blocking_issues else 'No'} |
| **Overall Release Decision** | **{release_decision}** |

**System Verdict:** {system_verdict}

---

### 3. Critical Checks Dashboard (Non-Negotiable)

| Check ID | What This Check Ensures | Threshold | Result | Status | Action Needed |
|----------|------------------------|-----------|--------|--------|---------------|
"""

    for check in critical_checks:
        status_emoji = "PASS" if check.passed else "FAIL"
        report += f"| {check.check_id} | {check.what_it_ensures[:50]}{'...' if len(check.what_it_ensures) > 50 else ''} | {check.threshold} | {check.result_summary} | {status_emoji} | {check.action_needed} |\n"

    report += f"""
**Key Insight:** {key_insight}

---

### 4. Flagged Quality Checks (Non-Blocking Signals)

| Check ID | What This Measures | Threshold | Average Score | Status | Recommendation |
|----------|-------------------|-----------|---------------|--------|----------------|
"""

    for check in flagged_checks:
        status_emoji = "PASS" if check.passed else "REVIEW"
        report += f"| {check.check_id} | {check.what_it_ensures[:40]}{'...' if len(check.what_it_ensures) > 40 else ''} | {check.threshold} | {check.score:.0%} | {status_emoji} | {check.action_needed} |\n"

    report += """
*These checks influence realism and polish but do NOT block release.*

---

"""

    # Section 5: What Failed (only if there are failures)
    if failure_summaries:
        report += """### 5. What Failed (Actionable Failure Summary)

"""
        for i, failure in enumerate(failure_summaries, 1):
            report += f"""**Failure {i}: {failure.failure_type}**

| Attribute | Details |
|-----------|---------|
| Affected Runs | {failure.affected_runs} |
| Example Issue | {failure.example_issue[:100]}{'...' if len(failure.example_issue) > 100 else ''} |
| Why This Matters | {failure.why_it_matters} |
| Where It Happens | {failure.where_it_happens} |
| Detection Stage | {failure.detection_stage} |
| Fix Scope | {failure.fix_scope} |
| Structural Risk | {failure.structural_risk} |

"""
    else:
        report += """### 5. What Failed (Actionable Failure Summary)

*No critical failures detected. All checks passed.*

---

"""

    # Section 6: Recommended Fixes
    report += """### 6. Recommended Fixes (Auto-Generated)

| Priority | Recommendation | Target Agent | Expected Impact |
|----------|----------------|--------------|-----------------|
"""

    if recommendations:
        for rec in recommendations[:5]:
            report += f"| {rec.priority} | {rec.recommendation[:60]}{'...' if len(rec.recommendation) > 60 else ''} | {rec.target_agent} | {rec.expected_impact} |\n"
    else:
        report += "| - | No fixes required | - | - |\n"

    report += f"""

---

### 7. Binary System Decision & Next Action

| Question | Answer |
|----------|--------|
| Can this ship as-is? | {'Yes' if can_ship else 'No'} |
| Is the failure well-scoped? | {'Yes' if is_failure_scoped else 'No'} |
| Is the fix isolated? | {'Yes' if is_fix_isolated else 'No'} |
| Can automation safely rerun? | {'Yes' if can_automation_rerun else 'No'} |

**System Instruction:** {system_instruction}

---

### 8. Final One-Line System Summary (Canonical)

> {one_line_summary}

---

*Report generated by Cartedo Validation Agent v2.0*
"""

    logger.info(f"[REPORT] Generated report: {release_decision}, {critical_pass_rate:.0%} pass rate")
    return report


# ============================================================================
# JSON REPORT FORMATTER
# ============================================================================

@traceable(name="generate_validation_report_json", run_type="chain")
async def generate_validation_report_json(
    agent_results: list,
    original_scenario: str,
    target_scenario: str,
    simulation_purpose: str = "Business Simulation Training",
    total_runs: int = 1,
    acceptance_threshold: float = 0.95,
) -> dict:
    """
    Generate the validation report as structured JSON.

    Returns a JSON object matching the ValidationReportData structure,
    suitable for programmatic consumption and downstream processing.
    """
    logger.info("[REPORT] Generating JSON validation report...")

    # Map agent results to checks
    critical_checks, flagged_checks = map_agent_results_to_checks(agent_results, total_runs)

    # Calculate release decision
    critical_pass_rate, meets_threshold, release_decision = calculate_release_decision(
        critical_checks, total_runs, acceptance_threshold
    )

    # Generate insights and recommendations
    key_insight = generate_key_insight(critical_checks)
    failure_summaries = generate_failure_summaries(critical_checks)
    recommendations = generate_recommendations(failure_summaries)

    # Determine binary decisions
    has_blocking_issues = any(not c.passed for c in critical_checks)
    can_ship = release_decision == "Approved"
    is_failure_scoped = len(failure_summaries) <= 2
    is_fix_isolated = all(f.structural_risk in ["Low", "Medium"] for f in failure_summaries)
    can_automation_rerun = is_failure_scoped and is_fix_isolated

    # Generate summaries
    if can_ship:
        system_verdict = f"System is READY FOR RELEASE. All critical checks passed with {critical_pass_rate:.0%} compliance."
        system_instruction = "Proceed to deployment. No manual intervention required."
        one_line_summary = f"Simulation adaptation PASSED all critical validation checks and is ready for deployment."
    elif can_automation_rerun:
        system_verdict = f"System requires FIXES before release. {critical_pass_rate:.0%} critical pass rate."
        system_instruction = f"Run automated fix cycle. Target agents: {', '.join(set(r.target_agent for r in recommendations[:3]))}."
        one_line_summary = f"Simulation adaptation REQUIRES FIXES: {sum(1 for c in critical_checks if not c.passed)} critical check(s) failed."
    else:
        system_verdict = f"System is BLOCKED. Multiple critical failures detected. {critical_pass_rate:.0%} critical pass rate."
        system_instruction = "Escalate to human review. Issues require manual inspection before proceeding."
        one_line_summary = f"Simulation adaptation BLOCKED: Multiple critical failures require manual review."

    return {
        "header": {
            "original_scenario": original_scenario,
            "target_scenario": target_scenario,
            "simulation_purpose": simulation_purpose,
            "system_mode": "Fully Automated",
            "validation_version": "v2.0",
            "total_runs": total_runs,
            "acceptance_threshold": acceptance_threshold,
            "timestamp": datetime.now().isoformat(),
        },
        "executive_decision": {
            "critical_pass_rate": critical_pass_rate,
            "meets_threshold": meets_threshold,
            "has_blocking_issues": has_blocking_issues,
            "release_decision": release_decision,
            "system_verdict": system_verdict,
        },
        "critical_checks": [
            {
                "check_id": c.check_id,
                "check_name": c.check_name,
                "what_it_ensures": c.what_it_ensures,
                "threshold": c.threshold,
                "passed": c.passed,
                "score": c.score,
                "result_summary": c.result_summary,
                "status": c.status,
                "action_needed": c.action_needed,
                "issues": c.issues,
                "locations": c.locations,
            }
            for c in critical_checks
        ],
        "flagged_checks": [
            {
                "check_id": c.check_id,
                "check_name": c.check_name,
                "what_it_measures": c.what_it_ensures,
                "threshold": c.threshold,
                "passed": c.passed,
                "score": c.score,
                "status": c.status,
                "recommendation": c.action_needed,
            }
            for c in flagged_checks
        ],
        "failure_analysis": {
            "key_insight": key_insight,
            "failures": [
                {
                    "failure_type": f.failure_type,
                    "affected_runs": f.affected_runs,
                    "example_issue": f.example_issue,
                    "why_it_matters": f.why_it_matters,
                    "where_it_happens": f.where_it_happens,
                    "detection_stage": f.detection_stage,
                    "fix_scope": f.fix_scope,
                    "structural_risk": f.structural_risk,
                }
                for f in failure_summaries
            ],
        },
        "recommendations": [
            {
                "priority": r.priority,
                "recommendation": r.recommendation,
                "target_agent": r.target_agent,
                "expected_impact": r.expected_impact,
            }
            for r in recommendations
        ],
        "binary_decision": {
            "can_ship_as_is": can_ship,
            "is_failure_scoped": is_failure_scoped,
            "is_fix_isolated": is_fix_isolated,
            "can_automation_rerun": can_automation_rerun,
            "system_instruction": system_instruction,
        },
        "summary": {
            "one_line_summary": one_line_summary,
        },
    }


# ============================================================================
# MULTI-RUN AGGREGATION
# ============================================================================

@traceable(name="aggregate_multi_run_results", run_type="chain")
async def aggregate_multi_run_results(
    run_results: list[dict],  # List of {agent_results, ...} from multiple runs
    original_scenario: str,
    target_scenario: str,
    acceptance_threshold: float = 0.95,
) -> dict:
    """
    Aggregate validation results across multiple runs.

    This produces a summary showing:
    - How many runs passed all critical checks
    - Per-check pass rates across all runs
    - Common failure patterns

    Args:
        run_results: List of run result dicts, each containing agent_results
        original_scenario: Source scenario
        target_scenario: Target scenario
        acceptance_threshold: Required pass rate

    Returns:
        Aggregated report dict
    """
    total_runs = len(run_results)

    if total_runs == 0:
        return {"error": "No run results provided"}

    # Aggregate per-check results
    check_pass_counts = {}  # check_id -> count of runs where it passed
    check_scores = {}  # check_id -> list of scores

    runs_passing_all_critical = 0

    for run_idx, run_result in enumerate(run_results):
        agent_results = run_result.get("agent_results", [])
        critical_checks, _ = map_agent_results_to_checks(agent_results, 1)

        all_passed = True
        for check in critical_checks:
            check_id = check.check_id
            if check_id not in check_pass_counts:
                check_pass_counts[check_id] = 0
                check_scores[check_id] = []

            if check.passed:
                check_pass_counts[check_id] += 1
            else:
                all_passed = False

            check_scores[check_id].append(check.score)

        if all_passed:
            runs_passing_all_critical += 1

    # Calculate overall critical pass rate
    critical_pass_rate = runs_passing_all_critical / total_runs
    meets_threshold = critical_pass_rate >= acceptance_threshold

    # Determine release decision
    if critical_pass_rate >= 0.95:
        release_decision = "Approved"
    elif critical_pass_rate >= 0.80:
        release_decision = "Fix Required"
    else:
        release_decision = "Blocked"

    # Build per-check summary
    check_summaries = []
    for check_def in CRITICAL_CHECKS:
        check_id = check_def["id"]
        passed_count = check_pass_counts.get(check_id, 0)
        scores = check_scores.get(check_id, [])
        avg_score = sum(scores) / len(scores) if scores else 0.0

        check_summaries.append({
            "check_id": check_id,
            "check_name": check_def["name"],
            "runs_passed": passed_count,
            "total_runs": total_runs,
            "pass_rate": passed_count / total_runs,
            "average_score": avg_score,
            "status": "Pass" if passed_count == total_runs else "Fail",
        })

    return {
        "summary": {
            "total_runs": total_runs,
            "runs_passing_all_critical": runs_passing_all_critical,
            "critical_pass_rate": critical_pass_rate,
            "meets_acceptance_threshold": meets_threshold,
            "release_decision": release_decision,
        },
        "per_check_results": check_summaries,
        "scenarios": {
            "original": original_scenario,
            "target": target_scenario,
        },
        "acceptance_threshold": acceptance_threshold,
    }
