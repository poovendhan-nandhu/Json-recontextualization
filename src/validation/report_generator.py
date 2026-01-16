"""
Validation Report Generator

Generates the canonical validation report in the standardized format.
Designed for PM/Client/QA decision-making, not debugging.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Any

from .check_definitions import (
    CRITICAL_CHECKS,
    FLAGGED_CHECKS,
    CheckDefinition,
    CheckTier,
    CheckStatus,
    get_check_by_id,
)
from .check_runner import (
    ValidationCheckRunner,
    RunValidationResult,
    CheckResult,
)


@dataclass
class AggregatedResults:
    """Aggregated validation results for report generation."""
    critical_passed: int
    critical_total: int
    flagged_passed: int
    flagged_total: int
    total_blocker_issues: int
    total_warning_issues: int
    total_info_issues: int
    runs_checked: int
    check_results: dict = field(default_factory=dict)

logger = logging.getLogger(__name__)


@dataclass
class FailureSummary:
    """Summary of a single failure for the report."""
    check_id: str
    failure_type: str
    affected_runs: list[str]
    example_issue: str
    why_it_matters: str
    where_it_happens: str
    detection_stage: str
    fix_scope: str
    structural_risk: str


@dataclass
class RecommendedFix:
    """A recommended fix action."""
    priority: str  # P0, P1, P2
    recommendation: str
    target_agent: str
    expected_impact: str


@dataclass
class ValidationReportData:
    """All data needed to generate the report."""

    # Header info
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

    # Aggregated results
    aggregated_results: AggregatedResults

    # Fields with defaults (must come after non-default fields)
    audience: str = "PM / Client / QA"  # Target audience for the report
    failure_summaries: list[FailureSummary] = field(default_factory=list)

    # Recommendations
    recommendations: list[RecommendedFix] = field(default_factory=list)

    # Binary decision
    can_ship_as_is: bool = False
    failure_well_scoped: bool = True
    fix_isolated: bool = True
    automation_can_rerun: bool = True
    next_action: str = ""

    # Final summary
    one_line_summary: str = ""


class ValidationReportGenerator:
    """
    Generates canonical validation reports.

    The report format follows the exact contract specified for
    PM/Client/QA consumption.
    """

    def __init__(
        self,
        original_scenario: str,
        target_scenario: str,
        simulation_purpose: str = "Training Simulation",
        system_mode: str = "Fully automated",
        acceptance_threshold: float = 0.95,
    ):
        self.original_scenario = original_scenario
        self.target_scenario = target_scenario
        self.simulation_purpose = simulation_purpose
        self.system_mode = system_mode
        self.acceptance_threshold = acceptance_threshold
        self.validation_version = "1.0"

        self.check_runner = ValidationCheckRunner(acceptance_threshold)

    def generate_report_data(
        self,
        run_results: list[RunValidationResult],
    ) -> ValidationReportData:
        """
        Generate all data needed for the report.

        Args:
            run_results: List of validation results from multiple runs

        Returns:
            ValidationReportData ready for formatting
        """
        # Aggregate results
        aggregated = self.check_runner.aggregate_results(run_results)

        # Determine release decision
        if aggregated.meets_acceptance and len(aggregated.failed_runs) == 0:
            release_decision = "Approved"
            has_blocking = False
        elif aggregated.meets_acceptance:
            release_decision = "Approved"  # Met threshold even with some failures
            has_blocking = True
        else:
            release_decision = "Fix Required"
            has_blocking = True

        # Generate system verdict
        if release_decision == "Approved" and not has_blocking:
            system_verdict = f"System is production-ready. {aggregated.critical_pass_rate:.0%} of runs passed all critical checks."
        elif release_decision == "Approved":
            system_verdict = f"System meets acceptance bar ({aggregated.critical_pass_rate:.0%} ≥ {self.acceptance_threshold:.0%}). Minor issues flagged for review."
        else:
            system_verdict = f"System requires fixes. Only {aggregated.critical_pass_rate:.0%} of runs passed (threshold: {self.acceptance_threshold:.0%})."

        # Generate failure summaries
        failure_summaries = self._generate_failure_summaries(aggregated)

        # Generate recommendations
        recommendations = self._generate_recommendations(aggregated, failure_summaries)

        # Binary decision
        can_ship = release_decision == "Approved"
        failure_scoped = len(aggregated.failed_runs) <= 2
        fix_isolated = len(set(f.check_id for f in failure_summaries)) <= 2

        if can_ship:
            next_action = "Proceed to deployment. No blocking issues."
        elif failure_scoped and fix_isolated:
            next_action = f"Rerun {recommendations[0].target_agent if recommendations else 'Semantic Fixer'} on affected runs, then re-validate."
        else:
            next_action = "Review failure patterns and run targeted fixes before re-validation."

        # One-line summary
        if can_ship and not has_blocking:
            one_line = f"Adaptation from {self.original_scenario} to {self.target_scenario} is complete and production-ready."
        elif can_ship:
            one_line = f"Adaptation meets acceptance bar ({aggregated.critical_pass_rate:.0%}); minor quality issues flagged for optional review."
        else:
            one_line = f"Adaptation requires fixes: {len(aggregated.failed_runs)} of {aggregated.total_runs} runs failed critical checks."

        return ValidationReportData(
            original_scenario=self.original_scenario,
            target_scenario=self.target_scenario,
            simulation_purpose=self.simulation_purpose,
            system_mode=self.system_mode,
            validation_version=self.validation_version,
            total_runs=aggregated.total_runs,
            acceptance_threshold=self.acceptance_threshold,
            timestamp=datetime.now(),
            critical_pass_rate=aggregated.critical_pass_rate,
            meets_threshold=aggregated.meets_acceptance,
            has_blocking_issues=has_blocking,
            release_decision=release_decision,
            system_verdict=system_verdict,
            aggregated_results=aggregated,
            failure_summaries=failure_summaries,
            recommendations=recommendations,
            can_ship_as_is=can_ship,
            failure_well_scoped=failure_scoped,
            fix_isolated=fix_isolated,
            automation_can_rerun=True,
            next_action=next_action,
            one_line_summary=one_line,
        )

    def _generate_failure_summaries(
        self,
        aggregated: AggregatedResults,
    ) -> list[FailureSummary]:
        """Generate human-readable failure summaries."""
        summaries = []

        # Group failures by check
        failure_by_check = {}
        for run_result in aggregated.run_results:
            for check_id in run_result.critical_failures:
                if check_id not in failure_by_check:
                    failure_by_check[check_id] = []
                failure_by_check[check_id].append(run_result)

        for check_id, failed_runs in failure_by_check.items():
            check = get_check_by_id(check_id)
            if not check:
                continue

            # Get example issue from first failed run
            example_issue = "No specific issue captured"
            if failed_runs and check_id in failed_runs[0].check_results:
                issues = failed_runs[0].check_results[check_id].issues_found
                if issues:
                    example_issue = issues[0]

            summaries.append(FailureSummary(
                check_id=check_id,
                failure_type=check.name,
                affected_runs=[r.run_id for r in failed_runs],
                example_issue=example_issue,
                why_it_matters=check.why_it_matters,
                where_it_happens="Simulation content",
                detection_stage=check.detection_stage,
                fix_scope=f"Run {check.fix_agent}",
                structural_risk="Low" if len(failed_runs) <= 2 else "Medium",
            ))

        return summaries

    def _generate_recommendations(
        self,
        aggregated: AggregatedResults,
        failures: list[FailureSummary],
    ) -> list[RecommendedFix]:
        """Generate prioritized fix recommendations."""
        recommendations = []

        # P0: Critical failures
        for failure in failures:
            check = get_check_by_id(failure.check_id)
            if not check:
                continue

            recommendations.append(RecommendedFix(
                priority="P0" if len(failure.affected_runs) > 1 else "P1",
                recommendation=f"Fix {failure.failure_type}: {failure.example_issue[:50]}...",
                target_agent=check.fix_agent,
                expected_impact=f"Resolves {len(failure.affected_runs)} failed run(s)",
            ))

        # P1: Flagged issues with low scores
        for check_id, agg in aggregated.check_aggregations.items():
            if check_id.startswith("F") and agg["avg_score"] < 0.85:
                check = get_check_by_id(check_id)
                if check:
                    recommendations.append(RecommendedFix(
                        priority="P2",
                        recommendation=f"Improve {check.name} (avg score: {agg['avg_score']:.0%})",
                        target_agent=check.fix_agent,
                        expected_impact="Improves quality metrics",
                    ))

        # Sort by priority
        priority_order = {"P0": 0, "P1": 1, "P2": 2}
        recommendations.sort(key=lambda r: priority_order.get(r.priority, 3))

        return recommendations[:5]  # Limit to top 5

    def validate_and_report(
        self,
        runs_data: list[dict],
    ) -> ValidationReportData:
        """
        Validate multiple runs and generate report data.

        Args:
            runs_data: List of dicts, each containing:
                - run_id: str
                - run_number: int
                - adapted_json: dict
                - validation_report: dict
                - alignment_report: dict
                - fix_results: dict
                - factsheet: dict

        Returns:
            ValidationReportData ready for formatting
        """
        run_results = []

        for run_data in runs_data:
            result = self.check_runner.validate_single_run(
                run_id=run_data["run_id"],
                run_number=run_data["run_number"],
                adapted_json=run_data["adapted_json"],
                validation_report=run_data.get("validation_report", {}),
                alignment_report=run_data.get("alignment_report", {}),
                fix_results=run_data.get("fix_results", {}),
                factsheet=run_data.get("factsheet", {}),
            )
            run_results.append(result)

        return self.generate_report_data(run_results)

    def from_existing_results(
        self,
        validation_report: dict,
        alignment_report: dict,
        compliance_result: dict,
        factsheet: dict,
    ) -> ValidationReportData:
        """
        FAST PATH: Create report data from existing results without re-running checks.

        This avoids re-running validation checks when we already have results from
        the pipeline. Much faster than validate_and_report() which runs all checks again.

        Args:
            validation_report: Output from validation stage
            alignment_report: Output from alignment checker
            compliance_result: Output from finisher compliance check
            factsheet: Global factsheet

        Returns:
            ValidationReportData ready for formatting
        """
        from datetime import datetime

        # Extract scores from compliance result
        score_data = compliance_result.get("score", {})
        blocker_pass_rate = score_data.get("blocker_pass_rate", 1.0)
        overall_score = score_data.get("overall_score", 1.0)
        passed = score_data.get("passed", True)

        # Determine decision
        if passed:
            release_decision = "Approved"
            system_verdict = "System auto-approves this adaptation"
        elif blocker_pass_rate >= 0.8:
            release_decision = "Fix Required"
            system_verdict = "Minor fixes needed before release"
        else:
            release_decision = "Blocked"
            system_verdict = "Critical issues require human review"

        # Build aggregated results from existing reports
        critical_passed = 0
        critical_total = len(CRITICAL_CHECKS)
        flagged_passed = 0
        flagged_total = len(FLAGGED_CHECKS)

        # Count from validation report
        if validation_report:
            shard_results = validation_report.get("shard_results", {})
            for shard_id, results in shard_results.items():
                for r in results:
                    if r.get("passed", False):
                        critical_passed += 1

        # Count from alignment report
        if alignment_report:
            for rule in alignment_report.get("passed_rules", []):
                flagged_passed += 1
            for rule in alignment_report.get("failed_rules", []):
                flagged_total += 1

        aggregated = AggregatedResults(
            critical_passed=critical_passed,
            critical_total=max(critical_total, 1),
            flagged_passed=flagged_passed,
            flagged_total=max(flagged_total, 1),
            total_blocker_issues=compliance_result.get("failing_shards", []).__len__(),
            total_warning_issues=0,
            total_info_issues=0,
            runs_checked=1,
            check_results={},
        )

        # Build failure summaries from failing shards
        failure_summaries = []
        for shard_id in compliance_result.get("failing_shards", []):
            failure_summaries.append(FailureSummary(
                check_id="shard_validation",
                failure_type="Shard Validation Failed",
                affected_runs=["run-1"],
                example_issue=f"Shard {shard_id} failed validation",
                why_it_matters="Content may have quality issues",
                where_it_happens=shard_id,
                detection_stage="validation",
                fix_scope="shard",
                structural_risk="low",
            ))

        return ValidationReportData(
            original_scenario=factsheet.get("source_scenario", "Source"),
            target_scenario=factsheet.get("target_scenario", "Target"),
            simulation_purpose=factsheet.get("simulation_purpose", "Training"),
            system_mode="Fully automated",
            validation_version="1.0",
            total_runs=1,
            acceptance_threshold=self.acceptance_threshold,
            timestamp=datetime.now(),
            critical_pass_rate=aggregated.critical_pass_rate,
            meets_threshold=passed,
            has_blocking_issues=blocker_pass_rate < 0.8,
            release_decision=release_decision,
            system_verdict=system_verdict,
            aggregated_results=aggregated,
            failure_summaries=failure_summaries,
            can_ship_as_is=passed,
            next_action="Ship" if passed else "Review failures",
            one_line_summary=f"Compliance: {overall_score:.1%}, {release_decision}",
        )
