"""
Validation Check Runner

Collects validation data from pipeline outputs and runs all checks.
Produces structured results for the report generator.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional
from pathlib import Path

from .check_definitions import (
    CRITICAL_CHECKS,
    FLAGGED_CHECKS,
    CheckDefinition,
    CheckTier,
    CheckStatus,
)

logger = logging.getLogger(__name__)


@dataclass
class CheckResult:
    """Result of a single check for a single run."""
    check_id: str
    check_name: str
    passed: bool
    score: float                     # 0.0 to 1.0
    status: CheckStatus
    issues_found: list[str] = field(default_factory=list)
    locations: list[str] = field(default_factory=list)
    action_needed: str = ""


@dataclass
class RunValidationResult:
    """Complete validation result for a single run."""
    run_id: str
    run_number: int

    # Overall status
    passed_all_critical: bool
    critical_pass_count: int
    critical_total: int
    flagged_pass_count: int
    flagged_total: int

    # Scores
    compliance_score: float

    # Individual check results
    check_results: dict[str, CheckResult] = field(default_factory=dict)

    # Failure details
    critical_failures: list[str] = field(default_factory=list)
    flagged_warnings: list[str] = field(default_factory=list)


@dataclass
class AggregatedResults:
    """Aggregated validation results across all runs."""
    total_runs: int
    runs_passing_critical: int
    critical_pass_rate: float
    meets_acceptance: bool
    acceptance_threshold: float

    # Per-check aggregation
    check_aggregations: dict[str, dict] = field(default_factory=dict)

    # Individual run results
    run_results: list[RunValidationResult] = field(default_factory=list)

    # Failure summary
    failed_runs: list[str] = field(default_factory=list)
    common_failure_patterns: list[str] = field(default_factory=list)


class ValidationCheckRunner:
    """
    Runs validation checks and collects results.

    Can operate in two modes:
    1. Live mode: Runs checks against actual pipeline output
    2. Report mode: Aggregates pre-collected check results
    """

    def __init__(self, acceptance_threshold: float = 0.95):
        self.acceptance_threshold = acceptance_threshold
        self.critical_checks = CRITICAL_CHECKS
        self.flagged_checks = FLAGGED_CHECKS

    def validate_single_run(
        self,
        run_id: str,
        run_number: int,
        adapted_json: dict,
        validation_report: dict,
        alignment_report: dict,
        fix_results: dict,
        factsheet: dict,
    ) -> RunValidationResult:
        """
        Validate a single run against all checks.

        Args:
            run_id: Unique run identifier
            run_number: Run sequence number
            adapted_json: The adapted simulation JSON
            validation_report: Output from validation stage
            alignment_report: Output from alignment checker
            fix_results: Output from fixers
            factsheet: Global factsheet used for adaptation

        Returns:
            RunValidationResult with all check outcomes
        """
        check_results = {}

        # Run all critical checks
        for check in self.critical_checks:
            result = self._run_check(
                check=check,
                adapted_json=adapted_json,
                validation_report=validation_report,
                alignment_report=alignment_report,
                fix_results=fix_results,
                factsheet=factsheet,
            )
            check_results[check.id] = result

        # Run all flagged checks
        for check in self.flagged_checks:
            result = self._run_check(
                check=check,
                adapted_json=adapted_json,
                validation_report=validation_report,
                alignment_report=alignment_report,
                fix_results=fix_results,
                factsheet=factsheet,
            )
            check_results[check.id] = result

        # Calculate aggregates
        critical_results = [r for cid, r in check_results.items() if cid.startswith("C")]
        flagged_results = [r for cid, r in check_results.items() if cid.startswith("F")]

        critical_passed = sum(1 for r in critical_results if r.passed)
        flagged_passed = sum(1 for r in flagged_results if r.passed)

        passed_all_critical = critical_passed == len(critical_results)

        # Get failure lists
        critical_failures = [r.check_id for r in critical_results if not r.passed]
        flagged_warnings = [r.check_id for r in flagged_results if not r.passed]

        # Calculate compliance score (weighted average)
        all_scores = [r.score for r in check_results.values()]
        compliance_score = sum(all_scores) / len(all_scores) if all_scores else 0.0

        return RunValidationResult(
            run_id=run_id,
            run_number=run_number,
            passed_all_critical=passed_all_critical,
            critical_pass_count=critical_passed,
            critical_total=len(critical_results),
            flagged_pass_count=flagged_passed,
            flagged_total=len(flagged_results),
            compliance_score=compliance_score,
            check_results=check_results,
            critical_failures=critical_failures,
            flagged_warnings=flagged_warnings,
        )

    def _run_check(
        self,
        check: CheckDefinition,
        adapted_json: dict,
        validation_report: dict,
        alignment_report: dict,
        fix_results: dict,
        factsheet: dict,
    ) -> CheckResult:
        """Run a single check and return result."""

        # Route to specific check implementation
        check_method = getattr(self, f"_check_{check.id.lower()}", None)

        if check_method:
            return check_method(
                check=check,
                adapted_json=adapted_json,
                validation_report=validation_report,
                alignment_report=alignment_report,
                fix_results=fix_results,
                factsheet=factsheet,
            )
        else:
            # Default: extract from validation/alignment reports
            return self._check_from_reports(
                check=check,
                validation_report=validation_report,
                alignment_report=alignment_report,
            )

    def _check_from_reports(
        self,
        check: CheckDefinition,
        validation_report: dict,
        alignment_report: dict,
    ) -> CheckResult:
        """Extract check result from existing reports."""

        # Map check IDs to report fields
        check_mappings = {
            "C1": ("entity_removal", "entityremoval"),
            "C2": ("domain_fidelity", "domainfidelity", "kpi_alignment"),
            "C3": ("structure_integrity", "structureintegrity", "schema"),
            "C4": ("rubric_integrity", "id_preservation"),
            "C5": ("content_completeness", "executability"),
            "C6": ("barrier_compliance", "structural"),
            "C7": ("klo_to_questions", "klo_alignment"),
            "C8": ("resource_completeness", "resources"),
            "F1": ("persona_realism", "tone"),
            "F2": ("resource_authenticity", "resources"),
            "F3": ("scenario_coherence", "narrative"),
            "F4": ("tone", "tone_consistency"),
            "F5": ("data_realism", "realism"),
            "F6": ("domain_fidelity", "terminology"),
        }

        # Get mapping keys for this check
        mapping_keys = check_mappings.get(check.id, [])

        # Search validation report
        score = 1.0
        issues = []
        passed = True

        # Check validation_report shard_results
        if "shard_results" in validation_report:
            for shard_id, results in validation_report.get("shard_results", {}).items():
                for result in results:
                    rule_id = result.get("rule_id", "").lower()
                    if any(key in rule_id for key in mapping_keys):
                        result_score = result.get("score", 1.0)
                        if result_score < score:
                            score = result_score
                        if result.get("issues"):
                            issues.extend([
                                i.get("message", str(i)) if isinstance(i, dict) else str(i)
                                for i in result.get("issues", [])[:3]
                            ])

        # Check alignment_report results
        if "results" in alignment_report:
            for result in alignment_report.get("results", []):
                rule_id = result.get("rule_id", "").lower()
                if any(key in rule_id for key in mapping_keys):
                    result_score = result.get("score", 1.0)
                    if result_score < score:
                        score = result_score
                    if result.get("issues"):
                        issues.extend([
                            i.get("description", str(i)) if isinstance(i, dict) else str(i)
                            for i in result.get("issues", [])[:3]
                        ])

        # Determine pass/fail based on threshold
        passed = score >= check.threshold_value
        status = CheckStatus.PASS if passed else CheckStatus.FAIL

        # Determine action needed
        action_needed = "None" if passed else f"Review and fix via {check.fix_agent}"

        return CheckResult(
            check_id=check.id,
            check_name=check.name,
            passed=passed,
            score=score,
            status=status,
            issues_found=issues[:5],  # Limit to 5 issues
            action_needed=action_needed,
        )

    # =========================================================================
    # SPECIFIC CHECK IMPLEMENTATIONS
    # =========================================================================

    def _check_c1(
        self,
        check: CheckDefinition,
        adapted_json: dict,
        validation_report: dict,
        alignment_report: dict,
        fix_results: dict,
        factsheet: dict,
    ) -> CheckResult:
        """C1: Entity Removal - No original scenario references remain."""

        poison_list = factsheet.get("poison_list", [])
        issues = []
        locations = []

        # Convert JSON to string for searching
        json_str = json.dumps(adapted_json, default=str).lower()

        # Check for poison terms
        for term in poison_list:
            if term.lower() in json_str:
                issues.append(f"Found stale reference: '{term}'")
                # Find approximate location
                locations.append("Multiple locations")

        passed = len(issues) == 0
        score = 1.0 if passed else max(0.0, 1.0 - (len(issues) * 0.1))

        return CheckResult(
            check_id=check.id,
            check_name=check.name,
            passed=passed,
            score=score,
            status=CheckStatus.PASS if passed else CheckStatus.FAIL,
            issues_found=issues[:5],
            locations=locations[:5],
            action_needed="None" if passed else "Run Semantic Fixer to remove stale references",
        )

    def _check_c3(
        self,
        check: CheckDefinition,
        adapted_json: dict,
        validation_report: dict,
        alignment_report: dict,
        fix_results: dict,
        factsheet: dict,
    ) -> CheckResult:
        """C3: Schema Validity - Output JSON conforms to schema."""

        issues = []

        # Check required top-level keys
        required_keys = ["topicWizardData"]
        for key in required_keys:
            if key not in adapted_json:
                issues.append(f"Missing required key: {key}")

        topic_data = adapted_json.get("topicWizardData", {})

        # Check required topic_data keys
        required_topic_keys = [
            "simulationName",
            "simulationFlow",
            "assessmentCriterion",
        ]
        for key in required_topic_keys:
            if key not in topic_data:
                issues.append(f"Missing required key: topicWizardData.{key}")

        # Check simulationFlow structure
        sim_flow = topic_data.get("simulationFlow", [])
        if not isinstance(sim_flow, list):
            issues.append("simulationFlow must be an array")
        elif len(sim_flow) == 0:
            issues.append("simulationFlow is empty")

        passed = len(issues) == 0
        score = 1.0 if passed else max(0.0, 1.0 - (len(issues) * 0.2))

        return CheckResult(
            check_id=check.id,
            check_name=check.name,
            passed=passed,
            score=score,
            status=CheckStatus.PASS if passed else CheckStatus.FAIL,
            issues_found=issues[:5],
            action_needed="None" if passed else "Run Structural Fixer to repair schema",
        )

    def _check_c7(
        self,
        check: CheckDefinition,
        adapted_json: dict,
        validation_report: dict,
        alignment_report: dict,
        fix_results: dict,
        factsheet: dict,
    ) -> CheckResult:
        """C7: KLO Preservation - Key Learning Outcomes preserved."""

        # Extract from alignment report
        klo_score = 1.0
        issues = []

        for result in alignment_report.get("results", []):
            rule_id = result.get("rule_id", "")
            if "klo" in rule_id.lower():
                result_score = result.get("score", 1.0)
                if result_score < klo_score:
                    klo_score = result_score
                for issue in result.get("issues", []):
                    if isinstance(issue, dict):
                        issues.append(issue.get("description", str(issue)))
                    else:
                        issues.append(str(issue))

        passed = klo_score >= check.threshold_value

        return CheckResult(
            check_id=check.id,
            check_name=check.name,
            passed=passed,
            score=klo_score,
            status=CheckStatus.PASS if passed else CheckStatus.FAIL,
            issues_found=issues[:5],
            action_needed="None" if passed else "Run Alignment Fixer to restore KLO mapping",
        )

    # =========================================================================
    # AGGREGATION
    # =========================================================================

    def aggregate_results(
        self,
        run_results: list[RunValidationResult],
    ) -> AggregatedResults:
        """
        Aggregate validation results across multiple runs.

        Args:
            run_results: List of individual run results

        Returns:
            AggregatedResults with pass rates and patterns
        """
        total_runs = len(run_results)

        if total_runs == 0:
            return AggregatedResults(
                total_runs=0,
                runs_passing_critical=0,
                critical_pass_rate=0.0,
                meets_acceptance=False,
                acceptance_threshold=self.acceptance_threshold,
            )

        # Count runs passing all critical checks
        runs_passing_critical = sum(
            1 for r in run_results if r.passed_all_critical
        )
        critical_pass_rate = runs_passing_critical / total_runs
        meets_acceptance = critical_pass_rate >= self.acceptance_threshold

        # Aggregate per-check results
        check_aggregations = {}
        all_check_ids = [c.id for c in self.critical_checks + self.flagged_checks]

        for check_id in all_check_ids:
            passes = sum(
                1 for r in run_results
                if check_id in r.check_results and r.check_results[check_id].passed
            )
            total = sum(
                1 for r in run_results if check_id in r.check_results
            )
            avg_score = sum(
                r.check_results[check_id].score
                for r in run_results if check_id in r.check_results
            ) / max(total, 1)

            check_aggregations[check_id] = {
                "passes": passes,
                "total": total,
                "pass_rate": passes / total if total > 0 else 0.0,
                "avg_score": avg_score,
            }

        # Get failed runs
        failed_runs = [
            r.run_id for r in run_results if not r.passed_all_critical
        ]

        # Find common failure patterns
        failure_counts = {}
        for r in run_results:
            for failure in r.critical_failures:
                failure_counts[failure] = failure_counts.get(failure, 0) + 1

        common_failure_patterns = [
            f"{check_id}: {count} runs"
            for check_id, count in sorted(
                failure_counts.items(), key=lambda x: x[1], reverse=True
            )
        ]

        return AggregatedResults(
            total_runs=total_runs,
            runs_passing_critical=runs_passing_critical,
            critical_pass_rate=critical_pass_rate,
            meets_acceptance=meets_acceptance,
            acceptance_threshold=self.acceptance_threshold,
            check_aggregations=check_aggregations,
            run_results=run_results,
            failed_runs=failed_runs,
            common_failure_patterns=common_failure_patterns,
        )
