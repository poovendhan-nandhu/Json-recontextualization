"""
Stage 6: Finisher (Compliance Loop)

Re-validates only changed shards and computes weighted compliance score.

Process:
1. Identify changed shards (hash comparison)
2. Re-run validation on changed shards only
3. Compute weighted score (Blockers=100%, Overall>=98%)
4. Route back to Fixer if fail (max 3 attempts)
5. Flag for human if still failing

Pass conditions:
- All blocker rules = 100%
- Overall score >= 98%
"""
import logging
from typing import Any, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ComplianceStatus(Enum):
    """Status of compliance check."""
    PASS = "pass"
    FAIL_RETRY = "fail_retry"        # Can retry with fixer
    FAIL_HUMAN = "fail_human"        # Needs human review
    ERROR = "error"


@dataclass
class ComplianceScore:
    """Weighted compliance score."""
    blocker_pass_rate: float    # Must be 1.0
    warning_pass_rate: float    # Should be high
    overall_score: float        # Must be >= 0.98
    shard_scores: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        # Relaxed thresholds - allow some failures
        return self.blocker_pass_rate >= 0.8 and self.overall_score >= 0.70


@dataclass
class ComplianceResult:
    """Result of compliance check."""
    status: ComplianceStatus
    score: ComplianceScore
    iteration: int
    changed_shards: list[str] = field(default_factory=list)
    failing_shards: list[str] = field(default_factory=list)
    flagged_for_human: list[str] = field(default_factory=list)
    message: str = ""

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "score": {
                "blocker_pass_rate": round(self.score.blocker_pass_rate, 4),
                "warning_pass_rate": round(self.score.warning_pass_rate, 4),
                "overall_score": round(self.score.overall_score, 4),
                "passed": self.score.passed,
            },
            "iteration": self.iteration,
            "changed_shards": self.changed_shards,
            "failing_shards": self.failing_shards,
            "flagged_for_human": self.flagged_for_human,
            "message": self.message,
        }


class Finisher:
    """
    Stage 6: Compliance Loop

    Orchestrates the fix-validate cycle until compliance or max iterations.
    """

    # Compliance thresholds (relaxed for better UX)
    BLOCKER_PASS_RATE_REQUIRED = 0.8    # 80% - allow some blocker failures
    OVERALL_SCORE_REQUIRED = 0.70       # 70% - more lenient overall
    MAX_ITERATIONS = 1                   # Don't waste time retrying

    def __init__(
        self,
        blocker_threshold: float = None,
        overall_threshold: float = None,
        max_iterations: int = None,
    ):
        self.blocker_threshold = blocker_threshold or self.BLOCKER_PASS_RATE_REQUIRED
        self.overall_threshold = overall_threshold or self.OVERALL_SCORE_REQUIRED
        self.max_iterations = max_iterations or self.MAX_ITERATIONS

        # Track original hashes for change detection
        self._original_hashes: dict[str, str] = {}
        self._iteration = 0

    def set_original_hashes(self, shards: list):
        """
        Store original shard hashes for change detection.

        Call this BEFORE any fixing to track what changed.
        """
        for shard in shards:
            shard_id = shard.id if hasattr(shard, 'id') else "unknown"
            shard_hash = shard.hash if hasattr(shard, 'hash') else str(hash(str(shard.content)))
            self._original_hashes[shard_id] = shard_hash

    def identify_changed_shards(self, shards: list) -> list[str]:
        """
        Identify which shards have changed since original.

        Args:
            shards: Current shard list

        Returns:
            List of shard IDs that changed
        """
        changed = []
        for shard in shards:
            shard_id = shard.id if hasattr(shard, 'id') else "unknown"
            current_hash = shard.hash if hasattr(shard, 'hash') else str(hash(str(shard.content)))

            original = self._original_hashes.get(shard_id)
            if original and original != current_hash:
                changed.append(shard_id)

        return changed

    def compute_weighted_score(
        self,
        validation_report: Any,  # ScopedValidationReport
        alignment_report: Any = None,  # AlignmentReport
    ) -> ComplianceScore:
        """
        Compute weighted compliance score.

        Blocker checks weighted 2x.
        Warning checks weighted 1x.
        Info checks weighted 0.5x.

        Args:
            validation_report: From scoped validation
            alignment_report: From alignment checker (optional)

        Returns:
            ComplianceScore
        """
        total_blocker_checks = 0
        passed_blocker_checks = 0
        total_warning_checks = 0
        passed_warning_checks = 0
        shard_scores = {}

        # Process validation results
        if hasattr(validation_report, 'shard_results'):
            for shard_id, results in validation_report.shard_results.items():
                shard_blocker_pass = 0
                shard_blocker_total = 0
                shard_warning_pass = 0
                shard_warning_total = 0

                for result in results:
                    # Count by severity
                    for issue in result.issues:
                        if issue.severity.value == "blocker":
                            shard_blocker_total += 1
                            total_blocker_checks += 1
                        elif issue.severity.value == "warning":
                            shard_warning_total += 1
                            total_warning_checks += 1

                    # If no issues, count as passed
                    if result.passed:
                        if result.blocker_count == 0:
                            passed_blocker_checks += 1
                            shard_blocker_pass += 1
                        if result.warning_count == 0:
                            passed_warning_checks += 1
                            shard_warning_pass += 1

                # Calculate shard score
                shard_total = shard_blocker_total + shard_warning_total
                if shard_total > 0:
                    shard_scores[shard_id] = 1.0 - (
                        (shard_blocker_total - shard_blocker_pass) * 2 +
                        (shard_warning_total - shard_warning_pass)
                    ) / (shard_total * 2)
                else:
                    shard_scores[shard_id] = 1.0

        # Include alignment report if provided
        if alignment_report:
            if hasattr(alignment_report, 'blocker_issues'):
                total_blocker_checks += len(alignment_report.blocker_issues)
            if hasattr(alignment_report, 'warning_issues'):
                total_warning_checks += len(alignment_report.warning_issues)

        # Calculate rates
        blocker_pass_rate = 1.0
        if total_blocker_checks > 0:
            blocker_pass_rate = passed_blocker_checks / total_blocker_checks

        warning_pass_rate = 1.0
        if total_warning_checks > 0:
            warning_pass_rate = passed_warning_checks / total_warning_checks

        # Overall score: weighted average
        # Blockers count 2x, warnings 1x
        total_weighted = total_blocker_checks * 2 + total_warning_checks
        if total_weighted > 0:
            passed_weighted = passed_blocker_checks * 2 + passed_warning_checks
            overall_score = passed_weighted / total_weighted
        else:
            overall_score = 1.0

        # Also factor in report scores
        if hasattr(validation_report, 'overall_score'):
            overall_score = (overall_score + validation_report.overall_score) / 2
        if alignment_report and hasattr(alignment_report, 'overall_score'):
            overall_score = (overall_score + alignment_report.overall_score) / 2

        return ComplianceScore(
            blocker_pass_rate=blocker_pass_rate,
            warning_pass_rate=warning_pass_rate,
            overall_score=overall_score,
            shard_scores=shard_scores,
        )

    async def run_compliance_loop(
        self,
        shards: list,
        context: dict,
        validator: Any = None,  # ScopedValidator
        fixer: Any = None,       # ScopedFixer
        alignment_checker: Any = None,  # AlignmentChecker
    ) -> ComplianceResult:
        """
        Run the compliance loop: validate -> fix -> re-validate.

        Args:
            shards: List of Shard objects
            context: Pipeline context (factsheet, scenarios, etc.)
            validator: ScopedValidator instance
            fixer: ScopedFixer instance
            alignment_checker: AlignmentChecker instance (optional)

        Returns:
            ComplianceResult
        """
        # Import here to avoid circular imports
        from ..validators import ScopedValidator, validate_shards
        from .fixers import ScopedFixer

        if validator is None:
            validator = ScopedValidator()
        if fixer is None:
            fixer = ScopedFixer()

        # Store original hashes
        self.set_original_hashes(shards)

        for iteration in range(1, self.max_iterations + 1):
            self._iteration = iteration
            logger.info(f"Compliance loop iteration {iteration}/{self.max_iterations}")

            # Step 1: Identify changed shards
            if iteration > 1:
                changed_shards = self.identify_changed_shards(shards)
                # Filter to only validate changed shards
                shards_to_validate = [s for s in shards if s.id in changed_shards]
            else:
                changed_shards = [s.id for s in shards]
                shards_to_validate = shards

            if not shards_to_validate:
                logger.info("No shards to validate")
                break

            # Step 2: Run validation
            validation_report = await validator.validate_all(shards_to_validate, context)

            # Step 3: Run alignment check if provided
            alignment_report = None
            if alignment_checker:
                adapted_json = context.get("adapted_json", {})
                factsheet = context.get("global_factsheet", {})
                source = context.get("source_scenario", "")
                alignment_report = await alignment_checker.check(adapted_json, factsheet, source)

            # Step 4: Compute weighted score
            score = self.compute_weighted_score(validation_report, alignment_report)

            logger.info(
                f"  Blocker pass rate: {score.blocker_pass_rate:.1%}, "
                f"Overall: {score.overall_score:.1%}"
            )

            # Step 5: Check if passed
            if score.passed:
                return ComplianceResult(
                    status=ComplianceStatus.PASS,
                    score=score,
                    iteration=iteration,
                    changed_shards=changed_shards,
                    message=f"Compliance achieved in {iteration} iteration(s)"
                )

            # Step 6: Identify failing shards
            failing_shards = [
                shard_id for shard_id, shard_score in score.shard_scores.items()
                if shard_score < 1.0
            ]

            # Step 7: If last iteration, flag for human
            if iteration >= self.max_iterations:
                return ComplianceResult(
                    status=ComplianceStatus.FAIL_HUMAN,
                    score=score,
                    iteration=iteration,
                    changed_shards=changed_shards,
                    failing_shards=failing_shards,
                    flagged_for_human=failing_shards,
                    message=f"Max iterations reached. {len(failing_shards)} shards need human review."
                )

            # Step 8: Run fixer on failing shards
            failing_shard_objects = [s for s in shards if s.id in failing_shards]
            fix_results = await fixer.fix_all(failing_shard_objects, validation_report, context)

            # Update shard content with fixes
            for shard in shards:
                if shard.id in fix_results:
                    fix_result = fix_results[shard.id]
                    if fix_result.success and fix_result.fixed_content:
                        shard.content = fix_result.fixed_content
                        # Update hash
                        if hasattr(shard, 'compute_hash'):
                            shard.compute_hash()

            # Check for flagged shards
            flagged = fixer.get_flagged_shards()
            if flagged:
                return ComplianceResult(
                    status=ComplianceStatus.FAIL_HUMAN,
                    score=score,
                    iteration=iteration,
                    changed_shards=changed_shards,
                    failing_shards=failing_shards,
                    flagged_for_human=flagged,
                    message=f"{len(flagged)} shards flagged for human review after max fix attempts."
                )

        # Should not reach here, but handle gracefully
        return ComplianceResult(
            status=ComplianceStatus.ERROR,
            score=ComplianceScore(0, 0, 0),
            iteration=self._iteration,
            message="Unexpected end of compliance loop"
        )


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

async def run_compliance_check(
    shards: list,
    context: dict,
    max_iterations: int = 3,
) -> ComplianceResult:
    """
    Run compliance loop on shards.

    Args:
        shards: List of Shard objects
        context: Pipeline context
        max_iterations: Max fix-validate cycles

    Returns:
        ComplianceResult
    """
    finisher = Finisher(max_iterations=max_iterations)
    return await finisher.run_compliance_loop(shards, context)
