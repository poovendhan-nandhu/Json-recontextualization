"""
Stage 7: Human Approval

Final review stage where human confirms REALISM, not correctness.

Human DOES:
- Confirm realism ("Does this feel like a real hotel simulation?")
- Confirm learning quality ("Would students learn the right skills?")
- Approve or flag feedback

Human does NOT:
- Fix formatting
- Fix missing data
- Fix broken logic

API endpoints for approve/reject workflow.
"""
import logging
from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid

logger = logging.getLogger(__name__)


class ApprovalStatus(Enum):
    """Status of human approval."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class FlaggedItem:
    """An item flagged for human attention."""
    shard_id: str
    issue_type: str
    description: str
    severity: str
    suggestion: Optional[str] = None


@dataclass
class ShardSummary:
    """Summary of a shard for human review."""
    shard_id: str
    shard_name: str
    score: float
    blocker_count: int
    warning_count: int
    key_changes: list[str] = field(default_factory=list)


@dataclass
class ApprovalPackage:
    """
    Package sent to human for review.

    Contains everything human needs to make approve/reject decision.
    """
    approval_id: str
    simulation_id: str
    created_at: datetime
    status: ApprovalStatus

    # Summary info
    summary: str                              # 2-3 sentence summary
    compliance_score: float                   # Overall score
    target_scenario: str                      # What it was adapted to

    # Detailed reports
    shard_summaries: list[ShardSummary] = field(default_factory=list)
    flagged_items: list[FlaggedItem] = field(default_factory=list)

    # Review guidance
    review_questions: list[str] = field(default_factory=list)

    # URLs
    approve_url: Optional[str] = None
    reject_url: Optional[str] = None
    preview_url: Optional[str] = None

    # Result (filled after human decision)
    reviewer: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    feedback: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "approval_id": self.approval_id,
            "simulation_id": self.simulation_id,
            "created_at": self.created_at.isoformat(),
            "status": self.status.value,
            "summary": self.summary,
            "compliance_score": round(self.compliance_score, 4),
            "target_scenario": self.target_scenario,
            "shard_summaries": [
                {
                    "shard_id": s.shard_id,
                    "shard_name": s.shard_name,
                    "score": round(s.score, 4),
                    "blocker_count": s.blocker_count,
                    "warning_count": s.warning_count,
                    "key_changes": s.key_changes[:5],
                }
                for s in self.shard_summaries
            ],
            "flagged_items": [
                {
                    "shard_id": f.shard_id,
                    "issue_type": f.issue_type,
                    "description": f.description,
                    "severity": f.severity,
                }
                for f in self.flagged_items
            ],
            "review_questions": self.review_questions,
            "approve_url": self.approve_url,
            "reject_url": self.reject_url,
            "preview_url": self.preview_url,
            "reviewer": self.reviewer,
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
            "feedback": self.feedback,
        }


class HumanApproval:
    """
    Stage 7: Human Approval System

    Creates approval packages and handles approve/reject workflow.
    """

    # Default review questions for human
    DEFAULT_REVIEW_QUESTIONS = [
        "Does this simulation feel realistic for the target industry?",
        "Would students learn the intended skills from this simulation?",
        "Is the language and tone appropriate for educational use?",
        "Are there any obvious errors or inconsistencies?",
    ]

    def __init__(self, base_url: str = ""):
        self.base_url = base_url.rstrip("/")
        self._pending_approvals: dict[str, ApprovalPackage] = {}

    def create_approval_package(
        self,
        simulation_id: str,
        compliance_result: any,  # ComplianceResult from finisher
        validation_report: any = None,  # ScopedValidationReport
        context: dict = None,
    ) -> ApprovalPackage:
        """
        Create an approval package for human review.

        Args:
            simulation_id: Unique simulation ID
            compliance_result: Result from compliance loop
            validation_report: Detailed validation report
            context: Pipeline context (factsheet, scenarios)

        Returns:
            ApprovalPackage ready for human review
        """
        context = context or {}
        approval_id = str(uuid.uuid4())[:8]

        # Build summary
        score = compliance_result.score if hasattr(compliance_result, 'score') else None
        compliance_score = score.overall_score if score else 0.0

        target_scenario_raw = context.get("target_scenario", "Unknown scenario")
        # Ensure it's a string before slicing
        if isinstance(target_scenario_raw, dict):
            target_scenario = str(target_scenario_raw.get("option", target_scenario_raw))[:200]
        else:
            target_scenario = str(target_scenario_raw)[:200] if target_scenario_raw else "Unknown scenario"

        summary = self._generate_summary(compliance_result, context)

        # Build shard summaries
        shard_summaries = []
        if validation_report and hasattr(validation_report, 'shard_results'):
            for shard_id, results in validation_report.shard_results.items():
                total_blockers = sum(r.blocker_count for r in results)
                total_warnings = sum(r.warning_count for r in results)
                avg_score = sum(r.score for r in results) / len(results) if results else 1.0

                shard_summaries.append(ShardSummary(
                    shard_id=shard_id,
                    shard_name=shard_id.replace("_", " ").title(),
                    score=avg_score,
                    blocker_count=total_blockers,
                    warning_count=total_warnings,
                ))

        # Build flagged items
        flagged_items = []
        if hasattr(compliance_result, 'flagged_for_human'):
            for shard_id in compliance_result.flagged_for_human:
                flagged_items.append(FlaggedItem(
                    shard_id=shard_id,
                    issue_type="max_fix_attempts",
                    description=f"Shard '{shard_id}' could not be auto-fixed after max attempts",
                    severity="warning",
                    suggestion="Review manually and provide feedback"
                ))

        # Create package
        package = ApprovalPackage(
            approval_id=approval_id,
            simulation_id=simulation_id,
            created_at=datetime.utcnow(),
            status=ApprovalStatus.PENDING,
            summary=summary,
            compliance_score=compliance_score,
            target_scenario=target_scenario,
            shard_summaries=shard_summaries,
            flagged_items=flagged_items,
            review_questions=self.DEFAULT_REVIEW_QUESTIONS,
            approve_url=f"{self.base_url}/api/v1/approval/{approval_id}/approve" if self.base_url else None,
            reject_url=f"{self.base_url}/api/v1/approval/{approval_id}/reject" if self.base_url else None,
            preview_url=f"{self.base_url}/api/v1/simulation/{simulation_id}/preview" if self.base_url else None,
        )

        # Store for later retrieval
        self._pending_approvals[approval_id] = package

        logger.info(f"Created approval package {approval_id} for simulation {simulation_id}")

        return package

    def _generate_summary(self, compliance_result: any, context: dict) -> str:
        """Generate human-readable summary."""
        parts = []

        # Score summary
        if hasattr(compliance_result, 'score'):
            score = compliance_result.score
            parts.append(f"Compliance score: {score.overall_score:.1%}")

            if score.passed:
                parts.append("All automated checks passed.")
            else:
                if score.blocker_pass_rate < 1.0:
                    parts.append(f"Blocker pass rate: {score.blocker_pass_rate:.1%}")

        # Iteration summary
        if hasattr(compliance_result, 'iteration'):
            parts.append(f"Completed in {compliance_result.iteration} iteration(s).")

        # Flagged items
        if hasattr(compliance_result, 'flagged_for_human') and compliance_result.flagged_for_human:
            parts.append(f"{len(compliance_result.flagged_for_human)} item(s) flagged for review.")

        # Target scenario
        target = context.get("target_scenario", "")
        if target:
            # Extract key info from scenario
            company = context.get("global_factsheet", {}).get("company", {}).get("name", "")
            industry = context.get("industry", "")
            if company or industry:
                parts.append(f"Adapted for: {company or industry}")

        return " ".join(parts) or "Simulation ready for review."

    def get_approval(self, approval_id: str) -> Optional[ApprovalPackage]:
        """Get an approval package by ID."""
        return self._pending_approvals.get(approval_id)

    def approve(
        self,
        approval_id: str,
        reviewer: str,
        feedback: str = None,
    ) -> ApprovalPackage:
        """
        Approve a simulation.

        Args:
            approval_id: Approval package ID
            reviewer: Name/ID of reviewer
            feedback: Optional feedback

        Returns:
            Updated ApprovalPackage
        """
        package = self._pending_approvals.get(approval_id)
        if not package:
            raise ValueError(f"Approval {approval_id} not found")

        if package.status != ApprovalStatus.PENDING:
            raise ValueError(f"Approval {approval_id} is not pending (status: {package.status.value})")

        package.status = ApprovalStatus.APPROVED
        package.reviewer = reviewer
        package.reviewed_at = datetime.utcnow()
        package.feedback = feedback

        logger.info(f"Simulation {package.simulation_id} approved by {reviewer}")

        return package

    def reject(
        self,
        approval_id: str,
        reviewer: str,
        feedback: str,
    ) -> ApprovalPackage:
        """
        Reject a simulation.

        Args:
            approval_id: Approval package ID
            reviewer: Name/ID of reviewer
            feedback: Required feedback explaining rejection

        Returns:
            Updated ApprovalPackage
        """
        if not feedback:
            raise ValueError("Feedback is required for rejection")

        package = self._pending_approvals.get(approval_id)
        if not package:
            raise ValueError(f"Approval {approval_id} not found")

        if package.status != ApprovalStatus.PENDING:
            raise ValueError(f"Approval {approval_id} is not pending (status: {package.status.value})")

        package.status = ApprovalStatus.REJECTED
        package.reviewer = reviewer
        package.reviewed_at = datetime.utcnow()
        package.feedback = feedback

        logger.info(f"Simulation {package.simulation_id} rejected by {reviewer}: {feedback[:100]}")

        return package

    def list_pending(self) -> list[ApprovalPackage]:
        """List all pending approvals."""
        return [
            pkg for pkg in self._pending_approvals.values()
            if pkg.status == ApprovalStatus.PENDING
        ]

    def get_stats(self) -> dict:
        """Get approval statistics."""
        all_packages = list(self._pending_approvals.values())

        return {
            "total": len(all_packages),
            "pending": sum(1 for p in all_packages if p.status == ApprovalStatus.PENDING),
            "approved": sum(1 for p in all_packages if p.status == ApprovalStatus.APPROVED),
            "rejected": sum(1 for p in all_packages if p.status == ApprovalStatus.REJECTED),
        }


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_human_approval: Optional[HumanApproval] = None


def get_human_approval(base_url: str = "") -> HumanApproval:
    """Get singleton HumanApproval instance."""
    global _human_approval
    if _human_approval is None:
        _human_approval = HumanApproval(base_url)
    return _human_approval


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_approval(
    simulation_id: str,
    compliance_result: any,
    validation_report: any = None,
    context: dict = None,
) -> ApprovalPackage:
    """Create approval package. See HumanApproval.create_approval_package."""
    return get_human_approval().create_approval_package(
        simulation_id, compliance_result, validation_report, context
    )


def approve_simulation(approval_id: str, reviewer: str, feedback: str = None) -> ApprovalPackage:
    """Approve simulation. See HumanApproval.approve."""
    return get_human_approval().approve(approval_id, reviewer, feedback)


def reject_simulation(approval_id: str, reviewer: str, feedback: str) -> ApprovalPackage:
    """Reject simulation. See HumanApproval.reject."""
    return get_human_approval().reject(approval_id, reviewer, feedback)
