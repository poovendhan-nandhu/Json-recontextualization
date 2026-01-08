"""
Base Validator for Scoped Validation.

All validators inherit from BaseValidator and implement validate().
Validators check WITHIN a single shard (not cross-shard).
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Literal
from enum import Enum


class ValidationSeverity(Enum):
    """Severity level for validation issues."""
    BLOCKER = "blocker"    # Must be fixed, blocks pipeline
    WARNING = "warning"    # Should be fixed, doesn't block
    INFO = "info"          # FYI, no action needed


@dataclass
class ValidationIssue:
    """A single validation issue found."""
    rule_id: str
    message: str
    location: str  # JSONPath or description
    severity: ValidationSeverity
    current_value: Any = None
    expected_value: Any = None
    suggestion: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "message": self.message,
            "location": self.location,
            "severity": self.severity.value,
            "current_value": str(self.current_value)[:200] if self.current_value else None,
            "expected_value": str(self.expected_value)[:200] if self.expected_value else None,
            "suggestion": self.suggestion,
        }


@dataclass
class ValidationResult:
    """Result of running a validator on a shard."""
    validator_name: str
    shard_id: str
    passed: bool
    score: float  # 0.0 to 1.0
    issues: list[ValidationIssue] = field(default_factory=list)
    details: dict = field(default_factory=dict)

    @property
    def blocker_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == ValidationSeverity.BLOCKER)

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == ValidationSeverity.WARNING)

    def to_dict(self) -> dict:
        return {
            "validator": self.validator_name,
            "shard_id": self.shard_id,
            "passed": self.passed,
            "score": round(self.score, 4),
            "blocker_count": self.blocker_count,
            "warning_count": self.warning_count,
            "issues": [i.to_dict() for i in self.issues],
            "details": self.details,
        }


class BaseValidator(ABC):
    """
    Abstract base class for shard validators.

    Each validator checks ONE aspect of a shard:
    - Structure integrity
    - ID preservation
    - Domain fidelity
    - Entity removal
    - Content completeness
    - Tone/format
    """

    # Validator identification
    name: str = "BaseValidator"
    description: str = "Base validator"

    # Is this a blocker? (failure = pipeline stops)
    is_blocker: bool = False

    # Which shard types this validator applies to (empty = all)
    applicable_shards: list[str] = []

    def __init__(self):
        pass

    @abstractmethod
    async def validate(
        self,
        shard: Any,  # Shard object
        context: dict,  # Additional context (factsheet, industry, etc.)
    ) -> ValidationResult:
        """
        Validate a shard.

        Args:
            shard: Shard object to validate
            context: Additional context including:
                - global_factsheet: Factsheet from adaptation
                - source_scenario: Original scenario text
                - target_scenario: Target scenario text
                - industry: Detected industry

        Returns:
            ValidationResult with pass/fail, score, and issues
        """
        pass

    def applies_to(self, shard_id: str) -> bool:
        """Check if this validator applies to a shard type."""
        if not self.applicable_shards:
            return True  # Applies to all if not specified
        return shard_id in self.applicable_shards

    def _create_result(
        self,
        shard_id: str,
        passed: bool,
        score: float,
        issues: list[ValidationIssue] = None,
        details: dict = None,
    ) -> ValidationResult:
        """Helper to create ValidationResult."""
        return ValidationResult(
            validator_name=self.name,
            shard_id=shard_id,
            passed=passed,
            score=score,
            issues=issues or [],
            details=details or {},
        )

    def _create_issue(
        self,
        message: str,
        location: str,
        severity: ValidationSeverity = None,
        current_value: Any = None,
        expected_value: Any = None,
        suggestion: str = None,
    ) -> ValidationIssue:
        """Helper to create ValidationIssue."""
        return ValidationIssue(
            rule_id=self.name.lower().replace(" ", "_"),
            message=message,
            location=location,
            severity=severity or (ValidationSeverity.BLOCKER if self.is_blocker else ValidationSeverity.WARNING),
            current_value=current_value,
            expected_value=expected_value,
            suggestion=suggestion,
        )
