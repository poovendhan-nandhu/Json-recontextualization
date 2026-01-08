"""Shard dataclass for chunked JSON processing."""
from dataclasses import dataclass, field
from typing import Literal, Optional, Any
from enum import Enum


class LockState(str, Enum):
    """Lock states for shards."""
    UNLOCKED = "UNLOCKED"
    STRUCTURE_LOCKED = "STRUCTURE_LOCKED"  # Content can change, structure cannot
    FULLY_LOCKED = "FULLY_LOCKED"  # Cannot modify at all


class ShardStatus(str, Enum):
    """Processing status for shards."""
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    VALIDATED = "VALIDATED"
    NEEDS_STRUCTURAL_FIX = "NEEDS_STRUCTURAL_FIX"
    NEEDS_SEMANTIC_FIX = "NEEDS_SEMANTIC_FIX"
    FIXED = "FIXED"
    FLAGGED_FOR_HUMAN = "FLAGGED_FOR_HUMAN"
    APPROVED = "APPROVED"


@dataclass
class ValidationResult:
    """Result from a single validator."""
    rule_name: str
    passed: bool
    shard_id: str
    details: str = ""
    severity: Literal["blocker", "warning", "info"] = "warning"
    suggested_fix: Optional[str] = None


@dataclass
class Shard:
    """
    Represents a chunk of the simulation JSON.

    Each shard is an independent unit that can be:
    - Processed in parallel
    - Validated independently
    - Fixed without affecting other shards
    """

    # Identity
    id: str                                    # "metadata", "assessment", "scenario", etc.
    name: str                                  # Human-readable name

    # Content
    paths: list[str]                           # JSONPaths covered by this shard
    content: dict[str, Any]                    # Extracted content from those paths

    # Hashing for change detection
    original_hash: str = ""                    # Hash at extraction time
    current_hash: str = ""                     # Hash after modifications

    # Lock state
    lock_state: LockState = LockState.UNLOCKED

    # Processing state
    status: ShardStatus = ShardStatus.PENDING
    is_blocker: bool = False                   # Must pass 100% validation?

    # Validation results
    validation_results: list[ValidationResult] = field(default_factory=list)

    # Fix tracking
    fix_attempts: int = 0
    max_fix_attempts: int = 3

    # Alignment info (which other shards this relates to)
    aligns_with: list[str] = field(default_factory=list)  # Other shard IDs

    # IDs extracted from this shard (for preservation)
    extracted_ids: list[str] = field(default_factory=list)

    def has_changed(self) -> bool:
        """Check if shard content has changed since extraction."""
        return self.current_hash != self.original_hash

    def can_modify(self, modification_type: Literal["structure", "semantic"]) -> bool:
        """Check if this shard can be modified."""
        if self.lock_state == LockState.FULLY_LOCKED:
            return False
        if modification_type == "structure" and self.lock_state == LockState.STRUCTURE_LOCKED:
            return False
        return True

    def can_retry_fix(self) -> bool:
        """Check if we can attempt another fix."""
        return self.fix_attempts < self.max_fix_attempts

    def get_blocker_failures(self) -> list[ValidationResult]:
        """Get all blocker-level validation failures."""
        return [r for r in self.validation_results if not r.passed and r.severity == "blocker"]

    def get_all_failures(self) -> list[ValidationResult]:
        """Get all validation failures."""
        return [r for r in self.validation_results if not r.passed]

    def pass_rate(self) -> float:
        """Calculate validation pass rate."""
        if not self.validation_results:
            return 0.0
        passed = sum(1 for r in self.validation_results if r.passed)
        return passed / len(self.validation_results)

    def clear_validation_results(self) -> None:
        """Clear previous validation results before re-validation."""
        self.validation_results = []


@dataclass
class ShardCollection:
    """
    Collection of all shards from a simulation JSON.
    Manages the full set and provides utilities.
    """

    shards: list[Shard] = field(default_factory=list)
    source_json_hash: str = ""                 # Hash of original full JSON
    scenario_prompt: str = ""                  # Target scenario prompt

    def get_shard(self, shard_id: str) -> Optional[Shard]:
        """Get a shard by ID."""
        for shard in self.shards:
            if shard.id == shard_id:
                return shard
        return None

    def get_unlocked_shards(self) -> list[Shard]:
        """Get all shards that can be modified."""
        return [s for s in self.shards if s.lock_state != LockState.FULLY_LOCKED]

    def get_blocker_shards(self) -> list[Shard]:
        """Get all shards marked as blockers."""
        return [s for s in self.shards if s.is_blocker]

    def get_changed_shards(self) -> list[Shard]:
        """Get all shards that have been modified."""
        return [s for s in self.shards if s.has_changed()]

    def get_shards_needing_fix(self) -> list[Shard]:
        """Get shards that need fixing."""
        return [s for s in self.shards
                if s.status in (ShardStatus.NEEDS_STRUCTURAL_FIX, ShardStatus.NEEDS_SEMANTIC_FIX)]

    def all_ids(self) -> list[str]:
        """Get all extracted IDs from all shards."""
        ids = []
        for shard in self.shards:
            ids.extend(shard.extracted_ids)
        return ids

    def overall_pass_rate(self) -> float:
        """Calculate overall validation pass rate across all shards."""
        total_results = []
        for shard in self.shards:
            total_results.extend(shard.validation_results)
        if not total_results:
            return 0.0
        passed = sum(1 for r in total_results if r.passed)
        return passed / len(total_results)

    def blocker_pass_rate(self) -> float:
        """Calculate pass rate for blocker-level validations only."""
        blocker_results = []
        for shard in self.shards:
            blocker_results.extend([r for r in shard.validation_results if r.severity == "blocker"])
        if not blocker_results:
            return 1.0  # No blockers = 100% pass
        passed = sum(1 for r in blocker_results if r.passed)
        return passed / len(blocker_results)

    def update_shard(self, updated_shard: Shard) -> None:
        """Update a shard in the collection by its ID."""
        for i, shard in enumerate(self.shards):
            if shard.id == updated_shard.id:
                self.shards[i] = updated_shard
                return
        # If not found, add it
        self.shards.append(updated_shard)


@dataclass
class ComplianceScore:
    """Weighted compliance score from Finisher stage."""
    blocker_pass_rate: float      # Must be 1.0 to pass
    overall_score: float          # Must be >= 0.98 to pass
    shard_scores: dict[str, float] = field(default_factory=dict)

    def is_passing(self) -> bool:
        """Check if scores meet thresholds."""
        return self.blocker_pass_rate == 1.0 and self.overall_score >= 0.98
