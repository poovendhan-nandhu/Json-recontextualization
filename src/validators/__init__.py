"""
Stage 4: Scoped Validators Module

Validates EACH SHARD independently (not full JSON).

Validators:
- StructureIntegrityValidator: All required fields present
- IDPreservationValidator: All IDs unchanged
- DomainFidelityValidator: KPIs valid for industry
- EntityRemovalValidator: Old names removed
- ContentCompletenessValidator: No empty/placeholder fields
- ToneValidator: Professional language
"""

from .base import (
    BaseValidator,
    ValidationResult,
    ValidationIssue,
    ValidationSeverity,
)
from .scoped_validators import (
    ScopedValidator,
    ScopedValidationReport,
    StructureIntegrityValidator,
    IDPreservationValidator,
    DomainFidelityValidator,
    EntityRemovalValidator,
    ContentCompletenessValidator,
    ToneValidator,
    validate_shards,
)

__all__ = [
    # Base
    "BaseValidator",
    "ValidationResult",
    "ValidationIssue",
    "ValidationSeverity",
    # Orchestrator
    "ScopedValidator",
    "ScopedValidationReport",
    "validate_shards",
    # Individual Validators
    "StructureIntegrityValidator",
    "IDPreservationValidator",
    "DomainFidelityValidator",
    "EntityRemovalValidator",
    "ContentCompletenessValidator",
    "ToneValidator",
]
