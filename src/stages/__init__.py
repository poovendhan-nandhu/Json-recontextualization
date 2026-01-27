"""
Cartedo Simulation Adaptation Pipeline - Simple Architecture

Stages:
1. Sharder - Split JSON into independent shards
2. Simple Adapter - Parallel shard transformation with Gemini
3. Simple Validators - Validate adapted output
4. Finisher - Compliance checking

Note: Old complex modules moved to backup/old_stages/
"""

from .sharder import Sharder, shard_json, merge_shards
from .simple_adapter import (
    adapt_simple,
    SimpleAdaptationResult,
)
from .simple_validators import (
    run_all_validators,
    validate_and_repair,
    ValidationReport,
)
from .finisher import (
    Finisher,
    ComplianceScore,
    ComplianceResult,
    ComplianceStatus,
    run_compliance_check,
)

__all__ = [
    # Sharder
    "Sharder",
    "shard_json",
    "merge_shards",
    # Simple Adapter
    "adapt_simple",
    "SimpleAdaptationResult",
    # Simple Validators
    "run_all_validators",
    "validate_and_repair",
    "ValidationReport",
    # Finisher
    "Finisher",
    "ComplianceScore",
    "ComplianceResult",
    "ComplianceStatus",
    "run_compliance_check",
]
