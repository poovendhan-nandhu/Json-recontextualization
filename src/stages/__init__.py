"""
Cartedo Simulation Adaptation Pipeline

Stages:
1. Sharder - Split JSON into independent shards
2. Adaptation Engine - Parallel shard transformation with Gemini
3. Alignment Checker - LLM-based cross-shard consistency validation
4. Scoped Validation - Per-shard validation (in src/validators/)
4B. Scoped Fixers - Fix failing shards (Structural + Semantic)
5. Merger - Reassemble shards (via sharder.merge_shards)
6. Finisher - Compliance loop (re-validate, weighted scoring)
7. Human Approval - Final review (approve/reject API)

Each stage is independent and can be tested/run separately.
"""

from .sharder import Sharder, shard_json, merge_shards
from .adaptation_engine import (
    AdaptationEngine,
    adapt_simulation,
    adapt_simulation_with_leaves,  # NEW: Leaf-based adaptation
    AdaptationResult,
)
from .alignment_checker import AlignmentChecker, check_alignment, AlignmentReport
from .alignment_fixer import AlignmentFixer, AlignmentFixResult, fix_alignment_issues
from .fixers import (
    ScopedFixer,
    StructuralFixer,
    SemanticFixer,
    FixResult,
    FixType,
    fix_shard,
    fix_all_shards,
)
from .finisher import (
    Finisher,
    ComplianceScore,
    ComplianceResult,
    ComplianceStatus,
    run_compliance_check,
)
from .human_approval import (
    HumanApproval,
    ApprovalPackage,
    ApprovalStatus,
    FlaggedItem,
    ShardSummary,
    get_human_approval,
    create_approval,
    approve_simulation,
    reject_simulation,
)

__all__ = [
    # Stage 1: Sharder
    "Sharder",
    "shard_json",
    "merge_shards",
    # Stage 2: Adaptation
    "AdaptationEngine",
    "adapt_simulation",
    "adapt_simulation_with_leaves",  # NEW: Leaf-based adaptation
    "AdaptationResult",
    # Stage 3: Alignment Checker
    "AlignmentChecker",
    "check_alignment",
    "AlignmentReport",
    # Stage 3B: Alignment Fixer (NEW!)
    "AlignmentFixer",
    "AlignmentFixResult",
    "fix_alignment_issues",
    # Stage 4B: Fixers
    "ScopedFixer",
    "StructuralFixer",
    "SemanticFixer",
    "FixResult",
    "FixType",
    "fix_shard",
    "fix_all_shards",
    # Stage 6: Finisher
    "Finisher",
    "ComplianceScore",
    "ComplianceResult",
    "ComplianceStatus",
    "run_compliance_check",
    # Stage 7: Human Approval
    "HumanApproval",
    "ApprovalPackage",
    "ApprovalStatus",
    "FlaggedItem",
    "ShardSummary",
    "get_human_approval",
    "create_approval",
    "approve_simulation",
    "reject_simulation",
]
