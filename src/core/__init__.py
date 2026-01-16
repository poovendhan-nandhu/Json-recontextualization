"""
Core processing modules for leaf-based JSON adaptation.

Modules:
- indexer: Extract leaf paths from JSON
- grouper: Group leaves by semantic context
- classifier: Classify leaves for processing strategy
- decider: LLM batch processing for rewrites (with smart prompts)
- context: AdaptationContext extraction
- smart_prompts: Smart prompt templates with validation rules
- leaf_validators: Validate leaf output (GPT 5.2)
- leaf_fixers: Fix validation issues (GPT 5.2)
- leaf_repair_loop: Validate -> Fix -> Re-validate cycle
- feedback_agent: Canonical validation report (GPT 5.2)
- leaf_adapter: Main orchestrator
"""

from .indexer import (
    index_leaves,
    filter_string_leaves,
    filter_modifiable_leaves,
    find_leaves_containing,
    get_leaf_stats,
)

from .grouper import (
    group_leaves_by_semantic_context,
    match_path_to_group,
    get_group_summary,
    redistribute_large_groups,
    SEMANTIC_GROUPS,
)

from .classifier import (
    LeafClassifier,
    LeafStrategy,
    ClassifiedLeaf,
    classify_leaves,
    get_leaves_by_strategy,
    apply_replacements,
)

from .context import (
    AdaptationContext,
    extract_adaptation_context,
)

from .smart_prompts import (
    build_smart_decision_prompt,
    build_reference_check_prompt,
    build_targeted_retry_prompt,
    check_poison_terms,
    check_klo_alignment,
)

from .decider import (
    LeafDecider,
    DecisionResult,
    decide_leaf_changes,
    get_changes_only,
    get_decision_stats,
    # Force replace logic
    is_force_replace_path,
    classify_leaf,
    log_force_replace_summary,
    FORCE_REPLACE_PATTERNS,
    # HTML utilities
    chunk_html_content,
    check_html_sanity,
    repair_html_issues,
)

from .leaf_adapter import (
    LeafAdapter,
    AdaptationResult,
    adapt_json_with_leaves,
    get_adaptation_diff,
)

from .leaf_validators import (
    LeafValidator,
    LeafValidationResult,
    ValidationIssue,
    ValidationSeverity,
    validate_leaf_decisions,
    get_blocker_issues,
)

from .leaf_fixers import (
    LeafFixer,
    LeafFixerResult,
    FixResult,
    fix_leaf_issues,
    apply_fixes_to_decisions,
)

from .leaf_repair_loop import (
    LeafRepairLoop,
    RepairLoopResult,
    RepairIteration,
    PathRepairHistory,
    SmartRetryFixer,
    AggressiveFixer,
    run_repair_loop,
    validate_and_fix,
    smart_fix_single,
)

from .feedback_agent import (
    FeedbackAgent,
    FeedbackReport,
    generate_feedback_report,
)

from .leaf_rag import (
    LeafRAG,
    LeafRAGResult,
    LeafExample,
    LEAF_COLLECTIONS,
    index_leaves_for_rag,
    retrieve_leaf_examples,
    get_rag_context_for_adaptation,
    get_rag_context_for_adaptation_parallel,
)

from .leaf_graph import (
    LeafPipelineState,
    create_leaf_workflow,
    run_leaf_pipeline,
    run_leaf_pipeline_streaming,
    leaf_workflow,
    # Individual nodes
    context_node,
    indexer_node,
    rag_node,
    decider_node,
    validation_node,
    repair_node,
    patcher_node,
    feedback_node,
)

__all__ = [
    # Indexer
    "index_leaves",
    "filter_string_leaves",
    "filter_modifiable_leaves",
    "find_leaves_containing",
    "get_leaf_stats",
    # Grouper
    "group_leaves_by_semantic_context",
    "match_path_to_group",
    "get_group_summary",
    "redistribute_large_groups",
    "SEMANTIC_GROUPS",
    # Classifier
    "LeafClassifier",
    "LeafStrategy",
    "ClassifiedLeaf",
    "classify_leaves",
    "get_leaves_by_strategy",
    "apply_replacements",
    # Context
    "AdaptationContext",
    "extract_adaptation_context",
    # Smart Prompts
    "build_smart_decision_prompt",
    "build_reference_check_prompt",
    "build_targeted_retry_prompt",
    "check_poison_terms",
    "check_klo_alignment",
    # Decider
    "LeafDecider",
    "DecisionResult",
    "decide_leaf_changes",
    "get_changes_only",
    "get_decision_stats",
    "is_force_replace_path",
    "classify_leaf",
    "log_force_replace_summary",
    "FORCE_REPLACE_PATTERNS",
    "chunk_html_content",
    "check_html_sanity",
    "repair_html_issues",
    # Leaf Adapter
    "LeafAdapter",
    "AdaptationResult",
    "adapt_json_with_leaves",
    "get_adaptation_diff",
    # Leaf Validators
    "LeafValidator",
    "LeafValidationResult",
    "ValidationIssue",
    "ValidationSeverity",
    "validate_leaf_decisions",
    "get_blocker_issues",
    # Leaf Fixers
    "LeafFixer",
    "LeafFixerResult",
    "FixResult",
    "fix_leaf_issues",
    "apply_fixes_to_decisions",
    # Leaf Repair Loop
    "LeafRepairLoop",
    "RepairLoopResult",
    "RepairIteration",
    "PathRepairHistory",
    "SmartRetryFixer",
    "AggressiveFixer",
    "run_repair_loop",
    "validate_and_fix",
    "smart_fix_single",
    # Feedback Agent
    "FeedbackAgent",
    "FeedbackReport",
    "generate_feedback_report",
    # Leaf RAG
    "LeafRAG",
    "LeafRAGResult",
    "LeafExample",
    "LEAF_COLLECTIONS",
    "index_leaves_for_rag",
    "retrieve_leaf_examples",
    "get_rag_context_for_adaptation",
    "get_rag_context_for_adaptation_parallel",
    # Leaf LangGraph
    "LeafPipelineState",
    "create_leaf_workflow",
    "run_leaf_pipeline",
    "run_leaf_pipeline_streaming",
    "leaf_workflow",
    "context_node",
    "indexer_node",
    "rag_node",
    "decider_node",
    "validation_node",
    "repair_node",
    "patcher_node",
    "feedback_node",
]
