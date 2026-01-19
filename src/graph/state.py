"""
State schema for 7-Stage LangGraph workflow.

Follows Shweta's architecture:
1. Sharder -> 2. Adaptation -> 3. Alignment -> 4. Validation ->
4B. Fixers -> 5. Merger -> 6. Finisher -> 7. Human Approval
"""
from typing import TypedDict, Optional, Literal, Any
from dataclasses import dataclass, field


class PipelineState(TypedDict, total=False):
    """
    Shared state across all 7 stages of the pipeline.

    Designed for shard-based processing with parallel validation.
    """

    # ==========================================================================
    # INPUT DATA
    # ==========================================================================
    input_json: dict                          # Original simulation JSON
    selected_scenario: str | int              # Target scenario index or text
    target_scenario_text: str                 # Full target scenario text

    # ==========================================================================
    # STAGE 1: SHARDER OUTPUT
    # ==========================================================================
    shard_collection: Any                     # ShardCollection object
    shards: list                              # List of Shard objects
    shard_ids: list[str]                      # IDs of all shards
    locked_shard_ids: list[str]               # IDs of locked shards (never modify)
    shard_hashes: dict[str, str]              # Original hashes for change detection

    # ==========================================================================
    # STAGE 2: ADAPTATION ENGINE OUTPUT
    # ==========================================================================
    adapted_shards: list                      # Adapted Shard objects
    entity_map: dict[str, str]                # Old entity -> New entity mapping
    industry: str                             # Detected target industry
    global_factsheet: dict                    # Factsheet with poison_list, hints
    rag_context: dict                         # Industry context from RAG

    # ==========================================================================
    # STAGE 3: ALIGNMENT CHECKER OUTPUT
    # ==========================================================================
    alignment_report: dict                    # AlignmentReport.to_dict()
    alignment_score: float                    # Overall alignment score
    alignment_passed: bool                    # Did alignment pass threshold?

    # ==========================================================================
    # STAGE 3B: ALIGNMENT FIXER OUTPUT
    # ==========================================================================
    alignment_retry_count: int                # Retry counter for alignment fixer
    alignment_fixes_applied: int              # Count of fixes applied
    alignment_fixer_skipped: bool             # Was alignment fixer skipped?
    alignment_fix_results: list               # Results from alignment fixer
    previous_alignment_score: float           # Score before alignment fixer ran
    alignment_feedback: dict                  # Feedback from alignment analysis

    # ==========================================================================
    # STAGE 4: SCOPED VALIDATION OUTPUT
    # ==========================================================================
    validation_report: dict                   # ScopedValidationReport.to_dict()
    validation_score: float                   # Overall validation score
    validation_passed: bool                   # Did validation pass threshold?
    blocker_count: int                        # Number of blocker issues
    warning_count: int                        # Number of warning issues

    # ==========================================================================
    # STAGE 4B: FIXERS OUTPUT
    # ==========================================================================
    fix_results: dict[str, dict]              # shard_id -> FixResult.to_dict()
    patches_applied: list[dict]               # All patches for rollback
    fixed_shards: list                        # Shards after fixing

    # ==========================================================================
    # STAGE 5: MERGER OUTPUT
    # ==========================================================================
    merged_json: dict                         # Reassembled full JSON
    merge_successful: bool                    # Did merge succeed?

    # ==========================================================================
    # STAGE 6: FINISHER OUTPUT
    # ==========================================================================
    compliance_result: dict                   # ComplianceResult.to_dict()
    compliance_score: float                   # Overall compliance score
    compliance_passed: bool                   # Did compliance pass?
    compliance_iteration: int                 # Which iteration of compliance loop
    flagged_for_human: list[str]              # Shard IDs needing human review
    human_readable_report: str                # Markdown validation report for PM/QA

    # ==========================================================================
    # STAGE 7: HUMAN APPROVAL
    # ==========================================================================
    approval_package: dict                    # ApprovalPackage.to_dict()
    approval_status: str                      # "pending", "approved", "rejected"
    reviewer: Optional[str]                   # Who reviewed
    feedback: Optional[str]                   # Review feedback

    # ==========================================================================
    # FINAL OUTPUT
    # ==========================================================================
    output_json: dict                         # Final adapted simulation JSON
    final_status: Literal["OK", "FAIL", "PENDING", "HUMAN_REVIEW"]

    # ==========================================================================
    # EXECUTION METADATA
    # ==========================================================================
    current_stage: str                        # Current stage name
    stage_timings: dict[str, int]             # Stage -> duration_ms
    total_runtime_ms: int                     # Total pipeline runtime
    retry_count: int                          # Compliance loop retries
    max_retries: int                          # Max allowed retries (default 3)

    # ==========================================================================
    # ERROR TRACKING
    # ==========================================================================
    errors: list[dict]                        # List of errors encountered
    warnings: list[str]                       # Non-blocking warnings

    # ==========================================================================
    # LLM STATS (for observability)
    # ==========================================================================
    llm_calls: list[dict]                     # Track all LLM calls
    total_tokens: int                         # Total tokens used
    total_cost: float                         # Estimated cost


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_initial_state(
    input_json: dict,
    selected_scenario: str | int,
    max_retries: int = 3,
) -> PipelineState:
    """
    Create initial pipeline state.

    Args:
        input_json: Original simulation JSON
        selected_scenario: Target scenario index or text
        max_retries: Max compliance loop retries

    Returns:
        Initial PipelineState
    """
    return PipelineState(
        # Input
        input_json=input_json,
        selected_scenario=selected_scenario,
        target_scenario_text="",

        # Stage outputs (initialized empty)
        shard_collection=None,
        shards=[],
        shard_ids=[],
        locked_shard_ids=[],
        shard_hashes={},
        adapted_shards=[],
        entity_map={},
        industry="unknown",
        global_factsheet={},
        rag_context={},
        alignment_report={},
        alignment_score=0.0,
        alignment_passed=False,
        # Alignment fixer state
        alignment_retry_count=0,
        alignment_fixes_applied=0,
        alignment_fixer_skipped=False,
        alignment_fix_results=[],
        previous_alignment_score=0.0,
        alignment_feedback={},
        validation_report={},
        validation_score=0.0,
        validation_passed=False,
        blocker_count=0,
        warning_count=0,
        fix_results={},
        patches_applied=[],
        fixed_shards=[],
        merged_json={},
        merge_successful=False,
        compliance_result={},
        compliance_score=0.0,
        compliance_passed=False,
        compliance_iteration=0,
        flagged_for_human=[],
        human_readable_report="",
        approval_package={},
        approval_status="pending",
        reviewer=None,
        feedback=None,
        output_json={},
        final_status="PENDING",

        # Metadata
        current_stage="init",
        stage_timings={},
        total_runtime_ms=0,
        retry_count=0,
        max_retries=max_retries,
        errors=[],
        warnings=[],
        llm_calls=[],
        total_tokens=0,
        total_cost=0.0,
    )


def add_stage_timing(state: PipelineState, stage: str, duration_ms: int) -> None:
    """Add timing for a stage."""
    state["stage_timings"][stage] = duration_ms
    state["total_runtime_ms"] = sum(state["stage_timings"].values())


def add_error(state: PipelineState, stage: str, error: str, is_fatal: bool = False) -> None:
    """Add an error to state."""
    state["errors"].append({
        "stage": stage,
        "error": error,
        "is_fatal": is_fatal,
    })
    if is_fatal:
        state["final_status"] = "FAIL"


def add_llm_call(
    state: PipelineState,
    stage: str,
    model: str,
    tokens: int,
    duration_ms: int,
) -> None:
    """Track an LLM call."""
    state["llm_calls"].append({
        "stage": stage,
        "model": model,
        "tokens": tokens,
        "duration_ms": duration_ms,
    })
    state["total_tokens"] += tokens
