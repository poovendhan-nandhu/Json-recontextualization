"""
State schema for Simple Adaptation Pipeline.

SIMPLIFIED PIPELINE:
  ADAPT → VALIDATE → REPAIR (loop) → FINALIZE
"""
from typing import TypedDict, Optional, Literal, Any


class PipelineState(TypedDict, total=False):
    """
    State that flows through the pipeline.

    Simpler than the old 7-stage state - focused on:
    - Input/output JSON
    - Validation scores
    - Repair tracking
    """

    # ==========================================================================
    # INPUT
    # ==========================================================================
    input_json: dict                          # Original simulation JSON
    scenario_prompt: str                      # Target scenario description

    # ==========================================================================
    # ADAPTATION OUTPUT
    # ==========================================================================
    adapted_json: dict                        # Adapted JSON (updated by repair)
    entity_map: dict                          # Old entity -> New entity mapping
    domain_profile: dict                      # Domain info (industry, forbidden terms)
    adaptation_time_ms: int                   # Time for adaptation
    shards_processed: int                     # Number of shards adapted

    # ==========================================================================
    # VALIDATION OUTPUT
    # ==========================================================================
    validation_score: float                   # Overall score (0.0 - 1.0)
    validation_passed: bool                   # Did it pass threshold?
    validation_issues: list                   # List of issue dicts
    agent_scores: dict                        # {agent_name: score}
    agent_results: list                       # List of AgentResult objects

    # ==========================================================================
    # REPAIR TRACKING
    # ==========================================================================
    repair_iteration: int                     # Current repair iteration
    repair_history: list                      # List of {iteration, issues_count, previous_score}

    # ==========================================================================
    # FINAL OUTPUT
    # ==========================================================================
    final_json: dict                          # Final adapted JSON
    final_score: float                        # Final validation score
    status: str                               # "success" | "partial" | "failed"
    validation_report: str                    # Markdown validation report

    # ==========================================================================
    # EXECUTION METADATA
    # ==========================================================================
    stage_timings: dict                       # {stage: duration_ms}
    total_runtime_ms: int                     # Total pipeline time
    errors: list                              # List of error messages


def create_initial_state(
    input_json: dict,
    scenario_prompt: str
) -> PipelineState:
    """
    Create initial pipeline state.

    Args:
        input_json: Original simulation JSON
        scenario_prompt: Target scenario description

    Returns:
        Initial PipelineState
    """
    return PipelineState(
        # Input
        input_json=input_json,
        scenario_prompt=scenario_prompt,

        # Adaptation output
        adapted_json={},
        entity_map={},
        domain_profile={},
        adaptation_time_ms=0,
        shards_processed=0,

        # Validation output
        validation_score=0.0,
        validation_passed=False,
        validation_issues=[],
        agent_scores={},
        agent_results=[],

        # Repair tracking
        repair_iteration=0,
        repair_history=[],

        # Final output
        final_json={},
        final_score=0.0,
        status="pending",

        # Metadata
        stage_timings={},
        total_runtime_ms=0,
        errors=[]
    )
