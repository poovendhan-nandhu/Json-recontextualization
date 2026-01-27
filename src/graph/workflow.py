"""
LangGraph Workflow - Re-exports from nodes.py

Simplified 4-stage pipeline:
  ADAPT → VALIDATE → REPAIR (loop) → FINALIZE
"""
from .nodes import (
    # Workflow creation
    build_pipeline,
    run_pipeline,
    run_pipeline_streaming,

    # Individual nodes
    node_adapt,
    node_validate,
    node_repair,
    node_finalize,

    # Routing functions
    should_repair,

    # Result class
    PipelineResult,
)

# Legacy aliases for backwards compatibility
create_adaptation_workflow = build_pipeline
create_workflow = build_pipeline
scenario_workflow = build_pipeline()  # Pre-compiled instance

# Node aliases (old names -> new names)
sharder_node = node_adapt  # Combined into adapt
adaptation_node = node_adapt
alignment_node = node_validate  # Combined into validate
validation_node = node_validate
fixers_node = node_repair
merger_node = node_finalize  # Combined into finalize
finisher_node = node_finalize
human_approval_node = node_finalize

# Routing aliases
should_fix = should_repair
should_retry_compliance = should_repair
should_abort = lambda state: "finalize"  # No abort in simplified pipeline

__all__ = [
    # New API
    "build_pipeline",
    "run_pipeline",
    "run_pipeline_streaming",
    "node_adapt",
    "node_validate",
    "node_repair",
    "node_finalize",
    "should_repair",
    "PipelineResult",

    # Legacy aliases
    "create_adaptation_workflow",
    "create_workflow",
    "scenario_workflow",
    "sharder_node",
    "adaptation_node",
    "alignment_node",
    "validation_node",
    "fixers_node",
    "merger_node",
    "finisher_node",
    "human_approval_node",
    "should_fix",
    "should_retry_compliance",
    "should_abort",
]
