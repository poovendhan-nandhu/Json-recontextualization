"""
LangGraph Workflow - Re-exports from nodes.py

All workflow logic is in nodes.py for simplicity.
This file provides backwards compatibility.
"""
from .nodes import (
    # Workflow creation
    create_adaptation_workflow,
    run_pipeline,
    run_pipeline_streaming,

    # Individual nodes (all have @traceable)
    sharder_node,
    adaptation_node,
    alignment_node,
    validation_node,
    fixers_node,
    merger_node,
    finisher_node,
    human_approval_node,

    # Routing functions
    should_fix,
    should_retry_compliance,
    should_abort,
)

# Legacy aliases
create_workflow = create_adaptation_workflow
scenario_workflow = create_adaptation_workflow()  # Pre-compiled instance

__all__ = [
    "create_adaptation_workflow",
    "create_workflow",
    "scenario_workflow",
    "run_pipeline",
    "run_pipeline_streaming",
    "sharder_node",
    "adaptation_node",
    "alignment_node",
    "validation_node",
    "fixers_node",
    "merger_node",
    "finisher_node",
    "human_approval_node",
]
