"""LangGraph workflow exports."""
from .workflow import (
    # New API
    build_pipeline,
    run_pipeline,
    run_pipeline_streaming,
    PipelineResult,

    # Legacy aliases
    create_adaptation_workflow,
    create_workflow,
    scenario_workflow,
)
from .state import PipelineState, create_initial_state

__all__ = [
    # New API
    "build_pipeline",
    "run_pipeline",
    "run_pipeline_streaming",
    "PipelineResult",
    "PipelineState",
    "create_initial_state",

    # Legacy aliases
    "create_adaptation_workflow",
    "create_workflow",
    "scenario_workflow",
]