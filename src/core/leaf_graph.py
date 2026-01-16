"""
LangGraph workflow for Leaf-Based Adaptation.

Converts the linear leaf pipeline to a proper LangGraph StateGraph.

Stages:
1. Context Extraction - Extract adaptation context (Gemini)
2. Indexer - Index all leaves in JSON
3. RAG - Index & retrieve similar examples (optional)
4. Decider - LLM decisions with smart prompts (Gemini)
5. Validation - Run 5 validators (GPT 5.2)
6. Repair Loop - Fix blockers with escalating strategies (GPT 5.2)
7. Patcher - Apply patches to JSON
8. Feedback - Generate canonical report (GPT 5.2)
"""

import time
import logging
import copy
from typing import TypedDict, Optional, Any, List
from dataclasses import dataclass

from langgraph.graph import StateGraph, END
from langsmith import traceable

logger = logging.getLogger(__name__)


# =============================================================================
# STATE SCHEMA
# =============================================================================

class LeafPipelineState(TypedDict, total=False):
    """Shared state across all leaf pipeline stages."""

    # INPUT
    input_json: dict
    scenario_prompt: str

    # STAGE 1: CONTEXT
    context: Any  # AdaptationContext object
    poison_terms: List[str]
    klo_terms: dict

    # STAGE 2: INDEXER
    all_leaves: List[tuple]
    total_leaves: int
    leaf_stats: dict

    # STAGE 3: RAG
    rag_context: dict
    rag_indexed: int
    use_rag: bool

    # STAGE 4: DECIDER
    decisions: List[Any]  # List of DecisionResult
    pre_filtered: int
    llm_evaluated: int
    changes_proposed: int

    # STAGE 5: VALIDATION
    validation_result: Any  # LeafValidationResult
    blockers: int
    warnings: int
    validation_passed: bool

    # STAGE 6: REPAIR LOOP
    repair_result: Any  # RepairLoopResult
    repair_iterations: int
    fixes_succeeded: int
    repair_passed: bool

    # STAGE 7: PATCHER
    adapted_json: dict
    patches_applied: int

    # STAGE 8: FEEDBACK
    feedback_report: Any  # FeedbackReport
    release_decision: str

    # METADATA
    current_stage: str
    stage_timings: dict
    total_runtime_ms: int
    errors: List[dict]

    # FINAL
    passed: bool
    final_status: str  # "OK", "FAIL", "FIX_REQUIRED"


def create_initial_state(
    input_json: dict,
    scenario_prompt: str,
    use_rag: bool = True,
) -> LeafPipelineState:
    """Create initial leaf pipeline state."""
    return LeafPipelineState(
        input_json=input_json,
        scenario_prompt=scenario_prompt,
        use_rag=use_rag,
        context=None,
        poison_terms=[],
        klo_terms={},
        all_leaves=[],
        total_leaves=0,
        leaf_stats={},
        rag_context={},
        rag_indexed=0,
        decisions=[],
        pre_filtered=0,
        llm_evaluated=0,
        changes_proposed=0,
        validation_result=None,
        blockers=0,
        warnings=0,
        validation_passed=False,
        repair_result=None,
        repair_iterations=0,
        fixes_succeeded=0,
        repair_passed=False,
        adapted_json={},
        patches_applied=0,
        feedback_report=None,
        release_decision="PENDING",
        current_stage="init",
        stage_timings={},
        total_runtime_ms=0,
        errors=[],
        passed=False,
        final_status="PENDING",
    )


def add_stage_timing(state: LeafPipelineState, stage: str, duration_ms: int):
    """Add timing for a stage."""
    state["stage_timings"][stage] = duration_ms
    state["total_runtime_ms"] = sum(state["stage_timings"].values())


def add_error(state: LeafPipelineState, stage: str, error: str, is_fatal: bool = False):
    """Add an error to state."""
    state["errors"].append({
        "stage": stage,
        "error": error,
        "is_fatal": is_fatal,
    })
    if is_fatal:
        state["final_status"] = "FAIL"


# =============================================================================
# STAGE 1: CONTEXT EXTRACTION NODE
# =============================================================================

@traceable(name="leaf_stage1_context")
async def context_node(state: LeafPipelineState) -> LeafPipelineState:
    """
    Stage 1: Extract adaptation context using Gemini.

    Extracts:
    - Company mappings (old → new)
    - KLO terms
    - Poison terms to remove
    - Industry context
    """
    start_time = time.time()
    state["current_stage"] = "context"

    try:
        from .context import extract_adaptation_context

        context = await extract_adaptation_context(
            input_json=state["input_json"],
            target_scenario=state["scenario_prompt"],
        )

        state["context"] = context
        state["poison_terms"] = context.poison_terms
        state["klo_terms"] = context.klo_terms

        duration_ms = int((time.time() - start_time) * 1000)
        add_stage_timing(state, "context", duration_ms)

        logger.info(f"Stage 1 complete: {len(context.poison_terms)} poison terms, "
                   f"{len(context.klo_terms)} KLO terms")

    except Exception as e:
        logger.error(f"Context extraction failed: {e}")
        add_error(state, "context", str(e), is_fatal=True)

    return state


# =============================================================================
# STAGE 2: INDEXER NODE
# =============================================================================

@traceable(name="leaf_stage2_indexer")
async def indexer_node(state: LeafPipelineState) -> LeafPipelineState:
    """
    Stage 2: Index all leaves in the JSON.

    Extracts all leaf paths and values for processing.
    """
    start_time = time.time()
    state["current_stage"] = "indexer"

    try:
        from .indexer import index_leaves, get_leaf_stats

        all_leaves = index_leaves(state["input_json"])
        leaf_stats = get_leaf_stats(all_leaves)

        state["all_leaves"] = all_leaves
        state["total_leaves"] = len(all_leaves)
        state["leaf_stats"] = leaf_stats

        duration_ms = int((time.time() - start_time) * 1000)
        add_stage_timing(state, "indexer", duration_ms)

        logger.info(f"Stage 2 complete: {len(all_leaves)} leaves indexed")

    except Exception as e:
        logger.error(f"Indexer failed: {e}")
        add_error(state, "indexer", str(e), is_fatal=True)

    return state


# =============================================================================
# STAGE 3: RAG NODE
# =============================================================================

@traceable(name="leaf_stage3_rag")
async def rag_node(state: LeafPipelineState) -> LeafPipelineState:
    """
    Stage 3: RAG - Index leaves and retrieve similar examples.

    Optional stage - skipped if use_rag=False.
    """
    start_time = time.time()
    state["current_stage"] = "rag"

    if not state.get("use_rag", True):
        logger.info("Stage 3 skipped: RAG disabled")
        add_stage_timing(state, "rag", 0)
        return state

    try:
        from .leaf_rag import LeafRAG, get_rag_context_for_adaptation_parallel

        rag = LeafRAG()

        if rag.available:
            # Index current leaves (PARALLEL - much faster!)
            rag_result = await rag.index_leaves_parallel(
                leaves=state["all_leaves"],
                simulation_id="current",
                industry="auto",
                clear_existing=True,
            )
            state["rag_indexed"] = rag_result.leaves_indexed

            # Retrieve similar examples (PARALLEL - much faster!)
            rag_context = await get_rag_context_for_adaptation_parallel(
                target_scenario=state["scenario_prompt"],
                groups=["questions", "resources", "rubrics", "scenarios", "klos"],
                n_per_group=2,
            )
            state["rag_context"] = rag_context

            logger.info(f"Stage 3 complete: {rag_result.leaves_indexed} indexed, "
                       f"{len(rag_context)} groups retrieved")
        else:
            logger.info("Stage 3 skipped: RAG not available")
            state["rag_context"] = {}

        duration_ms = int((time.time() - start_time) * 1000)
        add_stage_timing(state, "rag", duration_ms)

    except Exception as e:
        logger.warning(f"RAG failed (non-fatal): {e}")
        state["rag_context"] = {}
        add_stage_timing(state, "rag", int((time.time() - start_time) * 1000))

    return state


# =============================================================================
# STAGE 4: DECIDER NODE
# =============================================================================

@traceable(name="leaf_stage4_decider")
async def decider_node(state: LeafPipelineState) -> LeafPipelineState:
    """
    Stage 4: LLM decisions with smart prompts using Gemini.

    For each leaf, decides: keep or replace.
    """
    start_time = time.time()
    state["current_stage"] = "decider"

    try:
        from .decider import LeafDecider, get_decision_stats, get_changes_only

        decider = LeafDecider(
            context=state["context"],
            rag_context=state.get("rag_context", {}),
        )

        decisions = await decider.decide_all(state["all_leaves"])
        stats = get_decision_stats(decisions)

        state["decisions"] = decisions
        state["pre_filtered"] = stats["pre_filtered"]
        state["llm_evaluated"] = stats["llm_decided"]
        state["changes_proposed"] = stats["replace"]

        duration_ms = int((time.time() - start_time) * 1000)
        add_stage_timing(state, "decider", duration_ms)

        logger.info(f"Stage 4 complete: {stats['replace']} changes proposed, "
                   f"{stats['keep']} kept, {stats['pre_filtered']} pre-filtered")

    except Exception as e:
        logger.error(f"Decider failed: {e}")
        add_error(state, "decider", str(e), is_fatal=True)

    return state


# =============================================================================
# STAGE 5: VALIDATION NODE
# =============================================================================

@traceable(name="leaf_stage5_validation")
async def validation_node(state: LeafPipelineState) -> LeafPipelineState:
    """
    Stage 5: Run 5 validators on decisions using GPT 5.2.

    Validators:
    1. Entity Removal
    2. Domain Fidelity
    3. KLO Alignment
    4. Data Consistency
    5. Structure
    """
    start_time = time.time()
    state["current_stage"] = "validation"

    try:
        from .leaf_validators import validate_leaf_decisions

        validation_result = await validate_leaf_decisions(
            state["decisions"],
            state["context"],
        )

        state["validation_result"] = validation_result
        state["blockers"] = validation_result.blockers
        state["warnings"] = validation_result.warnings
        state["validation_passed"] = validation_result.passed

        duration_ms = int((time.time() - start_time) * 1000)
        add_stage_timing(state, "validation", duration_ms)

        logger.info(f"Stage 5 complete: {validation_result.blockers} blockers, "
                   f"{validation_result.warnings} warnings, passed={validation_result.passed}")

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        add_error(state, "validation", str(e), is_fatal=False)
        state["validation_passed"] = False

    return state


# =============================================================================
# STAGE 6: REPAIR LOOP NODE
# =============================================================================

@traceable(name="leaf_stage6_repair")
async def repair_node(state: LeafPipelineState) -> LeafPipelineState:
    """
    Stage 6: Run repair loop with escalating strategies using GPT 5.2.

    Strategies:
    1. Standard fix (semantic)
    2. Smart retry (targeted prompt)
    3. Aggressive fix (direct replacement)
    """
    start_time = time.time()
    state["current_stage"] = "repair"

    # Skip if no blockers
    if state.get("validation_passed", False) or state.get("blockers", 0) == 0:
        logger.info("Stage 6 skipped: no blockers to fix")
        state["repair_passed"] = True
        add_stage_timing(state, "repair", 0)
        return state

    try:
        from .leaf_repair_loop import run_repair_loop

        repair_result = await run_repair_loop(
            decisions=state["decisions"],
            context=state["context"],
            max_iterations=3,
        )

        state["repair_result"] = repair_result
        state["decisions"] = repair_result.decisions  # Updated decisions
        state["repair_iterations"] = repair_result.total_iterations
        state["fixes_succeeded"] = repair_result.total_fixes_succeeded
        state["repair_passed"] = repair_result.passed

        duration_ms = int((time.time() - start_time) * 1000)
        add_stage_timing(state, "repair", duration_ms)

        logger.info(f"Stage 6 complete: {repair_result.initial_blockers} → "
                   f"{repair_result.final_blockers} blockers, "
                   f"{repair_result.total_fixes_succeeded} fixes succeeded")

    except Exception as e:
        logger.error(f"Repair loop failed: {e}")
        add_error(state, "repair", str(e), is_fatal=False)
        state["repair_passed"] = False

    return state


# =============================================================================
# STAGE 7: PATCHER NODE
# =============================================================================

@traceable(name="leaf_stage7_patcher")
async def patcher_node(state: LeafPipelineState) -> LeafPipelineState:
    """
    Stage 7: Apply patches to JSON based on decisions.

    Uses JSON Pointer for surgical updates.
    """
    start_time = time.time()
    state["current_stage"] = "patcher"

    try:
        from .decider import get_changes_only
        from ..utils.patcher import PatchOp, get_patcher

        # Make a deep copy
        working_json = copy.deepcopy(state["input_json"])

        # Get changes
        changes = get_changes_only(state["decisions"])

        if not changes:
            state["adapted_json"] = working_json
            state["patches_applied"] = 0
            add_stage_timing(state, "patcher", int((time.time() - start_time) * 1000))
            logger.info("Stage 7 complete: no changes to apply")
            return state

        # Build patches
        patches = []
        for decision in changes:
            patch = PatchOp(
                op="replace",
                path=decision.path,
                value=decision.new_value,
                old_value=decision.old_value,
                reason=decision.reason,
            )
            patches.append(patch)

        # Apply patches
        patcher = get_patcher()
        patch_result = patcher.apply_patches(
            working_json,
            patches,
            validate_first=True,
            stop_on_error=False,
        )

        state["adapted_json"] = patch_result.patched_data
        state["patches_applied"] = len(patch_result.applied_patches)

        duration_ms = int((time.time() - start_time) * 1000)
        add_stage_timing(state, "patcher", duration_ms)

        logger.info(f"Stage 7 complete: {len(patch_result.applied_patches)} patches applied")

    except Exception as e:
        logger.error(f"Patcher failed: {e}")
        add_error(state, "patcher", str(e), is_fatal=True)
        state["adapted_json"] = state["input_json"]

    return state


# =============================================================================
# STAGE 8: FEEDBACK NODE
# =============================================================================

@traceable(name="leaf_stage8_feedback")
async def feedback_node(state: LeafPipelineState) -> LeafPipelineState:
    """
    Stage 8: Generate canonical feedback report using GPT 5.2.

    Produces decision-first report for PM/Client/QA.
    """
    start_time = time.time()
    state["current_stage"] = "feedback"

    try:
        from .feedback_agent import generate_feedback_report

        # Get final validation state
        validation_result = state.get("validation_result")
        if state.get("repair_result"):
            # Re-validate after repairs
            from .leaf_validators import validate_leaf_decisions
            validation_result = await validate_leaf_decisions(
                state["decisions"],
                state["context"],
            )

        feedback_report = await generate_feedback_report(
            decisions=state["decisions"],
            context=state["context"],
            validation_result=validation_result,
            repair_result=state.get("repair_result"),
            total_leaves=state["total_leaves"],
            time_ms=state["total_runtime_ms"],
        )

        state["feedback_report"] = feedback_report
        state["release_decision"] = feedback_report.release_decision

        # Set final status
        state["passed"] = feedback_report.can_ship
        if feedback_report.can_ship:
            state["final_status"] = "OK"
        elif feedback_report.release_decision == "Fix Required":
            state["final_status"] = "FIX_REQUIRED"
        else:
            state["final_status"] = "FAIL"

        duration_ms = int((time.time() - start_time) * 1000)
        add_stage_timing(state, "feedback", duration_ms)

        logger.info(f"Stage 8 complete: {feedback_report.release_decision}")

    except Exception as e:
        logger.error(f"Feedback generation failed: {e}")
        add_error(state, "feedback", str(e), is_fatal=False)
        state["release_decision"] = "UNKNOWN"
        state["final_status"] = "FAIL"

    return state


# =============================================================================
# ROUTING FUNCTIONS
# =============================================================================

def should_repair(state: LeafPipelineState) -> str:
    """Decide if repair loop should run."""
    if state.get("validation_passed", False):
        return "patcher"  # Skip repair, go to patcher
    if state.get("blockers", 0) == 0:
        return "patcher"  # No blockers, skip repair
    return "repair"


def should_abort(state: LeafPipelineState) -> str:
    """Check if pipeline should abort due to fatal error."""
    errors = state.get("errors", [])
    fatal_errors = [e for e in errors if e.get("is_fatal", False)]
    if fatal_errors:
        return "abort"
    return "continue"


# =============================================================================
# WORKFLOW GRAPH CREATION
# =============================================================================

def create_leaf_workflow():
    """
    Create the 8-stage LangGraph workflow for leaf adaptation.

    Flow:
    ┌─────────┐   ┌─────────┐   ┌─────┐   ┌─────────┐
    │ Context │ → │ Indexer │ → │ RAG │ → │ Decider │
    └─────────┘   └─────────┘   └─────┘   └─────────┘
                                               │
                                               ▼
                                        ┌────────────┐
                                        │ Validation │
                                        └─────┬──────┘
                                              │
                          ┌───────────────────┴───────────────────┐
                          │ blockers > 0?                         │
                          │  Yes → Repair Loop → Patcher          │
                          │  No  → Patcher                        │
                          └───────────────────────────────────────┘
                                              │
                                              ▼
                                        ┌──────────┐
                                        │ Feedback │ → END
                                        └──────────┘

    Returns:
        Compiled StateGraph workflow
    """
    # Create workflow graph
    workflow = StateGraph(LeafPipelineState)

    # ==========================================================================
    # ADD NODES
    # ==========================================================================
    workflow.add_node("context", context_node)
    workflow.add_node("indexer", indexer_node)
    workflow.add_node("rag", rag_node)
    workflow.add_node("decider", decider_node)
    workflow.add_node("validation", validation_node)
    workflow.add_node("repair", repair_node)
    workflow.add_node("patcher", patcher_node)
    workflow.add_node("feedback", feedback_node)

    # ==========================================================================
    # SET ENTRY POINT
    # ==========================================================================
    workflow.set_entry_point("context")

    # ==========================================================================
    # ADD EDGES
    # ==========================================================================

    # Linear flow: Context → Indexer → RAG → Decider → Validation
    workflow.add_edge("context", "indexer")
    workflow.add_edge("indexer", "rag")
    workflow.add_edge("rag", "decider")
    workflow.add_edge("decider", "validation")

    # Conditional: Validation → Repair or Patcher
    workflow.add_conditional_edges(
        "validation",
        should_repair,
        {
            "repair": "repair",
            "patcher": "patcher",
        }
    )

    # Repair → Patcher
    workflow.add_edge("repair", "patcher")

    # Patcher → Feedback
    workflow.add_edge("patcher", "feedback")

    # Feedback → END
    workflow.add_edge("feedback", END)

    # ==========================================================================
    # COMPILE
    # ==========================================================================
    return workflow.compile()


# =============================================================================
# RUN PIPELINE FUNCTIONS
# =============================================================================

@traceable(name="run_leaf_pipeline")
async def run_leaf_pipeline(
    input_json: dict,
    scenario_prompt: str,
    use_rag: bool = True,
) -> LeafPipelineState:
    """
    Run the full 8-stage leaf adaptation pipeline.

    Args:
        input_json: Original simulation JSON
        scenario_prompt: Target scenario description
        use_rag: Whether to use RAG (default True)

    Returns:
        Final LeafPipelineState with results
    """
    # Create workflow
    workflow = create_leaf_workflow()

    # Create initial state
    initial_state = create_initial_state(
        input_json=input_json,
        scenario_prompt=scenario_prompt,
        use_rag=use_rag,
    )

    # Run workflow
    config = {"recursion_limit": 25}
    final_state = await workflow.ainvoke(initial_state, config=config)

    # Log summary
    logger.info(
        f"Leaf pipeline complete: status={final_state.get('final_status')}, "
        f"runtime={final_state.get('total_runtime_ms')}ms, "
        f"patches={final_state.get('patches_applied')}"
    )

    return final_state


async def run_leaf_pipeline_streaming(
    input_json: dict,
    scenario_prompt: str,
    use_rag: bool = True,
):
    """
    Run leaf pipeline with streaming state updates.

    Yields state after each node completes.
    """
    workflow = create_leaf_workflow()

    initial_state = create_initial_state(
        input_json=input_json,
        scenario_prompt=scenario_prompt,
        use_rag=use_rag,
    )

    async for state in workflow.astream(initial_state):
        yield state


# =============================================================================
# PRE-COMPILED WORKFLOW INSTANCE
# =============================================================================

leaf_workflow = create_leaf_workflow()
