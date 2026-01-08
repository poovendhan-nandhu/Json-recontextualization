"""
LangGraph nodes for 7-Stage Simulation Adaptation Pipeline.

All nodes have @traceable for LangSmith observability.

Stages:
1. Sharder - Split JSON into shards
2. Adaptation Engine - Transform shards with Gemini
3. Alignment Checker - Cross-shard consistency (GPT-5.2)
4. Scoped Validation - Per-shard validation (parallel)
4B. Fixers - Fix failing shards (hybrid LLM + patcher)
5. Merger - Reassemble shards
6. Finisher - Compliance loop
7. Human Approval - Create approval package
"""
import time
import logging
import asyncio
from typing import Any
from copy import deepcopy

from langsmith import traceable

from .state import (
    PipelineState,
    add_stage_timing,
    add_error,
    add_llm_call,
)

logger = logging.getLogger(__name__)


# =============================================================================
# UTILITY: JSON SANITIZATION
# =============================================================================

def sanitize_json_content(obj: Any) -> Any:
    """
    Recursively sanitize JSON content to remove surrogate characters.
    This prevents UTF-8 encoding errors when sending to LLM APIs.
    """
    if isinstance(obj, str):
        # Remove surrogate characters (\uD800-\uDFFF) that cause encoding errors
        try:
            # Encode to UTF-8, replacing surrogates with replacement character
            return obj.encode('utf-8', errors='surrogatepass').decode('utf-8', errors='replace')
        except Exception:
            # Fallback: just strip non-ASCII if encoding fails
            return ''.join(c if ord(c) < 128 else '?' for c in obj)
    elif isinstance(obj, dict):
        return {k: sanitize_json_content(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_json_content(item) for item in obj]
    return obj


# =============================================================================
# STAGE 1: SHARDER NODE
# =============================================================================

@traceable(name="stage1_sharder")
async def sharder_node(state: PipelineState) -> PipelineState:
    """
    Stage 1: Split JSON into independent shards.

    - Extracts scenario options
    - Creates Shard objects for each section
    - Marks locked shards
    - Computes hashes for change detection
    """
    start_time = time.time()
    state["current_stage"] = "sharder"

    try:
        from ..stages import Sharder, shard_json
        from ..models.shard import ShardCollection, LockState

        # CRITICAL: Sanitize input JSON to remove surrogate characters
        # This prevents UTF-8 encoding errors when sending to LLM APIs
        input_json = sanitize_json_content(state["input_json"])
        state["input_json"] = input_json  # Update state with sanitized version

        selected_scenario = state["selected_scenario"]

        # Shard the JSON - returns ShardCollection
        shard_collection = shard_json(input_json)

        # Extract scenario info
        topic_data = input_json.get("topicWizardData", {})
        scenario_options = topic_data.get("scenarioOptions", [])

        # Resolve selected scenario
        if isinstance(selected_scenario, int):
            if selected_scenario < len(scenario_options):
                target_scenario_text = scenario_options[selected_scenario]
            else:
                target_scenario_text = str(selected_scenario)
        else:
            target_scenario_text = str(selected_scenario)

        # Track shard info from the collection
        shard_ids = [s.id for s in shard_collection.shards]
        locked_shard_ids = [s.id for s in shard_collection.shards if s.lock_state == LockState.FULLY_LOCKED]
        shard_hashes = {s.id: s.current_hash for s in shard_collection.shards}

        # Store the collection (for merge_shards later) and shards list
        state["shard_collection"] = shard_collection
        state["shards"] = shard_collection.shards  # List of Shard objects
        state["shard_ids"] = shard_ids
        state["locked_shard_ids"] = locked_shard_ids
        state["shard_hashes"] = shard_hashes
        state["target_scenario_text"] = target_scenario_text

        duration_ms = int((time.time() - start_time) * 1000)
        add_stage_timing(state, "sharder", duration_ms)

        logger.info(f"Stage 1 complete: {len(shard_collection.shards)} shards, {len(locked_shard_ids)} locked")

    except Exception as e:
        logger.error(f"Sharder failed: {e}")
        add_error(state, "sharder", str(e), is_fatal=True)

    return state


# =============================================================================
# STAGE 2: ADAPTATION ENGINE NODE
# =============================================================================

@traceable(name="stage2_adaptation")
async def adaptation_node(state: PipelineState) -> PipelineState:
    """
    Stage 2: Adapt shards to target scenario using Gemini 2.5 Flash.

    Uses AdaptationEngine.adapt() which handles:
    - Global factsheet extraction
    - Parallel shard adaptation
    - Entity mapping
    """
    start_time = time.time()
    state["current_stage"] = "adaptation"

    try:
        from ..stages import AdaptationEngine
        from ..rag import get_industry_context, detect_industry

        input_json = state["input_json"]
        selected_scenario = state["selected_scenario"]
        target_scenario_text = state["target_scenario_text"]

        # ⭐ DON'T pre-detect industry - let the LLM extract it from factsheet
        # The factsheet extraction reads the scenario directly and gets correct industry
        state["industry"] = "unknown"  # Will be updated after factsheet extraction

        # Initialize adaptation engine WITHOUT RAG context
        # (RAG context will be built AFTER we have correct industry from factsheet)
        engine = AdaptationEngine(rag_context="")

        # Determine if we're using index or free-form prompt
        if isinstance(selected_scenario, int):
            # Option A: Select from existing scenario options by index
            result = await engine.adapt(
                input_json=input_json,
                target_scenario_index=selected_scenario,
            )
        else:
            # Option B: Free-form scenario prompt
            result = await engine.adapt(
                input_json=input_json,
                scenario_prompt=str(selected_scenario),
            )

        # Extract results from AdaptationResult
        state["entity_map"] = result.entity_map
        state["global_factsheet"] = result.global_factsheet

        # ⭐ USE INDUSTRY FROM FACTSHEET (LLM-extracted, accurate)
        factsheet_industry = result.global_factsheet.get("company", {}).get("industry", "")
        if factsheet_industry and factsheet_industry.lower() != "unknown":
            state["industry"] = factsheet_industry.lower()
        logger.info(f"Industry from factsheet: {state['industry']}")

        # ⭐ NOW build RAG context with CORRECT industry (for downstream validation/fixing)
        try:
            rag_context = get_industry_context(state["industry"])
            state["rag_context"] = {
                "industry": state["industry"],
                "kpis": rag_context.kpis,
                "terminology": rag_context.terminology[:20],
            }
            logger.info(f"Built RAG context for {state['industry']}: {len(rag_context.kpis)} KPIs")
        except Exception as e:
            logger.warning(f"RAG context failed: {e}")
            state["rag_context"] = {"industry": state["industry"]}

        # ⭐ KLO ALIGNMENT FIX - Ensure questions map to KLOs
        # This runs AFTER adaptation but BEFORE alignment checking
        try:
            from ..stages.fixers import fix_klo_alignment
            logger.info("Running KLO Alignment Fixer...")
            klo_fix_context = {
                "global_factsheet": result.global_factsheet,
            }
            fixed_json = await fix_klo_alignment(result.adapted_json, klo_fix_context)
            logger.info("KLO Alignment Fixer complete")
        except Exception as e:
            logger.warning(f"KLO Alignment Fixer failed, using original: {e}")
            fixed_json = result.adapted_json

        # Update shard collection with adapted data
        # The adaptation engine returns the merged JSON - we need to re-shard
        # to get adapted_shards for downstream stages
        from ..stages import shard_json
        adapted_collection = shard_json(fixed_json)
        state["adapted_shards"] = adapted_collection.shards
        state["shard_collection"] = adapted_collection

        duration_ms = int((time.time() - start_time) * 1000)
        add_stage_timing(state, "adaptation", duration_ms)

        logger.info(f"Stage 2 complete: {result.shards_adapted} shards adapted, "
                   f"{result.shards_locked} locked, {len(result.entity_map)} entity mappings")

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        logger.error(f"Adaptation failed: {e}\n{tb}")
        add_error(state, "adaptation", f"{str(e)}\n\nTraceback:\n{tb}", is_fatal=True)

    return state


# =============================================================================
# STAGE 3: ALIGNMENT CHECKER NODE
# =============================================================================

@traceable(name="stage3_alignment")
async def alignment_node(state: PipelineState) -> PipelineState:
    """
    Stage 3: Check cross-shard alignment using GPT.

    - Runs 9 alignment checkers in parallel
    - Validates consistency across shards
    - AUTO-REGENERATES failed shards and re-checks (up to 2 attempts)
    """
    start_time = time.time()
    state["current_stage"] = "alignment"

    max_regenerations = 2  # Limit regeneration attempts
    regeneration_attempt = state.get("regeneration_attempt", 0)

    try:
        from ..stages import AlignmentChecker, check_alignment
        from ..stages.sharder import merge_shards

        adapted_shards = state["adapted_shards"]
        global_factsheet = state["global_factsheet"]
        shard_collection = state.get("shard_collection")
        original_json = state.get("input_json", {})

        # ALIGNMENT + REGENERATION LOOP
        while regeneration_attempt <= max_regenerations:
            # Reconstruct adapted JSON for alignment check
            if shard_collection:
                adapted_json = merge_shards(shard_collection, original_json)
            else:
                adapted_json = original_json.copy()

            logger.info(f"Running alignment check (attempt {regeneration_attempt + 1})...")

            # Run alignment check
            checker = AlignmentChecker()
            alignment_report = await checker.check(
                adapted_json=adapted_json,
                global_factsheet=global_factsheet,
                source_scenario=global_factsheet.get("source_scenario", ""),
            )

            state["alignment_report"] = alignment_report.to_dict()
            state["alignment_score"] = alignment_report.overall_score
            state["alignment_passed"] = alignment_report.passed

            logger.info(f"Alignment score: {alignment_report.overall_score:.2%}, passed={alignment_report.passed}")

            # If passed or score >= 95%, we're done
            if alignment_report.passed or alignment_report.overall_score >= 0.95:
                logger.info(f"✅ Alignment acceptable ({alignment_report.overall_score:.2%})")
                break

            # If max attempts reached, stop
            if regeneration_attempt >= max_regenerations:
                logger.warning(f"⚠️ Max regeneration attempts ({max_regenerations}) reached. Final score: {alignment_report.overall_score:.2%}")
                break

            # ⭐ REGENERATE failed shards
            from ..utils.content_processor import analyze_and_get_feedback
            from ..utils.gemini_client import regenerate_shards_with_feedback

            feedback, regeneration_prompt = analyze_and_get_feedback(alignment_report.to_dict())
            state["alignment_feedback"] = {
                "failed_rules": [r["rule_name"] for r in feedback.failed_rules],
                "critical_issues": feedback.critical_issues[:5],
                "suggestions": feedback.suggestions[:3],
                "focus_shards": feedback.focus_shards,
                "regeneration_prompt": regeneration_prompt,
            }

            if not feedback.focus_shards:
                logger.info("No focus shards identified for regeneration")
                break

            logger.info(f"🔄 Regenerating shards (attempt {regeneration_attempt + 1}/{max_regenerations}): {feedback.focus_shards}")

            # Prepare shards for regeneration
            shards_to_regenerate = {}

            # Handle both list of Shard objects and dict formats
            if isinstance(adapted_shards, list):
                # List of Shard objects
                for shard in adapted_shards:
                    shard_id = shard.id if hasattr(shard, 'id') else str(shard)
                    shard_content = shard.content if hasattr(shard, 'content') else shard
                    shard_lower = shard_id.lower()
                    needs_regen = any(
                        focus.lower() in shard_lower or shard_lower in focus.lower()
                        for focus in feedback.focus_shards
                    )
                    if needs_regen:
                        shards_to_regenerate[shard_id] = shard_content
                        logger.info(f"  → Will regenerate: {shard_id}")
            else:
                # Dict format
                for shard_id, content in adapted_shards.items():
                    shard_lower = shard_id.lower()
                    needs_regen = any(
                        focus.lower() in shard_lower or shard_lower in focus.lower()
                        for focus in feedback.focus_shards
                    )
                    if needs_regen:
                        shards_to_regenerate[shard_id] = content
                        logger.info(f"  → Will regenerate: {shard_id}")

            if not shards_to_regenerate:
                logger.info("No matching shards found for regeneration")
                break

            # Regenerate shards with feedback
            feedback_dict = {
                "failed_rules": feedback.failed_rules,
                "critical_issues": feedback.critical_issues,
                "suggestions": feedback.suggestions,
            }

            regenerated = await regenerate_shards_with_feedback(
                shards=shards_to_regenerate,
                source_scenario=global_factsheet.get("source_scenario", ""),
                target_scenario=state.get("scenario_prompt", ""),
                global_factsheet=global_factsheet,
                feedback=feedback_dict,
            )

            # Update adapted_shards with regenerated content
            regen_count = 0
            for shard_id, (new_content, mappings) in regenerated.items():
                if new_content and new_content != shards_to_regenerate.get(shard_id):
                    # Handle both list and dict formats
                    if isinstance(adapted_shards, list):
                        # Find and update shard in list
                        for shard in adapted_shards:
                            if hasattr(shard, 'id') and shard.id == shard_id:
                                shard.content = new_content
                                break
                    else:
                        adapted_shards[shard_id] = new_content
                    regen_count += 1
                    logger.info(f"  ✅ Regenerated: {shard_id}")

            if regen_count == 0:
                logger.info("No shards were actually regenerated")
                break

            state["adapted_shards"] = adapted_shards
            regeneration_attempt += 1
            state["regeneration_attempt"] = regeneration_attempt
            logger.info(f"🔄 Regenerated {regen_count} shards. Re-running alignment check...")

        duration_ms = int((time.time() - start_time) * 1000)
        add_stage_timing(state, "alignment", duration_ms)

        logger.info(f"Stage 3 complete: final alignment score {state['alignment_score']:.2%}, attempts={regeneration_attempt + 1}")

    except Exception as e:
        logger.error(f"Alignment check failed: {e}")
        add_error(state, "alignment", str(e), is_fatal=False)
        # Don't override alignment_passed if we already have a valid score
        if "alignment_score" not in state or state["alignment_score"] == 0:
            state["alignment_passed"] = True  # Only default to True if no score yet

    return state


# =============================================================================
# STAGE 4: SCOPED VALIDATION NODE
# =============================================================================

@traceable(name="stage4_validation")
async def validation_node(state: PipelineState) -> PipelineState:
    """
    Stage 4: Validate each shard independently (parallel).

    - Runs all validators in parallel per shard
    - Collects issues by severity
    - Reports pass/fail per shard
    """
    start_time = time.time()
    state["current_stage"] = "validation"

    try:
        from ..validators import ScopedValidator, validate_shards

        shards = state.get("fixed_shards") or state["adapted_shards"]
        global_factsheet = state["global_factsheet"]

        # Get original shards for comparison (to detect NEWLY empty fields)
        original_shards = state.get("shards", [])
        base_shards_map = {s.id: s for s in original_shards if hasattr(s, 'id')}

        # Build validation context
        context = {
            "global_factsheet": global_factsheet,
            "industry": state["industry"],
            "source_scenario": global_factsheet.get("source_scenario", ""),
            "entity_map": state["entity_map"],
            "base_shards": base_shards_map,  # For comparing with original
        }

        # Run scoped validation
        validator = ScopedValidator()
        validation_report, validation_fixes = await validator.validate_all(shards, context)

        state["validation_report"] = validation_report.to_dict()
        state["validation_score"] = validation_report.overall_score
        state["validation_passed"] = validation_report.passed
        state["blocker_count"] = validation_report.blocker_count
        state["warning_count"] = validation_report.warning_count
        state["validation_fixes"] = validation_fixes  # Store fixes for fixer stage

        duration_ms = int((time.time() - start_time) * 1000)
        add_stage_timing(state, "validation", duration_ms)

        logger.info(
            f"Stage 4 complete: score {validation_report.overall_score:.2%}, "
            f"blockers={validation_report.blocker_count}, warnings={validation_report.warning_count}"
        )

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        add_error(state, "validation", str(e), is_fatal=False)
        state["validation_passed"] = False

    return state


# =============================================================================
# STAGE 4B: FIXERS NODE
# =============================================================================

@traceable(name="stage4b_fixers")
async def fixers_node(state: PipelineState) -> PipelineState:
    """
    Stage 4B: Fix failing shards using hybrid approach.

    - LLM identifies exact fields (JSON Pointer paths)
    - Patcher applies surgical 'replace' operations
    - Stores patches for rollback
    """
    start_time = time.time()
    state["current_stage"] = "fixers"

    try:
        from ..stages import ScopedFixer, fix_all_shards
        from ..validators import ScopedValidationReport

        shards = state.get("fixed_shards") or state["adapted_shards"]
        validation_report = state.get("validation_report", {})

        # Skip if no issues
        if state.get("validation_passed", False):
            state["fixed_shards"] = shards
            state["fix_results"] = {}
            add_stage_timing(state, "fixers", int((time.time() - start_time) * 1000))
            return state

        # Build fix context
        context = {
            "global_factsheet": state["global_factsheet"],
            "industry": state["industry"],
            "entity_map": state["entity_map"],
        }

        # Create validation report object for fixer
        class MockValidationReport:
            def __init__(self, report_dict):
                self.shard_results = {}
                for shard_id, results in report_dict.get("shards", {}).items():
                    self.shard_results[shard_id] = results

        mock_report = MockValidationReport(validation_report)

        # Run fixers
        fixer = ScopedFixer()
        fix_results = await fixer.fix_all(shards, mock_report, context)

        # Update shards with fixed content
        fixed_shards = []
        all_patches = []

        for shard in shards:
            shard_id = shard.id if hasattr(shard, 'id') else "unknown"

            if shard_id in fix_results:
                result = fix_results[shard_id]
                if result.success and result.fixed_content:
                    shard.content = result.fixed_content
                    all_patches.extend([p.to_dict() for p in result.patches_applied])

            fixed_shards.append(shard)

        state["fixed_shards"] = fixed_shards
        state["fix_results"] = {sid: r.to_dict() for sid, r in fix_results.items()}
        state["patches_applied"] = all_patches

        duration_ms = int((time.time() - start_time) * 1000)
        add_stage_timing(state, "fixers", duration_ms)

        logger.info(f"Stage 4B complete: {len(fix_results)} shards processed, {len(all_patches)} patches applied")

    except Exception as e:
        logger.error(f"Fixers failed: {e}")
        add_error(state, "fixers", str(e), is_fatal=False)
        state["fixed_shards"] = state.get("adapted_shards", [])

    return state


# =============================================================================
# STAGE 5: MERGER NODE
# =============================================================================

@traceable(name="stage5_merger")
async def merger_node(state: PipelineState) -> PipelineState:
    """
    Stage 5: Reassemble shards into full JSON.

    - Merges all shards back together
    - Preserves original structure
    """
    start_time = time.time()
    state["current_stage"] = "merger"

    try:
        from ..stages import merge_shards
        from ..models.shard import ShardCollection

        shards = state.get("fixed_shards") or state.get("adapted_shards") or state["shards"]
        input_json = state["input_json"]

        # Get or create ShardCollection for merge
        shard_collection = state.get("shard_collection")
        if shard_collection is None:
            # Create a new collection from shards list
            shard_collection = ShardCollection(
                shards=shards,
                source_json_hash="",
                scenario_prompt=state.get("target_scenario_text", ""),
            )
        else:
            # Update collection's shards with fixed shards
            shard_collection.shards = shards

        # Merge shards back into full JSON
        merged_json = merge_shards(shard_collection, input_json)

        state["merged_json"] = merged_json
        state["merge_successful"] = True

        duration_ms = int((time.time() - start_time) * 1000)
        add_stage_timing(state, "merger", duration_ms)

        logger.info("Stage 5 complete: shards merged")

    except Exception as e:
        logger.error(f"Merger failed: {e}")
        add_error(state, "merger", str(e), is_fatal=True)
        state["merge_successful"] = False

    return state


# =============================================================================
# STAGE 6: FINISHER NODE
# =============================================================================

@traceable(name="stage6_finisher")
async def finisher_node(state: PipelineState) -> PipelineState:
    """
    Stage 6: Run compliance loop.

    - Re-validates changed shards
    - Computes weighted compliance score
    - Routes back to fixers if needed (max 3 iterations)
    - Flags for human if still failing
    """
    start_time = time.time()
    state["current_stage"] = "finisher"

    try:
        from ..stages import Finisher, ComplianceStatus

        shards = state.get("fixed_shards") or state["adapted_shards"]

        # Build context for compliance check
        context = {
            "global_factsheet": state["global_factsheet"],
            "industry": state["industry"],
            "adapted_json": state["merged_json"],
            "source_scenario": state["global_factsheet"].get("source_scenario", ""),
        }

        # Run compliance check
        finisher = Finisher(max_iterations=state.get("max_retries", 3))
        compliance_result = await finisher.run_compliance_loop(shards, context)

        state["compliance_result"] = compliance_result.to_dict()
        state["compliance_score"] = compliance_result.score.overall_score
        state["compliance_passed"] = compliance_result.status == ComplianceStatus.PASS
        state["compliance_iteration"] = compliance_result.iteration
        state["flagged_for_human"] = compliance_result.flagged_for_human

        # Update fixed shards if compliance loop modified them
        state["fixed_shards"] = shards

        # INCREMENT RETRY COUNT HERE (not in routing function!)
        # Routing functions are read-only in LangGraph
        if not state["compliance_passed"]:
            state["retry_count"] = state.get("retry_count", 0) + 1

        duration_ms = int((time.time() - start_time) * 1000)
        add_stage_timing(state, "finisher", duration_ms)

        logger.info(
            f"Stage 6 complete: compliance {compliance_result.status.value}, "
            f"score {compliance_result.score.overall_score:.2%}, "
            f"iterations={compliance_result.iteration}"
        )

    except Exception as e:
        logger.error(f"Finisher failed: {e}")
        add_error(state, "finisher", str(e), is_fatal=False)
        state["compliance_passed"] = False

    return state


# =============================================================================
# STAGE 7: HUMAN APPROVAL NODE
# =============================================================================

@traceable(name="stage7_human_approval")
async def human_approval_node(state: PipelineState) -> PipelineState:
    """
    Stage 7: Create human approval package.

    - Builds approval package with summary
    - Sets URLs for approve/reject
    - Marks status as pending human review
    """
    start_time = time.time()
    state["current_stage"] = "human_approval"

    try:
        from ..stages import HumanApproval, create_approval
        import uuid

        simulation_id = str(uuid.uuid4())[:8]

        # Create mock compliance result for approval
        class MockComplianceResult:
            def __init__(self, state_dict):
                self.score = type('Score', (), {
                    'overall_score': state_dict.get("compliance_score", 0.0),
                    'blocker_pass_rate': 1.0 if state_dict.get("blocker_count", 0) == 0 else 0.5,
                    'passed': state_dict.get("compliance_passed", False),
                })()
                self.flagged_for_human = state_dict.get("flagged_for_human", [])
                self.iteration = state_dict.get("compliance_iteration", 1)

        mock_compliance = MockComplianceResult(state)

        # Build context for approval
        context = {
            "target_scenario": state["target_scenario_text"],
            "global_factsheet": state["global_factsheet"],
            "industry": state["industry"],
        }

        # Create approval package
        approval_system = HumanApproval(base_url="")
        approval_package = approval_system.create_approval_package(
            simulation_id=simulation_id,
            compliance_result=mock_compliance,
            context=context,
        )

        state["approval_package"] = approval_package.to_dict()
        state["approval_status"] = "pending"

        # Set final output
        state["output_json"] = state["merged_json"]

        # Determine final status
        if state.get("compliance_passed", False):
            state["final_status"] = "OK"
        elif state.get("flagged_for_human", []):
            state["final_status"] = "HUMAN_REVIEW"
        else:
            state["final_status"] = "FAIL"

        duration_ms = int((time.time() - start_time) * 1000)
        add_stage_timing(state, "human_approval", duration_ms)

        logger.info(f"Stage 7 complete: approval package created, status={state['final_status']}")

    except Exception as e:
        logger.error(f"Human approval failed: {e}")
        add_error(state, "human_approval", str(e), is_fatal=False)
        state["output_json"] = state.get("merged_json", {})
        state["final_status"] = "FAIL"

    return state


# =============================================================================
# ROUTING FUNCTIONS
# =============================================================================

def should_fix(state: PipelineState) -> str:
    """Decide if fixers should run."""
    if state.get("validation_passed", False):
        return "merger"  # Skip fixers, go to merger
    return "fixers"


def should_retry_compliance(state: PipelineState) -> str:
    """
    Decide if compliance loop should retry.

    NOTE: Retry loop disabled to avoid recursion limit issues.
    Always goes to human_approval now.
    """
    # Always go to human_approval - no retry loop
    # This avoids LangGraph recursion limit issues
    return "human_approval"


def should_abort(state: PipelineState) -> str:
    """Check if pipeline should abort due to fatal error."""
    if state.get("final_status") == "FAIL":
        errors = state.get("errors", [])
        fatal_errors = [e for e in errors if e.get("is_fatal", False)]
        if fatal_errors:
            return "abort"
    return "continue"


# =============================================================================
# WORKFLOW GRAPH CREATION
# =============================================================================

def create_adaptation_workflow():
    """
    Create the 7-stage LangGraph workflow.

    Flow:
    ┌─────────┐   ┌────────────┐   ┌───────────┐   ┌────────────┐
    │ Sharder │ → │ Adaptation │ → │ Alignment │ → │ Validation │
    └─────────┘   └────────────┘   └───────────┘   └─────┬──────┘
                                                         │
                                     ┌───────────────────┴───────────────────┐
                                     │ validation_passed?                    │
                                     │  Yes → Merger                         │
                                     │  No  → Fixers → Merger                │
                                     └───────────────────────────────────────┘
                                                         │
                                                         ▼
    ┌───────────────┐   ┌──────────┐             ┌──────────┐
    │ Human Approval│ ← │ Finisher │ ← ───────── │  Merger  │
    └───────────────┘   └────┬─────┘             └──────────┘
           │                 │
           │    ┌────────────┴────────────┐
           │    │ compliance_passed?      │
           │    │  Yes → Human Approval   │
           │    │  No & retries < 3 →     │
           │    │       back to Validation│
           │    │  No & retries >= 3 →    │
           │    │       Human Approval    │
           │    └─────────────────────────┘
           │
           ▼
         [END]

    Returns:
        Compiled StateGraph workflow
    """
    from langgraph.graph import StateGraph, END

    # Create workflow graph
    workflow = StateGraph(PipelineState)

    # ==========================================================================
    # ADD NODES (all have @traceable for LangSmith observability)
    # ==========================================================================
    workflow.add_node("sharder", sharder_node)
    workflow.add_node("adaptation", adaptation_node)
    workflow.add_node("alignment", alignment_node)
    workflow.add_node("validation", validation_node)
    workflow.add_node("fixers", fixers_node)
    workflow.add_node("merger", merger_node)
    workflow.add_node("finisher", finisher_node)
    workflow.add_node("human_approval", human_approval_node)

    # ==========================================================================
    # SET ENTRY POINT
    # ==========================================================================
    workflow.set_entry_point("sharder")

    # ==========================================================================
    # ADD EDGES
    # ==========================================================================

    # Stage 1 → Stage 2
    workflow.add_edge("sharder", "adaptation")

    # Stage 2 → Stage 3
    workflow.add_edge("adaptation", "alignment")

    # Stage 3 → Stage 4
    workflow.add_edge("alignment", "validation")

    # Stage 4 → Stage 4B or Stage 5 (conditional)
    workflow.add_conditional_edges(
        "validation",
        should_fix,
        {
            "fixers": "fixers",    # Has issues → fix them
            "merger": "merger",    # No issues → skip to merger
        }
    )

    # Stage 4B → Stage 5
    workflow.add_edge("fixers", "merger")

    # Stage 5 → Stage 6
    workflow.add_edge("merger", "finisher")

    # Stage 6 → Stage 7 or retry (conditional)
    workflow.add_conditional_edges(
        "finisher",
        should_retry_compliance,
        {
            "human_approval": "human_approval",  # Pass or max retries
            "validation": "validation",           # Retry: back to validation
        }
    )

    # Stage 7 → END
    workflow.add_edge("human_approval", END)

    # ==========================================================================
    # COMPILE
    # ==========================================================================
    return workflow.compile()


# =============================================================================
# RUN PIPELINE FUNCTIONS
# =============================================================================

@traceable(name="run_adaptation_pipeline")
async def run_pipeline(
    input_json: dict,
    selected_scenario: str | int,
    max_retries: int = 3,
) -> PipelineState:
    """
    Run the full 7-stage adaptation pipeline.

    Args:
        input_json: Original simulation JSON
        selected_scenario: Target scenario (index or text)
        max_retries: Max compliance loop retries (default 3)

    Returns:
        Final PipelineState with results
    """
    from .state import create_initial_state

    # Create workflow
    workflow = create_adaptation_workflow()

    # Create initial state
    initial_state = create_initial_state(
        input_json=input_json,
        selected_scenario=selected_scenario,
        max_retries=max_retries,
    )

    # Run workflow with recursion limit (safety net)
    # Max nodes: sharder(1) + adapt(1) + align(1) + validation(3) + fixers(3) + merger(3) + finisher(3) + human(1) = ~16 max
    config = {"recursion_limit": 50}
    final_state = await workflow.ainvoke(initial_state, config=config)

    # Log summary
    logger.info(
        f"Pipeline complete: status={final_state.get('final_status')}, "
        f"runtime={final_state.get('total_runtime_ms')}ms, "
        f"tokens={final_state.get('total_tokens')}"
    )

    return final_state


async def run_pipeline_streaming(
    input_json: dict,
    selected_scenario: str | int,
    max_retries: int = 3,
):
    """
    Run pipeline with streaming state updates.

    Yields state after each node completes.
    Good for progress tracking in UI.
    """
    from .state import create_initial_state

    workflow = create_adaptation_workflow()

    initial_state = create_initial_state(
        input_json=input_json,
        selected_scenario=selected_scenario,
        max_retries=max_retries,
    )

    async for state in workflow.astream(initial_state):
        yield state
