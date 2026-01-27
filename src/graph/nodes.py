"""
LangGraph Nodes for Simple Adaptation Pipeline.

SIMPLIFIED 3-STAGE PIPELINE:
  ADAPT → VALIDATE → REPAIR (loop) → FINALIZE

Uses:
- src/stages/simple_adapter.py (Gemini adaptation)
- src/stages/simple_validators.py (GPT validation + repair)

Flow:
    ┌─────────┐
    │  START  │
    └────┬────┘
         │
         ▼
    ┌─────────┐
    │  ADAPT  │  (Gemini 2.5 Flash)
    └────┬────┘
         │
         ▼
    ┌──────────┐
    │ VALIDATE │  (6 GPT validators in parallel)
    └────┬─────┘
         │
         ▼
    ┌────────────────────┐
    │  score >= 95%?     │
    │   YES → FINALIZE   │
    │   NO  → REPAIR     │
    └─────────┬──────────┘
              │
         ┌────┴────┐
         │  REPAIR │  (Resource Fixer + Generic Patcher)
         └────┬────┘
              │
              ▼
         (back to VALIDATE, max 3 iterations)
              │
              ▼
    ┌──────────┐
    │ FINALIZE │
    └────┬─────┘
         │
         ▼
    ┌─────────┐
    │   END   │
    └─────────┘
"""
import json
import time
import logging
from typing import Literal
from dataclasses import dataclass

from langgraph.graph import StateGraph, END

try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

from .state import PipelineState

logger = logging.getLogger(__name__)

# Thresholds
PASS_THRESHOLD = 0.95
MAX_REPAIR_ITERATIONS = 2  # Reduced from 3 - over-repair causes regression


# =============================================================================
# NODE: ADAPT
# =============================================================================

@traceable(name="node_adapt", run_type="chain")
async def node_adapt(state: PipelineState) -> PipelineState:
    """
    Adaptation node - transforms JSON to target scenario using Gemini.

    Uses simple_adapter.adapt_simple() which:
    1. Generates entity_map + domain_profile in one LLM call
    2. Adapts all shards in parallel
    3. Merges back to full JSON
    4. **NEW**: Regenerates any truncated resources with GPT + full context
    """
    from src.stages.simple_adapter import adapt_simple

    logger.info("[NODE] ========== ADAPT ==========")
    start_time = time.time()

    try:
        result = await adapt_simple(
            input_json=state["input_json"],
            scenario_prompt=state["scenario_prompt"]
        )

        state["adapted_json"] = result.adapted_json
        state["entity_map"] = result.entity_map
        state["domain_profile"] = result.domain_profile
        state["adaptation_time_ms"] = result.time_ms
        state["shards_processed"] = result.shards_processed

        if result.errors:
            state["errors"].extend(result.errors)

        logger.info(f"[NODE] Adapt complete: {result.shards_processed} shards, {result.time_ms}ms")

        # === NEW: Check and regenerate truncated resources ===
        adapted_json = state["adapted_json"]
        if adapted_json:
            regenerated = await _regenerate_short_resources(
                adapted_json,
                state["scenario_prompt"],
                state["entity_map"],
                state["domain_profile"]
            )
            state["adapted_json"] = regenerated

    except Exception as e:
        logger.error(f"[NODE] Adapt failed: {e}")
        state["errors"].append(f"Adaptation error: {str(e)}")
        state["adapted_json"] = state["input_json"]  # Fallback
        state["status"] = "failed"

    state["stage_timings"]["adapt"] = int((time.time() - start_time) * 1000)
    return state


async def _regenerate_short_resources(
    adapted_json: dict,
    scenario_prompt: str,
    entity_map: dict,
    domain_profile: dict,
    min_words: int = 500
) -> dict:
    """
    Find and regenerate any MAIN resources with < min_words using GPT with full context.

    IMPORTANT: Only regenerates MAIN resources (markdown_text/content).
    SKIPS resource_options because they are METADATA (supposed to be 50-100 words).

    This runs AFTER Gemini adaptation but BEFORE validation.
    Uses GPT because it follows word count instructions better than Gemini.
    """
    import copy
    from src.stages.simple_validators import _call_gpt_async

    result = copy.deepcopy(adapted_json)

    # Find MAIN resources only (skip resource_options - they're metadata)
    resource_paths = []
    sim_flow = result.get("simulation_flow", result.get("simulationFlow", []))

    for i, stage in enumerate(sim_flow):
        if not isinstance(stage, dict):
            continue
        data = stage.get("data", {})
        if not isinstance(data, dict):
            continue

        # ONLY check main resources - NOT resource_options
        if "resource" in data and data["resource"]:
            resource_paths.append((f"simulation_flow/{i}/data/resource", data["resource"]))

        # SKIP resource_options - they are METADATA (supposed to be short ~50-100 words)
        # DO NOT regenerate them

    logger.info(f"[RESOURCE REGEN] Found {len(resource_paths)} MAIN resources to check (skipping resource_options)")

    # Check each resource
    regenerated_count = 0
    for path, resource in resource_paths:
        if not isinstance(resource, dict):
            continue

        # Get content - check all possible keys including markdown_text
        content = (
            resource.get("markdown_text") or  # Primary key in input data
            resource.get("content") or
            resource.get("html") or
            resource.get("text") or
            resource.get("body") or
            ""
        )

        if not content:
            continue

        word_count = len(str(content).split())

        if word_count < min_words:
            logger.info(f"[RESOURCE REGEN] {path}: {word_count} words < {min_words}, regenerating...")

            # Build context from adaptation
            company_name = entity_map.get("company", {}).get("name", "the company") if entity_map else "the company"
            terminology = domain_profile.get("terminology_map", {}) if domain_profile else {}
            forbidden = domain_profile.get("forbidden_terms", []) if domain_profile else []

            # Build regeneration prompt with FULL CONTEXT
            prompt = f"""Regenerate this resource content for a business simulation.

## SCENARIO:
{scenario_prompt}

## COMPANY NAME TO USE:
{company_name}

## TERMINOLOGY TO USE (source → target):
{json.dumps(dict(list(terminology.items())[:30]), indent=2) if terminology else "Use scenario-appropriate terms"}

## TERMS TO AVOID (from source scenario):
{', '.join(forbidden[:40]) if forbidden else "None specified"}

## CURRENT CONTENT (too short at {word_count} words):
{content[:3000]}

## REQUIREMENTS:
1. Generate 800-1200 words of FACTUAL DATA
2. Include specific numbers, percentages, statistics
3. Use tables and structured data where appropriate
4. NO recommendations or conclusions - just DATA
5. NO "should", "recommend", "therefore", "suggests"
6. Use the company name "{company_name}" consistently
7. Avoid ALL terms from the forbidden list

## OUTPUT:
Return ONLY the regenerated content text (no JSON wrapper, no explanation)."""

            try:
                new_content = await _call_gpt_async(
                    prompt,
                    system="You are a business data writer. Generate factual resource content with statistics and data. Never give recommendations."
                )

                # Clean response
                new_content = new_content.strip()
                if new_content.startswith("```"):
                    lines = new_content.split("\n")[1:]
                    if lines and lines[-1].strip() == "```":
                        lines = lines[:-1]
                    new_content = "\n".join(lines)

                new_word_count = len(new_content.split())
                logger.info(f"[RESOURCE REGEN] {path}: {word_count} → {new_word_count} words")

                # Update the resource
                # Navigate to the path and update
                parts = path.split("/")
                obj = result
                for part in parts[:-1]:
                    if part.isdigit():
                        obj = obj[int(part)]
                    else:
                        obj = obj.get(part, obj.get(part.replace("_", ""), {}))

                last_part = parts[-1]
                if last_part.isdigit():
                    target = obj[int(last_part)]
                else:
                    target = obj.get(last_part, {})

                if isinstance(target, dict):
                    # Update the content field
                    if "content" in target:
                        target["content"] = new_content
                    elif "html" in target:
                        target["html"] = new_content
                    elif "text" in target:
                        target["text"] = new_content
                    elif "body" in target:
                        target["body"] = new_content
                    else:
                        target["content"] = new_content

                    regenerated_count += 1

            except Exception as e:
                logger.error(f"[RESOURCE REGEN] Failed for {path}: {e}")

    logger.info(f"[RESOURCE REGEN] Regenerated {regenerated_count}/{len(resource_paths)} resources")
    return result


# =============================================================================
# NODE: VALIDATE
# =============================================================================

@traceable(name="node_validate", run_type="chain")
async def node_validate(state: PipelineState) -> PipelineState:
    """
    Validation node - runs 6 validators in parallel.

    Validators:
    1. Domain Fidelity - correct industry terminology
    2. Context Fidelity - goal/challenge preserved
    3. Resource Quality - data not answers, word count
    4. KLO-Question Alignment - questions map to KLOs
    5. Consistency - names/companies consistent
    6. Completeness - no missing content
    """
    from src.stages.simple_validators import run_all_validators

    logger.info("[NODE] ========== VALIDATE ==========")
    start_time = time.time()

    try:
        json_to_validate = state.get("adapted_json") or state["input_json"]

        report = await run_all_validators(
            adapted_json=json_to_validate,
            scenario_prompt=state["scenario_prompt"]
        )

        state["validation_score"] = report.overall_score
        state["validation_passed"] = report.passed
        state["validation_issues"] = []
        state["agent_scores"] = {}
        state["agent_results"] = report.agent_results  # Store for report generation

        # Collect issues and scores
        for agent_result in report.agent_results:
            state["agent_scores"][agent_result.agent_name] = agent_result.score
            for issue in agent_result.issues:
                state["validation_issues"].append({
                    "agent": issue.agent,
                    "location": issue.location,
                    "issue": issue.issue,
                    "suggestion": issue.suggestion,
                    "severity": issue.severity
                })

        logger.info(f"[NODE] Validate: {report.overall_score:.2%}, {len(state['validation_issues'])} issues")
        for name, score in state["agent_scores"].items():
            status = "[OK]" if score >= PASS_THRESHOLD else "[!!]"
            logger.info(f"[NODE]   {status} {name}: {score:.2%}")

    except Exception as e:
        logger.error(f"[NODE] Validate failed: {e}")
        state["errors"].append(f"Validation error: {str(e)}")
        state["validation_score"] = 0.0
        state["validation_passed"] = False

    state["stage_timings"]["validate"] = int((time.time() - start_time) * 1000)
    return state


# =============================================================================
# NODE: REPAIR
# =============================================================================

@traceable(name="node_repair", run_type="chain")
async def node_repair(state: PipelineState) -> PipelineState:
    """
    Repair node - fixes issues found by validators.

    Two repair strategies:
    1. Resource Fixer - regenerates content (for word count, direct answers)
    2. Generic Patcher - find/replace patches (for domain terms, consistency)
    """
    from src.stages.simple_validators import (
        fix_resource_quality,
        fix_context_fidelity,
        fix_completeness,
        repair_issues,
        ValidationReport,
        AgentResult,
        ValidationIssue
    )

    iteration = state.get("repair_iteration", 0) + 1
    state["repair_iteration"] = iteration

    logger.info(f"[NODE] ========== REPAIR (iter {iteration}/{MAX_REPAIR_ITERATIONS}) ==========")
    start_time = time.time()

    try:
        current_json = state.get("adapted_json") or state["input_json"]
        issues = state.get("validation_issues", [])

        if not issues:
            logger.info("[NODE] No issues to repair")
            state["stage_timings"]["repair"] = int((time.time() - start_time) * 1000)
            return state

        # Convert dict issues back to ValidationIssue objects
        validation_issues = []
        for issue_dict in issues:
            validation_issues.append(ValidationIssue(
                agent=issue_dict["agent"],
                location=issue_dict["location"],
                issue=issue_dict["issue"],
                suggestion=issue_dict["suggestion"],
                severity=issue_dict.get("severity", "warning")
            ))

        # STEP 1: Fix Resource Quality issues (regeneration) - NOW WITH CONTEXT
        resource_issues = [i for i in validation_issues if i.agent == "Resource Quality"]
        if resource_issues:
            logger.info(f"[NODE] Fixing {len(resource_issues)} resource issues (regeneration with context)")
            current_json = await fix_resource_quality(
                current_json,
                state["scenario_prompt"],
                resource_issues,
                entity_map=state.get("entity_map", {}),
                domain_profile=state.get("domain_profile", {})
            )

        # STEP 2: Fix Context Fidelity issues (specialized)
        context_issues = [i for i in validation_issues if i.agent == "Context Fidelity"]
        if context_issues:
            logger.info(f"[NODE] Fixing {len(context_issues)} context fidelity issues")
            current_json = await fix_context_fidelity(
                current_json,
                state["scenario_prompt"],
                context_issues
            )

        # STEP 3: Fix Completeness issues (specialized)
        completeness_issues = [i for i in validation_issues if i.agent == "Completeness"]
        if completeness_issues:
            logger.info(f"[NODE] Fixing {len(completeness_issues)} completeness issues")
            current_json = await fix_completeness(
                current_json,
                state["scenario_prompt"],
                completeness_issues
            )

        # STEP 4: Fix other issues (patching)
        other_issues = [i for i in validation_issues if i.agent not in ("Resource Quality", "Context Fidelity", "Completeness")]
        if other_issues:
            logger.info(f"[NODE] Fixing {len(other_issues)} other issues (patching)")

            mini_report = ValidationReport(
                overall_score=state["validation_score"],
                passed=False,
                agent_results=[
                    AgentResult(
                        agent_name="Mixed",
                        score=0.0,
                        passed=False,
                        issues=other_issues
                    )
                ],
                total_issues=len(other_issues),
                needs_repair=True
            )
            current_json = await repair_issues(
                current_json,
                state["scenario_prompt"],
                mini_report
            )

        state["adapted_json"] = current_json

        # Track repair history
        state["repair_history"].append({
            "iteration": iteration,
            "issues_count": len(issues),
            "previous_score": state["validation_score"]
        })

        logger.info(f"[NODE] Repair iteration {iteration} complete")

    except Exception as e:
        logger.error(f"[NODE] Repair failed: {e}")
        state["errors"].append(f"Repair error (iter {iteration}): {str(e)}")

    state["stage_timings"]["repair"] = int((time.time() - start_time) * 1000)
    return state


# =============================================================================
# NODE: FINALIZE
# =============================================================================

@traceable(name="node_finalize", run_type="chain")
async def node_finalize(state: PipelineState) -> PipelineState:
    """Finalize node - set final output and status."""

    logger.info("[NODE] ========== FINALIZE ==========")

    state["final_json"] = state.get("adapted_json") or state["input_json"]
    state["final_score"] = state.get("validation_score", 0.0)

    if state["final_score"] >= PASS_THRESHOLD:
        state["status"] = "success"
        state["validation_passed"] = True
        logger.info(f"[NODE] SUCCESS: {state['final_score']:.2%}")
    elif state["final_score"] >= 0.80:
        state["status"] = "partial"
        state["validation_passed"] = False
        logger.info(f"[NODE] PARTIAL: {state['final_score']:.2%} (below {PASS_THRESHOLD:.0%})")
    else:
        state["status"] = "failed"
        state["validation_passed"] = False
        logger.info(f"[NODE] FAILED: {state['final_score']:.2%}")

    # Calculate total runtime
    state["total_runtime_ms"] = sum(state["stage_timings"].values())

    return state


# =============================================================================
# ROUTING FUNCTIONS
# =============================================================================

def should_repair(state: PipelineState) -> Literal["repair", "finalize"]:
    """Decide whether to repair or finalize."""

    score = state.get("validation_score", 0.0)
    iteration = state.get("repair_iteration", 0)

    # Check if passed
    if score >= PASS_THRESHOLD:
        logger.info(f"[ROUTE] Score {score:.2%} >= {PASS_THRESHOLD:.0%} -> finalize")
        return "finalize"

    # Check max iterations (reduced to 2 to prevent over-repair)
    if iteration >= MAX_REPAIR_ITERATIONS:
        logger.info(f"[ROUTE] Max iterations ({MAX_REPAIR_ITERATIONS}) -> finalize")
        return "finalize"

    # Check if making progress - compare CURRENT score with LAST recorded score
    # This catches actual regression (when repairs make things worse)
    history = state.get("repair_history", [])
    if len(history) >= 1:
        # Last entry has the score BEFORE that repair iteration
        last_pre_repair_score = history[-1].get("previous_score", 0)
        # If current score is less than or equal to what we had before repairs started
        if score <= last_pre_repair_score * 0.95:  # Allow 5% tolerance
            logger.info(f"[ROUTE] Regression detected ({last_pre_repair_score:.2%} -> {score:.2%}) -> finalize")
            return "finalize"

    # Continue to repair - try to reach 95% target
    logger.info(f"[ROUTE] Score {score:.2%} < {PASS_THRESHOLD:.0%} -> repair (iter {iteration + 1}/{MAX_REPAIR_ITERATIONS})")
    return "repair"


# =============================================================================
# BUILD GRAPH
# =============================================================================

def build_pipeline() -> StateGraph:
    """Build the LangGraph pipeline."""

    graph = StateGraph(PipelineState)

    # Add nodes
    graph.add_node("adapt", node_adapt)
    graph.add_node("validate", node_validate)
    graph.add_node("repair", node_repair)
    graph.add_node("finalize", node_finalize)

    # Set entry point
    graph.set_entry_point("adapt")

    # Add edges
    graph.add_edge("adapt", "validate")

    # Conditional routing after validate
    graph.add_conditional_edges(
        "validate",
        should_repair,
        {
            "repair": "repair",
            "finalize": "finalize"
        }
    )

    # After repair, go back to validate
    graph.add_edge("repair", "validate")

    # Finalize goes to END
    graph.add_edge("finalize", END)

    return graph.compile()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

@dataclass
class PipelineResult:
    """Result from running the pipeline."""
    final_json: dict
    final_score: float
    status: str  # "success" | "partial" | "failed"
    adaptation_time_ms: int
    shards_processed: int
    repair_iterations: int
    agent_scores: dict
    issues_remaining: int
    total_runtime_ms: int
    errors: list


@traceable(name="run_pipeline", run_type="chain")
async def run_pipeline(
    input_json: dict,
    scenario_prompt: str
) -> PipelineResult:
    """
    Run the full adaptation pipeline.

    Args:
        input_json: The simulation JSON to adapt
        scenario_prompt: Description of the target scenario

    Returns:
        PipelineResult with final JSON and metrics
    """
    from .state import create_initial_state

    logger.info("[PIPELINE] ========================================")
    logger.info("[PIPELINE] STARTING ADAPTATION PIPELINE")
    logger.info("[PIPELINE] ========================================")
    logger.info(f"[PIPELINE] Scenario: {scenario_prompt[:100]}...")
    logger.info(f"[PIPELINE] Input size: {len(json.dumps(input_json)):,} chars")

    # Build and run the graph
    pipeline = build_pipeline()

    # Initial state
    initial_state = create_initial_state(
        input_json=input_json,
        scenario_prompt=scenario_prompt
    )

    # Run the pipeline
    final_state = await pipeline.ainvoke(initial_state)

    # Build result
    result = PipelineResult(
        final_json=final_state.get("final_json", {}),
        final_score=final_state.get("final_score", 0.0),
        status=final_state.get("status", "unknown"),
        adaptation_time_ms=final_state.get("adaptation_time_ms", 0),
        shards_processed=final_state.get("shards_processed", 0),
        repair_iterations=final_state.get("repair_iteration", 0),
        agent_scores=final_state.get("agent_scores", {}),
        issues_remaining=len(final_state.get("validation_issues", [])),
        total_runtime_ms=final_state.get("total_runtime_ms", 0),
        errors=final_state.get("errors", [])
    )

    logger.info("[PIPELINE] ========================================")
    logger.info(f"[PIPELINE] COMPLETE: {result.status.upper()}")
    logger.info(f"[PIPELINE] Final Score: {result.final_score:.2%}")
    logger.info(f"[PIPELINE] Total Time: {result.total_runtime_ms}ms")
    logger.info(f"[PIPELINE] Repair Iterations: {result.repair_iterations}")
    logger.info("[PIPELINE] ========================================")

    return result


# =============================================================================
# STREAMING VERSION
# =============================================================================

async def run_pipeline_streaming(
    input_json: dict,
    scenario_prompt: str
):
    """
    Run pipeline with streaming state updates.
    Yields state after each node completes.
    """
    from .state import create_initial_state

    pipeline = build_pipeline()
    initial_state = create_initial_state(
        input_json=input_json,
        scenario_prompt=scenario_prompt
    )

    async for state in pipeline.astream(initial_state):
        yield state
