"""FastAPI routes for scenario re-contextualization API."""
import json
import asyncio
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from typing import Optional
from src.models.schemas import (
    TransformRequest,
    TransformResponse,
    ValidationReport,
    ValidateOnlyRequest,
    HealthResponse
)
from src.utils.openai_client import openai_client
from src.utils.config import config
from src.utils.helpers import compute_sha256, generate_json_diff

# Import sharder
from src.stages.sharder import Sharder, get_shard_summary

# Import validation report agent
from src.stages.validation_report_agent import (
    generate_validation_report_markdown,
    generate_validation_report_json,
    aggregate_multi_run_results,
)

router = APIRouter()


# ============================================================================
# HEALTH CHECK
# ============================================================================

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API and OpenAI connectivity."""
    try:
        openai_connected = await openai_client.test_connection()
        return HealthResponse(
            status="healthy",
            version=config.APP_VERSION,
            openai_connected=openai_connected
        )
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            version=config.APP_VERSION,
            openai_connected=False
        )


# ============================================================================
# SHARD ENDPOINTS
# ============================================================================

class ShardRequest(BaseModel):
    """Request for sharding endpoint."""
    input_json: dict = Field(..., description="The simulation JSON to shard")
    scenario_prompt: Optional[str] = Field("", description="Target scenario prompt")


@router.post("/shard")
async def shard_json_endpoint(request: ShardRequest):
    """Split simulation JSON into shards."""
    try:
        sharder = Sharder()
        collection = sharder.shard(request.input_json, request.scenario_prompt)
        summary = get_shard_summary(collection)

        return {
            "status": "success",
            "summary": summary,
            "shard_details": [
                {
                    "id": shard.id,
                    "name": shard.name,
                    "lock_state": shard.lock_state.value,
                    "is_blocker": shard.is_blocker,
                    "paths": shard.paths,
                }
                for shard in collection.shards
            ]
        }
    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")


@router.post("/shard/{shard_id}")
async def get_shard_content(shard_id: str, request: ShardRequest):
    """Get content of a specific shard."""
    try:
        sharder = Sharder()
        collection = sharder.shard(request.input_json, request.scenario_prompt)
        shard = collection.get_shard(shard_id)

        if not shard:
            raise HTTPException(status_code=404, detail=f"Shard '{shard_id}' not found")

        return {
            "shard_id": shard.id,
            "name": shard.name,
            "lock_state": shard.lock_state.value,
            "content": shard.content
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# SIMPLE ADAPTER - Main adaptation endpoint
# ============================================================================

class SimpleAdaptRequest(BaseModel):
    """Request for simple adaptation endpoint."""
    input_json: dict = Field(..., description="The simulation JSON to adapt")
    scenario_prompt: str = Field(..., description="Target scenario - SINGLE SOURCE OF TRUTH")


@router.post("/adapt/simple")
async def adapt_simple_endpoint(request: SimpleAdaptRequest):
    """
    Adapt simulation JSON to a new scenario using parallel shards.

    The scenario_prompt is the SINGLE SOURCE OF TRUTH - all shards
    derive company name, KLOs, and terminology from it.
    """
    try:
        from src.stages.simple_adapter import adapt_simple

        result = await adapt_simple(
            input_json=request.input_json,
            scenario_prompt=request.scenario_prompt,
        )

        return {
            "status": "success",
            "mode": result.mode,
            "timing": {"time_ms": result.time_ms},
            "size": {
                "input_chars": result.input_chars,
                "output_chars": result.output_chars,
            },
            "shards_processed": result.shards_processed,
            "errors": result.errors,
            "adapted_json": result.adapted_json,
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")


@router.websocket("/adapt/simple/ws")
async def adapt_simple_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for adaptation with real-time progress streaming.

    Send: {"input_json": {...}, "scenario_prompt": "..."}
    Receive: progress messages and final result
    """
    await websocket.accept()

    try:
        await websocket.send_json({
            "type": "connected",
            "message": "WebSocket connected. Send {input_json, scenario_prompt} to start."
        })

        data = await websocket.receive_json()
        input_json = data.get("input_json")
        scenario_prompt = data.get("scenario_prompt")

        if not input_json or not scenario_prompt:
            await websocket.send_json({
                "type": "error",
                "message": "Missing required fields: input_json and scenario_prompt"
            })
            await websocket.close()
            return

        from src.stages.simple_adapter import adapt_simple_with_progress

        async def progress_callback(event_type: str, data: dict):
            try:
                await websocket.send_json({"type": event_type, **data})
            except Exception:
                pass

        await websocket.send_json({
            "type": "progress",
            "stage": "starting",
            "message": "Starting adaptation",
        })

        result = await adapt_simple_with_progress(
            input_json=input_json,
            scenario_prompt=scenario_prompt,
            progress_callback=progress_callback
        )

        await websocket.send_json({
            "type": "result",
            "status": "success",
            "data": {
                "mode": result.mode,
                "time_ms": result.time_ms,
                "shards_processed": result.shards_processed,
                "errors": result.errors,
                "adapted_json": result.adapted_json,
            }
        })

    except WebSocketDisconnect:
        pass
    except json.JSONDecodeError as e:
        await websocket.send_json({"type": "error", "message": f"Invalid JSON: {str(e)}"})
    except Exception as e:
        import traceback
        await websocket.send_json({
            "type": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        })
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


# ============================================================================
# FULL PIPELINE - Adapt + Validate + Repair
# ============================================================================

@router.websocket("/pipeline/ws")
async def pipeline_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for FULL PIPELINE (adapt + validate + repair).

    Pipeline stages:
    1. ADAPT - Gemini transforms JSON to target scenario
    2. VALIDATE - 8 GPT validators check quality
    3. REPAIR - Fix issues (loops back if needed, max 2 iterations)
    4. FINALIZE - Return final result

    Send: {"input_json": {...}, "scenario_prompt": "..."}
    """
    await websocket.accept()

    try:
        await websocket.send_json({
            "type": "connected",
            "message": "WebSocket connected. Send {input_json, scenario_prompt} to start full pipeline."
        })

        data = await websocket.receive_json()
        input_json = data.get("input_json")
        scenario_prompt = data.get("scenario_prompt")

        if not input_json or not scenario_prompt:
            await websocket.send_json({
                "type": "error",
                "message": "Missing required fields: input_json and scenario_prompt"
            })
            await websocket.close()
            return

        from src.graph.workflow import run_pipeline_streaming
        from src.graph.state import create_initial_state

        await websocket.send_json({
            "type": "stage_start",
            "stage": "pipeline",
            "message": "Starting full pipeline (adapt → validate → repair → finalize)",
        })

        last_stage = None
        final_state = None

        async for state in run_pipeline_streaming(input_json, scenario_prompt):
            for node_name, node_state in state.items():
                if isinstance(node_state, dict):
                    final_state = node_state

                    if "adapted_json" in node_state and last_stage != "adapt":
                        await websocket.send_json({
                            "type": "stage_complete",
                            "stage": "adapt",
                            "data": {
                                "shards_processed": node_state.get("shards_processed", 0),
                                "time_ms": node_state.get("adaptation_time_ms", 0),
                            }
                        })
                        last_stage = "adapt"

                    if "validation_score" in node_state:
                        score = node_state.get("validation_score", 0)
                        issues = len(node_state.get("validation_issues", []))
                        iteration = node_state.get("repair_iteration", 0)

                        if last_stage != f"validate_{iteration}":
                            await websocket.send_json({
                                "type": "validation_result",
                                "stage": "validate",
                                "iteration": iteration,
                                "score": score,
                                "issues": issues,
                                "passed": score >= 0.95,
                                "agent_scores": node_state.get("agent_scores", {})
                            })
                            last_stage = f"validate_{iteration}"

        if final_state:
            await websocket.send_json({
                "type": "result",
                "status": final_state.get("status", "unknown"),
                "data": {
                    "final_score": final_state.get("final_score", 0),
                    "validation_passed": final_state.get("validation_passed", False),
                    "repair_iterations": final_state.get("repair_iteration", 0),
                    "agent_scores": final_state.get("agent_scores", {}),
                    "final_json": final_state.get("final_json", {}),
                }
            })

    except WebSocketDisconnect:
        pass
    except json.JSONDecodeError as e:
        await websocket.send_json({"type": "error", "message": f"Invalid JSON: {str(e)}"})
    except Exception as e:
        import traceback
        await websocket.send_json({
            "type": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        })
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


# ============================================================================
# VALIDATION ENDPOINTS
# ============================================================================

@router.post("/validate")
async def validate_transformation(request: ValidateOnlyRequest):
    """Validate an already-transformed JSON."""
    try:
        original = request.original_json
        transformed = request.transformed_json
        locked_fields = request.locked_fields or config.LOCKED_FIELDS

        locked_field_hashes = {}
        validation_errors = []

        original_data = original.get("topicWizardData", {})
        transformed_data = transformed.get("topicWizardData", {})

        for field in locked_fields:
            if field in original_data:
                original_hash = compute_sha256(original_data[field])
                locked_field_hashes[field] = original_hash

                if field in transformed_data:
                    transformed_hash = compute_sha256(transformed_data[field])
                    if original_hash != transformed_hash:
                        validation_errors.append({
                            "field": field,
                            "error": "Locked field was modified"
                        })

        changed_paths = generate_json_diff(original, transformed)
        locked_pass = len(validation_errors) == 0

        return ValidationReport(
            schema_pass=True,
            locked_fields_compliance=locked_pass,
            locked_field_hashes=locked_field_hashes,
            changed_paths=changed_paths,
            scenario_consistency_score=1.0 if locked_pass else 0.0,
            old_scenario_keywords_found=[],
            runtime_ms=0,
            retries=0,
            openai_stats={},
            final_status="OK" if locked_pass else "FAIL"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# VALIDATION REPORT ENDPOINTS
# ============================================================================

class ValidationReportRequest(BaseModel):
    """Request for validation report generation."""
    input_json: dict = Field(..., description="The simulation JSON to validate")
    scenario_prompt: str = Field(..., description="Target scenario prompt")
    original_scenario: Optional[str] = Field("", description="Original scenario")
    simulation_purpose: Optional[str] = Field("Business Simulation Training")
    output_format: Optional[str] = Field("markdown", description="markdown or json")


class MultiRunReportRequest(BaseModel):
    """Request for multi-run aggregated report."""
    run_results: list[dict] = Field(..., description="List of run results")
    original_scenario: str = Field(..., description="Original scenario")
    target_scenario: str = Field(..., description="Target scenario")
    acceptance_threshold: float = Field(0.95)


class ValidationReportFromResultsRequest(BaseModel):
    """Request for generating report from validation results."""
    validation_results: dict = Field(..., description="Validation results dict")
    original_scenario: str = Field("")
    target_scenario: str = Field(...)
    simulation_purpose: str = Field("Business Simulation Training")
    output_format: str = Field("markdown")


@router.post("/validation/report")
async def generate_validation_report_endpoint(request: ValidationReportRequest):
    """Generate a canonical validation report for a simulation."""
    try:
        from src.stages.simple_validators import run_all_validators

        validation_report = await run_all_validators(
            adapted_json=request.input_json,
            scenario_prompt=request.scenario_prompt,
        )

        original_scenario = request.original_scenario
        if not original_scenario:
            topic_data = request.input_json.get("topicWizardData", {})
            selected = topic_data.get("selectedScenarioOption", "")
            if isinstance(selected, dict):
                original_scenario = selected.get("option", "Unknown")
            elif isinstance(selected, str):
                original_scenario = selected
            else:
                original_scenario = "Unknown"

        if request.output_format == "json":
            report_data = await generate_validation_report_json(
                agent_results=validation_report.agent_results,
                original_scenario=original_scenario,
                target_scenario=request.scenario_prompt,
                simulation_purpose=request.simulation_purpose,
            )
            return {"status": "success", "format": "json", "report": report_data}
        else:
            report_md = await generate_validation_report_markdown(
                agent_results=validation_report.agent_results,
                original_scenario=original_scenario,
                target_scenario=request.scenario_prompt,
                simulation_purpose=request.simulation_purpose,
            )
            return {"status": "success", "format": "markdown", "report": report_md}

    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")


@router.post("/validation/report/multi-run")
async def generate_multi_run_report_endpoint(request: MultiRunReportRequest):
    """Generate aggregated validation report across multiple runs."""
    try:
        aggregated = await aggregate_multi_run_results(
            run_results=request.run_results,
            original_scenario=request.original_scenario,
            target_scenario=request.target_scenario,
            acceptance_threshold=request.acceptance_threshold,
        )
        return {"status": "success", "aggregated_report": aggregated}

    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")


@router.post("/validation/report/from-results")
async def generate_report_from_validation_results(request: ValidationReportFromResultsRequest):
    """Generate validation report from pre-computed results."""
    try:
        from dataclasses import dataclass, field as dataclass_field

        @dataclass
        class MockIssue:
            agent: str
            location: str
            issue: str
            suggestion: str
            severity: str = "warning"

        @dataclass
        class MockAgentResult:
            agent_name: str
            score: float
            passed: bool
            issues: list = dataclass_field(default_factory=list)
            details: dict = dataclass_field(default_factory=dict)

        agent_results = []
        for ar_dict in request.validation_results.get("agent_results", []):
            issues = []
            for issue_dict in ar_dict.get("issues", []):
                issues.append(MockIssue(
                    agent=issue_dict.get("agent", ""),
                    location=issue_dict.get("location", ""),
                    issue=issue_dict.get("issue", ""),
                    suggestion=issue_dict.get("suggestion", ""),
                    severity=issue_dict.get("severity", "warning"),
                ))
            agent_results.append(MockAgentResult(
                agent_name=ar_dict.get("agent_name", ar_dict.get("name", "")),
                score=ar_dict.get("score", 0.0),
                passed=ar_dict.get("passed", False),
                issues=issues,
            ))

        if request.output_format == "json":
            report_data = await generate_validation_report_json(
                agent_results=agent_results,
                original_scenario=request.original_scenario,
                target_scenario=request.target_scenario,
                simulation_purpose=request.simulation_purpose,
            )
            return {"status": "success", "format": "json", "report": report_data}
        else:
            report_md = await generate_validation_report_markdown(
                agent_results=agent_results,
                original_scenario=request.original_scenario,
                target_scenario=request.target_scenario,
                simulation_purpose=request.simulation_purpose,
            )
            return {"status": "success", "format": "markdown", "report": report_md}

    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")


# ============================================================================
# UTILITY ENDPOINTS
# ============================================================================

@router.get("/scenarios")
async def list_scenarios(input_json: str = None):
    """List available scenarios from input JSON."""
    try:
        if input_json:
            data = json.loads(input_json)
            scenario_options = data.get("topicWizardData", {}).get("scenarioOptions", [])
            current_scenario = data.get("topicWizardData", {}).get("selectedScenarioOption", "")

            return {
                "total": len(scenario_options),
                "current_scenario": current_scenario,
                "scenarios": [
                    {"index": idx, "text": scenario, "is_current": scenario == current_scenario}
                    for idx, scenario in enumerate(scenario_options)
                ]
            }

        return {"message": "Provide input_json parameter to see available scenarios"}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/status/langsmith")
async def langsmith_status():
    """Check LangSmith observability configuration."""
    from src.utils.gemini_client import get_langsmith_status

    status = get_langsmith_status()

    return {
        "langsmith": status,
        "instructions": {
            "step_1": "Get API key from https://smith.langchain.com",
            "step_2": "Set LANGCHAIN_API_KEY in .env",
            "step_3": "Ensure LANGCHAIN_TRACING_V2=true",
            "step_4": "Restart server",
        },
    }
