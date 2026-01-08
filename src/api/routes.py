"""FastAPI routes for scenario re-contextualization API."""
import json
import asyncio
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional
from src.models.schemas import (
    TransformRequest,
    TransformResponse,
    ValidationReport,
    ValidateOnlyRequest,
    HealthResponse
)
from src.graph.workflow import scenario_workflow
from src.utils.openai_client import openai_client
from src.utils.config import config
from src.utils.helpers import compute_sha256, generate_json_diff, search_keywords

# Import sharder
from src.stages.sharder import Sharder, get_shard_summary, merge_shards

router = APIRouter()


# ============================================================================
# SHARD ENDPOINTS - For testing the new sharder
# ============================================================================

class ShardRequest(BaseModel):
    """Request for sharding endpoint."""
    input_json: dict = Field(..., description="The simulation JSON to shard")
    scenario_prompt: Optional[str] = Field("", description="Target scenario prompt")


@router.post("/shard")
async def shard_json_endpoint(request: ShardRequest):
    """
    Split simulation JSON into shards.

    Returns summary of all shards with their lock states and IDs.
    """
    try:
        # Debug: Check input structure
        input_keys = list(request.input_json.keys()) if request.input_json else []
        topic_data = request.input_json.get("topicWizardData", {})
        topic_keys = list(topic_data.keys()) if topic_data else []

        sharder = Sharder()
        collection = sharder.shard(request.input_json, request.scenario_prompt)
        summary = get_shard_summary(collection)

        return {
            "status": "success",
            "debug": {
                "input_keys": input_keys,
                "topicWizardData_keys": topic_keys[:10],  # First 10 keys
                "has_assessmentCriterion": "assessmentCriterion" in topic_data,
                "has_simulationFlow": "simulationFlow" in topic_data,
                "simulationFlow_count": len(topic_data.get("simulationFlow", [])),
            },
            "summary": summary,
            "shard_details": [
                {
                    "id": shard.id,
                    "name": shard.name,
                    "lock_state": shard.lock_state.value,
                    "is_blocker": shard.is_blocker,
                    "paths": shard.paths,
                    "ids_count": len(shard.extracted_ids),
                    "sample_ids": shard.extracted_ids[:5],
                    "aligns_with": shard.aligns_with,
                    "hash": shard.current_hash[:16] + "...",
                    "content_keys": list(shard.content.keys())[:5] if shard.content else []
                }
                for shard in collection.shards
            ]
        }
    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")


@router.post("/shard/{shard_id}")
async def get_shard_content(shard_id: str, request: ShardRequest):
    """
    Get content of a specific shard.

    Args:
        shard_id: ID of the shard (e.g., "workplace_scenario", "emails")
    """
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
            "is_blocker": shard.is_blocker,
            "paths": shard.paths,
            "extracted_ids": shard.extracted_ids,
            "content": shard.content
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# RAG ENDPOINTS - For testing the RAG system
# ============================================================================

class IndexRequest(BaseModel):
    """Request for indexing a simulation."""
    input_json: dict = Field(..., description="The simulation JSON to index")
    simulation_id: str = Field(..., description="Unique identifier for this simulation")
    clear_existing: bool = Field(False, description="Clear existing index for this simulation")


class QueryRequest(BaseModel):
    """Request for querying the RAG system."""
    query: str = Field(..., description="Search query")
    simulation_id: Optional[str] = Field(None, description="Filter to specific simulation")
    n_results: int = Field(5, description="Number of results to return")


@router.post("/rag/index")
async def index_simulation_endpoint(request: IndexRequest):
    """
    Index a simulation into the RAG system.

    Shards the JSON and stores embeddings in ChromaDB.
    """
    try:
        from src.rag.retriever import SimulationRetriever
        from src.stages.sharder import Sharder

        # First shard the simulation
        sharder = Sharder()
        collection = sharder.shard(request.input_json)

        # Then index the shards
        retriever = SimulationRetriever()
        result = retriever.index_simulation(
            simulation_id=request.simulation_id,
            shards=collection.shards,
            clear_existing=request.clear_existing,
        )

        return {
            "status": "success",
            "message": f"Indexed {result['indexed']} shards",
            "simulation_id": request.simulation_id,
            "shards_indexed": result.get("shards", []),
        }
    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")


@router.post("/rag/query")
async def query_rag_endpoint(request: QueryRequest):
    """
    Query the RAG system for relevant context.

    Returns similar chunks from indexed simulations.
    """
    try:
        from src.rag.retriever import SimulationRetriever

        retriever = SimulationRetriever()
        contexts = retriever.retrieve_context(
            query=request.query,
            simulation_id=request.simulation_id,
            n_results=request.n_results,
        )

        return {
            "status": "success",
            "query": request.query,
            "results": [
                {
                    "shard_id": ctx.shard_id,
                    "score": ctx.score,
                    "document": ctx.document[:500] + "..." if len(ctx.document) > 500 else ctx.document,
                    "metadata": ctx.metadata,
                }
                for ctx in contexts
            ]
        }
    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")


@router.get("/rag/collections")
async def list_rag_collections():
    """
    List all RAG collections and their document counts.
    """
    try:
        from src.rag.vector_store import get_vector_store

        store = get_vector_store()
        collections = store.list_collections()

        return {
            "status": "success",
            "collections": collections,
        }
    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")


@router.delete("/rag/clear/{collection_name}")
async def clear_rag_collection(collection_name: str):
    """
    Clear a RAG collection.
    """
    try:
        from src.rag.vector_store import get_vector_store

        store = get_vector_store()
        store.clear_collection(collection_name)

        return {
            "status": "success",
            "message": f"Cleared collection: {collection_name}",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# ADAPTATION ENGINE - Stage 1: Parallel shard transformation with Gemini
# ============================================================================

class AdaptRequest(BaseModel):
    """Request for adaptation endpoint."""
    input_json: dict = Field(..., description="The BASE simulation JSON to adapt")
    # Option A: Select from existing scenarios in JSON
    target_scenario_index: Optional[int] = Field(None, description="Target scenario index (select from scenarioOptions in JSON)")
    # Option B: Free-form prompt
    scenario_prompt: Optional[str] = Field(None, description="FREE-FORM scenario prompt (e.g. 'Gen Z organic T-shirts brand...')")
    use_rag: bool = Field(True, description="Use RAG for additional context")


@router.post("/adapt")
async def adapt_simulation_endpoint(request: AdaptRequest):
    """
    Adapt simulation to a new scenario using PARALLEL shard processing.

    TWO INPUT OPTIONS:
    - Option A: target_scenario_index → Select from existing scenarioOptions in JSON
    - Option B: scenario_prompt → Free-form text prompt (any custom scenario)

    This is Stage 1 of the pipeline:
    1. Shards the JSON into 13 pieces
    2. Filters to unlocked shards only (8 shards)
    3. Adapts ALL unlocked shards IN PARALLEL using Gemini 2.5 Flash
    4. Merges back with locked shards

    NO hardcoded mappings - LLM infers everything dynamically.
    """
    try:
        from src.stages.adaptation_engine import AdaptationEngine
        from src.rag.retriever import SimulationRetriever

        # Validate input - must have either scenario_prompt OR target_scenario_index
        if request.scenario_prompt is None and request.target_scenario_index is None:
            raise ValueError("Provide either 'scenario_prompt' (free-form text) OR 'target_scenario_index' (select from JSON options)")

        # Setup RAG if requested
        rag_retriever = None
        if request.use_rag:
            try:
                rag_retriever = SimulationRetriever()
            except Exception:
                pass  # RAG is optional

        # Run parallel adaptation
        engine = AdaptationEngine(rag_retriever=rag_retriever)
        result = await engine.adapt(
            input_json=request.input_json,
            target_scenario_index=request.target_scenario_index,
            scenario_prompt=request.scenario_prompt,
        )

        return {
            "status": "success",
            "mode": "free_prompt" if request.scenario_prompt else "scenario_index",
            "timing": {
                "total_time_ms": result.total_time_ms,
                "parallel_time_ms": result.parallel_time_ms,
            },
            "shards": {
                "adapted": result.shards_adapted,
                "locked": result.shards_locked,
            },
            "global_factsheet": {
                "company": result.global_factsheet.get("company", {}),
                "poison_list_count": len(result.global_factsheet.get("poison_list", [])),
            },
            "source_scenario": result.source_scenario[:100] + "..." if len(result.source_scenario) > 100 else result.source_scenario,
            "target_scenario": result.target_scenario[:100] + "..." if len(result.target_scenario) > 100 else result.target_scenario,
            "entity_mappings": result.entity_map,
            "llm_stats": result.stats,
            "adapted_json": result.adapted_json,
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")


# ============================================================================
# ALIGNMENT CHECKER - Stage 3: LLM-based cross-shard consistency validation
# ============================================================================

class AlignmentRequest(BaseModel):
    """Request for alignment check endpoint."""
    adapted_json: dict = Field(..., description="The ADAPTED simulation JSON to validate")
    global_factsheet: dict = Field(default_factory=dict, description="Factsheet from adaptation (for consistency checks)")
    source_scenario: str = Field("", description="Original scenario text (for poison term detection)")
    threshold: float = Field(0.98, description="Minimum alignment score required (0.0-1.0)")


@router.post("/align/check")
async def check_alignment_endpoint(request: AlignmentRequest):
    """
    Stage 3: LLM-based alignment validation.

    Checks:
    1. Reporting Manager Consistency - same person everywhere
    2. Company Name Consistency - same org name everywhere
    3. Poison Term Avoidance - no old scenario terms leaked
    4. KLO ↔ Task Alignment - learning outcomes match activities
    5. Scenario Coherence - emails match scenario context

    Uses Gemini for intelligent semantic checking (not hardcoded rules).
    """
    try:
        from src.stages.alignment_checker import check_alignment

        report = await check_alignment(
            adapted_json=request.adapted_json,
            global_factsheet=request.global_factsheet,
            source_scenario=request.source_scenario,
            threshold=request.threshold,
        )

        return {
            "status": "success",
            "alignment": report.to_dict(),
        }

    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")


@router.post("/adapt-and-check")
async def adapt_and_check_endpoint(request: AdaptRequest):
    """
    Full pipeline: Adapt + Alignment Check in one call.

    Runs Stage 2 (Adaptation) → Stage 3 (Alignment Check) sequentially.
    """
    try:
        from src.stages.adaptation_engine import AdaptationEngine
        from src.stages.alignment_checker import check_alignment
        from src.rag.retriever import SimulationRetriever

        # Validate input
        if request.scenario_prompt is None and request.target_scenario_index is None:
            raise ValueError("Provide either 'scenario_prompt' or 'target_scenario_index'")

        # Setup RAG
        rag_retriever = None
        if request.use_rag:
            try:
                rag_retriever = SimulationRetriever()
            except Exception:
                pass

        # Stage 2: Adaptation
        engine = AdaptationEngine(rag_retriever=rag_retriever)
        adapt_result = await engine.adapt(
            input_json=request.input_json,
            target_scenario_index=request.target_scenario_index,
            scenario_prompt=request.scenario_prompt,
        )

        # Stage 3: Alignment Check
        alignment_report = await check_alignment(
            adapted_json=adapt_result.adapted_json,
            global_factsheet=adapt_result.global_factsheet,
            source_scenario=adapt_result.source_scenario,
            threshold=0.98,
        )

        return {
            "status": "success",
            "mode": "free_prompt" if request.scenario_prompt else "scenario_index",
            "adaptation": {
                "timing": {
                    "total_time_ms": adapt_result.total_time_ms,
                    "parallel_time_ms": adapt_result.parallel_time_ms,
                },
                "shards": {
                    "adapted": adapt_result.shards_adapted,
                    "locked": adapt_result.shards_locked,
                },
                "source_scenario": adapt_result.source_scenario[:100] + "...",
                "target_scenario": adapt_result.target_scenario[:100] + "...",
            },
            "alignment": alignment_report.to_dict(),
            "adapted_json": adapt_result.adapted_json,
            "global_factsheet": adapt_result.global_factsheet,
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")


@router.get("/adapt/scenarios")
async def list_available_scenarios(request: ShardRequest = None):
    """
    List all available scenario options from a simulation JSON.

    POST your JSON to see all 37 scenarios you can adapt to.
    """
    return {
        "message": "POST to /api/v1/adapt/scenarios with input_json to see available scenarios",
        "example": {
            "input_json": {"topicWizardData": {"scenarioOptions": ["..."]}}
        }
    }


@router.post("/adapt/scenarios")
async def list_scenarios_from_json(request: ShardRequest):
    """
    List all scenario options from the provided JSON.
    """
    try:
        topic_data = request.input_json.get("topicWizardData", {})
        scenarios_raw = topic_data.get("scenarioOptions", [])

        # Handle both formats: list of strings OR list of dicts
        if scenarios_raw and isinstance(scenarios_raw[0], dict):
            scenarios = [s.get("option", str(s)) for s in scenarios_raw]
        else:
            scenarios = scenarios_raw

        # Handle selectedScenarioOption as dict or int
        selected = topic_data.get("selectedScenarioOption", 0)
        if isinstance(selected, dict):
            selected_text = selected.get("option", "")
            current_idx = 0
            for i, opt in enumerate(scenarios):
                if opt == selected_text:
                    current_idx = i
                    break
        else:
            current_idx = int(selected) if selected else 0

        return {
            "total_scenarios": len(scenarios),
            "current_scenario_index": current_idx,
            "current_scenario": scenarios[current_idx] if current_idx < len(scenarios) else None,
            "scenarios": [
                {
                    "index": i,
                    "text": s[:150] + "..." if len(s) > 150 else s,
                    "is_current": i == current_idx,
                }
                for i, s in enumerate(scenarios)
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/transform/stream-openai")
async def transform_scenario_stream_openai(request: TransformRequest):
    """
    Transform JSON with real-time OpenAI streaming output.
    
    Shows the actual JSON being generated token-by-token from OpenAI.
    """
    async def event_generator():
        """Generate SSE events including OpenAI streaming chunks."""
        try:
            from src.graph.nodes import (
                ingestor_node, analyzer_node, transformer_node_streaming,
                consistency_checker_node, validator_node, finalizer_node
            )
            import time as time_module
            
            # Send start event
            yield f"data: {json.dumps({'event': 'start', 'message': 'Starting transformation'})}\n\n"
            await asyncio.sleep(0.05)
            
            # Prepare initial state
            state = {
                "input_json": request.input_json,
                "selected_scenario": request.selected_scenario,
                "node_logs": [],
                "validation_errors": [],
                "retry_count": 0,
                "final_status": "PENDING"
            }
            
            # Run nodes in thread pool (except transformer which we'll stream)
            import concurrent.futures
            loop = asyncio.get_event_loop()
            
            # 1. Ingestor
            yield f"data: {json.dumps({'event': 'node_start', 'node': 'IngestorNode'})}\n\n"
            with concurrent.futures.ThreadPoolExecutor() as pool:
                state = await loop.run_in_executor(pool, ingestor_node, state)
            yield f"data: {json.dumps({'event': 'node_complete', 'node': 'IngestorNode'})}\n\n"
            await asyncio.sleep(0.05)
            
            # 2. Analyzer
            yield f"data: {json.dumps({'event': 'node_start', 'node': 'AnalyzerNode'})}\n\n"
            with concurrent.futures.ThreadPoolExecutor() as pool:
                state = await loop.run_in_executor(pool, analyzer_node, state)
            yield f"data: {json.dumps({'event': 'node_complete', 'node': 'AnalyzerNode'})}\n\n"
            await asyncio.sleep(0.05)
            
            # 3. Transformer with streaming (progress only, no chunks)
            yield f"data: {json.dumps({'event': 'node_start', 'node': 'TransformerNode', 'message': 'Starting OpenAI transformation'})}\n\n"
            await asyncio.sleep(0.05)
            
            # Run transformer streaming in thread pool but don't send chunks
            updated_state = None
            
            def run_transformer():
                final_state = None
                error_msg = None
                try:
                    for chunk in transformer_node_streaming(state):
                        if isinstance(chunk, dict) and chunk.get("__complete__"):
                            final_state = chunk.get("__state__")
                            if chunk.get("__error__"):
                                error_msg = chunk["__error__"]
                        # Skip sending individual chunks - just collect final state
                except Exception as e:
                    import traceback
                    error_msg = f"{str(e)}\n{traceback.format_exc()}"
                    
                return final_state, error_msg
            
            # Execute in thread pool with heartbeat progress updates
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = loop.run_in_executor(pool, run_transformer)
                
                # Send heartbeat while waiting for OpenAI
                heartbeat = 0
                while not future.done():
                    await asyncio.sleep(2)
                    heartbeat += 2
                    yield f"data: {json.dumps({'event': 'openai_progress', 'message': f'OpenAI generating... ({heartbeat}s)'})}\n\n"
                
                # Get results
                updated_state, error_msg = await future
            
            # Check for errors
            if error_msg:
                yield f"data: {json.dumps({'event': 'error', 'message': f'TransformerNode error: {error_msg}'})}\n\n"
                return
            
            # Update state with result
            if updated_state:
                state = updated_state
            else:
                # If no state returned, something went wrong
                yield f"data: {json.dumps({'event': 'error', 'message': 'TransformerNode did not return updated state'})}\n\n"
                return
            
            yield f"data: {json.dumps({'event': 'node_complete', 'node': 'TransformerNode', 'message': 'Transformation complete'})}\n\n"
            await asyncio.sleep(0.05)
            
            # 4. Consistency Checker
            yield f"data: {json.dumps({'event': 'node_start', 'node': 'ConsistencyCheckerNode'})}\n\n"
            with concurrent.futures.ThreadPoolExecutor() as pool:
                state = await loop.run_in_executor(pool, consistency_checker_node, state)
            yield f"data: {json.dumps({'event': 'node_complete', 'node': 'ConsistencyCheckerNode'})}\n\n"
            await asyncio.sleep(0.05)
            
            # 5. Validator
            yield f"data: {json.dumps({'event': 'node_start', 'node': 'ValidatorNode'})}\n\n"
            with concurrent.futures.ThreadPoolExecutor() as pool:
                state = await loop.run_in_executor(pool, validator_node, state)
            yield f"data: {json.dumps({'event': 'node_complete', 'node': 'ValidatorNode'})}\n\n"
            await asyncio.sleep(0.05)
            
            # 6. Finalizer
            with concurrent.futures.ThreadPoolExecutor() as pool:
                state = await loop.run_in_executor(pool, finalizer_node, state)
            
            # Build response
            output_json = state.get("transformed_json", request.input_json)
            
            validation_report = ValidationReport(
                schema_pass=state.get("final_status") == "OK",
                locked_fields_compliance=len([e for e in state.get("validation_errors", []) 
                                              if e.get("field") in config.LOCKED_FIELDS]) == 0,
                locked_field_hashes=state.get("locked_field_hashes", {}),
                changed_paths=state.get("changed_paths", []),
                scenario_consistency_score=state.get("consistency_score", 0.0),
                old_scenario_keywords_found=search_keywords(
                    state.get("transformed_json", {}),
                    list(state.get("entity_map", {}).keys()),
                    exclude_paths=config.LOCKED_FIELDS
                ),
                runtime_ms=state.get("runtime_ms", 0),
                retries=state.get("retry_count", 0),
                openai_stats=state.get("openai_stats", {}),
                final_status=state.get("final_status", "FAIL")
            )
            
            response = TransformResponse(
                output_json=output_json,
                validation_report=validation_report,
                execution_time_ms=state.get("runtime_ms", 0)
            )
            
            # Send final result
            yield f"data: {json.dumps({'event': 'complete', 'result': response.model_dump()})}\n\n"
            
        except Exception as e:
            import traceback
            yield f"data: {json.dumps({'event': 'error', 'message': str(e), 'traceback': traceback.format_exc()})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )


@router.post("/validate")
async def validate_transformation(request: ValidateOnlyRequest):
    """
    Validate an already-transformed JSON without performing transformation.
    
    Args:
        request: ValidateOnlyRequest with original and transformed JSON
    
    Returns:
        ValidationReport
    """
    try:
        original = request.original_json
        transformed = request.transformed_json
        locked_fields = request.locked_fields or config.LOCKED_FIELDS
        
        # Compute hashes for locked fields
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
        
        # Generate diff
        changed_paths = generate_json_diff(original, transformed)
        
        # Determine status
        locked_pass = len(validation_errors) == 0
        final_status = "OK" if locked_pass else "FAIL"
        
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
            final_status=final_status
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API and OpenAI connectivity."""
    try:
        # Test OpenAI connection
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


@router.get("/status/langsmith")
async def langsmith_status():
    """
    Check LangSmith observability configuration.

    Returns current LangSmith setup status and instructions.
    """
    from src.utils.gemini_client import get_langsmith_status

    status = get_langsmith_status()

    return {
        "langsmith": status,
        "instructions": {
            "step_1": "Get API key from https://smith.langchain.com",
            "step_2": "Set LANGCHAIN_API_KEY in .env file",
            "step_3": "Ensure LANGCHAIN_TRACING_V2=true",
            "step_4": "Restart server",
        },
        "traced_functions": [
            "extract_global_factsheet",
            "adapt_shard_content",
        ],
        "dashboard_url": f"https://smith.langchain.com/o/default/projects/p/{status['project']}" if status['tracing_enabled'] else None
    }


# ============================================================================
# FULL 7-STAGE PIPELINE - New LangGraph workflow
# ============================================================================

class PipelineRequest(BaseModel):
    """Request for full 7-stage pipeline."""
    input_json: dict = Field(..., description="The simulation JSON to adapt")
    target_scenario_index: Optional[int] = Field(None, description="Target scenario index")
    scenario_prompt: Optional[str] = Field(None, description="Free-form scenario prompt")
    max_retries: int = Field(3, description="Max compliance loop retries")


@router.post("/pipeline")
async def run_full_pipeline(request: PipelineRequest):
    """
    Run the FULL 7-stage adaptation pipeline.

    Stages:
    1. Sharder - Split JSON into shards
    2. Adaptation Engine - Transform shards (Gemini 2.5 Flash)
    3. Alignment Checker - Cross-shard consistency (GPT-5.2)
    4. Scoped Validation - Per-shard validation (parallel)
    4B. Fixers - Fix failing shards (hybrid LLM + patcher)
    5. Merger - Reassemble shards
    6. Finisher - Compliance loop
    7. Human Approval - Create approval package

    All nodes have @traceable for LangSmith observability.
    """
    try:
        from src.graph.nodes import run_pipeline

        # Determine selected scenario
        if request.scenario_prompt:
            selected_scenario = request.scenario_prompt
        elif request.target_scenario_index is not None:
            selected_scenario = request.target_scenario_index
        else:
            raise ValueError("Provide either 'scenario_prompt' or 'target_scenario_index'")

        # Run full pipeline
        final_state = await run_pipeline(
            input_json=request.input_json,
            selected_scenario=selected_scenario,
            max_retries=request.max_retries,
        )

        return {
            "status": "success",
            "final_status": final_state.get("final_status"),
            "pipeline_summary": {
                "total_runtime_ms": final_state.get("total_runtime_ms"),
                "stage_timings": final_state.get("stage_timings"),
                "compliance_score": final_state.get("compliance_score"),
                "compliance_passed": final_state.get("compliance_passed"),
                "blocker_count": final_state.get("blocker_count"),
                "warning_count": final_state.get("warning_count"),
            },
            "shards": {
                "total": len(final_state.get("shard_ids", [])),
                "locked": len(final_state.get("locked_shard_ids", [])),
            },
            "alignment": {
                "score": final_state.get("alignment_score"),
                "passed": final_state.get("alignment_passed"),
                "report": final_state.get("alignment_report"),  # Detailed check results
            },
            "validation": {
                "score": final_state.get("validation_score"),
                "passed": final_state.get("validation_passed"),
                "report": final_state.get("validation_report"),  # Detailed validation results
            },
            "fixes": {
                "patches_applied": len(final_state.get("patches_applied", [])),
            },
            "human_review": {
                "flagged_shards": final_state.get("flagged_for_human", []),
                "approval_status": final_state.get("approval_status"),
            },
            "errors": final_state.get("errors", []),
            "debug": {
                "global_factsheet": final_state.get("global_factsheet"),
                "rag_context_length": len(final_state.get("rag_context", "") or ""),
                "source_scenario": final_state.get("source_scenario", "")[:200] if final_state.get("source_scenario") else None,
                "target_scenario": final_state.get("target_scenario", "")[:200] if final_state.get("target_scenario") else None,
            },
            "output_json": final_state.get("output_json"),
            "approval_package": final_state.get("approval_package"),
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")


@router.post("/pipeline/stream")
async def run_pipeline_streaming(request: PipelineRequest):
    """
    Run full pipeline with streaming progress updates.

    Returns SSE events for each stage completion.
    Sends heartbeat every 10s to prevent Heroku timeout.
    """
    async def event_generator():
        import asyncio
        from collections import deque

        try:
            from src.graph.nodes import run_pipeline_streaming

            # Determine selected scenario
            if request.scenario_prompt:
                selected_scenario = request.scenario_prompt
            elif request.target_scenario_index is not None:
                selected_scenario = request.target_scenario_index
            else:
                yield f"data: {json.dumps({'event': 'error', 'message': 'Provide scenario_prompt or target_scenario_index'})}\n\n"
                return

            yield f"data: {json.dumps({'event': 'start', 'message': 'Starting 7-stage pipeline'})}\n\n"

            # Use queue to collect states from pipeline
            state_queue = asyncio.Queue()
            pipeline_done = asyncio.Event()
            final_state = {}
            pipeline_error = None

            async def run_pipeline():
                nonlocal final_state, pipeline_error
                try:
                    last_stage = None
                    async for state in run_pipeline_streaming(
                        input_json=request.input_json,
                        selected_scenario=selected_scenario,
                        max_retries=request.max_retries,
                    ):
                        current_stage = state.get("current_stage", "unknown")
                        if current_stage != last_stage:
                            await state_queue.put({
                                'event': 'stage_complete',
                                'stage': current_stage,
                                'timing_ms': state.get('stage_timings', {}).get(current_stage, 0)
                            })
                            last_stage = current_stage
                        final_state = state
                except Exception as e:
                    import traceback
                    pipeline_error = {'message': str(e), 'traceback': traceback.format_exc()}
                finally:
                    pipeline_done.set()

            # Start pipeline in background
            pipeline_task = asyncio.create_task(run_pipeline())

            # Send heartbeats while waiting for pipeline
            heartbeat_count = 0
            while not pipeline_done.is_set():
                try:
                    # Check for new states (non-blocking with timeout)
                    state_event = await asyncio.wait_for(state_queue.get(), timeout=10.0)
                    yield f"data: {json.dumps(state_event)}\n\n"
                except asyncio.TimeoutError:
                    # No state update - send heartbeat to keep connection alive
                    heartbeat_count += 1
                    yield f"data: {json.dumps({'event': 'heartbeat', 'count': heartbeat_count})}\n\n"

            # Drain any remaining states in queue
            while not state_queue.empty():
                state_event = await state_queue.get()
                yield f"data: {json.dumps(state_event)}\n\n"

            # Check for errors
            if pipeline_error:
                yield f"data: {json.dumps({'event': 'error', **pipeline_error})}\n\n"
            else:
                # Final result
                yield f"data: {json.dumps({'event': 'complete', 'final_status': final_state.get('final_status'), 'total_runtime_ms': final_state.get('total_runtime_ms')})}\n\n"

        except Exception as e:
            import traceback
            yield f"data: {json.dumps({'event': 'error', 'message': str(e), 'traceback': traceback.format_exc()})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


@router.get("/scenarios")
async def list_scenarios(input_json: str = None):
    """
    List available scenarios from input JSON.
    
    Args:
        input_json: Base64 encoded JSON or JSON string
    
    Returns:
        List of scenario options
    """
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
