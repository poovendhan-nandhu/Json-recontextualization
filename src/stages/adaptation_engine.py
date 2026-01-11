"""
Stage 1: Adaptation Engine

Parallel shard-based transformation using Gemini 2.5 Flash.

KEY IMPROVEMENTS:
1. Global Factsheet extraction BEFORE parallel processing (consistency)
2. Poison list to avoid source scenario terms
3. Statistics tracking
4. LangSmith tracing

Flow:
1. Extract Global Factsheet (one LLM call)
2. Shard the JSON (13 shards)
3. Filter to UNLOCKED shards only (8 shards)
4. Adapt ALL unlocked shards IN PARALLEL (with shared factsheet)
5. Merge back with locked shards
"""
import asyncio
import copy
import json
import logging
import time
from typing import Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class AdaptationResult:
    """Result from adaptation engine."""
    adapted_json: dict
    entity_map: dict  # Combined from all shards
    source_scenario: str
    target_scenario: str
    global_factsheet: dict  # The shared factsheet
    shards_adapted: int
    shards_locked: int
    parallel_time_ms: int
    total_time_ms: int
    rag_context: str = ""
    stats: dict = field(default_factory=dict)


class AdaptationEngine:
    """
    Stage 1: Parallel shard-based adaptation with global consistency.

    KEY PRINCIPLES:
    - Global Factsheet extracted FIRST (ensures consistency)
    - Poison list prevents source term leakage
    - PARALLEL processing of unlocked shards
    - Locked shards NEVER sent to LLM
    - Statistics tracking for cost/performance
    - PER-SHARD RAG: Each shard gets similar examples from its type collection
    """

    def __init__(self, rag_retriever=None, rag_context: str = "", use_per_shard_rag: bool = True):
        """
        Args:
            rag_retriever: Optional RAG retriever for context (dynamic)
            rag_context: Optional pre-built RAG context string (static)
            use_per_shard_rag: Enable per-shard-type RAG retrieval (default True)
        """
        self.rag_retriever = rag_retriever
        self._static_rag_context = rag_context  # Pre-built context
        self.use_per_shard_rag = use_per_shard_rag
        self._shard_examples_cache = {}  # Cache for per-shard examples

    async def adapt(
        self,
        input_json: dict,
        target_scenario_index: int = None,
        scenario_prompt: str = None,
    ) -> AdaptationResult:
        """
        Adapt simulation to new scenario using parallel shard processing.

        TWO INPUT OPTIONS:
        - Option A: target_scenario_index → Select from existing scenarioOptions
        - Option B: scenario_prompt → Free-form text (any custom scenario)

        Args:
            input_json: Original simulation JSON
            target_scenario_index: Index in scenarioOptions (Option A)
            scenario_prompt: Free-form scenario text (Option B)

        Returns:
            AdaptationResult with adapted JSON and stats
        """
        from ..utils.llm_stats import get_stats_summary, reset_stats

        # Reset stats for this run
        reset_stats()

        total_start = time.time()

        # 1. Determine source scenario (what the JSON currently represents)
        topic_data = input_json.get("topicWizardData", {})
        scenario_options_raw = topic_data.get("scenarioOptions", [])

        # Extract scenario texts (handle both formats)
        if scenario_options_raw:
            if isinstance(scenario_options_raw[0], dict):
                scenario_options = [s.get("option", str(s)) for s in scenario_options_raw]
            else:
                scenario_options = scenario_options_raw
        else:
            scenario_options = []

        # Get current/source scenario
        selected = topic_data.get("selectedScenarioOption", 0)
        if isinstance(selected, dict):
            source_scenario = selected.get("option", "")
            if not source_scenario and scenario_options:
                source_scenario = scenario_options[0]
        elif isinstance(selected, int) and scenario_options:
            source_scenario = scenario_options[selected] if selected < len(scenario_options) else scenario_options[0]
        elif scenario_options:
            source_scenario = scenario_options[0]
        else:
            source_scenario = "Unknown source scenario"

        # 2. Determine target scenario (Option A or Option B)
        if scenario_prompt:
            # OPTION B: Free-form prompt
            target_scenario = scenario_prompt
            logger.info(f"Mode: FREE-FORM PROMPT")
        elif target_scenario_index is not None:
            # OPTION A: Select from existing options
            if not scenario_options:
                raise ValueError("No scenarioOptions in JSON. Use 'scenario_prompt' for free-form input.")
            if target_scenario_index >= len(scenario_options):
                raise ValueError(f"Invalid index {target_scenario_index}. Max: {len(scenario_options)-1}")
            target_scenario = scenario_options[target_scenario_index]
            logger.info(f"Mode: SELECT FROM INDEX [{target_scenario_index}]")
        else:
            raise ValueError("Provide either 'scenario_prompt' or 'target_scenario_index'")

        logger.info(f"Source: {source_scenario[:60]}...")
        logger.info(f"Target: {target_scenario[:60]}...")

        # 2. ⭐ EXTRACT GLOBAL FACTSHEET FIRST (critical for consistency)
        from ..utils.gemini_client import extract_global_factsheet

        logger.info("Extracting global factsheet...")
        global_factsheet = await extract_global_factsheet(
            source_scenario=source_scenario,
            target_scenario=target_scenario,
        )
        # Safe logging - handle unexpected factsheet types
        if isinstance(global_factsheet, dict):
            company_info = global_factsheet.get('company', {})
            company_name = company_info.get('name', 'Unknown') if isinstance(company_info, dict) else 'Unknown'
            poison_list = global_factsheet.get('poison_list', [])
            poison_count = len(poison_list) if isinstance(poison_list, list) else 0
            logger.info(f"Factsheet: company={company_name}, poison_list={poison_count} terms")
        else:
            logger.warning(f"Unexpected factsheet type: {type(global_factsheet)}")

        # 2b. ⭐ BUILD RAG CONTEXT from LLM-generated industry_context in factsheet
        # No hardcoded lookups - everything comes from LLM
        if not self._static_rag_context and isinstance(global_factsheet, dict):
            try:
                # Extract LLM-generated industry context
                industry_ctx = global_factsheet.get("industry_context", {})
                industry_ctx = industry_ctx if isinstance(industry_ctx, dict) else {}

                company_data = global_factsheet.get("company", {})
                company_data = company_data if isinstance(company_data, dict) else {}
                industry_name = company_data.get("industry", "Unknown")
                industry_name = industry_name if isinstance(industry_name, str) else "Unknown"

                # Get LLM-generated lists
                kpis = industry_ctx.get("kpis", [])
                kpis = kpis if isinstance(kpis, list) else []
                terminology = industry_ctx.get("terminology", [])
                terminology = terminology if isinstance(terminology, list) else []
                wrong_terms = industry_ctx.get("wrong_terms", [])
                wrong_terms = wrong_terms if isinstance(wrong_terms, list) else []

                if kpis or terminology:
                    self._static_rag_context = f"""## Industry Context: {industry_name.upper()}

### Key Performance Indicators (KPIs) for this industry:
{', '.join(str(k) for k in kpis[:15])}

### Industry-Specific Terminology to use:
{', '.join(str(t) for t in terminology[:20])}

### Terms to AVOID (wrong for this industry):
{', '.join(str(w) for w in wrong_terms[:15])}

Use these KPIs and terminology when adapting content. Replace any wrong terms with appropriate industry equivalents."""
                    logger.info(f"Built RAG context from factsheet for {industry_name}: {len(self._static_rag_context)} chars")
                else:
                    logger.warning("No industry_context in factsheet - RAG context will be empty")
            except Exception as e:
                logger.warning(f"Failed to build RAG context from factsheet: {e}")

        # 3. Shard the JSON
        from .sharder import Sharder, merge_shards
        from ..models.shard import LockState

        sharder = Sharder()
        collection = sharder.shard(input_json)

        # 4. Separate locked vs unlocked shards
        locked_shards = [s for s in collection.shards if s.lock_state == LockState.FULLY_LOCKED]
        unlocked_shards = [s for s in collection.shards if s.lock_state != LockState.FULLY_LOCKED]

        logger.info(f"Shards: {len(locked_shards)} locked, {len(unlocked_shards)} unlocked")

        # 5. Get RAG context (if available)
        rag_context = ""
        # Option 1: Use pre-built static context
        if self._static_rag_context:
            rag_context = self._static_rag_context
            logger.info(f"Using static RAG context: {len(rag_context)} chars")
        # Option 2: Use dynamic retriever
        elif self.rag_retriever:
            rag_context = await self._get_rag_context(target_scenario)
            logger.info(f"Retrieved RAG context: {len(rag_context)} chars")

        # 6. Build adaptation context (shared across all shards)
        adaptation_context = {
            "source_scenario": source_scenario,
            "target_scenario": target_scenario,
            "global_factsheet": global_factsheet,  # ⭐ SHARED FACTSHEET
            "rag_context": rag_context,
        }

        # 7. ⭐ PARALLEL ADAPTATION with shared factsheet
        parallel_start = time.time()
        logger.info(f"Starting parallel adaptation of {len(unlocked_shards)} shards...")

        adapted_shards, all_entity_maps = await self._adapt_shards_parallel(
            unlocked_shards, adaptation_context
        )

        parallel_time_ms = int((time.time() - parallel_start) * 1000)

        # 8. Update collection with adapted shards
        for adapted_shard in adapted_shards:
            collection.update_shard(adapted_shard)

        # 9. Merge back to full JSON
        adapted_json = merge_shards(collection, input_json)

        # 9b. ⭐ POST-PROCESSING: Comprehensive content fixes
        from ..utils.content_processor import post_process_content
        logger.info("Applying comprehensive post-processing...")
        adapted_json = post_process_content(adapted_json, global_factsheet)

        # 10. Update selectedScenarioOption with full scenario object
        if "topicWizardData" in adapted_json:
            scenario_options = adapted_json["topicWizardData"].get("scenarioOptions", [])
            if target_scenario_index is not None and target_scenario_index < len(scenario_options):
                # Set to the actual scenario option object
                adapted_json["topicWizardData"]["selectedScenarioOption"] = scenario_options[target_scenario_index]
            elif scenario_prompt:
                # For free-form prompts, create a new scenario option object
                adapted_json["topicWizardData"]["selectedScenarioOption"] = {
                    "id": f"custom_{hash(scenario_prompt) % 10000}",
                    "option": target_scenario,
                    "recommendedTasks": []
                }

        total_time_ms = int((time.time() - total_start) * 1000)
        stats = get_stats_summary()

        logger.info(f"Adaptation complete: {total_time_ms}ms total, {parallel_time_ms}ms parallel")

        return AdaptationResult(
            adapted_json=adapted_json,
            entity_map=all_entity_maps,
            source_scenario=source_scenario,
            target_scenario=target_scenario,
            global_factsheet=global_factsheet,
            shards_adapted=len(unlocked_shards),
            shards_locked=len(locked_shards),
            parallel_time_ms=parallel_time_ms,
            total_time_ms=total_time_ms,
            rag_context=rag_context,
            stats=stats,
        )

    async def _get_rag_context(self, target_scenario: str) -> str:
        """Query RAG for relevant context (legacy method)."""
        try:
            results = self.rag_retriever.retrieve_context(
                query=target_scenario,
                n_results=5,
            )
            if results:
                context_parts = [r.document for r in results[:3]]
                return "\n\n".join(context_parts)
        except Exception as e:
            logger.warning(f"RAG retrieval failed: {e}")
        return ""

    async def _get_per_shard_rag_examples(
        self,
        shard_names: list[str],
        target_scenario: str,
        industry: str = None,
    ) -> dict[str, str]:
        """
        Retrieve similar examples for each shard type.

        This is the KEY RAG method - it retrieves similar content from
        the same shard type across indexed simulations to guide generation.

        Args:
            shard_names: List of shard IDs to get examples for
            target_scenario: Target scenario description for similarity search
            industry: Target industry for filtering

        Returns:
            Dict mapping shard_id to formatted RAG context string
        """
        from ..rag import SimulationRetriever

        try:
            retriever = SimulationRetriever()

            # Get examples for all shards
            all_examples = retriever.retrieve_all_shard_examples(
                target_scenario=target_scenario,
                shard_names=shard_names,
                n_results_per_shard=2,  # 2 examples per shard type
                exclude_simulation=None,  # Include all indexed simulations
                industry=industry,
            )

            # Format examples for each shard
            shard_contexts = {}
            for shard_name, examples in all_examples.items():
                if examples:
                    formatted = retriever.format_examples_for_prompt(
                        examples,
                        max_chars=4000,  # Limit per-shard context
                    )
                    shard_contexts[shard_name] = formatted
                    logger.debug(f"RAG: {shard_name} got {len(examples)} examples ({len(formatted)} chars)")
                else:
                    shard_contexts[shard_name] = ""

            logger.info(f"Per-shard RAG: Retrieved examples for {len(shard_contexts)} shards")
            return shard_contexts

        except Exception as e:
            logger.warning(f"Per-shard RAG retrieval failed: {e}")
            return {name: "" for name in shard_names}

    async def _adapt_shards_parallel(
        self,
        shards: list,
        context: dict,
    ) -> tuple[list, dict]:
        """
        Adapt multiple shards in PARALLEL using asyncio.gather().

        All shards receive the SAME global_factsheet for consistency.
        Each shard receives UNIQUE RAG examples from its shard type collection.

        Args:
            shards: List of unlocked shards to adapt
            context: Shared adaptation context (includes factsheet)

        Returns:
            (list of adapted shards, combined entity map)
        """
        # =====================================================================
        # STEP 1: Get per-shard RAG examples (if enabled)
        # =====================================================================
        shard_rag_contexts = {}
        if self.use_per_shard_rag:
            shard_names = [s.id for s in shards]
            target_scenario = context.get("target_scenario", "")
            industry = context.get("global_factsheet", {}).get("industry", None)

            logger.info(f"Retrieving per-shard RAG examples for {len(shard_names)} shards...")
            shard_rag_contexts = await self._get_per_shard_rag_examples(
                shard_names=shard_names,
                target_scenario=target_scenario,
                industry=industry,
            )

            # Cache for potential reuse
            self._shard_examples_cache = shard_rag_contexts

        # =====================================================================
        # STEP 2: Create tasks with per-shard context
        # =====================================================================
        tasks = []
        for shard in shards:
            # Build shard-specific context
            shard_context = context.copy()

            # Add per-shard RAG examples (if available)
            per_shard_rag = shard_rag_contexts.get(shard.id, "")
            if per_shard_rag:
                # Combine static RAG context with per-shard examples
                existing_rag = shard_context.get("rag_context", "")
                shard_context["rag_context"] = f"{existing_rag}\n\n{per_shard_rag}".strip()
                shard_context["per_shard_rag_examples"] = per_shard_rag

            tasks.append(self._adapt_single_shard(shard, shard_context))

        # =====================================================================
        # STEP 3: Run ALL tasks in parallel
        # =====================================================================
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        adapted_shards = []
        all_entity_maps = {}

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Shard {shards[i].id} failed: {result}")
                # Keep original shard on failure
                adapted_shards.append(shards[i])
            else:
                adapted_shard, entity_map = result
                adapted_shards.append(adapted_shard)
                all_entity_maps.update(entity_map)

        return adapted_shards, all_entity_maps

    # Max content size before splitting (in characters)
    MAX_SHARD_SIZE = 10000

    async def _adapt_single_shard(
        self,
        shard,
        context: dict,
    ) -> tuple[Any, dict]:
        """
        Adapt a single shard using Gemini with shared factsheet.
        Large shards (like simulation_flow) are split into sub-parts.

        Args:
            shard: Shard to adapt
            context: Adaptation context (includes global_factsheet)

        Returns:
            (adapted shard, entity mappings)
        """
        import json

        content_size = len(json.dumps(shard.content))
        logger.info(f"Shard {shard.id}: {content_size} chars")

        # Check if shard needs splitting
        if content_size > self.MAX_SHARD_SIZE and shard.id == "simulation_flow":
            logger.info(f"Splitting large shard: {shard.id} ({content_size} chars)")
            return await self._adapt_simulation_flow_by_stages(shard, context)

        # Normal single-call adaptation for smaller shards
        return await self._adapt_shard_direct(shard, context)

    async def _adapt_shard_direct(
        self,
        shard,
        context: dict,
    ) -> tuple[Any, dict]:
        """Direct adaptation for normal-sized shards."""
        from ..utils.gemini_client import adapt_shard_content

        logger.debug(f"Adapting shard directly: {shard.id}")

        try:
            adapted_content, entity_map = await adapt_shard_content(
                shard_id=shard.id,
                shard_name=shard.name,
                content=shard.content,
                source_scenario=context["source_scenario"],
                target_scenario=context["target_scenario"],
                global_factsheet=context["global_factsheet"],
                rag_context=context["rag_context"],
            )

            shard.content = adapted_content
            shard.current_hash = ""
            logger.debug(f"Shard {shard.id} adapted with {len(entity_map)} mappings")
            return shard, entity_map

        except Exception as e:
            logger.error(f"Failed to adapt shard {shard.id}: {e}")
            raise

    async def _adapt_simulation_flow_by_stages(
        self,
        shard,
        context: dict,
    ) -> tuple[Any, dict]:
        """
        Split simulation_flow into stages to stay under Gemini token limits.
        Each stage is processed in PARALLEL, then merged back.
        """
        from ..utils.gemini_client import adapt_shard_content

        # Extract the actual stages array from shard content
        # shard.content = {"topicWizardData.simulationFlow": [stages...]}
        content = shard.content
        if isinstance(content, dict):
            stages = content.get("topicWizardData.simulationFlow", [])
        else:
            stages = content

        if not isinstance(stages, list) or len(stages) == 0:
            # Fallback to direct if not a list
            return await self._adapt_shard_direct(shard, context)

        logger.info(f"Processing {len(stages)} stages in parallel...")

        # Create tasks for each stage
        tasks = []
        for i, stage in enumerate(stages):
            stage_id = f"simulation_flow_stage_{i}"
            stage_name = stage.get("name", f"Stage {i}")

            task = adapt_shard_content(
                shard_id=stage_id,
                shard_name=f"SimFlow: {stage_name}",
                content=stage,
                source_scenario=context["source_scenario"],
                target_scenario=context["target_scenario"],
                global_factsheet=context["global_factsheet"],
                rag_context=context["rag_context"],
            )
            tasks.append(task)

        # Run all stages in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Merge results back
        adapted_stages = []
        all_entity_maps = {}

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Stage {i} failed: {result}")
                # Keep original stage on failure
                adapted_stages.append(stages[i])
            else:
                adapted_content, entity_map = result
                adapted_stages.append(adapted_content)
                all_entity_maps.update(entity_map)

        # Update shard with merged stages - maintain dict structure
        if isinstance(shard.content, dict):
            shard.content["topicWizardData.simulationFlow"] = adapted_stages
        else:
            shard.content = adapted_stages
        shard.current_hash = ""

        logger.info(f"Merged {len(adapted_stages)} stages with {len(all_entity_maps)} mappings")
        return shard, all_entity_maps


# Convenience function
async def adapt_simulation(
    input_json: dict,
    target_scenario_index: int,
    rag_retriever=None,
    use_per_shard_rag: bool = True,
) -> AdaptationResult:
    """
    Adapt simulation to new scenario with parallel processing.

    Args:
        input_json: Original simulation JSON
        target_scenario_index: Target scenario index (0-36)
        rag_retriever: Optional RAG retriever for legacy context
        use_per_shard_rag: Enable per-shard-type RAG retrieval (default True)

    Returns:
        AdaptationResult with adapted JSON and per-shard RAG context
    """
    engine = AdaptationEngine(
        rag_retriever=rag_retriever,
        use_per_shard_rag=use_per_shard_rag,
    )
    return await engine.adapt(input_json, target_scenario_index)
