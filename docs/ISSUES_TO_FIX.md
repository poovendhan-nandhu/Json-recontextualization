⏺ Now I have a comprehensive view of the codebase. Let me provide a detailed review.

  ---
  JSON Recontextualization Pipeline Review

  Overview

  This is a 7-stage LangGraph pipeline for adapting business simulation JSON from one industry scenario to another. The pipeline uses Gemini 2.5 Flash for adaptation and GPT-5.2 for validation/fixing.

  ---
  🔴 CRITICAL ISSUES

  1. Compliance Retry Loop Disabled (nodes.py:757-766)

  def should_retry_compliance(state: PipelineState) -> str:
      # Always go to human_approval - no retry loop
      return "human_approval"
  Problem: The compliance retry loop is completely disabled. Even if validation fails, it goes straight to human approval instead of attempting fixes.

  Impact: Pipeline cannot self-correct validation failures.

  Fix: Re-enable with proper recursion limit handling.

  ---
  2. Missing Error Recovery in Alignment Regeneration (nodes.py:395-400)

  except Exception as e:
      logger.error(f"Alignment check failed: {e}")
      add_error(state, "alignment", str(e), is_fatal=False)
      if "alignment_score" not in state or state["alignment_score"] == 0:
          state["alignment_passed"] = True  # Only default to True if no score yet
  Problem: On exception, alignment_passed defaults to True which lets invalid content pass through.

  Impact: Corrupted data may proceed to later stages.

  ---
  3. Race Condition in Parallel Shard Updates (nodes.py:370-378)

  # Update adapted_shards with regenerated content
  if isinstance(adapted_shards, list):
      for shard in adapted_shards:
          if hasattr(shard, 'id') and shard.id == shard_id:
              shard.content = new_content  # Mutation during iteration
  Problem: Mutating shards in-place during iteration can cause race conditions in async context.

  ---
  4. Hardcoded Model Reference (fixers.py:45, scoped_validators.py:41)

  FIXER_MODEL = os.getenv("FIXER_MODEL", "gpt-5.2-2025-12-11")
  VALIDATION_MODEL = os.getenv("VALIDATION_MODEL", "gpt-5.2-2025-12-11")
  Problem: Model gpt-5.2-2025-12-11 is a future model (Dec 2025). This will fail if the model doesn't exist.

  ---
  🟠 MISSING PARTS

  1. No KLO Alignment Validator (Missing from scoped_validators.py)

  The KLOAlignmentValidator class is mentioned in comments but not implemented. Only KLOAlignmentFixer exists in fixers.py.

  Impact: KLO-Question alignment issues detected by fixer but no dedicated validator to catch them during validation phase.

  ---
  2. Missing base_shards Context Passing (nodes.py:428-437)

  base_shards_map = {s.id: s for s in original_shards if hasattr(s, 'id')}
  context = {
      ...
      "base_shards": base_shards_map,  # For comparing with original
  }
  Problem: base_shards is created but individual validators access base_shard (singular) via:
  base_shard = context.get("base_shard")  # NOT base_shards!
  The mapping happens in validate_shard() but relies on base_shards being present.

  ---
  3. Incomplete Industry Term Coverage (config.py:252-308)

  Only 6 industries defined:
  - beverage, hospitality, retail, manufacturing, healthcare, tech_saas

  Missing: Airlines, banking, insurance, real estate, education, pharmaceuticals, etc.

  ---
  4. No Rollback Integration (fixers.py:798-821)

  rollback_shard() method exists but is never called anywhere in the pipeline. Patches are stored but never used for recovery.

  ---
  5. Missing UTF-8 Sanitization in All Entry Points

  Sanitization exists in nodes.py:38-55 but not consistently applied:
  - alignment_node doesn't sanitize before regeneration
  - fixers_node doesn't sanitize LLM responses

  ---
  🟡 OPTIMIZATION OPPORTUNITIES

  1. Duplicate LLM Calls for KLO Alignment (nodes.py:199-209)

  # KLO Alignment Fixer runs AFTER adaptation but BEFORE alignment checking
  klo_fix_context = {
      "global_factsheet": result.global_factsheet,
  }
  fixed_json = await fix_klo_alignment(result.adapted_json, klo_fix_context)
  Problem: KLO alignment is checked twice:
  1. In fix_klo_alignment() during adaptation
  2. In AlignmentChecker during alignment node

  Optimization: Run KLO alignment only once, cache results.

  ---
  2. Redundant JSON Parsing/Serialization (fixers.py, scoped_validators.py)

  content_text = json.dumps(content) if isinstance(content, (dict, list)) else str(content)
  This pattern appears 10+ times. Each dumps/loads cycle costs ~1-5ms.

  Optimization: Cache serialized content or pass content_text in context.

  ---
  3. Non-Batched Validator LLM Calls (scoped_validators.py:584-660)

  ToneValidator makes individual LLM calls per shard.

  Optimization: Batch tone validation with BatchedShardChecker or remove it (low value).

  ---
  4. Inefficient Per-Question KLO Checking (fixers.py:1321-1327)

  tasks = [
      self._check_and_fix_klo(klo, questions, company_name, industry)
      for klo in klos
  ]
  results = await asyncio.gather(*tasks, return_exceptions=True)
  Each KLO triggers a separate LLM call. For 5 KLOs = 5 API calls.

  Optimization: Batch all KLOs into single prompt, parse structured output.

  ---
  5. No Caching of RAG Retrieval (adaptation_engine.py:331-383)

  async def _get_per_shard_rag_examples(...):
      retriever = SimulationRetriever()  # New instance each time
  Optimization: Singleton pattern or connection pooling for ChromaDB.

  ---
  6. Large Context in LLM Prompts (scoped_validators.py:1202-1208)

  ## CURRENT CONTENT:
  ```json
  {json.dumps(content, indent=2)[:8000]}
  8000 chars = ~2000 tokens per shard validation call.

  **Optimization**: Compress content, use summaries, or only include relevant sections.

  ---

  ## 🔵 CODE QUALITY ISSUES

  ### 1. **Mock Classes in Production Code** (`nodes.py:505-511, 691-700`)
  ```python
  class MockValidationReport:
      def __init__(self, report_dict):
          ...
  class MockComplianceResult:
      ...
  Problem: Mock objects used in production flow. Should use proper dataclasses/Pydantic models.

  ---
  2. Inconsistent Issue Format Handling (fixers.py:631-655)

  def get_rule_id(issue):
      if isinstance(issue, dict):
          return issue.get('rule_id', '')
      return getattr(issue, 'rule_id', '')
  Same pattern repeated 5+ times. Should normalize at validation boundary.

  ---
  3. Debug Print Statements in Production (adaptation_engine.py:211, 419)

  print(f"[RAG] use_per_shard_rag = {self.use_per_shard_rag}")
  print(f"[RAG] Indexing input for RAG retrieval...")
  Problem: Print statements instead of proper logging.

  ---
  4. Overly Broad Exception Handling (Multiple locations)

  except Exception as e:
      logger.warning(f"... failed: {e}")
      return default_value  # Swallows all errors
  Problem: Swallowing exceptions hides bugs and makes debugging difficult.

  ---
  📊 SUMMARY TABLE
  ┌─────────────────┬───────┬───────────────┐
  │    Category     │ Count │   Severity    │
  ├─────────────────┼───────┼───────────────┤
  │ Critical Issues │ 4     │ 🔴 High       │
  ├─────────────────┼───────┼───────────────┤
  │ Missing Parts   │ 5     │ 🟠 Medium     │
  ├─────────────────┼───────┼───────────────┤
  │ Optimizations   │ 6     │ 🟡 Low-Medium │
  ├─────────────────┼───────┼───────────────┤
  │ Code Quality    │ 4     │ 🔵 Low        │
  └─────────────────┴───────┴───────────────┘
  ---
  🎯 PRIORITY FIXES

  1. P0: Re-enable compliance retry loop with proper recursion handling
  2. P0: Fix alignment exception handler (don't default to passed=True)
  3. P1: Validate model name exists before use
  4. P1: Add base_shards → base_shard mapping consistently
  5. P2: Batch KLO alignment checks into single LLM call
  6. P2: Remove debug print statements
  7. P3: Add more industry term definitions

✻ Worked for 3m 3s