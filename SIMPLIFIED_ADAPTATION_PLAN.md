# Simplified Adaptation Plan

**Date:** 2026-01-16
**Goal:** Simplify adaptation to just: Scenario Prompt + JSON → LLM → Adapted JSON

---

## Current Approach (Over-engineered)

### File: `src/stages/adaptation_engine.py` (847 LOC)

### Current Flow

```
1. INPUT: JSON + target_scenario_index OR scenario_prompt

2. DETERMINE SOURCE SCENARIO
   - Extract from JSON's selectedScenarioOption
   - ~20 lines of code handling different formats

3. EXTRACT GLOBAL FACTSHEET (Separate LLM Call - Gemini)
   - Calls extract_global_factsheet() from gemini_client.py
   - Extracts: company, poison_list, KLOs, industry_context
   - This is injected into EVERY shard adaptation

4. BUILD RAG CONTEXT
   - Extracts industry_context from factsheet
   - Builds KPIs, terminology, wrong_terms lists
   - ~40 lines of code

5. SHARD THE JSON
   - Splits into 15 shards (2 locked, 13 unlocked)
   - Uses Sharder class

6. INDEX FOR RAG (Optional)
   - Indexes input into ChromaDB
   - For per-shard example retrieval

7. GET PER-SHARD RAG EXAMPLES
   - Retrieves similar examples from ChromaDB
   - For each shard type

8. PARALLEL SHARD ADAPTATION
   - Each shard sent to Gemini with:
     - source_scenario
     - target_scenario
     - global_factsheet (from step 3)
     - rag_context (from step 4 + 7)
   - 13 parallel LLM calls

9. MERGE SHARDS
   - Combines adapted shards back

10. POST-PROCESSING
    - Calls post_process_content()
    - Content fixes, cleanup

11. UPDATE selectedScenarioOption
    - Sets the new scenario in output

OUTPUT: AdaptationResult with adapted_json
```

### Current LLM Calls

| Step | LLM Calls | Model | Purpose |
|------|-----------|-------|---------|
| Factsheet | 1 | Gemini | Extract company, poison list, KLOs |
| Adaptation | 13 | Gemini | Per-shard adaptation |
| **Total** | **14** | | |

### Key Functions

| Function | Lines | What It Does |
|----------|-------|--------------|
| `adapt()` | 152-412 | Main entry point, orchestrates everything |
| `extract_global_factsheet()` | External | Separate LLM call for context |
| `_adapt_shards_parallel()` | 482-571 | Parallel shard processing |
| `_adapt_single_shard()` | 576-633 | Single shard adaptation |
| `_adapt_simulation_flow_by_stages()` | 635-702 | Splits large shards |

### Problems with Current Approach

1. **Factsheet Extraction is Separate** - Extra LLM call that may miss context
2. **Poison List Incomplete** - Misses domain-implied terms (185 "HR" occurrences)
3. **Cross-Shard Blindness** - Each shard adapted without seeing others
4. **Over-engineered** - RAG indexing, per-shard examples, etc.
5. **Complex** - 847 lines for what should be simple

---

## Shweta's Approach (Simple)

### The Insight

From Shweta's guidance:
> "I'm not changing the context. I'm changing the scenario."
> "Swap out the scenario for this"

The scenario prompt already contains everything the LLM needs:
- Company name (EcoChic Threads)
- Industry (organic T-shirts)
- Role (junior consultant)
- Challenge (go/no-go market entry)

### Simplified Flow

```
1. INPUT:
   - Scenario Prompt: "learners will act as a junior consultant for
     an exciting Gen Z organic T-shirts brand..."
   - JSON: sample_main.json

2. ADAPTATION (Single concept, may still shard for size):
   - Give LLM the scenario prompt
   - Give LLM the JSON (or shards)
   - LLM figures out what to change
   - No separate factsheet extraction

3. OUTPUT:
   - Adapted JSON matching the scenario
```

### The Key Prompt

```
You are adapting a business simulation to a new scenario.

## TARGET SCENARIO (this is what the simulation should be about):
{scenario_prompt}

## RULES:
1. The simulation structure (KLOs, rubric format, activity flow) stays the same
2. Replace ALL company names, people names, industry terms with TARGET scenario
3. Make resources relevant to the TARGET industry
4. Ensure submission questions are answerable from the adapted resources
5. Keep the same educational purpose, just change the context

## INPUT JSON:
{json_content}

## OUTPUT:
Return the adapted JSON with all content changed to match the TARGET SCENARIO.
```

### Why This Works

1. **LLM is Smart** - It can infer what the current JSON is about by reading it
2. **Scenario Prompt is Context** - Contains all needed info (company, industry, role)
3. **No Poison List Needed** - LLM understands "replace HR terms with T-shirt terms"
4. **One Concept** - Simpler to understand, debug, and maintain

---

## Implementation Plan

### Option A: Minimal Change (Keep Sharding)

Keep sharding for context window limits, but remove factsheet extraction.

```python
async def adapt(self, input_json: dict, scenario_prompt: str) -> AdaptationResult:
    """Simplified adaptation - just scenario prompt + JSON."""

    # 1. Shard the JSON (for size only)
    sharder = Sharder()
    collection = sharder.shard(input_json)

    # 2. Adapt each shard with scenario prompt
    # NO factsheet extraction - scenario prompt IS the context
    for shard in unlocked_shards:
        adapted = await adapt_shard_simple(
            shard=shard,
            scenario_prompt=scenario_prompt,  # This is all we need
        )

    # 3. Merge and return
    return merge_shards(collection, input_json)
```

### Option B: Monolithic (No Sharding)

Try adapting the entire JSON in one call (if context window allows).

```python
async def adapt_simple(input_json: dict, scenario_prompt: str) -> dict:
    """Ultra-simple: one LLM call for entire JSON."""

    prompt = f"""
    Adapt this simulation to the following scenario:

    SCENARIO: {scenario_prompt}

    RULES:
    - Keep structure (KLOs, rubric format, activities)
    - Replace all company/industry/people references
    - Make resources match the new industry

    JSON:
    {json.dumps(input_json)}
    """

    return await call_gemini(prompt)
```

### What to Remove

| Current Code | Action |
|--------------|--------|
| `extract_global_factsheet()` call | Remove |
| RAG context building | Remove |
| RAG indexing | Remove |
| Per-shard RAG examples | Remove |
| Poison list handling | Remove (LLM infers) |
| Complex source scenario detection | Simplify |

### What to Keep

| Current Code | Action |
|--------------|--------|
| Sharding (for large JSON) | Keep (optional) |
| Parallel processing | Keep |
| Post-processing cleanup | Keep (for now) |
| Basic structure | Keep |

---

## New File Structure

### Simplified `adaptation_engine.py`

```python
"""
Simplified Adaptation Engine

Just: Scenario Prompt + JSON → LLM → Adapted JSON
"""

class AdaptationEngine:
    """Simple scenario-based adaptation."""

    async def adapt(
        self,
        input_json: dict,
        scenario_prompt: str,
    ) -> AdaptationResult:
        """
        Adapt simulation to match scenario prompt.

        Args:
            input_json: Original simulation JSON
            scenario_prompt: Target scenario description

        Returns:
            AdaptationResult with adapted JSON
        """
        # 1. Shard for size (if needed)
        shards = self._shard_if_needed(input_json)

        # 2. Adapt with scenario prompt only
        adapted_shards = await self._adapt_parallel(shards, scenario_prompt)

        # 3. Merge and return
        return self._merge_result(adapted_shards)

    async def _adapt_shard(self, shard: dict, scenario_prompt: str) -> dict:
        """Adapt single shard using scenario prompt."""

        prompt = self._build_prompt(shard, scenario_prompt)
        return await call_gemini(prompt)

    def _build_prompt(self, shard: dict, scenario_prompt: str) -> str:
        """Build the adaptation prompt."""

        return f"""
You are adapting a business simulation to a new scenario.

## TARGET SCENARIO:
{scenario_prompt}

## RULES:
1. Keep the same structure (this is a {shard['type']} section)
2. Replace all company/industry/people references with TARGET scenario
3. Make content relevant to the TARGET industry
4. Preserve educational purpose

## CONTENT TO ADAPT:
{json.dumps(shard['content'], indent=2)}

## OUTPUT:
Return ONLY the adapted JSON. No explanations.
"""
```

---

## Metrics to Track

| Metric | Current | Target |
|--------|---------|--------|
| LLM Calls | 14 | 1-13 (depending on sharding) |
| Lines of Code | 847 | ~200 |
| Factsheet Extraction | Yes | No |
| RAG Complexity | High | None |
| Alignment Score | 92.33% | ≥95% |

---

## Next Steps

1. **Create simplified `adapt_simple()` function** - Test with one shard first
2. **Test monolithic approach** - See if entire JSON fits in context
3. **Compare results** - Run both approaches, compare alignment scores
4. **Iterate on prompt** - Refine the adaptation prompt based on results

---

## Questions to Resolve

1. **Context window**: Can we fit entire JSON in one call? (~65K chars)
2. **Sharding strategy**: If we shard, how do we ensure cross-shard consistency?
3. **Validation**: Do we still need alignment checking, or trust the LLM?
4. **Model choice**: Gemini vs GPT for this simpler approach?

---

*Plan created: 2026-01-16*
*Based on Shweta's guidance and current codebase analysis*
