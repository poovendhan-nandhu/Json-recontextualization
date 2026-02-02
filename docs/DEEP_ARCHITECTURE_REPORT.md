# JSON Recontextualization - Deep Architecture Report

## Table of Contents
1. [Project Goal](#1-project-goal)
2. [Current Architecture](#2-current-architecture)
3. [Current Issues](#3-current-issues)
4. [Proposed Solution: JSON Whisperer + RAG](#4-proposed-solution-json-whisperer--rag)
5. [Implementation Plan](#5-implementation-plan)
6. [Success Metrics](#6-success-metrics)

---

## 1. Project Goal

### What Shweta Wants

**Primary Goal:** Transform educational simulation JSONs from one business scenario to another while preserving:
- Structure (all IDs, keys, relationships)
- Pedagogical alignment (KLOs ↔ Questions ↔ Resources)
- Quality (no placeholders, complete content)

### Input
```
┌─────────────────────────────────┐     ┌─────────────────────────────────┐
│     BASE SIMULATION JSON        │     │      SCENARIO PROMPT            │
│     (sample_main.json)          │     │                                 │
│                                 │     │  "Gen Z organic T-shirts brand, │
│  - Company: TechCorp            │  +  │   market entry analysis,        │
│  - Industry: Technology         │     │   junior consultant role..."    │
│  - KLOs, Questions, Resources   │     │                                 │
│  - 50,000+ characters           │     └─────────────────────────────────┘
└─────────────────────────────────┘
```

### Expected Output
```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ADAPTED SIMULATION JSON                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ✅ Company: BurgerBlitz (consistently used everywhere)                  │
│  ✅ Industry: QSR/Fast Food (domain-specific terminology)                │
│  ✅ KLOs: Adapted to new context, aligned with questions                 │
│  ✅ Questions: Test the adapted KLOs                                     │
│  ✅ Resources: Real data tables, charts with consistent numbers          │
│  ✅ Structure: 100% preserved (same IDs, same keys)                      │
│  ✅ Length: Similar to input (0.9x - 1.3x, NO explosion)                 │
│                                                                          │
│  Validation Score: ≥ 95%                                                 │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Key Requirements (from Shweta)
| Requirement | Target | Priority |
|-------------|--------|----------|
| KLO ↔ Question Alignment | 100% | CRITICAL |
| Structure Preservation | 100% | CRITICAL |
| Consistency (names, numbers) | 100% | HIGH |
| Domain Fidelity | ≥ 90% | HIGH |
| No Content Explosion | 0.9x-1.3x | HIGH |
| No Placeholders | 0 | HIGH |
| Overall Validation | ≥ 95% | HIGH |

---

## 2. Current Architecture

### How It Works Now

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        CURRENT PIPELINE                                  │
└─────────────────────────────────────────────────────────────────────────┘

     INPUT JSON (50K chars)
           │
           ▼
┌─────────────────────────┐
│  STAGE 1: DERIVE KLOs   │  ← LLM extracts Key Learning Outcomes
└─────────────────────────┘
           │
           ▼
┌─────────────────────────┐
│  STAGE 2: ENTITY MAP    │  ← LLM generates company/people/competitor names
└─────────────────────────┘
           │
           ▼
┌─────────────────────────┐
│  STAGE 3: FACTSHEET     │  ← LLM generates canonical numbers/metrics
└─────────────────────────┘
           │
           ▼
┌─────────────────────────┐
│  STAGE 4: SHARDING      │  ← Split JSON into 12+ batches by structure
│                         │
│  batch_0: questions     │
│  batch_1: rubrics       │
│  batch_2: resources     │
│  batch_3: emails        │
│  batch_4: scenarios     │
│  ...                    │
└─────────────────────────┘
           │
           ▼
┌─────────────────────────┐
│  STAGE 5: ADAPT SHARDS  │  ← Each shard sent to Gemini LLM
│  (PARALLEL)             │     with full prompt (4K-8K tokens each)
│                         │
│  Problem: LLM regenerates│
│  ENTIRE content, causing │
│  content explosion       │
└─────────────────────────┘
           │
           ▼
┌─────────────────────────┐
│  STAGE 6: MERGE         │  ← Reassemble shards into final JSON
└─────────────────────────┘
           │
           ▼
┌─────────────────────────┐
│  STAGE 7: VALIDATE      │  ← 8 parallel validators check quality
│                         │
│  - Domain Fidelity      │
│  - KLO-Question         │
│  - Consistency          │
│  - Completeness         │
│  - etc.                 │
└─────────────────────────┘
           │
           ▼
┌─────────────────────────┐
│  STAGE 8: REPAIR        │  ← If validation < 95%, repair issues
│  (up to 2 iterations)   │
└─────────────────────────┘
           │
           ▼
     OUTPUT JSON

```

### Current File Structure
```
src/stages/simple_adapter.py    # Main pipeline (2500+ lines)
├── derive_klos()               # Extract KLOs from scenario
├── generate_entity_map()       # Generate names mapping
├── generate_numeric_factsheet()# Generate canonical numbers
├── adapt_shard()               # Adapt single shard via LLM
├── normalize_consistency()     # Post-process name/number fixes
├── validate_adapted_content()  # Run 8 validators
└── repair_issues()             # Fix validation failures
```

### RAG Usage (Current)
```python
# Currently RAG is used for:
1. Indexing NumericFactsheet values for cross-shard consistency
2. Retrieving business facts for resource enrichment

# NOT used for:
- Querying specific JSON parts
- Reducing context size
- Patch generation
```

---

## 3. Current Issues

### Issue 1: Content Explosion (CRITICAL)
```
INPUT:  chat_history = 251 chars
OUTPUT: chat_history = 20,716 chars  (82x explosion!)

INPUT:  batch_0 = 7,941 chars
OUTPUT: batch_0 = 25,890 chars (3.3x explosion)
```

**Root Cause:** LLM is asked to "adapt" but it REGENERATES entire content, adding new sentences, expanding explanations.

### Issue 2: Consistency Failures
```
Shard 1 generates: "BurgerBlitz"
Shard 2 generates: "Burger Blitz"
Shard 3 generates: "BurgerBlitz Inc."

Same person appears as:
- "Ethan Reed" in shard 1
- "Mark Caldwell" in shard 2
```

**Root Cause:** Each shard is adapted independently. Even with entity_map in prompt, LLM doesn't always follow it.

### Issue 3: KLO-Question Misalignment
```
KLO: "Analyze competitor pricing strategies"
Question: "What is the company's mission statement?"  ← WRONG!
```

**Root Cause:** Questions are adapted without strong coupling to KLOs.

### Issue 4: Wrong Consistency Fixes
```
Before: "Solution Relevance" (rubric criterion)
After:  "Sophia Chen" (person name)  ← WRONG!
```

**Root Cause:** Similarity threshold too low, pattern matches non-names.

### Issue 5: Context Length
```
Each shard prompt includes:
- Entity map (~500 tokens)
- Factsheet (~300 tokens)
- KLOs (~400 tokens)
- Instructions (~800 tokens)
- Shard content (~2000 tokens)
─────────────────────────────
Total: ~4000 tokens PER SHARD

With 12 shards = 48,000 tokens input
Plus 12 outputs = 24,000+ tokens output
─────────────────────────────
Total: ~72,000 tokens per adaptation
```

---

## 4. Proposed Solution: JSON Whisperer + RAG

### Core Idea: Don't Regenerate, PATCH

Instead of asking LLM to output entire adapted JSON, ask it to output **patches** (changes only).

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    JSON WHISPERER APPROACH                               │
└─────────────────────────────────────────────────────────────────────────┘

CURRENT (Bad):
  LLM Input:  "Here's the JSON, adapt it for BurgerBlitz"
  LLM Output: [ENTIRE 5000 char JSON with changes + extras + explosion]

PROPOSED (Good):
  LLM Input:  "Here's the JSON, what needs to change for BurgerBlitz?"
  LLM Output: [
    {"op": "replace", "path": "/company", "value": "BurgerBlitz"},
    {"op": "replace", "path": "/industry", "value": "QSR"},
    {"op": "replace", "path": "/metric_1", "value": "15%"}
  ]
```

### Architecture: JSON Whisperer + RAG

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    NEW PIPELINE: JSON WHISPERER + RAG                    │
└─────────────────────────────────────────────────────────────────────────┘

     INPUT JSON (50K chars)
           │
           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STAGE 1: INDEX JSON IN RAG                                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Break JSON into semantic chunks and index in Pinecone:                  │
│                                                                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │ Chunk: Q1   │  │ Chunk: Q2   │  │ Chunk: R1   │  │ Chunk: E1   │     │
│  │ path: q[0]  │  │ path: q[1]  │  │ path: r[0]  │  │ path: e[0]  │     │
│  │ type: quest │  │ type: quest │  │ type: rsrc  │  │ type: email │     │
│  │ klo_ref: 1  │  │ klo_ref: 2  │  │ klo_ref: 1  │  │ stage: 1    │     │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘     │
│                                                                          │
│  Each chunk indexed with:                                                │
│  - JSON path (for reassembly)                                            │
│  - Content type (question/resource/email/rubric)                         │
│  - Related KLO reference                                                 │
│  - Embedding of content                                                  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STAGE 2: ANALYZE & PLAN CHANGES                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  LLM analyzes scenario prompt and determines what types of changes:      │
│                                                                          │
│  Input: "Gen Z organic T-shirts brand, market entry analysis"            │
│                                                                          │
│  Output (Change Plan):                                                   │
│  {                                                                       │
│    "company": {"from": "TechCorp", "to": "ThreadWell"},                  │
│    "industry": {"from": "Technology", "to": "Fashion/Apparel"},          │
│    "domain_terms": ["market penetration", "brand positioning", ...],     │
│    "metrics_needed": ["market_share", "revenue", "growth_rate"],         │
│    "klo_adaptations": [                                                  │
│      {"klo_id": 1, "new_focus": "analyze Gen Z consumer behavior"},      │
│      {"klo_id": 2, "new_focus": "evaluate sustainable sourcing"}         │
│    ]                                                                     │
│  }                                                                       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STAGE 3: RAG-GUIDED PATCH GENERATION                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  For each change type, query RAG for relevant chunks:                    │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  Query: "company name references"                                │    │
│  │  Retrieved: [chunk_3, chunk_7, chunk_12, chunk_15]               │    │
│  │                                                                  │    │
│  │  Generate Patches:                                               │    │
│  │  [                                                               │    │
│  │    {"op": "replace", "path": "/chunk_3/company", "value": "X"},  │    │
│  │    {"op": "replace", "path": "/chunk_7/org", "value": "X"},      │    │
│  │    {"op": "replace", "path": "/chunk_12/name", "value": "X"},    │    │
│  │    {"op": "replace", "path": "/chunk_15/firm", "value": "X"}     │    │
│  │  ]                                                               │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  Query: "questions related to KLO 1"                             │    │
│  │  Retrieved: [question_1, question_4]                             │    │
│  │                                                                  │    │
│  │  Generate Patches (with KLO context):                            │    │
│  │  [                                                               │    │
│  │    {"op": "replace", "path": "/q1/text", "value": "..."},        │    │
│  │    {"op": "replace", "path": "/q4/text", "value": "..."}         │    │
│  │  ]                                                               │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  Benefits:                                                               │
│  - Only retrieve chunks that need changes                                │
│  - LLM sees minimal context (just the relevant chunk)                    │
│  - Output is patches, not full content                                   │
│  - Guaranteed structure preservation                                     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STAGE 4: APPLY PATCHES                                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  import jsonpatch                                                        │
│                                                                          │
│  original_json = load("input.json")                                      │
│  patches = [...]  # All generated patches                                │
│                                                                          │
│  # Apply patches (RFC 6902 compliant)                                    │
│  adapted_json = jsonpatch.apply_patch(original_json, patches)            │
│                                                                          │
│  Benefits:                                                               │
│  - Structure 100% preserved (patches can't add/remove keys)              │
│  - Atomic operations (either all apply or none)                          │
│  - Auditable (can see exactly what changed)                              │
│  - Reversible (can undo patches)                                         │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STAGE 5: VALIDATE & REPAIR                                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Same 8 validators, but repair is also patch-based:                      │
│                                                                          │
│  If KLO-Question alignment fails:                                        │
│  1. Query RAG for misaligned question                                    │
│  2. Generate repair patch with KLO context                               │
│  3. Apply patch                                                          │
│                                                                          │
│  If consistency fails:                                                   │
│  1. Query RAG for inconsistent references                                │
│  2. Generate patches to standardize                                      │
│  3. Apply patches                                                        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
           │
           ▼
     OUTPUT JSON (same structure, adapted content)
```

### Patch Types (JSON Patch RFC 6902)

```json
// Replace a value (most common)
{"op": "replace", "path": "/company/name", "value": "BurgerBlitz"}

// Replace nested content
{"op": "replace", "path": "/questions/0/text", "value": "What is BurgerBlitz's market share?"}

// Replace array element
{"op": "replace", "path": "/resources/2/data/rows/0/revenue", "value": "$2.5M"}
```

### RAG Schema for JSON Chunks

```python
# Pinecone vector schema
{
    "id": "chunk_uuid",
    "values": [0.1, 0.2, ...],  # Embedding vector
    "metadata": {
        "json_path": "topicWizardData.simulationFlow[0].data.questions[0]",
        "content_type": "question",  # question|resource|email|rubric|scenario
        "klo_reference": "KLO_1",    # Which KLO this relates to
        "stage": 1,                   # Simulation stage number
        "original_text": "What is the company's revenue?",
        "char_count": 45
    }
}
```

### Query Examples

```python
# Find all company name references
results = index.query(
    vector=embed("company name organization firm"),
    filter={"content_type": {"$in": ["question", "resource", "email"]}},
    top_k=50
)

# Find questions for specific KLO
results = index.query(
    vector=embed("analyze competitor pricing strategy"),
    filter={"content_type": "question", "klo_reference": "KLO_2"},
    top_k=10
)

# Find resources that need domain adaptation
results = index.query(
    vector=embed("financial data revenue metrics"),
    filter={"content_type": "resource"},
    top_k=20
)
```

### Token Comparison

```
CURRENT APPROACH:
─────────────────
Per shard: ~4,000 tokens input + ~3,000 tokens output
12 shards × 7,000 = 84,000 tokens total
Plus repairs: +20,000 tokens
TOTAL: ~100,000 tokens per adaptation


JSON WHISPERER + RAG:
─────────────────────
Change plan: ~1,000 tokens
Per patch query: ~500 tokens input + ~100 tokens output
~50 patches × 600 = 30,000 tokens
Plus repairs: +5,000 tokens
TOTAL: ~36,000 tokens per adaptation

SAVINGS: 64% reduction in token usage
```

---

## 5. Implementation Plan

### Phase 1: RAG Infrastructure (Week 1)

```python
# src/rag/json_indexer.py

class JSONIndexer:
    """Index JSON into semantic chunks for RAG retrieval."""

    def __init__(self, pinecone_index):
        self.index = pinecone_index
        self.embedder = get_embedder()

    def index_json(self, json_data: dict, simulation_id: str) -> int:
        """Break JSON into chunks and index."""
        chunks = self._extract_chunks(json_data)

        vectors = []
        for chunk in chunks:
            embedding = self.embedder.embed(chunk['text'])
            vectors.append({
                "id": f"{simulation_id}_{chunk['path']}",
                "values": embedding,
                "metadata": {
                    "json_path": chunk['path'],
                    "content_type": chunk['type'],
                    "klo_reference": chunk.get('klo_ref'),
                    "original_text": chunk['text'][:1000]
                }
            })

        self.index.upsert(vectors)
        return len(vectors)

    def _extract_chunks(self, data: dict, path: str = "") -> list:
        """Recursively extract indexable chunks."""
        chunks = []

        # Identify chunk boundaries based on content type
        if self._is_question(data):
            chunks.append({
                "path": path,
                "type": "question",
                "text": self._get_question_text(data),
                "klo_ref": self._find_klo_reference(data)
            })
        elif self._is_resource(data):
            chunks.append({
                "path": path,
                "type": "resource",
                "text": self._get_resource_text(data)
            })
        # ... etc for emails, rubrics, scenarios

        return chunks
```

### Phase 2: Change Planner (Week 1)

```python
# src/stages/change_planner.py

async def plan_changes(original_json: dict, scenario_prompt: str) -> ChangePlan:
    """Analyze scenario and plan what needs to change."""

    prompt = f"""
    Analyze this scenario prompt and determine what needs to change.

    SCENARIO: {scenario_prompt}

    Output a JSON change plan:
    {{
        "company": {{"from": "detect from JSON", "to": "new name"}},
        "industry": {{"from": "...", "to": "..."}},
        "domain_terms": ["list of new industry terms to use"],
        "people": [
            {{"role": "manager", "name": "new name"}},
            ...
        ],
        "metrics": {{
            "revenue": "realistic value for new industry",
            "market_share": "...",
            ...
        }},
        "klo_adaptations": [
            {{"klo_id": 1, "new_focus": "..."}},
            ...
        ]
    }}
    """

    result = await llm.generate(prompt)
    return ChangePlan.parse(result)
```

### Phase 3: Patch Generator (Week 2)

```python
# src/stages/patch_generator.py

async def generate_patches(
    original_json: dict,
    change_plan: ChangePlan,
    rag_index: JSONIndexer
) -> list[dict]:
    """Generate JSON patches using RAG-retrieved context."""

    all_patches = []

    # 1. Company name patches
    company_chunks = rag_index.query(
        "company name organization firm",
        filter={"content_type": {"$ne": "metadata"}}
    )

    for chunk in company_chunks:
        patch = await _generate_patch_for_chunk(
            chunk=chunk,
            change_type="company_name",
            old_value=change_plan.company["from"],
            new_value=change_plan.company["to"]
        )
        all_patches.extend(patch)

    # 2. Question patches (with KLO alignment)
    for klo in change_plan.klo_adaptations:
        question_chunks = rag_index.query(
            klo["new_focus"],
            filter={"content_type": "question", "klo_reference": klo["klo_id"]}
        )

        for chunk in question_chunks:
            patch = await _generate_question_patch(
                chunk=chunk,
                klo=klo,
                change_plan=change_plan
            )
            all_patches.extend(patch)

    # 3. Resource patches
    # 4. Email patches
    # ...

    return all_patches


async def _generate_patch_for_chunk(chunk, change_type, old_value, new_value) -> list:
    """Generate patch for a single chunk."""

    prompt = f"""
    Generate a JSON patch to update this content.

    ORIGINAL:
    {chunk['original_text']}

    CHANGE: Replace "{old_value}" with "{new_value}"

    Output JSON patch format:
    [{{"op": "replace", "path": "{chunk['json_path']}/field", "value": "new content"}}]

    Rules:
    - ONLY output the patch, nothing else
    - Keep same length (don't expand content)
    - Preserve all formatting
    """

    result = await llm.generate(prompt)
    return json.loads(result)
```

### Phase 4: Patch Applier (Week 2)

```python
# src/stages/patch_applier.py

import jsonpatch

def apply_patches(original_json: dict, patches: list[dict]) -> dict:
    """Apply all patches to original JSON."""

    # Validate patches first
    validated_patches = []
    for patch in patches:
        try:
            # Check path exists
            jsonpatch.JsonPointer(patch['path']).resolve(original_json)
            validated_patches.append(patch)
        except jsonpatch.JsonPointerException:
            logger.warning(f"Invalid patch path: {patch['path']}")

    # Apply patches
    patch_obj = jsonpatch.JsonPatch(validated_patches)
    adapted_json = patch_obj.apply(original_json)

    return adapted_json
```

### Phase 5: Integration (Week 3)

```python
# src/stages/json_whisperer.py

async def adapt_with_patches(
    input_json: dict,
    scenario_prompt: str
) -> dict:
    """Main entry point for JSON Whisperer adaptation."""

    # 1. Index original JSON in RAG
    indexer = JSONIndexer(get_pinecone_index())
    chunk_count = indexer.index_json(input_json, simulation_id=uuid4())
    logger.info(f"Indexed {chunk_count} chunks")

    # 2. Plan changes
    change_plan = await plan_changes(input_json, scenario_prompt)
    logger.info(f"Change plan: {change_plan.summary()}")

    # 3. Generate patches using RAG
    patches = await generate_patches(input_json, change_plan, indexer)
    logger.info(f"Generated {len(patches)} patches")

    # 4. Apply patches
    adapted_json = apply_patches(input_json, patches)

    # 5. Validate
    validation = await validate_adapted_content(adapted_json, input_json, scenario_prompt)

    # 6. Repair if needed (also patch-based)
    if validation.overall_score < 0.95:
        repair_patches = await generate_repair_patches(
            adapted_json, validation.issues, indexer
        )
        adapted_json = apply_patches(adapted_json, repair_patches)

    return adapted_json
```

---

## 6. Success Metrics

### Current vs Target

| Metric | Current | Target (JSON Whisperer) |
|--------|---------|-------------------------|
| KLO-Question Alignment | 60-70% | ≥ 95% |
| Consistency | 70-80% | ≥ 98% |
| Content Explosion | 3x-80x | 1.0x (no expansion) |
| Token Usage | ~100K | ~36K |
| Validation Score | 85-91% | ≥ 95% |
| Adaptation Time | 60-90s | 30-45s |

### Why This Will Work

1. **No Content Explosion**: Patches can only REPLACE, not ADD. Structure is preserved by design.

2. **Better Consistency**: RAG finds ALL references to a term, patches update them ALL at once.

3. **KLO-Question Alignment**: RAG retrieves questions BY their KLO reference, ensures they stay coupled.

4. **Lower Token Usage**: Only retrieve and process chunks that need changes.

5. **Auditability**: Can see exact patches applied, easy to debug.

6. **Reversibility**: Can undo patches if validation fails.

---

## Appendix: Comparison Table

| Aspect | Current (Shard-Based) | Proposed (JSON Whisperer + RAG) |
|--------|----------------------|----------------------------------|
| **Approach** | Regenerate entire shards | Generate minimal patches |
| **Structure Risk** | LLM can break structure | Structure guaranteed |
| **Content Length** | Can explode | Fixed (patches only replace) |
| **Consistency** | Each shard independent | RAG finds all references |
| **Token Usage** | High (~100K) | Low (~36K) |
| **Complexity** | Medium | Higher (need RAG infra) |
| **Debugging** | Hard (full regeneration) | Easy (see exact patches) |
| **KLO Alignment** | Weak coupling | Strong (query by KLO ref) |

---

## Next Steps

1. **Prototype**: Build minimal JSON Whisperer with 1 change type (company name)
2. **Validate**: Test on sample_main.json, measure token usage and accuracy
3. **Expand**: Add question patches, resource patches, etc.
4. **Integrate**: Replace current simple_adapter.py
5. **Deploy**: Test on Heroku with real scenarios

---

*Report generated: 2026-01-29*
*Author: Claude Code*
