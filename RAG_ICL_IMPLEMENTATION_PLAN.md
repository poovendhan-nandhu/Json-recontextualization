# RAG + ICL Fix: Simple Summary

## The Problem

Collections were **EMPTY** → Retrieval returned **NOTHING** → No examples in prompts

```
BEFORE:
Input → Adapt (tries to retrieve but collections EMPTY) → Output (misaligned)
```

## The Fix (20 lines)

**Added indexing step in `adaptation_engine.py` (line 209-230):**

```python
# INDEX THE INPUT for RAG retrieval
if self.use_per_shard_rag:
    retriever = SimulationRetriever()
    index_result = retriever.index_simulation_by_shard_type(
        simulation_id="base_input",
        shards=collection.shards,
        industry=industry,
        clear_existing=True,
    )
```

## New Flow

```
Input → INDEX IT → Adapt (retrieves from indexed input) → Output (aligned)
            ↓
    Collections now populated!
```

## Why It Works

1. **Input = Golden Example** - Base simulation is already well-aligned
2. **Index before adapt** - Put KLOs, questions, resources into ChromaDB
3. **Retrieve during adapt** - LLM sees patterns from original
4. **Output follows pattern** - Maintains alignment

## Logs to Check

```
Indexed input for RAG: {'klos': 1, 'activities': 4, 'resources': 1, ...}
RAG retrieval: 5/8 shards got examples
```

## Files Changed

- `src/stages/adaptation_engine.py` - Added indexing step + fixed industry extraction

That's it. No external golden examples needed. The input IS the reference.
