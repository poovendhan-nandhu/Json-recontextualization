
⏺ Let me outline the approach and then implement it:

⏺ Here's the approach:

  ┌─────────────────────────────────────────────────────────────────┐
  │                   ADAPTATION ENGINE FLOW                        │
  ├─────────────────────────────────────────────────────────────────┤
  │                                                                 │
  │  INPUT: base_simulation.json + target_scenario_index (0-36)     │
  │                           │                                     │
  │                           ▼                                     │
  │  ┌─────────────────────────────────────────────────────────┐   │
  │  │  1. ENTITY EXTRACTION                                    │   │
  │  │     - Regex: Find company names, person names            │   │
  │  │     - NLP: Extract KPIs, metrics, industry terms         │   │
  │  │     - From: current selectedScenarioOption               │   │
  │  └─────────────────────────────────────────────────────────┘   │
  │                           │                                     │
  │                           ▼                                     │
  │  ┌─────────────────────────────────────────────────────────┐   │
  │  │  2. TARGET ANALYSIS                                      │   │
  │  │     - Parse scenarioOptions[target_index]                │   │
  │  │     - Extract: company, industry, challenge              │   │
  │  │     - Detect industry type (hospitality, airline, etc.)  │   │
  │  └─────────────────────────────────────────────────────────┘   │
  │                           │                                     │
  │                           ▼                                     │
  │  ┌─────────────────────────────────────────────────────────┐   │
  │  │  3. BUILD ENTITY MAP                                     │   │
  │  │     {                                                    │   │
  │  │       "HarvestBowls": "BlueHaven Hotels",               │   │
  │  │       "foot traffic": "occupancy rate",                  │   │
  │  │       "Alex Chen": "Jordan Smith",                       │   │
  │  │       ...                                                │   │
  │  │     }                                                    │   │
  │  └─────────────────────────────────────────────────────────┘   │
  │                           │                                     │
  │                           ▼                                     │
  │  ┌─────────────────────────────────────────────────────────┐   │
  │  │  4. RAG CONTEXT RETRIEVAL                                │   │
  │  │     - Query: "{industry} KPIs metrics terminology"       │   │
  │  │     - Get: realistic values, industry-specific terms     │   │
  │  └─────────────────────────────────────────────────────────┘   │
  │                           │                                     │
  │                           ▼                                     │
  │  ┌─────────────────────────────────────────────────────────┐   │
  │  │  5. LLM TRANSFORMATION (OpenAI)                          │   │
  │  │     - System: "You are adapting, not generating"         │   │
  │  │     - Input: entity_map + rag_context + shard_content    │   │
  │  │     - Output: transformed content (same structure)       │   │
  │  └─────────────────────────────────────────────────────────┘   │
  │                           │                                     │
  │                           ▼                                     │
  │  ┌─────────────────────────────────────────────────────────┐   │
  │  │  6. LOCK GUARD                                           │   │
  │  │     - Force-restore all FULLY_LOCKED fields              │   │
  │  │     - Verify structure integrity                         │   │
  │  └─────────────────────────────────────────────────────────┘   │
  │                           │                                     │
  │                           ▼                                     │
  │  OUTPUT: adapted_json + entity_map + industry_context           │
  │                                                                 │
  └─────────────────────────────────────────────────────────────────┘

  Let me implement this now: