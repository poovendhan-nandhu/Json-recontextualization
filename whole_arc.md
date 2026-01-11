
⏺ Now I understand correctly. Let me lay out the complete architecture:

  ---
  CARTEDO SIMULATION ADAPTATION FRAMEWORK - FINAL ARCHITECTURE

  The Flow

  ┌─────────────────────────────────────────────────────────────────────────┐
  │                              INPUTS                                      │
  ├─────────────────────────────────────────────────────────────────────────┤
  │                                                                          │
  │  ┌──────────────────────────┐    ┌──────────────────────────────────┐   │
  │  │  BASE SIMULATION JSON    │    │  NEW SCENARIO PROMPT             │   │
  │  │  (sample_main.json)      │    │  "Gen Z organic T-shirts brand,  │   │
  │  │                          │    │   market entry analysis..."      │   │
  │  │  - Structure with IDs    │    │                                  │   │
  │  │  - KLOs, Criteria        │    │  - Industry context              │   │
  │  │  - Simulation Flow       │    │  - Role description              │   │
  │  │  - Emails, Resources     │    │  - Task requirements             │   │
  │  └──────────────────────────┘    └──────────────────────────────────┘   │
  │               │                              │                           │
  │               └──────────────┬───────────────┘                           │
  │                              ▼                                           │
  └─────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                     STAGE 1: INGESTION & SHARDING                        │
  ├─────────────────────────────────────────────────────────────────────────┤
  │                                                                          │
  │  1. Parse base simulation JSON                                           │
  │  2. Extract and CHUNK into independent shards:                           │
  │                                                                          │
  │     ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐     │
  │     │   SHARD 1   │ │   SHARD 2   │ │   SHARD 3   │ │   SHARD 4   │     │
  │     │    KLOs     │ │  Scenario   │ │  Sim Flow   │ │  Resources  │     │
  │     │  Criteria   │ │ Background  │ │   Emails    │ │   Rubrics   │     │
  │     └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘     │
  │           │               │               │               │              │
  │     ┌─────┴─────┐   ┌─────┴─────┐   ┌─────┴─────┐   ┌─────┴─────┐       │
  │     │  hash_1   │   │  hash_2   │   │  hash_3   │   │  hash_4   │       │
  │     │  lock=NO  │   │  lock=NO  │   │  lock=NO  │   │  lock=NO  │       │
  │     └───────────┘   └───────────┘   └───────────┘   └───────────┘       │
  │                                                                          │
  │  3. Store shards in VECTOR DB (ChromaDB) with embeddings                 │
  │                                                                          │
  └─────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                  STAGE 2: PARALLEL GENERATION (RAG-Assisted)             │
  ├─────────────────────────────────────────────────────────────────────────┤
  │                                                                          │
  │  Process chunks IN PARALLEL using the new scenario prompt:               │
  │                                                                          │
  │  ┌─────────────────────────────────────────────────────────────────┐    │
  │  │                    NEW PROMPT (shared context)                   │    │
  │  │     "Gen Z organic T-shirts brand, market entry analysis..."     │    │
  │  └─────────────────────────────────────────────────────────────────┘    │
  │           │               │               │               │              │
  │           ▼               ▼               ▼               ▼              │
  │     ┌───────────┐   ┌───────────┐   ┌───────────┐   ┌───────────┐       │
  │     │ Generate  │   │ Generate  │   │ Generate  │   │ Generate  │       │
  │     │   KLOs    │   │ Scenario  │   │ Sim Flow  │   │ Resources │       │
  │     │  (GPT-4)  │   │  (GPT-4)  │   │  (GPT-4)  │   │  (GPT-4)  │       │
  │     └───────────┘   └───────────┘   └───────────┘   └───────────┘       │
  │           │               │               │               │              │
  │           ▼               ▼               ▼               ▼              │
  │     ┌───────────┐   ┌───────────┐   ┌───────────┐   ┌───────────┐       │
  │     │  Shard 1' │   │  Shard 2' │   │  Shard 3' │   │  Shard 4' │       │
  │     │(generated)│   │(generated)│   │(generated)│   │(generated)│       │
  │     └───────────┘   └───────────┘   └───────────┘   └───────────┘       │
  │                                                                          │
  │  RAG retrieves relevant base structure for each chunk                    │
  │  IDs PRESERVED throughout generation                                     │
  │                                                                          │
  └─────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                   STAGE 3: ALIGNMENT CHECKER                             │
  ├─────────────────────────────────────────────────────────────────────────┤
  │                                                                          │
  │  Check CROSS-SHARD alignment (the critical part!):                       │
  │                                                                          │
  │  ┌─────────────────────────────────────────────────────────────────┐    │
  │  │                     ALIGNMENT MATRIX                             │    │
  │  ├─────────────────┬─────────────────┬─────────────────────────────┤    │
  │  │ KLOs            │ ←→ Submission   │ Do questions assess KLOs?   │    │
  │  │                 │    Questions    │                             │    │
  │  ├─────────────────┼─────────────────┼─────────────────────────────┤    │
  │  │ KLOs            │ ←→ Resources    │ Do resources support KLOs?  │    │
  │  ├─────────────────┼─────────────────┼─────────────────────────────┤    │
  │  │ Scenario        │ ←→ Emails       │ Is context consistent?      │    │
  │  ├─────────────────┼─────────────────┼─────────────────────────────┤    │
  │  │ Scenario        │ ←→ Resources    │ Industry-specific data?     │    │
  │  ├─────────────────┼─────────────────┼─────────────────────────────┤    │
  │  │ Role            │ ←→ Tasks        │ Tasks match role?           │    │
  │  └─────────────────┴─────────────────┴─────────────────────────────┘    │
  │                                                                          │
  │  OUTPUT: Alignment Score per pair + Overall Score                        │
  │  TARGET: ≥ 98% alignment                                                 │
  │                                                                          │
  └─────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                    STAGE 4: SCOPED VALIDATION                            │
  ├─────────────────────────────────────────────────────────────────────────┤
  │                                                                          │
  │  Validate EACH SHARD independently (not full JSON!):                     │
  │                                                                          │
  │  ┌───────────────────────────────────────────────────────────────────┐  │
  │  │  SHARD 1 (KLOs)           │  SHARD 2 (Scenario)                   │  │
  │  │  ✓ Structure intact       │  ✓ Structure intact                   │  │
  │  │  ✓ IDs preserved          │  ✓ IDs preserved                      │  │
  │  │  ✓ Content valid          │  ✓ Domain fidelity                    │  │
  │  │  ✓ Criteria complete      │  ✓ Industry terms correct             │  │
  │  └───────────────────────────────────────────────────────────────────┘  │
  │  ┌───────────────────────────────────────────────────────────────────┐  │
  │  │  SHARD 3 (Sim Flow)       │  SHARD 4 (Resources)                  │  │
  │  │  ✓ Stage structure ok     │  ✓ Data realistic                     │  │
  │  │  ✓ Email format valid     │  ✓ Numbers make sense                 │  │
  │  │  ✓ Activities complete    │  ✓ Context matches scenario           │  │
  │  └───────────────────────────────────────────────────────────────────┘  │
  │                                                                          │
  │  Each shard validated ONLY against its own rules                         │
  │  NO reprocessing of full JSON                                            │
  │                                                                          │
  └─────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
                      ┌─────────────────────┐
                      │  All Checks Pass?   │
                      └─────────────────────┘
                        │               │
                       YES              NO
                        │               │
                        ▼               ▼
  ┌─────────────────────────┐  ┌─────────────────────────────────────────────┐
  │    STAGE 5: MERGE       │  │        STAGE 4B: SCOPED FIXERS              │
  ├─────────────────────────┤  ├─────────────────────────────────────────────┤
  │                         │  │                                             │
  │  Reassemble shards      │  │  Fix ONLY the failing shard:               │
  
  │  into final JSON        │  │                                             │
  │                         │  │  ┌─────────────────────────────────────┐   │
  │  Preserve all IDs       │  │  │  If STRUCTURE issue → Structural Fix│   │
  │  Verify hash integrity  │  │  │  If SEMANTIC issue → Semantic Fix   │   │
  │                         │  │  │  If ALIGNMENT issue → Re-generate   │   │
  │                         │  │  └─────────────────────────────────────┘   │
  │                         │  │                                             │
  │                         │  │  Max 3 attempts per shard                   │
  │                         │  │  Then → Flag for human                      │
  │                         │  │                                             │
  └─────────────────────────┘  └─────────────────────────────────────────────┘
            │                              │
            │                              │ (loop back to validation)
            │                              │
            ▼                              
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                      STAGE 6: HUMAN APPROVAL                             │
  ├─────────────────────────────────────────────────────────────────────────┤
  │                                                                          │
  │  Human reviews:                                                          │
  │  • Does it feel realistic for the industry?                              │
  │  • Would students learn the right skills?                                │
  │  • Is alignment between sections logical?                                │
  │                                                                          │
  │  [APPROVE] → Ship to production                                          │
  │  [REJECT]  → Feedback for improvement                                    │
  │                                                                          │
  └─────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                            OUTPUT                                        │
  ├─────────────────────────────────────────────────────────────────────────┤
  │                                                                          │
  │  GOLDEN ADAPTED SIMULATION JSON                                          │
  │  • Same structure as base (IDs preserved)                                │
  │  • New context from prompt                                               │
  │  • KLOs ↔ Questions ↔ Resources ALIGNED                                  │
  │  • Validation score ≥ 98%                                                │
  │                                                                          │
  └─────────────────────────────────────────────────────────────────────────┘

  ---
  FILE STRUCTURE

  src/
  ├── main.py                              # FastAPI entry point
  ├── api/
  │   ├── routes.py                        # /generate, /validate endpoints
  │   └── approval_routes.py               # /approve, /reject endpoints
  │
  ├── stages/
  │   ├── __init__.py
  │   ├── ingestor.py                      # Stage 1: Parse & shard base JSON
  │   ├── sharder.py                       # Split JSON into chunks with hashes
  │   ├── generator.py                     # Stage 2: Parallel generation
  │   ├── alignment_checker.py             # Stage 3: Cross-shard alignment
  │   ├── validator.py                     # Stage 4: Scoped validation per shard
  │   ├── fixer.py                         # Stage 4B: Structural/Semantic fixes
  │   ├── merger.py                        # Stage 5: Reassemble shards
  │   └── human_approval.py                # Stage 6: Approval API
  │
  ├── rag/
  │   ├── __init__.py
  │   ├── vector_store.py                  # ChromaDB setup
  │   ├── embeddings.py                    # OpenAI embeddings
  │   ├── indexer.py                       # Index base simulation chunks
  │   └── retriever.py                     # Retrieve relevant chunks
  │
  ├── validators/
  │   ├── __init__.py
  │   ├── base.py                          # BaseValidator ABC
  │   ├── structure_validator.py           # Check IDs, keys, schema
  │   ├── alignment_validator.py           # Check KLO↔Questions↔Resources
  │   ├── domain_validator.py              # Check industry terms, KPIs
  │   └── content_validator.py             # Check completeness, format
  │
  ├── models/
  │   ├── schemas.py                       # Pydantic models
  │   ├── shard.py                         # Shard dataclass
  │   └── alignment.py                     # Alignment matrix models
  │
  ├── graph/
  │   ├── state.py                         # LangGraph state
  │   └── workflow.py                      # LangGraph workflow definition
  │
  └── utils/
      ├── config.py                        # Configuration
      ├── hash.py                          # SHA-256 utilities
      ├── openai_client.py                 # OpenAI wrapper
      └── helpers.py                       # Utility functions

  ---
  SHARD DEFINITIONS (from sample_main.json)

  SHARD_DEFINITIONS = {
      "metadata": {
          "paths": [
              "topicWizardData.id",
              "topicWizardData.workspace",
              "topicWizardData.lessonInformation"
          ],
          "locked": True,  # IDs never change
          "parallel": False
      },

      "assessment": {
          "paths": [
              "topicWizardData.assessmentCriterion",
              "topicWizardData.selectedAssessmentCriterion"
          ],
          "locked": False,  # Content can change, structure locked
          "parallel": True,
          "aligns_with": ["submission_questions", "resources"]
      },

      "scenario": {
          "paths": [
              "topicWizardData.scenarioOptions",
              "topicWizardData.selectedScenarioOption",
              "topicWizardData.chatHistory.scenarioDescription"
          ],
          "locked": False,
          "parallel": True,
          "aligns_with": ["simulation_flow", "resources"]
      },

      "simulation_flow": {
          "paths": [
              "topicWizardData.simulationFlow"
          ],
          "locked": False,
          "parallel": True,
          "contains": ["emails", "resources", "activities", "rubrics"]
      },

      "launch_settings": {
          "paths": [
              "topicWizardData.launchSettings"
          ],
          "locked": False,
          "parallel": True
      },

      "industry_activities": {
          "paths": [
              "topicWizardData.industryAlignedActivities",
              "topicWizardData.selectedIndustryAlignedActivities"
          ],
          "locked": False,
          "parallel": True
      }
  }

  ---
  ALIGNMENT RULES (Critical for 98% accuracy)

  ALIGNMENT_RULES = {
      "klo_to_questions": {
          "source": "assessmentCriterion[*].keyLearningOutcome",
          "target": "simulationFlow[*].data.questions",
          "rule": "Each KLO must have at least one question assessing it",
          "weight": 2.0,  # High priority
          "blocker": True
      },

      "klo_to_resources": {
          "source": "assessmentCriterion[*].keyLearningOutcome",
          "target": "simulationFlow[*].data.resource",
          "rule": "Resources must support learning outcomes",
          "weight": 1.5,
          "blocker": True
      },

      "scenario_to_emails": {
          "source": "chatHistory.scenarioDescription[*].message.background",
          "target": "simulationFlow[*].data.introEmail",
          "rule": "Email context must match scenario organization",
          "weight": 1.0,
          "blocker": False
      },

      "scenario_to_resources": {
          "source": "chatHistory.scenarioDescription[*].message.background.organizationName",
          "target": "simulationFlow[*].data.resource",
          "rule": "Resource data must reference correct organization",
          "weight": 1.5,
          "blocker": True
      },

      "role_to_tasks": {
          "source": "chatHistory.scenarioDescription[*].message.learnerRoleReportingManager.learnerRole",
          "target": "simulationFlow[*].data.activity",
          "rule": "Tasks must be appropriate for learner role",
          "weight": 1.0,
          "blocker": False
      }
  }

  ---
  RAG IMPLEMENTATION

  # rag/vector_store.py

  class SimulationVectorStore:
      def __init__(self):
          self.client = chromadb.PersistentClient(path="./chroma_db")
          self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

          # Collections for different chunk types
          self.collections = {
              "scenarios": self.client.get_or_create_collection("scenarios"),
              "klos": self.client.get_or_create_collection("klos"),
              "resources": self.client.get_or_create_collection("resources"),
              "emails": self.client.get_or_create_collection("emails"),
              "activities": self.client.get_or_create_collection("activities")
          }

      def index_simulation(self, simulation_json: dict, simulation_id: str):
          """Index a base simulation into vector DB for RAG retrieval"""

          # Extract and embed each shard type
          for shard_name, shard_config in SHARD_DEFINITIONS.items():
              content = extract_by_paths(simulation_json, shard_config["paths"])
              embedding = self.embeddings.embed(json.dumps(content))

              self.collections[shard_name].add(
                  ids=[f"{simulation_id}_{shard_name}"],
                  embeddings=[embedding],
                  documents=[json.dumps(content)],
                  metadatas=[{"simulation_id": simulation_id, "shard": shard_name}]
              )

      def retrieve_similar(self, prompt: str, shard_type: str, k: int = 3):
          """Retrieve similar chunks for generation context"""

          query_embedding = self.embeddings.embed(prompt)
          results = self.collections[shard_type].query(
              query_embeddings=[query_embedding],
              n_results=k
          )
          return results

  ---
  PARALLEL GENERATION

  # stages/generator.py

  async def generate_parallel(
      base_shards: dict[str, Shard],
      new_prompt: str,
      vector_store: SimulationVectorStore
  ) -> dict[str, Shard]:
      """Generate all shards in parallel"""

      tasks = []

      for shard_name, shard in base_shards.items():
          if SHARD_DEFINITIONS[shard_name]["parallel"]:
              # Create generation task for this shard
              task = generate_shard(
                  shard_name=shard_name,
                  base_shard=shard,
                  new_prompt=new_prompt,
                  vector_store=vector_store
              )
              tasks.append(task)

      # Run all in parallel
      results = await asyncio.gather(*tasks)

      return {r.name: r for r in results}


  async def generate_shard(
      shard_name: str,
      base_shard: Shard,
      new_prompt: str,
      vector_store: SimulationVectorStore
  ) -> Shard:
      """Generate a single shard using RAG context"""

      # 1. Retrieve similar examples from vector DB
      similar = vector_store.retrieve_similar(new_prompt, shard_name, k=3)

      # 2. Build prompt with base structure + similar examples + new prompt
      system_prompt = f"""
      You are recontextualizing a simulation {shard_name} section.
      
      RULES:
      - Preserve ALL IDs exactly as they are
      - Keep the same JSON structure
      - Only change content to match the new scenario
      - Maintain professional, instructional tone
      
      BASE STRUCTURE (preserve this):
      {json.dumps(base_shard.content, indent=2)}
      
      SIMILAR EXAMPLES FOR CONTEXT:
      {json.dumps(similar, indent=2)}
      """

      user_prompt = f"""
      NEW SCENARIO:
      {new_prompt}
      
      Generate the {shard_name} section for this new scenario.
      Return valid JSON only.
      """

      # 3. Generate
      result = await openai_client.generate_json(system_prompt, user_prompt)

      # 4. Create new shard with preserved IDs
      new_shard = Shard(
          name=shard_name,
          content=result,
          original_ids=base_shard.ids,  # Preserve!
          hash=compute_hash(result)
      )

      return new_shard

  ---
  IMPLEMENTATION ORDER

  | Phase   | Tasks                           | Priority |
  |---------|---------------------------------|----------|
  | Phase 1 | Sharder + Shard dataclass       | HIGH     |
  | Phase 2 | RAG setup (ChromaDB + indexing) | HIGH     |
  | Phase 3 | Parallel generator              | HIGH     |
  | Phase 4 | Alignment checker               | CRITICAL |
  | Phase 5 | Scoped validators               | HIGH     |
  | Phase 6 | Fixers (structural + semantic)  | MEDIUM   |
  | Phase 7 | Merger + final output           | MEDIUM   |
  | Phase 8 | Human approval API              | LOW      |

  ---
  SUCCESS CRITERIA

  | Metric                | Target       |
  |-----------------------|--------------|
  | Alignment Score       | ≥ 98%        |
  | KLO ↔ Questions Match | 100%         |
  | IDs Preserved         | 100%         |
  | Structure Intact      | 100%         |
  | Generation Time       | < 60 seconds |
  | Validation Time       | < 10 seconds |

  ---
  Is this the architecture you want to implement?


















⏺ The sharder is working. Let me update the todos and provide a summary:

⏺ Phase 1 Complete! Here's what was built:

  Files Created

  src/
  ├── models/
  │   ├── __init__.py          # Updated with exports
  │   └── shard.py             # NEW - Shard, ShardCollection, ValidationResult
  ├── stages/
  │   ├── __init__.py          # NEW - Stage module
  │   └── sharder.py           # NEW - JSON to shards parser
  └── utils/
      ├── config.py            # Updated with SHARD_DEFINITIONS (13 shards)
      └── hash.py              # NEW - Hash & path utilities

  Test Results

  Total shards: 13
  ├── Locked shards: 4 (metadata, assessment_criteria, scenario_options, industry_activities)
  ├── Unlocked shards: 9 (can be recontextualized)
  ├── Blocker shards: 7 (must pass 100% validation)
  └── Total IDs extracted: 170

  ✅ Merge successful - shards reassemble into original JSON

  Shard Breakdown

  | Shard                 | Locked | Blocker | Aligns With                                    |
  |-----------------------|--------|---------|------------------------------------------------|
  | metadata              | ✅     | ✅      | -                                              |
  | assessment_criteria   | ✅     | ✅      | simulation_flow, rubrics                       |
  | scenario_options      | ✅     | ✅      | -                                              |
  | industry_activities   | ✅     | ❌      | -                                              |
  | selected_scenario     | ❌     | ✅      | workplace_scenario, simulation_flow            |
  | workplace_scenario    | ❌     | ✅      | selected_scenario, simulation_flow, emails     |
  | scenario_chat_history | ❌     | ❌      | workplace_scenario                             |
  | simulation_flow       | ❌     | ✅      | workplace_scenario, emails, rubrics, resources |
  | emails                | ❌     | ❌      | workplace_scenario, simulation_flow            |
  | rubrics               | ❌     | ✅      | assessment_criteria, simulation_flow           |
  | resources             | ❌     | ❌      | simulation_flow, workplace_scenario            |
  | launch_settings       | ❌     | ❌      | workplace_scenario                             |





  ┌──────────────────────────┬─────────────────────────────────────────────────────────┬───────────┐
  │          Issue           │                           Fix                           │  Effort   │
  ├──────────────────────────┼─────────────────────────────────────────────────────────┼───────────┤
  │ KLO questions incomplete │ Enhance KLO Alignment Fixer prompt to be more specific  │ 2-3 hours │
  ├──────────────────────────┼─────────────────────────────────────────────────────────┼───────────┤
  │ Template truncation      │ Increase Gemini max_tokens or add truncation detection  │ 1 hour    │
  ├──────────────────────────┼─────────────────────────────────────────────────────────┼───────────┤
  │ All scores capped at 90% │ Alignment checker may be too strict - review thresholds │ 1-2 hours │
  └──────────────────────────┴─────────────────────────────────────────────────────────┴───────────┘
  ---




  