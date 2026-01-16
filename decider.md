⏺ Great question! Here's how the LLM decides what to change:

  Step 1: Filter BEFORE LLM

  First, the code filters out things that should never be changed:

  # decider.py:524-552
  def should_modify_leaf(path: str, value: Any, scenario_context: str) -> bool:

      # Skip non-strings (numbers, booleans)
      if not isinstance(value, str):
          return False

      # Skip empty strings
      if not value.strip():
          return False

      # Skip technical stuff (IDs, URLs, tokens)
      if any(indicator in path.lower() for indicator in ['url', 'id', 'key', 'token']):
          return False

      # Skip alternative scenarios (shouldn't be changed)
      if 'scenarioOptions' in path:
          return False

      return True  # Send to LLM for decision

  ---
  Step 2: LLM Gets Context

  For each leaf that passes the filter, LLM receives:

  # decider.py:228-241 - The prompt sent to GPT:

  """
  **Scenario Context:**
  {scenario_context}  # e.g., "Convert to airline industry"

  **JSON Path:** /workplaceScenario/companyName

  **Current Value:** "HarvestBowls Restaurant"

  **Parent Context:**
  - Type: Object
  - Sibling keys: companyName, industry, competitor

  **Task:**
  Decide whether this value should be modified to fit the scenario.
  """

  ---
  Step 3: LLM Responds with Decision

  # decider.py:58-62 - Structured response:
  class DecisionResponse(BaseModel):
      action: Literal["keep", "replace"]  # Only 2 choices
      new_value: Optional[str] = None     # If replace, what to?
      reason: str                         # Why?

  Example LLM Response:
  {
    "action": "replace",
    "new_value": "SkyHigh Airlines",
    "reason": "Company name needs to reflect airline industry"
  }

  ---
  The Flow Visualized

  ┌─────────────────────────────────────────────────────────────────┐
  │                  HOW LLM DECIDES                                │
  ├─────────────────────────────────────────────────────────────────┤
  │                                                                 │
  │  538 Total Leaves                                               │
  │        │                                                        │
  │        ▼                                                        │
  │  ┌─────────────┐                                               │
  │  │   FILTER    │  Skip: IDs, URLs, tokens, empty, non-strings  │
  │  └──────┬──────┘                                               │
  │         │                                                       │
  │         ▼                                                       │
  │  274 Modifiable Leaves                                          │
  │         │                                                       │
  │         ▼                                                       │
  │  ┌─────────────────────────────────────────────────────────┐   │
  │  │                    FOR EACH LEAF                         │   │
  │  │                                                          │   │
  │  │  LLM receives:                                          │   │
  │  │  • Scenario: "Convert to airline industry"              │   │
  │  │  • Path: /workplaceScenario/companyName                 │   │
  │  │  • Value: "HarvestBowls Restaurant"                     │   │
  │  │  • Context: sibling keys, parent type                   │   │
  │  │                                                          │   │
  │  │  LLM thinks:                                            │   │
  │  │  "This is a company name, scenario is airline..."       │   │
  │  │  "Should change to an airline name"                     │   │
  │  │                                                          │   │
  │  │  LLM responds:                                          │   │
  │  │  { action: "replace", new_value: "SkyHigh Airlines" }   │   │
  │  │                                                          │   │
  │  └─────────────────────────────────────────────────────────┘   │
  │         │                                                       │
  │         ▼                                                       │
  │  Only changed leaves become PatchOps                            │
  │                                                                 │
  └─────────────────────────────────────────────────────────────────┘

  ---
  Example Decisions
  ┌───────────────┬─────────────────────────┬───────────────────────────────┬──────────────────────────────────┐
  │     Path      │          Value          │         LLM Decision          │               Why                │
  ├───────────────┼─────────────────────────┼───────────────────────────────┼──────────────────────────────────┤
  │ /companyName  │ "HarvestBowls"          │ REPLACE → "SkyHigh Airlines"  │ Company name must match scenario │
  ├───────────────┼─────────────────────────┼───────────────────────────────┼──────────────────────────────────┤
  │ /lesson       │ "Analyze market trends" │ KEEP                          │ Generic, works for any industry  │
  ├───────────────┼─────────────────────────┼───────────────────────────────┼──────────────────────────────────┤
  │ /email        │ "sarah@harvest.com"     │ REPLACE → "sarah@skyhigh.com" │ Email domain must match          │
  ├───────────────┼─────────────────────────┼───────────────────────────────┼──────────────────────────────────┤
  │ /criterion/id │ "abc-123"               │ SKIP (filtered)               │ Never sent to LLM                │
  ├───────────────┼─────────────────────────┼───────────────────────────────┼──────────────────────────────────┤
  │ /kpiValue     │ "15%"                   │ KEEP                          │ Number is reasonable             │
  └───────────────┴─────────────────────────┴───────────────────────────────┴──────────────────────────────────┘
  ---
  The System Prompt (Guidelines for LLM)

  # decider.py:217-226
  system_prompt = """
  You are a JSON contextualization expert.

  Guidelines:
  - ONLY modify values that directly contradict the scenario
  - Preserve factual accuracy and structural integrity
  - Keep proper nouns, URLs, IDs unchanged unless part of scenario
  - Maintain same style and tone as original
  - Be conservative: when in doubt, keep the original value
  """

  The LLM is told to be conservative - only change what's necessary!

─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
❯ can we run it with llm mode                                                                                     ↵ send
─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  ? for shortcuts                                                                           ⧉ In domain_validation.json




