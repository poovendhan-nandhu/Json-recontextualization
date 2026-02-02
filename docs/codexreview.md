Below is a focused review of your **Simple Adapter** prompt logic (build_simple_prompt) and the failures it caused, plus concrete prompt rewrites and code guardrails. I’m treating this as the **adapter prompt** (not the validators).

---

## 1. Clarifying Questions (if any)
None needed. I can proceed with the current context.

---

## 2. Diagnosis (with categorized failures)

### A) Instruction conflicts **inside your adapter prompt**
**Root cause:** The prompt tells the model to invent sources and people, while your scenario_prompt says “use only simulation data.”

- **Conflict: “Use only simulation’s provided data” vs “real citations (e.g., McKinsey 2024)”**  
  Prompt excerpt:  
  > “Resources: 500-1500 words with real citations (e.g., ‘Source: McKinsey 2024’)”  
  Output example (hallucinated):  
  > “McKinsey 2024”, “Sustainable Apparel Coalition (2023)”, “U.S. Department of Commerce framework”  
  These violate your scenario constraint and are **explicitly induced** by the prompt.

- **Conflict: “scenario prompt is source of truth” vs “invent company identity + replace all names”**  
  Prompt excerpt:  
  > “You MUST invent these NOW...”  
  This forces invention even when the scenario prompt **might** already specify names, or when you need to **preserve existing names** in locked shards.

### B) Structure drift enabled by prompt (locked sections not protected)
- The prompt doesn’t mention locked paths. As a result, the model rewrites fields that should remain unchanged (e.g., `scenarioOptions`).
- Example drift from your output:  
  Input:  
  ```json
  "selectedScenarioOption": {"id": "", "option": "", "recommendedTasks": []}
  ```  
  Output:  
  ```json
  "selectedScenarioOption": {"id": "VA-US-ENTRY-2024", "option": "...", "recommendedTasks": [...]}
  ```  
  This is **structure/meaning drift** from empty → filled.

### C) Consistency errors encouraged by prompt
- You require “ONE manager name” but also instruct to “replace ALL names” and “invent names” while not enforcing **consistent role titles**.  
- Output shows **Marcus Rivera** with multiple titles (“Senior Strategy Director”, “Director of Global Strategy”, “Senior Strategy Manager”).  
- The prompt lacks an explicit “lock the manager’s role title everywhere.”

### D) KLO preservation bug (code-level, prompt-level)
- You extract KLOs, but **build_simple_prompt doesn’t accept them** and therefore doesn’t enforce them.  
  In `_adapt_monolithic` you call:  
  ```python
  build_simple_prompt(..., klos_text=klos_text)
  ```  
  But the function signature **does not include `klos_text`**, so KLOs are ignored.  
- This explains KLO alignment failures in validation.

### E) “Real citations” instruction causes hallucination
- You **force** 500–1500 words + citations, causing over-generation and invented sources.  
- This is a direct prompt-level cause of your “factuality” validator failures.

---

## 3. Prompt Rewrites (3 versions)

> These are **drop-in replacements** for the `build_simple_prompt` text.  
> Each version includes: root-type constraint, JSON-only, structure/type preservation, citation handling, locked paths.

---

### (1) Minimal Fix Prompt
```
You are adapting a business simulation JSON to match a new scenario.

SCENARIO PROMPT (source of truth):
{scenario_prompt}

LOCKED PATHS (DO NOT CHANGE VALUES):
- topicWizardData.scenarioOptions
- topicWizardData.workspaceIds

CRITICAL OUTPUT RULES:
- Output must be a single JSON object (same root type).
- Output JSON ONLY (no markdown, no extra text).
- Preserve all keys, array lengths, and value types.
- Do NOT add or remove keys/objects/arrays.
- Only update string content where needed.

CITATIONS / SOURCES:
- Do NOT invent sources, citations, or stats.
- Only use sources that already exist in the input JSON or scenario_prompt.
- If none exist, omit citations entirely.

CONSISTENCY:
- Use one company name, one manager name, one manager role throughout.
- If scenario_prompt provides names, use them exactly.
- If not provided, keep existing names from input JSON.

CONTENT TO ADAPT:
{shard_hint}
```json
{content_str}
```

Return ONLY the adapted JSON object.
```

---

### (2) Structured Version (explicit invariants + locked paths)
```
ROLE: JSON adapter for domain transformation.

[ROOT TYPE]
- Output must be a single JSON object (same root type).

[JSON-ONLY]
- Output JSON only. No markdown, no commentary, no extra objects.

[LOCKED PATHS — DO NOT CHANGE VALUES]
- topicWizardData.scenarioOptions
- topicWizardData.workspaceIds

[STRUCTURE + TYPE INVARIANTS]
- Keep all keys exactly the same.
- Keep arrays the same length and order.
- Keep IDs unchanged.
- Do not add/remove/rename keys.

[ALLOWED CHANGES]
- Only update string content to match the scenario_prompt.

[CONSISTENCY]
- One company name, one manager name, one manager role across all fields.
- If scenario_prompt does not provide names, preserve existing names in JSON.

[SOURCES / CITATIONS]
- No invented citations.
- Only use sources explicitly present in input JSON or scenario_prompt.

SCENARIO PROMPT:
{scenario_prompt}

CONTENT TO ADAPT:
{shard_hint}
```json
{content_str}
```

Return ONLY the adapted JSON object.
```

---

### (3) Robust Version (guardrails + self-check)
```
You are a strict JSON transformer.

HARD RULES:
1) Output must be a single JSON object (same root type).
2) JSON only — no markdown, no extra text.
3) Preserve all keys, array lengths, and types.
4) Preserve all IDs exactly.
5) DO NOT change values at locked paths:
   - topicWizardData.scenarioOptions
   - topicWizardData.workspaceIds
6) Only update string values to align with the scenario_prompt.
7) Do NOT invent citations or sources.

CONSISTENCY:
- One company name, one manager name, one manager role throughout.
- If scenario_prompt doesn’t provide names, keep existing names.

SELF-CHECK BEFORE OUTPUT:
- Root type correct ✅
- Keys unchanged ✅
- Arrays same length ✅
- No new citations ✅
- Locked paths unchanged ✅

If unsure, return the original JSON unchanged.

SCENARIO PROMPT:
{scenario_prompt}

CONTENT:
{shard_hint}
```json
{content_str}
```

Return ONLY the JSON object.
```

---

## 4. Why These Work (changes + tradeoffs)

### Minimal Fix
- **Changes:** Adds locked paths, “no invented citations,” keep existing names if not provided.  
- **Why it helps:** Stops hallucinated sources and prevents drift in locked sections.  
- **Tradeoff:** Less explicit about consistency role titles.

### Structured Version
- **Changes:** Formal invariants + locked paths + citation policy.  
- **Why it helps:** Better schema preservation and reduces inconsistencies.  
- **Tradeoff:** More verbose.

### Robust Version
- **Changes:** Adds self-check + “return original JSON if unsure.”  
- **Why it helps:** Prevents partial corruption and forced hallucinations.  
- **Tradeoff:** May under-adapt if model is uncertain.

---

## 5. Test Plan + Rubric

### 5 Test Cases
1) **No-source scenario prompt** → verify no new citations.  
2) **Scenario prompt with explicit names** → ensure names preserved exactly.  
3) **Locked path content** in `scenarioOptions` remains unchanged.  
4) **HTML-heavy fields** (emails) remain valid JSON.  
5) **Consistency test**: verify manager role is identical across all sections.

### Scoring Rubric (0–2 each)
- JSON validity  
- Structure preservation  
- Domain fidelity  
- Consistency (names/roles)  
- Completeness (no placeholders/truncation)

---

## 6. Recommended Code Guardrails

### 🔧 Code-level fixes you should make

**(1) Fix KLO preservation bug**  
`build_simple_prompt` is missing `klos_text`. Add it:

```python
def build_simple_prompt(..., klos_text: str = "") -> str:
    ...
    return f"""...
{klos_text}
...
"""
```

**(2) Remove hallucination trigger**  
Delete or conditionally disable this line in the prompt:  
> “Resources: 500-1500 words with real citations (e.g., ‘Source: McKinsey 2024’)”

**(3) Prevent “the company” fallback**  
In `extract_company_context`, if company/manager not found, return **None** and do **not** lock names:

```python
if not company_name or company_name in ["the company"]:
    company_name = ""
```

Then only include `company_section` if you have real values.

**(4) Add locked paths to adapter**  
Pass locked paths into prompt or exclude them in sharding.

**(5) Add schema diff checks after adaptation**  
- key set diff  
- array length diff  
- locked path equality check  
- detection of new citations (regex blacklist)

---

If you want, I can also rewrite your **validation agent prompts** directly (Domain/Context/Consistency/Completeness) to reduce false positives and align with your scenario constraints.