# KLO Alignment Fixer - Clear Implementation Plan

## Goal
Improve KLO → Questions alignment from **90% to 95%+** by rewriting questions to match KLO terminology.

---

## The Problem

**KLO says:**
> "Develop a weighted evaluation framework for market entry options"

**Question says:**
> "What factors did you consider?"

**Alignment checker:** These don't match! Score: 72%

---

## The Solution

Rewrite the question to use **SAME TERMINOLOGY** as KLO:

**Fixed question:**
> "Design a weighted evaluation framework with criteria for assessing market entry options"

**Alignment checker:** Now they match! Score: 95%+

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         ADAPTED JSON                              │
│  ┌─────────────────────┐        ┌─────────────────────────────┐  │
│  │ assessmentCriterion │        │ simulationFlow / questions  │  │
│  │ - KLO 1             │   ?    │ - Question 1                │  │
│  │ - KLO 2             │ ←───→  │ - Question 2                │  │
│  │ - KLO 3             │        │ - Question 3                │  │
│  └─────────────────────┘        └─────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────┐
                    │     KLO FIXER (NEW)      │
                    │                          │
                    │  1. Extract KLOs         │
                    │  2. Extract Questions    │
                    │  3. Check alignment      │
                    │  4. Rewrite if needed    │
                    │  5. Return FIXED JSON    │
                    └──────────────────────────┘
```

---

## Implementation: 3 Steps

### Step 1: Check + Fix in One Call (Per KLO)

```python
async def _check_and_fix_klo(self, klo, questions, company, industry):
    """Check one KLO and generate fix if misaligned."""

    prompt = f"""
## KLO:
"{klo['klo']}"

## AVAILABLE QUESTIONS:
{questions}

## TASK:
1. Find a question that assesses this KLO (uses same terminology)
2. If none found, rewrite the BEST candidate question

## OUTPUT (JSON):
{{
  "is_aligned": true/false,
  "matched_question_id": "id or null",
  "fix": null or {{
    "question_id": "id to rewrite",
    "original_text": "...",
    "new_text": "rewritten to match KLO terminology"
  }}
}}
"""
    # Call LLM and return result
```

### Step 2: Run All KLOs in Parallel

```python
async def fix(self, adapted_json, context):
    klos = self._extract_klos(adapted_json)
    questions = self._extract_questions(adapted_json)

    # Parallel check for all KLOs
    tasks = [
        self._check_and_fix_klo(klo, questions, company, industry)
        for klo in klos
    ]
    results = await asyncio.gather(*tasks)

    # Collect fixes
    fixes = [r["fix"] for r in results if r["fix"]]

    # Apply fixes
    if fixes:
        return self._apply_fixes(adapted_json, fixes)
    return adapted_json
```

### Step 3: Apply Fixes to JSON

```python
def _apply_fixes(self, adapted_json, fixes):
    """Replace question text in JSON."""

    fixed = copy.deepcopy(adapted_json)

    for fix in fixes:
        question_id = fix["question_id"]
        new_text = fix["new_text"]

        # Find and replace in submissionQuestions
        for q in fixed["topicWizardData"].get("submissionQuestions", []):
            if q.get("id") == question_id:
                q["question"] = new_text

        # Find and replace in simulationFlow
        for stage in fixed["topicWizardData"].get("simulationFlow", []):
            for q in stage.get("data", {}).get("questions", []):
                if q.get("id") == question_id:
                    q["question"] = new_text

    return fixed
```

---

## Latency

```
KLO 1 ──┐
KLO 2 ──┼── PARALLEL (~5-7s total)
KLO 3 ──┘

NOT sequential (would be 15-20s)
```

**Added time:** ~10-15 seconds
**Benefit:** +5% alignment score

---

## Example

### Input
```
KLO: "Analyze competitive landscape using Porter's Five Forces"

Questions:
1. "Describe the market conditions" (id: q1)
2. "What competitors exist?" (id: q2)
```

### LLM Output
```json
{
  "is_aligned": false,
  "matched_question_id": null,
  "fix": {
    "question_id": "q2",
    "original_text": "What competitors exist?",
    "new_text": "Analyze the competitive landscape using Porter's Five Forces framework. Identify threats from new entrants, supplier power, buyer power, substitutes, and competitive rivalry."
  }
}
```

### Result
Question q2 gets rewritten to match KLO terminology.

---

## Files to Change

| File | Change |
|------|--------|
| `src/stages/fixers.py` | Rewrite `KLOAlignmentFixer` class |

**No other files need changes** - already integrated in pipeline.

---

## Success Criteria

| Metric | Before | After |
|--------|--------|-------|
| KLO → Questions | 72-90% | 95%+ |
| Overall Score | 90% | 95%+ |
| Added Latency | - | +10-15s |

---

## Risks

| Risk | Mitigation |
|------|------------|
| LLM timeout | Small prompts per KLO, parallel execution |
| Bad rewrite | LLM sees original + KLO, rewrites sensibly |
| Wrong question selected | Prefer submissionQuestions over others |
| JSON breaks | Use deepcopy, preserve IDs |

---

## Ready to Implement

1. Rewrite `_check_and_fix_klo()` method
2. Update `fix()` to apply fixes
3. Add `_apply_fixes()` method
4. Test

**Estimated time:** 1-2 hours
