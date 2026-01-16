# Failed14 Analysis Report

## Overall Score: 88.67% (Target: 95%)

**Progress:** 59.78% → 69.5% → 79.3% → **88.67%**

---

## Rule Scores

| Rule | Score | Status | Notes |
|------|-------|--------|-------|
| reporting_manager_consistency | 0.97 | PASSED | Fixed |
| company_consistency | 0.97 | PASSED | Fixed |
| poison_term_avoidance | 0.70 | FAILED | 3 "poison" terms found |
| klo_to_questions | 0.88 | PASSED | Minor alignment issues |
| klo_to_resources | 0.88 | PASSED | Resource gaps noted |
| scenario_to_resources | 0.90 | PASSED | Empty resource entry |
| role_to_tasks | 0.90 | PASSED | Warnings only |
| klo_task_alignment | 0.88 | PASSED | Duplicate activities |
| scenario_coherence | 0.90 | PASSED | Minor scope blur |

---

## Primary Blocker: Poison Term Avoidance (0.70)

The 3 "poison terms" flagged appear to be **FALSE POSITIVES**:

1. **"role"** - Found in "junior consultant role context"
   - "role" is a common English word, not scenario-specific

2. **"ensure"** - Flagged but actually found "consistent"
   - Seems like wrong term detection

3. **"consistent"** - Found as "consistency"
   - "consistency" is valid business language

**Recommendation:** Review poison list - may be too aggressive with common words.

---

## Secondary Issues (Warnings)

1. **Duplicate activities** - "Formulate a Strategic Recommendation..." appears twice
2. **Truncated content** - Some activity descriptions cut off mid-sentence
3. **Empty resource** - One resource entry has blank title/content
4. **Generic stats** - Some statistics not directly verifiable

---

## Gap to 95%

Current: 88.67%
Target: 95%
Gap: **6.33%**

To close the gap:
1. Fix poison term detection (false positives)
2. Remove duplicate activities
3. Complete truncated descriptions
4. Remove empty resource entry




