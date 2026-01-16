# Current Issues & Problems (Updated 2026-01-15)

## Status Summary

| Metric | Current (FAILED20) | Target | Status |
|--------|-------------------|--------|--------|
| Alignment Score | **92.33%** | 95% | ⚠️ 2.67% gap |
| Validation Score | 78.96% | 95% | ❌ 16% gap |
| Compliance Score | **100%** | 100% | ✅ PASS |
| Blockers | 48 | 0 | ❌ High |

---

## FIXED ISSUES ✅

### 1. URL Truncation (TRAILING DOTS) - ✅ FIXED
**Status:** URLs in `output_json` are now correct (`.webp`, `.pdf`)
- Truncated URLs only appear in validation reports (showing issues BEFORE fix)
- `fix_truncated_urls()` in `content_fixer.py` working
- `cleanup_merged_json()` in merger stage applies fix

### 2. Placeholder Text - ✅ FIXED
**Status:** No placeholders found in output_json
- `remove_placeholders()` function expanded with 30+ patterns
- Catches `[brackets]`, `[placeholder]`, `[YOUR_NAME]`, etc.

### 3. Manager Name - ✅ FIXED
**Status:** Manager name is consistent
- Found: "Maya Sharma" x21 occurrences
- No more garbage text like "practices and"

### 4. Poison Term Avoidance - ✅ 100%
**Status:** No "Elizabeth Carter" or old scenario entities
- Score: 1.0 (100%)

### 5. Company Consistency - ✅ 97%
**Status:** Company name consistent throughout

### 6. Verification Blocking on Trailing Dots - ✅ FIXED
**Status:** Verification now logs but doesn't block on trailing dots
- Dots are fixed later in merger stage

---

## REMAINING ISSUES ❌

### 7. KLO-to-Questions Alignment - ⚠️ 90% (needs 95%)

**Problems:**
- Submission questions are **duplicated verbatim**
- Questions assess rubric/meta-skills, not direct KLO criteria
- KLO1 segmentation requirement only implicitly covered
- KLO2 financial viability (CAC, Gross Margin) not explicitly required
- No explicit go/no-go deliverable for KLO3

**Suggestions from Alignment Report:**
1. Remove/consolidate duplicated submission questions
2. Add explicit TAM/SAM calculation requirement
3. Separate assessment items for KLO2 (internal capabilities, financial projections, risk register)
4. Require explicit go/no-go recommendation deliverable

### 8. KLO-to-Resources Alignment - ⚠️ 88% (needs 95%)

**Problems:**
- Market entry document **truncated** (cuts off mid-PESTEL)
- Limited concrete financial numbers (CAC/LTV, margins, break-even)
- Competitive landscape is qualitative, not structured benchmark
- No TAM/SAM/SOM worksheet for US Gen Z organic T-shirts
- PDF resources may be generic, not scenario-specific

**Suggestions from Alignment Report:**
1. Provide complete, untruncated market entry document
2. Add lightweight financial model (pricing, COGS, margins, CAC, break-even)
3. Add competitor benchmarking table (prices, certifications, channels)
4. Include market sizing worksheet with sources

### 9. Scenario-to-Resources Alignment - ⚠️ 90% (needs 95%)

**Problems:**
- Only 1 resource explicitly references "VerdeThreads"
- Market entry analysis truncated at end
- Cited statistics not verifiable (no links/citations)
- PDFs may not be sustainable fashion specific

### 10. Role-to-Tasks Alignment - ⚠️ 90% (needs 95%)

**Problems:**
- Task 3 edges into senior-level ownership
- **ACTIVITIES list has duplicates** (same activity appears twice)
- Activity descriptions truncated
- Scope creep risk (deep financial modeling for junior consultant)

### 11. KLO-Task Alignment - ⚠️ 90% (needs 95%)

**Problems:**
- Two activities have exact same name (duplication)
- Financial viability/risk assessment not explicitly assessed
- No clear go/no-go deliverable
- Activity descriptions truncated

### 12. Scenario Coherence - ⚠️ 90% (needs 95%)

**Problems:**
- Mismatch: WORKPLACE says "optimizing US penetration" vs FACTSHEET says "go/no-go entry"
- Email truncated (ends mid-word)
- Citable facts list truncated

### 13. Poison Terms Still Leaking (HR Domain)

**From latest output check:**
```
Summit: 1 occurrence
Innovations: 1 occurrence
HR: 185 occurrences (!)
hiring: 4 occurrences
candidate: 3 occurrences
interview: 5 occurrences
recruitment: 1 occurrence
```

**Root Cause:** These appear to be from:
- Generic business terms in resources
- Base JSON structure not fully cleaned
- Rubric/assessment criteria containing HR terminology

### 14. Alignment Fixer Not Finding Questions/Resources

**From logs:**
```
[ALIGNMENT FIXER] Questions: Existing IDs = []
[ALIGNMENT FIXER] Resources: Existing IDs = []
```

**Root Cause:**
- Questions/resources stored inside `simulationFlow` stages, not at top level
- Alignment fixer only looked at top-level `submissionQuestions`

**Fix Applied:** Updated to search ALL locations including simulationFlow stages

### 15. Content Truncation

**Problems:**
- Email body preview truncated ("Junior Consultan")
- Resource content truncated mid-section
- Activity descriptions incomplete
- Citable facts truncated mid-sentence

### 16. Word Count Violations

**From validation report:**
- 14 sections below minimum word count (200 words)
- resourceOptions descriptions too short (31-48 words)
- taskEmail.body is 0 words

---

## ALIGNMENT SCORE BREAKDOWN

| Rule | Score | Status | Priority |
|------|-------|--------|----------|
| reporting_manager_consistency | 96% | ✅ | - |
| company_consistency | 97% | ✅ | - |
| poison_term_avoidance | 100% | ✅ | - |
| **klo_to_questions** | 90% | ⚠️ | P0 |
| **klo_to_resources** | 88% | ⚠️ | P0 |
| scenario_to_resources | 90% | ⚠️ | P1 |
| role_to_tasks | 90% | ⚠️ | P1 |
| klo_task_alignment | 90% | ⚠️ | P1 |
| scenario_coherence | 90% | ⚠️ | P2 |

**To reach 95% overall:** Need to improve klo_to_resources (88%) and klo_to_questions (90%)

---

## ROOT CAUSE ANALYSIS

### Why Alignment Fixer Isn't Working
1. **Empty IDs:** Questions/resources stored in `simulationFlow`, not top-level
2. **LLM returns 2-char response:** When no questions found, LLM returns `[]`
3. **Position fallback not triggered:** No questions to fall back on

### Why Resources Are Sparse
1. **Prompt doesn't require specifics:** No minimum word count enforced
2. **Financial data missing:** No template for CAC/LTV/margins
3. **Truncation during generation:** LLM cuts off mid-section

### Why Questions Don't Assess KLOs
1. **Generic rubric questions:** "Clarity of Thought" instead of "Calculate TAM/SAM"
2. **Duplication:** Same question copied for multiple KLOs
3. **Indirect coverage:** Activities might cover KLO, but questions don't require it

---

## PRIORITY FIXES NEEDED

### P0 (Critical - Blocking 95%)
1. ❌ **Fix alignment fixer to find questions in simulationFlow** - Code added, needs testing
2. ❌ **Add explicit KLO-specific questions** - Need to rewrite submission questions
3. ❌ **Add financial model to resources** - Need template with CAC, margins, break-even

### P1 (High)
4. ❌ **Remove HR terminology** - 185 "HR" occurrences still present
5. ❌ **Remove duplicate activities** - Same activity appears 2x
6. ❌ **Complete truncated content** - Market entry doc cut off mid-PESTEL

### P2 (Medium)
7. ⚠️ **Improve resource generation prompts** - Add word count minimums
8. ⚠️ **Fix scenario coherence** - Align "optimization" vs "go/no-go" framing
9. ⚠️ **Add citations to resources** - Make statistics verifiable

---

## FILES MODIFIED (This Session)

| File | Changes |
|------|---------|
| `src/stages/sharder.py` | Added `fix_truncated_urls()` and `remove_placeholders()` to `cleanup_merged_json()` |
| `src/utils/content_fixer.py` | Improved URL fix, expanded placeholder patterns, added debug logging |
| `src/utils/prompts.py` | Changed trailing dots check from blocker to debug log |
| `src/stages/alignment_fixer.py` | Search ALL locations for questions (simulationFlow), aggressive KLO prompt, position fallback |

---

## NEXT STEPS

1. **Run pipeline again** with new alignment fixer code
2. **Check logs** for `[ALIGNMENT FIXER] Found N questions total from all locations`
3. **If questions still empty**, the issue is in JSON structure - need to inspect actual simulationFlow
4. **If 95% not reached**, need to:
   - Add explicit financial deliverable question
   - Add go/no-go recommendation question
   - Expand resource content with financial template

---

*Last Updated: 2026-01-15 (After FAILED20 analysis)*
