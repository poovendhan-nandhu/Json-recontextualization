# Cartedo Simulation Adaptation Framework — Validation Summary Report

**Scenario Change:** HR Hiring/Selection → Market Entry Analysis (EcoChic Apparel U.S. Gen Z Organic T-shirts)

**Report ID:** VAL-2026-01-14-ECOCHIC-14

**Generated on:** Jan 14, 2026 (IST)

**Audience:** PM / Client / QA

---

## Run Definition

One run = the full end-to-end conversion of one simulation into a fully recontextualized "ideal" simulation, including all internal agent passes and the compliance loop.

---

## 1) Executive Gate (Go / No-Go)

| Metric | Value | Status |
|--------|-------|--------|
| **Overall Status** | ⚠️ CONDITIONAL | Requires fixes before release |
| **Alignment Score** | 88.67% | ❌ (need ≥95%) |
| **Validation Score** | 79.52% | ❌ (need ≥95%) |
| **Compliance Score** | 100% | ✅ |
| **Blocker Count** | 44 | ❌ |
| **Warning Count** | 161 | ⚠️ |

**Release recommendation:** Do NOT ship. Address Critical blockers (poison term leakage, HR language in KLOs, duplicate activities) before approval.

---

## 2) Agent-by-Agent Pipeline Outcomes (Process Transparency)

### Stage 1 — Adaptation Engine (Planner + Generator)

| Output artifacts | Status |
|------------------|--------|
| adapted_working.json | ✅ Produced |
| entity_map.json | ✅ Complete |
| global_factsheet.json | ✅ Extracted |

| Critical outcome | Result |
|------------------|--------|
| Draft adapted simulation produced | ✅ |
| Entity mapping complete | ✅ |
| KLOs generated for target scenario | ✅ |

**Notable flagged patterns:**
- HR evaluation language ("consistent and equitable") persisted in assessmentCriterion shard

**Timing:** 410,463ms (6.8 min)

---

### Stage 2 — Sharder

| Output artifacts | Status |
|------------------|--------|
| shards/* (15 total) | ✅ Created |
| locked shards | 2 (workspace_ids, scenario_options) |

| Critical outcome | Result |
|------------------|--------|
| Sharded successfully | ✅ 15/15 |
| Unlocked shards for adaptation | ✅ 13/15 |

**Timing:** 4ms

---

### Stage 3 — Alignment Checker (QA Scanner)

| Output | Status |
|--------|--------|
| alignment_report.json | ✅ Produced |
| rule_scorecard | ✅ Complete |

| Critical outcome | Result |
|------------------|--------|
| Completed rule scan | ✅ |
| Critical blockers detected | 🔴 15 blockers |

**Top issues detected:**

| Issue | Severity | Location |
|-------|----------|----------|
| Poison term "role" found | Blocker | workplaceScenario |
| Poison term "consistent" found | Blocker | keyLearningOutcomes[1] |
| Poison term "consistency" found | Blocker | keyLearningOutcomes[2] |
| Duplicate submission questions | Blocker | klo_to_questions |
| HR language in KLO phrasing | Blocker | assessmentCriterion |

**Timing:** 96,825ms (1.6 min)

---

### Stage 4 — Validation (Structural + Semantic)

| Output | Status |
|--------|--------|
| validation_report.json | ✅ Produced |
| shard_scores | ✅ Complete |

| Critical outcome | Result |
|------------------|--------|
| Structural integrity | ✅ Passed |
| ID preservation | ✅ Passed |
| Content completeness | ⚠️ 10 placeholders found |

**Timing:** 63,816ms (1.1 min)

---

### Stage 5 — Fixers (Structural + Semantic)

| Output | Status |
|--------|--------|
| structural_fixes | ✅ Applied |
| semantic_fixes | ⚠️ Partial (LLM parse errors) |

| Critical outcome | Result |
|------------------|--------|
| Structural fixes applied | ✅ |
| Semantic fixes applied | ⚠️ Some skipped due to malformed LLM output |
| Barrier compliance | ✅ |

**Timing:** 151,968ms (2.5 min)

---

### Stage 6 — Finisher (Merger + Final Assembly)

| Output | Status |
|--------|--------|
| output_json | ✅ Produced |
| golden_simulation.json | ✅ Created |

| Critical outcome | Result |
|------------------|--------|
| JSON valid | ✅ |
| All shards merged | ✅ |

**Timing:** 33ms

---

### Stage 7 — Human-in-the-Loop Approval

| What human sees | Status |
|-----------------|--------|
| Scorecard (Critical + Flagged) | ✅ Available |
| Validation report | ✅ Available |
| Output JSON | ✅ Available |

**Human decision:** ⚠️ PENDING — Requires fixes before approval

---

## 3) Critical Checks (Blocking) — Aggregated Results

**Acceptance rule:** A run passes only if all Critical checks pass with score ≥0.95.

| Critical Check | What it Ensures | Threshold | Result | Status |
|----------------|-----------------|-----------|--------|--------|
| C1: Reporting Manager Consistency | Same manager across all references | ≥0.95 | 0.97 | ✅ |
| C2: Company Name Consistency | Same company across all references | ≥0.95 | 0.97 | ✅ |
| C3: Poison Term Avoidance | No old scenario entities remain | ≥0.95 | 0.70 | 🔴 |
| C4: KLO-Question Alignment | Questions map to learning outcomes | ≥0.85 | 0.88 | ✅ |
| C5: KLO-Resource Alignment | Resources support KLOs | ≥0.85 | 0.88 | ✅ |
| C6: Scenario-Resource Alignment | Resources match scenario context | ≥0.85 | 0.90 | ✅ |
| C7: Role-Task Alignment | Tasks appropriate for learner role | ≥0.85 | 0.90 | ✅ |
| C8: KLO-Task Alignment | Tasks assess KLOs | ≥0.85 | 0.88 | ✅ |
| C9: Scenario Coherence | Narrative consistency | ≥0.85 | 0.90 | ✅ |

**Critical Gate Summary:** 8/9 checks passed = 88.9% ❌ (need 100%)

---

## 4) Validation Scores by Shard — Detailed Results

| Shard | Avg Score | Blockers | Warnings | Status |
|-------|-----------|----------|----------|--------|
| lesson_information | 0.93 | 1 | 2 | ⚠️ |
| assessment_criteria | 0.67 | 6 | 4 | 🔴 |
| industry_activities | 0.79 | 4 | 8 | 🔴 |
| activities_chat_history | 0.74 | 5 | 12 | 🔴 |
| selected_scenario | 0.88 | 2 | 6 | ⚠️ |
| workplace_scenario | 0.83 | 3 | 14 | ⚠️ |
| scenario_chat_history | 0.74 | 5 | 18 | 🔴 |
| simulation_flow | 0.74 | 5 | 32 | 🔴 |
| emails | 0.81 | 4 | 16 | ⚠️ |
| rubrics | 0.83 | 2 | 12 | ⚠️ |
| resources | 0.78 | 3 | 18 | 🔴 |
| launch_settings | 0.83 | 2 | 8 | ⚠️ |
| videos | 0.81 | 2 | 11 | ⚠️ |

**Worst performing shards:**
1. **assessment_criteria** (0.67) — HR language mixing, count mismatch
2. **activities_chat_history** (0.74) — Incomplete adaptation
3. **scenario_chat_history** (0.74) — Legacy content patterns
4. **simulation_flow** (0.74) — Duplicate activities, truncation

---

## 5) What Failed (Critical) + Why + Drill-Down

### ❌ FAILED: Poison Term Avoidance (C3) — Score: 0.70

**Short reason:** Found 3 legacy HR/hiring terms still present in market analysis context.

**Impact:** Breaks scenario authenticity; HR evaluation language ("consistent and equitable evaluations") inappropriate for market analysis simulation.

**Where it happened:**

| Term | Location | Context |
|------|----------|---------|
| "role" | workplaceScenario | "junior consultant role context" |
| "consistent" | keyLearningOutcomes[1] | "confirm thorough, consistent, and equitable evaluations" |
| "consistency" | keyLearningOutcomes[2] | "enhancing the consistency, accuracy..." |

**Root cause:** assessmentCriterion shard adapted TOPIC (market analysis) but preserved SOURCE STRUCTURE's phrasing patterns (HR evaluation language).

**Pointers:**
- Alignment report: `failed14.json → alignment.report.results[2]`
- Affected content: `output_json.topicWizardData.assessmentCriterion`

**Disposition:** Requires semantic rewrite of KLO phrasing to remove HR evaluation terminology.

---

### ⚠️ FLAGGED: assessment_criteria Shard — Score: 0.67

**Short reason:** Two sets of KLOs exist — one properly adapted, one with HR language.

**Good KLOs (lines 3705-3761):**
```
1. "Analyze market data to identify growth opportunities..."
2. "Evaluate the competitive landscape..."
3. "Apply structured frameworks (SWOT, PESTEL)..."
4. "Develop a comprehensive go/no-go recommendation..."
5. "Formulate a strategic plan..."
```

**Bad KLOs (lines 3777-3813):**
```
1. "...confirm thorough, consistent, and equitable evaluations..."
2. "...enhancing the consistency, accuracy, and impartiality..."
```

**Detected by:** Batched_context_fidelity validator

**Disposition:** Remove/rewrite second set of KLOs to match market analysis context.

---

### ⚠️ FLAGGED: Duplicate Activities

**Short reason:** Same activity appears multiple times in simulation_flow.

**Example:**
- "Formulate a Strategic Recommendation and Impact Evaluation Framework" appears 2x

**Impact:** Confuses assessment coverage; learner sees redundant tasks.

**Detected by:** role_to_tasks, klo_task_alignment checks

**Disposition:** Deduplicate activities and ensure unique mapping to KLOs.

---

## 6) Adapted Simulation Content Summary

### Simulation Metadata

| Field | Value |
|-------|-------|
| **Name** | WALK IN THE MANAGER'S SHOES by Analyzing Market Opportunities |
| **Company** | EcoChic Apparel |
| **Industry** | Fashion/Sustainable Apparel |
| **Target Market** | U.S. Gen Z Organic T-shirts |
| **Learner Role** | Junior Consultant |
| **Manager** | (Consistent across simulation) |

### Key Learning Outcomes (5)

| # | KLO |
|---|-----|
| 1 | Analyze market data to identify growth opportunities and challenges for organic T-shirts in the U.S. Gen Z segment |
| 2 | Evaluate the competitive landscape and assess strengths/weaknesses of existing market players |
| 3 | Apply structured frameworks (SWOT, PESTEL) to assess market potential, competition, capabilities, finances, and risks |
| 4 | Develop a comprehensive go/no-go market entry recommendation supported by data-driven insights |
| 5 | Formulate a strategic plan that considers market positioning, target audience, and operational feasibility |

### Simulation Flow (4 stages)

1. Introduction
2. Market Strategy Assessment Guide
3. Manager Chat
4. Final Submission

---

## 7) Approval Status

| Category | Runs | Status |
|----------|------|--------|
| ✅ Approved (Production-Ready) | 0 | — |
| ⚠️ Approved with Notes | 0 | — |
| ❌ Not Approved (Blocked) | 1 | Current run |

**Blocker:** Poison term leakage (C3), HR language in KLOs

---

## 8) Recommended Next Actions

| Priority | Action | Owner | Effort |
|----------|--------|-------|--------|
| 🔴 P0 | Rewrite assessmentCriterion KLO phrasing to remove HR language | Semantic Fixer | Medium |
| 🔴 P0 | Review poison term detection — "role", "consistent" may be false positives | Alignment Checker | Low |
| 🟡 P1 | Deduplicate activities in simulation_flow | Content Fixer | Low |
| 🟡 P1 | Complete truncated activity descriptions | Content Fixer | Low |
| 🟡 P1 | Remove empty resource entries | Content Processor | Low |
| 🟢 P2 | Add explicit data-source requirements for KLO alignment | Prompt Engineering | Medium |

---

## 9) Score Progression

| Run | Alignment | Validation | Status |
|-----|-----------|------------|--------|
| failed11 | 59.78% | — | ❌ |
| failed12 | 69.56% | — | ❌ |
| failed13 | 79.33% | — | ❌ |
| **failed14** | **88.67%** | **79.52%** | ⚠️ |
| Target | 95.00% | 95.00% | — |

**Gap to close:** 6.33% alignment, 15.48% validation

---

## Appendix: Artifacts Index

| Artifact | Path |
|----------|------|
| Full report JSON | `failed14.json` |
| Output simulation | `failed14_output_content.json` |
| Analysis summary | `failed14_analysis.md` |
| This report | `failed14_validation_report.md` |

---

*End of Validation Summary Report*
