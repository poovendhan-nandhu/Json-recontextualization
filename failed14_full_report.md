# Failed14 Full Report

## Overall Scores

| Metric | Score | Status |
|--------|-------|--------|
| **Alignment Score** | 88.67% | FAILED (need 95%) |
| **Validation Score** | 79.52% | FAILED |
| **Compliance Score** | 100% | PASSED |

---

## Alignment Scores by Rule

| Rule | Score | Status |
|------|-------|--------|
| reporting_manager_consistency | 0.97 | PASSED |
| company_consistency | 0.97 | PASSED |
| poison_term_avoidance | 0.70 | FAILED |
| klo_to_questions | 0.88 | PASSED |
| klo_to_resources | 0.88 | PASSED |
| scenario_to_resources | 0.90 | PASSED |
| role_to_tasks | 0.90 | PASSED |
| klo_task_alignment | 0.88 | PASSED |
| scenario_coherence | 0.90 | PASSED |

---

## Validation Scores by Shard

| Shard | Avg Score | Blockers |
|-------|-----------|----------|
| lesson_information | 0.93 | 1 |
| assessment_criteria | 0.67 | 6 |
| industry_activities | 0.79 | 4 |
| activities_chat_history | 0.74 | 5 |
| selected_scenario | 0.88 | 2 |
| workplace_scenario | 0.83 | 3 |
| scenario_chat_history | 0.74 | 5 |
| simulation_flow | 0.74 | 5 |
| emails | 0.81 | 4 |
| rubrics | 0.83 | 2 |
| resources | 0.78 | 3 |
| launch_settings | 0.83 | 2 |
| videos | 0.81 | 2 |

**Worst performing shards:**
- assessment_criteria: 0.67 (6 blockers)
- activities_chat_history: 0.74 (5 blockers)
- scenario_chat_history: 0.74 (5 blockers)
- simulation_flow: 0.74 (5 blockers)

---

## Adapted Simulation Content

**Simulation Name:** WALK IN THE MANAGER'S SHOES by Analyzing Market Opportunities

**KLOs (5):**
1. Analyze market data to identify growth opportunities and challenges for organic T-shirts in the U.S. Gen Z segment
2. Evaluate the competitive landscape and assess strengths/weaknesses of existing market players
3. Apply structured frameworks (SWOT, PESTEL) to assess market potential, competition, capabilities, finances, and risks
4. Develop a comprehensive go/no-go market entry recommendation supported by data-driven insights
5. Formulate a strategic plan that considers market positioning, target audience, and operational feasibility

**Simulation Flow (4 stages):**
1. Introduction
2. Market Strategy Assessment Guide
3. Manager Chat
4. (Additional stages...)

---

## Key Issues to Fix

1. **Poison terms (0.70):** "role", "consistent", "equitable" flagged as poison - may be false positives
2. **assessment_criteria shard (0.67):** HR language mixed with market analysis ("consistent and equitable evaluations")
3. **Duplicate activities:** Same activity appears multiple times
4. **Truncated content:** Some descriptions cut off mid-sentence
5. **Empty resources:** Blank resource entries

---

## Output Files

- `failed14_output_content.json` - Full adapted simulation JSON
- `failed14_full_report.md` - This report
