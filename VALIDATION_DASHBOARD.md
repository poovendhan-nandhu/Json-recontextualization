# Cartedo Simulation Adaptation Framework - Validation Dashboard

**Report Type:** Agent Glossary + KPI Dashboard
**Date:** 2026-01-16
**Scenario:** HR Hiring/Selection → Market Entry Analysis (EcoChic Threads)

---

## Part 1: Agent Glossary

This table shows all agents in the pipeline, what each agent checks, and the threshold values for pass/fail.

### Pipeline Stages

| Stage | Agent Name | What It Does | Threshold | Why It Matters |
|-------|-----------|--------------|-----------|----------------|
| 1 | **Adaptation Engine** | Generates adapted simulation from source + factsheet | N/A (generator) | Produces the initial recontextualized content |
| 2 | **Sharder** | Splits JSON into 15 shards for parallel processing | 15/15 shards | Enables parallel validation and fixes |
| 3 | **Alignment Checker** | Validates cross-shard semantic alignment | **≥95%** | Ensures KLOs, questions, resources are coherent |
| 4 | **Alignment Fixer** | Fixes failed alignment rules | score < 95% triggers | Repairs misaligned content automatically |
| 5 | **Validation Agent** | Runs critical + flagged checks | **≥95%** | Ensures structural and semantic correctness |
| 6 | **Compliance Loop** | Iterates until compliance or max retries | **100%** blockers pass | Guarantees no critical issues remain |
| 7 | **Finisher** | Final cleanup, URL fixes, placeholder removal | Pass all checks | Produces clean output JSON |

---

### Alignment Checker Rules (9 Rules)

| Rule ID | Rule Name | What It Checks | Threshold | Fix Agent |
|---------|-----------|----------------|-----------|-----------|
| R1 | Reporting Manager Consistency | Same manager name used throughout simulation | **≥95%** | Semantic Fixer |
| R2 | Company Name Consistency | Same company name used throughout simulation | **≥95%** | Semantic Fixer |
| R3 | Poison Term Avoidance | No old scenario terms (Elizabeth Carter, old company) | **100%** | Semantic Fixer |
| R4 | KLO ↔ Questions Alignment | Each KLO has specific submission questions | **≥95%** | Alignment Fixer |
| R5 | KLO ↔ Resources Alignment | Resources support each KLO adequately | **≥95%** | Alignment Fixer |
| R6 | Scenario ↔ Resources Alignment | Resources match new scenario context | **≥95%** | Semantic Fixer |
| R7 | Role ↔ Tasks Alignment | Tasks appropriate for learner role (Junior Consultant) | **≥90%** | Alignment Fixer |
| R8 | KLO ↔ Task Alignment | Activities mapped to specific KLOs | **≥90%** | Alignment Fixer |
| R9 | Scenario Coherence | Consistent story flow after recontextualization | **≥90%** | Semantic Fixer |

---

### Critical Checks (8 Checks - Blocking)

| Check ID | Check Name | What It Ensures | Threshold | Fix Agent |
|----------|-----------|-----------------|-----------|-----------|
| C1 | Entity Removal | No original scenario references remain | **100%** | Semantic Fixer |
| C2 | KPI Alignment | Industry KPIs correctly updated | **100%** | Semantic Fixer |
| C3 | Schema Validity | Output JSON conforms to simulation schema | **100%** | Structural Fixer |
| C4 | Rubric Integrity | Rubric levels, scoring logic preserved | **100%** | Structural Fixer |
| C5 | End-to-End Executability | Simulation executes without missing references | **100%** | Finisher |
| C6 | Barrier Compliance | Locked elements never modified | **100%** | Structural Fixer |
| C7 | KLO Preservation | Key Learning Outcomes preserved and mapped | **≥95%** | Alignment Fixer |
| C8 | Resource Completeness | All referenced resources exist with valid content | **100%** | Semantic Fixer |

---

### Flagged Checks (6 Checks - Non-Blocking)

| Check ID | Check Name | What It Ensures | Threshold | Fix Agent |
|----------|-----------|-----------------|-----------|-----------|
| F1 | Persona Realism | Character personas feel authentic | **≥85%** | Semantic Fixer |
| F2 | Resource Authenticity | Resources resemble real-world artifacts | **≥85%** | Semantic Fixer |
| F3 | Narrative Coherence | Story flow reads naturally | **≥90%** | Semantic Fixer |
| F4 | Tone Consistency | Professional tone maintained | **≥90%** | Semantic Fixer |
| F5 | Data Realism | Numbers, dates, statistics are plausible | **≥85%** | Semantic Fixer |
| F6 | Industry Terminology | Correct industry jargon used | **≥90%** | Semantic Fixer |

---

## Part 2: Run Dashboard (20 Runs)

### Executive Summary

| Metric | Target | Current Best | Status |
|--------|--------|--------------|--------|
| Alignment Score | ≥95% | 92.33% (FAILED20) | ❌ Gap: 2.67% |
| Validation Score | ≥95% | 80.9% (FAILED15) | ❌ Gap: 14.1% |
| Compliance Score | 100% | 100% | ✅ PASS |
| Poison Terms | 0 | 0 | ✅ PASS |
| Blockers | 0 | 48 | ❌ HIGH |

---

### Pass/Fail Dashboard Across Runs

| Run | Alignment | Validation | Compliance | Blockers | Overall | Reason for Failure |
|-----|-----------|------------|------------|----------|---------|-------------------|
| FAILED14 | 88.67% ❌ | 79.52% ❌ | 100% ✅ | 44 | **FAIL** | Poison terms (HR language in KLOs), duplicate activities |
| FAILED15 | 92.11% ❌ | 80.9% ❌ | 100% ✅ | 36 | **FAIL** | Duplicate submission questions, truncated resources |
| FAILED18 | 89.0% ❌ | 78.0% ❌ | 100% ✅ | 52 | **FAIL** | Alignment fixer not finding questions (empty IDs) |
| FAILED20 | 92.33% ❌ | 78.96% ❌ | 100% ✅ | 48 | **FAIL** | KLO-resources gap (88%), HR terminology (185 occurrences) |

**Progress Trend:** Alignment improved from 88.67% → 92.33% (+3.66%)

---

### Alignment Rule Scores (FAILED20)

| Rule | Score | Status | Gap to 95% |
|------|-------|--------|------------|
| Reporting Manager Consistency | 96% | ✅ PASS | +1% |
| Company Name Consistency | 97% | ✅ PASS | +2% |
| **Poison Term Avoidance** | **100%** | ✅ PASS | +5% |
| KLO ↔ Questions | 90% | ⚠️ FAIL | -5% |
| **KLO ↔ Resources** | **88%** | ❌ FAIL | **-7%** |
| Scenario ↔ Resources | 90% | ⚠️ FAIL | -5% |
| Role ↔ Tasks | 90% | ⚠️ FAIL | -5% |
| KLO ↔ Task Alignment | 90% | ⚠️ FAIL | -5% |
| Scenario Coherence | 90% | ⚠️ FAIL | -5% |

**Weighted Average:** 92.33%

---

## Part 3: Failure Analysis & Suggestions

### Top 3 Failures Preventing 95%

#### 1. KLO ↔ Resources Alignment (88%) - CRITICAL

**Why it failed:**
- Market entry document truncated mid-PESTEL analysis
- No concrete financial numbers (CAC/LTV, margins, break-even)
- Competitive landscape is qualitative, not structured benchmark
- No TAM/SAM/SOM worksheet for US Gen Z organic T-shirts

**Suggestion to fix:**
1. Add minimum word count (500+ words) for resource generation
2. Include financial template with CAC, margins, break-even in prompt
3. Add competitor benchmarking table (prices, certifications, channels)
4. Include market sizing worksheet with sources

---

#### 2. KLO ↔ Questions Alignment (90%) - HIGH

**Why it failed:**
- Submission questions are duplicated verbatim across KLOs
- Questions assess rubric/meta-skills, not direct KLO criteria
- KLO1 segmentation requirement only implicitly covered
- KLO2 financial viability (CAC, Gross Margin) not explicitly required
- No explicit go/no-go deliverable for KLO3

**Suggestion to fix:**
1. Remove/consolidate duplicated submission questions
2. Add explicit TAM/SAM calculation requirement
3. Separate assessment items for KLO2 (internal capabilities, financial projections, risk register)
4. Require explicit go/no-go recommendation deliverable

---

#### 3. HR Terminology Leakage (185 occurrences) - HIGH

**Why it failed:**
- Generic business terms in resources still contain "HR" references
- Base JSON structure not fully cleaned
- Rubric/assessment criteria containing HR terminology from source

**Terms found:**
- "HR": 185 occurrences
- "hiring": 4 occurrences
- "candidate": 3 occurrences
- "interview": 5 occurrences
- "recruitment": 1 occurrence

**Suggestion to fix:**
1. Add HR-specific terms to poison list
2. Run semantic replacement on all text fields
3. Add post-processing step to detect and replace HR terminology

---

## Part 4: What's Working (Green)

| Check | Score | Notes |
|-------|-------|-------|
| Poison Term Avoidance | 100% | No "Elizabeth Carter" or old scenario entities |
| Company Consistency | 97% | "EcoChic Threads" used throughout |
| Manager Consistency | 96% | "Maya Sharma" x21 occurrences |
| Compliance Score | 100% | All structural barriers respected |
| URL Truncation | Fixed | 0 truncated URLs in output_json |
| Placeholders | Fixed | 0 placeholders in output_json |

---

## Part 5: Action Items to Reach 95%

### Priority P0 (Critical - Blocking 95%)

| # | Action | Owner | Expected Impact |
|---|--------|-------|-----------------|
| 1 | Fix alignment fixer to find questions in simulationFlow | Dev | +2% alignment |
| 2 | Add explicit KLO-specific submission questions | Dev | +3% alignment |
| 3 | Add financial model to resources (CAC, margins, break-even) | Dev | +4% alignment |

### Priority P1 (High)

| # | Action | Owner | Expected Impact |
|---|--------|-------|-----------------|
| 4 | Remove HR terminology (185 occurrences) | Dev | Cleaner output |
| 5 | Remove duplicate activities | Dev | +1% alignment |
| 6 | Complete truncated content (market entry doc) | Dev | +2% alignment |

### Priority P2 (Medium)

| # | Action | Owner | Expected Impact |
|---|--------|-------|-----------------|
| 7 | Improve resource generation prompts (word count minimums) | Dev | Better resources |
| 8 | Fix scenario coherence ("optimization" vs "go/no-go") | Dev | +1% alignment |
| 9 | Add citations to resources | Dev | Better traceability |

---

## Part 6: Decision Gate

### Current State

| Criteria | Value | Gate |
|----------|-------|------|
| Alignment Score | 92.33% | ❌ FAIL (need ≥95%) |
| Validation Score | 78.96% | ❌ FAIL (need ≥95%) |
| Compliance Score | 100% | ✅ PASS |
| Critical Blockers | 48 | ❌ FAIL (need 0) |

### Recommendation

**DO NOT SHIP** - Address P0 items before release.

**Gap Analysis:**
- Current: 92.33% alignment
- Target: 95% alignment
- Gap: 2.67% (achievable with P0 fixes)

**Estimated runs to 95%:** 3-5 more iterations with P0 fixes applied

---

*Generated: 2026-01-16*
*Report Format: Per Shewta's Requirements (Agent Glossary + KPI Dashboard)*
