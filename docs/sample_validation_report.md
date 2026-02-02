# Cartedo Simulation Adaptation Framework — Validation Summary Report

**Scenario Change:** Retail Banking → Airline Operations (Customer Service Training)
**Report ID:** VAL-2026-01-12-AO-19
**Generated on:** Jan 12, 2026
**Audience:** PM / Client / QA

---

## Run Definition

> One run = the full end-to-end conversion of one simulation into a fully recontextualized "ideal" simulation, including all internal agent passes and the compliance loop.

## 1) Executive Gate (Go / No-Go)

| Metric | Result |
|--------|--------|
| **Overall Status** | ✅ **APPROVED (Meets Acceptance Bar)** |
| Total runs evaluated | 20 |
| Runs passing all Critical checks | 19 / 20 (95%) ✅ (meets ≥95% requirement) |
| Runs with only Flagged issues | 6 / 20 (review recommended, not blocking) |
| Gold output produced | 20 / 20 (golden artifact produced for every run) |

**Release recommendation:** Ship is allowed, but address Flagged Issue Cluster "Persona Realism" before client demo.

## 2) Agent-by-Agent Pipeline Outcomes (Process Transparency)

### Stage 1 — Adaptation Engine (Planner + Generator)

**Output artifacts**
- `adapted_working.json`
- `entity_map.json`

**Critical outcome**
- ✅ 20/20 produced draft adapted simulation
- ✅ Entity mapping complete in all runs

**Notable flagged patterns**
- 2 runs: minor tone drift in persona emails (non-blocking)

**Drill-down**
- Logs: `logs/run-*/stage1_adaptation_engine.log`
- `artifacts/run-*/adapted_working.json`
- `artifacts/run-*/entity_map.json`

### Stage 2 — Sharder

**Output artifacts**
- `shards/* (rubrics, options, personas, resources)`
- `cas_hashes.json`

**Critical outcome**
- ✅ 20/20 sharded successfully
- ✅ 20/20 generated CAS hashes
- ✅ 20/20 slice contracts attached

**Drill-down**
- Logs: `logs/run-*/stage2_sharder.log`
- `artifacts/run-*/cas_hashes.json`
- `artifacts/run-*/shards/`

### Stage 3 — Unified Checker (QA Scanner)

**Output artifacts**
- `rule_scorecard.json`
- `check_report.md (human-readable)`

**Critical outcome**
- ✅ 20/20 completed rule scan
- 🔴 1/20 runs had a Critical blocker detected pre-fix (resolved in later stages)

**Notable flagged patterns**
- Entity Removal

**Drill-down**
- Logs: `logs/run-*/stage3_unified_checker.log`
- `artifacts/run-*/rule_scorecard.json`
- `artifacts/run-*/check_report.md`

### Stage 4 — Structural Fixers (Shape Only)

**Output artifacts**
- `structural_patch.json`
- `barrier_manifest.json (locked slices)`

**Critical outcome**
- ✅ 20/20 fixed all structural issues where present
- ✅ 20/20 enforced barrier locks after structure fixes

**Rules enforced**
- No wording changes
- No edits to locked slices after barrier=true

**Drill-down**
- Logs: `logs/run-*/stage4_structural_fixers.log`
- `artifacts/run-*/structural_patch.json`
- `artifacts/run-*/barrier_manifest.json`

### Stage 5 — Semantic Fixers (Meaning + Realism)

**Output artifacts**
- `semantic_patch.json`

**Critical outcome**
- ✅ 20/20 applied semantic fixes without violating barriers
- ✅ 20/20 updated KPIs, personas, resources per scenario context
- ⚠️ 6/20 produced minor realism warnings (Flagged)

**Drill-down**
- Logs: `logs/run-*/stage5_semantic_fixers.log`
- `artifacts/run-*/semantic_patch.json`

### Stage 6 — Finisher (Compliance Loop + Global Guard)

**Output artifacts**
- `compliance_summary.json`
- `dependency_recheck_trace.json`
- `global_guard_report.json`

**Pass criteria**
- Critical blockers = 0
- Weighted compliance score ≥ 98%
- No infinite loops / oscillation

**Critical outcome**
- ✅ 19/20 met all pass criteria
- 🔴 1/20 failed Critical gate (details below)

**Drill-down**
- Logs: `logs/run-*/stage6_finisher.log`
- `artifacts/run-*/compliance_summary.json`
- `artifacts/run-*/dependency_recheck_trace.json`
- `artifacts/run-*/global_guard_report.json`

### Stage 7 — Human-in-the-Loop Approval

**Output artifacts**
- `human_review_packet.html`
- `visual_diff.html`

**What human sees**
- Scorecard (Critical + Flagged)
- Visual diff
- Explanations + links to patches

**Human decision (sample)**
- ✅ 13 approved immediately
- ✅ 6 approved with note (flagged issues)
- ⚠️ 1 flagged (failed Critical gate; withheld)

**Drill-down**
- `artifacts/run-*/human_review_packet.html`
- `artifacts/run-*/visual_diff.html`

## 3) Critical Checks (Blocking) — Aggregated Results

> **Acceptance rule:** A run passes only if all Critical checks pass. Release is acceptable if ≥95% of runs pass all Critical checks.

| Check | What it Ensures | Threshold | Result | Status |
|-------|-----------------|-----------|--------|--------|
| C1 | No original scenario references remain (company names, locations...) | 100% | 19/20 | 🔴 |
| C2 | Industry KPIs correctly updated (e.g., 'order accuracy' → 'on-time...) | 100% | 20/20 | ✅ |
| C3 | Output JSON conforms to required simulation schema | 100% | 20/20 | ✅ |
| C4 | Rubric levels, scoring logic, and evaluation criteria preserved | 100% | 20/20 | ✅ |
| C5 | Simulation executes from start to finish without missing references | 100% | 20/20 | ✅ |
| C6 | Locked structural elements were never modified after barrier lock | 100% | 20/20 | ✅ |
| C7 | Key Learning Outcomes preserved and mapped to activities | ≥95% | 20/20 | ✅ |
| C8 | All referenced resources exist and contain valid content | 100% | 20/20 | ✅ |

**Critical Gate Summary:** 19/20 runs passed all Critical checks = 95% ✅

## 4) Flagged Checks (Non-Blocking) — Aggregated Results

| Check | What it Flags | Threshold | Result | Status |
|-------|---------------|-----------|--------|--------|
| F1 | Character personas feel authentic to target industry | ≥0.85 | 14/20 | ⚠️ |
| F2 | Resources resemble real-world artifacts for scenario | ≥0.85 | 17/20 | ⚠️ |
| F3 | Story flow reads naturally after recontextualization | ≥0.90 | 18/20 | ✅ |
| F4 | Professional tone maintained throughout | ≥0.90 | 19/20 | ✅ |
| F5 | Numbers, dates, and statistics are plausible for scenario | ≥0.85 | 18/20 | ✅ |
| F6 | Correct industry jargon used consistently | ≥0.90 | 19/20 | ✅ |

**Flagged cluster to address next:** Persona Realism (F1) — non-blocking but client-visible.

## 5) What Failed (Critical) + Why (Human Readable) + Drill-Down

### Run 11 — ❌ FAILED (Critical Check C1: Entity Removal)

| Aspect | Detail |
|--------|--------|
| **Short reason** | Found 3 legacy retail-banking entities still present in passenger disruption scenario |
| **Impact** | Breaks scenario authenticity and can confuse learners; violates "no stale references" non-negotiable |
| **Where it happened** | Persona email + one resource snippet |
| **Detected by** | Unified Checker (Stage 3) and confirmed by Global Guard (Stage 6) |

**Pointers**
- Scorecard: `artifacts/run-11/rule_scorecard.json` → findings[C1]
- Leaked entities list: `artifacts/run-11/global_guard_report.json`
- Offending shards:
  - `artifacts/run-11/shards/personas/persona_02.json`
  - `artifacts/run-11/shards/resources/resource_04.json`
- Logs: `logs/run-11/stage3_unified_checker.log`, `logs/run-11/stage6_finisher.log`

**Disposition:** Run Semantic Fixer. Requires targeted fix on affected shards, then Finisher recheck.

## 6) Downstream Effects Validation (Scope Beyond "Correctness")

> These checks ensure we didn't just "swap words," but that the adapted simulation behaves correctly and feels realistic.

**Downstream effects score (weighted)**
| Metric | Value |
|--------|-------|
| Weighted compliance score (avg across runs) | 98.6 / 100 ✅ |
| Lowest run score | 97.9 (run-11) |
| Top driver of flagged score drops | Persona Realism |

**Examples of downstream validations performed**
- Learner-facing coherence across turns (no industry-jumping mid-flow)
- Rubric language consistency with the new industry (e.g., "load factor", "on-time performance")
- Resource plausibility for the scenario (memos/emails/tools match airline ops context)
- Option alignment (system's recommended options match scenario intent)

**Pointers**
- `artifacts/run-*/downstream_effects_report.json`
- `artifacts/run-*/behavior_trace_eval.json`
- `artifacts/run-*/rubric_alignment_eval.json`

## 7) "Approved vs Not Approved" PR-Style Review

### ✅ Approved (Production-Ready)
- **Runs:** 1–10, 12–20 (except flagged notes below)
- **Meets gate:** ≥95% of runs pass all Critical checks

### ⚠️ Approved with Notes (Non-Blocking)
- **Runs:** 3, 7, 9, 14, 16, 19
- **Notes:** Persona realism tone drift; review recommended before client demo

### ❌ Not Approved (Blocked)
- **Run:** 11
- **Blocker:** C1: Entity Removal

## 8) Recommended Next Actions

1. **Fix Entity Removal** via targeted Semantic Fixer patch on identified shards → rerun Finisher (dependency recheck only).
2. **Improve Persona Realism:** update semantic templates/examples and add additional quality checks for "role-appropriate tone."
3. **Add client-facing "Confidence" banner:** "Critical Gate Passed (95%), Flagged Issues Present (30%)" for transparency.

---

## Appendix: Artifacts Summary

| Artifact Type | Path |
|--------------|------|
| Per-run review packet | `artifacts/run-*/human_review_packet.html` |
| Root JSON | `artifacts/run-*/golden_adapted_simulation.json` |
| Unified scorecard | `artifacts/run-*/rule_scorecard.json` |
| Global Guard report | `artifacts/run-*/global_guard_report.json` |
| Logs | `logs/run-*/stage*.log` |

---
*End of Validation Report*
*Generated by Cartedo Validation Agent v1.0*
