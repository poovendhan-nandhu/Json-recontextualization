# Implementation Plan: Pipeline Architectural Fixes

**Date:** 2026-01-17
**Status:** Ready for Implementation
**Estimated LOC:** ~116 lines across 6 files

---

## Current State

| Metric | Current | Target | Gap |
|--------|---------|--------|-----|
| Alignment Score | 92.33% | 95% | 2.67% |
| Validation Score | 78.96% | 95% | 16% |
| Domain Terms Leaked | 185 | 0 | Critical |
| Blockers | 48 | 0 | Critical |

---

## Root Causes Identified

1. **Domain terminology leakage** - Poison list filters out domain-specific terms as "common words"
2. **Cross-shard KLO blindness** - KLOs only injected into 4/15 shards
3. **Alignment fixer path bug** - Can't find questions in `simulationFlow[].data.submissionQuestions`
4. **Missing state fields** - Causes KeyError crashes
5. **Relaxed thresholds** - 70%/80% instead of 95%
6. **Fixer integration gap** - Semantic fixer overwrites alignment fixes
7. **Duplicate KLO checkers** - 4 different checks for "KLO alignment" with different logic (C7, R4, R5, R8)

---

## Implementation Phases

### Phase 1: Fix Crashes (P0)

#### 1.1 Initialize Missing State Fields

**File:** `src/graph/state.py`

**Location:** Find `create_initial_state()` function and add to return statement:

```python
alignment_retry_count=0,
alignment_fixes_applied=0,
alignment_fixer_skipped=False,
alignment_fix_results=[],
previous_alignment_score=0.0,
alignment_feedback={},
```

**Why:** Nodes access these fields but they're never initialized, causing KeyError.

---

### Phase 2: Fix Domain Terminology Leakage (P1)

> **Design Principle:** Domain-agnostic. Works for ANY source (HR, Finance, Marketing, etc.)

#### 2.1 Add Domain Detection to Factsheet Prompt

**File:** `src/utils/prompts.py`

**Location:** In `FACTSHEET_PROMPT` (around line 44), add BEFORE poison list instructions:

```
## DOMAIN DETECTION (CRITICAL):
First, identify the PRIMARY DOMAIN of the SOURCE scenario:
- What industry/function is the SOURCE about? (e.g., "HR/Recruitment", "Finance", "Marketing", "Operations", "Healthcare")
- What are the CORE VOCABULARY TERMS unique to this domain?

You MUST include a "source_domain" field with:
- domain_name: The identified domain
- domain_vocabulary: 20+ terms CENTRAL to this domain that MUST be replaced
```

**Location:** In JSON structure (around line 91), add after `poison_list`:

```json
"source_domain": {
  "domain_name": "Primary domain of SOURCE (e.g., HR/Recruitment, Finance, Marketing)",
  "domain_vocabulary": ["20+ domain-specific terms - these are CRITICAL and must ALL be in poison_list"]
}
```

#### 2.2 Domain-Aware Poison List Filtering

**File:** `src/utils/gemini_client.py`

**Location:** Replace `filter_poison_list()` function (lines 129-168):

```python
def filter_poison_list(poison_list: list, domain_vocabulary: list = None) -> list:
    """
    Filter out common English words from the poison list.
    NEVER filters domain_vocabulary terms - those are critical for domain separation.
    """
    if not isinstance(poison_list, list):
        return []

    # Domain vocabulary is PROTECTED - never filter these
    protected_terms = set()
    if domain_vocabulary:
        protected_terms = {term.lower().strip() for term in domain_vocabulary if isinstance(term, str)}
        logger.info(f"[POISON] Protecting {len(protected_terms)} domain vocabulary terms from filtering")

    filtered = []
    for term in poison_list:
        if not isinstance(term, str):
            continue
        term_lower = term.lower().strip()

        if not term_lower or len(term_lower) <= 2:
            continue

        # NEVER filter domain vocabulary terms
        if term_lower in protected_terms:
            filtered.append(term)
            continue

        # Filter common words (only for non-domain terms)
        if term_lower in COMMON_ENGLISH_WORDS:
            logger.debug(f"Filtered common word: {term}")
            continue

        filtered.append(term)

    # Ensure ALL domain vocabulary is in the list
    existing_lower = {t.lower() for t in filtered}
    for domain_term in (domain_vocabulary or []):
        if isinstance(domain_term, str) and domain_term.lower() not in existing_lower:
            filtered.append(domain_term)
            logger.debug(f"Added missing domain term: {domain_term}")

    logger.info(f"[POISON] Poison list: {len(poison_list)} raw -> {len(filtered)} filtered")
    return filtered
```

**Location:** In `extract_global_factsheet()`, update the filter call:

```python
# Get domain vocabulary from factsheet
domain_vocab = factsheet.get("source_domain", {}).get("domain_vocabulary", [])
filtered_poison_list = filter_poison_list(raw_poison_list, domain_vocabulary=domain_vocab)
```

---

### Phase 3: Fix Cross-Shard KLO Blindness (P1)

#### 3.1 Inject KLOs into ALL Shards

**File:** `src/utils/prompts.py`

**Location:** Find KLO injection logic (search for `simulation_flow.*resources.*rubrics`):

**Current:**
```python
if klos and shard_id.lower() in ['simulation_flow', 'resources', 'rubrics', 'assessment_criteria']:
```

**Change to:**
```python
if klos:  # Inject KLOs into ALL shards for cross-shard alignment
```

**Location:** After KLO text generation, add cross-shard context:

```python
# Add cross-shard alignment requirements
from ..utils.config import SHARD_DEFINITIONS
shard_def = SHARD_DEFINITIONS.get(shard_id.lower(), {})
aligned_shards = shard_def.get("aligns_with", [])
if aligned_shards:
    klo_text += f"\n## Cross-Shard Alignment:\n"
    klo_text += f"This shard MUST be consistent with: {', '.join(aligned_shards)}\n"
    klo_text += "Ensure terminology, entities, and KLO references match across shards.\n"
```

---

### Phase 4: Fix Alignment Fixer Path Bug (P1)

#### 4.1 Fix Stage Matching for SimulationFlow Questions

**File:** `src/stages/alignment_fixer.py`

**Location:** Find `apply_question_fix_at_location()` function, stage matching loop (around line 515-527):

**Replace with:**
```python
for stage_idx, stage in enumerate(topic.get("simulationFlow", [])):
    s_name = stage.get("name", "")

    # Match by: exact name, stage_N index, or parsed index
    stage_match = (
        s_name == stage_name or
        stage_name == f"stage_{stage_idx}" or
        (stage_name.startswith("stage_") and
         stage_name.split("_")[-1].isdigit() and
         int(stage_name.split("_")[-1]) == stage_idx)
    )

    if stage_match:
        stage_data = stage.get("data", {})
        q_list = stage_data.get(q_type, [])
        for q in q_list:
            if q.get("id") == q_id:
                if "question" in q:
                    q["question"] = new_text
                else:
                    q["text"] = new_text
                logger.info(f"[ALIGNMENT FIXER] Applied fix for {q_id} at stage {stage_idx}")
                return True
```

---

### Phase 5: Consolidate KLO Checkers (P1)

> **Problem:** 4 different checks for "KLO alignment" with different logic creates confusion

#### Current Duplicate Checks

| Check | Location | What It Checks |
|-------|----------|----------------|
| C7 | `check_definitions.py` | KLO Preservation |
| R4 | `alignment_checker.py` | KLO-to-Questions |
| R5 | `alignment_checker.py` | KLO-to-Resources |
| R8 | `alignment_checker.py` | KLO-Task Alignment |

#### 5.1 Create Unified KLO Validator

**New File:** `src/validators/klo_validator.py`

```python
"""
Unified KLO Validator - Single source of truth for ALL KLO alignment checks.

Consolidates: C7, R4, R5, R8 into one coherent validator.
"""
import logging
from typing import Any

logger = logging.getLogger(__name__)


class UnifiedKLOValidator:
    """
    Single source of truth for ALL KLO alignment checks.

    Replaces:
    - C7 (KLO Preservation) from check_definitions.py
    - R4 (KLO-to-Questions) from alignment_checker.py
    - R5 (KLO-to-Resources) from alignment_checker.py
    - R8 (KLO-Task Alignment) from alignment_checker.py
    """

    THRESHOLD = 0.95  # Single threshold for all KLO checks

    def __init__(self):
        self.results = {}

    def validate(self, adapted_json: dict, factsheet: dict) -> dict:
        """
        Run all KLO alignment checks and return unified result.

        Returns:
            {
                "overall_score": float,
                "passed": bool,
                "checks": {
                    "preservation": {"score": float, "issues": []},
                    "questions": {"score": float, "issues": []},
                    "resources": {"score": float, "issues": []},
                    "tasks": {"score": float, "issues": []},
                }
            }
        """
        klos = self._extract_klos(factsheet)
        if not klos:
            logger.warning("[KLO VALIDATOR] No KLOs found in factsheet")
            return {"overall_score": 0, "passed": False, "checks": {}}

        topic_data = adapted_json.get("topicWizardData", {})

        # Run all checks
        checks = {
            "preservation": self._check_preservation(topic_data, klos),
            "questions": self._check_questions(topic_data, klos),
            "resources": self._check_resources(topic_data, klos),
            "tasks": self._check_tasks(topic_data, klos),
        }

        # Calculate overall score (weighted average)
        weights = {"preservation": 1.0, "questions": 1.5, "resources": 1.5, "tasks": 1.0}
        total_weight = sum(weights.values())
        overall_score = sum(
            checks[k]["score"] * weights[k] for k in checks
        ) / total_weight

        result = {
            "overall_score": overall_score,
            "passed": overall_score >= self.THRESHOLD,
            "checks": checks,
        }

        logger.info(f"[KLO VALIDATOR] Overall: {overall_score:.2%} ({'PASS' if result['passed'] else 'FAIL'})")
        return result

    def _extract_klos(self, factsheet: dict) -> list:
        """Extract KLOs from factsheet."""
        klos = factsheet.get("klos", [])
        if isinstance(klos, list):
            return klos
        return []

    def _check_preservation(self, topic_data: dict, klos: list) -> dict:
        """Check if KLOs are preserved in assessment_criterion."""
        assessment = topic_data.get("assessmentCriterion", [])
        if not assessment:
            assessment = topic_data.get("selectedAssessmentCriterion", [])

        preserved = 0
        issues = []

        for i, klo in enumerate(klos):
            klo_text = klo if isinstance(klo, str) else klo.get("outcome", "")
            # Check if KLO essence is in assessment
            found = any(
                klo_text.lower()[:50] in str(a).lower()
                for a in assessment
            )
            if found:
                preserved += 1
            else:
                issues.append(f"KLO{i+1} not found in assessment criteria")

        score = preserved / len(klos) if klos else 0
        return {"score": score, "issues": issues}

    def _check_questions(self, topic_data: dict, klos: list) -> dict:
        """Check if questions align with KLOs."""
        questions = self._get_all_questions(topic_data)

        if not questions:
            return {"score": 0, "issues": ["No questions found"]}

        aligned = 0
        issues = []

        for klo_idx, klo in enumerate(klos):
            klo_text = klo if isinstance(klo, str) else klo.get("outcome", "")
            klo_keywords = self._extract_keywords(klo_text)

            # Check if any question covers this KLO
            covered = False
            for q in questions:
                q_text = q.get("question", q.get("text", "")).lower()
                if any(kw in q_text for kw in klo_keywords):
                    covered = True
                    break

            if covered:
                aligned += 1
            else:
                issues.append(f"KLO{klo_idx+1} has no aligned question")

        score = aligned / len(klos) if klos else 0
        return {"score": score, "issues": issues}

    def _check_resources(self, topic_data: dict, klos: list) -> dict:
        """Check if resources support KLOs."""
        resources = topic_data.get("resources", [])

        if not resources:
            return {"score": 0, "issues": ["No resources found"]}

        supported = 0
        issues = []

        for klo_idx, klo in enumerate(klos):
            klo_text = klo if isinstance(klo, str) else klo.get("outcome", "")
            klo_keywords = self._extract_keywords(klo_text)

            # Check if any resource supports this KLO
            covered = False
            for r in resources:
                r_text = str(r.get("markdownText", "") + r.get("title", "")).lower()
                if any(kw in r_text for kw in klo_keywords):
                    covered = True
                    break

            if covered:
                supported += 1
            else:
                issues.append(f"KLO{klo_idx+1} has no supporting resource")

        score = supported / len(klos) if klos else 0
        return {"score": score, "issues": issues}

    def _check_tasks(self, topic_data: dict, klos: list) -> dict:
        """Check if tasks/activities align with KLOs."""
        activities = topic_data.get("activities", [])

        if not activities:
            # Try simulationFlow
            for stage in topic_data.get("simulationFlow", []):
                stage_activities = stage.get("data", {}).get("activities", [])
                activities.extend(stage_activities)

        if not activities:
            return {"score": 0.5, "issues": ["No activities found (partial pass)"]}

        aligned = 0
        issues = []

        for klo_idx, klo in enumerate(klos):
            klo_text = klo if isinstance(klo, str) else klo.get("outcome", "")
            klo_keywords = self._extract_keywords(klo_text)

            covered = False
            for a in activities:
                a_text = str(a.get("description", "") + a.get("name", "")).lower()
                if any(kw in a_text for kw in klo_keywords):
                    covered = True
                    break

            if covered:
                aligned += 1
            else:
                issues.append(f"KLO{klo_idx+1} has no aligned activity")

        score = aligned / len(klos) if klos else 0
        return {"score": score, "issues": issues}

    def _get_all_questions(self, topic_data: dict) -> list:
        """Get questions from all locations."""
        questions = []

        # Top-level
        questions.extend(topic_data.get("submissionQuestions", []))

        # Inside simulationFlow
        for stage in topic_data.get("simulationFlow", []):
            stage_qs = stage.get("data", {}).get("submissionQuestions", [])
            questions.extend(stage_qs)

        return questions

    def _extract_keywords(self, text: str) -> list:
        """Extract meaningful keywords from KLO text."""
        # Simple keyword extraction - get words > 4 chars
        words = text.lower().split()
        keywords = [w for w in words if len(w) > 4 and w.isalpha()]
        return keywords[:5]  # Top 5 keywords
```

#### 5.2 Remove Duplicate Checks from alignment_checker.py

**File:** `src/stages/alignment_checker.py`

**Remove these methods** (they're now in UnifiedKLOValidator):
- `_check_klo_to_questions()` (R4)
- `_check_klo_to_resources()` (R5)
- `_check_klo_alignment()` (R8)

**Update `check()` method** to use UnifiedKLOValidator:

```python
from ..validators.klo_validator import UnifiedKLOValidator

async def check(self, adapted_json, global_factsheet, source_scenario):
    # Use unified KLO validator instead of separate checks
    klo_validator = UnifiedKLOValidator()
    klo_result = klo_validator.validate(adapted_json, global_factsheet)

    # Run remaining checks (non-KLO)
    check_coroutines = [
        self._check_reporting_manager_consistency(...),  # R1
        self._check_company_consistency(...),             # R2
        self._check_poison_terms(...),                    # R3
        self._check_scenario_to_resources(...),           # R6
        self._check_role_to_tasks(...),                   # R7
        self._check_scenario_coherence(...),              # R9
    ]

    # Add KLO result as single check
    results.append(AlignmentResult(
        rule_id="klo_alignment",
        score=klo_result["overall_score"],
        passed=klo_result["passed"],
        details=klo_result["checks"],
    ))
```

#### 5.3 Remove C7 from check_definitions.py

**File:** `src/validation/check_definitions.py`

**Remove or comment out** the C7 check definition - it's now part of UnifiedKLOValidator.

---

### Phase 6: Fix Thresholds & Integration (P1/P2)

#### 6.1 Correct Finisher Thresholds

**File:** `src/stages/finisher.py`

**Location:** Find threshold constants (around line 82-84):

**Current:**
```python
BLOCKER_PASS_RATE_REQUIRED = 0.8
OVERALL_SCORE_REQUIRED = 0.70
MAX_ITERATIONS = 1
```

**Change to:**
```python
BLOCKER_PASS_RATE_REQUIRED = 1.0    # Blockers MUST pass
OVERALL_SCORE_REQUIRED = 0.95       # Per project requirements
MAX_ITERATIONS = 3                  # Allow retries
```

**Location:** Find `passed` property (around line 43):

**Change to:**
```python
return self.blocker_pass_rate >= 1.0 and self.overall_score >= 0.95
```

#### 6.2 Resource Word Count Validation

**File:** `src/validators/scoped_validators.py`

**Location:** Find `MIN_RESOURCE_WORDS` (around line 729):

**Change:**
```python
MIN_RESOURCE_WORDS = 500  # Was 300, must match prompt requirement
```

#### 6.3 Protect Alignment Fixes from Semantic Fixer

**File:** `src/stages/fixers.py`

**Location:** In `SemanticFixer.fix()` method, add before applying fixes:

```python
# Build set of paths already fixed by alignment fixer - don't overwrite
alignment_fixes = context.get("alignment_fix_results", [])
protected_paths = set()
for fix_result in alignment_fixes:
    if isinstance(fix_result, dict):
        for change in fix_result.get("changes", []):
            if "at " in change:
                protected_paths.add(change.split("at ")[-1].strip())
            elif ":" in change:
                protected_paths.add(change.split(":")[0].strip())

if protected_paths:
    logger.info(f"[SEMANTIC FIXER] Protecting {len(protected_paths)} paths from alignment fixer")
```

**In the fix application loop, skip protected paths:**
```python
for fix in fixes:
    if any(p in fix.path for p in protected_paths):
        logger.debug(f"[SEMANTIC FIXER] Skipping {fix.path} - protected by alignment fixer")
        continue
    # ... apply fix
```

---

## File Change Summary

| File | Changes | LOC |
|------|---------|-----|
| `src/graph/state.py` | Add 6 state fields | 8 |
| `src/utils/prompts.py` | Domain detection + KLO injection | 25 |
| `src/utils/gemini_client.py` | Domain-aware poison filtering | 40 |
| `src/stages/alignment_fixer.py` | Fix stage matching | 15 |
| `src/validators/klo_validator.py` | **NEW** Unified KLO validator | 150 |
| `src/stages/alignment_checker.py` | Remove duplicate KLO checks, use unified | -80 |
| `src/validation/check_definitions.py` | Remove C7 (now in unified) | -20 |
| `src/stages/finisher.py` | Correct thresholds | 6 |
| `src/stages/fixers.py` | Protect alignment fixes | 22 |
| `src/validators/scoped_validators.py` | Resource word count | 2 |
| **Total** | | **~170** |

---

## Implementation Order

```
Phase 1: state.py              [P0 - Fix crashes]
Phase 2: prompts.py            [P1 - Domain detection]
Phase 2: gemini_client.py      [P1 - Domain-aware filtering]
Phase 3: prompts.py            [P1 - KLO injection for all shards]
Phase 4: alignment_fixer.py    [P1 - Path bug fix]
Phase 5: klo_validator.py      [P1 - NEW unified KLO validator]
Phase 5: alignment_checker.py  [P1 - Remove duplicate checks]
Phase 5: check_definitions.py  [P1 - Remove C7]
Phase 6: finisher.py           [P1 - Thresholds]
Phase 6: scoped_validators.py  [P2 - Resource word count]
Phase 6: fixers.py             [P2 - Fixer integration]
```

---

## Verification

After implementation:

```bash
# 1. Run full pipeline
python test_with_sample.py

# 2. Check logs for domain detection
# Should see: "[POISON] Protecting N domain vocabulary terms"

# 3. Check for domain term elimination
grep -i "hiring\|candidate\|interview" transformed_data_only.json
# Expected: 0 matches (for HR source)

# 4. Verify alignment score >= 95%
# 5. Verify validation score >= 95%
```

---

## Design Decisions

### Why Domain-Agnostic?

The poison list fix does NOT hardcode any specific domain (HR, Finance, etc.).

**How it works:**
1. Factsheet extraction **detects** the SOURCE domain dynamically
2. LLM **extracts** domain vocabulary (20+ terms central to that domain)
3. Filter **protects** those terms - they are NEVER filtered as "common words"
4. All domain vocabulary is **guaranteed** to be in the final poison list

**Result:** Works for ANY source scenario - HR, Finance, Marketing, Healthcare, etc.

### Why Inject KLOs into ALL Shards?

Previously only 4 shards got KLO context. But:
- `emails` shard mentions KLO-related tasks
- `workplace_scenario` sets context for KLO activities
- `characters` describes roles aligned to KLOs
- All 15 shards need awareness for consistency

### Why Consolidate KLO Checkers?

**Before:** 4 different checks (C7, R4, R5, R8) with different logic:
- A run could PASS C7 but FAIL R4/R5
- Confusing about what "KLO alignment" means
- Duplicate code, harder to maintain

**After:** 1 unified `UnifiedKLOValidator`:
- Single source of truth
- One threshold (95%)
- Weighted scoring across all KLO aspects
- Clearer failure messages

---

## Expected Results

| Metric | Before | After |
|--------|--------|-------|
| Alignment Score | 92.33% | >=95% |
| Validation Score | 78.96% | >=95% |
| Domain Terms Leaked | 185 | 0 |
| Blockers | 48 | 0 |
| KLO-Questions | 90% | >=95% |
| KLO-Resources | 88% | >=95% |
