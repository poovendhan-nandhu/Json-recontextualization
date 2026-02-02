# JSON Recontextualization System - Complete Issues Analysis

**Date:** January 2026
**Status:** Deep Analysis Complete
**Purpose:** Document all identified issues for prioritized fixing

---

## Executive Summary

The current shard-based adaptation system has **21 identified issues** across 7 categories that impact quality, consistency, and performance. The most critical issues are:

1. **Content Explosion** - Empty shards inflate 100x (chat_history: 251 → 23,724 chars)
2. **Domain Term Leakage** - Only 3 forbidden terms captured vs 30-50 needed
3. **Inconsistent Thresholds** - 4 different sources of truth for pass/fail criteria
4. **Prompt Bloat** - 168-line prompts with conflicting rules

---

## Table of Contents

1. [Content Quality Issues](#1-content-quality-issues)
2. [Content Explosion Issues](#2-content-explosion-issues)
3. [Pipeline Logic Issues](#3-pipeline-logic-issues)
4. [Configuration Inconsistencies](#4-configuration-inconsistencies)
5. [Performance Issues](#5-performance-issues)
6. [Error Handling Issues](#6-error-handling-issues)
7. [Testing Gaps](#7-testing-gaps)

---

## 1. Content Quality Issues

### 1.1 Entity Map Generation is Too Weak

**Location:** `src/stages/simple_adapter.py` lines 763-992

**Evidence from logs:**
```
[MAPS] Generated: 2 people, 2 roles, 22 term mappings, 3 forbidden terms
[MAPS] Forbidden terms: ['FreshTaste', 'HarvestBowls', "Nature's Crust"]
```

**Problems:**
| Issue | Current | Expected |
|-------|---------|----------|
| Forbidden terms | 3 | 30-50 |
| JSON sample size | 1500-2000 chars per section | Full scan |
| Validation | None | Verify coverage |

**Root Cause:**
- Single LLM call with limited JSON sample
- No comprehensive regex pre-scan
- No validation that key terms were captured

**Impact:** Source domain terms leak through adaptation.

---

### 1.2 Resource Quality - Inference Map Violations

**Location:** `src/stages/simple_validators.py` lines 321-509

**Problem:** Direct answer phrases leak through.

**Incomplete forbidden list (line 418-425):**
```python
FORBIDDEN in resources:
- "should" / "recommend" / "therefore" / "thus"
# MISSING:
# - "the solution is"
# - "we recommend implementing"
# - "the data shows that you should"
# - "our analysis suggests"
# - "it would be best to"
```

**Impact:** Resources give answers instead of data for learners to analyze.

---

### 1.3 Domain Term Leakage - Forbidden Terms Still Appear

**Location:** `src/stages/simple_adapter.py` lines 215-256

**Evidence from previous tests:**
```
HR: 185 occurrences (!)
hiring: 4 occurrences
candidate: 3 occurrences
interview: 5 occurrences
```

**Problems:**
1. `cleanup_forbidden_terms()` runs AFTER adaptation (too late)
2. Simple string replacement can break JSON structure
3. Domain fidelity validator samples only first 50K chars

**Code Issue (line 234-246):**
```python
def cleanup_forbidden_terms(adapted_json: dict, forbidden_terms: list[str], ...):
    # Runs AFTER all adaptation is done
    # By then, damage is already in the output
    adapted_str = json.dumps(adapted_json, ensure_ascii=False)
    for term in forbidden_terms:       
        adapted_str = adapted_str.replace(term, replacement)  # Dangerous global replace
```

---

### 1.4 Inconsistent Numeric Values Across Shards

**Location:** `src/stages/simple_adapter.py` lines 756-778, 984-986

**Problem:** Same metric shows different numbers in different shards.

**Evidence:**
- Factsheet indexed in RAG (line 984-986)
- But retrieval is optional (line 1145: `numeric_factsheet = None`)
- No post-validation that output numbers match factsheet

**Example of drift:**
```
Shard A: "Market share: 38%"
Shard B: "Market share: 42%"
Shard C: "Market share: 35%"
```

---

### 1.5 Completeness Validator - False Negatives

**Location:** `src/stages/simple_validators.py` lines 743-896

**Missing placeholder patterns (line 751-780):**
```python
placeholder_patterns = [
    r'\[TBD\]',
    r'\[TODO\]',
    # MISSING:
    # r'\[Company Name\]'
    # r'\[Product Name\]'
    # r'\[Industry\]'
    # r'\[Manager\]'
]
```

**Overly permissive exclusions (line 854-881):**
```python
structural_patterns = [
    "overview",      # But overview SHOULD have content!
    "description",   # Description should NOT be empty in resources!
]
```

---

### 1.6 Context Fidelity - Goal/Challenge Drift

**Location:** `src/stages/simple_validators.py` lines 259-315

**Evidence:**
```
Mismatch: WORKPLACE says "optimizing US penetration" vs
FACTSHEET says "go/no-go entry"
```

**Problem:** Learning objectives change between shards because:
- Context fidelity instructions buried in 168-line prompt
- Binary check (preserved: true/false) instead of degree
- No cross-shard consistency verification

---

### 1.7 Question-KLO Alignment - Terminology Mismatch

**Location:** `src/stages/simple_validators.py` lines 516-643

**Problem:** Questions don't use KLO terminology.

**Example:**
```
KLO: "Analyze competitor strategy and market positioning"
Question: "What are the critical success factors?"
# No mention of "competitor", "strategy", or "positioning"
```

**Root Cause (line 580-603):** Validator prompt doesn't require EXACT terminology matching.

---

## 2. Content Explosion Issues

### 2.1 Empty Shards Inflate Massively

**Evidence from test runs:**

| Shard | Input | Output | Ratio | Status |
|-------|-------|--------|-------|--------|
| `chat_history` | 251 | 23,724 | **94x** | CRITICAL |
| `chat_history` (test 2) | 251 | 27,751 | **110x** | CRITICAL |
| `batch_0` | 7,941 | 22,040 | **2.8x** | BAD |
| `batch_0` (test 2) | 7,941 | 26,158 | **3.3x** | BAD |
| `batch_1` | 9,764 | 14,619 | **1.5x** | HIGH |
| `resources` | 15,814 | 22,643 | **1.4x** | HIGH |

**Root Cause:** Prompt says (line 1427-1432):
```
**IMPORTANT FOR CHAT HISTORY / CONVERSATION CONTENT:**
If this JSON contains chat messages, conversation history, or previous dialogue:
- Transform ALL content within messages...
```

LLM interprets "transform" as "generate" when content is empty.

---

### 2.2 Conflicting Word Count Rules

**Location:** `src/stages/simple_adapter.py` lines 1498-1526

**CONFLICT:**
```python
# Rule 1 (line 1521-1523):
"OUTPUT LENGTH RULE: Keep output SIMILAR length to input"
"Target: 0.9x to 1.3x of input length"

# Rule 2 (line 1498-1501):
"**WORD LIMIT: 800-1400 words per resource (STRICT)**"
"Under 800 words = TOO SHORT, will be rejected"
```

**If input resource is 400 words:**
- Rule 1 says: 360-520 words
- Rule 2 says: 800-1400 words

**LLM follows stricter rule → 2x explosion**

---

### 2.3 No Output Length Enforcement

**Location:** `src/stages/simple_adapter.py`

**Current behavior:**
```python
# Just logs the difference
logger.info(f"<<< DONE: {shard_name} (input={input_size}, output={output_size})")
# No check, no rejection, no retry
```

**Missing:**
```python
if output_size > input_size * 2.0:
    logger.warning(f"Content explosion: {output_size/input_size:.1f}x")
    # Should: retry with strict prompt OR reject
```

---

### 2.4 Shard-Agnostic Prompt Rules

**Problem:** All shards get same content generation rules.

**Current:**
```python
# Same 168-line prompt for ALL shards including:
"REQUIRED in resources: Raw statistics, Percentages..."
"ALL content must be complete and realistic"
```

**Effect:** chat_history (which should be minimal) gets told to add statistics and complete content.

---

## 3. Pipeline Logic Issues

### 3.1 Shard Dependency Violations

**Location:** `src/stages/simple_adapter.py` lines 1789-1937

**Problem:** All shards processed in parallel without dependency resolution.

```python
# Line 1789-1937
tasks = []
for shard in unlocked_shards:  # ALL shards in parallel
    task = _adapt_single_shard_simple(...)
    tasks.append(task)
results = await asyncio.gather(*tasks, return_exceptions=True)
```

**Impact:**
- Shard A references "Sarah Chen"
- Shard B adapted same person to "Maya Sharma"
- Both use entity_map but may interpret differently
- Result: Name inconsistencies

**If entity_map generation fails (line 1781-1787), ALL shards get empty maps and invent their own names.**

---

### 3.2 Repair Loop Regression

**Location:** `src/graph/nodes.py` lines 512-542

**Problem:** Repairs can make quality WORSE.

```python
# Line 524
MAX_REPAIR_ITERATIONS = 2  # Comment: "Reduced from 3 - over-repair causes regression"

# Line 529-537: Regression detection is BROKEN
if len(history) >= 1:
    last_pre_repair_score = history[-1].get("previous_score", 0)
    if score <= last_pre_repair_score * 0.95:  # Only checks LAST iteration
        # Should compare to FIRST score (before ANY repairs)
```

**Missing:** Comparison to ORIGINAL score before any repairs started.

---

### 3.3 Resource Regeneration Wrong Trigger

**Location:** `src/graph/nodes.py` lines 133-285

**Conflicting thresholds:**
```python
# nodes.py line 138
min_words: int = 500  # Regeneration trigger

# simple_validators.py line 1498
"800-1400 words per resource"  # Validation requirement

# config.py line 337
"resource": {"min": 200, "max": 1500}  # Config setting

# scoped_validators.py line 622
"minimum 300 recommended"  # Another standard
```

**4 different word count standards!**

**Also:** Regeneration uses GPT (line 233-237) but adaptation uses Gemini → style inconsistency.

---

### 3.4 Prompt Bloat - Critical Instructions Buried

**Location:** `src/stages/simple_adapter.py` lines 1135-1547

**Statistics:**
- `build_simple_prompt()` function: 412 lines
- Main prompt template: 168 lines
- Critical sections buried in middle

**Section positions:**
| Section | Line | Position |
|---------|------|----------|
| Domain Transformation | 1405-1432 | Near start (good) |
| Context Fidelity | 1387-1402 | Middle (bad) |
| Inference Map Rules | 1479-1507 | Late (bad) |
| Word Count Limits | 1498-1501 | Single mention (bad) |

**Effect:** LLM follows FIRST instruction (domain transformation) but ignores later constraints.

---

### 3.5 Validation Score Inflation

**Location:** `src/stages/simple_validators.py` throughout

**Problems:**

**1. No severity weighting (line 627-632):**
```python
total_issues = unaligned + wrong_terms
score = max(0.0, (total_questions - total_issues) / total_questions)
# 1 critical issue = 1 minor issue
```

**2. Arbitrary thresholds (line 707-725):**
```python
if total_variations <= 2:
    score = 1.0  # Perfect
elif total_variations <= 4:
    score = 0.9  # Minor
# Why these numbers? Not data-driven
```

**3. Inconsistent pass criteria:**
```python
# Resource Quality: passes at 0.95 (ACCEPTABLE_THRESHOLD)
# Domain Fidelity: passes at 0.98 (PASS_THRESHOLD)
```

---

## 4. Configuration Inconsistencies

### 4.1 Multiple Sources of Truth for Thresholds

**Found in codebase:**

| Location | Variable | Value |
|----------|----------|-------|
| `nodes.py:71` | PASS_THRESHOLD | 0.95 |
| `simple_validators.py:43` | PASS_THRESHOLD | **0.98** |
| `simple_validators.py:44` | ACCEPTABLE_THRESHOLD | 0.95 |
| `config.py:349` | MINIMUM_ACCEPTABLE_THRESHOLD | 0.95 |

**nodes.py uses 0.95 but validators use 0.98 for the same concept!**

---

### 4.2 Word Count Limits Chaos

| Location | Context | Value |
|----------|---------|-------|
| `nodes.py:138` | Regeneration trigger | 500 words |
| `simple_adapter.py:1498` | Prompt to LLM | 800-1400 words |
| `simple_validators.py:327` | Validation check | 800-1400 words |
| `config.py:337` | Config dict | 200-1500 words |
| `scoped_validators.py:622` | Sparse warning | 300 words |

**5 different standards for same requirement!**

---

### 4.3 Hardcoded Values That Should Be Configurable

**Location:** Multiple files

```python
# simple_adapter.py line 644
DEFAULT_MODEL = "gemini-2.5-flash"  # Hardcoded

# simple_adapter.py line 1799
MAX_SHARD_SIZE = 20000  # Magic number

# nodes.py line 72
MAX_REPAIR_ITERATIONS = 2  # Hardcoded

# simple_validators.py line 194
content_text = json.dumps(adapted_json, indent=2)[:50000]  # Truncation limit
```

All should be in `config.py` or environment variables.

---

## 5. Performance Issues

### 5.1 Excessive LLM Calls - No Batching

**Location:** `src/stages/simple_adapter.py` lines 1801-1925

**Problem:** Large shards split into MANY small calls.

```python
# Line 1862: Split by deepkey
for deepkey, deepval in subval.items():
    if deepval_size > SPLIT_THRESHOLD:
        task = _adapt_single_item(...)  # Separate LLM call
```

**Evidence from logs:**
```
[ADAPT ITEM] Simulation Flow (Stages).simulation_flow[1].data.review
[ADAPT ITEM] Simulation Flow (Stages).simulation_flow[1].data.resource
[ADAPT ITEM] Simulation Flow (Stages).simulation_flow[1].batch_0
[ADAPT ITEM] Simulation Flow (Stages).simulation_flow[1].batch_1
# 4 calls for ONE stage
```

**Inefficiency:**
- `MAX_SHARD_SIZE = 20000` (20K chars ≈ 5K tokens)
- Gemini supports 2M tokens input
- Could easily handle 200K char shards in one call

---

### 5.2 Redundant Validation Calls

**Location:** `src/stages/simple_validators.py` lines 1954-2005

```python
# Line 1962
for i in range(max_iter):
    report = await run_all_validators(...)  # ALL 8 agents each iteration
```

**Waste:** Runs ALL 8 validators even if only 1 agent failed.

**Should:** Only re-run failed agents on subsequent iterations.

---

## 6. Error Handling Issues

### 6.1 Silent Failures - Fallbacks Too Permissive

**Location:** `src/stages/simple_adapter.py`

**Entity map failure (line 989-992):**
```python
except Exception as e:
    logger.error(f"[MAPS] Generation failed: {e}")
    return EntityMap({}, {}, {}, {}), DomainProfile(...), [], NumericFactsheet({}, {}, {})
    # Returns EMPTY map, pipeline continues
    # Should FAIL FAST
```

**LLM extraction fallback (line 1046-1054):**
```python
return CompanyContext(
    company_name="Target Company",  # Generic!
    manager_name="Project Manager",  # Generic!
)
# Violates own rule against "the company"
```

---

### 6.2 Missing Error Propagation

**Location:** `src/stages/simple_adapter.py` lines 1938-1999

```python
# Line 1938
results = await asyncio.gather(*tasks, return_exceptions=True)

# Line 1945-1948
if isinstance(result, Exception):
    logger.error(f"[SIMPLE ADAPTER] Task {i} failed: {result}")
    errors.append(f"{shard.id}: {str(result)}")
    # errors list populated but NEVER CHECKED
    # Pipeline continues even if ALL shards failed
```

---

## 7. Testing Gaps

### 7.1 No Validation Unit Tests

**Current test files:**
- `test_pipeline.py` - Integration test only
- `test_3_prompts.py` - Prompt testing

**Missing:**
1. Domain fidelity with known source terms
2. Resource quality with forbidden phrases
3. Completeness with truncated content
4. Consistency with name variations
5. Each validator in isolation
6. Regression tests for fixed bugs

---

## Summary: Issues by Severity

### CRITICAL (P0) - Blocking Quality
| # | Issue | Impact |
|---|-------|--------|
| 2.1 | Empty shards inflate 100x | Massive content explosion |
| 1.3 | Domain term leakage | Source terms in output |
| 1.1 | Weak entity map (3 vs 50 terms) | Incomplete transformation |
| 1.7 | Question-KLO misalignment | Core validation failing |

### HIGH (P1) - Causing Inconsistency
| # | Issue | Impact |
|---|-------|--------|
| 2.2 | Conflicting word count rules | Content explosion |
| 1.4 | Inconsistent numeric values | Different numbers per shard |
| 3.1 | Parallel shards without deps | Name variations |
| 3.2 | Repair regression | Fixes make it worse |

### MEDIUM (P2) - Performance & Reliability
| # | Issue | Impact |
|---|-------|--------|
| 5.1 | Excessive LLM calls | Slow and expensive |
| 3.4 | Prompt bloat | LLM misses instructions |
| 4.1 | Threshold inconsistency | Unpredictable pass/fail |
| 6.1 | Silent failures | Errors hidden |

### LOW (P3) - Technical Debt
| # | Issue | Impact |
|---|-------|--------|
| 4.3 | Hardcoded values | Can't tune without code |
| 7.1 | No unit tests | Can't verify fixes |
| 3.5 | Score inflation | False confidence |

---

## Recommended Fix Order

### Phase 1: Stop the Bleeding (1-2 days)
1. **Skip empty shards** - Fix 2.1 (chat_history 100x explosion)
2. **Remove 800-1400 rule** - Fix 2.2 (conflicting word counts)
3. **Add output length check** - Fix 2.3 (catch all explosions)

### Phase 2: Core Quality (3-5 days)
4. **Deep entity scan** - Fix 1.1 (30-50 forbidden terms)
5. **Anchor shard first** - Fix 3.1 (consistency)
6. **Simplify prompts** - Fix 3.4 (one job per call)

### Phase 3: Validation (2-3 days)
7. **Unify thresholds** - Fix 4.1, 4.2
8. **Full content validation** - Fix 1.5 (no truncation)
9. **Fix regression detection** - Fix 3.2

### Phase 4: Hardening (ongoing)
10. **Move hardcoded values to config** - Fix 4.3
11. **Add unit tests** - Fix 7.1
12. **Improve error handling** - Fix 6.1, 6.2

---

## Appendix: File Reference

| File | Issues Found |
|------|--------------|
| `src/stages/simple_adapter.py` | 1.1, 1.3, 2.1, 2.2, 2.3, 3.1, 3.4, 4.3, 5.1, 6.1, 6.2 |
| `src/stages/simple_validators.py` | 1.2, 1.3, 1.5, 1.6, 1.7, 3.5, 4.1, 5.2 |
| `src/graph/nodes.py` | 3.2, 3.3, 4.1, 4.3 |
| `src/utils/config.py` | 4.1, 4.2 |
| `src/validators/scoped_validators.py` | 4.2 |

---

*Document generated from deep analysis of codebase and test outputs.*
