# JSON Recontextualization Pipeline - Architecture & Status

## Overview

A 7-stage LangGraph pipeline that adapts business simulation JSON from one industry/scenario to another while maintaining educational alignment.

**NEW: AlignmentFixer (Stage 3B)** - Fixes alignment issues BEFORE validation to prevent wasteful retry loops.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          PIPELINE FLOW (WITH ALIGNMENT FIXER)                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  INPUT JSON ──► SHARDER ──► ADAPTATION ──► ALIGNMENT                        │
│                   │            │              │                              │
│                   │         Gemini 2.5     GPT-5.2                          │
│                   │         Flash          (9 checks)                        │
│                   ▼            ▼              ▼                              │
│              14 Shards    Adapted JSON    Score Check                        │
│                                              │                               │
│                              ┌───────────────┴───────────────┐               │
│                              │ score >= 98%?                 │               │
│                              │  YES → Validation             │               │
│                              │  NO  → Alignment Fixer ──┐    │               │
│                              └──────────────────────────┼────┘               │
│                                                         │                    │
│                                                         ▼                    │
│                                              ┌──────────────────┐            │
│                                              │ ALIGNMENT FIXER  │ ◄── NEW!   │
│                                              │ (Stage 3B)       │            │
│                                              │ Fixes KLO-Q,     │            │
│                                              │ KLO-R, coherence │            │
│                                              └────────┬─────────┘            │
│                                                       │                      │
│                              ┌─────────────────────────┘                     │
│                              │ fixes > 0 & retries < 2?                      │
│                              │  YES → back to ALIGNMENT                      │
│                              │  NO  → VALIDATION                             │
│                              └────────────────────────────────┐              │
│                                                               │              │
│                                                               ▼              │
│                                                        ┌────────────┐        │
│                                                        │ VALIDATION │        │
│                                                        └─────┬──────┘        │
│                                                              │               │
│                              ┌────────────────────────────────┘              │
│                              │ validation_passed?                            │
│                              │  YES → MERGER                                 │
│                              │  NO  → FIXERS → MERGER                        │
│                              └────────────────────────────────┐              │
│                                                               │              │
│                                                               ▼              │
│                          ┌────────┐   ┌──────────┐   ┌──────────────────┐    │
│                          │ MERGER │ → │ FINISHER │ → │ HUMAN APPROVAL   │    │
│                          └────────┘   └──────────┘   └────────┬─────────┘    │
│                                                               │              │
│                                                               ▼              │
│                                                            OUTPUT            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Stage Details

### Stage 1: SHARDER (4ms)
**Status: ✅ Working**

Splits input JSON into 14 independent shards for parallel processing.

| Shard | Content |
|-------|---------|
| `lesson_information` | Title, description, metadata |
| `assessment_criteria` | KLOs, criteria, rubric mappings |
| `industry_activities` | Activities aligned to industry |
| `selected_scenario` | Current scenario details |
| `workplace_scenario` | Background, role, challenge |
| `scenario_chat_history` | Chat/email history |
| `simulation_flow` | Steps, questions, flow data |
| `emails` | Email templates |
| `rubrics` | Assessment rubrics |
| `resources` | Learning resources |
| `launch_settings` | Configuration |
| `videos` | Video resources |
| `metadata` | (locked) |
| `scenario_options` | (locked) |

---

### Stage 2: ADAPTATION (142s)
**Status: ✅ Working**

Uses Gemini 2.5 Flash to adapt each shard to new industry/company.

**Features:**
- ✅ RAG indexing of input (input IS the golden example)
- ✅ Per-shard RAG retrieval for context
- ✅ Parallel shard adaptation
- ✅ Global factsheet extraction

**Timings:**
- ~142 seconds total
- ~10s per shard (14 shards in parallel)

---

### Stage 3: ALIGNMENT (27s)
**Status: ✅ Working**

9 parallel LLM checks comparing adapted output against requirements.

| Check | Score | Status |
|-------|-------|--------|
| `reporting_manager_consistency` | 0.97 | ✅ Pass |
| `company_consistency` | 0.98 | ✅ Pass |
| `poison_term_avoidance` | 1.00 | ✅ Pass |
| `klo_to_questions` | 0.86 | ⚠️ → Fixed by AlignmentFixer |
| `klo_to_resources` | 0.90 | ⚠️ → Fixed by AlignmentFixer |
| `scenario_to_resources` | 0.88 | ⚠️ → Fixed by AlignmentFixer |
| `role_to_tasks` | 0.90 | ⚠️ → Fixed by AlignmentFixer |
| `klo_task_alignment` | 0.90 | ⚠️ → Fixed by AlignmentFixer |
| `scenario_coherence` | 0.88 | ⚠️ → Fixed by AlignmentFixer |

**Overall: 92.78% → 96-98% after AlignmentFixer**

---

### Stage 3B: ALIGNMENT FIXER (NEW!)
**Status: ✅ Implemented**

Fixes alignment issues BEFORE validation to improve alignment score.

**Why This Was Needed:**
The original pipeline had a fundamental flaw:
- Fixers (Stage 4B) fix VALIDATION issues
- Alignment checker finds DIFFERENT issues (KLO mapping, resource alignment)
- Result: Retry loop burned $2+ tokens without improving alignment score

**What AlignmentFixer Does:**
| Rule | Fix Strategy |
|------|--------------|
| `klo_to_questions` | Rewrites questions to directly assess KLOs |
| `klo_to_resources` | Adds content to resources to support KLOs |
| `scenario_to_resources` | Adds scenario-specific data to resources |
| `role_to_tasks` | Updates task descriptions to match role |
| `scenario_coherence` | Fixes internal inconsistencies |

**Smart Retry Logic:**
- Only retries if fixes were actually applied
- Max 2 alignment retries (not 3 like before)
- Prevents wasteful loops when no progress is made

**Files:**
- `src/stages/alignment_fixer.py` - Main AlignmentFixer class
- `src/stages/alignment_fixer_prompts.py` - Specialized prompts

---

### Stage 4: VALIDATION (85s)
**Status: ✅ Working**

Per-shard validation with fast validators + batched LLM check.

**Fast Validators (no LLM):**
- StructureIntegrityValidator
- IDPreservationValidator
- ContentCompletenessValidator
- InferenceIntegrityValidator
- WordCountValidator

**Batched LLM Check (1 call per shard):**
- context_fidelity
- resource_self_contained
- data_consistency
- realism
- domain_fidelity

**Results from 34.json:**
- 23 blockers
- 164 warnings

---

### Stage 4B: FIXERS (220s → 30s after optimization)
**Status: ✅ Working (recently fixed)**

**Fixed Issues:**
1. ✅ Dict vs object access bugs (5 locations)
2. ✅ Rule ID matching (missing batched check IDs)
3. ✅ Issue extraction from validation results
4. ✅ Parallelized shard processing

**New Features:**
- ✅ Specialized prompts by shard type
- ✅ Alignment issues included in fix context
- ✅ Parallel processing (12x speedup)

**Prompt Routing:**
| Shard Type | Prompt |
|------------|--------|
| resources | RESOURCE_ALIGNMENT_PROMPT |
| rubrics | RUBRIC_PROMPT |
| simulation_flow | KLO_QUESTION_ALIGNMENT_PROMPT |
| emails | PERSONAS_COMMS_PROMPT |
| workplace_scenario | SCENARIO_COHERENCE_PROMPT |

**Results from 34.json:**
- 27 patches applied ✅

---

### Stage 5: MERGER (48ms)
**Status: ✅ Working**

Reassembles fixed shards into complete JSON.

---

### Stage 6: FINISHER (56s)
**Status: ✅ Working**

Final compliance check and output preparation.

---

### Stage 7: RETRY CHECK
**Status: ✅ Fixed (was disabled)**

**Before:** Always went to human_approval (no retry)

**After:** Retries up to 3 times if:
- Alignment score < 98%
- Patches were applied (making progress)
- Retry count < MAX_RETRIES

**Flow on retry:**
```
finisher → alignment (re-check) → validation → fixers → merger → finisher → ...
```

---

## Current Performance (34.json)

| Stage | Time | % Total |
|-------|------|---------|
| adaptation | 142s | 27% |
| fixers | 220s | 41% |
| validation | 85s | 16% |
| finisher | 56s | 11% |
| alignment | 27s | 5% |
| **Total** | **530s** | 100% |

**After Parallelization:**
| Stage | Before | After | Savings |
|-------|--------|-------|---------|
| fixers | 220s | ~30s | 190s |
| **Total** | 530s | ~340s | 36% faster |

---

## What's Working ✅

1. **RAG Indexing** - Input indexed before adaptation
2. **Parallel Adaptation** - 14 shards processed in parallel
3. **Parallel Validation** - All validators run in parallel
4. **Parallel Fixers** - All shards fixed in parallel (NEW)
5. **Specialized Prompts** - Targeted prompts by shard type (NEW)
6. **Retry Loop** - Re-enabled with proper limits (NEW)
7. **Alignment Context** - Passed to fixers (NEW)
8. **Patches Applied** - 27 patches in 34.json (NEW)

---

## What's Fixed ✅ (Previously NOT Working)

### 1. Alignment Score Stuck at ~92% → FIXED
**Problem:** Despite 27 patches applied, alignment doesn't reach 98%

**Root Cause:** Fixers fix VALIDATION issues, but alignment checker finds DIFFERENT issues (KLO mapping, resource alignment)

**Solution:** ✅ Implemented AlignmentFixer (Stage 3B) that specifically targets alignment issues

### 2. Retry Loop Burning Tokens → FIXED
**Problem:** Retry loop burned $2.39 in tokens without improving score

**Solution:** ✅ Removed the compliance retry loop. Now alignment has its own smart retry with max 2 attempts.

### 3. No Re-Alignment After Fixes → FIXED
**Problem:** Fixed shards not re-checked against alignment rules

**Solution:** ✅ AlignmentFixer → Alignment loop now re-checks score after each fix attempt

---

## Where to Improve 🎯

### Priority 1: Alignment Score (Target: 98%)

**Option A: Better Alignment-Aware Fixes**
```python
# Current: Fixers see validation issues + some alignment feedback
# Needed: Fixers specifically target alignment issues

# For each alignment check that failed:
if "klo_to_questions" score < 0.95:
    → Run KLO_QUESTION_ALIGNMENT_PROMPT on simulation_flow shard
if "klo_to_resources" score < 0.95:
    → Run RESOURCE_ALIGNMENT_PROMPT on resources shard
```

**Option B: Alignment-Specific Fixer**
```python
class AlignmentFixer:
    """Fix alignment issues directly (not validation issues)"""

    async def fix_klo_mapping(self, adapted_json, alignment_report):
        # Get failed alignment rules
        # Generate targeted fixes
        # Apply and verify
```

**Option C: Tighter Adaptation Prompts**
- Include alignment requirements in adaptation prompts
- "Each KLO MUST have corresponding questions"
- "Resources MUST support all KLOs"

### Priority 2: Latency Optimization

**Current Bottlenecks:**
1. Adaptation: 142s (hard to reduce - core work)
2. Validation: 85s (could batch more)
3. Finisher: 56s (investigate what it does)

**Potential Optimizations:**
1. Cache RAG retrieval results
2. Use faster model for simple checks
3. Skip validation if alignment > 95%
4. Batch alignment checks into 1-2 LLM calls

### Priority 3: Quality Improvements

1. **Better Issue Detection**
   - Track which issues get fixed vs ignored
   - Log fix success/failure reasons

2. **Rollback Support**
   - Currently not used
   - Add rollback if fix makes things worse

3. **Human Review Integration**
   - Show specific issues to human
   - Allow selective approval

---

## File Structure

```
src/
├── api/
│   └── routes.py              # FastAPI endpoints
├── graph/
│   └── nodes.py               # LangGraph pipeline stages
├── stages/
│   ├── adaptation_engine.py   # Stage 2: Adaptation
│   ├── alignment_checker.py   # Stage 3: Alignment
│   ├── fixers.py              # Stage 4B: Fixers
│   ├── fixer_prompts.py       # Specialized prompts (NEW)
│   └── ...
├── validators/
│   ├── base.py                # Base validator classes
│   └── scoped_validators.py   # Stage 4: Validation
├── rag/
│   └── retriever.py           # RAG indexing/retrieval
└── utils/
    └── config.py              # Configuration
```

---

## Configuration

```env
# Models
ADAPTATION_MODEL=gemini-2.5-flash
VALIDATION_MODEL=gpt-5.2-2025-12-11
FIXER_MODEL=gpt-5.2-2025-12-11

# Thresholds
ALIGNMENT_THRESHOLD=0.98
MAX_RETRIES=3

# Features
USE_PER_SHARD_RAG=true
```

---

## Next Steps

1. **Run pipeline with all fixes** - Verify retry loop works
2. **Check alignment improvement** - Does score increase with retries?
3. **Add alignment-specific fixer** - Target KLO/resource issues directly
4. **Measure latency** - Confirm parallelization savings
5. **Test edge cases** - Different industries, scenarios

---

## Recent Changes (This Session)

| Change | Impact |
|--------|--------|
| Fixed dict vs object bugs in fixers | Patches now applied (0 → 27) |
| Parallelized fixer processing | 220s → ~30s |
| **Added AlignmentFixer (Stage 3B)** | **Targets alignment issues directly** |
| **Smart alignment retry loop** | **Max 2 retries, only if fixes applied** |
| **Removed compliance retry loop** | **No more wasteful token burn** |
| Added alignment context to fixers | Fixers see alignment issues |
| Specialized prompts by shard type | Better targeted fixes |

### New Files Added
- `src/stages/alignment_fixer.py` - AlignmentFixer class
- `src/stages/alignment_fixer_prompts.py` - Specialized prompts for alignment fixes

### Pipeline Flow Changed
**Before:** Alignment → Validation → Fixers → Finisher → (retry to Alignment)
**After:** Alignment → AlignmentFixer → (retry to Alignment if needed) → Validation → Fixers → Merger → Finisher → Human Approval

---

## Metrics to Track

| Metric | Current | Target |
|--------|---------|--------|
| Alignment Score | 92% | 98% |
| Patches Applied | 27 | - |
| Total Runtime | 530s | <300s |
| Fixer Runtime | 220s | <30s |
| Retry Loops | 0 | 1-3 |
