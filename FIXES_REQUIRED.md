# Fixes Required - Shweta & Rachit Requirements Analysis

## Status Legend
- ✅ DONE - Already implemented/fixed
- ⚠️ PARTIAL - Partially implemented, needs improvement
- ❌ TODO - Not implemented, needs to be built

---

## Critical Fixes (Blocking Issues)

### 1. ✅ DONE - Literal Replacement Bug
**Problem:** `industry_term_map` in `context.py` was doing find-replace:
- "structured interview process" → "organic T-shirts"
- "event coordinator" → "Gen Z"

**Fixed:** Removed `industry_term_map` entirely. LLM now does semantic transformation.

**File:** `src/core/context.py`

---

### 2. ✅ DONE - Company Name Generation
**Problem:** When scenario was "Gen Z organic T-shirts brand", no company name was generated.

**Fixed:** Added `_generate_company_name()` function that creates appropriate names like "EcoThread Co."

**File:** `src/core/context.py`

---

### 3. ❌ TODO - Resource Answerability Validator
**Shweta's Requirement (Dec 22):**
> "does the resource contain all the information the student needs to answer the submission questions"

**What's Needed:**
```python
class ResourceAnswerabilityValidator:
    """
    Check that EVERY submission question can be answered
    using ONLY the resources provided.
    """
    async def validate(self, questions, resources, context):
        # For each question:
        # 1. Check if resources contain data to answer it
        # 2. Flag if external knowledge required
        # 3. Return specific question IDs that fail
```

**Priority:** HIGH - This is a BLOCKER

---

### 4. ❌ TODO - Answer Leakage Validator
**Shweta's Requirement (Dec 22):**
> "the resource does not have the answer... it should basically have all the dots for inference to connect the dots, but it doesn't really give the connected dots"

**What's Needed:**
```python
class AnswerLeakageValidator:
    """
    Check that resources provide DATA for inference,
    but do NOT directly state the answer.

    BAD: "The best strategy is Option B because..."
    GOOD: "Option B has 20% higher ROI, 15% lower risk..."
    """
    async def validate(self, questions, resources, context):
        # For each question:
        # 1. Check if resource directly answers it
        # 2. Flag if conclusion is given instead of data
```

**Priority:** HIGH - This is a BLOCKER

---

### 5. ❌ TODO - Resource Word Count Limit
**Shweta's Requirement (Dec 22):**
> "is the resource within like 1500 words"

**Current State:** `WordCountValidator` exists but doesn't enforce 1500 word limit for resources.

**What's Needed:**
- Update `WORD_COUNT_LIMITS` in `config.py`:
  ```python
  "resource": {"min": 200, "max": 1500},  # Changed from 3000
  ```
- Make it a BLOCKER for resources over 1500 words

**Priority:** MEDIUM

---

### 6. ❌ TODO - Cross-Shard Alignment Validator
**Problem:** When we chunk/shard, we lose global alignment:
- KLOs in one shard
- Questions in another shard
- Resources in another shard

They get processed independently and don't align.

**What's Needed:**
```python
class CrossShardAlignmentValidator:
    """
    Run AFTER all shards processed.
    Check that:
    1. Every KLO has at least one question assessing it
    2. Every question maps to a KLO
    3. Resources support all questions
    4. Same company name used everywhere
    5. Same industry terms used consistently
    """
```

**Priority:** HIGH - This is why chunking broke alignment

---

### 7. ❌ TODO - Human-Readable Dashboard Output
**Shweta's Requirement (Jan 9):**
> "I'm not going to review your JSON Poovendhan. I want you to give me reports and dashboards"
> "pass fail column, and reason why"
> "19 on 20 and the reason and how to fix"

**What's Needed:**
```
═══════════════════════════════════════════════════════════════
              VALIDATION REPORT - Beverage Simulation
═══════════════════════════════════════════════════════════════

Overall Score: 94% (19/20 checks passed)
Status: ⚠️ NEEDS REVIEW

┌─────────────────────────┬────────┬─────────────────────────────┐
│ Check                   │ Status │ Issue                       │
├─────────────────────────┼────────┼─────────────────────────────┤
│ Domain Fidelity         │ ✅     │                             │
│ Context Fidelity        │ ✅     │                             │
│ ID Preservation         │ ✅     │                             │
│ Resource Self-Contained │ ❌     │ Q3 not answerable from data │
│ Answer Leakage          │ ✅     │                             │
│ Word Count              │ ⚠️     │ Resource 2 is 1823 words    │
│ KLO Alignment           │ ✅     │                             │
│ Company Name            │ ✅     │ "ThriveBite Nutrition"      │
└─────────────────────────┴────────┴─────────────────────────────┘

BLOCKERS (must fix):
  1. Q3: "What market entry strategy..." cannot be answered
     from resources. Missing: competitor pricing data.

WARNINGS (review recommended):
  1. Resource 2 exceeds 1500 word limit (1823 words)

SUGGESTIONS:
  1. Add competitor pricing table to Resource 2
  2. Trim Resource 2 by ~300 words
```

**Priority:** HIGH - Shweta explicitly asked for this

---

### 8. ❌ TODO - Enforce 98% Threshold
**Shweta's Requirement (Dec 22):**
> "You basically say you start with 98%, and if you are not getting it, you come down to 95%"

**Current State:** Validators run but don't block at threshold.

**What's Needed:**
```python
# In pipeline after validation:
if validation_score < 0.98:
    # Don't proceed - require fixes
    return {
        "status": "BLOCKED",
        "score": validation_score,
        "message": "Score 94% below 98% threshold",
        "blockers": [...],
        "action_required": "Fix blockers and re-run"
    }
```

**Priority:** MEDIUM

---

### 9. ⚠️ PARTIAL - Inference Integrity Check
**Current State:** `InferenceIntegrityValidator` checks for:
- Ranges like "10-15"
- Placeholders like "TBD", "N/A"
- Vague terms like "approximately"

**What's Missing:**
- Checking that resources provide SPECIFIC numbers, not ranges
- Checking that conclusions are NOT stated (only data)

**Priority:** MEDIUM - Enhance existing validator

---

### 10. ❌ TODO - Agent Glossary Documentation
**Shweta's Requirement (Jan 9):**
> "a very simple table of the Agents that you are using, what is each agent performing and checking for? What is your threshold value?"

**What's Needed:** Create documentation file:
```markdown
# Agent Glossary

| Agent | Tasks | Threshold | Blocker |
|-------|-------|-----------|---------|
| DomainFidelityValidator | Checks industry terms match target | 100% | YES |
| ContextFidelityValidator | KLO/criteria counts match base | 100% | YES |
| ResourceAnswerabilityValidator | Questions answerable from resources | 100% | YES |
| ... | ... | ... | ... |
```

**Priority:** LOW - Documentation for Shweta

---

## Implementation Order

1. **Resource Answerability Validator** (HIGH)
2. **Answer Leakage Validator** (HIGH)
3. **Cross-Shard Alignment Validator** (HIGH)
4. **Human-Readable Dashboard** (HIGH)
5. **Resource 1500 Word Limit** (MEDIUM)
6. **98% Threshold Enforcement** (MEDIUM)
7. **Enhanced Inference Check** (MEDIUM)
8. **Agent Glossary** (LOW)

---

## Already Fixed (Verification Needed)

These were fixed in the last session - need to verify they work:

1. ✅ `context.py` - No more `industry_term_map`
2. ✅ `context.py` - Company name generation works
3. ✅ `smart_prompts.py` - Semantic transformation, not literal replacement
4. ✅ `scoped_validators.py` - 8 validators implemented
5. ✅ `fixers.py` - BatchedSemanticFixer applies fixes

---

## Test Scenarios

After implementing fixes, test with:

1. **Prompt 1:** Gen Z organic T-shirts brand (Sample 1)
2. **Prompt 2:** Fast food $1 menu response (Sample 2)
3. **Prompt 3:** ThriveBite Nutrition beverage (Sample 3)

For each, verify:
- [ ] Company name generated (not just scenario text)
- [ ] Domain terms correct for industry
- [ ] All questions answerable from resources
- [ ] No direct answers in resources
- [ ] Resources under 1500 words
- [ ] KLOs align with questions
- [ ] Dashboard output is human-readable
- [ ] Overall score >= 98%


####################################################################################
\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
!!!!!!!!!!!  UPDATE THE PROGESS BELOW ---- CRITICAL!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
####################################################################################
\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\


## PROGRESS LOG

### Jan 13, 2026 - Session 2

#### ✅ COMPLETED: Decider Retry Loop (THE KEY FIX)

**Problem:** Decider was ONE-SHOT - if it produced garbage, we were stuck with it.

**Solution Implemented in `src/core/decider.py`:**

1. **Added Quick Validation** (`quick_validate_decisions()`)
   - Checks for poison terms in output
   - Checks for old company names remaining
   - Detects nonsensical content (company name > 8 words)
   - Detects empty or identical replacements

2. **Added Retry Loop** (up to 3 attempts)
   - After LLM response, validate output immediately
   - If validation fails, retry with feedback
   - Increase temperature slightly on retry (0.2 → 0.3 → 0.4)
   - Return best effort after max retries

3. **Added Retry Prompt** (`_build_retry_prompt()`)
   - Shows LLM what went wrong
   - Lists specific failed indices and their bad outputs
   - Gives correction instructions
   - Forces LLM to not repeat same mistakes

**New Flow:**
```
┌─────────────────────────────────────────────────────────────────────┐
│  BEFORE (Broken):                                                   │
│  Decider → ONE SHOT → Accept garbage                               │
│                                                                     │
│  AFTER (Fixed):                                                     │
│  Decider → Validate → PASS? → Done                                 │
│              ↓                                                      │
│            FAIL? → Retry with feedback → Validate → ...            │
│              ↓                                                      │
│          (up to 3 attempts)                                         │
└─────────────────────────────────────────────────────────────────────┘
```

---

### Still TODO

| # | Task | Status |
|---|------|--------|
| 1 | Resource Answerability Validator | ❌ TODO |
| 2 | Answer Leakage Validator | ❌ TODO |
| 3 | Cross-Shard Alignment Validator | ❌ TODO |
| 4 | Human-Readable Dashboard | ❌ TODO |
| 5 | Resource 1500 Word Limit | ❌ TODO |
| 6 | 98% Threshold Enforcement | ❌ TODO |
| 7 | Test the retry mechanism | ⏳ PENDING |

