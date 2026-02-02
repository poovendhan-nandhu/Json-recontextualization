Here is **the founder’s architecture**, explained **cleanly, end-to-end, in plain language**, exactly the way **Shweta expects you to understand and implement it**.

I’ll keep this **conceptual + practical**, not theoretical.

---

## 🧠 Founder’s Architecture (Shweta’s Mental Model)

This is **NOT** about models, LangGraph, or tools first.
It’s about **control, safety, and debuggability**.

> **Core idea:**
> *“Safely adapt an existing simulation into a new scenario without breaking anything.”*

---

## 🧩 High-level Flow (What happens, in order)

```
Base Simulation JSON
        ↓
1. Adaptation Engine
        ↓
2. Sharder
        ↓
3. Unified Checker
        ↓
4. Structural Fixers
        ↓
5. Semantic Fixers
        ↓
6. Finisher (Compliance Loop)
        ↓
7. Human Approval
```

---

## 1️⃣ Adaptation Engine (Generator)

**What it does**

* Takes a **base simulation JSON**
* Applies the **new scenario** (industry, company, context)

**Critical rules**

* ❌ Do NOT change structure
* ❌ Do NOT infer simulation type
* ❌ Do NOT redesign logic
   
**Founder phrase**

> “   ”

**Output**

* `adapted_working.json` → first draft
* `entity_map.json` → old → new names mapping

---

## 2️⃣ Sharder (Safety step most people miss)

**What it does**

* Splits the JSON into **independent shards**

  * Scenario
  * Resources
  * Rubric
  * Submission questions
  * Emails, personas, etc.

**Why this exists**

* Prevents whole-JSON rewrites
* Enables **scoped validation and repair**
* Makes chunking actually useful

**Each shard gets**

* A hash
* A lock state

> This is where your earlier latency issue came from
> (you were chunking *after* generation, not *before*)

---

## 3️⃣ Unified Checker (Diagnosis only)

**What it does**

* Runs **many micro checks**
* Does **NOT fix anything**
* Produces a **scorecard**

**Examples of checks**

* Domain fidelity
* Context preservation
* Inference integrity
* Structure integrity
* Data consistency

**Output**

```
Shard: Resources
❌ Inference Integrity failed
❌ Missing numeric driver
✅ Domain Fidelity passed
```

> Founder rule:
> **“Agents should point, not fix.”**

---

## 4️⃣ Structural Fixers (Shape only)

**Golden rule**

> ❌ No wording changes
> ✅ Shape only

**What they fix**

* Missing keys
* Wrong ordering
* Broken rubric levels
* Schema mismatches

**After this step**

* Shard is **locked**
* It cannot be changed again structurally

---

## 5️⃣ Semantic Fixers (Meaning only)

**What they fix**

* Industry-specific KPIs
* Language tone
* Personas
* Resources realism
* Emails sounding right

**Hard limits**

* ❌ Cannot change structure
* ❌ Cannot touch locked shards

> This is where “realism” is handled
> (but never called realism)

---

## 6️⃣ Finisher (Compliance Loop)

**What it does**

* Re-checks **only the shards that changed**
* Applies **global guardrails**
* Computes a **weighted score**

**Pass conditions**

* All blocker rules = **100%**
* Overall score ≥ **98%**

**If it fails**

* Routes back to:

  * Structural Fixer OR
  * Semantic Fixer
* No infinite loops allowed

---

## 7️⃣ Human-in-the-Loop (Very specific role)

**Human does NOT**

* Fix formatting
* Fix missing data
* Fix broken logic

**Human DOES**

* Confirm realism
* Confirm learning quality
* Approve or flag feedback

> “Humans confirm realism, not correctness.”

---

## 🔐 Invisible Safety Systems (Always on)

These are not stages — they are **rules**:

* **Frozen structure**
* **Shard locks**
* **Hash comparisons**
* **No cross-shard mutation**
* **No oscillation loops**

This is why the system is scalable.

---

## 🚨 Most important founder rule (memorize this)

> **You are adapting an existing simulation, not generating a new one.**

Everything flows from this.

---

## Why Shweta told you:

**“Use your model, just score the agents”**

Because:

* You already built a working version
* The missing part was:

  * clear shard boundaries
  * micro-validators
  * explicit scoring

She wanted you to **align execution**, not restart.

---

## One-sentence founder architecture summary

> **“Adapt first, slice second, diagnose precisely, fix narrowly, re-verify safely, then let a human approve.”**

---

If you want next, I can:

* Map **your current pipeline → this architecture**
* Show **exactly where you were drifting**
* Help you rewrite this as a **design doc** you can reuse internally


Implementing Shweta's Architecture
@Poovendhan updates @Shweta on their progress implementing the architecture she shared, including mapping agents to the stages and clarifying the use of prompts.

More details
@Poovendhan reported that the architecture @Shweta shared was really helpful and they would start implementing the method and mapping their agents to the stages outlined [1]
@Poovendhan clarified that when they mentioned prompts earlier, they meant scenarios, not the prompts specifically required [1]
Seeking Alignment on the Framework
@Poovendhan seeks a quick huddle with @Shweta to ensure they are interpreting the fixer stages and validation loops correctly, as they are finding it hard to apply the framework end-to-end without drifting.

More details
@Poovendhan expressed that they were finding it hard to apply the framework end-to-end without drifting, and wanted to ensure they were interpreting the fixer stages and validation loops correctly [2]
@Poovendhan requested a quick huddle with @Shweta to get alignment on how the stages are expected to work together [2]
Using @Poovendhan's Own Model
@Shweta tells @Poovendhan to just use their own model, not Shweta's, and to focus on scoring the agents as requested.

More details
@Shweta told @Poovendhan to just use their own model, not Shweta's [3]
@Shweta asked @Poovendhan to focus on scoring the agents as requested [4]
Finalizing the Simulation Flow
@Poovendhan reports that the end-to-end simulation flow is now wired and stable on their model, with all checks, sharding, locks, and scoped repairs functioning as intended. They are proceeding to finalize the simulation and close remaining gaps.

More details
@Poovendhan reported that the end-to-end simulation flow was now wired and stable on their model, with all checks, sharding, locks, and scoped repairs functioning as intended [5]
@Poovendhan was proceeding to finalize the simulation and close the remaining gaps, working with Rachit [5]
Exploring Chunked/Parallel Generation
@Poovendhan is adjusting the approach to explore chunked/parallel generation for better efficiency, while keeping the overall flow unchanged. They are working on this now and ask @Shweta if it's okay to continue on Ken's simulation as part of the task.

More details
@Poovendhan was adjusting the approach to explore chunked/parallel generation for better efficiency, while keeping the overall flow unchanged [6]
@Poovendhan asked @Shweta if it was okay to continue on Ken's simulation as part of the task [6]
Committing to Completion by Wednesday
@Poovendhan commits to completing the work by Wednesday, focusing on content validity and accuracy, specifically consistent alignment between KLOs, submission questions, and resources, and passing validation checks reliably.

More details
@Poovendhan committed to completing the work by Wednesday, focusing on content validity and accuracy, specifically consistent alignment between KLOs, submission questions, and resources, and passing validation checks reliably [7]
Scheduling a Demo Meeting
@Shweta requests a meeting with @Poovendhan and Rachit the next day to have @Poovendhan show the process and full execution.

More details
@Shweta requested a meeting with @Poovendhan and Rachit the next day to have @Poovendhan show the process and full execution [8]
Summarizing Completed Work and Remaining Issues
@Poovendhan provides a detailed summary of what was completed, including fixing realism KPI issues, removing hardcoded assumptions, and reworking realism checking. They also explain where things went wrong, with optimization attempts causing alignment issues that they were unable to resolve within the timeline.

More details
@Poovendhan summarized the key issues they had fixed, including realism KPI issues, removing hardcoded assumptions, and reworking realism checking [9]
@Poovendhan explained that their optimization attempts had caused alignment issues they were unable to resolve within the timeline [9]
Committing to a Stable Version by Friday
@Poovendhan requests a short huddle before the meeting, as they feel nervous about presenting the current system and have some doubts. They commit to having a stable version running by Friday and will share a clear checkpoint update sooner if anything blocks that.

More details
@Poovendhan requested a short huddle before the meeting, as they felt nervous about presenting the current system and had some doubts [10]
@Poovendhan committed to having a stable version running by Friday and sharing a clear checkpoint update sooner if anything blocked that [10]


Jan 7, 2026
JSON LEASSON PACKAGE GENERATOR  - Transcript
00:00:00
 
Rachit Sharma: Hey Ben, how are you doing?
Poovendhan Velrajan: Hello. Um hi. So I I tried to use the chunking and um
Rachit Sharma: Hi
Poovendhan Velrajan: embeddings uh on Monday. I have shared uh rep so to you. So basically it didn't work for me even though I tried when chunking the latency was very high because my whole architecture were based on monolithic. So whenever I tried to use chunking it basically like called too many validation and too many repeats. So the latency increased again. Sure.
Rachit Sharma: Can you show me the um your land graph for it? What graph did you use?
Poovendhan Velrajan: Sure.
Rachit Sharma: Uh can you show me your land chain?
Poovendhan Velrajan: Sure. I guess you could see the screen here.
Rachit Sharma: Not yet. No, I can't.
Poovendhan Velrajan: Yeah. So basically we have generate node and we have like multiple checkers like uh seven different checkers and like domain fatality and other things here. So I can show you basic output right now we are getting.
 
 
00:10:07
 
Poovendhan Velrajan: So each every time it uh in the first uh it recontext to every like whole JSON right now and it fixes the structure and it checks the domain fatality and uh like other verifications like we have like some different uh verification like data
Rachit Sharma: Can you can you show me how is it running on Lang
Poovendhan Velrajan: uh yeah
Rachit Sharma: Smith?
Poovendhan Velrajan: So it's exactly like know we have one gender. So initially it's like uh does the one shot and it recontest the whole JSON again after that it goes by chunks by chunk and it repairs. the that's what here
Rachit Sharma: Okay,
Poovendhan Velrajan: happening
Rachit Sharma: just give me a second. I'm also opening langu.
Poovendhan Velrajan: sure I guess you have the access for
Rachit Sharma: Yeah,
Poovendhan Velrajan: that
Rachit Sharma: that is what I am trying to open. Which project is
Poovendhan Velrajan: yeah
Rachit Sharma: this?
Poovendhan Velrajan: um so it's lesson simulation package I feel like I burnt the system later. Uh initially I was like trying to do the monolytic so it was like a single shot and it had latency but I got like some solid results like a proper results but right now after like doing the parallelization and uh chunking it messed up the whole system.
 
 
00:11:59
 
Rachit Sharma: I'm also sharing my screen.
Poovendhan Velrajan: Sure.
Rachit Sharma: Let me know if it is visible.
Poovendhan Velrajan: Okay.
Rachit Sharma: This is the one that you're talking about, right?
Poovendhan Velrajan: Oh, give me a second. Yes. Uh, that's the one. The first one. Yeah.
Rachit Sharma: Sorry, I sending the whole JSON here.
Poovendhan Velrajan: Yeah, in the first shot we are like you know uh we are sending whole JSON and uh doing recontextuation.
Rachit Sharma: Where did you get this JSON from?
Poovendhan Velrajan: Uh SWE has shared me a PDF. So basically what I done was like converted that PDF into JSON so I could get a whole package system.
Rachit Sharma: Can you show me the PDF?
Poovendhan Velrajan: Yes sure Yep. This is the PDF I got or should I share the PDF with you? Hello.
Rachit Sharma: Sorry. So you converted this into a JSON, right?
Poovendhan Velrajan: Yeah,
Rachit Sharma: Okay.
Poovendhan Velrajan: exactly.
Rachit Sharma: So you don't need to convert like uh I'll just share the JSON with you.
 
 
00:13:54
 
Rachit Sharma: You just convert the thatJSON.
Poovendhan Velrajan: Sure. Okay.
Rachit Sharma: I'll just share one with you.
Poovendhan Velrajan: Uh I have seen the cartido simulation template. So when I was going through this, it seems to be very different than what I'm doing right now.
Rachit Sharma: Different
Poovendhan Velrajan: It's like total different. So I don't know if I'm going.
Rachit Sharma: to different to what exactly? Like
Poovendhan Velrajan: So basically yeah so basically what set has shared me
Rachit Sharma: it's
Poovendhan Velrajan: the methods like you know like basically we have like sharers basically we convert every chunks into hashable thing and we will log it and every time we like change that part not like a like section by section. So after that we merge everything and give an output over here. So that's what what we're doing here.
Rachit Sharma: Uh,
Poovendhan Velrajan: Yeah.
Rachit Sharma: I'll just share a new file with you.
Poovendhan Velrajan: Yeah.
Rachit Sharma: You don't need to convert anything into a JSON. You will provide be provided with the JSON.
Poovendhan Velrajan: Okay.
 
 
00:15:24
 
Rachit Sharma: I've shared the JSON with you. Try to uh use this JSON to create the structure. Remember you created the same thing in your PC as well.
Poovendhan Velrajan: Yeah,
Rachit Sharma: U the S.
Poovendhan Velrajan: we done. Yeah. So basically we didn't get the proper results over and the P itself right when I was doing in the
Rachit Sharma: So,
Poovendhan Velrajan: interview. So that's what I was like telling you the latency was the same issue we got like we locked
Rachit Sharma: So,
Poovendhan Velrajan: the every you know fields but uh didn't get a proper you know because of the latency we didn't get a proper uh
Rachit Sharma: how do you plan to do
Poovendhan Velrajan: results.
Rachit Sharma: that?
Poovendhan Velrajan: So right now yeah so this the JSON structure over here is very different than what I was like working on.
Rachit Sharma: Okay, just check the JSON structure that I sent you right
Poovendhan Velrajan: So yeah,
Rachit Sharma: now.
Poovendhan Velrajan: so I guess this method works I guess like you know we could like lock the system or recontextrate whole uh chunks of you know like send batch by batch that's what I done in the interview when I was
 
 
00:16:30
 
Rachit Sharma: So, can you see the JSON that I sent you right
Poovendhan Velrajan: doing yeah I could see right now
Rachit Sharma: now? Does it look familiar?
Poovendhan Velrajan: Yeah. So
Rachit Sharma: So, does this help you also?
Poovendhan Velrajan: yeah, I guess it will help me right now. But I'm like little confused on how how to proceed this again with doing that.
Rachit Sharma: Can you run this JSON through your structure?
Poovendhan Velrajan: My structure was very different, right? Should I share my
Rachit Sharma: Yeah.
Poovendhan Velrajan: Yeah. Give me a second. I sent you.
Rachit Sharma: Okay, give me a second. I'm checking right now. So, give me a sec. Give me a minute. I'll be back. I want to just give me a minute.
Poovendhan Velrajan: Yeah, notice. Yeah.
Rachit Sharma: Yeah, I'm back. So,
Poovendhan Velrajan: Yeah.
Rachit Sharma: you've sent me the Yeah,
Poovendhan Velrajan: So I was going through the
Rachit Sharma: you've sent me the uh structure,
Poovendhan Velrajan: Yeah.
Rachit Sharma: the JSON structure.
 
 
00:22:25
 
Rachit Sharma: This looks similar,
Poovendhan Velrajan: Yeah.
Rachit Sharma: but you need to preserve all the ids. For example, the one that I sent you,
Poovendhan Velrajan: Yeah, exactly. Yeah.
Rachit Sharma: uh the JSON that I've sent you, you can see there are different different ids inside it,
Poovendhan Velrajan: Mhm.
Rachit Sharma: right? there's a structure there objects that does and then there are arrays with it you need to preserve those
Poovendhan Velrajan: Yeah. Yeah.
Rachit Sharma: structures so now while you're working with it and u now that you have a new structure work on the P that you had uh given that in your interview and work on it try to use chunking there or embedding might be a long shot but try to use it there and preserve the whole JSON structure okay previously you did
Poovendhan Velrajan: Got it.
Rachit Sharma: not have to work with ids or any other small rules but now you'll have to
Poovendhan Velrajan: Yeah. Got it.
Rachit Sharma: Okay.
Poovendhan Velrajan: Yeah. But we we get to see here like message and sender.
 
 
00:23:19
 
Poovendhan Velrajan: So it's more of like topic with Danny simulation you know.
Rachit Sharma: So the simulation is based on the topic vizard
Poovendhan Velrajan: Oh okay.
Rachit Sharma: data. That is right.
Poovendhan Velrajan: Okay.
Rachit Sharma: So by when can you show it to us with the new
Poovendhan Velrajan: Uh yeah.
Rachit Sharma: JSON?
Poovendhan Velrajan: Uh first I will have like yeah at five I guess I will send you an proper report. I will be using rag.
Rachit Sharma: Can you show that drag
Poovendhan Velrajan: Uh I you I have done the rag but yeah. So
Rachit Sharma: implementation?
Poovendhan Velrajan: sure.
Rachit Sharma: Did it work for
Poovendhan Velrajan: So Okay. Not not um yeah as I told you know uh initially what I was like doing
Rachit Sharma: you?
Poovendhan Velrajan: was monolytic method right so when I tried to include like chunking it uh didn't probably work for me. I guess I have to go back to the ones which I used in the interview and try to use that one right now.
Rachit Sharma: Okay. Okay. Cool.
Poovendhan Velrajan: Yeah,
Rachit Sharma: So, send me by five the implementation and the lang link also.
Poovendhan Velrajan: sure.
Rachit Sharma: Uh where Yeah.
Poovendhan Velrajan: Sure.
Rachit Sharma: Tell me you were saying trying to say something.
Poovendhan Velrajan: Sure. Yeah. Yeah. After um five I can come and meet with you, right?
Rachit Sharma: Yeah.
Poovendhan Velrajan: So, sure. Thank you. Thank you. Got it. Sure. Got it. Raj. Sure.
 
 
Transcription ended after 00:30:34

This editable transcript was computer generated and might contain errors. People can also change the text after it was created.




















 Core Principle (from Shweta):
 "You are adapting an existing simulation, not generating a new one."
 "Same bones, new skin."

 ---
 Current State vs Target State

 | Aspect     | Current (6-Node)        | Target (7-Stage Cartedo)              |
 |------------|-------------------------|---------------------------------------|
 | Processing | Monolithic (whole JSON) | Shard-based (6-8 independent shards)  |
 | Validation | 3 simple checks         | 5+ micro-validators with scorecard    |
 | Repair     | None (retry all)        | Structural + Semantic fixers (scoped) |
 | Scoring    | Pass/Fail               | Weighted: Blockers=100%, Overall≥98%  |
 | Human Loop | None                    | Final approval stage                  |
 | RAG        | None                    | Industry knowledge for semantic fixes |

 ---
 Architecture Overview

 ┌─────────────────────────────────────────────────────────────────────┐
 │                    CARTEDO 7-STAGE PIPELINE                         │
 ├─────────────────────────────────────────────────────────────────────┤
 │                                                                     │
 │  INPUT: Base Simulation JSON + Target Scenario Index                │
 │                           │                                         │
 │                           ▼                                         │
 │  ┌─────────────────────────────────────────────────────────────┐   │
 │  │  STAGE 1: ADAPTATION ENGINE                                  │   │
 │  │  - First-pass transformation with entity mapping             │   │
 │  │  - RAG: Retrieve industry context                            │   │
 │  │  - Output: adapted_working.json + entity_map.json            │   │
 │  └─────────────────────────────────────────────────────────────┘   │
 │                           │                                         │
 │                           ▼                                         │
 │  ┌─────────────────────────────────────────────────────────────┐   │
 │  │  STAGE 2: SHARDER                                            │   │
 │  │  - Split into 6-8 independent shards                         │   │
 │  │  - Compute hash per shard                                    │   │
 │  │  - Initialize lock state = UNLOCKED                          │   │
 │  └─────────────────────────────────────────────────────────────┘   │
 │                           │                                         │
 │                           ▼                                         │
 │  ┌─────────────────────────────────────────────────────────────┐   │
 │  │  STAGE 3: UNIFIED CHECKER                                    │   │
 │  │  - Run 5+ micro-validators per shard                         │   │
 │  │  - Output: Scorecard (pass/fail per rule per shard)          │   │
 │  │  - NO FIXING - diagnosis only                                │   │
 │  └─────────────────────────────────────────────────────────────┘   │
 │                           │                                         │
 │              ┌────────────┴────────────┐                           │
 │              ▼                         ▼                           │
 │  ┌──────────────────────┐  ┌──────────────────────┐                │
 │  │  STAGE 4: STRUCTURAL │  │  STAGE 5: SEMANTIC   │                │
 │  │  FIXERS              │  │  FIXERS              │                │
 │  │  - Fix shape only    │  │  - Fix meaning only  │                │
 │  │  - No wording change │  │  - RAG: Industry KPIs│                │
 │  │  - Lock after fix    │  │  - No structure change│               │
 │  └──────────────────────┘  └──────────────────────┘                │
 │              │                         │                           │
 │              └────────────┬────────────┘                           │
 │                           ▼                                         │
 │  ┌─────────────────────────────────────────────────────────────┐   │
 │  │  STAGE 6: FINISHER (COMPLIANCE LOOP)                         │   │
 │  │  - Re-check only changed shards                              │   │
 │  │  - Weighted scoring: Blockers=100%, Overall≥98%              │   │
 │  │  - Route back to Fixer if fail (max 3 attempts)              │   │
 │  └─────────────────────────────────────────────────────────────┘   │
 │                           │                                         │
 │                           ▼                                         │
 │  ┌─────────────────────────────────────────────────────────────┐   │
 │  │  STAGE 7: HUMAN APPROVAL                                     │   │
 │  │  - Visual diff + scorecard                                   │   │
 │  │  - Human confirms REALISM, not correctness                   │   │
 │  │  - Approve → Ship | Reject → Feedback                        │   │
 │  └─────────────────────────────────────────────────────────────┘   │
 │                           │                                         │
 │                           ▼                                         │
 │  OUTPUT: Golden Adapted Simulation                                  │
 │                                                                     │
 └─────────────────────────────────────────────────────────────────────┘

 ---
 Stage-by-Stage Implementation Details

 STAGE 1: Adaptation Engine

 Purpose: Create first-pass transformation with entity mapping

 File: src/stages/adaptation_engine.py

 Process:
 1. Extract entities from current scenario (regex + NER)
 2. Extract entities from target scenario
 3. Query RAG for industry context (KPIs, terminology, competitors)
 4. Build comprehensive entity_map
 5. Call OpenAI for initial transformation
 6. Force-restore locked fields

 Inputs:
 {
     "input_json": dict,           # Original simulation
     "selected_scenario": int,      # Target scenario index
     "rag_context": dict           # Retrieved industry knowledge
 }

 Outputs:
 {
     "adapted_json": dict,          # First-pass transformed JSON
     "entity_map": dict,            # {old_entity: new_entity, ...}
     "industry_context": dict       # KPIs, terminology from RAG
 }

 RAG Integration:
 # Query ChromaDB for industry knowledge
 industry = detect_industry(target_scenario)  # "hospitality", "airline", etc.
 rag_results = vector_store.similarity_search(
     query=f"{industry} business simulation KPIs metrics terminology",
     k=5
 )

 ---
 STAGE 2: Sharder

 Purpose: Split JSON into independent, hashable shards

 File: src/stages/sharder.py

 Shard Definitions:
 SHARD_DEFINITIONS = {
     "scenario": {
         "paths": [
             "topicWizardData.selectedScenarioOption",
             "topicWizardData.workplaceScenario.scenario"
         ],
         "locked": False,
         "blocker": True  # Must pass 100%
     },
     "background": {
         "paths": [
             "topicWizardData.workplaceScenario.background.*"
         ],
         "locked": False,
         "blocker": True
     },
     "challenge": {
         "paths": [
             "topicWizardData.workplaceScenario.challenge.*"
         ],
         "locked": False,
         "blocker": True
     },
     "learner_role": {
         "paths": [
             "topicWizardData.workplaceScenario.learnerRoleReportingManager.*"
         ],
         "locked": False,
         "blocker": False
     },
     "simulation_flow": {
         "paths": [
             "topicWizardData.simulationFlow.*"
         ],
         "locked": False,
         "blocker": False
     },
     "resources": {
         "paths": [
             "topicWizardData.simulationFlow[*].data.resource",
             "topicWizardData.simulationFlow[*].data.resourceOptions"
         ],
         "locked": False,
         "blocker": False
     },
     "rubric": {
         "paths": [
             "topicWizardData.simulationFlow[*].data.review.rubric"
         ],
         "locked": False,  # Content can change, structure locked
         "blocker": True
     },
     "locked_fields": {
         "paths": [
             "topicWizardData.scenarioOptions",
             "topicWizardData.assessmentCriterion",
             "topicWizardData.selectedAssessmentCriterion",
             "topicWizardData.industryAlignedActivities",
             "topicWizardData.selectedIndustryAlignedActivities"
         ],
         "locked": True,   # NEVER modify
         "blocker": True
     }
 }

 Shard Data Structure:
 @dataclass
 class Shard:
     id: str                    # "scenario", "background", etc.
     paths: list[str]           # JSONPaths covered
     content: dict              # Extracted content
     hash: str                  # SHA-256 of content
     lock_state: Literal["UNLOCKED", "STRUCTURE_LOCKED", "FULLY_LOCKED"]
     is_blocker: bool           # Must pass 100%?
     validation_results: dict   # {rule_name: pass/fail}
     fix_attempts: int          # Track retry count (max 3)

 Output:
 {
     "shards": list[Shard],
     "shard_dependency_graph": dict  # Which shards depend on which
 }

 ---
 STAGE 3: Unified Checker

 Purpose: Diagnose issues per shard with micro-validators (NO FIXING)

 File: src/stages/unified_checker.py

 Micro-Validators:

 | Validator                    | What It Checks                     | Blocker? |
 |------------------------------|------------------------------------|----------|
 | DomainFidelityValidator      | KPIs match industry (RAG-assisted) | Yes      |
 | ContextPreservationValidator | Scenario context maintained        | Yes      |
 | InferenceIntegrityValidator  | Inferred values consistent         | Yes      |
 | StructureIntegrityValidator  | All required fields present        | Yes      |
 | DataConsistencyValidator     | Values align cross-field           | No       |
 | EntityRemovalValidator       | Old entities removed               | Yes      |
 | ToneValidator                | Professional/instructional tone    | No       |
 | EmailFormatValidator         | Emails follow pattern              | No       |

 Validator Interface:
 class BaseValidator(ABC):
     name: str
     is_blocker: bool

     @abstractmethod
     def validate(self, shard: Shard, context: dict) -> ValidationResult:
         """Returns pass/fail + details"""
         pass

 @dataclass
 class ValidationResult:
     rule_name: str
     passed: bool
     shard_id: str
     details: str
     severity: Literal["blocker", "warning", "info"]
     suggested_fix: Optional[str]

 Scorecard Output:
 {
     "shards": {
         "scenario": {
             "domain_fidelity": {"passed": True},
             "context_preservation": {"passed": True},
             "entity_removal": {"passed": False, "details": "Found 'HarvestBowls' in line 42"}
         },
         "background": {
             "domain_fidelity": {"passed": False, "details": "KPI 'foot traffic' invalid for hospitality"},
             ...
         }
     },
     "summary": {
         "total_checks": 48,
         "passed": 45,
         "failed": 3,
         "blocker_failures": 2,
         "overall_score": 0.9375
     }
 }

 ---
 STAGE 4: Structural Fixers

 Purpose: Fix SHAPE only, no wording changes

 File: src/stages/structural_fixer.py

 What Gets Fixed:
 - Missing required keys
 - Wrong key ordering
 - Broken rubric levels (e.g., only 3 of 5 ratings)
 - Schema mismatches (array vs object)
 - Empty required fields

 What Does NOT Get Fixed:
 - Wording/text content
 - Tone issues
 - KPI values
 - Industry terminology

 Golden Rule:
 # ALLOWED
 fix_missing_key(shard, "organizationName", default="[PLACEHOLDER]")
 fix_array_order(shard, "starRatings", sort_key="rating")
 fix_schema_type(shard, "scopeOfWork", target_type=list)

 # NOT ALLOWED
 change_text(shard, "organizationName", "HarvestBowls", "BlueHaven")  # NO!

 After Fix:
 shard.lock_state = "STRUCTURE_LOCKED"  # Cannot change structure again
 shard.hash = compute_hash(shard.content)

 ---
 STAGE 5: Semantic Fixers

 Purpose: Fix MEANING only, using RAG for industry knowledge

 File: src/stages/semantic_fixer.py

 What Gets Fixed:
 - Industry-specific KPIs (foot traffic → occupancy rate)
 - Language tone (casual → professional)
 - Persona names/emails
 - Resource realism (menu items → room packages)
 - Email content sounding authentic

 What Does NOT Get Fixed:
 - Structure (handled by Stage 4)
 - Locked shards (cannot touch)

 RAG-Assisted Fixing:
 async def fix_kpis(shard: Shard, industry: str, rag_store: VectorStore):
     """Replace KPIs with industry-appropriate ones"""

     # Find invalid KPIs
     invalid_kpis = find_invalid_kpis(shard, industry)

     for kpi in invalid_kpis:
         # Query RAG for replacement
         results = rag_store.similarity_search(
             f"What KPI replaces '{kpi}' in {industry} industry?",
             k=3
         )
         replacement = extract_best_kpi(results)

         # Apply replacement
         shard.content = replace_text(shard.content, kpi, replacement)

     return shard

 Hard Constraints:
 if shard.lock_state == "FULLY_LOCKED":
     raise LockedShardError(f"Cannot modify locked shard: {shard.id}")

 if shard.lock_state != "STRUCTURE_LOCKED":
     raise OrderError("Structural fixer must run before semantic fixer")

 ---
 STAGE 6: Finisher (Compliance Loop)

 Purpose: Re-verify changed shards, compute weighted score, route back if needed

 File: src/stages/finisher.py

 Process:
 def compliance_loop(shards: list[Shard], max_iterations: int = 3):
     for iteration in range(max_iterations):
         # 1. Identify changed shards (hash comparison)
         changed_shards = [s for s in shards if s.hash != s.original_hash]

         # 2. Re-run Unified Checker on changed shards only
         scorecard = unified_checker.validate(changed_shards)

         # 3. Compute weighted score
         score = compute_weighted_score(scorecard)

         # 4. Check pass conditions
         if score.blocker_pass_rate == 1.0 and score.overall >= 0.98:
             return {"status": "PASS", "score": score, "shards": shards}

         # 5. Route to appropriate fixer
         for shard in changed_shards:
             if shard.has_structural_failures():
                 structural_fixer.fix(shard)
                 shard.fix_attempts += 1
             elif shard.has_semantic_failures():
                 semantic_fixer.fix(shard)
                 shard.fix_attempts += 1

             # Anti-oscillation: max 3 attempts per shard
             if shard.fix_attempts >= 3:
                 shard.status = "FLAGGED_FOR_HUMAN"

     return {"status": "NEEDS_HUMAN", "score": score, "shards": shards}

 Weighted Scoring:
 @dataclass
 class ComplianceScore:
     blocker_pass_rate: float   # Must be 1.0
     overall_score: float       # Must be >= 0.98
     shard_scores: dict         # Per-shard breakdown

 def compute_weighted_score(scorecard: dict) -> ComplianceScore:
     blocker_checks = [c for c in scorecard if c.is_blocker]
     blocker_pass_rate = sum(1 for c in blocker_checks if c.passed) / len(blocker_checks)

     # Weighted by severity
     weights = {"blocker": 2.0, "warning": 1.0, "info": 0.5}
     weighted_sum = sum(c.weight * (1 if c.passed else 0) for c in scorecard)
     overall = weighted_sum / sum(c.weight for c in scorecard)

     return ComplianceScore(blocker_pass_rate, overall, ...)

 ---
 STAGE 7: Human Approval

 Purpose: Human confirms REALISM, not correctness

 File: src/stages/human_approval.py

 What Human Sees:
 @dataclass
 class ApprovalPackage:
     simulation_id: str
     summary: str                    # 2-3 sentence summary of changes
     compliance_score: float         # Overall score
     shard_report: list[ShardSummary]
     visual_diff: str                # Side-by-side diff
     flagged_items: list[FlaggedItem]  # Items needing attention
     approve_url: str
     reject_url: str

 Human Decision Points:
 - Realism: "Does this feel like a real hotel simulation?"
 - Learning quality: "Would students learn the right skills?"
 - Tone: "Is the language appropriate?"

 Human Does NOT:
 - Fix formatting
 - Fix missing data
 - Fix broken logic

 API Endpoints:
 @router.post("/api/v1/approval/{simulation_id}/approve")
 async def approve_simulation(simulation_id: str, reviewer: str):
     """Human approves - simulation ships"""

 @router.post("/api/v1/approval/{simulation_id}/reject")
 async def reject_simulation(simulation_id: str, feedback: str):
     """Human rejects - feedback stored for training"""

 ---
 RAG System Design

 Data Flow Clarification

 Source Data: You receive simulation JSON files from Cartedo (like sample_input.json)
 - Contains topicWizardData with all simulation content
 - Has scenarioOptions (37 different industry scenarios)
 - Transformation is scenario-to-scenario within the same JSON structure

 RAG Purpose: Enhance transformation with industry-specific context
 - NOT for storing source simulation data (that comes from Cartedo JSON)
 - FOR: KPI mappings, terminology, industry-specific validation rules

 Vector Database: ChromaDB (Local Persistent)

 File: src/rag/vector_store.py

 Storage: ./chroma_db/ (persisted locally)

 Collections:
 COLLECTIONS = {
     "industry_mappings": {
         "description": "KPI and terminology mappings between industries",
         "embedding_model": "text-embedding-3-small"
     },
     "validation_context": {
         "description": "Industry-specific validation rules and realistic values",
         "embedding_model": "text-embedding-3-small"
     }
 }

 Auto-Generated from Scenarios:
 The RAG knowledge can be bootstrapped from the scenarioOptions in the simulation JSON:
 # Extract industry context from scenario text
 scenarios = input_json["topicWizardData"]["scenarioOptions"]

 # Each scenario contains industry hints:
 # "BlueHaven Hotels faces occupancy challenges..." → hospitality
 # "SkyLink Airlines sees bookings dip..." → airline
 # "TechNova faces slowing sales..." → technology

 # Build industry mapping from scenario patterns
 industry_contexts = extract_industry_contexts(scenarios)

 Retrieval Functions:
 async def get_kpi_mapping(source_industry: str, target_industry: str) -> dict:
     """Get KPI translation between industries"""
     # foot traffic (restaurant) → occupancy rate (hotel)

 async def get_industry_validation_rules(industry: str) -> list:
     """Get realistic value ranges for industry"""
     # hospitality: occupancy 60-90%, ADR $100-$500

 ---
 File Structure

 src/
 ├── main.py                          # FastAPI app
 ├── api/
 │   ├── routes.py                    # API endpoints
 │   └── approval_routes.py           # Human approval endpoints
 ├── stages/                          # 7-stage pipeline
 │   ├── __init__.py
 │   ├── adaptation_engine.py         # Stage 1
 │   ├── sharder.py                   # Stage 2
 │   ├── unified_checker.py           # Stage 3
 │   ├── structural_fixer.py          # Stage 4
 │   ├── semantic_fixer.py            # Stage 5
 │   ├── finisher.py                  # Stage 6
 │   └── human_approval.py            # Stage 7
 ├── validators/                       # Micro-validators
 │   ├── __init__.py
 │   ├── base.py                      # BaseValidator ABC
 │   ├── domain_fidelity.py
 │   ├── context_preservation.py
 │   ├── inference_integrity.py
 │   ├── structure_integrity.py
 │   ├── data_consistency.py
 │   ├── entity_removal.py
 │   ├── tone_validator.py
 │   └── email_format.py
 ├── rag/                              # RAG system
 │   ├── __init__.py
 │   ├── vector_store.py              # ChromaDB client
 │   ├── embeddings.py                # OpenAI embeddings
 │   ├── retriever.py                 # Retrieval functions
 │   └── documents/                   # Industry knowledge docs
 │       ├── hospitality.json
 │       ├── airline.json
 │       ├── retail.json
 │       ├── technology.json
 │       └── ...
 ├── models/
 │   ├── schemas.py                   # Pydantic models
 │   ├── shard.py                     # Shard dataclass
 │   └── scorecard.py                 # Scorecard models
 ├── graph/                            # LangGraph workflow
 │   ├── state.py                     # Extended state
 │   └── workflow.py                  # 7-stage graph
 ├── utils/
 │   ├── config.py                    # Configuration
 │   ├── helpers.py                   # Utility functions
 │   ├── openai_client.py             # OpenAI wrapper
 │   └── hash.py                      # Hashing utilities
 └── safety/                           # Safety systems
     ├── __init__.py
     ├── shard_locks.py               # Shard lock management
     ├── anti_oscillation.py          # Prevent infinite loops
     └── global_guard.py              # Final regression check

 ---
 Implementation Order

 Phase 1: Foundation (Day 1-2)

 1. Create src/stages/ directory structure
 2. Implement Sharder (Stage 2) - most critical
 3. Create Shard dataclass and utilities
 4. Update state management for shard-based flow

 Phase 2: Validators (Day 2-3)

 5. Create src/validators/ with BaseValidator
 6. Implement 5 core validators:
   - EntityRemovalValidator
   - StructureIntegrityValidator
   - DomainFidelityValidator (basic, no RAG yet)
   - ContextPreservationValidator
   - DataConsistencyValidator
 7. Implement Unified Checker (Stage 3)

 Phase 3: Fixers (Day 3-4)

 8. Implement Structural Fixer (Stage 4)
 9. Implement Semantic Fixer (Stage 5) - basic version
 10. Implement Finisher loop (Stage 6)

 Phase 4: RAG Integration (Day 4-5)

 11. Set up ChromaDB
 12. Create industry knowledge documents
 13. Implement retrieval functions
 14. Enhance Semantic Fixer with RAG
 15. Enhance DomainFidelityValidator with RAG

 Phase 5: Human Loop & Polish (Day 5-6)

 16. Implement Human Approval (Stage 7)
 17. Add approval API endpoints
 18. Create visual diff generation
 19. Add comprehensive logging
 20. Write tests

 ---
 Key Files to Modify

 | File                  | Changes                                      |
 |-----------------------|----------------------------------------------|
 | src/graph/state.py    | Add shard-related fields                     |
 | src/graph/workflow.py | Rewrite for 7-stage flow                     |
 | src/graph/nodes.py    | Deprecate, move to src/stages/               |
 | src/utils/config.py   | Add shard definitions, RAG config            |
 | src/models/schemas.py | Add Shard, Scorecard, ApprovalPackage models |
 | src/api/routes.py     | Add approval endpoints                       |
 | requirements.txt      | Add chromadb, langchain                      |

 ---
 Safety Systems (Always Running)

 # src/safety/shard_locks.py
 class ShardLockManager:
     """Manages lock state transitions"""

     VALID_TRANSITIONS = {
         "UNLOCKED": ["STRUCTURE_LOCKED", "FULLY_LOCKED"],
         "STRUCTURE_LOCKED": ["FULLY_LOCKED"],
         "FULLY_LOCKED": []  # Cannot unlock
     }

     def can_modify(self, shard: Shard, modification_type: str) -> bool:
         if shard.lock_state == "FULLY_LOCKED":
             return False
         if modification_type == "structure" and shard.lock_state == "STRUCTURE_LOCKED":
             return False
         return True

 # src/safety/anti_oscillation.py
 class AntiOscillationGuard:
     """Prevents infinite fix loops"""

     MAX_ATTEMPTS_PER_SHARD = 3

     def can_retry(self, shard: Shard) -> bool:
         return shard.fix_attempts < self.MAX_ATTEMPTS_PER_SHARD

 # src/safety/global_guard.py
 class GlobalGuard:
     """Final regression check before output"""

     def check(self, original: dict, final: dict) -> bool:
         # 1. All locked fields unchanged
         # 2. Overall structure preserved
         # 3. No empty required fields
         # 4. All shards accounted for
         pass

 ---
 Success Criteria

 1. Latency: < 30 seconds for full pipeline (vs current 4+ minutes)
 2. Accuracy: Blocker pass rate = 100%
 3. Quality: Overall score >= 98%
 4. Safety: Zero locked field modifications
 5. Human Effort: < 5 minutes review time per simulation

 ---
 User Decisions (Confirmed)

 | Decision       | Choice                                       |
 |----------------|----------------------------------------------|
 | Source Data    | JSON files provided by Cartedo (not created) |
 | RAG Data       | Auto-bootstrap from scenarioOptions in JSON  |
 | Human Approval | API only (endpoints for approve/reject)      |
 | Vector Storage | Local persistent ChromaDB (./chroma_db/)     |

 ---
 Ready to Implement

 The architecture is complete. Implementation will follow this order:

 1. Sharder (Stage 2) - Foundation for shard-based processing
 2. Unified Checker (Stage 3) - Micro-validators with scorecard
 3. Structural Fixer (Stage 4) - Shape-only repairs
 4. Semantic Fixer (Stage 5) - Meaning repairs with RAG
 5. Finisher Loop (Stage 6) - Compliance verification
 6. Human Approval API (Stage 7) - Approve/reject endpoints
 7. RAG Integration - ChromaDB for industry context