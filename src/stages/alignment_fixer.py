"""
Stage 3B: Alignment Fixer

Fixes ALIGNMENT issues specifically (not validation issues).

The key insight: Fixers were fixing validation issues, but alignment checker
finds DIFFERENT issues (KLO mapping, resource alignment, scenario coherence).
This was causing the retry loop to burn tokens without improving alignment score.

This AlignmentFixer specifically targets alignment-specific problems:
- KLO-Question mapping (klo_to_questions)
- KLO-Resource support (klo_to_resources)
- Scenario-Resource alignment (scenario_to_resources)
- Role-Task alignment (role_to_tasks)
- Scenario coherence (scenario_coherence)

Runs AFTER alignment check but BEFORE validation.
"""

import os
import re
import json
import copy
import asyncio
import logging
from typing import Any
from dataclasses import dataclass, field

import httpx
from langchain_openai import ChatOpenAI
from langsmith import traceable

# Global semaphore for controlling concurrent LLM calls
MAX_CONCURRENT_LLM_CALLS = int(os.getenv("MAX_CONCURRENT_LLM_CALLS", "10"))
_llm_semaphore = None  # Will be initialized lazily in async context


def _get_semaphore():
    """Get or create the semaphore (must be called in async context)."""
    global _llm_semaphore
    if _llm_semaphore is None:
        _llm_semaphore = asyncio.Semaphore(MAX_CONCURRENT_LLM_CALLS)
    return _llm_semaphore

from .alignment_fixer_prompts import (
    KLO_QUESTION_FIX_PROMPT,
    KLO_RESOURCE_FIX_PROMPT,
    SCENARIO_RESOURCE_FIX_PROMPT,
    ROLE_TASK_FIX_PROMPT,
    SCENARIO_COHERENCE_FIX_PROMPT,
)

logger = logging.getLogger(__name__)

# Model for alignment fixing
ALIGNMENT_FIXER_MODEL = os.getenv("ALIGNMENT_FIXER_MODEL", "gpt-5.2-2025-12-11")


def _get_alignment_fixer_llm():
    """
    Get OpenAI client for alignment fixing.

    CRITICAL: Each call creates a NEW httpx.AsyncClient to enable TRUE parallel execution.
    Without this, all LLM calls share the same connection and run sequentially.
    """
    # Create NEW http client for each LLM instance - enables true parallelism
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(300.0, connect=30.0),  # 5 min read, 30s connect
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=50),  # Higher limits for parallelism
    )

    return ChatOpenAI(
        model=ALIGNMENT_FIXER_MODEL,
        temperature=0.2,  # Slightly higher for creative fixes
        max_retries=3,
        request_timeout=300,
        http_async_client=http_client,  # CRITICAL: Use separate HTTP client
        api_key=os.getenv("OPENAI_API_KEY"),
    )


async def _invoke_with_semaphore(llm, prompt: str) -> str:
    """
    Invoke LLM with semaphore for controlled parallelism.

    This is the KEY to true parallel execution:
    1. Semaphore limits concurrent requests (prevents serialization)
    2. Each call gets its own HTTP client (via _get_llm())
    3. asyncio.gather then runs them truly in parallel
    """
    semaphore = _get_semaphore()
    async with semaphore:
        logger.debug(f"[PARALLEL] Acquired semaphore, invoking LLM...")
        result = await llm.ainvoke(prompt)
        return result.content if hasattr(result, 'content') else str(result)


@dataclass
class AlignmentFixResult:
    """Result of an alignment fix attempt."""
    rule_id: str
    fixes_applied: list[dict] = field(default_factory=list)
    changes_made: list[str] = field(default_factory=list)
    score_before: float = 0.0
    success: bool = True
    error: str = None

    def to_dict(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "fixes_applied": len(self.fixes_applied),
            "changes": self.changes_made[:10],
            "score_before": self.score_before,
            "success": self.success,
            "error": self.error,
        }


class AlignmentFixer:
    """
    Fixes ALIGNMENT issues specifically.

    Unlike regular fixers (which fix validation issues),
    this fixer targets alignment-specific problems:
    - KLO-Question mapping
    - KLO-Resource support
    - Scenario coherence
    - Role-Task alignment

    Key difference from validation fixers:
    - Uses alignment_report (not validation_report)
    - Fixes are targeted at improving alignment SCORE
    - Runs AFTER alignment check, BEFORE validation
    """

    # Threshold for considering a rule as "failed"
    RULE_FAIL_THRESHOLD = 0.95

    def __init__(self):
        # Don't create shared LLM - each parallel task gets its own
        pass

    def _get_llm(self):
        """Get a NEW LLM instance for each parallel task (avoids rate limiting on shared instance)."""
        return _get_alignment_fixer_llm()

    @traceable(name="alignment_fixer")
    async def fix(
        self,
        adapted_json: dict,
        alignment_report: dict,
        global_factsheet: dict,
    ) -> tuple[dict, list[AlignmentFixResult]]:
        """
        Fix alignment issues in the adapted JSON using BATCHED approach.

        Strategy:
        1. FAST fixes first (string replacements, no LLM): manager/company consistency
        2. BATCHED LLM fix: all KLO/resource/coherence issues in ONE call

        Args:
            adapted_json: The adapted simulation JSON
            alignment_report: Report from alignment checker
            global_factsheet: Context for fixes (company, industry, KLOs)

        Returns:
            (fixed_json, list of AlignmentFixResult)
        """
        results = []
        fixed_json = copy.deepcopy(adapted_json)

        # Get failed rules (score < threshold) - safely handle non-dict inputs
        alignment_report = alignment_report if isinstance(alignment_report, dict) else {}
        rule_results = alignment_report.get("results", []) if isinstance(alignment_report.get("results"), list) else []
        failed_rules = [
            r for r in rule_results
            if isinstance(r, dict) and r.get("score", 1.0) < self.RULE_FAIL_THRESHOLD
        ]

        if not failed_rules:
            logger.info("No alignment issues to fix (all rules passed)")
            return adapted_json, [AlignmentFixResult(
                rule_id="none",
                changes_made=["No alignment fixes needed - all rules passed"],
                success=True,
            )]

        logger.info(f"AlignmentFixer: {len(failed_rules)} failed rules")
        print(f"[ALIGNMENT FIXER] Processing {len(failed_rules)} failed rules")

        # Debug: show all failed rules
        for r in failed_rules:
            if isinstance(r, dict):
                print(f"[ALIGNMENT FIXER] Failed rule: {r.get('rule_id', 'unknown')} = {r.get('score', 0):.2%}")

        # =================================================================
        # PHASE 1: FAST FIXES (no LLM) - manager/company consistency
        # =================================================================
        fast_fix_rules = ["reporting_manager_consistency", "company_consistency", "poison_term_avoidance"]
        llm_fix_rules = []

        for rule in failed_rules:
            if not isinstance(rule, dict):
                continue
            rule_id = rule.get("rule_id", "unknown")
            score = rule.get("score", 0)
            issues = rule.get("issues", []) if isinstance(rule.get("issues"), list) else []

            logger.info(f"  - {rule_id}: {score:.2%} ({len(issues)} issues)")

            if rule_id in fast_fix_rules:
                # Run fast string-replacement fixes immediately
                if rule_id == "reporting_manager_consistency":
                    _, fixed_json, result = await self._fix_manager_consistency(
                        fixed_json, issues, "", global_factsheet, score
                    )
                    results.append(result)
                elif rule_id == "company_consistency":
                    _, fixed_json, result = await self._fix_company_consistency(
                        fixed_json, issues, "", global_factsheet, score
                    )
                    results.append(result)
                # poison_term_avoidance handled by semantic fixers
            else:
                # Collect for batched LLM fix
                llm_fix_rules.append(rule)

        # =================================================================
        # PHASE 2: BATCHED LLM FIX (ONE call for all KLO/resource issues)
        # =================================================================
        if llm_fix_rules:
            print(f"[ALIGNMENT FIXER] Running BATCHED LLM fix for {len(llm_fix_rules)} rules")
            logger.info(f"Running BATCHED LLM fix for {len(llm_fix_rules)} rules (1 LLM call instead of {len(llm_fix_rules)})")
            batched_result = await self._fix_all_batched(
                json_data=fixed_json,
                failed_rules=llm_fix_rules,
                factsheet=global_factsheet,
            )
            if batched_result:
                rule_id, fixed_json, batch_results = batched_result
                results.extend(batch_results if isinstance(batch_results, list) else [batch_results])
                print(f"[ALIGNMENT FIXER] Batched fix returned {len(batch_results) if isinstance(batch_results, list) else 1} results")
        else:
            print("[ALIGNMENT FIXER] No LLM fix rules to process")

        # Log summary
        total_changes = sum(len(r.changes_made) for r in results if r.changes_made)
        logger.info(f"AlignmentFixer complete: {len(results)} rules processed, {total_changes} changes made")

        return fixed_json, results

    async def _fix_all_batched(
        self,
        json_data: dict,
        failed_rules: list,
        factsheet: dict,
    ) -> tuple[str, dict, list[AlignmentFixResult]]:
        """
        BATCHED FIX: Fix ALL alignment issues in PARALLEL batches.

        Fixes:
        1. Submission questions (KLO alignment)
        2. Guidelines/overview (poison terms, domain alignment)
        3. Resources (industry-specific content)
        4. Workplace scenario text (coherence)
        """
        # Extract context - safely handle non-dict inputs
        json_data = json_data if isinstance(json_data, dict) else {}
        factsheet = factsheet if isinstance(factsheet, dict) else {}
        topic_data = json_data.get("topicWizardData", {}) if isinstance(json_data.get("topicWizardData"), dict) else {}
        company_obj = factsheet.get("company", {}) if isinstance(factsheet.get("company"), dict) else {}
        company_name = company_obj.get("name", "the company") if isinstance(company_obj, dict) else "the company"
        industry = company_obj.get("industry", "business") if isinstance(company_obj, dict) else "business"
        poison_list = factsheet.get("poison_list", []) if isinstance(factsheet.get("poison_list"), list) else []

        # Collect all issues by category
        question_issues = []
        content_issues = []  # Guidelines, overview, scenario text
        resource_issues = []

        for rule in (failed_rules if isinstance(failed_rules, list) else []):
            if not isinstance(rule, dict):
                continue
            rule_id = rule.get("rule_id", "")
            issues_list = rule.get("issues", []) if isinstance(rule.get("issues"), list) else []
            for issue in issues_list:
                desc = issue.get("description", "") if isinstance(issue, dict) else str(issue)
                suggestion = issue.get("suggestion", "") if isinstance(issue, dict) else ""
                issue_text = f"[{rule_id}] {desc}\n  Suggestion: {suggestion}"

                if rule_id in ["klo_to_questions", "klo_task_alignment"]:
                    question_issues.append(issue_text)
                elif rule_id in ["klo_to_resources", "scenario_to_resources"]:
                    resource_issues.append(issue_text)
                else:
                    content_issues.append(issue_text)

        # Run fixes in PARALLEL
        fix_tasks = []

        # Task 1: Fix questions if needed
        if question_issues:
            fix_tasks.append(self._fix_questions_batch(json_data, topic_data, question_issues, company_name, industry, factsheet))

        # Task 2: Fix content (guidelines, overview, scenario) if needed
        if content_issues or poison_list:
            fix_tasks.append(self._fix_content_batch(json_data, topic_data, content_issues, company_name, industry, poison_list, factsheet))

        # Task 3: Fix resources if needed
        if resource_issues:
            fix_tasks.append(self._fix_resources_batch(json_data, topic_data, resource_issues, company_name, industry, factsheet))

        if not fix_tasks:
            return ("batched", json_data, [AlignmentFixResult(
                rule_id="batched",
                success=True,
                changes_made=["No batched fixes needed"],
            )])

        # Run all fix tasks in parallel
        print(f"[ALIGNMENT FIXER] Running {len(fix_tasks)} batched fix tasks in PARALLEL")
        results = await asyncio.gather(*fix_tasks, return_exceptions=True)

        # Merge results
        fixed_json = copy.deepcopy(json_data)
        all_results = []
        all_changes = []

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Batched fix task failed: {result}")
                continue

            task_name, task_json, task_results = result

            # Merge the fixed content back
            if task_json and isinstance(task_json, dict) and "topicWizardData" in task_json:
                task_topic = task_json.get("topicWizardData", {}) if isinstance(task_json.get("topicWizardData"), dict) else {}

                # Merge specific sections
                for key in ["submissionQuestions", "overview", "guidelines", "resources",
                           "lessonInformation", "workplaceScenario"]:
                    if key in task_topic:
                        fixed_json["topicWizardData"][key] = task_topic[key]

            if isinstance(task_results, list):
                all_results.extend(task_results)
                for r in task_results:
                    all_changes.extend(r.changes_made)
            else:
                all_results.append(task_results)
                all_changes.extend(task_results.changes_made)

        logger.info(f"Batched fix complete: {len(all_changes)} total changes")
        print(f"[ALIGNMENT FIXER] All changes made: {all_changes[:10]}")  # Debug
        print(f"[ALIGNMENT FIXER] All results count: {len(all_results)}")

        return ("batched", fixed_json, all_results or [AlignmentFixResult(
            rule_id="batched",
            success=True,
            changes_made=all_changes or ["No changes made"],
        )])

    async def _fix_questions_batch(
        self,
        json_data: dict,
        topic_data: dict,
        issues: list,
        company_name: str,
        industry: str,
        factsheet: dict,
    ) -> tuple[str, dict, AlignmentFixResult]:
        """Fix submission questions for KLO alignment."""
        # Extract KLOs - safely handle non-dict inputs
        topic_data = topic_data if isinstance(topic_data, dict) else {}

        # DEBUG: Log topic_data structure
        topic_keys = list(topic_data.keys())[:20]
        print(f"[ALIGNMENT FIXER] Questions: topic_data keys = {topic_keys}")
        sim_flow = topic_data.get("simulationFlow", [])
        print(f"[ALIGNMENT FIXER] Questions: simulationFlow has {len(sim_flow)} stages")
        klos = []
        assessment_criteria = topic_data.get("assessmentCriterion", []) if isinstance(topic_data.get("assessmentCriterion"), list) else []
        for criterion in assessment_criteria:
            if not isinstance(criterion, dict):
                continue
            klo_text = criterion.get("keyLearningOutcome", "")
            if klo_text:
                klos.append({"id": criterion.get("id", ""), "klo": klo_text})

        # Extract submission questions from ALL locations
        questions = []
        questions_by_id = {}  # Track by ID to avoid duplicates

        # 1. Top-level submissionQuestions
        submission_questions = topic_data.get("submissionQuestions", []) if isinstance(topic_data.get("submissionQuestions"), list) else []
        for q in submission_questions:
            if not isinstance(q, dict):
                continue
            q_id = q.get("id", "")
            q_text = q.get("question", "")
            if q_id and q_id not in questions_by_id:
                questions_by_id[q_id] = {"id": q_id, "question": q_text, "location": "submissionQuestions"}

        # 2. Inside simulationFlow stages
        sim_flow = topic_data.get("simulationFlow", []) if isinstance(topic_data.get("simulationFlow"), list) else []
        for stage_idx, stage in enumerate(sim_flow):
            if not isinstance(stage, dict):
                continue
            stage_data = stage.get("data", {}) if isinstance(stage.get("data"), dict) else {}
            stage_name = stage.get("name", f"stage_{stage_idx}")

            # Check data.submissionQuestions
            stage_sq = stage_data.get("submissionQuestions", []) if isinstance(stage_data.get("submissionQuestions"), list) else []
            for q in stage_sq:
                if not isinstance(q, dict):
                    continue
                q_id = q.get("id", "")
                q_text = q.get("question", "")
                if q_id and q_id not in questions_by_id:
                    questions_by_id[q_id] = {"id": q_id, "question": q_text, "location": f"simulationFlow/{stage_name}/submissionQuestions"}

            # Check data.questions
            stage_q = stage_data.get("questions", []) if isinstance(stage_data.get("questions"), list) else []
            for q in stage_q:
                if not isinstance(q, dict):
                    continue
                q_id = q.get("id", "")
                q_text = q.get("question", q.get("text", ""))
                if q_id and q_id not in questions_by_id:
                    questions_by_id[q_id] = {"id": q_id, "question": q_text, "location": f"simulationFlow/{stage_name}/questions"}

        questions = list(questions_by_id.values())
        print(f"[ALIGNMENT FIXER] Found {len(questions)} questions total from all locations")

        issues_text = "\n\n".join(issues)

        # Build a list of valid IDs for the prompt
        valid_ids = [q.get("id", "") for q in questions]

        prompt = f"""You are fixing KLO-Question alignment in a business simulation for {company_name} ({industry}).

## CURRENT KLOs (Key Learning Outcomes) - Questions MUST assess these:
{json.dumps(klos, indent=2)}

## CURRENT SUBMISSION QUESTIONS (with their EXACT IDs):
{json.dumps(questions, indent=2)}

## ALIGNMENT ISSUES TO FIX:
{issues_text}

## YOUR TASK:
AGGRESSIVELY rewrite the submission questions so they DIRECTLY assess each KLO.

**WHAT GOOD KLO-ALIGNED QUESTIONS LOOK LIKE**:
- If KLO mentions "market sizing" → Question asks for TAM/SAM calculations with sources
- If KLO mentions "competitive analysis" → Question asks to compare 3+ competitors with data
- If KLO mentions "financial viability" → Question asks for CAC, LTV, gross margin projections
- If KLO mentions "risk assessment" → Question asks to identify 3+ risks with mitigations
- If KLO mentions "recommendation" → Question asks for go/no-go decision with criteria

**WHAT TO AVOID**:
- Generic rubric questions like "Clarity of Thought", "Critical Thinking"
- Meta-evaluation questions about "approach" or "methodology"
- Questions that don't require specific data/analysis

**CRITICAL RULES**:
1. You MUST use the EXACT IDs from the list above. Valid IDs are: {json.dumps(valid_ids)}
2. Do NOT generate new IDs - copy them EXACTLY
3. REPLACE generic questions with KLO-specific assessment questions
4. Each question should map to ONE specific KLO requirement
5. Questions should require specific deliverables (numbers, tables, decisions)

Return ONLY a JSON array with the EXACT IDs from above:
[{{"id": "<one of the valid IDs above>", "question": "Rewritten KLO-specific question..."}}]"""

        try:
            content = await _invoke_with_semaphore(self._get_llm(), prompt)

            # DEBUG: Log LLM response
            print(f"[ALIGNMENT FIXER] Questions LLM response length: {len(content)} chars")

            json_match = re.search(r'\[[\s\S]*\]', content)

            if not json_match:
                print(f"[ALIGNMENT FIXER] Questions: No JSON array found in response")
                print(f"[ALIGNMENT FIXER] Questions RAW response (first 500 chars): {content[:500]}")
                return ("questions", json_data, AlignmentFixResult(
                    rule_id="questions_batch", success=False, changes_made=["No valid JSON"]))

            fixed_questions = json.loads(json_match.group())
            print(f"[ALIGNMENT FIXER] Questions: Parsed {len(fixed_questions)} fixed questions")
            fixed_json = copy.deepcopy(json_data)
            changes = []

            # Use questions_by_id which has location info from ALL sources
            existing_ids = list(questions_by_id.keys())
            print(f"[ALIGNMENT FIXER] Questions: Existing IDs = {existing_ids[:10]}")

            def apply_question_fix_at_location(json_obj: dict, q_id: str, new_text: str, location: str) -> bool:
                """Apply fix to question at the specified location."""
                topic = json_obj.get("topicWizardData", {})

                if location == "submissionQuestions":
                    for q in topic.get("submissionQuestions", []):
                        if q.get("id") == q_id:
                            q["question"] = new_text
                            return True
                elif location.startswith("simulationFlow/"):
                    # Inside simulationFlow - parse location
                    parts = location.split("/")
                    stage_name = parts[1] if len(parts) > 1 else ""
                    q_type = parts[2] if len(parts) > 2 else "submissionQuestions"

                    for stage in topic.get("simulationFlow", []):
                        s_name = stage.get("name", "")
                        # Match by stage name or if stage_name starts with "stage_"
                        if s_name == stage_name or (stage_name.startswith("stage_") and s_name):
                            stage_data = stage.get("data", {})
                            q_list = stage_data.get(q_type, [])
                            for q in q_list:
                                if q.get("id") == q_id:
                                    if "question" in q:
                                        q["question"] = new_text
                                    else:
                                        q["text"] = new_text
                                    return True
                return False

            for idx, fixed_q in enumerate(fixed_questions):
                q_id = fixed_q.get("id", "")
                new_q = fixed_q.get("question", "")
                print(f"[ALIGNMENT FIXER] Questions: Trying to apply fix for ID '{q_id}'")

                if not new_q:
                    continue

                # Try exact match using questions_by_id which has location
                if q_id in questions_by_id:
                    location = questions_by_id[q_id].get("location", "submissionQuestions")
                    old_q = questions_by_id[q_id].get("question", "")
                    if old_q != new_q:
                        success = apply_question_fix_at_location(fixed_json, q_id, new_q, location)
                        if success:
                            changes.append(f"Question {q_id}: updated at {location}")
                            print(f"[ALIGNMENT FIXER] Questions: Applied fix for {q_id} at {location}")
                        else:
                            print(f"[ALIGNMENT FIXER] Questions: Failed to apply fix for {q_id}")
                    else:
                        print(f"[ALIGNMENT FIXER] Questions: No change for {q_id} (same text)")
                else:
                    # FALLBACK: Try position-based matching
                    print(f"[ALIGNMENT FIXER] Questions: ID '{q_id}' not found, trying position fallback...")
                    if idx < len(existing_ids):
                        fallback_id = existing_ids[idx]
                        location = questions_by_id[fallback_id].get("location", "submissionQuestions")
                        old_q = questions_by_id[fallback_id].get("question", "")
                        if old_q != new_q:
                            success = apply_question_fix_at_location(fixed_json, fallback_id, new_q, location)
                            if success:
                                changes.append(f"Question {fallback_id}: updated via fallback")
                                print(f"[ALIGNMENT FIXER] Questions: Applied FALLBACK fix for {fallback_id}")

            return ("questions", fixed_json, AlignmentFixResult(
                rule_id="questions_batch",
                success=len(changes) > 0,
                changes_made=changes or ["No question changes needed"],
            ))
        except Exception as e:
            logger.error(f"Questions batch fix failed: {e}")
            return ("questions", json_data, AlignmentFixResult(
                rule_id="questions_batch", success=False, error=str(e)))

    async def _fix_content_batch(
        self,
        json_data: dict,
        topic_data: dict,
        issues: list,
        company_name: str,
        industry: str,
        poison_list: list,
        factsheet: dict,
    ) -> tuple[str, dict, AlignmentFixResult]:
        """Fix guidelines, overview, and scenario text for poison terms and domain alignment."""
        # Extract content to fix - safely handle non-dict inputs
        topic_data = topic_data if isinstance(topic_data, dict) else {}
        overview = topic_data.get("overview", "")
        guidelines = topic_data.get("guidelines", "")
        lesson_info = topic_data.get("lessonInformation", {}) if isinstance(topic_data.get("lessonInformation"), dict) else {}

        # Also check workplaceScenario text
        workplace = topic_data.get("workplaceScenario", {}) if isinstance(topic_data.get("workplaceScenario"), dict) else {}
        scenario_text = workplace.get("scenario", "") if isinstance(workplace, dict) else ""
        background = workplace.get("background", {}) if isinstance(workplace.get("background"), dict) else {}
        about_org = background.get("aboutOrganization", "") if isinstance(background, dict) else ""

        issues_text = "\n\n".join(issues) if issues else "Check for poison terms"

        prompt = f"""You are fixing content alignment in a business simulation for {company_name} ({industry}).

## POISON TERMS TO REMOVE (terms from OLD scenario that must NOT appear):
{json.dumps(poison_list[:30], indent=2)}

## CURRENT CONTENT TO FIX:

### Overview:
{overview[:2000] if overview else "N/A"}

### Guidelines:
{guidelines[:2000] if guidelines else "N/A"}

### Lesson Information:
{json.dumps(lesson_info, indent=2)[:2000] if lesson_info else "N/A"}

### Scenario Text:
{scenario_text[:1500] if scenario_text else "N/A"}

### About Organization:
{about_org[:1500] if about_org else "N/A"}

## ISSUES:
{issues_text}

## YOUR TASK:
1. REMOVE all poison terms - replace with {industry}-appropriate equivalents
2. Ensure ALL content refers to {company_name} and {industry}
3. Replace domain-specific jargon from old scenario with new industry terms
4. Keep the same structure and formatting

Return JSON with fixes:
{{
    "overview": "Fixed overview text (or null if no changes)",
    "guidelines": "Fixed guidelines text (or null if no changes)",
    "lessonInformation": {{}},  // Fixed lesson info object (or null)
    "scenario": "Fixed scenario text (or null if no changes)",
    "aboutOrganization": "Fixed about text (or null if no changes)",
    "changes": ["list of changes made"]
}}"""

        try:
            content = await _invoke_with_semaphore(self._get_llm(), prompt)

            # DEBUG: Log LLM response
            print(f"[ALIGNMENT FIXER] Content LLM response length: {len(content)} chars")

            # Extract JSON
            json_match = re.search(r'\{[\s\S]*\}', content)
            if not json_match:
                print(f"[ALIGNMENT FIXER] Content: No JSON found in response")
                print(f"[ALIGNMENT FIXER] Content RAW response (first 500 chars): {content[:500]}")
                return ("content", json_data, AlignmentFixResult(
                    rule_id="content_batch", success=False, changes_made=["No valid JSON"]))

            fixes = json.loads(json_match.group())
            fixed_json = copy.deepcopy(json_data)
            changes = []

            # Apply fixes
            if fixes.get("overview") and fixes["overview"] != overview:
                fixed_json["topicWizardData"]["overview"] = fixes["overview"]
                changes.append("Fixed overview: removed poison terms")

            if fixes.get("guidelines") and fixes["guidelines"] != guidelines:
                fixed_json["topicWizardData"]["guidelines"] = fixes["guidelines"]
                changes.append("Fixed guidelines: removed poison terms")

            if fixes.get("lessonInformation") and isinstance(fixes["lessonInformation"], dict):
                fixed_json["topicWizardData"]["lessonInformation"] = fixes["lessonInformation"]
                changes.append("Fixed lessonInformation: updated for new domain")

            if fixes.get("scenario") and fixes["scenario"] != scenario_text:
                if "workplaceScenario" not in fixed_json["topicWizardData"]:
                    fixed_json["topicWizardData"]["workplaceScenario"] = {}
                fixed_json["topicWizardData"]["workplaceScenario"]["scenario"] = fixes["scenario"]
                changes.append("Fixed scenario text: removed poison terms")

            if fixes.get("aboutOrganization") and fixes["aboutOrganization"] != about_org:
                if "workplaceScenario" not in fixed_json["topicWizardData"]:
                    fixed_json["topicWizardData"]["workplaceScenario"] = {}
                if "background" not in fixed_json["topicWizardData"]["workplaceScenario"]:
                    fixed_json["topicWizardData"]["workplaceScenario"]["background"] = {}
                fixed_json["topicWizardData"]["workplaceScenario"]["background"]["aboutOrganization"] = fixes["aboutOrganization"]
                changes.append("Fixed aboutOrganization: removed poison terms")

            # Add LLM-reported changes
            if fixes.get("changes"):
                changes.extend(fixes["changes"][:5])

            return ("content", fixed_json, AlignmentFixResult(
                rule_id="content_batch",
                success=len(changes) > 0,
                changes_made=changes or ["No content changes needed"],
            ))
        except Exception as e:
            logger.error(f"Content batch fix failed: {e}")
            return ("content", json_data, AlignmentFixResult(
                rule_id="content_batch", success=False, error=str(e)))

    async def _fix_resources_batch(
        self,
        json_data: dict,
        topic_data: dict,
        issues: list,
        company_name: str,
        industry: str,
        factsheet: dict,
    ) -> tuple[str, dict, AlignmentFixResult]:
        """Fix resources for KLO support and industry alignment."""
        # DEBUG: Log what's in topic_data
        topic_keys = list(topic_data.keys()) if isinstance(topic_data, dict) else []
        print(f"[ALIGNMENT FIXER] Resources: topic_data keys = {topic_keys[:15]}")

        # Extract KLOs
        klos = []
        assessment_criteria = topic_data.get("assessmentCriterion", []) if isinstance(topic_data.get("assessmentCriterion"), list) else []
        for criterion in assessment_criteria:
            if not isinstance(criterion, dict):
                continue
            klo_text = criterion.get("keyLearningOutcome", "")
            if klo_text:
                klos.append({"id": criterion.get("id", ""), "klo": klo_text})

        # Extract resources (simplified)
        resources = topic_data.get("resources", []) if isinstance(topic_data.get("resources"), list) else []
        print(f"[ALIGNMENT FIXER] Resources: Found {len(resources)} resources at top level")
        resources_simplified = []
        for r in resources[:10]:
            if not isinstance(r, dict):
                continue
            resources_simplified.append({
                "id": r.get("id", ""),
                "title": r.get("title", r.get("name", "")),
                "content_preview": r.get("content", r.get("markdownText", r.get("body", "")))[:500],
            })

        issues_text = "\n\n".join(issues)

        # Build list of valid resource IDs
        valid_resource_ids = [r.get("id", "") for r in resources_simplified]

        prompt = f"""You are fixing resource alignment in a business simulation for {company_name} ({industry}).

## KLOs (Key Learning Outcomes) - Resources must support these:
{json.dumps(klos, indent=2)}

## CURRENT RESOURCES (with their EXACT IDs):
{json.dumps(resources_simplified, indent=2)}

## ISSUES:
{issues_text}

## YOUR TASK:
1. Add content to resources so they support the KLOs
2. Include SPECIFIC data, numbers, facts for {company_name} ({industry})
3. Ensure students can answer KLO-related questions using resource data

**CRITICAL RULES**:
- You MUST use the EXACT IDs from the list above. Valid IDs are: {json.dumps(valid_resource_ids)}
- Do NOT generate new IDs - copy them EXACTLY as shown above

Return JSON array of resource updates with EXACT IDs from above:
[{{"id": "<one of the valid IDs above>", "content_to_append": "Specific content to add..."}}]"""

        try:
            content = await _invoke_with_semaphore(self._get_llm(), prompt)

            # DEBUG: Log LLM response
            print(f"[ALIGNMENT FIXER] Resources LLM response length: {len(content)} chars")

            json_match = re.search(r'\[[\s\S]*\]', content)

            if not json_match:
                print(f"[ALIGNMENT FIXER] Resources: No JSON array found in response")
                print(f"[ALIGNMENT FIXER] Resources RAW response (first 500 chars): {content[:500]}")
                return ("resources", json_data, AlignmentFixResult(
                    rule_id="resources_batch", success=False, changes_made=["No valid JSON"]))

            updates = json.loads(json_match.group())
            fixed_json = copy.deepcopy(json_data)
            changes = []

            # Create lookup
            resource_map = {r.get("id"): i for i, r in enumerate(resources)}
            existing_ids = list(resource_map.keys())

            print(f"[ALIGNMENT FIXER] Resources: Parsed {len(updates)} updates")
            print(f"[ALIGNMENT FIXER] Resources: Existing IDs = {existing_ids[:10]}")

            for idx, update in enumerate(updates):
                r_id = update.get("id", "")
                content_to_add = update.get("content_to_append", "")
                print(f"[ALIGNMENT FIXER] Resources: Trying to apply fix for ID '{r_id}', has content: {bool(content_to_add)}")

                if not content_to_add:
                    print(f"[ALIGNMENT FIXER] Resources: No content to add for '{r_id}'")
                    continue

                # Try exact match first
                if r_id in resource_map:
                    target_idx = resource_map[r_id]
                    resource = fixed_json["topicWizardData"]["resources"][target_idx]
                    # Try markdownText first, then content, then body
                    content_field = "markdownText" if "markdownText" in resource else ("content" if "content" in resource else "body")
                    existing = resource.get(content_field, "")
                    resource[content_field] = existing + "\n\n" + content_to_add
                    changes.append(f"Resource {r_id}: added KLO-supporting content")
                    print(f"[ALIGNMENT FIXER] Updated resource {r_id} ({content_field})")
                else:
                    # FALLBACK: Try position-based matching if LLM returned wrong IDs
                    print(f"[ALIGNMENT FIXER] Resources: ID '{r_id}' not found, trying position fallback...")
                    if idx < len(resources):
                        fallback_id = existing_ids[idx] if idx < len(existing_ids) else None
                        if fallback_id and fallback_id in resource_map:
                            target_idx = resource_map[fallback_id]
                            resource = fixed_json["topicWizardData"]["resources"][target_idx]
                            content_field = "markdownText" if "markdownText" in resource else ("content" if "content" in resource else "body")
                            existing = resource.get(content_field, "")
                            resource[content_field] = existing + "\n\n" + content_to_add
                            changes.append(f"Resource {fallback_id}: added content via position fallback")
                            print(f"[ALIGNMENT FIXER] Updated resource {fallback_id} via FALLBACK (was '{r_id}')")

            return ("resources", fixed_json, AlignmentFixResult(
                rule_id="resources_batch",
                success=len(changes) > 0,
                changes_made=changes or ["No resource changes needed"],
            ))
        except Exception as e:
            logger.error(f"Resources batch fix failed: {e}")
            return ("resources", json_data, AlignmentFixResult(
                rule_id="resources_batch", success=False, error=str(e)))

    def _get_fix_coroutine(
        self,
        rule_id: str,
        score: float,
        issues: list,
        feedback: str,
        json_data: dict,
        factsheet: dict,
    ):
        """
        Get a coroutine (NOT awaited) for a specific rule fix.
        Returns None if no handler exists for the rule.
        """
        # Route to appropriate fix method based on rule_id
        # NOTE: These return coroutines (not awaited) for parallel execution
        if rule_id == "klo_to_questions":
            return self._fix_klo_questions(json_data, issues, feedback, factsheet, score)

        elif rule_id == "klo_to_resources":
            return self._fix_klo_resources(json_data, issues, feedback, factsheet, score)

        elif rule_id == "scenario_to_resources":
            return self._fix_scenario_resources(json_data, issues, feedback, factsheet, score)

        elif rule_id == "role_to_tasks":
            return self._fix_role_tasks(json_data, issues, feedback, factsheet, score)

        elif rule_id == "scenario_coherence":
            return self._fix_coherence(json_data, issues, feedback, factsheet, score)

        elif rule_id == "reporting_manager_consistency":
            return self._fix_manager_consistency(json_data, issues, feedback, factsheet, score)

        elif rule_id == "company_consistency":
            return self._fix_company_consistency(json_data, issues, feedback, factsheet, score)

        elif rule_id == "poison_term_avoidance":
            # Poison terms should be caught by semantic fixers earlier
            logger.info(f"Skipping {rule_id} - handled by semantic fixers")
            return None

        elif rule_id == "klo_task_alignment":
            # This is essentially the same as klo_to_questions
            return self._fix_klo_questions(json_data, issues, feedback, factsheet, score)

        else:
            logger.warning(f"No fix handler for rule: {rule_id}")
            return None

    def _merge_fixes(self, base_json: dict, patches_json: dict) -> dict:
        """Merge fixes from a parallel task into the base JSON."""
        if not patches_json:
            return base_json

        # Deep merge the topicWizardData
        result = copy.deepcopy(base_json)
        patches_topic = patches_json.get("topicWizardData", {})
        result_topic = result.get("topicWizardData", {})

        # Merge specific sections that might have been modified
        for key in ["submissionQuestions", "resources", "simulationFlow", "workplaceScenario"]:
            if key in patches_topic:
                result_topic[key] = patches_topic[key]

        result["topicWizardData"] = result_topic
        return result

    @traceable(name="fix_klo_questions")
    async def _fix_klo_questions(
        self,
        json_data: dict,
        issues: list,
        feedback: str,
        factsheet: dict,
        score_before: float,
    ) -> tuple[str, dict, AlignmentFixResult]:
        """Fix KLO-Question alignment issues."""

        # Extract context - safely handle non-dict inputs
        json_data = json_data if isinstance(json_data, dict) else {}
        factsheet = factsheet if isinstance(factsheet, dict) else {}
        topic_data = json_data.get("topicWizardData", {}) if isinstance(json_data.get("topicWizardData"), dict) else {}
        company_obj = factsheet.get("company", {}) if isinstance(factsheet.get("company"), dict) else {}
        company_name = company_obj.get("name", "the company") if isinstance(company_obj, dict) else "the company"
        industry = company_obj.get("industry", "business") if isinstance(company_obj, dict) else "business"

        # Extract KLOs
        klos = []
        assessment_criteria = topic_data.get("assessmentCriterion", []) if isinstance(topic_data.get("assessmentCriterion"), list) else []
        for criterion in assessment_criteria:
            if not isinstance(criterion, dict):
                continue
            klo_text = criterion.get("keyLearningOutcome", "")
            if klo_text:
                klos.append({
                    "id": criterion.get("id", ""),
                    "klo": klo_text,
                })

        # Extract questions from simulation flow
        questions = []
        sim_flow = topic_data.get("simulationFlow", []) if isinstance(topic_data.get("simulationFlow"), list) else []
        for stage in sim_flow:
            if not isinstance(stage, dict):
                continue
            stage_data = stage.get("data", {}) if isinstance(stage.get("data"), dict) else {}
            stage_questions = stage_data.get("questions", []) if isinstance(stage_data.get("questions"), list) else []
            for q in stage_questions:
                if not isinstance(q, dict):
                    continue
                questions.append({
                    "id": q.get("id", ""),
                    "question": q.get("question", q.get("text", "")),
                    "location": f"simulationFlow/{stage.get('name', 'unknown')}/questions",
                })
            submission_qs = stage_data.get("submissionQuestions", []) if isinstance(stage_data.get("submissionQuestions"), list) else []
            for q in submission_qs:
                if not isinstance(q, dict):
                    continue
                questions.append({
                    "id": q.get("id", ""),
                    "question": q.get("question", ""),
                    "location": f"simulationFlow/{stage.get('name', 'unknown')}/submissionQuestions",
                })

        # Add top-level submission questions
        for q in topic_data.get("submissionQuestions", []):
            questions.append({
                "id": q.get("id", ""),
                "question": q.get("question", ""),
                "location": "submissionQuestions",
            })

        # Format issues
        issues_text = "\n".join([
            f"- {i}" if isinstance(i, str) else f"- {i.get('message', str(i))}"
            for i in issues[:10]
        ])
        if feedback:
            issues_text += f"\n\nFeedback: {feedback}"

        # Build prompt
        prompt = KLO_QUESTION_FIX_PROMPT.format(
            company_name=company_name,
            industry=industry,
            klos=json.dumps(klos, indent=2),
            questions=json.dumps(questions[:20], indent=2),
            issues=issues_text,
        )

        try:
            # Use semaphore for controlled parallelism
            content = await _invoke_with_semaphore(self._get_llm(), prompt)

            # Parse JSON response
            fixes = self._parse_json_response(content)

            # DEBUG: Log what LLM returned
            print(f"[ALIGNMENT FIXER] KLO-Question LLM response length: {len(content)} chars")
            print(f"[ALIGNMENT FIXER] Parsed fixes: {len(fixes.get('fixes', [])) if fixes else 0} fixes")
            if not fixes:
                print(f"[ALIGNMENT FIXER] RAW LLM response (first 500 chars): {content[:500]}")

            if not fixes:
                return ("klo_to_questions", json_data, AlignmentFixResult(
                    rule_id="klo_to_questions",
                    score_before=score_before,
                    changes_made=["LLM returned no fixes"],
                    success=True,
                ))

            # Apply fixes
            changes = []
            fixed_json = copy.deepcopy(json_data)

            for fix in fixes.get("fixes", []):
                question_id = fix.get("question_id")
                new_text = fix.get("fixed_question", fix.get("new_text"))
                klo_ref = fix.get("klo", "")

                if not question_id or not new_text:
                    continue

                # Find and update the question
                applied = self._apply_question_fix(fixed_json, question_id, new_text)
                if applied:
                    changes.append(f"Updated question {question_id} to align with KLO: {klo_ref[:50]}...")

            return ("klo_to_questions", fixed_json, AlignmentFixResult(
                rule_id="klo_to_questions",
                fixes_applied=fixes.get("fixes", []),
                changes_made=changes,
                score_before=score_before,
                success=len(changes) > 0,
            ))

        except Exception as e:
            logger.error(f"KLO-Question fix failed: {e}")
            return ("klo_to_questions", json_data, AlignmentFixResult(
                rule_id="klo_to_questions",
                score_before=score_before,
                success=False,
                error=str(e),
            ))

    @traceable(name="fix_klo_resources")
    async def _fix_klo_resources(
        self,
        json_data: dict,
        issues: list,
        feedback: str,
        factsheet: dict,
        score_before: float,
    ) -> tuple[str, dict, AlignmentFixResult]:
        """Fix KLO-Resource alignment issues."""

        # Safely handle non-dict inputs
        json_data = json_data if isinstance(json_data, dict) else {}
        factsheet = factsheet if isinstance(factsheet, dict) else {}
        topic_data = json_data.get("topicWizardData", {}) if isinstance(json_data.get("topicWizardData"), dict) else {}
        company_obj = factsheet.get("company", {}) if isinstance(factsheet.get("company"), dict) else {}
        company_name = company_obj.get("name", "the company") if isinstance(company_obj, dict) else "the company"
        industry = company_obj.get("industry", "business") if isinstance(company_obj, dict) else "business"

        # Extract KLOs
        klos = []
        assessment_criteria = topic_data.get("assessmentCriterion", []) if isinstance(topic_data.get("assessmentCriterion"), list) else []
        for criterion in assessment_criteria:
            if not isinstance(criterion, dict):
                continue
            klo_text = criterion.get("keyLearningOutcome", "")
            if klo_text:
                klos.append({"id": criterion.get("id", ""), "klo": klo_text})

        # Extract resources
        resources = topic_data.get("resources", []) if isinstance(topic_data.get("resources"), list) else []
        resources_simplified = []
        for r in resources[:10]:
            if not isinstance(r, dict):
                continue
            resources_simplified.append({
                "id": r.get("id", ""),
                "title": r.get("title", r.get("name", "")),
                "content": r.get("content", r.get("body", ""))[:500],
            })

        # Format issues
        issues_text = "\n".join([
            f"- {i}" if isinstance(i, str) else f"- {i.get('message', str(i))}"
            for i in issues[:10]
        ])

        prompt = KLO_RESOURCE_FIX_PROMPT.format(
            company_name=company_name,
            industry=industry,
            klos=json.dumps(klos, indent=2),
            resources=json.dumps(resources_simplified, indent=2),
            issues=issues_text,
        )

        try:
            # Use semaphore for controlled parallelism
            content = await _invoke_with_semaphore(self._get_llm(), prompt)

            fixes = self._parse_json_response(content)

            if not fixes:
                return ("klo_to_resources", json_data, AlignmentFixResult(
                    rule_id="klo_to_resources",
                    score_before=score_before,
                    changes_made=["LLM returned no fixes"],
                    success=True,
                ))

            # Apply fixes
            changes = []
            fixed_json = copy.deepcopy(json_data)

            for fix in fixes.get("fixes", []):
                resource_id = fix.get("resource_id")
                content_to_add = fix.get("content_to_add", "")
                klo_ref = fix.get("klo", "")

                if not resource_id or not content_to_add:
                    continue

                # Find and update the resource
                applied = self._apply_resource_fix(fixed_json, resource_id, content_to_add)
                if applied:
                    changes.append(f"Added content to resource {resource_id} for KLO: {klo_ref[:50]}...")

            return ("klo_to_resources", fixed_json, AlignmentFixResult(
                rule_id="klo_to_resources",
                fixes_applied=fixes.get("fixes", []),
                changes_made=changes,
                score_before=score_before,
                success=len(changes) > 0,
            ))

        except Exception as e:
            logger.error(f"KLO-Resource fix failed: {e}")
            return ("klo_to_resources", json_data, AlignmentFixResult(
                rule_id="klo_to_resources",
                score_before=score_before,
                success=False,
                error=str(e),
            ))

    @traceable(name="fix_scenario_resources")
    async def _fix_scenario_resources(
        self,
        json_data: dict,
        issues: list,
        feedback: str,
        factsheet: dict,
        score_before: float,
    ) -> tuple[str, dict, AlignmentFixResult]:
        """Fix Scenario-Resource alignment issues."""

        topic_data = json_data.get("topicWizardData", {})
        company_name = factsheet.get("company", {}).get("name", "the company")
        industry = factsheet.get("company", {}).get("industry", "business")

        # Extract scenario info
        scenario = topic_data.get("selectedScenario", {})
        scenario_text = scenario.get("scenario", scenario.get("description", ""))
        workplace = topic_data.get("workplaceScenario", {})

        # Extract resources
        resources = topic_data.get("resources", [])

        issues_text = "\n".join([
            f"- {i}" if isinstance(i, str) else f"- {i.get('message', str(i))}"
            for i in issues[:10]
        ])

        prompt = SCENARIO_RESOURCE_FIX_PROMPT.format(
            company_name=company_name,
            industry=industry,
            scenario=scenario_text[:500],
            workplace_context=json.dumps(workplace, indent=2)[:500],
            resources=json.dumps([{
                "id": r.get("id", ""),
                "title": r.get("title", ""),
                "content": r.get("content", "")[:300]
            } for r in resources[:10]], indent=2),
            issues=issues_text,
        )

        try:
            # Use semaphore for controlled parallelism
            content = await _invoke_with_semaphore(self._get_llm(), prompt)

            fixes = self._parse_json_response(content)
            changes = []
            fixed_json = copy.deepcopy(json_data)

            if fixes:
                for fix in fixes.get("fixes", []):
                    resource_id = fix.get("resource_id")
                    content_to_add = fix.get("content_to_add", "")

                    if resource_id and content_to_add:
                        applied = self._apply_resource_fix(fixed_json, resource_id, content_to_add)
                        if applied:
                            changes.append(f"Added scenario-specific content to resource {resource_id}")

            return ("scenario_to_resources", fixed_json, AlignmentFixResult(
                rule_id="scenario_to_resources",
                fixes_applied=fixes.get("fixes", []) if fixes else [],
                changes_made=changes or ["No fixes applied"],
                score_before=score_before,
                success=True,
            ))

        except Exception as e:
            logger.error(f"Scenario-Resource fix failed: {e}")
            return ("scenario_to_resources", json_data, AlignmentFixResult(
                rule_id="scenario_to_resources",
                score_before=score_before,
                success=False,
                error=str(e),
            ))

    @traceable(name="fix_role_tasks")
    async def _fix_role_tasks(
        self,
        json_data: dict,
        issues: list,
        feedback: str,
        factsheet: dict,
        score_before: float,
    ) -> tuple[str, dict, AlignmentFixResult]:
        """Fix Role-Task alignment issues."""

        topic_data = json_data.get("topicWizardData", {})
        company_name = factsheet.get("company", {}).get("name", "the company")
        industry = factsheet.get("company", {}).get("industry", "business")

        # Extract role info
        workplace = topic_data.get("workplaceScenario", {})
        learner_role = workplace.get("role", workplace.get("learnerRole", ""))

        # Extract tasks from simulation flow
        tasks = []
        for stage in topic_data.get("simulationFlow", []):
            stage_data = stage.get("data", {})
            task_name = stage_data.get("name", stage.get("name", ""))
            task_desc = stage_data.get("description", "")
            if task_name:
                tasks.append({"name": task_name, "description": task_desc})

        issues_text = "\n".join([
            f"- {i}" if isinstance(i, str) else f"- {i.get('message', str(i))}"
            for i in issues[:10]
        ])

        prompt = ROLE_TASK_FIX_PROMPT.format(
            company_name=company_name,
            industry=industry,
            role=learner_role,
            tasks=json.dumps(tasks[:10], indent=2),
            issues=issues_text,
        )

        try:
            # Use semaphore for controlled parallelism
            content = await _invoke_with_semaphore(self._get_llm(), prompt)

            fixes = self._parse_json_response(content)
            changes = []
            fixed_json = copy.deepcopy(json_data)

            if fixes:
                # Apply task description fixes
                for fix in fixes.get("fixes", []):
                    task_name = fix.get("task_name")
                    new_description = fix.get("new_description")

                    if task_name and new_description:
                        applied = self._apply_task_fix(fixed_json, task_name, new_description)
                        if applied:
                            changes.append(f"Updated task '{task_name}' to align with role")

            return ("role_to_tasks", fixed_json, AlignmentFixResult(
                rule_id="role_to_tasks",
                fixes_applied=fixes.get("fixes", []) if fixes else [],
                changes_made=changes or ["No fixes applied"],
                score_before=score_before,
                success=True,
            ))

        except Exception as e:
            logger.error(f"Role-Task fix failed: {e}")
            return ("role_to_tasks", json_data, AlignmentFixResult(
                rule_id="role_to_tasks",
                score_before=score_before,
                success=False,
                error=str(e),
            ))

    @traceable(name="fix_coherence")
    async def _fix_coherence(
        self,
        json_data: dict,
        issues: list,
        feedback: str,
        factsheet: dict,
        score_before: float,
    ) -> tuple[str, dict, AlignmentFixResult]:
        """Fix scenario coherence issues."""

        topic_data = json_data.get("topicWizardData", {})
        company_name = factsheet.get("company", {}).get("name", "the company")
        industry = factsheet.get("company", {}).get("industry", "business")

        # Extract key scenario elements
        workplace = topic_data.get("workplaceScenario", {})
        scenario = topic_data.get("selectedScenario", {})

        issues_text = "\n".join([
            f"- {i}" if isinstance(i, str) else f"- {i.get('message', str(i))}"
            for i in issues[:10]
        ])

        prompt = SCENARIO_COHERENCE_FIX_PROMPT.format(
            company_name=company_name,
            industry=industry,
            workplace_scenario=json.dumps(workplace, indent=2)[:1000],
            selected_scenario=json.dumps(scenario, indent=2)[:500],
            issues=issues_text,
        )

        try:
            # Use semaphore for controlled parallelism
            content = await _invoke_with_semaphore(self._get_llm(), prompt)

            fixes = self._parse_json_response(content)
            changes = []
            fixed_json = copy.deepcopy(json_data)

            if fixes:
                # Apply coherence fixes
                for fix in fixes.get("fixes", []):
                    path = fix.get("path")
                    new_value = fix.get("new_value")

                    if path and new_value:
                        applied = self._apply_path_fix(fixed_json, path, new_value)
                        if applied:
                            changes.append(f"Fixed coherence issue at {path}")

            return ("scenario_coherence", fixed_json, AlignmentFixResult(
                rule_id="scenario_coherence",
                fixes_applied=fixes.get("fixes", []) if fixes else [],
                changes_made=changes or ["No fixes applied"],
                score_before=score_before,
                success=True,
            ))

        except Exception as e:
            logger.error(f"Coherence fix failed: {e}")
            return ("scenario_coherence", json_data, AlignmentFixResult(
                rule_id="scenario_coherence",
                score_before=score_before,
                success=False,
                error=str(e),
            ))

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _parse_json_response(self, content: str) -> dict:
        """Parse JSON from LLM response."""
        try:
            # Find JSON block
            start = content.find('{')
            if start == -1:
                return {}

            # Find matching closing brace
            depth = 0
            for i, char in enumerate(content[start:], start):
                if char == '{':
                    depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 0:
                        json_str = content[start:i+1]
                        return json.loads(json_str)

            return {}
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON: {e}")
            return {}

    def _apply_question_fix(self, json_data: dict, question_id: str, new_text: str) -> bool:
        """Apply a question text fix."""
        topic_data = json_data.get("topicWizardData", {})

        # Check submissionQuestions
        for q in topic_data.get("submissionQuestions", []):
            if q.get("id") == question_id:
                q["question"] = new_text
                return True

        # Check simulationFlow
        for stage in topic_data.get("simulationFlow", []):
            stage_data = stage.get("data", {})

            for q in stage_data.get("questions", []):
                if q.get("id") == question_id:
                    if "question" in q:
                        q["question"] = new_text
                    else:
                        q["text"] = new_text
                    return True

            for q in stage_data.get("submissionQuestions", []):
                if q.get("id") == question_id:
                    q["question"] = new_text
                    return True

        return False

    def _apply_resource_fix(self, json_data: dict, resource_id: str, content_to_add: str) -> bool:
        """Apply a resource content fix (append content)."""
        topic_data = json_data.get("topicWizardData", {})

        for r in topic_data.get("resources", []):
            if r.get("id") == resource_id:
                existing = r.get("content", r.get("body", ""))
                r["content"] = existing + "\n\n" + content_to_add
                return True

        return False

    def _apply_task_fix(self, json_data: dict, task_name: str, new_description: str) -> bool:
        """Apply a task description fix."""
        topic_data = json_data.get("topicWizardData", {})

        for stage in topic_data.get("simulationFlow", []):
            stage_data = stage.get("data", {})
            if stage_data.get("name") == task_name or stage.get("name") == task_name:
                stage_data["description"] = new_description
                return True

        return False

    def _apply_path_fix(self, json_data: dict, path: str, new_value: Any) -> bool:
        """Apply a fix at a JSON path."""
        try:
            # Skip invalid paths that reference non-existent root keys
            invalid_root_keys = ["globalFactsheet", "communications", "klo"]
            path_lower = path.lower()
            for invalid_key in invalid_root_keys:
                if f"/{invalid_key}/" in path_lower or path_lower.endswith(f"/{invalid_key}"):
                    logger.debug(f"Skipping path with non-existent key '{invalid_key}': {path}")
                    return False

            # Handle dot notation (convert to slash notation)
            if '.' in path and '/' not in path:
                path = '/' + path.replace('.', '/')

            # Handle array wildcards [*] - skip these as they require special handling
            if '[*]' in path:
                logger.debug(f"Skipping wildcard path: {path}")
                return False

            # Handle array index notation [0], [1], etc.
            path = re.sub(r'\[(\d+)\]', r'/\1', path)

            # Simple path parsing (supports /key1/key2/key3)
            parts = [p for p in path.strip('/').split('/') if p]

            if not parts:
                return False

            # Auto-prepend topicWizardData if the path doesn't start with it
            # and the first part isn't a valid root-level key
            valid_root_keys = ["topicWizardData", "metadata", "version", "id"]
            topic_data = json_data.get("topicWizardData", {})

            if parts[0] not in valid_root_keys and parts[0] not in json_data:
                # Check if it exists under topicWizardData
                valid_topic_keys = [
                    "workplaceScenario", "simulationFlow", "assessmentCriterion",
                    "resources", "submissionQuestions", "emails", "rubric",
                    "industryAlignedActivities", "selectedIndustryAlignedActivities",
                    "scenarioOptions", "scenarioDescription", "lessonInformation",
                    "overview", "simulationName", "launchSettings", "videos",
                    "selectedScenarioOption", "selectedSubmissionQuestions"
                ]
                if parts[0] in topic_data or parts[0] in valid_topic_keys:
                    parts = ["topicWizardData"] + parts
                    logger.debug(f"Auto-prepended topicWizardData to path: /{'/'.join(parts)}")
                else:
                    logger.debug(f"Unknown root key '{parts[0]}', skipping path: {path}")
                    return False

            current = json_data

            for part in parts[:-1]:
                if part.isdigit():
                    idx = int(part)
                    if isinstance(current, list) and idx < len(current):
                        current = current[idx]
                    else:
                        logger.debug(f"Index {idx} out of range for list of length {len(current) if isinstance(current, list) else 'N/A'}")
                        return False
                else:
                    if isinstance(current, dict) and part in current:
                        current = current[part]
                    else:
                        logger.debug(f"Path part '{part}' not found. Available: {list(current.keys())[:10] if isinstance(current, dict) else 'N/A'}")
                        return False

            # Set the value
            final_key = parts[-1]
            if final_key.isdigit():
                idx = int(final_key)
                if isinstance(current, list) and idx < len(current):
                    current[idx] = new_value
                    return True
                else:
                    return False
            else:
                if isinstance(current, dict):
                    current[final_key] = new_value
                    return True
                return False

        except (KeyError, IndexError, TypeError) as e:
            logger.debug(f"Failed to apply path fix at {path}: {e}")
            return False

    # =========================================================================
    # CONSISTENCY FIX METHODS
    # =========================================================================

    @traceable(name="fix_manager_consistency")
    async def _fix_manager_consistency(
        self,
        json_data: dict,
        issues: list,
        feedback: str,
        factsheet: dict,
        score_before: float,
    ) -> tuple[str, dict, AlignmentFixResult]:
        """Fix reporting manager consistency issues by replacing incorrect manager references."""

        # Get expected manager from factsheet - safely handle non-dict inputs
        factsheet = factsheet if isinstance(factsheet, dict) else {}
        expected_manager = factsheet.get("reporting_manager", {}) if isinstance(factsheet.get("reporting_manager"), dict) else {}
        expected_name = expected_manager.get("name", "") if isinstance(expected_manager, dict) else ""
        expected_email = expected_manager.get("email", "") if isinstance(expected_manager, dict) else ""
        expected_role = expected_manager.get("role", "") if isinstance(expected_manager, dict) else ""
        expected_gender = expected_manager.get("gender", "") if isinstance(expected_manager, dict) else ""

        if not expected_name:
            logger.warning("No expected manager name in factsheet, skipping manager consistency fix")
            return ("reporting_manager_consistency", json_data, AlignmentFixResult(
                rule_id="reporting_manager_consistency",
                changes_made=["No expected manager in factsheet"],
                score_before=score_before,
                success=False,
            ))

        # Get poison list to find old names to replace
        poison_list = list(factsheet.get("poison_list", []))  # Copy to avoid mutating original

        # Debug: Log what we're working with
        print(f"[MANAGER FIX] Expected manager: {expected_name}")
        print(f"[MANAGER FIX] Initial poison_list: {poison_list[:5]}...")
        print(f"[MANAGER FIX] Issues to process: {len(issues)}")

        # FALLBACK: Extract names from issues if poison_list doesn't have them
        # Parse issues for quoted names that might be old scenario content
        for issue in issues:
            issue_text = str(issue.get("description", "")) if isinstance(issue, dict) else str(issue)
            print(f"[MANAGER FIX] Processing issue: {issue_text[:100]}...")

            # Extract quoted names - both single and double quotes
            # e.g., "Elizabeth", 'Elizabeth', "Velocity Dome"
            double_quotes = re.findall(r'"([A-Z][^"]+)"', issue_text)
            single_quotes = re.findall(r"'([A-Z][^']+)'", issue_text)
            quoted_matches = double_quotes + single_quotes

            for match in quoted_matches:
                # Skip if it matches expected values
                if expected_name and match.lower() == expected_name.lower():
                    continue
                if expected_email and match.lower() == expected_email.lower():
                    continue
                if expected_role and match.lower() == expected_role.lower():
                    continue
                # Add to poison list if it's not already there and looks like a name/company
                if match not in poison_list and len(match.split()) <= 4:
                    poison_list.append(match)
                    print(f"[MANAGER FIX] Extracted '{match}' from issue")
                    logger.info(f"[FALLBACK] Added '{match}' to poison list from issue")

        print(f"[MANAGER FIX] Final poison_list: {poison_list[:10]}")

        fixed_json = copy.deepcopy(json_data)
        changes = []

        def replace_manager_refs(obj: Any, path: str = "") -> Any:
            """Recursively replace manager references."""
            if isinstance(obj, str):
                modified = obj
                # Replace old manager names from poison list (case-insensitive)
                for poison in poison_list:
                    if poison.lower() in modified.lower():
                        # Check if this looks like a name (capitalized words)
                        if poison[0].isupper():
                            # Case-insensitive replacement
                            pattern = re.compile(re.escape(poison), re.IGNORECASE)
                            new_val = pattern.sub(expected_name, modified)
                            if new_val != modified:
                                changes.append(f"Replaced '{poison}' with '{expected_name}' at {path}")
                                modified = new_val
                return modified
            elif isinstance(obj, dict):
                result = {}
                for key, value in obj.items():
                    new_path = f"{path}.{key}" if path else key

                    # Special handling for manager/sender objects
                    if key in ("reportingManager", "sender", "manager") and isinstance(value, dict):
                        # Replace manager fields with expected values
                        if value.get("name") and value["name"] != expected_name:
                            old_name = value["name"]
                            value["name"] = expected_name
                            changes.append(f"Fixed manager name: '{old_name}' → '{expected_name}' at {new_path}")
                        if expected_email and value.get("email") and expected_email not in value["email"]:
                            old_email = value["email"]
                            value["email"] = expected_email
                            changes.append(f"Fixed manager email at {new_path}")
                        if expected_role and value.get("role"):
                            value["role"] = expected_role
                        if expected_gender and value.get("gender"):
                            value["gender"] = expected_gender.capitalize()
                        result[key] = value
                    else:
                        result[key] = replace_manager_refs(value, new_path)
                return result
            elif isinstance(obj, list):
                return [replace_manager_refs(item, f"{path}[{i}]") for i, item in enumerate(obj)]
            return obj

        fixed_json = replace_manager_refs(fixed_json)

        return ("reporting_manager_consistency", fixed_json, AlignmentFixResult(
            rule_id="reporting_manager_consistency",
            fixes_applied=[{"type": "manager_replacement"}] if changes else [],
            changes_made=changes or ["No manager inconsistencies found to fix"],
            score_before=score_before,
            success=len(changes) > 0,
        ))

    @traceable(name="fix_company_consistency")
    async def _fix_company_consistency(
        self,
        json_data: dict,
        issues: list,
        feedback: str,
        factsheet: dict,
        score_before: float,
    ) -> tuple[str, dict, AlignmentFixResult]:
        """Fix company name consistency by replacing old company names with the correct one."""

        # Get expected company from factsheet - safely handle non-dict inputs
        factsheet = factsheet if isinstance(factsheet, dict) else {}
        company_obj = factsheet.get("company", {}) if isinstance(factsheet.get("company"), dict) else {}
        expected_company = company_obj.get("name", "") if isinstance(company_obj, dict) else ""

        if not expected_company:
            logger.warning("No expected company name in factsheet, skipping company consistency fix")
            return ("company_consistency", json_data, AlignmentFixResult(
                rule_id="company_consistency",
                changes_made=["No expected company in factsheet"],
                score_before=score_before,
                success=False,
            ))

        # Get poison list to find old company names
        poison_list_raw = factsheet.get("poison_list", []) if isinstance(factsheet.get("poison_list"), list) else []
        poison_list = list(poison_list_raw)  # Copy to avoid mutating original

        # Debug: Log what we're working with
        print(f"[COMPANY FIX] Expected company: {expected_company}")
        print(f"[COMPANY FIX] Initial poison_list: {poison_list[:5]}...")
        print(f"[COMPANY FIX] Issues to process: {len(issues)}")

        # FALLBACK: Extract company names from issues if poison_list doesn't have them
        for issue in issues:
            issue_text = str(issue.get("description", "")) if isinstance(issue, dict) else str(issue)
            print(f"[COMPANY FIX] Processing issue: {issue_text[:100]}...")

            # Extract quoted names - both single and double quotes
            double_quotes = re.findall(r'"([A-Z][^"]+)"', issue_text)
            single_quotes = re.findall(r"'([A-Z][^']+)'", issue_text)
            quoted_matches = double_quotes + single_quotes

            for match in quoted_matches:
                # Skip if it matches expected company
                if match.lower() == expected_company.lower():
                    continue
                # Add to poison list if it looks like a company name
                if match not in poison_list and len(match.split()) <= 4:
                    poison_list.append(match)
                    print(f"[COMPANY FIX] Extracted '{match}' from issue")
                    logger.info(f"[FALLBACK] Added '{match}' to poison list from issue (company fix)")

        print(f"[COMPANY FIX] Final poison_list: {poison_list[:10]}")

        fixed_json = copy.deepcopy(json_data)
        changes = []

        def replace_company_refs(obj: Any, path: str = "") -> Any:
            """Recursively replace company name references."""
            if isinstance(obj, str):
                modified = obj
                # Replace old company names from poison list
                for poison in poison_list:
                    if poison.lower() in modified.lower():
                        # Case-insensitive replacement
                        pattern = re.compile(re.escape(poison), re.IGNORECASE)
                        new_val = pattern.sub(expected_company, modified)
                        if new_val != modified:
                            changes.append(f"Replaced '{poison}' with '{expected_company}' at {path}")
                            modified = new_val
                return modified
            elif isinstance(obj, dict):
                result = {}
                for key, value in obj.items():
                    new_path = f"{path}.{key}" if path else key

                    # Special handling for organization/company fields
                    if key in ("organizationName", "companyName", "company") and isinstance(value, str):
                        if value != expected_company:
                            for poison in poison_list:
                                if poison.lower() in value.lower():
                                    changes.append(f"Fixed company name: '{value}' → '{expected_company}' at {new_path}")
                                    value = expected_company
                                    break
                        result[key] = value
                    else:
                        result[key] = replace_company_refs(value, new_path)
                return result
            elif isinstance(obj, list):
                return [replace_company_refs(item, f"{path}[{i}]") for i, item in enumerate(obj)]
            return obj

        fixed_json = replace_company_refs(fixed_json)

        return ("company_consistency", fixed_json, AlignmentFixResult(
            rule_id="company_consistency",
            fixes_applied=[{"type": "company_replacement"}] if changes else [],
            changes_made=changes or ["No company inconsistencies found to fix"],
            score_before=score_before,
            success=len(changes) > 0,
        ))


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

async def fix_alignment_issues(
    adapted_json: dict,
    alignment_report: dict,
    global_factsheet: dict,
) -> tuple[dict, list[AlignmentFixResult]]:
    """
    Convenience function to fix alignment issues.

    Call this AFTER alignment check if score < threshold.

    Args:
        adapted_json: The adapted JSON from adaptation stage
        alignment_report: Report from alignment checker
        global_factsheet: Context with company, industry, KLOs

    Returns:
        (fixed_json, list of AlignmentFixResult)
    """
    fixer = AlignmentFixer()
    return await fixer.fix(adapted_json, alignment_report, global_factsheet)
