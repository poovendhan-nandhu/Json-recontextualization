"""
Simple Validation Agents - Based on PROMPT_SIMPLIFICATION.md

6 Validation Agents that run in PARALLEL after adaptation:
1. Domain Fidelity Agent - checks industry terminology matches target
2. Context Fidelity Agent - verifies goal/challenge preserved
3. Resource Quality Agent - ensures resources provide data, not answers
4. KLO-Question Alignment Agent - confirms questions assess KLOs
5. Consistency Agent - validates same names throughout
6. Completeness Agent - detects placeholders/truncation

Plus: Repair Agent to fix issues found.

NO HARDCODING - everything derived from scenario prompt.
"""
import os
import json
import logging
import asyncio
import re
from dataclasses import dataclass, field
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

# LangSmith tracing
try:
    from langsmith import traceable
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    def traceable(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=10)

PASS_THRESHOLD = 0.98
ACCEPTABLE_THRESHOLD = 0.95

# Locked shard paths - these should NOT be validated for domain terms
LOCKED_PATHS = ["scenarioOptions", "workspaceIds"]


def strip_locked_content(adapted_json: dict) -> dict:
    """Remove locked shard content before validation.

    Locked shards (scenarioOptions, workspaceIds) contain original content
    that should NOT be flagged by validators.
    """
    import copy
    result = copy.deepcopy(adapted_json)

    topic_data = result.get("topicWizardData", {})
    if isinstance(topic_data, dict):
        for path in LOCKED_PATHS:
            if path in topic_data:
                # Replace with marker so structure is preserved
                topic_data[path] = "[LOCKED - NOT VALIDATED]"

    return result


@dataclass
class ValidationIssue:
    agent: str
    location: str
    issue: str
    suggestion: str
    severity: str = "warning"


@dataclass
class AgentResult:
    agent_name: str
    score: float
    passed: bool
    issues: list[ValidationIssue] = field(default_factory=list)
    details: dict = field(default_factory=dict)


@dataclass
class ValidationReport:
    overall_score: float
    passed: bool
    agent_results: list[AgentResult]
    total_issues: int
    needs_repair: bool

    def to_dict(self) -> dict:
        return {
            "overall_score": self.overall_score,
            "passed": self.passed,
            "needs_repair": self.needs_repair,
            "total_issues": self.total_issues,
            "agents": [
                {
                    "name": r.agent_name,
                    "score": r.score,
                    "passed": r.passed,
                    "issues_count": len(r.issues),
                }
                for r in self.agent_results
            ]
        }


def _get_openai_client():
    from openai import OpenAI
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


async def _call_gpt_async(prompt: str, system: str = "You are a validation agent.") -> str:
    loop = asyncio.get_event_loop()

    def _call():
        client = _get_openai_client()
        response = client.chat.completions.create(
            model="gpt-5.2-2025-12-11",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
        )
        return response.choices[0].message.content

    return await loop.run_in_executor(_executor, _call)


def _parse_json_response(response: str) -> dict:
    """Parse JSON from LLM response."""
    response = response.strip()
    if response.startswith("```"):
        response = response.split("```")[1]
        if response.startswith("json"):
            response = response[4:]
    return json.loads(response)


# =============================================================================
# AGENT 1: DOMAIN FIDELITY
# =============================================================================

@traceable(name="validate_domain_fidelity", run_type="chain")
async def validate_domain_fidelity(adapted_json: dict, scenario_prompt: str) -> AgentResult:
    """Check if ALL content uses TARGET industry terminology (derived from scenario)."""
    logger.info("[VALIDATOR] Running Domain Fidelity Agent...")

    content_text = json.dumps(adapted_json, indent=2)[:50000]

    prompt = f"""You are validating DOMAIN FIDELITY for a business simulation.

## TARGET SCENARIO (derive the target industry from this):
{scenario_prompt}

## YOUR TASK:
1. First, identify what INDUSTRY the target scenario is about
2. Then, scan the content for ANY terms from a DIFFERENT industry
3. Flag any terms that don't belong to the target industry

The content should be 100% about the TARGET industry.
Any terms from other industries (whatever they may be) are violations.

## CONTENT TO CHECK:
{content_text}

## OUTPUT (JSON only):
{{
    "target_industry": "what industry you derived from scenario",
    "score": 0.0 to 1.0,
    "invalid_terms": [
        {{"term": "found term", "belongs_to": "what industry it belongs to", "should_be": "target equivalent"}}
    ]
}}"""

    try:
        response = await _call_gpt_async(prompt)
        result = _parse_json_response(response)

        issues = []
        for item in result.get("invalid_terms", []):
            issues.append(ValidationIssue(
                agent="Domain Fidelity",
                location="content",
                issue=f"'{item.get('term')}' belongs to {item.get('belongs_to')}",
                suggestion=f"Replace with: {item.get('should_be')}",
                severity="error"
            ))

        score = result.get("score", 0.0)
        return AgentResult(
            agent_name="Domain Fidelity",
            score=score,
            passed=score >= PASS_THRESHOLD,
            issues=issues,
            details=result
        )
    except Exception as e:
        logger.error(f"[VALIDATOR] Domain Fidelity failed: {e}")
        return AgentResult(agent_name="Domain Fidelity", score=0.0, passed=False, issues=[])


# =============================================================================
# AGENT 2: CONTEXT FIDELITY
# =============================================================================

@traceable(name="validate_context_fidelity", run_type="chain")
async def validate_context_fidelity(adapted_json: dict, scenario_prompt: str) -> AgentResult:
    """Verify the goal/challenge/learning purpose from scenario is preserved."""
    logger.info("[VALIDATOR] Running Context Fidelity Agent...")

    content_text = json.dumps(adapted_json, indent=2)[:30000]

    prompt = f"""You are validating CONTEXT FIDELITY for a business simulation.

## TARGET SCENARIO (extract the goal/challenge from this):
{scenario_prompt}

## YOUR TASK:
1. Extract from scenario: What is the learner's GOAL? What CHALLENGE must they solve? What ROLE do they play?
2. Check if the adapted content PRESERVES these elements
3. The specific context may change, but the TYPE of challenge should remain

## CONTENT TO CHECK:
{content_text}

## OUTPUT (JSON only):
{{
    "extracted_goal": "what you derived from scenario",
    "extracted_challenge": "the challenge type",
    "extracted_role": "learner's role",
    "goal_preserved": true/false,
    "challenge_preserved": true/false,
    "role_preserved": true/false,
    "score": 0.0 to 1.0,
    "issues": [{{"aspect": "what changed", "expected": "from scenario", "found": "in content"}}]
}}"""

    try:
        response = await _call_gpt_async(prompt)
        result = _parse_json_response(response)

        issues = []
        for item in result.get("issues", []):
            issues.append(ValidationIssue(
                agent="Context Fidelity",
                location=item.get("aspect", "unknown"),
                issue=f"Expected: {item.get('expected')}",
                suggestion=f"Found: {item.get('found')}",
                severity="warning"
            ))

        score = result.get("score", 0.0)
        return AgentResult(
            agent_name="Context Fidelity",
            score=score,
            passed=score >= PASS_THRESHOLD,
            issues=issues,
            details=result
        )
    except Exception as e:
        logger.error(f"[VALIDATOR] Context Fidelity failed: {e}")
        return AgentResult(agent_name="Context Fidelity", score=0.0, passed=False, issues=[])


# =============================================================================
# AGENT 3: RESOURCE QUALITY (INFERENCE MAP)
# =============================================================================

@traceable(name="validate_resource_quality", run_type="chain")
async def validate_resource_quality(adapted_json: dict, scenario_prompt: str) -> AgentResult:
    """Ensure resources provide DATA not answers (inference map)."""
    logger.info("[VALIDATOR] Running Resource Quality Agent...")

    topic_data = adapted_json.get("topicWizardData", {})
    resources = topic_data.get("resources", [])

    if not resources:
        return AgentResult(agent_name="Resource Quality", score=1.0, passed=True, issues=[])

    resources_text = json.dumps(resources, indent=2)[:40000]

    prompt = f"""You are validating RESOURCE QUALITY for a business simulation.

## CRITICAL RULE: Resources must provide DATA, not ANSWERS

Resources should give "dots to connect" - raw data, statistics, facts.
Resources should NOT give "connected dots" - conclusions, recommendations, answers.

BAD examples (giving answers):
- "The market is attractive because..."
- "Based on this data, you should..."
- "The recommendation is to..."

GOOD examples (giving data):
- "Market size: $X billion (Source: Report 2024)"
- "Competitor A has X% market share"
- "Customer segment prefers Y"

## ALSO CHECK:
- Word count: 500-1500 words per resource
- Has citations/sources
- Self-contained (has data needed)

## RESOURCES TO CHECK:
{resources_text}

## OUTPUT (JSON only):
{{
    "score": 0.0 to 1.0,
    "resources_checked": number,
    "direct_answers_found": ["any conclusions/recommendations found in resources"],
    "issues": [{{"resource": "name", "issue": "what's wrong", "example": "problematic text"}}]
}}"""

    try:
        response = await _call_gpt_async(prompt)
        result = _parse_json_response(response)

        issues = []
        for answer in result.get("direct_answers_found", []):
            issues.append(ValidationIssue(
                agent="Resource Quality",
                location="resources",
                issue=f"Direct answer: {answer[:100]}",
                suggestion="Rephrase to provide data without conclusion",
                severity="error"
            ))

        score = result.get("score", 0.0)
        return AgentResult(
            agent_name="Resource Quality",
            score=score,
            passed=score >= ACCEPTABLE_THRESHOLD,
            issues=issues,
            details=result
        )
    except Exception as e:
        logger.error(f"[VALIDATOR] Resource Quality failed: {e}")
        return AgentResult(agent_name="Resource Quality", score=0.0, passed=False, issues=[])


# =============================================================================
# AGENT 4: KLO-QUESTION ALIGNMENT
# =============================================================================

@traceable(name="validate_klo_question_alignment", run_type="chain")
async def validate_klo_question_alignment(adapted_json: dict, scenario_prompt: str) -> AgentResult:
    """Confirm each question assesses a KLO implied by the scenario."""
    logger.info("[VALIDATOR] Running KLO-Question Alignment Agent...")

    topic_data = adapted_json.get("topicWizardData", {})
    klos = topic_data.get("assessmentCriterion", [])
    klos_text = json.dumps(klos, indent=2)[:10000] if klos else "No KLOs found"

    # Find questions in all possible locations
    questions = []

    def extract_questions(obj):
        """Recursively find all question arrays."""
        if isinstance(obj, dict):
            for key, val in obj.items():
                if 'question' in key.lower() and isinstance(val, list):
                    questions.extend(val)
                else:
                    extract_questions(val)
        elif isinstance(obj, list):
            for item in obj:
                extract_questions(item)

    extract_questions(topic_data)
    logger.info(f"[VALIDATOR] Found {len(questions)} questions, {len(klos)} KLOs")

    questions_text = json.dumps(questions, indent=2)[:20000] if questions else "No questions found"

    prompt = f"""You are validating KLO-QUESTION ALIGNMENT.

## SCENARIO:
{scenario_prompt}

## KEY LEARNING OUTCOMES (KLOs):
{klos_text}

## SUBMISSION QUESTIONS:
{questions_text}

## YOUR TASK:
1. For each question, check if it assesses one of the KLOs
2. Check if question terminology matches the TARGET scenario
3. Check if question can be answered from resources

## OUTPUT (JSON only):
{{
    "score": 0.0 to 1.0,
    "questions_checked": number,
    "unaligned_questions": ["questions not mapping to any KLO"],
    "wrong_terminology_questions": ["questions using wrong industry terms"]
}}"""

    try:
        response = await _call_gpt_async(prompt)
        result = _parse_json_response(response)

        issues = []
        for q in result.get("unaligned_questions", []):
            issues.append(ValidationIssue(
                agent="KLO-Question",
                location="questions",
                issue=f"Not aligned: {q[:80]}",
                suggestion="Map to KLO",
                severity="warning"
            ))
        for q in result.get("wrong_terminology_questions", []):
            issues.append(ValidationIssue(
                agent="KLO-Question",
                location="questions",
                issue=f"Wrong terms: {q[:80]}",
                suggestion="Use target terminology",
                severity="error"
            ))

        score = result.get("score", 0.0)
        return AgentResult(
            agent_name="KLO-Question Alignment",
            score=score,
            passed=score >= PASS_THRESHOLD,
            issues=issues,
            details=result
        )
    except Exception as e:
        logger.error(f"[VALIDATOR] KLO-Question failed: {e}")
        return AgentResult(agent_name="KLO-Question Alignment", score=0.0, passed=False, issues=[])


# =============================================================================
# AGENT 5: CONSISTENCY
# =============================================================================

@traceable(name="validate_consistency", run_type="chain")
async def validate_consistency(adapted_json: dict, scenario_prompt: str) -> AgentResult:
    """Validate same company/manager names used throughout."""
    logger.info("[VALIDATOR] Running Consistency Agent...")

    content_text = json.dumps(adapted_json, indent=2)[:40000]

    prompt = f"""You are validating NAMING CONSISTENCY.

## SCENARIO:
{scenario_prompt}

## YOUR TASK:
Extract all company names, person names, and emails from content.
Check for consistency:
- ONE company name should be used everywhere
- ONE manager name should be used everywhere
- Email format: firstname.lastname@company.com

Inconsistent = "EcoChic" vs "EcoChic Threads" vs "Eco-Chic"
Consistent = "EcoChic Threads" used everywhere

## CONTENT:
{content_text}

## OUTPUT (JSON only):
{{
    "score": 0.0 to 1.0,
    "company_variations": ["all variations found"],
    "is_company_consistent": true/false,
    "manager_variations": ["all name variations"],
    "is_manager_consistent": true/false,
    "issues": [{{"type": "company/manager/email", "found": "variations", "recommendation": "use this"}}]
}}"""

    try:
        response = await _call_gpt_async(prompt)
        result = _parse_json_response(response)

        issues = []
        for item in result.get("issues", []):
            issues.append(ValidationIssue(
                agent="Consistency",
                location=item.get("type", "unknown"),
                issue=f"Inconsistent: {item.get('found')}",
                suggestion=item.get("recommendation", ""),
                severity="warning"
            ))

        score = result.get("score", 0.0)
        return AgentResult(
            agent_name="Consistency",
            score=score,
            passed=score >= PASS_THRESHOLD,
            issues=issues,
            details=result
        )
    except Exception as e:
        logger.error(f"[VALIDATOR] Consistency failed: {e}")
        return AgentResult(agent_name="Consistency", score=0.0, passed=False, issues=[])


# =============================================================================
# AGENT 6: COMPLETENESS
# =============================================================================

@traceable(name="validate_completeness", run_type="chain")
async def validate_completeness(adapted_json: dict, scenario_prompt: str) -> AgentResult:
    """Detect placeholders, truncated content, or missing data."""
    logger.info("[VALIDATOR] Running Completeness Agent...")

    content_text = json.dumps(adapted_json, indent=2)[:40000]

    # Quick regex pre-check
    placeholder_patterns = [r'\[TBD\]', r'\[TODO\]', r'\[INSERT\]', r'XXX', r'\[.*?\]']
    found = []
    for p in placeholder_patterns:
        found.extend(re.findall(p, content_text, re.IGNORECASE))

    prompt = f"""You are validating COMPLETENESS.

## SCAN FOR:
1. Placeholders: [TBD], [INSERT], [Your Name], TODO, XXX, [anything in brackets]
2. Truncated content: sentences ending mid-word or abruptly
3. Empty fields that should have content
4. Incomplete lists

## PRE-DETECTED PLACEHOLDERS: {found[:20]}

## CONTENT:
{content_text}

## OUTPUT (JSON only):
{{
    "score": 0.0 to 1.0,
    "placeholders": ["list"],
    "truncated": ["truncated sentences"],
    "empty_fields": ["field paths"]
}}"""

    try:
        response = await _call_gpt_async(prompt)
        result = _parse_json_response(response)

        issues = []
        for p in result.get("placeholders", []):
            issues.append(ValidationIssue(
                agent="Completeness",
                location="content",
                issue=f"Placeholder: {p}",
                suggestion="Replace with actual content",
                severity="error"
            ))

        score = result.get("score", 0.0)
        return AgentResult(
            agent_name="Completeness",
            score=score,
            passed=score >= PASS_THRESHOLD,
            issues=issues,
            details=result
        )
    except Exception as e:
        logger.error(f"[VALIDATOR] Completeness failed: {e}")
        return AgentResult(agent_name="Completeness", score=0.0, passed=False, issues=[])


# =============================================================================
# ORCHESTRATOR
# =============================================================================

@traceable(name="run_all_validators", run_type="chain")
async def run_all_validators(adapted_json: dict, scenario_prompt: str) -> ValidationReport:
    """Run all 6 validation agents in PARALLEL."""
    logger.info("[VALIDATOR] Running all 6 agents in parallel...")

    # Strip locked shards before validation - they contain original content
    unlocked_json = strip_locked_content(adapted_json)
    logger.info("[VALIDATOR] Stripped locked shards (scenarioOptions, workspaceIds)")

    results = await asyncio.gather(
        validate_domain_fidelity(unlocked_json, scenario_prompt),
        validate_context_fidelity(unlocked_json, scenario_prompt),
        validate_resource_quality(unlocked_json, scenario_prompt),
        validate_klo_question_alignment(unlocked_json, scenario_prompt),
        validate_consistency(unlocked_json, scenario_prompt),
        validate_completeness(unlocked_json, scenario_prompt),
        return_exceptions=True
    )

    agent_results = []
    for r in results:
        if isinstance(r, Exception):
            agent_results.append(AgentResult(agent_name="Error", score=0.0, passed=False))
        else:
            agent_results.append(r)

    scores = [r.score for r in agent_results]
    overall = sum(scores) / len(scores) if scores else 0.0
    total_issues = sum(len(r.issues) for r in agent_results)
    all_passed = all(r.passed for r in agent_results)

    logger.info(f"[VALIDATOR] Overall: {overall:.2%}, Issues: {total_issues}, Passed: {all_passed}")

    return ValidationReport(
        overall_score=overall,
        passed=all_passed,
        agent_results=agent_results,
        total_issues=total_issues,
        needs_repair=not all_passed
    )


# =============================================================================
# REPAIR AGENT
# =============================================================================

def apply_patches(json_obj: dict, patches: list) -> dict:
    """Apply a list of find/replace patches to JSON content."""
    import copy
    result = copy.deepcopy(json_obj)
    content_str = json.dumps(result, ensure_ascii=False)

    for patch in patches:
        find = patch.get("find", "")
        replace = patch.get("replace", "")
        if find and find != replace:
            content_str = content_str.replace(find, replace)

    return json.loads(content_str)


@traceable(name="repair_issues", run_type="chain")
async def repair_issues(adapted_json: dict, scenario_prompt: str, report: ValidationReport) -> dict:
    """Repair Agent: Fix issues using targeted patches (not full JSON replacement)."""
    if report.passed:
        return adapted_json

    logger.info(f"[REPAIR] Fixing {report.total_issues} issues with patches...")

    all_issues = []
    for ar in report.agent_results:
        for issue in ar.issues:
            all_issues.append({"agent": issue.agent, "issue": issue.issue, "suggestion": issue.suggestion})

    # Only send a sample of content for context, not the full JSON
    content_sample = json.dumps(adapted_json, indent=2)[:15000]

    prompt = f"""You are a REPAIR AGENT. Generate find/replace patches to fix issues.

## SCENARIO (source of truth):
{scenario_prompt}

## ISSUES TO FIX:
{json.dumps(all_issues[:30], indent=2)}

## CONTENT SAMPLE (for context):
{content_sample}
...

## TASK:
Generate a list of FIND/REPLACE patches to fix the issues.
Each patch should find a specific string and replace it.

## OUTPUT FORMAT (JSON array only):
[
    {{"find": "exact string to find", "replace": "replacement string"}},
    {{"find": "another string", "replace": "its replacement"}}
]

Return ONLY the JSON array of patches. No explanations."""

    try:
        response = await _call_gpt_async(prompt)
        patches = _parse_json_response(response)

        if isinstance(patches, list) and patches:
            logger.info(f"[REPAIR] Applying {len(patches)} patches")
            return apply_patches(adapted_json, patches)
        else:
            logger.warning("[REPAIR] No valid patches returned")
            return adapted_json
    except Exception as e:
        logger.error(f"[REPAIR] Failed: {e}")
        return adapted_json


@traceable(name="validate_and_repair", run_type="chain")
async def validate_and_repair(adapted_json: dict, scenario_prompt: str, max_iter: int = 3) -> tuple[dict, ValidationReport]:
    """Full validation + repair loop."""
    current = adapted_json

    for i in range(max_iter):
        logger.info(f"[V+R] Iteration {i+1}/{max_iter}")
        report = await run_all_validators(current, scenario_prompt)

        if report.passed or report.overall_score >= ACCEPTABLE_THRESHOLD:
            return current, report

        current = await repair_issues(current, scenario_prompt, report)

    final_report = await run_all_validators(current, scenario_prompt)
    return current, final_report
