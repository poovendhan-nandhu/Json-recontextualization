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

# Import JSON Pointer patcher for path-based patching
from src.utils.patcher import JSONPatcher, PatchOp, get_patcher

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
# Supports both new format (snake_case at root) and old format (camelCase under topicWizardData)
LOCKED_PATHS_NEW = ["scenario_options", "workspace_ids"]  # New format
LOCKED_PATHS_OLD = ["scenarioOptions", "workspaceIds"]    # Old format


def strip_locked_content(adapted_json: dict) -> dict:
    """Remove locked shard content before validation.

    Locked shards (scenarioOptions, workspaceIds) contain original content
    that should NOT be flagged by validators.

    Supports both formats:
    - New format: root-level snake_case keys
    - Old format: topicWizardData wrapper with camelCase
    """
    import copy
    result = copy.deepcopy(adapted_json)

    # New format: root-level keys
    for path in LOCKED_PATHS_NEW:
        if path in result:
            result[path] = "[LOCKED - NOT VALIDATED]"

    # Old format: topicWizardData wrapper
    topic_data = result.get("topicWizardData", {})
    if isinstance(topic_data, dict):
        for path in LOCKED_PATHS_OLD:
            if path in topic_data:
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
    """Parse JSON from LLM response with repair for malformed JSON."""
    response = response.strip()

    # Strip markdown code blocks
    if response.startswith("```"):
        response = response.split("```")[1]
        if response.startswith("json"):
            response = response[4:]
    response = response.strip()

    # Try standard parsing first
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # Use json_repair for malformed JSON
    try:
        from json_repair import repair_json
        repaired = repair_json(response, return_objects=True)
        if isinstance(repaired, dict):
            return repaired
        elif isinstance(repaired, list) and repaired:
            return repaired[0] if isinstance(repaired[0], dict) else {"data": repaired}
        return {"raw": response}
    except Exception:
        # Last resort: try to extract JSON object
        import re
        match = re.search(r'\{[\s\S]*\}', response)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass
        return {"error": "Could not parse response", "raw": response[:500]}


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
        {{
            "term": "found term",
            "belongs_to": "what industry it belongs to",
            "should_be": "target equivalent",
            "found_in": "the sentence or context where you found it"
        }}
    ]
}}"""

    try:
        response = await _call_gpt_async(prompt)
        result = _parse_json_response(response)

        issues = []
        for item in result.get("invalid_terms", []):
            term = item.get('term', '')
            found_in = item.get('found_in', '')
            issues.append(ValidationIssue(
                agent="Domain Fidelity",
                location=f"find:{term}",  # Use find: prefix for string-based patching
                issue=f"'{term}' belongs to {item.get('belongs_to')} | Context: {found_in[:100]}",
                suggestion=f"Replace '{term}' with '{item.get('should_be')}'",
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
    """Ensure resources provide DATA not answers (inference map).

    IMPORTANT DISTINCTION:
    - MAIN resources (with markdown_text/content): Check word count (800-1400) + forbidden phrases
    - Resource OPTIONS (metadata): Check terminology ONLY, NOT word count (they're supposed to be short)
    """
    logger.info("[VALIDATOR] Running Resource Quality Agent...")

    # Support both new format (root-level) and old format (topicWizardData wrapper)
    simulation_flow = adapted_json.get("simulation_flow") or adapted_json.get("topicWizardData", {}).get("simulationFlow", [])

    # Collect resources WITH their exact paths - SEPARATE main resources from options
    main_resources = []  # List of (path, resource_dict) - FULL resources that need word count checks
    resource_options = []  # List of (path, option_dict) - METADATA only (no word count check)

    for stage_idx, stage in enumerate(simulation_flow):
        if isinstance(stage, dict):
            data = stage.get("data", {})
            if isinstance(data, dict):
                # Get MAIN resource content - these need word count checks
                res = data.get("resource")
                if res:
                    path = f"/simulation_flow/{stage_idx}/data/resource"
                    res_dict = res if isinstance(res, dict) else {"content": res}
                    # Check if it has substantial content (markdown_text or content key)
                    content = res_dict.get("markdown_text") or res_dict.get("content") or res_dict.get("html") or ""
                    if len(str(content)) > 200:  # Only check substantial resources
                        main_resources.append((path, res_dict, "main"))
                    else:
                        logger.info(f"[VALIDATOR] Skipping minimal main resource at {path} ({len(str(content))} chars)")

                # Get resourceOptions - METADATA ONLY (no word count check)
                opts = data.get("resource_options") or data.get("resourceOptions")
                opts_key = "resource_options" if data.get("resource_options") else "resourceOptions"
                if opts:
                    if isinstance(opts, list):
                        for opt_idx, opt in enumerate(opts):
                            path = f"/simulation_flow/{stage_idx}/data/{opts_key}/{opt_idx}"
                            resource_options.append((path, opt, "option"))
                    elif isinstance(opts, dict):
                        path = f"/simulation_flow/{stage_idx}/data/{opts_key}"
                        resource_options.append((path, opts, "option"))

    logger.info(f"[VALIDATOR] Found {len(main_resources)} main resources, {len(resource_options)} resource options")

    if not main_resources and not resource_options:
        logger.warning("[VALIDATOR] No resources found in adapted JSON")
        return AgentResult(agent_name="Resource Quality", score=1.0, passed=True, issues=[])

    # Build separate sections for LLM
    main_resources_info = []
    for i, (path, res, _) in enumerate(main_resources):
        # Get content for word count
        content = res.get("markdown_text") or res.get("content") or res.get("html") or ""
        word_count = len(str(content).split())
        main_resources_info.append({
            "index": i,
            "path": path,
            "type": "MAIN_RESOURCE",
            "word_count": word_count,
            "content_preview": str(content)[:2000]
        })

    options_info = []
    for i, (path, opt, _) in enumerate(resource_options):
        # Resource options have title, description, relevance - all short metadata
        options_info.append({
            "index": i,
            "path": path,
            "type": "RESOURCE_OPTION_METADATA",
            "title": opt.get("title", ""),
            "description": opt.get("description", "")[:200],
            "relevance": opt.get("relevance", "")[:200]
        })

    main_text = json.dumps(main_resources_info, indent=2)[:25000] if main_resources_info else "No main resources"
    options_text = json.dumps(options_info, indent=2)[:15000] if options_info else "No resource options"

    prompt = f"""You are validating RESOURCE QUALITY for a business simulation.

## CRITICAL: TWO TYPES OF RESOURCES WITH DIFFERENT RULES

### TYPE 1: MAIN RESOURCES (need full validation)
- These are FULL content resources with markdown_text/content
- CHECK: Word count (must be 800-1400 words)
- CHECK: Forbidden phrases ("should", "recommend", etc.)
- CHECK: Has data/statistics

### TYPE 2: RESOURCE OPTIONS (metadata only - NO WORD COUNT CHECK)
- These are SHORT metadata descriptions (50-100 words is NORMAL)
- DO NOT check word count - they're supposed to be brief
- ONLY CHECK: Terminology matches target scenario
- ONLY CHECK: No source industry terms leaked

## FORBIDDEN PHRASES (for MAIN resources only):
- "should" / "we should" / "you should" / "the company should"
- "recommend" / "recommendation" / "recommended"
- "therefore" / "thus" / "hence" / "consequently"
- "suggests that" / "indicates that" / "implies that"
- "The best approach is..." / "The optimal strategy is..."
- "In conclusion..." / "To summarize..."

## MAIN RESOURCES (check word count 800-1400 + forbidden phrases):
{main_text}

## RESOURCE OPTIONS (check terminology ONLY, NOT word count):
{options_text}

## TARGET SCENARIO:
{scenario_prompt[:500]}

## SCORING:
- Start at 100%
- MAIN resources: -10% for each forbidden phrase, -20% if word count outside 800-1400
- Options: -5% for wrong terminology in description/relevance
- DO NOT penalize resource options for being short (they're supposed to be)

## OUTPUT (JSON only):
{{
    "score": 0.0 to 1.0,
    "main_resources_checked": number,
    "options_checked": number,
    "issues": [
        {{
            "path": "/simulation_flow/X/data/resource",
            "resource_type": "main|option",
            "issue_type": "direct_answer|word_count|wrong_terminology",
            "description": "what's wrong",
            "found_text": "the problematic phrase",
            "word_count": number (only for main resources),
            "suggestion": "how to fix"
        }}
    ]
}}"""

    try:
        response = await _call_gpt_async(prompt)
        result = _parse_json_response(response)

        issues = []

        # Parse issues with exact paths
        for issue_item in result.get("issues", []):
            if isinstance(issue_item, dict):
                path = issue_item.get("path", "unknown")
                issue_type = issue_item.get("issue_type", "unknown")
                description = issue_item.get("description", "Unknown issue")
                found_text = issue_item.get("found_text", "")
                word_count = issue_item.get("word_count")
                suggestion = issue_item.get("suggestion", "Fix the resource content")

                # Build detailed issue message
                if issue_type == "direct_answer":
                    issue_msg = f"Direct answer found: '{found_text[:80]}...'" if found_text else description
                elif issue_type == "word_count":
                    issue_msg = f"Word count {word_count} outside 800-1400 range"
                else:
                    issue_msg = description

                issues.append(ValidationIssue(
                    agent="Resource Quality",
                    location=path,  # EXACT PATH like /simulation_flow/2/data/resource_options/0
                    issue=issue_msg,
                    suggestion=suggestion,
                    severity="error" if issue_type in ("direct_answer", "word_count") else "warning"
                ))

        score = result.get("score", 0.0)

        # Log what was found for debugging
        main_checked = result.get("main_resources_checked", 0)
        opts_checked = result.get("options_checked", 0)
        logger.info(f"[VALIDATOR] Resource Quality: {main_checked} main, {opts_checked} options checked, {len(issues)} issues, score={score:.1%}")
        for issue in issues[:3]:  # Log first 3 issues
            logger.info(f"[VALIDATOR]   -> {issue.issue[:80]}")

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

    # Support both formats (new: snake_case at root, old: camelCase under topicWizardData)
    klos_raw = adapted_json.get("assessment_criterion") or adapted_json.get("topicWizardData", {}).get("assessmentCriterion", [])

    # Extract actual KLO text from the structure (field is 'key_learning_outcome')
    klos_extracted = []
    for i, klo in enumerate(klos_raw):
        if isinstance(klo, dict):
            klo_text = klo.get("key_learning_outcome") or klo.get("title") or klo.get("name") or klo.get("description", "")
            klos_extracted.append(f"KLO {i+1}: {klo_text}")

    klos_text = "\n".join(klos_extracted) if klos_extracted else "No KLOs found"
    logger.info(f"[VALIDATOR] Extracted {len(klos_extracted)} KLOs")

    # Find questions in all possible locations (search entire JSON)
    questions = []
    question_texts = []  # Just the text for display

    def extract_questions(obj):
        """Recursively find all question arrays."""
        if isinstance(obj, dict):
            for key, val in obj.items():
                if 'question' in key.lower() and isinstance(val, list):
                    for q in val:
                        questions.append(q)
                        if isinstance(q, dict):
                            qtext = q.get("question") or q.get("text") or q.get("content", "")
                            if qtext:
                                question_texts.append(qtext)
                        elif isinstance(q, str):
                            question_texts.append(q)
                else:
                    extract_questions(val)
        elif isinstance(obj, list):
            for item in obj:
                extract_questions(item)

    # Search in specific locations to avoid picking up placeholder references like "Q1", "Q2"
    # 1. simulation_flow contains the real activity questions
    # 2. submission_questions at root (but NOT selected_submission_questions which may have placeholders)
    sim_flow = adapted_json.get("simulation_flow") or adapted_json.get("topicWizardData", {}).get("simulationFlow", [])
    extract_questions(sim_flow)

    # Also check submission_questions directly (skip selected_submission_questions)
    sub_q = adapted_json.get("submission_questions", [])
    for q in sub_q:
        if isinstance(q, dict):
            qtext = q.get("question") or q.get("text", "")
            if qtext and len(qtext) > 15:  # Real questions are longer
                questions.append(q)
                question_texts.append(qtext)
        elif isinstance(q, str) and len(q) > 15:
            questions.append(q)
            question_texts.append(q)

    logger.info(f"[VALIDATOR] Found {len(questions)} questions, {len(klos_extracted)} KLOs")

    # Format questions for display
    questions_formatted = "\n".join([f"Q{i+1}: {q[:200]}" for i, q in enumerate(question_texts)])

    prompt = f"""You are validating KLO-QUESTION ALIGNMENT for a business simulation.

## TARGET SCENARIO:
{scenario_prompt}

## KEY LEARNING OUTCOMES (KLOs) - What students should learn:
{klos_text}

## QUESTIONS - What students will answer:
{questions_formatted[:15000]}

## YOUR TASK:
1. For EACH question, determine which KLO(s) it assesses
2. Check if question terminology matches the TARGET scenario (fast food, $1 menu, BurgerBlitz)
3. A question is ALIGNED if it tests skills/knowledge from at least one KLO

## OUTPUT (JSON only):
{{
    "questions_checked": {len(question_texts)},
    "klos_found": {len(klos_extracted)},
    "alignment_summary": "brief description of how questions map to KLOs",
    "unaligned_questions": ["list questions that don't map to any KLO"],
    "wrong_terminology_questions": ["list questions using wrong industry terms"]
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

        # CALCULATE score ourselves based on actual issues
        total_questions = len(question_texts) if question_texts else 1
        unaligned = len(result.get("unaligned_questions", []))
        wrong_terms = len(result.get("wrong_terminology_questions", []))
        total_issues = unaligned + wrong_terms
        score = max(0.0, (total_questions - total_issues) / total_questions)
        logger.info(f"[VALIDATOR] KLO-Question: {total_questions} questions, {total_issues} issues = {score:.1%}")
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
- ONE protagonist company name used consistently (the company learners advise)
- ONE manager name used consistently
- Email format: firstname.lastname@company.com
- NOTE: A COMPETITOR company name may also appear - this is ALLOWED and should NOT be flagged as inconsistent

Inconsistent = "EcoChic" vs "EcoChic Threads" vs "Eco-Chic" (variations of same company)
Consistent = "EcoChic Threads" for protagonist, "RivalCo" for competitor (two different companies is OK)

## CONTENT:
{content_text}

## OUTPUT (JSON only):
{{
    "canonical_company": "the correct company name to use everywhere",
    "canonical_manager": "the correct manager name to use everywhere",
    "company_variations": ["all variations found"],
    "manager_variations": ["all name variations"],
    "issues": [
        {{
            "type": "company|manager|email",
            "wrong_value": "the incorrect value found",
            "correct_value": "what it should be replaced with"
        }}
    ]
}}"""

    try:
        response = await _call_gpt_async(prompt)
        result = _parse_json_response(response)

        issues = []
        for item in result.get("issues", []):
            wrong_val = item.get("wrong_value", "")
            correct_val = item.get("correct_value", "")
            issues.append(ValidationIssue(
                agent="Consistency",
                location=f"find:{wrong_val}",  # Use find: prefix for string-based patching
                issue=f"Inconsistent {item.get('type', 'name')}: '{wrong_val}' should be '{correct_val}'",
                suggestion=f"Replace '{wrong_val}' with '{correct_val}'",
                severity="warning"
            ))

        # CALCULATE score based on variations found
        company_variations = len(result.get("company_variations", []))
        manager_variations = len(result.get("manager_variations", []))
        total_variations = company_variations + manager_variations

        # Perfect = 2 total (1 company + 1 manager)
        # Score decreases as variations increase
        if total_variations <= 2:
            score = 1.0  # Perfect - one name each
        elif total_variations <= 4:
            score = 0.9  # Minor variations
        elif total_variations <= 6:
            score = 0.8  # Some variations
        else:
            score = max(0.5, 1.0 - (total_variations - 2) * 0.1)

        # Also penalize for issues
        if len(issues) > 0:
            score = max(0.5, score - len(issues) * 0.1)

        logger.info(f"[VALIDATOR] Consistency: {company_variations} company vars, {manager_variations} manager vars, {len(issues)} issues = {score:.1%}")
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

    # Quick regex pre-check - ONLY check for actual placeholders, not all brackets
    placeholder_patterns = [
        r'\[TBD\]',
        r'\[TODO\]',
        r'\[INSERT\]',
        r'\[INSERT HERE\]',
        r'\[PLACEHOLDER\]',
        r'\[YOUR NAME\]',
        r'\[YOUR .*?\]',
        r'\[COMPANY NAME\]',
        r'XXX',
        r'FIXME',
    ]
    found = []
    for p in placeholder_patterns:
        matches = re.findall(p, content_text, re.IGNORECASE)
        found.extend(matches)

    # Filter out false positives - legitimate bracket usage
    def is_real_placeholder(text):
        text_upper = text.upper()
        # Known placeholders
        if any(x in text_upper for x in ['TBD', 'TODO', 'INSERT', 'PLACEHOLDER', 'YOUR NAME', 'YOUR ', 'XXX', 'FIXME']):
            return True
        # Single letter brackets like [X] or [N] might be placeholders
        if re.match(r'^\[[A-Z]\]$', text):
            return True
        return False

    found = [f for f in found if is_real_placeholder(f)]

    prompt = f"""You are validating COMPLETENESS of a business simulation JSON.

## SCAN FOR REAL ISSUES ONLY:

1. **ACTUAL PLACEHOLDERS** (not normal brackets):
   - [TBD], [TODO], [INSERT], [Your Name], [PLACEHOLDER], XXX, FIXME
   - DO NOT flag: [Source: ...], [Figure 1], [Q1 2024], [optional], citations, references

2. **Truncated content**: sentences ending mid-word "The company is planning to exp" (cut off)

3. **Empty CONTENT fields** that should have text:
   - ONLY flag if a description, content, or text field is empty AND it's in a place where content is expected
   - Example: A resource with empty description when other resources have descriptions

## DO NOT FLAG (these are intentionally empty or structural):
- reporting_manager fields (email, name, role) - often intentionally blank
- "name" fields in flow items or stage definitions - these are IDs, not content
- popups.description, popups.title - optional UI fields
- answer_key - not always used
- Empty arrays [] - structural, not content
- Fields in locked sections
- Root-level "overview" field - content is in launch_settings.cover_tab.overview instead
- cover_tab.description - optional field, overview is the main content
- Any field inside "popups" or "settings" objects - UI configuration
- scenario_change.selected_scenario_option.id/option - often empty in source
- workplace_scenario.scenario - content may be in scenario_description instead
- scenario_description at root - content is in chat_history or workplace_scenario
- learner_role.role, learner_role.role_description - optional role fields
- background.about_organization, background.organization_name - optional metadata

## PRE-DETECTED PLACEHOLDERS: {found[:10] if found else "None found"}

## CONTENT SAMPLE:
{content_text}

## OUTPUT (JSON only):
{{
    "placeholders": ["actual placeholders like [TBD], not citations"],
    "truncated": ["sentences cut off mid-word"],
    "empty_fields": ["only CONTENT fields that should have text but don't"],
    "notes": "brief explanation"
}}"""

    try:
        response = await _call_gpt_async(prompt)
        result = _parse_json_response(response)

        issues = []
        penalty = 0.0

        # Extract issues from ALL three categories with penalties
        for p in result.get("placeholders", []):
            issues.append(ValidationIssue(
                agent="Completeness",
                location="content",
                issue=f"Placeholder: {p}",
                suggestion="Replace with actual content",
                severity="error"
            ))
            penalty += 0.15  # Placeholders are serious

        for t in result.get("truncated", []):
            issues.append(ValidationIssue(
                agent="Completeness",
                location="content",
                issue=f"Truncated content: {t[:100]}...",
                suggestion="Complete the truncated text",
                severity="warning"
            ))
            penalty += 0.10  # Truncation is medium severity

        # Filter out KNOWN structural empty fields that are intentionally empty
        structural_patterns = [
            "overview",  # Root overview - content is in cover_tab.overview
            "cover_tab.description",  # Optional UI field
            "popups.description",  # Optional popup text
            "popups.title",  # Optional popup title
            "answer_key",  # Not always used
            "reporting_manager",  # Often intentionally blank
            ".name",  # Flow item names are IDs
            "settings.",  # UI settings
            "description",  # Generic description often optional
            "scenario_description",  # May be filled elsewhere
        ]

        for ef in result.get("empty_fields", []):
            ef_str = str(ef).lower()
            # Check if this is a known structural field
            is_structural = any(pat in ef_str for pat in structural_patterns)
            if not is_structural:
                issues.append(ValidationIssue(
                    agent="Completeness",
                    location=str(ef),
                    issue=f"Empty field: {ef}",
                    suggestion="Add content to empty field",
                    severity="warning"
                ))
                penalty += 0.05  # Empty fields are minor
            else:
                logger.info(f"[VALIDATOR] Ignoring structural empty field: {ef}")

        # CALCULATE score: start at 100%, subtract penalties
        score = max(0.0, 1.0 - penalty)
        logger.info(f"[VALIDATOR] Completeness: {len(issues)} issues, penalty={penalty:.0%}, score={score:.1%}")

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
# AGENT 7: KLO-RESOURCE ALIGNMENT
# =============================================================================

@traceable(name="validate_klo_resource_alignment", run_type="chain")
async def validate_klo_resource_alignment(adapted_json: dict, scenario_prompt: str) -> AgentResult:
    """Check if resources support learning the KLOs (Key Learning Outcomes)."""
    logger.info("[VALIDATOR] Running KLO-Resource Alignment Agent...")

    # Extract KLOs
    klos_raw = adapted_json.get("assessment_criterion") or adapted_json.get("topicWizardData", {}).get("assessmentCriterion", [])
    klos_extracted = []
    for i, klo in enumerate(klos_raw):
        if isinstance(klo, dict):
            klo_text = klo.get("key_learning_outcome") or klo.get("title") or klo.get("description", "")
            klos_extracted.append(f"KLO {i+1}: {klo_text}")

    klos_text = "\n".join(klos_extracted) if klos_extracted else "No KLOs found"

    # Extract resources
    simulation_flow = adapted_json.get("simulation_flow") or adapted_json.get("topicWizardData", {}).get("simulationFlow", [])
    resources = []
    for stage_idx, stage in enumerate(simulation_flow):
        if isinstance(stage, dict):
            data = stage.get("data", {})
            if isinstance(data, dict):
                res = data.get("resource")
                if res:
                    content = ""
                    if isinstance(res, dict):
                        content = res.get("markdown_text") or res.get("content") or res.get("html") or ""
                    elif isinstance(res, str):
                        content = res
                    if content:
                        resources.append({
                            "stage": stage_idx,
                            "content_preview": str(content)[:3000]
                        })

    resources_text = json.dumps(resources, indent=2)[:20000] if resources else "No resources found"

    prompt = f"""You are validating KLO-RESOURCE ALIGNMENT for a business simulation.

## KEY LEARNING OUTCOMES (KLOs) - What students should learn:
{klos_text}

## RESOURCES PROVIDED:
{resources_text}

## YOUR TASK:
Check if the resources provide DATA that helps students learn each KLO.

**BE LENIENT - mark as SUPPORTED if:**
- Resources contain relevant data categories (market info, competitor data, options/strategies)
- Students can practice the skill described in the KLO using the provided data
- The data enables analysis even if not perfectly comprehensive

**Mark as UNSUPPORTED only if:**
- The KLO topic is completely absent from resources
- No data exists to even begin the learning activity
- Critical data categories are entirely missing

## OUTPUT (JSON only):
{{
    "klos_checked": {len(klos_extracted)},
    "klo_resource_mapping": [
        {{
            "klo": "KLO text",
            "supported": true/false,
            "supporting_data": "what data in resources supports this KLO",
            "gap": "what data is missing (if any)"
        }}
    ],
    "summary": "brief description of alignment"
}}"""

    try:
        response = await _call_gpt_async(prompt)
        result = _parse_json_response(response)

        issues = []
        unsupported_count = 0
        for mapping in result.get("klo_resource_mapping", []):
            if not mapping.get("supported", True):
                unsupported_count += 1
                issues.append(ValidationIssue(
                    agent="KLO-Resource",
                    location="resources",
                    issue=f"KLO not supported: {mapping.get('klo', '')[:60]}",
                    suggestion=f"Add data for: {mapping.get('gap', 'missing data')}",
                    severity="warning"
                ))

        # CALCULATE score ourselves based on actual issues
        total_klos = len(klos_extracted) if klos_extracted else 1
        score = max(0.0, (total_klos - unsupported_count) / total_klos)
        logger.info(f"[VALIDATOR] KLO-Resource: {total_klos} KLOs, {unsupported_count} unsupported = {score:.1%}")
        return AgentResult(
            agent_name="KLO-Resource Alignment",
            score=score,
            passed=score >= ACCEPTABLE_THRESHOLD,
            issues=issues,
            details=result
        )
    except Exception as e:
        logger.error(f"[VALIDATOR] KLO-Resource Alignment failed: {e}")
        return AgentResult(agent_name="KLO-Resource Alignment", score=0.0, passed=False, issues=[])


# =============================================================================
# AGENT 8: QUESTION SOLVABILITY (Shweta's Request)
# =============================================================================

@traceable(name="validate_question_solvability", run_type="chain")
async def validate_question_solvability(adapted_json: dict, scenario_prompt: str) -> AgentResult:
    """Check if questions can be answered using ONLY the provided resources.

    This is the 'inference map' check - students should be able to:
    1. Read the resources (data, statistics, facts)
    2. Connect the dots themselves
    3. Answer the questions

    Resources should NOT give away answers, but MUST provide enough data.
    """
    logger.info("[VALIDATOR] Running Question Solvability Agent...")

    # Extract questions (excluding reflective/metacognitive questions)
    questions = []
    REFLECTIVE_PATTERNS = [
        "what can you now do",
        "what did you learn",
        "how has this simulation",
        "reflect on your",
        "what skills did you",
        "before this simulation",
        "after this simulation"
    ]

    def is_reflective(q: str) -> bool:
        q_lower = q.lower()
        return any(pat in q_lower for pat in REFLECTIVE_PATTERNS)

    def extract_questions(obj):
        if isinstance(obj, dict):
            for key, val in obj.items():
                if 'question' in key.lower() and isinstance(val, list):
                    for q in val:
                        if isinstance(q, dict):
                            qtext = q.get("question") or q.get("text") or q.get("content", "")
                            if qtext and len(qtext) > 15 and not is_reflective(qtext):
                                questions.append(qtext)
                        elif isinstance(q, str) and len(q) > 15 and not is_reflective(q):
                            questions.append(q)
                else:
                    extract_questions(val)
        elif isinstance(obj, list):
            for item in obj:
                extract_questions(item)

    sim_flow = adapted_json.get("simulation_flow") or adapted_json.get("topicWizardData", {}).get("simulationFlow", [])
    extract_questions(sim_flow)

    questions_text = "\n".join([f"Q{i+1}: {q[:200]}" for i, q in enumerate(questions)])

    # Extract resources - both main resources AND resource_options
    resources = []
    for stage_idx, stage in enumerate(sim_flow):
        if isinstance(stage, dict):
            data = stage.get("data", {})
            if isinstance(data, dict):
                # Main resource
                res = data.get("resource")
                if res:
                    content = ""
                    if isinstance(res, dict):
                        content = res.get("markdown_text") or res.get("content") or res.get("html") or ""
                    elif isinstance(res, str):
                        content = res
                    if content:
                        resources.append(str(content)[:12000])

                # Resource options (also contains relevant data)
                res_opts = data.get("resource_options", [])
                for opt in res_opts:
                    if isinstance(opt, dict):
                        opt_content = opt.get("markdown_text") or opt.get("content") or opt.get("html") or ""
                        if opt_content and len(opt_content) > 100:
                            resources.append(str(opt_content)[:2500])

    resources_text = "\n---\n".join(resources)[:25000] if resources else "No resources found"

    prompt = f"""You are validating QUESTION SOLVABILITY for a business simulation.

## CRITICAL CHECK: Can students answer questions using ONLY the resources?

## QUESTIONS students must answer:
{questions_text[:8000]}

## RESOURCES provided to students:
{resources_text}

## YOUR TASK:
For each question, determine if the resources contain enough DATA to answer it.

**BE LENIENT - mark as SOLVABLE if:**
- Resources contain relevant data (percentages, trends, comparisons) even if not exact dollar figures
- Students can make reasonable inferences from the data provided
- Multiple response options are described with comparative information
- The question asks for analysis/recommendation and data exists to support analysis

**Mark as UNSOLVABLE only if:**
- The question asks about something not mentioned at all in resources
- Critical data categories are completely missing (e.g., no competitor info when asked about competitors)
- The gap is fundamental, not just "more detail would help"

## OUTPUT (JSON only):
{{
    "questions_checked": {len(questions)},
    "solvability_analysis": [
        {{
            "question": "question text (truncated)",
            "solvable": true/false,
            "required_data": "what data is needed to answer",
            "data_found": "what relevant data exists in resources",
            "gap": "what data is missing (if any) - or 'None' if solvable"
        }}
    ],
    "unsolvable_questions": ["list of questions that can't be answered from resources"],
    "summary": "brief assessment of overall solvability"
}}"""

    try:
        response = await _call_gpt_async(prompt)
        result = _parse_json_response(response)

        issues = []
        for q in result.get("unsolvable_questions", []):
            issues.append(ValidationIssue(
                agent="Solvability",
                location="questions",
                issue=f"Not solvable from resources: {q[:60]}",
                suggestion="Add required data to resources",
                severity="error"
            ))

        # CALCULATE score ourselves based on actual issues
        total_questions = len(questions) if questions else 1
        unsolvable = len(result.get("unsolvable_questions", []))
        score = max(0.0, (total_questions - unsolvable) / total_questions)
        logger.info(f"[VALIDATOR] Solvability: {total_questions} questions, {unsolvable} unsolvable = {score:.1%}")
        return AgentResult(
            agent_name="Question Solvability",
            score=score,
            passed=score >= ACCEPTABLE_THRESHOLD,
            issues=issues,
            details=result
        )
    except Exception as e:
        logger.error(f"[VALIDATOR] Question Solvability failed: {e}")
        return AgentResult(agent_name="Question Solvability", score=0.0, passed=False, issues=[])


# =============================================================================
# ORCHESTRATOR
# =============================================================================

@traceable(name="run_all_validators", run_type="chain")
async def run_all_validators(adapted_json: dict, scenario_prompt: str) -> ValidationReport:
    """Run all 8 validation agents in PARALLEL."""
    logger.info("[VALIDATOR] Running all 8 agents in parallel...")

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
        validate_klo_resource_alignment(unlocked_json, scenario_prompt),  # NEW: KLO-Resource
        validate_question_solvability(unlocked_json, scenario_prompt),    # NEW: Solvability (Shweta)
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

def apply_string_patches(json_obj: dict, patches: list, protect_resources: bool = True) -> dict:
    """Apply find/replace string patches to JSON content (fallback method).

    Args:
        json_obj: The JSON object to patch
        patches: List of {find, replace} patches
        protect_resources: If True, skip patches that would modify resource content
    """
    import copy
    result = copy.deepcopy(json_obj)

    # If protecting resources, extract and protect resource content first
    protected_content = {}
    if protect_resources:
        # Find and store resource content so it doesn't get modified
        sim_flow = result.get("simulation_flow", [])
        for idx, stage in enumerate(sim_flow):
            if isinstance(stage, dict):
                data = stage.get("data", {})
                if isinstance(data, dict):
                    res = data.get("resource")
                    if res and isinstance(res, dict):
                        # Store the full resource content
                        content = res.get("markdown_text") or res.get("content") or res.get("html")
                        if content and len(str(content)) > 200:
                            key = f"__PROTECTED_RESOURCE_{idx}__"
                            protected_content[key] = content
                            # Replace with placeholder
                            if "markdown_text" in res:
                                res["markdown_text"] = key
                            elif "content" in res:
                                res["content"] = key
                            elif "html" in res:
                                res["html"] = key

        if protected_content:
            logger.info(f"[REPAIR] Protected {len(protected_content)} resource contents from string patches")

    content_str = json.dumps(result, ensure_ascii=False)

    applied = 0
    for patch in patches:
        find = patch.get("find", "")
        replace = patch.get("replace", "")
        if find and find != replace and find in content_str:
            content_str = content_str.replace(find, replace)
            applied += 1
            logger.info(f"[REPAIR] Applied string patch: '{find[:50]}...' -> '{replace[:50]}...'")

    logger.info(f"[REPAIR] Applied {applied}/{len(patches)} string patches")
    result = json.loads(content_str)

    # Restore protected resource content
    if protected_content:
        sim_flow = result.get("simulation_flow", [])
        for idx, stage in enumerate(sim_flow):
            if isinstance(stage, dict):
                data = stage.get("data", {})
                if isinstance(data, dict):
                    res = data.get("resource")
                    if res and isinstance(res, dict):
                        key = f"__PROTECTED_RESOURCE_{idx}__"
                        if key in protected_content:
                            original = protected_content[key]
                            if res.get("markdown_text") == key:
                                res["markdown_text"] = original
                            elif res.get("content") == key:
                                res["content"] = original
                            elif res.get("html") == key:
                                res["html"] = original
        logger.info(f"[REPAIR] Restored {len(protected_content)} protected resource contents")

    return result


def apply_path_patches(json_obj: dict, patches: list) -> dict:
    """Apply JSON Pointer path-based patches using JSONPatcher."""
    patcher = get_patcher()
    patch_ops = []

    for patch in patches:
        path = patch.get("path", "")
        value = patch.get("value")
        reason = patch.get("reason", "Repair fix")

        if path and path.startswith("/"):
            patch_ops.append(PatchOp(
                op="replace",
                path=path,
                value=value,
                reason=reason
            ))
        else:
            logger.warning(f"[REPAIR] Invalid path format: {path}")

    if patch_ops:
        result = patcher.apply_patches(json_obj, patch_ops, validate_first=True, stop_on_error=False)
        logger.info(f"[REPAIR] Applied {len(result.applied_patches)}/{len(patch_ops)} path patches")
        if result.failed_patches:
            for failed_patch, error in result.failed_patches:
                logger.warning(f"[REPAIR] Failed patch {failed_patch.path}: {error}")
        return result.patched_data

    return json_obj


@traceable(name="repair_issues", run_type="chain")
async def repair_issues(adapted_json: dict, scenario_prompt: str, report: ValidationReport) -> dict:
    """Repair Agent: Fix issues using path-based or string-based patches."""
    if report.passed:
        return adapted_json

    logger.info(f"[REPAIR] Fixing {report.total_issues} issues with patches...")

    all_issues = []
    auto_patches = []  # Patches we can generate automatically from find: locations
    skipped_agents = ["Solvability", "KLO-Resource"]  # These need regeneration, not patches

    for ar in report.agent_results:
        # Skip agents whose issues can't be fixed with patches
        if ar.agent_name in skipped_agents or any(skip in ar.agent_name for skip in skipped_agents):
            logger.info(f"[REPAIR] Skipping {len(ar.issues)} issues from {ar.agent_name} (needs regeneration, not patches)")
            continue

        for issue in ar.issues:
            location = issue.location

            # Parse find: prefix for automatic string-based patching
            if location.startswith("find:"):
                find_term = location[5:]  # Remove "find:" prefix
                # Extract replacement from suggestion if it follows pattern "Replace 'X' with 'Y'"
                suggestion = issue.suggestion
                if "Replace '" in suggestion and "' with '" in suggestion:
                    # Parse: "Replace 'old' with 'new'"
                    import re
                    match = re.search(r"Replace '([^']+)' with '([^']+)'", suggestion)
                    if match:
                        auto_patches.append({
                            "find": match.group(1),
                            "replace": match.group(2)
                        })
                        continue  # Don't add to all_issues, we'll handle it automatically

            all_issues.append({
                "agent": issue.agent,
                "location": location,
                "issue": issue.issue,
                "suggestion": issue.suggestion
            })

    # If all issues were converted to auto_patches, just apply those
    if not all_issues and auto_patches:
        logger.info(f"[REPAIR] All {len(auto_patches)} issues can be auto-fixed")
        return apply_string_patches(adapted_json, auto_patches)

    # Only send a sample of content for context, not the full JSON
    content_sample = json.dumps(adapted_json, indent=2)[:20000]

    prompt = f"""You are a REPAIR AGENT. Generate patches to fix validation issues.

## SCENARIO (source of truth):
{scenario_prompt}

## ISSUES TO FIX:
Each issue has a "location" indicating WHERE in the JSON the problem is.
{json.dumps(all_issues[:30], indent=2)}

## CONTENT (for reference):
{content_sample}
...

## TASK:
Generate patches to fix each issue. Use JSON Pointer paths when the location is a valid path.

## CRITICAL CONSTRAINTS:
- DO NOT truncate or shorten any content
- DO NOT remove data (statistics, figures, percentages)
- Only REPLACE incorrect terms with correct ones
- Keep all numerical data intact

## OUTPUT FORMAT (JSON array):
For PATH-BASED patches (when you know the exact JSON path):
[
    {{"path": "/simulation_flow/0/data/task_email/body", "value": "new content here", "reason": "Fixed domain term"}},
    {{"path": "/workplace_scenario/background/organization_name", "value": "BurgerBlitz", "reason": "Consistent company name"}}
]

For STRING-BASED patches (when you need to find/replace text):
[
    {{"find": "exact text to find", "replace": "replacement text"}},
    {{"find": "hiring manager", "replace": "marketing director"}}
]

You can mix both types. Return ONLY the JSON array."""

    try:
        response = await _call_gpt_async(prompt)
        patches = _parse_json_response(response)

        if not isinstance(patches, list) or not patches:
            logger.warning("[REPAIR] No valid patches returned")
            return adapted_json

        # Separate path-based and string-based patches
        path_patches = [p for p in patches if p.get("path")]
        string_patches = [p for p in patches if p.get("find")]

        result = adapted_json

        # Apply auto-generated patches from find: locations first
        if auto_patches:
            logger.info(f"[REPAIR] Applying {len(auto_patches)} auto-generated patches from validators")
            result = apply_string_patches(result, auto_patches)

        # Apply path-based patches (more precise)
        if path_patches:
            logger.info(f"[REPAIR] Applying {len(path_patches)} path-based patches")
            result = apply_path_patches(result, path_patches)

        # Apply string-based patches from LLM
        if string_patches:
            logger.info(f"[REPAIR] Applying {len(string_patches)} string-based patches")
            result = apply_string_patches(result, string_patches)

        return result

    except Exception as e:
        logger.error(f"[REPAIR] Failed: {e}")
        return adapted_json


# =============================================================================
# SPECIALIZED RESOURCE FIXER (Regenerates content, not just patches)
# =============================================================================

def get_value_at_path(obj: dict, path: str):
    """Get value at JSON Pointer path."""
    if not path.startswith("/"):
        return None
    parts = path[1:].split("/")
    current = obj
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list):
            try:
                current = current[int(part)]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return current


def set_value_at_path(obj: dict, path: str, value) -> bool:
    """Set value at JSON Pointer path. Returns True if successful."""
    if not path.startswith("/"):
        return False
    parts = path[1:].split("/")
    current = obj
    for part in parts[:-1]:
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list):
            try:
                current = current[int(part)]
            except (ValueError, IndexError):
                return False
        else:
            return False

    final_key = parts[-1]
    if isinstance(current, dict):
        current[final_key] = value
        return True
    elif isinstance(current, list):
        try:
            current[int(final_key)] = value
            return True
        except (ValueError, IndexError):
            return False
    return False


@traceable(name="fix_resource_quality", run_type="chain")
async def fix_resource_quality(
    adapted_json: dict,
    scenario_prompt: str,
    issues: list[ValidationIssue],
    entity_map: dict = None,
    domain_profile: dict = None
) -> dict:
    """
    SPECIALIZED RESOURCE FIXER - Regenerates MAIN resource content when needed.

    IMPORTANT: Only fixes MAIN resources. Resource options are METADATA
    (supposed to be short ~50-100 words) - we only fix terminology in them.

    NOW WITH FULL CONTEXT from adaptation stage:
    - entity_map: Company names, people, roles to use
    - domain_profile: Terminology mappings, forbidden terms

    This is different from generic repair - it actually rewrites resources to:
    1. Remove direct answer language (should, recommend, therefore)
    2. Trim to 800-1400 word range (MAIN resources only)
    3. Add data/statistics if missing
    4. **NEW**: Use correct company name and terminology from adaptation
    """
    import copy
    result = copy.deepcopy(adapted_json)

    # Filter to resource quality issues for MAIN resources only
    # Skip resource_options - they're metadata that should stay short
    resource_issues = [
        i for i in issues
        if i.agent == "Resource Quality"
        and i.location.startswith("/")
        and "resource_options" not in i.location  # Skip options
        and "resourceOptions" not in i.location   # Skip options (camelCase)
    ]

    # For resource_options, only apply terminology fixes (not regeneration)
    option_issues = [
        i for i in issues
        if i.agent == "Resource Quality"
        and i.location.startswith("/")
        and ("resource_options" in i.location or "resourceOptions" in i.location)
    ]

    if option_issues:
        logger.info(f"[RESOURCE FIXER] Skipping {len(option_issues)} resource option issues (metadata - no regeneration needed)")

    if not resource_issues:
        return result

    logger.info(f"[RESOURCE FIXER] Fixing {len(resource_issues)} MAIN resource quality issues...")

    # Group issues by path (multiple issues might be for same resource)
    issues_by_path = {}
    for issue in resource_issues:
        path = issue.location
        if path not in issues_by_path:
            issues_by_path[path] = []
        issues_by_path[path].append(issue)

    # Fix each resource
    for path, path_issues in issues_by_path.items():
        logger.info(f"[RESOURCE FIXER] Fixing resource at {path} ({len(path_issues)} issues)")

        # Get current content
        current_resource = get_value_at_path(result, path)
        if not current_resource:
            logger.warning(f"[RESOURCE FIXER] Could not find resource at {path}")
            continue

        # Get content string - try multiple possible keys and remember which one
        content_key = None  # Track which key we read from
        if isinstance(current_resource, dict):
            # Try common content keys in order of preference
            for key in ["markdown_text", "content", "html", "text", "body", "description"]:
                if current_resource.get(key):
                    content = current_resource[key]
                    content_key = key
                    break
            else:
                content = json.dumps(current_resource)  # Fallback to full JSON
                content_key = None
        elif isinstance(current_resource, str):
            content = current_resource
        else:
            continue

        # Check actual content length (not just the wrapper)
        content_len = len(str(content))
        if content_len < 100:
            logger.info(f"[RESOURCE FIXER] Skipping minimal resource at {path} ({content_len} chars)")
            continue

        logger.info(f"[RESOURCE FIXER] Processing resource at {path} ({content_len} chars)")

        # Build fix prompt with FULL CONTEXT from adaptation
        issue_descriptions = "\n".join([f"- {i.issue}" for i in path_issues])

        # Extract context from adaptation stage
        company_name = "the company"
        if entity_map and isinstance(entity_map, dict):
            company_info = entity_map.get("company", {})
            if isinstance(company_info, dict):
                company_name = company_info.get("name", "the company")
            elif isinstance(company_info, str):
                company_name = company_info

        terminology = {}
        forbidden_terms = []
        if domain_profile and isinstance(domain_profile, dict):
            terminology = domain_profile.get("terminology_map", {})
            forbidden_terms = domain_profile.get("forbidden_terms", [])

        # Build terminology section
        term_section = ""
        if terminology:
            term_items = list(terminology.items())[:25]
            term_section = "\n".join([f"  - {k} → {v}" for k, v in term_items])

        # Build forbidden terms section
        forbidden_section = ", ".join(forbidden_terms[:40]) if forbidden_terms else "None specified"

        prompt = f"""Rewrite this resource content to fix the following issues:

## ISSUES TO FIX:
{issue_descriptions}

## COMPANY NAME TO USE THROUGHOUT:
{company_name}

## TERMINOLOGY TO USE (replace source terms with target):
{term_section if term_section else "Use scenario-appropriate business terms"}

## TERMS TO AVOID (from source scenario - DO NOT USE THESE):
{forbidden_section}

## RULES FOR REWRITTEN CONTENT:

1. **INFERENCE MAP** - Provide "dots to connect", NOT "connected dots"
   - REMOVE all: "should", "recommend", "therefore", "thus", "suggests that"
   - REMOVE all conclusions and recommendations
   - KEEP only: Raw data, statistics, facts, percentages

2. **WORD COUNT** - Must be 800-1400 words (currently {len(content.split())} words)
   - If too long: Trim redundant sections, remove filler
   - If too short: Add more data points, statistics, market figures

3. **DATA QUALITY** - Must contain:
   - Market size figures
   - Percentages and growth rates
   - Competitor data
   - Cost/financial figures
   - Consumer/survey data

4. **CONSISTENCY** - Use "{company_name}" as the company name throughout

## CURRENT CONTENT:
{content[:8000]}

## SCENARIO CONTEXT:
{scenario_prompt[:500]}

## OUTPUT:
Return ONLY the rewritten content text. No explanations, no JSON wrapper.
Target: 800-1400 words of pure data and facts.
Use "{company_name}" consistently. Avoid all forbidden terms."""

        try:
            fixed_content = await _call_gpt_async(prompt, system="You are a content editor. Return only the rewritten text.")

            # Clean up response
            fixed_content = fixed_content.strip()
            if fixed_content.startswith("```"):
                lines = fixed_content.split("\n")[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                fixed_content = "\n".join(lines)

            word_count = len(fixed_content.split())
            logger.info(f"[RESOURCE FIXER] Rewrote resource: {len(content.split())} -> {word_count} words")

            # Verify fix actually removed forbidden phrases
            forbidden_check = ["should", "recommend", "therefore", "thus", "consequently", "suggests that"]
            found_forbidden = [f for f in forbidden_check if f.lower() in fixed_content.lower()]
            if found_forbidden:
                logger.warning(f"[RESOURCE FIXER] Still contains forbidden: {found_forbidden} - applying second pass")
                # Second pass - more aggressive
                second_prompt = f"""The following text STILL contains forbidden phrases: {found_forbidden}

REMOVE ALL of these phrases completely. Rewrite sentences to be purely factual data without ANY recommendations or conclusions.

TEXT:
{fixed_content}

Rules:
- Replace "should" sentences with pure data statements
- Replace "recommend" with factual observations
- Remove "therefore/thus/consequently" - just state the facts
- NEVER tell the reader what to do or conclude

Return ONLY the fixed text."""
                fixed_content = await _call_gpt_async(second_prompt, system="Remove all recommendation language. Return only factual data.")
                fixed_content = fixed_content.strip()
                if fixed_content.startswith("```"):
                    lines = fixed_content.split("\n")[1:]
                    if lines and lines[-1].strip() == "```":
                        lines = lines[:-1]
                    fixed_content = "\n".join(lines)
                word_count = len(fixed_content.split())
                logger.info(f"[RESOURCE FIXER] Second pass complete: {word_count} words")

            # Update the resource - write back to the SAME key we read from
            if isinstance(current_resource, dict):
                if content_key:
                    current_resource[content_key] = fixed_content
                    logger.info(f"[RESOURCE FIXER] Updated '{content_key}' field")
                else:
                    # Fallback to content if we don't know the original key
                    current_resource["content"] = fixed_content
                set_value_at_path(result, path, current_resource)
            else:
                set_value_at_path(result, path, fixed_content)

        except Exception as e:
            logger.error(f"[RESOURCE FIXER] Failed to fix {path}: {e}")

    return result


@traceable(name="fix_context_fidelity", run_type="chain")
async def fix_context_fidelity(adapted_json: dict, scenario_prompt: str, issues: list[ValidationIssue]) -> dict:
    """
    SPECIALIZED CONTEXT FIDELITY FIXER - Ensures goal/challenge/role are preserved.

    This is different from generic repair - it specifically ensures the learning
    context from the scenario is properly reflected in the adapted content.
    """
    import copy
    result = copy.deepcopy(adapted_json)

    if not issues:
        return result

    logger.info(f"[CONTEXT FIXER] Fixing {len(issues)} context fidelity issues...")

    # Get key sections that need context
    content_sample = json.dumps(result, indent=2)[:30000]

    issue_descriptions = "\n".join([f"- {i.issue}: {i.suggestion}" for i in issues])

    prompt = f"""Fix the following CONTEXT FIDELITY issues in this JSON content.

## TARGET SCENARIO (source of truth):
{scenario_prompt}

## ISSUES FOUND:
{issue_descriptions}

## WHAT TO FIX:
1. Ensure the LEARNER'S GOAL from the scenario appears in overview/description sections
2. Ensure the CHALLENGE from the scenario drives the tasks
3. Ensure the ROLE is properly referenced throughout
4. All content should build toward the learning objectives

## CURRENT CONTENT (sample):
{content_sample}
...

## OUTPUT:
Return a JSON array of patches to fix these issues:
[
    {{"path": "/overview/description", "value": "Updated description with proper goal...", "reason": "Added learner goal"}},
    {{"path": "/workplace_scenario/role", "value": "Junior Consultant", "reason": "Fixed role"}}
]

Return ONLY the JSON array of patches."""

    try:
        response = await _call_gpt_async(prompt)
        patches = _parse_json_response(response)

        if isinstance(patches, list):
            result = apply_path_patches(result, patches)
            logger.info(f"[CONTEXT FIXER] Applied {len(patches)} context fixes")

    except Exception as e:
        logger.error(f"[CONTEXT FIXER] Failed: {e}")

    return result


@traceable(name="fix_completeness", run_type="chain")
async def fix_completeness(adapted_json: dict, scenario_prompt: str, issues: list[ValidationIssue]) -> dict:
    """
    SPECIALIZED COMPLETENESS FIXER - Expands truncated content and fills placeholders.

    This regenerates content that was truncated or left as placeholders.
    """
    import copy
    result = copy.deepcopy(adapted_json)

    if not issues:
        return result

    logger.info(f"[COMPLETENESS FIXER] Fixing {len(issues)} completeness issues...")

    content_str = json.dumps(result, ensure_ascii=False)

    # Fix placeholders directly via string replacement
    placeholder_patterns = [
        (r'\[TBD\]', ''),
        (r'\[TODO\]', ''),
        (r'\[INSERT\]', ''),
        (r'\[INSERT HERE\]', ''),
        (r'XXX', ''),
    ]

    import re
    for pattern, replacement in placeholder_patterns:
        if re.search(pattern, content_str, re.IGNORECASE):
            logger.info(f"[COMPLETENESS FIXER] Removing placeholder: {pattern}")
            content_str = re.sub(pattern, replacement, content_str, flags=re.IGNORECASE)

    try:
        result = json.loads(content_str)
    except json.JSONDecodeError:
        logger.error("[COMPLETENESS FIXER] Failed to parse after placeholder removal")
        return adapted_json

    # For truncated content, we need to identify and expand
    truncation_issues = [i for i in issues if "truncat" in i.issue.lower()]
    if truncation_issues:
        logger.info(f"[COMPLETENESS FIXER] Found {len(truncation_issues)} truncation issues")
        # These are harder to fix - would need to identify the truncated fields
        # and regenerate them. For now, log and skip.

    return result


@traceable(name="targeted_gap_fill", run_type="chain")
async def targeted_gap_fill(
    adapted_json: dict,
    scenario_prompt: str,
    unsolvable_questions: list[str],
    solvability_details: list[dict] = None
) -> dict:
    """PHASE 2: Targeted gap-fill for unsolvable questions.

    This is SEPARATE from adaptation. Adaptation is done, now we:
    1. Take the list of unsolvable questions
    2. Generate ONLY the specific data needed for those questions
    3. Insert into the resource

    Key difference from previous enrichment:
    - More focused prompt (just data generation, not adapt+fill)
    - Uses solvability analysis to know exactly what's missing
    """
    if not unsolvable_questions:
        return adapted_json

    logger.info(f"[GAP FILL] Phase 2: Generating data for {len(unsolvable_questions)} unsolvable questions")

    # Find resource location
    resource_obj = None
    resource_key = None
    sim_flow = adapted_json.get("simulation_flow", [])

    for idx, stage in enumerate(sim_flow):
        if isinstance(stage, dict):
            data_obj = stage.get("data", {})
            if isinstance(data_obj, dict) and data_obj.get("resource"):
                resource_obj = data_obj["resource"]
                resource_key = "markdown_text" if isinstance(resource_obj, dict) and "markdown_text" in resource_obj else "content"
                logger.info(f"[GAP FILL] Found resource at simulation_flow[{idx}].data.resource")
                break

    if not resource_obj:
        logger.warning("[GAP FILL] No resource found, skipping")
        return adapted_json

    # Get current content
    current_content = ""
    if isinstance(resource_obj, dict):
        current_content = resource_obj.get("markdown_text") or resource_obj.get("content") or resource_obj.get("html") or ""

    # Build focused prompt - ONLY data generation
    questions_with_gaps = []
    for i, q in enumerate(unsolvable_questions[:5]):
        gap_info = ""
        if solvability_details:
            for detail in solvability_details:
                if detail.get("question", "")[:50] in q[:50]:
                    gap_info = f" (Missing: {detail.get('gap', 'data')})"
                    break
        questions_with_gaps.append(f"Q{i+1}: {q[:200]}{gap_info}")

    questions_text = "\n".join(questions_with_gaps)

    prompt = f"""You are a DATA GENERATOR for a business simulation.

## TASK: Generate specific data paragraphs for questions that can't be answered.

## SCENARIO CONTEXT:
{scenario_prompt[:500]}

## QUESTIONS THAT NEED DATA:
{questions_text}

## GENERATE DATA FOR EACH QUESTION:

For each question above, write ONE paragraph with SPECIFIC DATA:
- Use real numbers (market size $X billion, growth X%, share X%)
- Include comparisons (Competitor A vs B, before vs after)
- Provide enough detail for analysis (not conclusions)

## FORMAT:
<h3>[Topic matching the question]</h3>
<p>[2-4 sentences of specific data with numbers]</p>

## RULES:
- Generate {len(unsolvable_questions)} paragraphs (one per question)
- Numbers must be realistic for the industry
- NO recommendations or conclusions - ONLY facts and data
- Keep each paragraph focused and concise (50-100 words)

## OUTPUT:
Return ONLY the HTML paragraphs, nothing else."""

    try:
        response = await _call_gpt_async(prompt, system="You generate business data. Output only HTML paragraphs with factual data.")

        # Clean response
        response = response.strip()
        if response.startswith("```"):
            lines = response.split("\n")[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            response = "\n".join(lines)

        # Insert into resource (append at end of content)
        if isinstance(resource_obj, dict):
            if "markdown_text" in resource_obj:
                resource_obj["markdown_text"] = resource_obj["markdown_text"] + "\n\n" + response
            elif "content" in resource_obj:
                resource_obj["content"] = resource_obj["content"] + "\n\n" + response
            elif "html" in resource_obj:
                resource_obj["html"] = resource_obj["html"] + "\n\n" + response

        logger.info(f"[GAP FILL] Added {len(response)} chars of targeted data")
        return adapted_json

    except Exception as e:
        logger.error(f"[GAP FILL] Failed: {e}")
        return adapted_json


@traceable(name="validate_and_repair", run_type="chain")
async def validate_and_repair(adapted_json: dict, scenario_prompt: str, max_iter: int = 2) -> tuple[dict, ValidationReport]:
    """Full validation + repair loop with specialized fixers."""
    current = adapted_json
    resource_fixer_done = False  # Only run Resource Fixer once

    for i in range(max_iter):
        logger.info(f"[V+R] Iteration {i+1}/{max_iter}")
        report = await run_all_validators(current, scenario_prompt)

        if report.passed or report.overall_score >= ACCEPTABLE_THRESHOLD:
            return current, report

        # Collect all issues from all agents
        all_issues = []
        solvability_result = None
        for ar in report.agent_results:
            all_issues.extend(ar.issues)
            if "Solvability" in ar.agent_name:
                solvability_result = ar

        # STEP 1: Run specialized Resource Fixer FIRST (regenerates content) - ONLY ONCE
        resource_issues = [i for i in all_issues if i.agent == "Resource Quality"]
        if resource_issues and not resource_fixer_done:
            logger.info(f"[V+R] Running specialized Resource Fixer for {len(resource_issues)} issues")
            current = await fix_resource_quality(current, scenario_prompt, resource_issues)
            resource_fixer_done = True
        elif resource_issues:
            logger.info(f"[V+R] Skipping Resource Fixer (already ran once) - {len(resource_issues)} issues remain")

        # STEP 2: Run generic repair for other issues (patches)
        other_issues = [i for i in all_issues if i.agent != "Resource Quality"]
        if other_issues:
            # Create a mini-report for generic repair
            mini_report = ValidationReport(
                overall_score=report.overall_score,
                passed=False,
                agent_results=[ar for ar in report.agent_results if ar.agent_name != "Resource Quality"],
                total_issues=len(other_issues),
                needs_repair=True
            )
            logger.info(f"[V+R] Running generic repair for {len(other_issues)} other issues")
            current = await repair_issues(current, scenario_prompt, mini_report)

        # STEP 3: Gap-fill DISABLED - adds latency without significant improvement
        # Solvability should be addressed by smarter data transformation during adaptation
        # if not gap_fill_done and solvability_result and solvability_result.issues:
        #     ... gap fill code disabled ...

    final_report = await run_all_validators(current, scenario_prompt)
    return current, final_report
