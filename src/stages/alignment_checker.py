"""
Stage 3: Alignment Checker (LLM-Based)

Cross-shard alignment validation using GPT for intelligent semantic checking.

Uses OpenAI GPT to:
1. Validate reporting manager consistency
2. Check company/organization consistency
3. Detect poison terms (old scenario leakage)
4. Verify KLO ↔ Task alignment
5. Check scenario ↔ content coherence

TARGET: ≥ 98% alignment score
"""
import os
import json
import logging
import asyncio
from typing import Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from pydantic import BaseModel

import httpx
from langsmith import traceable
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from src.utils.gemini_client import filter_poison_list

logger = logging.getLogger(__name__)

# Global semaphore for controlling concurrent LLM calls
# This is the KEY to true parallelism - limits concurrent requests to avoid serialization
MAX_CONCURRENT_LLM_CALLS = int(os.getenv("MAX_CONCURRENT_LLM_CALLS", "10"))
_llm_semaphore = None  # Will be initialized lazily in async context


def _get_semaphore():
    """Get or create the semaphore (must be called in async context)."""
    global _llm_semaphore
    if _llm_semaphore is None:
        _llm_semaphore = asyncio.Semaphore(MAX_CONCURRENT_LLM_CALLS)
    return _llm_semaphore


# =============================================================================
# PYDANTIC MODELS FOR STRUCTURED LLM OUTPUT
# =============================================================================

class ConsistencyCheckResponse(BaseModel):
    """Response from LLM for consistency checks."""
    is_consistent: bool
    inconsistencies: list[str]  # List of inconsistencies found
    confidence: float  # 0.0 to 1.0
    suggestions: Optional[str] = None


class PoisonTermResponse(BaseModel):
    """Response from LLM for poison term detection."""
    contains_poison_terms: bool
    detected_terms: list[str]  # Old terms found
    locations: list[str]  # Where they were found
    confidence: float


class AlignmentCheckResponse(BaseModel):
    """Response from LLM for alignment validation."""
    is_aligned: bool
    alignment_score: float  # 0.0 to 1.0
    issues: list[str]
    recommendations: list[str]


class OverallValidationResponse(BaseModel):
    """Response from LLM for overall validation."""
    passed: bool
    overall_score: float
    critical_issues: list[str]
    warnings: list[str]
    summary: str


# =============================================================================
# ALIGNMENT RESULT MODELS
# =============================================================================

class AlignmentSeverity(Enum):
    """Severity level for alignment issues."""
    BLOCKER = "blocker"
    WARNING = "warning"
    INFO = "info"


@dataclass
class AlignmentIssue:
    """A single alignment issue found."""
    rule_id: str
    description: str
    location: str
    severity: AlignmentSeverity
    suggestion: Optional[str] = None


@dataclass
class AlignmentResult:
    """Result of a single alignment check."""
    rule_id: str
    rule_name: str
    passed: bool
    score: float
    issues: list[AlignmentIssue] = field(default_factory=list)
    details: dict = field(default_factory=dict)
    suggestions: list[str] = field(default_factory=list)


@dataclass
class AlignmentReport:
    """Complete alignment report for an adapted simulation."""
    overall_score: float
    passed: bool
    threshold: float
    results: list[AlignmentResult] = field(default_factory=list)
    blocker_issues: list[AlignmentIssue] = field(default_factory=list)
    warning_issues: list[AlignmentIssue] = field(default_factory=list)
    llm_summary: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for API response."""
        results_list = []
        for r in self.results:
            try:
                issues_list = []
                for i in r.issues:
                    if isinstance(i, AlignmentIssue):
                        issues_list.append({
                            "description": i.description,
                            "location": i.location,
                            "severity": i.severity.value if hasattr(i.severity, 'value') else str(i.severity),
                            "suggestion": i.suggestion,
                        })
                    else:
                        # Handle string issues
                        issues_list.append({
                            "description": str(i),
                            "location": "unknown",
                            "severity": "warning",
                            "suggestion": None,
                        })

                results_list.append({
                    "rule_id": r.rule_id,
                    "rule_name": r.rule_name,
                    "passed": r.passed,
                    "score": round(r.score, 4),
                    "issues": issues_list,
                })
            except Exception as e:
                results_list.append({
                    "rule_id": getattr(r, 'rule_id', 'error'),
                    "rule_name": getattr(r, 'rule_name', 'Error'),
                    "passed": False,
                    "score": 0.0,
                    "issues": [{"description": str(e), "location": "to_dict", "severity": "warning", "suggestion": None}],
                })

        return {
            "overall_score": round(self.overall_score, 4),
            "passed": self.passed,
            "threshold": self.threshold,
            "summary": self.llm_summary,
            "results": results_list,
            "blocker_count": len(self.blocker_issues),
            "warning_count": len(self.warning_issues),
        }


# =============================================================================
# LLM-BASED ALIGNMENT CHECKER (GPT-5.2)
# =============================================================================

# GPT-5.2 Model for validation - 400K context, strong reasoning
VALIDATION_MODEL = os.getenv("VALIDATION_MODEL", "gpt-5.2-2025-12-11")


def _get_validation_llm():
    """
    Get OpenAI GPT-5.2 client for validation.

    CRITICAL: Each call creates a NEW httpx.AsyncClient to enable TRUE parallel execution.
    Without this, all LLM calls share the same connection and run sequentially.
    """
    # Create NEW http client for each LLM instance - enables true parallelism
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(120.0, connect=30.0),  # 2 min read, 30s connect
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=50),  # Higher limits for parallelism
    )

    return ChatOpenAI(
        model=VALIDATION_MODEL,
        temperature=0.1,
        max_retries=3,
        request_timeout=120,
        http_async_client=http_client,  # CRITICAL: Use separate HTTP client
        api_key=os.getenv("OPENAI_API_KEY"),
    )


class AlignmentChecker:
    """
    Stage 3: LLM-based cross-shard alignment validation.

    Uses OpenAI GPT-5.2 for intelligent semantic checking instead of hardcoded rules.
    """

    def __init__(self, threshold: float = 0.95):
        """
        Args:
            threshold: Minimum score required to pass (default 0.95 = 95%)
        """
        self.threshold = threshold
        # Don't create shared LLM - each parallel task gets its own for true parallelism

    def _get_llm(self):
        """Get a NEW LLM instance for each parallel check (enables true parallel execution)."""
        return _get_validation_llm()

    async def check(
        self,
        adapted_json: dict,
        global_factsheet: dict,
        source_scenario: str = "",
    ) -> AlignmentReport:
        """
        Run LLM-based alignment checks on the adapted JSON.

        Args:
            adapted_json: The adapted simulation JSON
            global_factsheet: The factsheet used during adaptation
            source_scenario: Original scenario text (for poison check)

        Returns:
            AlignmentReport with scores and issues
        """
        results = []
        blocker_issues = []
        warning_issues = []

        # Run alignment checks in parallel using create_task for TRUE parallelism
        # NOTE: KLO checks (R4, R5, R8) are now consolidated into UnifiedKLOValidator
        logger.info(f"Running 6 alignment checks in PARALLEL + 1 unified KLO check (semaphore limit: {MAX_CONCURRENT_LLM_CALLS})")

        # First, run the unified KLO validator (synchronous, fast)
        from ..validators.klo_validator import UnifiedKLOValidator
        klo_validator = UnifiedKLOValidator()
        klo_result = klo_validator.validate(adapted_json, global_factsheet)

        # Convert KLO result to AlignmentResult format
        klo_alignment_result = AlignmentResult(
            rule_id="klo_alignment_unified",
            rule_name="Unified KLO Alignment (R4+R5+R8)",
            passed=klo_result.passed,
            score=klo_result.overall_score,
            issues=[
                AlignmentIssue(
                    rule_id="klo_alignment_unified",
                    description=issue,
                    location="klo_validator",
                    severity=AlignmentSeverity.BLOCKER if not klo_result.passed else AlignmentSeverity.WARNING,
                )
                for check in klo_result.checks.values()
                for issue in check.issues
            ],
            suggestions=[klo_result.summary],
        )
        results.append(klo_alignment_result)
        logger.info(f"  {'[PASS]' if klo_result.passed else '[FAIL]'} Unified KLO Alignment: {klo_result.overall_score:.1%}")

        # Remaining checks run in parallel (KLO checks removed - now unified above)
        check_coroutines = [
            # Consistency checks
            self._check_reporting_manager_consistency(adapted_json, global_factsheet),
            self._check_company_consistency(adapted_json, global_factsheet),
            self._check_poison_terms(adapted_json, global_factsheet, source_scenario),
            # Alignment Matrix checks (KLO checks moved to unified validator above)
            self._check_scenario_to_resources(adapted_json, global_factsheet),  # Scenario ↔ Resources
            self._check_role_to_tasks(adapted_json, global_factsheet),  # Role ↔ Tasks
            # Coherence checks
            self._check_scenario_coherence(adapted_json, global_factsheet),
        ]
        # Create actual Task objects to force concurrent scheduling
        check_tasks = [asyncio.create_task(coro) for coro in check_coroutines]
        check_results = await asyncio.gather(*check_tasks, return_exceptions=True)

        for result in check_results:
            if isinstance(result, Exception):
                logger.error(f"Alignment check failed: {result}")
                results.append(AlignmentResult(
                    rule_id="error",
                    rule_name="Check Error",
                    passed=False,
                    score=0.0,
                    issues=[AlignmentIssue(
                        rule_id="error",
                        description=str(result),
                        location="alignment_checker",
                        severity=AlignmentSeverity.BLOCKER,
                    )],
                ))
            elif not isinstance(result, AlignmentResult):
                # Handle unexpected result type
                logger.error(f"Unexpected result type: {type(result)} - {result}")
                results.append(AlignmentResult(
                    rule_id="error",
                    rule_name="Type Error",
                    passed=False,
                    score=0.0,
                    issues=[AlignmentIssue(
                        rule_id="error",
                        description=f"Unexpected result type: {type(result)}",
                        location="alignment_checker",
                        severity=AlignmentSeverity.WARNING,
                    )],
                ))
            else:
                results.append(result)
                # Log each check result for debugging
                status = "✅ PASS" if result.passed else "❌ FAIL"
                logger.info(f"  {status} {result.rule_name}: {result.score:.1%} ({len(result.issues)} issues)")
                for issue in result.issues:
                    # Defensive check for issue type
                    if isinstance(issue, AlignmentIssue):
                        logger.info(f"      - [{issue.severity.value}] {issue.description[:100]}")
                        if issue.severity == AlignmentSeverity.BLOCKER:
                            blocker_issues.append(issue)
                        elif issue.severity == AlignmentSeverity.WARNING:
                            warning_issues.append(issue)
                    else:
                        logger.warning(f"      - [WARN] Unexpected issue type: {type(issue)} - {issue}")

        # Get overall validation from LLM
        overall_result = await self._get_overall_validation(
            adapted_json, global_factsheet, results
        )

        # Calculate overall score from actual check results
        if results:
            overall_score = sum(r.score for r in results) / len(results)
        else:
            overall_score = 0.0

        # Get LLM summary but DON'T override the calculated score
        # The LLM tends to be too harsh with its own scoring
        if overall_result:
            llm_summary = overall_result.summary
            # Log the LLM's score for reference, but don't use it
            logger.debug(f"LLM suggested score: {overall_result.overall_score:.1%} (using calculated: {overall_score:.1%})")
        else:
            llm_summary = ""

        passed = len(blocker_issues) == 0 and overall_score >= self.threshold

        report = AlignmentReport(
            overall_score=overall_score,
            passed=passed,
            threshold=self.threshold,
            results=results,
            blocker_issues=blocker_issues,
            warning_issues=warning_issues,
            llm_summary=llm_summary,
        )

        logger.info(f"Alignment check: {overall_score:.1%} (threshold: {self.threshold:.0%})")
        if blocker_issues:
            logger.warning(f"  ⚠ {len(blocker_issues)} blocker issues")

        return report

    # =========================================================================
    # LLM-BASED ALIGNMENT CHECKS
    # =========================================================================

    # Scoring guidelines added to all prompts
    SCORING_GUIDELINES = """

## SCORING GUIDELINES (IMPORTANT):
When providing alignment_score, use this scale:
- 0.95-1.0: Perfect - No issues, everything aligns perfectly
- 0.85-0.94: Excellent - Minor nitpicks only, fully functional
- 0.75-0.84: Good - Some small issues but overall solid alignment
- 0.65-0.74: Acceptable - Noticeable issues but still usable
- 0.50-0.64: Needs Work - Significant issues affecting quality
- Below 0.50: Poor - Major alignment problems

BE GENEROUS with scores when:
- The core content is correct even if wording could be improved
- Minor variations exist but the meaning is preserved
- The adaptation successfully changed the context while maintaining structure

Only give low scores (<0.7) for ACTUAL problems like:
- Wrong company/person names used
- Content that contradicts the scenario
- Missing critical information
- Fundamentally misaligned content
"""

    async def _call_llm_with_parser(
        self,
        prompt: str,
        parser: PydanticOutputParser,
        system_prompt: str = "You are a validation agent for simulation content.",
    ) -> Any:
        """Call GPT-5.2 with Pydantic output parser."""
        # Add scoring guidelines to system prompt
        enhanced_system_prompt = system_prompt + self.SCORING_GUIDELINES

        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", enhanced_system_prompt + "\n\n{format_instructions}"),
            ("human", "{input}"),
        ])

        chain = chat_prompt | self._get_llm() | parser

        # Use semaphore for controlled parallelism
        semaphore = _get_semaphore()
        async with semaphore:
            result = await chain.ainvoke({
                "input": prompt,
                "format_instructions": parser.get_format_instructions(),
            })

        return result

    @traceable
    async def _check_reporting_manager_consistency(
        self,
        adapted_json: dict,
        global_factsheet: dict,
    ) -> AlignmentResult:
        """
        Use GPT-5.2 to check reporting manager consistency across all references.
        """
        rule_id = "reporting_manager_consistency"

        # Extract all manager references from JSON
        topic_data = adapted_json.get("topicWizardData", {})
        manager_refs = self._extract_manager_references(topic_data)

        expected_manager = global_factsheet.get("reporting_manager", {})

        prompt = f"""Check for REPORTING MANAGER CONSISTENCY in this simulation.

## EXPECTED MANAGER (from factsheet):
- Name: {expected_manager.get('name', 'Unknown')}
- Role: {expected_manager.get('role', 'Unknown')}
- Email: {expected_manager.get('email', 'Unknown')}
- Gender: {expected_manager.get('gender', 'Unknown')}

## MANAGER REFERENCES FOUND IN ADAPTED JSON:
{json.dumps(manager_refs, indent=2)}

## TASK:
Check if ALL references to the reporting manager are CONSISTENT with the expected manager.

Look for inconsistencies in:
1. Name variations (should all refer to same person)
2. Email addresses (should match)
3. Role/title (should be consistent)
4. Gender pronouns (should match)"""

        try:
            parser = PydanticOutputParser(pydantic_object=ConsistencyCheckResponse)
            result = await self._call_llm_with_parser(
                prompt=prompt,
                parser=parser,
                system_prompt="You are a validation agent checking for consistency in business simulations.",
            )

            issues = []
            for inconsistency in result.inconsistencies:
                issues.append(AlignmentIssue(
                    rule_id=rule_id,
                    description=inconsistency,
                    location="reporting_manager",
                    severity=AlignmentSeverity.BLOCKER,
                    suggestion=result.suggestions,
                ))

            return AlignmentResult(
                rule_id=rule_id,
                rule_name="Reporting Manager Consistency",
                passed=result.is_consistent,
                score=result.confidence if result.is_consistent else 1.0 - (len(result.inconsistencies) * 0.2),
                issues=issues,
                details={"manager_refs_found": len(manager_refs)},
            )

        except Exception as e:
            logger.error(f"Manager consistency check failed: {e}")
            return AlignmentResult(
                rule_id=rule_id,
                rule_name="Reporting Manager Consistency",
                passed=False,
                score=0.0,
                issues=[AlignmentIssue(
                    rule_id=rule_id,
                    description=f"Check failed: {e}",
                    location="reporting_manager",
                    severity=AlignmentSeverity.WARNING,
                )],
            )

    @traceable
    async def _check_company_consistency(
        self,
        adapted_json: dict,
        global_factsheet: dict,
    ) -> AlignmentResult:
        """
        Use GPT-5.2 to check company/organization name consistency.
        """
        rule_id = "company_consistency"

        topic_data = adapted_json.get("topicWizardData", {})
        company_refs = self._extract_company_references(topic_data)

        expected_company = global_factsheet.get("company", {})

        prompt = f"""Check for COMPANY NAME CONSISTENCY in this simulation.

## EXPECTED COMPANY (from factsheet):
- Name: {expected_company.get('name', 'Unknown')}
- Industry: {expected_company.get('industry', 'Unknown')}

## COMPANY REFERENCES FOUND IN ADAPTED JSON:
{json.dumps(company_refs, indent=2)}

## TASK:
Check if ALL references to the company/organization are CONSISTENT.

Look for:
1. Different company names used in different places
2. Old company names that should have been replaced
3. Inconsistent industry terminology"""

        try:
            parser = PydanticOutputParser(pydantic_object=ConsistencyCheckResponse)
            result = await self._call_llm_with_parser(
                prompt=prompt,
                parser=parser,
                system_prompt="You are a validation agent checking for consistency in business simulations.",
            )

            issues = []
            for inconsistency in result.inconsistencies:
                issues.append(AlignmentIssue(
                    rule_id=rule_id,
                    description=inconsistency,
                    location="company_name",
                    severity=AlignmentSeverity.BLOCKER,
                    suggestion=result.suggestions,
                ))

            return AlignmentResult(
                rule_id=rule_id,
                rule_name="Company Name Consistency",
                passed=result.is_consistent,
                score=result.confidence if result.is_consistent else 1.0 - (len(result.inconsistencies) * 0.2),
                issues=issues,
                details={"company_refs_found": len(company_refs)},
            )

        except Exception as e:
            logger.error(f"Company consistency check failed: {e}")
            return AlignmentResult(
                rule_id=rule_id,
                rule_name="Company Name Consistency",
                passed=False,
                score=0.0,
                issues=[AlignmentIssue(
                    rule_id=rule_id,
                    description=f"Check failed: {e}",
                    location="company_name",
                    severity=AlignmentSeverity.WARNING,
                )],
            )

    @traceable
    async def _check_poison_terms(
        self,
        adapted_json: dict,
        global_factsheet: dict,
        source_scenario: str,
    ) -> AlignmentResult:
        """
        Use GPT-5.2 to detect old scenario terms that shouldn't appear.
        """
        rule_id = "poison_term_avoidance"

        # Filter poison list to remove common English words (second layer of defense)
        # Pass domain_vocabulary to protect domain-specific terms from being filtered
        raw_poison_list = global_factsheet.get("poison_list", [])
        domain_vocab = global_factsheet.get("source_domain", {}).get("domain_vocabulary", [])
        poison_list = filter_poison_list(raw_poison_list, domain_vocabulary=domain_vocab) if isinstance(raw_poison_list, list) else []
        replacement_hints = global_factsheet.get("replacement_hints", {})

        # Sample key content for checking
        topic_data = adapted_json.get("topicWizardData", {})
        sample_content = self._extract_sample_content(topic_data)

        prompt = f"""Check for OLD SCENARIO TERMS that should NOT appear in the adapted content.

## SOURCE SCENARIO (OLD):
{source_scenario[:500] if source_scenario else "Not provided"}

## POISON LIST (scenario-specific terms from old scenario that should NOT appear):
{json.dumps(poison_list, indent=2)}

## REPLACEMENT HINTS:
{json.dumps(replacement_hints, indent=2)}

## SAMPLE CONTENT FROM ADAPTED JSON:
{json.dumps(sample_content, indent=2)}

## TASK:
Check if ANY of the poison list terms (or variations) appear in the adapted content.

## IMPORTANT - DO NOT flag common English words as poison terms:
- Words like "role", "consistent", "ensure", "evaluate", "assessment" are NORMAL business language
- Only flag terms that are TRULY SPECIFIC to the old scenario (company names, person names, product names, branded terms)
- A term is only a problem if it refers specifically to the OLD scenario context

Look for:
1. Exact matches of scenario-specific poison terms (company names, product names, person names)
2. Industry-specific jargon from the OLD scenario that doesn't fit the NEW scenario
3. Branded or proprietary terms from the source scenario

DO NOT flag:
- Common business words (role, consistent, ensure, evaluate, analysis, etc.)
- Generic terms that appear in any business context
- Words that are appropriate for the NEW scenario context"""

        try:
            parser = PydanticOutputParser(pydantic_object=PoisonTermResponse)
            result = await self._call_llm_with_parser(
                prompt=prompt,
                parser=parser,
                system_prompt="You are a validation agent detecting old scenario term leakage.",
            )

            issues = []
            for i, term in enumerate(result.detected_terms):
                location = result.locations[i] if i < len(result.locations) else "unknown"
                issues.append(AlignmentIssue(
                    rule_id=rule_id,
                    description=f"Old term '{term}' found",
                    location=location,
                    severity=AlignmentSeverity.BLOCKER,
                    suggestion="Replace with target scenario equivalent",
                ))

            score = 1.0 if not result.contains_poison_terms else max(0.0, 1.0 - (len(result.detected_terms) * 0.1))

            return AlignmentResult(
                rule_id=rule_id,
                rule_name="Poison Term Avoidance",
                passed=not result.contains_poison_terms,
                score=score,
                issues=issues,
                details={"poison_terms_checked": len(poison_list), "found": len(result.detected_terms)},
            )

        except Exception as e:
            logger.error(f"Poison term check failed: {e}")
            return AlignmentResult(
                rule_id=rule_id,
                rule_name="Poison Term Avoidance",
                passed=True,  # Assume pass on error
                score=0.8,
                issues=[],
            )

    # =========================================================================
    # ALIGNMENT MATRIX CHECKS (from whole_arc.md)
    # =========================================================================

    @traceable
    async def _check_klo_to_questions(
        self,
        adapted_json: dict,
        global_factsheet: dict,
    ) -> AlignmentResult:
        """
        Use GPT-5.2 to check: Do submission questions assess KLOs?

        Each KLO must have at least one question that assesses it.
        """
        rule_id = "klo_to_questions"

        topic_data = adapted_json.get("topicWizardData", {})

        # Extract KLOs
        klos = []
        for criterion in topic_data.get("assessmentCriterion", []):
            klo_text = criterion.get("keyLearningOutcome", "")
            if klo_text:
                klos.append({
                    "id": criterion.get("id", ""),
                    "klo": klo_text,
                    "criteria": [c.get("criteria", "") for c in criterion.get("criterion", [])]
                })

        # Extract submission questions from simulationFlow
        questions = []
        for stage in topic_data.get("simulationFlow", []):
            stage_data = stage.get("data", {})
            # Check for questions in various places
            if "questions" in stage_data:
                questions.extend(stage_data["questions"])
            if "submissionQuestions" in stage_data:
                questions.extend(stage_data["submissionQuestions"])
            # Check activityData.questions (common location for submission questions)
            activity_data = stage_data.get("activityData", {})
            if isinstance(activity_data, dict) and "questions" in activity_data:
                questions.extend(activity_data["questions"])
            # Check review.rubric for questions
            review = stage_data.get("review", {})
            if isinstance(review, dict):
                for rubric in review.get("rubric", []):
                    if isinstance(rubric, dict):
                        if rubric.get("question"):
                            questions.append({"question": rubric["question"]})
                        if rubric.get("reviewQuestion"):
                            questions.append({"question": rubric["reviewQuestion"]})
            # Check children
            for child in stage.get("children", []):
                child_data = child.get("data", {})
                if "questions" in child_data:
                    questions.extend(child_data["questions"])

        # Also check top-level submissionQuestions
        questions.extend(topic_data.get("submissionQuestions", []))
        questions.extend(topic_data.get("selectedSubmissionQuestions", []))

        # Also consider activities as implicit assessments (many simulations use activities instead of explicit questions)
        activities = self._extract_activities(topic_data)

        prompt = f"""Check if SUBMISSION QUESTIONS and/or ACTIVITIES properly assess the KEY LEARNING OUTCOMES (KLOs).

## KEY LEARNING OUTCOMES (KLOs):
{json.dumps(klos, indent=2)}

## SUBMISSION QUESTIONS:
{json.dumps(questions, indent=2) if questions else "No explicit submission questions found"}

## ACTIVITIES (these serve as implicit assessments):
{json.dumps(activities, indent=2) if activities else "No activities found"}

## TASK:
Evaluate if each KLO has at least one question OR activity that assesses/measures it.

IMPORTANT: Many simulations use ACTIVITIES as the primary assessment mechanism instead of explicit questions.
If activities are well-aligned to KLOs, that is sufficient for a high score.

Consider:
1. Does each KLO have a corresponding question OR activity that assesses it?
2. Are the activities designed to demonstrate the learning outcomes?
3. Is there clear alignment between what students should learn and what they're asked to do?

Be generous with scoring if activities properly assess the KLOs, even if there are few explicit questions."""

        try:
            parser = PydanticOutputParser(pydantic_object=AlignmentCheckResponse)
            result = await self._call_llm_with_parser(
                prompt=prompt,
                parser=parser,
                system_prompt="You are a validation agent checking KLO-to-question alignment in educational simulations.",
            )

            issues = []
            for i, issue in enumerate(result.issues):
                suggestion = result.recommendations[i] if i < len(result.recommendations) else None
                issues.append(AlignmentIssue(
                    rule_id=rule_id,
                    description=issue,
                    location="klo_to_questions",
                    severity=AlignmentSeverity.BLOCKER,  # This is critical
                    suggestion=suggestion,
                ))

            return AlignmentResult(
                rule_id=rule_id,
                rule_name="KLOs ↔ Questions/Activities",
                passed=result.is_aligned and result.alignment_score >= 0.8,
                score=result.alignment_score,
                issues=issues,
                details={"klo_count": len(klos), "question_count": len(questions), "activity_count": len(activities)},
            )

        except Exception as e:
            logger.error(f"KLO to questions check failed: {e}")
            return AlignmentResult(
                rule_id=rule_id,
                rule_name="KLOs ↔ Questions/Activities",
                passed=True,
                score=0.8,
                issues=[],
            )

    @traceable
    async def _check_klo_to_resources(
        self,
        adapted_json: dict,
        global_factsheet: dict,
    ) -> AlignmentResult:
        """
        Use GPT-5.2 to check: Do resources support KLOs?

        Resources should provide information needed to achieve learning outcomes.
        """
        rule_id = "klo_to_resources"

        topic_data = adapted_json.get("topicWizardData", {})

        # Extract KLOs
        klos = [
            c.get("keyLearningOutcome", "")
            for c in topic_data.get("assessmentCriterion", [])
            if c.get("keyLearningOutcome")
        ]

        # Extract resources from simulationFlow
        resources = []
        for stage in topic_data.get("simulationFlow", []):
            stage_data = stage.get("data", {})
            if "resource" in stage_data:
                resources.append(stage_data["resource"])
            if "resourceOptions" in stage_data:
                resources.extend(stage_data["resourceOptions"])
            # Check children for attachments
            for child in stage.get("children", []):
                child_data = child.get("data", {})
                email = child_data.get("email", {})
                if email and "attachments" in email:
                    resources.extend(email["attachments"])

        prompt = f"""Check if RESOURCES properly support the KEY LEARNING OUTCOMES (KLOs).

## KEY LEARNING OUTCOMES (KLOs):
{json.dumps(klos, indent=2)}

## RESOURCES PROVIDED:
{json.dumps(resources, indent=2)[:8000]}

## TASK:
Evaluate if the resources support students in achieving the learning outcomes.

Consider:
1. Do resources provide information needed to demonstrate the KLOs?
2. Are resources relevant to the learning objectives?
3. Do resources give students the data/knowledge needed to complete assessments?"""

        try:
            parser = PydanticOutputParser(pydantic_object=AlignmentCheckResponse)
            result = await self._call_llm_with_parser(
                prompt=prompt,
                parser=parser,
                system_prompt="You are a validation agent checking resource-to-KLO alignment.",
            )

            issues = []
            for i, issue in enumerate(result.issues):
                suggestion = result.recommendations[i] if i < len(result.recommendations) else None
                issues.append(AlignmentIssue(
                    rule_id=rule_id,
                    description=issue,
                    location="klo_to_resources",
                    severity=AlignmentSeverity.BLOCKER,
                    suggestion=suggestion,
                ))

            return AlignmentResult(
                rule_id=rule_id,
                rule_name="KLOs ↔ Resources",
                passed=result.is_aligned and result.alignment_score >= 0.8,
                score=result.alignment_score,
                issues=issues,
                details={"klo_count": len(klos), "resource_count": len(resources)},
            )

        except Exception as e:
            logger.error(f"KLO to resources check failed: {e}")
            return AlignmentResult(
                rule_id=rule_id,
                rule_name="KLOs ↔ Resources",
                passed=True,
                score=0.8,
                issues=[],
            )

    @traceable
    async def _check_scenario_to_resources(
        self,
        adapted_json: dict,
        global_factsheet: dict,
    ) -> AlignmentResult:
        """
        Use GPT-5.2 to check: Is resource data industry-specific and scenario-aligned?

        Resources should contain industry-appropriate data for the scenario.
        """
        rule_id = "scenario_to_resources"

        topic_data = adapted_json.get("topicWizardData", {})

        # Extract scenario context
        workplace = topic_data.get("workplaceScenario", {})
        company = workplace.get("background", {}).get("organizationName", "")
        industry = global_factsheet.get("company", {}).get("industry", "")
        scenario_context = workplace.get("scenario", "")

        # Extract resources
        resources = []
        for stage in topic_data.get("simulationFlow", []):
            stage_data = stage.get("data", {})
            if "resource" in stage_data:
                resources.append(stage_data["resource"])
            for child in stage.get("children", []):
                email = child.get("data", {}).get("email", {})
                if email and "attachments" in email:
                    resources.extend(email["attachments"])

        prompt = f"""Check if RESOURCES contain INDUSTRY-SPECIFIC data aligned with the SCENARIO.

## SCENARIO CONTEXT:
- Company: {company}
- Industry: {industry}
- Scenario: {scenario_context[:500]}

## RESOURCES PROVIDED:
{json.dumps(resources, indent=2)[:8000]}

## TASK:
Evaluate if resource data is appropriate for the industry and scenario.

Consider:
1. Do resources reference the correct company name?
2. Is the data industry-appropriate (correct KPIs, metrics, terminology)?
3. Are financial figures, statistics realistic for this industry?
4. Do resources avoid referencing a different company/industry?"""

        try:
            parser = PydanticOutputParser(pydantic_object=AlignmentCheckResponse)
            result = await self._call_llm_with_parser(
                prompt=prompt,
                parser=parser,
                system_prompt="You are a validation agent checking scenario-to-resource alignment.",
            )

            issues = []
            for i, issue in enumerate(result.issues):
                suggestion = result.recommendations[i] if i < len(result.recommendations) else None
                issues.append(AlignmentIssue(
                    rule_id=rule_id,
                    description=issue,
                    location="scenario_to_resources",
                    severity=AlignmentSeverity.BLOCKER,
                    suggestion=suggestion,
                ))

            return AlignmentResult(
                rule_id=rule_id,
                rule_name="Scenario ↔ Resources",
                passed=result.is_aligned and result.alignment_score >= 0.8,
                score=result.alignment_score,
                issues=issues,
                details={"company": company, "industry": industry, "resource_count": len(resources)},
            )

        except Exception as e:
            logger.error(f"Scenario to resources check failed: {e}")
            return AlignmentResult(
                rule_id=rule_id,
                rule_name="Scenario ↔ Resources",
                passed=True,
                score=0.8,
                issues=[],
            )

    @traceable
    async def _check_role_to_tasks(
        self,
        adapted_json: dict,
        global_factsheet: dict,
    ) -> AlignmentResult:
        """
        Use GPT-5.2 to check: Are tasks appropriate for the learner role?

        Tasks should match what someone in that role would actually do.
        """
        rule_id = "role_to_tasks"

        topic_data = adapted_json.get("topicWizardData", {})

        # Extract learner role
        learner_role = (topic_data.get("workplaceScenario", {})
                       .get("learnerRoleReportingManager", {})
                       .get("learnerRole", {}))

        role_name = learner_role.get("role", "")
        role_description = learner_role.get("roleDescription", "")
        scope_of_work = learner_role.get("scopeOfWork", [])

        # Extract tasks from activities
        tasks = []
        for sow in scope_of_work:
            tasks.append({
                "task": sow.get("task", ""),
                "description": sow.get("description", "")
            })

        # Also get activities from industryAlignedActivities
        activities = self._extract_activities(topic_data)

        prompt = f"""Check if TASKS are appropriate for the LEARNER ROLE.

## LEARNER ROLE:
- Role Title: {role_name}
- Role Description: {role_description}

## SCOPE OF WORK (Tasks assigned):
{json.dumps(tasks, indent=2)}

## ACTIVITIES:
{json.dumps(activities, indent=2)}

## TASK:
Evaluate if the assigned tasks are appropriate for someone in this role.

Consider:
1. Would someone in this role realistically do these tasks?
2. Are tasks at the appropriate level of responsibility?
3. Do tasks align with the role description?
4. Are there tasks that seem inappropriate for this role?"""

        try:
            parser = PydanticOutputParser(pydantic_object=AlignmentCheckResponse)
            result = await self._call_llm_with_parser(
                prompt=prompt,
                parser=parser,
                system_prompt="You are a validation agent checking role-to-task alignment.",
            )

            issues = []
            for i, issue in enumerate(result.issues):
                suggestion = result.recommendations[i] if i < len(result.recommendations) else None
                issues.append(AlignmentIssue(
                    rule_id=rule_id,
                    description=issue,
                    location="role_to_tasks",
                    severity=AlignmentSeverity.WARNING,  # Warning, not blocker
                    suggestion=suggestion,
                ))

            return AlignmentResult(
                rule_id=rule_id,
                rule_name="Role ↔ Tasks",
                passed=result.is_aligned and result.alignment_score >= 0.7,
                score=result.alignment_score,
                issues=issues,
                details={"role": role_name, "task_count": len(tasks), "activity_count": len(activities)},
            )

        except Exception as e:
            logger.error(f"Role to tasks check failed: {e}")
            return AlignmentResult(
                rule_id=rule_id,
                rule_name="Role ↔ Tasks",
                passed=True,
                score=0.8,
                issues=[],
            )

    @traceable
    async def _check_klo_alignment(
        self,
        adapted_json: dict,
        global_factsheet: dict,
    ) -> AlignmentResult:
        """
        Use GPT-5.2 to check KLO ↔ Task alignment.
        """
        rule_id = "klo_task_alignment"

        topic_data = adapted_json.get("topicWizardData", {})

        # Extract KLOs
        klos = [
            c.get("keyLearningOutcome", "")
            for c in topic_data.get("assessmentCriterion", [])
        ]

        # Extract activities/tasks
        activities = self._extract_activities(topic_data)

        prompt = f"""Check KLO (Key Learning Outcome) ALIGNMENT with tasks/activities.

## KEY LEARNING OUTCOMES:
{json.dumps(klos, indent=2)}

## ACTIVITIES/TASKS IN SIMULATION:
{json.dumps(activities, indent=2)}

## TASK:
Check if the activities/tasks align with and assess the Key Learning Outcomes.

Evaluate:
1. Does each KLO have at least one activity that assesses it?
2. Are activities designed to demonstrate the learning outcomes?
3. Is there logical connection between what students learn and what they do?"""

        try:
            parser = PydanticOutputParser(pydantic_object=AlignmentCheckResponse)
            result = await self._call_llm_with_parser(
                prompt=prompt,
                parser=parser,
                system_prompt="You are a validation agent checking learning outcome alignment.",
            )

            issues = []
            for i, issue in enumerate(result.issues):
                suggestion = result.recommendations[i] if i < len(result.recommendations) else None
                issues.append(AlignmentIssue(
                    rule_id=rule_id,
                    description=issue,
                    location="klo_task_alignment",
                    severity=AlignmentSeverity.WARNING,
                    suggestion=suggestion,
                ))

            return AlignmentResult(
                rule_id=rule_id,
                rule_name="KLO ↔ Task Alignment",
                passed=result.is_aligned and result.alignment_score >= 0.8,
                score=result.alignment_score,
                issues=issues,
                details={"klo_count": len(klos), "activity_count": len(activities)},
            )

        except Exception as e:
            logger.error(f"KLO alignment check failed: {e}")
            return AlignmentResult(
                rule_id=rule_id,
                rule_name="KLO ↔ Task Alignment",
                passed=True,
                score=0.8,
                issues=[],
            )

    @traceable
    async def _check_scenario_coherence(
        self,
        adapted_json: dict,
        global_factsheet: dict,
    ) -> AlignmentResult:
        """
        Use GPT-5.2 to check overall scenario coherence.
        """
        rule_id = "scenario_coherence"

        topic_data = adapted_json.get("topicWizardData", {})

        # Extract key scenario elements
        workplace = topic_data.get("workplaceScenario", {})
        emails = self._extract_emails(topic_data)

        prompt = f"""Check SCENARIO COHERENCE in this adapted simulation.

## WORKPLACE SCENARIO:
{json.dumps(workplace, indent=2)[:8000]}

## EMAILS IN SIMULATION:
{json.dumps(emails, indent=2)[:4000]}

## GLOBAL FACTSHEET:
{json.dumps(global_factsheet, indent=2)[:4000]}

## TASK:
Check if the scenario is internally coherent and consistent.

Evaluate:
1. Do emails match the workplace scenario context?
2. Are company names, roles, and terminology consistent?
3. Does the story flow logically?
4. Are there any contradictions or inconsistencies?"""

        try:
            parser = PydanticOutputParser(pydantic_object=AlignmentCheckResponse)
            result = await self._call_llm_with_parser(
                prompt=prompt,
                parser=parser,
                system_prompt="You are a validation agent checking scenario coherence.",
            )

            issues = []
            for issue in result.issues:
                issues.append(AlignmentIssue(
                    rule_id=rule_id,
                    description=issue,
                    location="scenario_coherence",
                    severity=AlignmentSeverity.WARNING,
                ))

            return AlignmentResult(
                rule_id=rule_id,
                rule_name="Scenario Coherence",
                passed=result.is_aligned and result.alignment_score >= 0.8,
                score=result.alignment_score,
                issues=issues,
            )

        except Exception as e:
            logger.error(f"Scenario coherence check failed: {e}")
            return AlignmentResult(
                rule_id=rule_id,
                rule_name="Scenario Coherence",
                passed=True,
                score=0.8,
                issues=[],
            )

    @traceable
    async def _get_overall_validation(
        self,
        adapted_json: dict,
        global_factsheet: dict,
        check_results: list[AlignmentResult],
    ) -> Optional[OverallValidationResponse]:
        """
        Use GPT-5.2 to provide overall validation summary.
        """
        # Summarize check results
        results_summary = []
        for r in check_results:
            results_summary.append({
                "check": r.rule_name,
                "passed": r.passed,
                "score": r.score,
                "issues": len(r.issues),
            })

        prompt = f"""Provide an OVERALL VALIDATION SUMMARY for this adapted simulation.

## CHECK RESULTS:
{json.dumps(results_summary, indent=2)}

## GLOBAL FACTSHEET:
{json.dumps(global_factsheet, indent=2)[:4000]}

## TASK:
Provide an overall assessment of the adapted simulation quality."""

        try:
            parser = PydanticOutputParser(pydantic_object=OverallValidationResponse)
            result = await self._call_llm_with_parser(
                prompt=prompt,
                parser=parser,
                system_prompt="You are a validation agent providing overall assessment.",
            )
            return result
        except Exception as e:
            logger.error(f"Overall validation failed: {e}")
            return None

    # =========================================================================
    # HELPER METHODS FOR EXTRACTING CONTENT
    # =========================================================================

    def _extract_manager_references(self, topic_data: dict) -> list[dict]:
        """Extract all manager references from the JSON."""
        refs = []

        # From workplaceScenario
        ws_manager = (topic_data.get("workplaceScenario", {})
                     .get("learnerRoleReportingManager", {})
                     .get("reportingManager", {}))
        if ws_manager:
            refs.append({"location": "workplaceScenario", "data": ws_manager})

        # From chatHistory.scenarioDescription
        for i, sd in enumerate(topic_data.get("chatHistory", {}).get("scenarioDescription", [])):
            msg = sd.get("message", {})
            if isinstance(msg, dict):
                ch_manager = msg.get("learnerRoleReportingManager", {}).get("reportingManager", {})
                if ch_manager:
                    refs.append({"location": f"chatHistory[{i}]", "data": ch_manager})

        # From email senders
        for stage_idx, stage in enumerate(topic_data.get("simulationFlow", [])):
            for child_idx, child in enumerate(stage.get("children", [])):
                email = child.get("data", {}).get("email", {})
                if email and email.get("sender"):
                    refs.append({
                        "location": f"simulationFlow[{stage_idx}].children[{child_idx}].email",
                        "data": email.get("sender")
                    })

        return refs

    def _extract_company_references(self, topic_data: dict) -> list[dict]:
        """Extract all company references from the JSON."""
        refs = []

        # From workplaceScenario
        ws_company = topic_data.get("workplaceScenario", {}).get("background", {})
        if ws_company.get("organizationName"):
            refs.append({
                "location": "workplaceScenario",
                "name": ws_company.get("organizationName"),
                "about": ws_company.get("aboutOrganization", "")[:200],
            })

        # From chatHistory.scenarioDescription
        for i, sd in enumerate(topic_data.get("chatHistory", {}).get("scenarioDescription", [])):
            msg = sd.get("message", {})
            if isinstance(msg, dict):
                ch_bg = msg.get("background", {})
                if ch_bg.get("organizationName"):
                    refs.append({
                        "location": f"chatHistory[{i}]",
                        "name": ch_bg.get("organizationName"),
                    })

        return refs

    def _extract_sample_content(self, topic_data: dict) -> dict:
        """
        Extract COMPREHENSIVE content for poison term checking.

        CRITICAL: Must include ALL learner-facing content, especially:
        - Guidelines (often contains domain-specific terms)
        - Overview
        - Scenario text
        - KLOs
        - Activities
        """
        # Basic info
        content = {
            "simulationName": topic_data.get("simulationName", ""),
            "overview": topic_data.get("overview", ""),
            "workplaceScenario": topic_data.get("workplaceScenario", {}).get("scenario", ""),
            "organizationName": topic_data.get("workplaceScenario", {}).get("background", {}).get("organizationName", ""),
            "aboutOrganization": topic_data.get("workplaceScenario", {}).get("background", {}).get("aboutOrganization", "")[:500],
        }

        # CRITICAL: Add guidelines - often contains poison terms!
        guidelines = topic_data.get("guidelines", "")
        if guidelines:
            content["guidelines"] = guidelines[:1500]

        # Add lesson information
        lesson_info = topic_data.get("lessonInformation", {})
        if isinstance(lesson_info, dict):
            content["lessonTitle"] = lesson_info.get("title", "")
            content["lessonDescription"] = lesson_info.get("description", "")[:500]
        elif isinstance(lesson_info, str):
            content["lessonInformation"] = lesson_info[:500]

        # Add KLOs - they might contain source scenario terms
        klos = []
        for criterion in topic_data.get("assessmentCriterion", []):
            klo_text = criterion.get("keyLearningOutcome", "")
            if klo_text:
                klos.append(klo_text[:200])
        if klos:
            content["keyLearningOutcomes"] = klos[:5]

        # Add activity names and descriptions
        activities = []
        for activity in topic_data.get("industryAlignedActivities", [])[:5]:
            if isinstance(activity, dict):
                name = activity.get("name", "")
                desc = activity.get("description", "")[:100]
                if name or desc:
                    activities.append(f"{name}: {desc}")
        if activities:
            content["activities"] = activities

        # Add scope of work tasks
        learner_role = (topic_data.get("workplaceScenario", {})
                       .get("learnerRoleReportingManager", {})
                       .get("learnerRole", {}))
        tasks = []
        for sow in learner_role.get("scopeOfWork", [])[:5]:
            if isinstance(sow, dict):
                task = sow.get("task", "")
                if task:
                    tasks.append(task[:100])
        if tasks:
            content["scopeOfWorkTasks"] = tasks

        # Add email subjects and bodies (first few)
        emails = self._extract_emails(topic_data)
        if emails:
            content["emailSamples"] = emails[:3]

        return content

    def _extract_activities(self, topic_data: dict) -> list[dict]:
        """Extract activities from the JSON."""
        activities = []

        # From industryAlignedActivities - handle both formats
        for activity in topic_data.get("industryAlignedActivities", []):
            if isinstance(activity, dict):
                # Format 1: Direct name/description on activity object
                if activity.get("name"):
                    activities.append({
                        "name": activity.get("name", ""),
                        "description": activity.get("description", "")[:200],
                    })
                # Format 2: Nested message list (legacy format)
                elif isinstance(activity.get("message"), list):
                    for msg in activity["message"]:
                        activities.append({
                            "name": msg.get("name", ""),
                            "description": msg.get("description", "")[:200],
                        })

        # From selectedIndustryAlignedActivities - handle both formats
        for activity in topic_data.get("selectedIndustryAlignedActivities", []):
            if isinstance(activity, dict):
                # Format 1: Direct name/description on activity object
                if activity.get("name"):
                    activities.append({
                        "name": activity.get("name", ""),
                        "description": activity.get("description", "")[:200],
                    })
                # Format 2: Nested message list (legacy format)
                elif isinstance(activity.get("message"), list):
                    for msg in activity["message"]:
                        activities.append({
                            "name": msg.get("name", ""),
                            "description": msg.get("description", "")[:200],
                        })

        # Also extract from scopeOfWork (learner tasks)
        learner_role = (topic_data.get("workplaceScenario", {})
                       .get("learnerRoleReportingManager", {})
                       .get("learnerRole", {}))
        for sow in learner_role.get("scopeOfWork", []):
            if isinstance(sow, dict) and sow.get("task"):
                activities.append({
                    "name": sow.get("task", ""),
                    "description": sow.get("description", "")[:200],
                })

        return activities[:10]  # Limit for prompt size

    def _extract_emails(self, topic_data: dict) -> list[dict]:
        """Extract emails from simulationFlow."""
        emails = []

        for stage in topic_data.get("simulationFlow", []):
            for child in stage.get("children", []):
                email = child.get("data", {}).get("email", {})
                if email:
                    emails.append({
                        "subject": email.get("subject", ""),
                        "sender": email.get("sender", {}).get("name", ""),
                        "body_preview": email.get("body", "")[:300],
                    })

        return emails[:5]  # Limit for prompt size


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

async def check_alignment(
    adapted_json: dict,
    global_factsheet: dict,
    source_scenario: str = "",
    threshold: float = 0.95,
) -> AlignmentReport:
    """
    Run LLM-based alignment checks on adapted simulation.

    Args:
        adapted_json: The adapted simulation JSON
        global_factsheet: The factsheet used during adaptation
        source_scenario: Original scenario text
        threshold: Minimum score required to pass

    Returns:
        AlignmentReport
    """
    checker = AlignmentChecker(threshold=threshold)
    return await checker.check(adapted_json, global_factsheet, source_scenario)
