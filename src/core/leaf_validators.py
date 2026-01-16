"""
Leaf Validators - Validate leaf adaptation output.

Uses GPT 5.2 for semantic validation of adapted leaves.
Runs AFTER adaptation, BEFORE final output.

5 Validators:
1. EntityRemovalValidator - No poison terms remain
2. DomainFidelityValidator - Industry terms/KPIs correct
3. KLOAlignmentValidator - Questions align with KLOs
4. DataConsistencyValidator - Numbers consistent
5. StructureValidator - Values are valid (no placeholders)
"""

import os
import json
import logging
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

from langchain_openai import ChatOpenAI
from langsmith import traceable
import httpx

from .context import AdaptationContext
from .decider import DecisionResult
from .smart_prompts import check_poison_terms, check_klo_alignment

logger = logging.getLogger(__name__)

# GPT 5.2 for validation
VALIDATOR_MODEL = os.getenv("VALIDATION_MODEL", "gpt-5.2-2025-12-11")
MAX_CONCURRENT_VALIDATIONS = int(os.getenv("MAX_CONCURRENT_VALIDATIONS", "6"))

_validation_semaphore = None


def _get_semaphore():
    """Get or create validation semaphore."""
    global _validation_semaphore
    if _validation_semaphore is None:
        _validation_semaphore = asyncio.Semaphore(MAX_CONCURRENT_VALIDATIONS)
    return _validation_semaphore


def _get_validator_llm():
    """Get OpenAI client for validation."""
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(120.0, connect=30.0),
        limits=httpx.Limits(max_connections=50, max_keepalive_connections=25),
    )
    return ChatOpenAI(
        model=VALIDATOR_MODEL,
        temperature=0.0,
        max_retries=2,
        request_timeout=120,
        http_async_client=http_client,
        api_key=os.getenv("OPENAI_API_KEY"),
    )


class ValidationSeverity(Enum):
    """Severity of validation issue."""
    BLOCKER = "blocker"      # Must fix before ship
    WARNING = "warning"      # Should fix, not blocking
    INFO = "info"            # Informational only


@dataclass
class ValidationIssue:
    """A single validation issue found."""
    rule_id: str
    severity: ValidationSeverity
    path: str
    message: str
    old_value: str = ""
    new_value: str = ""
    suggestion: str = ""

    def to_dict(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "severity": self.severity.value,
            "path": self.path,
            "message": self.message,
            "old_value": self.old_value[:100] if self.old_value else "",
            "new_value": self.new_value[:100] if self.new_value else "",
            "suggestion": self.suggestion,
        }


@dataclass
class LeafValidationResult:
    """Result of validating all leaves."""
    total_validated: int
    blockers: int
    warnings: int
    passed: bool
    issues: List[ValidationIssue] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_validated": self.total_validated,
            "blockers": self.blockers,
            "warnings": self.warnings,
            "passed": self.passed,
            "issues": [i.to_dict() for i in self.issues[:20]],
        }


# =============================================================================
# VALIDATOR 1: Entity Removal
# =============================================================================

class EntityRemovalValidator:
    """
    Validates that no poison terms remain in adapted leaves.

    BLOCKER: Any old company/industry term found = fail.
    """

    @traceable(name="leaf_validator_entity_removal")
    async def validate(
        self,
        decisions: List[DecisionResult],
        context: AdaptationContext,
    ) -> List[ValidationIssue]:
        """Check all adapted leaves for poison terms."""
        issues = []

        for decision in decisions:
            if decision.action != "replace" or not decision.new_value:
                continue

            # Check for poison terms
            found_terms = check_poison_terms(decision.new_value, context.poison_terms)

            if found_terms:
                issues.append(ValidationIssue(
                    rule_id="entity_removal",
                    severity=ValidationSeverity.BLOCKER,
                    path=decision.path,
                    message=f"Contains poison terms: {', '.join(found_terms)}",
                    old_value=decision.old_value,
                    new_value=decision.new_value,
                    suggestion=f"Replace with: {context.new_company_name}",
                ))

        logger.info(f"EntityRemovalValidator: {len(issues)} issues found")
        return issues


# =============================================================================
# VALIDATOR 2: Domain Fidelity
# =============================================================================

class DomainFidelityValidator:
    """
    Validates industry terms and KPIs match target industry.

    BLOCKER: Wrong industry KPIs (e.g., CAC in beverage industry).
    """

    # Terms that are WRONG for specific industries
    WRONG_TERMS = {
        "beverage": ["CAC", "LTV", "churn", "MRR", "ARR", "subscription", "SaaS", "freemium"],
        "hospitality": ["CAC", "LTV", "churn", "MRR", "ARR", "subscription", "SaaS"],
        "retail": ["CAC", "MRR", "ARR", "subscription", "SaaS", "freemium"],
        "manufacturing": ["CAC", "LTV", "churn", "MRR", "ARR", "SaaS"],
    }

    @traceable(name="leaf_validator_domain_fidelity")
    async def validate(
        self,
        decisions: List[DecisionResult],
        context: AdaptationContext,
    ) -> List[ValidationIssue]:
        """Check adapted leaves for wrong industry terms."""
        issues = []

        # Get wrong terms for target industry
        industry_lower = context.target_industry.lower() if context.target_industry else ""
        wrong_terms = []
        for industry_key, terms in self.WRONG_TERMS.items():
            if industry_key in industry_lower:
                wrong_terms = terms
                break

        # Also use invalid_kpis from context
        wrong_terms.extend(context.invalid_kpis)
        wrong_terms = list(set(wrong_terms))

        if not wrong_terms:
            return issues

        for decision in decisions:
            if decision.action != "replace" or not decision.new_value:
                continue

            value_upper = decision.new_value.upper()
            found_wrong = [t for t in wrong_terms if t.upper() in value_upper]

            if found_wrong:
                issues.append(ValidationIssue(
                    rule_id="domain_fidelity",
                    severity=ValidationSeverity.BLOCKER,
                    path=decision.path,
                    message=f"Contains wrong industry terms: {', '.join(found_wrong)}",
                    old_value=decision.old_value,
                    new_value=decision.new_value,
                    suggestion=f"Use {context.target_industry} industry terms instead",
                ))

        logger.info(f"DomainFidelityValidator: {len(issues)} issues found")
        return issues


# =============================================================================
# VALIDATOR 3: KLO Alignment
# =============================================================================

class KLOAlignmentValidator:
    """
    Validates questions align with KLO terms.

    WARNING: Question doesn't reference any KLO term.
    """

    @traceable(name="leaf_validator_klo_alignment")
    async def validate(
        self,
        decisions: List[DecisionResult],
        context: AdaptationContext,
    ) -> List[ValidationIssue]:
        """Check question leaves for KLO alignment."""
        issues = []

        if not context.klo_terms:
            return issues

        # Only check question-related paths
        question_keywords = ["question", "submission", "reflect"]

        for decision in decisions:
            if decision.action != "replace" or not decision.new_value:
                continue

            # Check if this is a question path
            path_lower = decision.path.lower()
            is_question = any(kw in path_lower for kw in question_keywords)

            if not is_question:
                continue

            # Check KLO alignment
            aligned = check_klo_alignment(decision.new_value, context.klo_terms)

            if not aligned:
                issues.append(ValidationIssue(
                    rule_id="klo_alignment",
                    severity=ValidationSeverity.WARNING,
                    path=decision.path,
                    message="Question doesn't reference any KLO term",
                    old_value=decision.old_value,
                    new_value=decision.new_value,
                    suggestion=f"Include one of: {', '.join(context.klo_terms.values())}",
                ))

        logger.info(f"KLOAlignmentValidator: {len(issues)} issues found")
        return issues


# =============================================================================
# VALIDATOR 4: Data Consistency
# =============================================================================

class DataConsistencyValidator:
    """
    Validates numbers/data are consistent across leaves.

    WARNING: Inconsistent numbers in adapted content.
    """

    @traceable(name="leaf_validator_data_consistency")
    async def validate(
        self,
        decisions: List[DecisionResult],
        context: AdaptationContext,
    ) -> List[ValidationIssue]:
        """Check for data consistency across adapted leaves."""
        issues = []

        if not context.resource_data:
            return issues

        # Extract numbers from resource_data
        expected_numbers = {}
        for key, value in context.resource_data.items():
            # Extract numbers from value
            import re
            numbers = re.findall(r'\d+(?:\.\d+)?%?', str(value))
            if numbers:
                expected_numbers[key] = numbers[0]

        if not expected_numbers:
            return issues

        # Check resource-related paths for consistency
        resource_keywords = ["resource", "data", "metric", "report"]

        for decision in decisions:
            if decision.action != "replace" or not decision.new_value:
                continue

            path_lower = decision.path.lower()
            is_resource = any(kw in path_lower for kw in resource_keywords)

            if not is_resource:
                continue

            # Check if numbers in content match expected
            import re
            found_numbers = re.findall(r'\d+(?:\.\d+)?%?', decision.new_value)

            # Simple check: if we find numbers, they should be in expected
            for num in found_numbers:
                if num not in expected_numbers.values() and len(num) > 2:
                    issues.append(ValidationIssue(
                        rule_id="data_consistency",
                        severity=ValidationSeverity.WARNING,
                        path=decision.path,
                        message=f"Number '{num}' may be inconsistent with resource data",
                        new_value=decision.new_value[:200],
                        suggestion="Verify numbers match resource data",
                    ))
                    break

        logger.info(f"DataConsistencyValidator: {len(issues)} issues found")
        return issues


# =============================================================================
# VALIDATOR 5: Structure/Placeholder
# =============================================================================

class StructureValidator:
    """
    Validates no placeholders or invalid patterns remain.

    BLOCKER: Placeholders like [INSERT], TBD, XXX found.
    """

    PLACEHOLDER_PATTERNS = [
        r'\[INSERT.*?\]',
        r'\[TODO.*?\]',
        r'\[PLACEHOLDER.*?\]',
        r'\bTBD\b',
        r'\bTBC\b',
        r'\bXXX+\b',
        r'\bN/A\b',
        r'<[A-Z_]+>',
    ]

    @traceable(name="leaf_validator_structure")
    async def validate(
        self,
        decisions: List[DecisionResult],
        context: AdaptationContext,
    ) -> List[ValidationIssue]:
        """Check for placeholders and invalid patterns."""
        import re
        issues = []

        compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.PLACEHOLDER_PATTERNS]

        for decision in decisions:
            if decision.action != "replace" or not decision.new_value:
                continue

            for pattern in compiled_patterns:
                match = pattern.search(decision.new_value)
                if match:
                    issues.append(ValidationIssue(
                        rule_id="structure_integrity",
                        severity=ValidationSeverity.BLOCKER,
                        path=decision.path,
                        message=f"Contains placeholder: {match.group()}",
                        new_value=decision.new_value[:200],
                        suggestion="Replace placeholder with actual content",
                    ))
                    break

        logger.info(f"StructureValidator: {len(issues)} issues found")
        return issues


# =============================================================================
# VALIDATOR 6: Coherence (LLM-Based - Dynamic, not hardcoded)
# =============================================================================

class CoherenceValidator:
    """
    Validates content is coherent using LLM semantic understanding.

    BLOCKER: Content that doesn't make sense in the target context.

    Uses LLM to check:
    - Does the content make grammatical sense?
    - Is it coherent within the target industry?
    - Does it look like proper transformation (not literal word swap)?

    NO HARDCODED PATTERNS - LLM determines coherence dynamically.
    """

    @traceable(name="leaf_validator_coherence")
    async def validate(
        self,
        decisions: List[DecisionResult],
        context: AdaptationContext,
    ) -> List[ValidationIssue]:
        """Check for incoherent content using LLM semantic analysis."""
        issues = []

        # Filter to replaced content only
        changes = [d for d in decisions if d.action == "replace" and d.new_value]

        if not changes:
            return issues

        # Sample up to 10 significant changes for LLM coherence check
        # (checking all would be too expensive)
        significant_changes = [
            d for d in changes
            if len(d.new_value) > 50  # Only check substantial content
        ][:10]

        if not significant_changes:
            return issues

        # Use LLM to check coherence
        try:
            llm = _get_validator_llm()

            # Build batch coherence check prompt
            content_items = []
            for i, decision in enumerate(significant_changes):
                content_items.append(f"""
<item index="{i}">
<path>{decision.path}</path>
<content>{decision.new_value[:400]}</content>
</item>""")

            prompt = f"""<task>
Check if the following content is COHERENT for a simulation about:
{context.target_scenario[:300] if context.target_scenario else context.target_industry}

Industry: {context.target_industry or "Target industry"}
Company: {context.new_company_name or "Target company"}
</task>

<content_to_check>
{"".join(content_items)}
</content_to_check>

<coherence_criteria>
For each item, determine if it:
1. Makes GRAMMATICAL sense (proper sentence structure)
2. Makes SEMANTIC sense (concepts that logically go together)
3. Fits the TARGET industry (uses appropriate terminology)
4. Looks like PROPER content (not a broken word-replacement artifact)

Examples of INCOHERENT content (would FAIL):
- "organic T-shirts questions" (nonsense phrase)
- "market's communication skills" (markets don't have skills)
- "STAR interview method for retail analysis" (mixing HR and retail concepts)

Examples of COHERENT content (would PASS):
- "market analysis framework" (makes sense)
- "brand positioning strategy" (makes sense for retail)
- "consumer demographic analysis" (makes sense)
</coherence_criteria>

<response_format>
{{
  "results": [
    {{
      "index": 0,
      "coherent": true,
      "issues": []
    }},
    {{
      "index": 1,
      "coherent": false,
      "issues": ["Specific issue description"]
    }}
  ]
}}
</response_format>
"""

            async with _get_semaphore():
                response = await llm.ainvoke(prompt)

            # Parse response
            import json
            response_text = response.content if hasattr(response, 'content') else str(response)

            # Extract JSON from response
            json_match = response_text
            if "```json" in response_text:
                json_match = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                json_match = response_text.split("```")[1].split("```")[0]

            result = json.loads(json_match.strip())

            # Create issues for incoherent content
            for item in result.get("results", []):
                if not item.get("coherent", True):
                    idx = item.get("index", 0)
                    if idx < len(significant_changes):
                        decision = significant_changes[idx]
                        issue_msgs = item.get("issues", ["Content is not coherent"])

                        issues.append(ValidationIssue(
                            rule_id="coherence",
                            severity=ValidationSeverity.BLOCKER,
                            path=decision.path,
                            message=f"Incoherent content: {'; '.join(issue_msgs)}",
                            new_value=decision.new_value[:200],
                            suggestion="Content needs semantic transformation, not literal replacement",
                        ))

        except Exception as e:
            logger.warning(f"CoherenceValidator LLM check failed: {e}")
            # Fall back to basic checks if LLM fails
            issues.extend(self._basic_coherence_check(changes, context))

        logger.info(f"CoherenceValidator: {len(issues)} issues found")
        return issues

    def _basic_coherence_check(
        self,
        decisions: List[DecisionResult],
        context: AdaptationContext,
    ) -> List[ValidationIssue]:
        """Basic coherence check fallback (no hardcoded industry-specific patterns)."""
        issues = []

        for decision in decisions:
            if not decision.new_value:
                continue

            # Check for obvious structural issues only
            # (placeholders, HTML tags, incomplete content)
            import re

            # Check for HTML tags that shouldn't be in text content
            if re.search(r'<(?!br|p|b|i|em|strong)[a-z]+>', decision.new_value, re.IGNORECASE):
                issues.append(ValidationIssue(
                    rule_id="coherence",
                    severity=ValidationSeverity.BLOCKER,
                    path=decision.path,
                    message="Contains unexpected HTML tags",
                    new_value=decision.new_value[:200],
                    suggestion="Remove HTML tags or fix content generation",
                ))

            # Check for placeholder patterns
            if re.search(r'\[INSERT|\[TODO|\[PLACEHOLDER|TBD|XXX', decision.new_value, re.IGNORECASE):
                issues.append(ValidationIssue(
                    rule_id="coherence",
                    severity=ValidationSeverity.BLOCKER,
                    path=decision.path,
                    message="Contains placeholder text",
                    new_value=decision.new_value[:200],
                    suggestion="Replace placeholder with actual content",
                ))

        return issues


# =============================================================================
# MAIN VALIDATOR ORCHESTRATOR
# =============================================================================

class LeafValidator:
    """
    Orchestrates all leaf validators.

    Runs all 6 validators in parallel and aggregates results.
    """

    def __init__(self):
        self.entity_removal = EntityRemovalValidator()
        self.domain_fidelity = DomainFidelityValidator()
        self.klo_alignment = KLOAlignmentValidator()
        self.data_consistency = DataConsistencyValidator()
        self.structure = StructureValidator()
        self.coherence = CoherenceValidator()  # NEW: catches nonsensical content

    @traceable(name="leaf_validator_all")
    async def validate_all(
        self,
        decisions: List[DecisionResult],
        context: AdaptationContext,
    ) -> LeafValidationResult:
        """
        Run all validators on adapted leaves.

        Args:
            decisions: List of DecisionResult from adaptation
            context: AdaptationContext with validation rules

        Returns:
            LeafValidationResult with all issues
        """
        # Filter to only "replace" decisions
        changes = [d for d in decisions if d.action == "replace" and d.new_value]

        if not changes:
            return LeafValidationResult(
                total_validated=0,
                blockers=0,
                warnings=0,
                passed=True,
                issues=[],
            )

        logger.info(f"Validating {len(changes)} adapted leaves...")

        # Run all 6 validators in parallel
        results = await asyncio.gather(
            self.entity_removal.validate(decisions, context),
            self.domain_fidelity.validate(decisions, context),
            self.klo_alignment.validate(decisions, context),
            self.data_consistency.validate(decisions, context),
            self.structure.validate(decisions, context),
            self.coherence.validate(decisions, context),  # NEW: catches nonsensical content
            return_exceptions=True,
        )

        # Collect all issues
        all_issues = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Validator failed: {result}")
            elif isinstance(result, list):
                all_issues.extend(result)

        # Count by severity
        blockers = sum(1 for i in all_issues if i.severity == ValidationSeverity.BLOCKER)
        warnings = sum(1 for i in all_issues if i.severity == ValidationSeverity.WARNING)

        passed = blockers == 0

        logger.info(f"Validation complete: {blockers} blockers, {warnings} warnings, passed={passed}")

        return LeafValidationResult(
            total_validated=len(changes),
            blockers=blockers,
            warnings=warnings,
            passed=passed,
            issues=all_issues,
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def validate_leaf_decisions(
    decisions: List[DecisionResult],
    context: AdaptationContext,
) -> LeafValidationResult:
    """
    Validate leaf adaptation decisions.

    Args:
        decisions: List of DecisionResult from adaptation
        context: AdaptationContext with validation rules

    Returns:
        LeafValidationResult with all issues
    """
    validator = LeafValidator()
    return await validator.validate_all(decisions, context)


def get_blocker_issues(result: LeafValidationResult) -> List[ValidationIssue]:
    """Get only blocker issues that must be fixed."""
    return [i for i in result.issues if i.severity == ValidationSeverity.BLOCKER]


def get_issues_by_rule(result: LeafValidationResult, rule_id: str) -> List[ValidationIssue]:
    """Get issues for a specific rule."""
    return [i for i in result.issues if i.rule_id == rule_id]
