"""
Leaf Fixers - Fix validation issues in adapted leaves.

Uses GPT 5.2 for semantic fixes.
Targeted fixes based on validation issue type.

3 Fixers:
1. LeafEntityFixer - Remove poison terms
2. LeafDomainFixer - Fix wrong industry terms
3. LeafStructureFixer - Fix placeholders/invalid patterns
"""

import os
import json
import logging
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from langchain_openai import ChatOpenAI
from langsmith import traceable
import httpx

from .context import AdaptationContext
from .decider import DecisionResult
from .leaf_validators import ValidationIssue, ValidationSeverity

logger = logging.getLogger(__name__)

# GPT 5.2 for fixing
FIXER_MODEL = os.getenv("FIXER_MODEL", "gpt-5.2-2025-12-11")
MAX_CONCURRENT_FIXES = int(os.getenv("MAX_CONCURRENT_FIXES", "6"))

_fixer_semaphore = None


def _get_semaphore():
    """Get or create fixer semaphore."""
    global _fixer_semaphore
    if _fixer_semaphore is None:
        _fixer_semaphore = asyncio.Semaphore(MAX_CONCURRENT_FIXES)
    return _fixer_semaphore


def _get_fixer_llm():
    """Get OpenAI client for fixing."""
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(120.0, connect=30.0),
        limits=httpx.Limits(max_connections=50, max_keepalive_connections=25),
    )
    return ChatOpenAI(
        model=FIXER_MODEL,
        temperature=0.1,
        max_retries=2,
        request_timeout=120,
        http_async_client=http_client,
        api_key=os.getenv("OPENAI_API_KEY"),
    )


@dataclass
class FixResult:
    """Result of fixing a single leaf."""
    path: str
    success: bool
    original_value: str
    fixed_value: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "success": self.success,
            "original_value": self.original_value[:100],
            "fixed_value": self.fixed_value[:100] if self.fixed_value else None,
            "error": self.error,
        }


@dataclass
class LeafFixerResult:
    """Result of fixing all leaves."""
    total_issues: int
    fixes_attempted: int
    fixes_succeeded: int
    fixes_failed: int
    fix_results: List[FixResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_issues": self.total_issues,
            "fixes_attempted": self.fixes_attempted,
            "fixes_succeeded": self.fixes_succeeded,
            "fixes_failed": self.fixes_failed,
            "success_rate": self.fixes_succeeded / max(1, self.fixes_attempted),
            "fix_results": [r.to_dict() for r in self.fix_results[:10]],
        }


# =============================================================================
# FIXER 1: Entity Removal Fixer
# =============================================================================

class LeafEntityFixer:
    """
    Fixes leaves that contain poison terms (old company/industry names).

    Uses GPT 5.2 to intelligently replace terms while preserving meaning.
    """

    @traceable(name="leaf_fixer_entity")
    async def fix(
        self,
        issue: ValidationIssue,
        context: AdaptationContext,
    ) -> FixResult:
        """Fix a single leaf with entity issues."""
        try:
            llm = _get_fixer_llm()

            prompt = f"""Fix this text by removing old company/industry references.

## OLD REFERENCES TO REMOVE:
{json.dumps(context.poison_terms)}

## REPLACEMENT MAPPINGS:
- Old company -> {context.new_company_name}
- Old industry -> {context.target_industry}
{self._format_entity_map(context.entity_map)}

## TEXT TO FIX:
"{issue.new_value}"

## ISSUE FOUND:
{issue.message}

## RULES:
1. Replace ALL old references with new equivalents
2. Preserve the exact meaning and tone
3. Keep same length and style
4. Do NOT add new content

## OUTPUT (JSON only):
{{"fixed_text": "your corrected text here"}}"""

            semaphore = _get_semaphore()
            async with semaphore:
                response = await llm.ainvoke(prompt)

            # Parse response
            content = response.content if hasattr(response, 'content') else str(response)
            fixed_text = self._extract_fixed_text(content)

            if fixed_text:
                return FixResult(
                    path=issue.path,
                    success=True,
                    original_value=issue.new_value,
                    fixed_value=fixed_text,
                )
            else:
                return FixResult(
                    path=issue.path,
                    success=False,
                    original_value=issue.new_value,
                    error="Could not extract fixed text from LLM response",
                )

        except Exception as e:
            logger.error(f"Entity fix failed for {issue.path}: {e}")
            return FixResult(
                path=issue.path,
                success=False,
                original_value=issue.new_value,
                error=str(e),
            )

    def _format_entity_map(self, entity_map: Dict[str, str]) -> str:
        """Format entity map for prompt."""
        if not entity_map:
            return ""
        lines = [f"- {old} -> {new}" for old, new in entity_map.items()]
        return "\n".join(lines)

    def _extract_fixed_text(self, content: str) -> Optional[str]:
        """Extract fixed_text from LLM response."""
        try:
            # Try to parse as JSON
            start = content.find('{')
            end = content.rfind('}') + 1
            if start >= 0 and end > start:
                data = json.loads(content[start:end])
                return data.get("fixed_text")
        except:
            pass
        return None


# =============================================================================
# FIXER 2: Domain Fidelity Fixer
# =============================================================================

class LeafDomainFixer:
    """
    Fixes leaves that contain wrong industry terms/KPIs.

    Replaces tech/SaaS terms with industry-appropriate alternatives.
    """

    # Replacement suggestions by industry
    REPLACEMENTS = {
        "beverage": {
            "CAC": "customer acquisition cost",
            "LTV": "customer lifetime value",
            "churn": "customer attrition",
            "MRR": "monthly revenue",
            "ARR": "annual revenue",
            "subscription": "purchase pattern",
            "SaaS": "consumer goods",
        },
        "hospitality": {
            "CAC": "guest acquisition cost",
            "LTV": "guest lifetime value",
            "churn": "guest attrition",
            "MRR": "monthly revenue",
            "subscription": "booking pattern",
        },
        "retail": {
            "CAC": "customer acquisition cost",
            "MRR": "monthly sales",
            "ARR": "annual sales",
            "subscription": "purchase frequency",
        },
    }

    @traceable(name="leaf_fixer_domain")
    async def fix(
        self,
        issue: ValidationIssue,
        context: AdaptationContext,
    ) -> FixResult:
        """Fix a single leaf with domain/industry issues."""
        try:
            llm = _get_fixer_llm()

            # Get industry-specific replacements
            industry_lower = context.target_industry.lower() if context.target_industry else ""
            replacements = {}
            for key, reps in self.REPLACEMENTS.items():
                if key in industry_lower:
                    replacements = reps
                    break

            prompt = f"""Fix this text by replacing wrong industry terms.

## TARGET INDUSTRY:
{context.target_industry}

## WRONG TERMS TO REPLACE:
{json.dumps(context.invalid_kpis)}

## SUGGESTED REPLACEMENTS:
{json.dumps(replacements)}

## VALID TERMS FOR THIS INDUSTRY:
{json.dumps(context.valid_kpis)}

## TEXT TO FIX:
"{issue.new_value}"

## ISSUE FOUND:
{issue.message}

## RULES:
1. Replace wrong industry terms with appropriate alternatives
2. Use industry-standard terminology for {context.target_industry}
3. Preserve the exact meaning
4. Keep same length and style

## OUTPUT (JSON only):
{{"fixed_text": "your corrected text here"}}"""

            semaphore = _get_semaphore()
            async with semaphore:
                response = await llm.ainvoke(prompt)

            content = response.content if hasattr(response, 'content') else str(response)
            fixed_text = self._extract_fixed_text(content)

            if fixed_text:
                return FixResult(
                    path=issue.path,
                    success=True,
                    original_value=issue.new_value,
                    fixed_value=fixed_text,
                )
            else:
                return FixResult(
                    path=issue.path,
                    success=False,
                    original_value=issue.new_value,
                    error="Could not extract fixed text",
                )

        except Exception as e:
            logger.error(f"Domain fix failed for {issue.path}: {e}")
            return FixResult(
                path=issue.path,
                success=False,
                original_value=issue.new_value,
                error=str(e),
            )

    def _extract_fixed_text(self, content: str) -> Optional[str]:
        """Extract fixed_text from LLM response."""
        try:
            start = content.find('{')
            end = content.rfind('}') + 1
            if start >= 0 and end > start:
                data = json.loads(content[start:end])
                return data.get("fixed_text")
        except:
            pass
        return None


# =============================================================================
# FIXER 3: Structure Fixer
# =============================================================================

class LeafStructureFixer:
    """
    Fixes leaves that contain placeholders or invalid patterns.

    Replaces [INSERT], TBD, XXX with actual content.
    """

    @traceable(name="leaf_fixer_structure")
    async def fix(
        self,
        issue: ValidationIssue,
        context: AdaptationContext,
    ) -> FixResult:
        """Fix a single leaf with structure/placeholder issues."""
        try:
            llm = _get_fixer_llm()

            prompt = f"""Fix this text by replacing placeholders with actual content.

## CONTEXT:
- Company: {context.new_company_name}
- Industry: {context.target_industry}
- Scenario: {context.target_scenario[:500] if context.target_scenario else 'Business simulation'}

## TEXT TO FIX:
"{issue.new_value}"

## ISSUE FOUND:
{issue.message}

## RULES:
1. Replace ALL placeholders ([INSERT], TBD, XXX, etc.) with real content
2. Content must be realistic for {context.target_industry} industry
3. Keep same length and style as surrounding text
4. Use specific numbers/names, not generic ones

## OUTPUT (JSON only):
{{"fixed_text": "your corrected text with real content"}}"""

            semaphore = _get_semaphore()
            async with semaphore:
                response = await llm.ainvoke(prompt)

            content = response.content if hasattr(response, 'content') else str(response)
            fixed_text = self._extract_fixed_text(content)

            if fixed_text:
                return FixResult(
                    path=issue.path,
                    success=True,
                    original_value=issue.new_value,
                    fixed_value=fixed_text,
                )
            else:
                return FixResult(
                    path=issue.path,
                    success=False,
                    original_value=issue.new_value,
                    error="Could not extract fixed text",
                )

        except Exception as e:
            logger.error(f"Structure fix failed for {issue.path}: {e}")
            return FixResult(
                path=issue.path,
                success=False,
                original_value=issue.new_value,
                error=str(e),
            )

    def _extract_fixed_text(self, content: str) -> Optional[str]:
        """Extract fixed_text from LLM response."""
        try:
            start = content.find('{')
            end = content.rfind('}') + 1
            if start >= 0 and end > start:
                data = json.loads(content[start:end])
                return data.get("fixed_text")
        except:
            pass
        return None


# =============================================================================
# MAIN FIXER ORCHESTRATOR
# =============================================================================

class LeafFixer:
    """
    Orchestrates all leaf fixers.

    Routes issues to appropriate fixer based on rule_id.
    Runs fixes in parallel with rate limiting.
    """

    def __init__(self):
        self.entity_fixer = LeafEntityFixer()
        self.domain_fixer = LeafDomainFixer()
        self.structure_fixer = LeafStructureFixer()

    def _get_fixer_for_issue(self, issue: ValidationIssue):
        """Get appropriate fixer for issue type."""
        if issue.rule_id == "entity_removal":
            return self.entity_fixer
        elif issue.rule_id == "domain_fidelity":
            return self.domain_fixer
        elif issue.rule_id == "structure_integrity":
            return self.structure_fixer
        else:
            # Default to entity fixer for unknown issues
            return self.entity_fixer

    @traceable(name="leaf_fixer_all")
    async def fix_all(
        self,
        issues: List[ValidationIssue],
        context: AdaptationContext,
    ) -> LeafFixerResult:
        """
        Fix all validation issues.

        Args:
            issues: List of ValidationIssue from validation
            context: AdaptationContext with fix context

        Returns:
            LeafFixerResult with all fix results
        """
        # Only fix blockers
        blockers = [i for i in issues if i.severity == ValidationSeverity.BLOCKER]

        if not blockers:
            return LeafFixerResult(
                total_issues=len(issues),
                fixes_attempted=0,
                fixes_succeeded=0,
                fixes_failed=0,
            )

        logger.info(f"Fixing {len(blockers)} blocker issues...")

        # Create fix tasks
        async def fix_one(issue: ValidationIssue) -> FixResult:
            fixer = self._get_fixer_for_issue(issue)
            return await fixer.fix(issue, context)

        # Run all fixes in parallel
        tasks = [fix_one(issue) for issue in blockers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        fix_results = []
        succeeded = 0
        failed = 0

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Fix failed: {result}")
                failed += 1
            elif isinstance(result, FixResult):
                fix_results.append(result)
                if result.success:
                    succeeded += 1
                else:
                    failed += 1

        logger.info(f"Fixing complete: {succeeded} succeeded, {failed} failed")

        return LeafFixerResult(
            total_issues=len(issues),
            fixes_attempted=len(blockers),
            fixes_succeeded=succeeded,
            fixes_failed=failed,
            fix_results=fix_results,
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def fix_leaf_issues(
    issues: List[ValidationIssue],
    context: AdaptationContext,
) -> LeafFixerResult:
    """
    Fix validation issues in leaves.

    Args:
        issues: List of ValidationIssue from validation
        context: AdaptationContext with fix context

    Returns:
        LeafFixerResult with all fix results
    """
    fixer = LeafFixer()
    return await fixer.fix_all(issues, context)


def apply_fixes_to_decisions(
    decisions: List[DecisionResult],
    fix_results: List[FixResult],
) -> List[DecisionResult]:
    """
    Apply fix results back to decisions.

    Updates the new_value in decisions with fixed values.
    """
    # Build path -> fixed_value map
    fix_map = {r.path: r.fixed_value for r in fix_results if r.success and r.fixed_value}

    # Update decisions
    for decision in decisions:
        if decision.path in fix_map:
            decision.new_value = fix_map[decision.path]
            decision.reason += " [FIXED]"

    return decisions
