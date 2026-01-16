"""
Leaf Repair Loop - Validate → Fix → Re-validate cycle with SMART RETRIES.

Enhanced repair workflow with escalating strategies:
1. Validate adapted leaves
2. If blockers found, try STANDARD FIX (semantic fixer)
3. If still failing, escalate to TARGETED RETRY (smart prompt)
4. If still failing, escalate to AGGRESSIVE FIX (direct replacement)
5. Track failed paths to avoid infinite loops
6. Repeat up to max_iterations

Uses GPT 5.2 for validation and fixing.
"""

import os
import json
import logging
import asyncio
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field

from langchain_openai import ChatOpenAI
from langsmith import traceable
import httpx

from .context import AdaptationContext
from .decider import DecisionResult
from .smart_prompts import build_targeted_retry_prompt, check_poison_terms
from .leaf_validators import (
    LeafValidator,
    LeafValidationResult,
    ValidationIssue,
    ValidationSeverity,
    get_blocker_issues,
)
from .leaf_fixers import (
    LeafFixer,
    LeafFixerResult,
    FixResult,
    apply_fixes_to_decisions,
)

logger = logging.getLogger(__name__)

# GPT 5.2 for smart retries
RETRY_MODEL = os.getenv("FIXER_MODEL", "gpt-5.2-2025-12-11")
MAX_CONCURRENT_RETRIES = int(os.getenv("MAX_CONCURRENT_RETRIES", "4"))

_retry_semaphore = None


def _get_retry_semaphore():
    """Get or create retry semaphore."""
    global _retry_semaphore
    if _retry_semaphore is None:
        _retry_semaphore = asyncio.Semaphore(MAX_CONCURRENT_RETRIES)
    return _retry_semaphore


def _get_retry_llm():
    """Get OpenAI client for smart retries."""
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(120.0, connect=30.0),
        limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
    )
    return ChatOpenAI(
        model=RETRY_MODEL,
        temperature=0.0,  # More deterministic for retries
        max_retries=2,
        request_timeout=120,
        http_async_client=http_client,
        api_key=os.getenv("OPENAI_API_KEY"),
    )


# =============================================================================
# REPAIR TRACKING
# =============================================================================

@dataclass
class PathRepairHistory:
    """Track repair attempts for a single path."""
    path: str
    original_value: str
    attempts: List[Dict[str, Any]] = field(default_factory=list)
    current_value: str = ""
    strategy_used: str = "none"
    resolved: bool = False

    def add_attempt(self, strategy: str, new_value: str, success: bool, failures: List[str]):
        self.attempts.append({
            "strategy": strategy,
            "new_value": new_value[:200],
            "success": success,
            "failures": failures,
        })
        self.current_value = new_value
        self.strategy_used = strategy
        if success:
            self.resolved = True


@dataclass
class RepairIteration:
    """Record of a single repair iteration."""
    iteration: int
    validation_result: LeafValidationResult
    fix_result: Optional[LeafFixerResult] = None
    smart_retries: int = 0
    aggressive_fixes: int = 0
    blockers_before: int = 0
    blockers_after: int = 0
    strategy: str = "standard"


@dataclass
class RepairLoopResult:
    """Final result of the repair loop."""
    passed: bool
    total_iterations: int
    initial_blockers: int
    final_blockers: int
    total_fixes_attempted: int
    total_fixes_succeeded: int
    smart_retries_used: int
    aggressive_fixes_used: int
    decisions: List[DecisionResult]
    iterations: List[RepairIteration] = field(default_factory=list)
    unresolved_paths: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "total_iterations": self.total_iterations,
            "initial_blockers": self.initial_blockers,
            "final_blockers": self.final_blockers,
            "total_fixes_attempted": self.total_fixes_attempted,
            "total_fixes_succeeded": self.total_fixes_succeeded,
            "smart_retries_used": self.smart_retries_used,
            "aggressive_fixes_used": self.aggressive_fixes_used,
            "unresolved_paths": self.unresolved_paths,
            "iterations": [
                {
                    "iteration": it.iteration,
                    "blockers_before": it.blockers_before,
                    "blockers_after": it.blockers_after,
                    "strategy": it.strategy,
                    "smart_retries": it.smart_retries,
                    "aggressive_fixes": it.aggressive_fixes,
                    "fixes_attempted": it.fix_result.fixes_attempted if it.fix_result else 0,
                    "fixes_succeeded": it.fix_result.fixes_succeeded if it.fix_result else 0,
                }
                for it in self.iterations
            ],
        }


# =============================================================================
# SMART RETRY FIXER
# =============================================================================

class SmartRetryFixer:
    """
    Uses build_targeted_retry_prompt for intelligent retries.

    This is the escalation when standard fixers fail.
    It tells the LLM exactly what went wrong and demands a fix.
    """

    def __init__(self):
        self.llm = _get_retry_llm()

    @traceable(name="smart_retry_fix")
    async def fix(
        self,
        decision: DecisionResult,
        context: AdaptationContext,
        failures: List[str],
    ) -> FixResult:
        """
        Smart retry with targeted prompt.

        Args:
            decision: The DecisionResult that failed validation
            context: AdaptationContext
            failures: List of specific failure reasons

        Returns:
            FixResult with fixed value or error
        """
        try:
            # Build the targeted retry prompt
            prompt = build_targeted_retry_prompt(
                context=context,
                path=decision.path,
                previous_value=decision.new_value,
                original_value=decision.old_value,
                failures=failures,
            )

            semaphore = _get_retry_semaphore()
            async with semaphore:
                response = await self.llm.ainvoke(prompt)

            content = response.content if hasattr(response, 'content') else str(response)
            fixed_text = self._extract_fixed_text(content)

            if fixed_text:
                # Verify fix actually worked (quick poison check)
                remaining_poison = check_poison_terms(fixed_text, context.poison_terms)
                if remaining_poison:
                    logger.warning(f"Smart retry still has poison: {remaining_poison[:3]}")
                    # Continue anyway, will be caught in next validation

                return FixResult(
                    path=decision.path,
                    success=True,
                    original_value=decision.new_value,
                    fixed_value=fixed_text,
                )
            else:
                return FixResult(
                    path=decision.path,
                    success=False,
                    original_value=decision.new_value,
                    error="Could not extract fixed text from smart retry",
                )

        except Exception as e:
            logger.error(f"Smart retry failed for {decision.path}: {e}")
            return FixResult(
                path=decision.path,
                success=False,
                original_value=decision.new_value,
                error=str(e),
            )

    def _extract_fixed_text(self, content: str) -> Optional[str]:
        """Extract new_value from smart retry response."""
        try:
            start = content.find('{')
            end = content.rfind('}') + 1
            if start >= 0 and end > start:
                data = json.loads(content[start:end])
                return data.get("new_value")
        except:
            pass
        return None


# =============================================================================
# AGGRESSIVE FIXER
# =============================================================================

class AggressiveFixer:
    """
    Last resort: Direct string replacement without LLM.

    Used when smart retries still fail.
    Does brute-force find/replace of poison terms.
    """

    def fix(
        self,
        decision: DecisionResult,
        context: AdaptationContext,
    ) -> FixResult:
        """
        Aggressive fix: direct replacement of poison terms.

        No LLM call - just string manipulation.
        """
        try:
            text = decision.new_value

            # Direct replacement of poison terms
            for term in context.poison_terms:
                if term.lower() in text.lower():
                    # Find replacement
                    replacement = context.entity_map.get(term)
                    if not replacement:
                        replacement = context.new_company_name or "[COMPANY]"

                    # Case-insensitive replacement
                    import re
                    pattern = re.compile(re.escape(term), re.IGNORECASE)
                    text = pattern.sub(replacement, text)

            # Check if we actually changed anything
            if text != decision.new_value:
                return FixResult(
                    path=decision.path,
                    success=True,
                    original_value=decision.new_value,
                    fixed_value=text,
                )
            else:
                return FixResult(
                    path=decision.path,
                    success=False,
                    original_value=decision.new_value,
                    error="No poison terms found to replace",
                )

        except Exception as e:
            logger.error(f"Aggressive fix failed for {decision.path}: {e}")
            return FixResult(
                path=decision.path,
                success=False,
                original_value=decision.new_value,
                error=str(e),
            )


# =============================================================================
# MAIN REPAIR LOOP
# =============================================================================

class LeafRepairLoop:
    """
    Orchestrates the Validate → Fix → Re-validate cycle with ESCALATING STRATEGIES.

    STRATEGY ESCALATION:
    1. STANDARD: Use LeafFixer (semantic LLM fix)
    2. SMART RETRY: Use build_targeted_retry_prompt (tells LLM what went wrong)
    3. AGGRESSIVE: Direct string replacement (no LLM, brute force)

    Flow per iteration:
    1. VALIDATE: Run all validators on decisions
    2. CHECK: If no blockers, we're done
    3. CATEGORIZE: Group issues by path, check repair history
    4. FIX: Apply appropriate strategy based on history
       - First attempt → STANDARD
       - Second attempt → SMART RETRY
       - Third attempt → AGGRESSIVE
    5. UPDATE: Apply fixes back to decisions
    6. RE-VALIDATE: Run validators again
    7. REPEAT: Up to max_iterations

    Stops when:
    - No blockers remain (success)
    - Max iterations reached (partial success)
    - All paths exhausted strategies (stuck)
    """

    def __init__(self, max_iterations: int = 3):
        """
        Args:
            max_iterations: Maximum repair attempts before giving up
        """
        self.max_iterations = max_iterations
        self.validator = LeafValidator()
        self.standard_fixer = LeafFixer()
        self.smart_fixer = SmartRetryFixer()
        self.aggressive_fixer = AggressiveFixer()

        # Track repair history per path
        self.repair_history: Dict[str, PathRepairHistory] = {}

    def _get_strategy_for_path(self, path: str) -> str:
        """Determine which strategy to use based on previous attempts."""
        if path not in self.repair_history:
            return "standard"

        history = self.repair_history[path]
        attempts = len(history.attempts)

        if attempts == 0:
            return "standard"
        elif attempts == 1:
            return "smart_retry"
        else:
            return "aggressive"

    def _update_history(
        self,
        path: str,
        original_value: str,
        new_value: str,
        strategy: str,
        success: bool,
        failures: List[str],
    ):
        """Update repair history for a path."""
        if path not in self.repair_history:
            self.repair_history[path] = PathRepairHistory(
                path=path,
                original_value=original_value,
            )

        self.repair_history[path].add_attempt(strategy, new_value, success, failures)

    def _get_failures_for_path(
        self,
        path: str,
        issues: List[ValidationIssue],
    ) -> List[str]:
        """Get failure messages for a specific path."""
        return [
            issue.message
            for issue in issues
            if issue.path == path
        ]

    def _get_decision_for_path(
        self,
        path: str,
        decisions: List[DecisionResult],
    ) -> Optional[DecisionResult]:
        """Get decision for a specific path."""
        for d in decisions:
            if d.path == path:
                return d
        return None

    @traceable(name="leaf_repair_loop_smart")
    async def run(
        self,
        decisions: List[DecisionResult],
        context: AdaptationContext,
    ) -> RepairLoopResult:
        """
        Run the complete repair loop with escalating strategies.

        Args:
            decisions: List of DecisionResult from adaptation
            context: AdaptationContext with validation/fix context

        Returns:
            RepairLoopResult with final decisions and stats
        """
        iterations = []
        total_fixes_attempted = 0
        total_fixes_succeeded = 0
        smart_retries_used = 0
        aggressive_fixes_used = 0
        initial_blockers = None

        # Reset repair history
        self.repair_history = {}

        for iteration in range(1, self.max_iterations + 1):
            logger.info(f"{'='*60}")
            logger.info(f"REPAIR LOOP - Iteration {iteration}/{self.max_iterations}")
            logger.info(f"{'='*60}")

            # Step 1: VALIDATE
            validation_result = await self.validator.validate_all(decisions, context)
            blockers_before = validation_result.blockers

            if initial_blockers is None:
                initial_blockers = blockers_before

            logger.info(f"Blockers: {blockers_before}, Warnings: {validation_result.warnings}")

            # Step 2: CHECK - No blockers = success
            if blockers_before == 0:
                logger.info("✓ No blockers found - repair loop complete!")
                iterations.append(RepairIteration(
                    iteration=iteration,
                    validation_result=validation_result,
                    blockers_before=0,
                    blockers_after=0,
                    strategy="none",
                ))
                break

            # Step 3: CATEGORIZE issues by path and determine strategy
            blocker_issues = get_blocker_issues(validation_result)
            paths_to_fix = set(issue.path for issue in blocker_issues)

            logger.info(f"Paths with blockers: {len(paths_to_fix)}")

            # Step 4: FIX - Apply appropriate strategy per path
            fix_results = []
            iteration_smart_retries = 0
            iteration_aggressive = 0
            iteration_strategy = "standard"

            for path in paths_to_fix:
                strategy = self._get_strategy_for_path(path)
                decision = self._get_decision_for_path(path, decisions)
                failures = self._get_failures_for_path(path, blocker_issues)

                if not decision:
                    continue

                logger.info(f"  {path}: strategy={strategy}")

                if strategy == "standard":
                    # Use standard fixer via fix_all (batched)
                    path_issues = [i for i in blocker_issues if i.path == path]
                    result = await self.standard_fixer.fix_all(path_issues, context)
                    if result.fix_results:
                        fix_results.extend(result.fix_results)
                        for r in result.fix_results:
                            self._update_history(path, decision.old_value, r.fixed_value or decision.new_value, "standard", r.success, failures)

                elif strategy == "smart_retry":
                    # Use smart retry with targeted prompt
                    result = await self.smart_fixer.fix(decision, context, failures)
                    fix_results.append(result)
                    self._update_history(path, decision.old_value, result.fixed_value or decision.new_value, "smart_retry", result.success, failures)
                    iteration_smart_retries += 1
                    smart_retries_used += 1
                    iteration_strategy = "smart_retry"

                else:  # aggressive
                    # Use aggressive fixer (no LLM)
                    result = self.aggressive_fixer.fix(decision, context)
                    fix_results.append(result)
                    self._update_history(path, decision.old_value, result.fixed_value or decision.new_value, "aggressive", result.success, failures)
                    iteration_aggressive += 1
                    aggressive_fixes_used += 1
                    iteration_strategy = "aggressive"

            # Count successes
            succeeded = sum(1 for r in fix_results if r.success)
            total_fixes_attempted += len(fix_results)
            total_fixes_succeeded += succeeded

            # Step 5: UPDATE - Apply fixes back to decisions
            if succeeded > 0:
                decisions = apply_fixes_to_decisions(decisions, fix_results)
                logger.info(f"Applied {succeeded} fixes to decisions")

            # Step 6: RE-VALIDATE to get blockers_after
            revalidation = await self.validator.validate_all(decisions, context)
            blockers_after = revalidation.blockers

            # Record iteration
            iterations.append(RepairIteration(
                iteration=iteration,
                validation_result=validation_result,
                fix_result=LeafFixerResult(
                    total_issues=len(blocker_issues),
                    fixes_attempted=len(fix_results),
                    fixes_succeeded=succeeded,
                    fixes_failed=len(fix_results) - succeeded,
                    fix_results=fix_results,
                ),
                smart_retries=iteration_smart_retries,
                aggressive_fixes=iteration_aggressive,
                blockers_before=blockers_before,
                blockers_after=blockers_after,
                strategy=iteration_strategy,
            ))

            # Log progress
            if blockers_after < blockers_before:
                logger.info(f"✓ Progress: {blockers_before} → {blockers_after} blockers")
            elif blockers_after == blockers_before:
                logger.warning(f"⚠ No progress: {blockers_before} → {blockers_after} blockers")
            else:
                logger.error(f"✗ Regression: {blockers_before} → {blockers_after} blockers")

            # Check if all blockers resolved
            if blockers_after == 0:
                logger.info(f"✓ All blockers resolved after iteration {iteration}!")
                break

        # Final validation
        final_validation = await self.validator.validate_all(decisions, context)
        final_blockers = final_validation.blockers
        passed = final_blockers == 0

        # Get unresolved paths
        unresolved_paths = [
            path for path, history in self.repair_history.items()
            if not history.resolved
        ]

        logger.info(f"{'='*60}")
        logger.info(f"REPAIR LOOP COMPLETE")
        logger.info(f"  Passed: {passed}")
        logger.info(f"  Initial blockers: {initial_blockers}")
        logger.info(f"  Final blockers: {final_blockers}")
        logger.info(f"  Smart retries used: {smart_retries_used}")
        logger.info(f"  Aggressive fixes used: {aggressive_fixes_used}")
        logger.info(f"  Unresolved paths: {len(unresolved_paths)}")
        logger.info(f"{'='*60}")

        return RepairLoopResult(
            passed=passed,
            total_iterations=len(iterations),
            initial_blockers=initial_blockers or 0,
            final_blockers=final_blockers,
            total_fixes_attempted=total_fixes_attempted,
            total_fixes_succeeded=total_fixes_succeeded,
            smart_retries_used=smart_retries_used,
            aggressive_fixes_used=aggressive_fixes_used,
            decisions=decisions,
            iterations=iterations,
            unresolved_paths=unresolved_paths,
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def run_repair_loop(
    decisions: List[DecisionResult],
    context: AdaptationContext,
    max_iterations: int = 3,
) -> RepairLoopResult:
    """
    Run the repair loop on adapted decisions with escalating strategies.

    Args:
        decisions: List of DecisionResult from adaptation
        context: AdaptationContext with validation/fix context
        max_iterations: Maximum repair attempts

    Returns:
        RepairLoopResult with final decisions and stats
    """
    loop = LeafRepairLoop(max_iterations=max_iterations)
    return await loop.run(decisions, context)


async def validate_and_fix(
    decisions: List[DecisionResult],
    context: AdaptationContext,
) -> tuple[List[DecisionResult], LeafValidationResult]:
    """
    Single validate-and-fix pass (no loop, just standard fixer).

    Args:
        decisions: List of DecisionResult from adaptation
        context: AdaptationContext with validation/fix context

    Returns:
        (updated_decisions, validation_result)
    """
    validator = LeafValidator()
    fixer = LeafFixer()

    # Validate
    validation_result = await validator.validate_all(decisions, context)

    if validation_result.blockers == 0:
        return decisions, validation_result

    # Fix blockers
    blocker_issues = get_blocker_issues(validation_result)
    fix_result = await fixer.fix_all(blocker_issues, context)

    # Apply fixes
    if fix_result.fixes_succeeded > 0:
        decisions = apply_fixes_to_decisions(decisions, fix_result.fix_results)

    # Re-validate
    final_validation = await validator.validate_all(decisions, context)

    return decisions, final_validation


async def smart_fix_single(
    decision: DecisionResult,
    context: AdaptationContext,
    failures: List[str],
) -> FixResult:
    """
    Smart fix a single decision using targeted retry prompt.

    Args:
        decision: The DecisionResult that failed
        context: AdaptationContext
        failures: List of failure reasons

    Returns:
        FixResult with fixed value
    """
    fixer = SmartRetryFixer()
    return await fixer.fix(decision, context, failures)
