"""
Feedback Agent - Generate canonical validation report.

Runs at the END of the leaf adaptation pipeline.
Uses GPT 5.2 to generate a decision-first validation report.

The report is designed for PM/Client/QA to make ship/no-ship decisions.
"""

import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from langchain_openai import ChatOpenAI
from langsmith import traceable
import httpx

from .context import AdaptationContext
from .decider import DecisionResult, get_changes_only
from .leaf_validators import LeafValidationResult, ValidationIssue, ValidationSeverity
from .leaf_repair_loop import RepairLoopResult

logger = logging.getLogger(__name__)

# GPT 5.2 for feedback generation
FEEDBACK_MODEL = os.getenv("FEEDBACK_MODEL", "gpt-5.2-2025-12-11")


def _get_feedback_llm():
    """Get OpenAI client for feedback generation."""
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(180.0, connect=30.0),
        limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
    )
    return ChatOpenAI(
        model=FEEDBACK_MODEL,
        temperature=0.1,
        max_retries=2,
        request_timeout=180,
        http_async_client=http_client,
        api_key=os.getenv("OPENAI_API_KEY"),
    )


@dataclass
class FeedbackReport:
    """Canonical feedback report."""
    # Metadata
    original_scenario: str
    target_scenario: str
    timestamp: datetime

    # Decision
    release_decision: str  # "Approved" / "Fix Required" / "Blocked"
    can_ship: bool
    system_verdict: str

    # Stats
    total_leaves: int
    changes_made: int
    blockers_found: int
    blockers_fixed: int
    warnings_found: int

    # Repair loop
    repair_iterations: int
    repair_passed: bool

    # Report content
    markdown_report: str = ""

    def to_dict(self) -> dict:
        return {
            "original_scenario": self.original_scenario,
            "target_scenario": self.target_scenario,
            "timestamp": self.timestamp.isoformat(),
            "release_decision": self.release_decision,
            "can_ship": self.can_ship,
            "system_verdict": self.system_verdict,
            "total_leaves": self.total_leaves,
            "changes_made": self.changes_made,
            "blockers_found": self.blockers_found,
            "blockers_fixed": self.blockers_fixed,
            "warnings_found": self.warnings_found,
            "repair_iterations": self.repair_iterations,
            "repair_passed": self.repair_passed,
        }


# =============================================================================
# FEEDBACK AGENT PROMPT
# =============================================================================

FEEDBACK_AGENT_PROMPT = '''You are the Cartedo Validation Agent.

Your role is to evaluate the output of the Cartedo Simulation Adaptation Framework and produce a
NON-TECHNICAL, DECISION-FIRST validation report that allows PMs, clients, QA, and prompt engineers
to quickly and confidently decide whether a simulation adaptation is ready to ship.

You MUST follow the output format and language contract exactly as defined below.

You are validating an automated scenario change where:
- Original scenario: {original_scenario}
- Target scenario: {target_scenario}
- Purpose: {purpose}
- System mode: Fully automated (no human-in-the-loop)

Your primary question to answer is:
"Did the system correctly convert the simulation without breaking anything?"

----------------------------------------------------------------
ADAPTATION RESULTS
----------------------------------------------------------------
{adaptation_stats}

----------------------------------------------------------------
VALIDATION RESULTS
----------------------------------------------------------------
{validation_stats}

----------------------------------------------------------------
REPAIR LOOP RESULTS
----------------------------------------------------------------
{repair_stats}

----------------------------------------------------------------
ISSUES FOUND
----------------------------------------------------------------
{issues_summary}

----------------------------------------------------------------
GLOBAL RULES (NON-NEGOTIABLE)
----------------------------------------------------------------
1. Do NOT expose raw JSON in the main report
2. Do NOT use technical jargon unless absolutely necessary
3. Write for non-technical decision makers first
4. Be concise, factual, and confidence-oriented
5. Use tables wherever possible
6. All conclusions must be directly supported by validation results
7. If something fails, clearly say:
   - What failed
   - Why it matters
   - What to fix
   - Which component should fix it
8. Assume the reader will NOT inspect logs unless explicitly pointed to do so
9. Treat this output as a RELEASE GATE contract

----------------------------------------------------------------
OUTPUT FORMAT (MUST MATCH EXACTLY)
----------------------------------------------------------------

# Cartedo Simulation Adaptation
## Canonical Validation Output (Standard Contract)

### 1️⃣ Canonical Header (Contract Metadata)
Produce a table with:
- Original Scenario
- Target Scenario
- Simulation Purpose
- System Mode
- Validation Version
- Total Leaves Evaluated
- Acceptance Threshold
- Validation Timestamp

----------------------------------------------------------------

### 2️⃣ Executive Decision Gate (Single-Glance)
Produce a table with:
- Critical Pass Rate
- Acceptance Threshold Met (Yes/No)
- Blocking Issues Present (Yes/No)
- Overall Release Decision (Approved / Fix Required / Blocked)

Follow with a single-sentence **System Verdict** summarizing readiness.

----------------------------------------------------------------

### 3️⃣ Critical Checks Dashboard (Non-Negotiable)
Produce a table with columns:
- Check ID
- Check Ensures (Plain English)
- Result
- Status (Pass / Fail)
- Action Needed

Critical checks MUST include:
- No old scenario references remain
- Industry KPIs correctly replaced
- No placeholders remain
- Structure preserved

After the table, include a short **Key Insight** summarizing the dominant failure pattern (if any).

----------------------------------------------------------------

### 4️⃣ Flagged Quality Checks (Non-Blocking Signals)
Produce a table with:
- Check ID
- What This Measures
- Status
- Recommendation

These checks influence quality but do NOT block release.

----------------------------------------------------------------

### 5️⃣ What Failed (Actionable Failure Summary)
If any Critical check failed, include a clear, human-readable summary with:
- Failure Type
- Affected Count
- Example Issue (plain English)
- Why This Matters
- Fix Scope

This section MUST make it obvious how to fix the issue.

----------------------------------------------------------------

### 6️⃣ Recommended Fixes (Auto-Generated)
Produce a table with:
- Priority (P0, P1, P2)
- Recommendation
- Target Component
- Expected Impact

Ensure recommendations are scoped, actionable, and safe to automate.

----------------------------------------------------------------

### 7️⃣ Binary System Decision & Next Action
Produce a table answering:
- Can this ship as-is?
- Is the failure well-scoped?
- Is the fix isolated?
- Can automation safely rerun?

Then include a short **System Instruction** describing exactly what to do next.

----------------------------------------------------------------

### 8️⃣ Final One-Line System Summary (Canonical)
End with ONE sentence that a PM or client could quote verbatim.

----------------------------------------------------------------

Produce the report now.'''


# =============================================================================
# FEEDBACK AGENT
# =============================================================================

class FeedbackAgent:
    """
    Generates canonical validation report using GPT 5.2.

    Runs at the END of the leaf adaptation pipeline.
    Produces a decision-first report for PM/Client/QA.
    """

    def __init__(self):
        self.llm = _get_feedback_llm()

    @traceable(name="feedback_agent_generate")
    async def generate_report(
        self,
        decisions: List[DecisionResult],
        context: AdaptationContext,
        validation_result: LeafValidationResult,
        repair_result: Optional[RepairLoopResult] = None,
        total_leaves: int = 0,
        time_ms: int = 0,
    ) -> FeedbackReport:
        """
        Generate the canonical feedback report.

        Args:
            decisions: Final list of DecisionResult
            context: AdaptationContext with scenario info
            validation_result: Final validation result
            repair_result: Result from repair loop (if run)
            total_leaves: Total leaves processed
            time_ms: Total processing time

        Returns:
            FeedbackReport with markdown report
        """
        # Build stats for prompt
        changes = get_changes_only(decisions)

        adaptation_stats = f"""
- Total Leaves Indexed: {total_leaves}
- Leaves Adapted (Changed): {len(changes)}
- Leaves Kept Unchanged: {total_leaves - len(changes)}
- Processing Time: {time_ms}ms ({time_ms/1000:.1f}s)
"""

        validation_stats = f"""
- Total Validated: {validation_result.total_validated}
- Blockers Found: {validation_result.blockers}
- Warnings Found: {validation_result.warnings}
- Validation Passed: {'Yes' if validation_result.passed else 'No'}
"""

        if repair_result:
            repair_stats = f"""
- Repair Loop Ran: Yes
- Total Iterations: {repair_result.total_iterations}
- Initial Blockers: {repair_result.initial_blockers}
- Final Blockers: {repair_result.final_blockers}
- Fixes Attempted: {repair_result.total_fixes_attempted}
- Fixes Succeeded: {repair_result.total_fixes_succeeded}
- Repair Passed: {'Yes' if repair_result.passed else 'No'}
"""
        else:
            repair_stats = "- Repair Loop: Not run (no blockers found)"

        # Format issues
        issues_summary = self._format_issues(validation_result.issues)

        # Build prompt
        prompt = FEEDBACK_AGENT_PROMPT.format(
            original_scenario=context.source_scenario[:200] if context.source_scenario else "Original Simulation",
            target_scenario=context.target_scenario[:200] if context.target_scenario else "Target Simulation",
            purpose="Business Training Simulation",
            adaptation_stats=adaptation_stats,
            validation_stats=validation_stats,
            repair_stats=repair_stats,
            issues_summary=issues_summary,
        )

        try:
            # Generate report
            response = await self.llm.ainvoke(prompt)
            markdown_report = response.content if hasattr(response, 'content') else str(response)

            # Determine decision
            passed = validation_result.passed
            if repair_result:
                passed = repair_result.passed

            if passed:
                release_decision = "Approved"
                system_verdict = "System is production-ready. All critical checks passed."
            elif validation_result.blockers <= 2:
                release_decision = "Fix Required"
                system_verdict = f"System requires minor fixes. {validation_result.blockers} blocker(s) remain."
            else:
                release_decision = "Blocked"
                system_verdict = f"System blocked. {validation_result.blockers} critical issues must be resolved."

            return FeedbackReport(
                original_scenario=context.source_scenario or "Original",
                target_scenario=context.target_scenario or "Target",
                timestamp=datetime.now(),
                release_decision=release_decision,
                can_ship=passed,
                system_verdict=system_verdict,
                total_leaves=total_leaves,
                changes_made=len(changes),
                blockers_found=repair_result.initial_blockers if repair_result else validation_result.blockers,
                blockers_fixed=repair_result.total_fixes_succeeded if repair_result else 0,
                warnings_found=validation_result.warnings,
                repair_iterations=repair_result.total_iterations if repair_result else 0,
                repair_passed=repair_result.passed if repair_result else validation_result.passed,
                markdown_report=markdown_report,
            )

        except Exception as e:
            logger.error(f"Feedback agent failed: {e}")
            # Return basic report on failure
            return self._build_fallback_report(
                context, validation_result, repair_result, total_leaves, len(changes)
            )

    def _format_issues(self, issues: List[ValidationIssue]) -> str:
        """Format issues for prompt."""
        if not issues:
            return "No issues found."

        lines = []

        # Group by severity
        blockers = [i for i in issues if i.severity == ValidationSeverity.BLOCKER]
        warnings = [i for i in issues if i.severity == ValidationSeverity.WARNING]

        if blockers:
            lines.append(f"**BLOCKERS ({len(blockers)}):**")
            for i, issue in enumerate(blockers[:10], 1):
                lines.append(f"{i}. [{issue.rule_id}] {issue.message}")
                if issue.suggestion:
                    lines.append(f"   -> Suggestion: {issue.suggestion}")

        if warnings:
            lines.append(f"\n**WARNINGS ({len(warnings)}):**")
            for i, issue in enumerate(warnings[:5], 1):
                lines.append(f"{i}. [{issue.rule_id}] {issue.message}")

        return "\n".join(lines)

    def _build_fallback_report(
        self,
        context: AdaptationContext,
        validation_result: LeafValidationResult,
        repair_result: Optional[RepairLoopResult],
        total_leaves: int,
        changes_made: int,
    ) -> FeedbackReport:
        """Build fallback report if LLM fails."""
        passed = validation_result.passed
        if repair_result:
            passed = repair_result.passed

        markdown = f"""# Cartedo Simulation Adaptation
## Validation Report (Fallback)

### Executive Summary
- **Release Decision**: {'Approved' if passed else 'Fix Required'}
- **Blockers**: {validation_result.blockers}
- **Warnings**: {validation_result.warnings}

### Stats
- Total Leaves: {total_leaves}
- Changes Made: {changes_made}
- Validation Passed: {passed}

*Note: Full report generation failed. This is a fallback summary.*
"""

        return FeedbackReport(
            original_scenario=context.source_scenario or "Original",
            target_scenario=context.target_scenario or "Target",
            timestamp=datetime.now(),
            release_decision="Approved" if passed else "Fix Required",
            can_ship=passed,
            system_verdict="See fallback report above.",
            total_leaves=total_leaves,
            changes_made=changes_made,
            blockers_found=validation_result.blockers,
            blockers_fixed=0,
            warnings_found=validation_result.warnings,
            repair_iterations=repair_result.total_iterations if repair_result else 0,
            repair_passed=passed,
            markdown_report=markdown,
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def generate_feedback_report(
    decisions: List[DecisionResult],
    context: AdaptationContext,
    validation_result: LeafValidationResult,
    repair_result: Optional[RepairLoopResult] = None,
    total_leaves: int = 0,
    time_ms: int = 0,
) -> FeedbackReport:
    """
    Generate canonical feedback report.

    Args:
        decisions: Final list of DecisionResult
        context: AdaptationContext with scenario info
        validation_result: Final validation result
        repair_result: Result from repair loop (if run)
        total_leaves: Total leaves processed
        time_ms: Total processing time

    Returns:
        FeedbackReport with markdown report
    """
    agent = FeedbackAgent()
    return await agent.generate_report(
        decisions=decisions,
        context=context,
        validation_result=validation_result,
        repair_result=repair_result,
        total_leaves=total_leaves,
        time_ms=time_ms,
    )
