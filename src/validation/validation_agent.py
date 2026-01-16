"""
Cartedo Validation Agent

LLM-powered validation agent that produces human-readable,
decision-first validation reports for PM/Client/QA.
"""

import os
import json
import logging
from datetime import datetime
from typing import Any, Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable

from .check_definitions import CRITICAL_CHECKS, FLAGGED_CHECKS
from .check_runner import ValidationCheckRunner, AggregatedResults
from .report_generator import ValidationReportGenerator, ValidationReportData
from .report_formatter import format_markdown_report, format_json_report

logger = logging.getLogger(__name__)

# Validation agent model
VALIDATION_AGENT_MODEL = os.getenv("VALIDATION_AGENT_MODEL", "gpt-5.2")


VALIDATION_AGENT_SYSTEM_PROMPT = """You are the Cartedo Validation Agent.

Your role is to evaluate the output of the Cartedo Simulation Adaptation Framework and produce a
NON-TECHNICAL, DECISION-FIRST validation report that allows PMs, clients, QA, and prompt engineers
to quickly and confidently decide whether a simulation adaptation is ready to ship.

You MUST follow the output format and language contract exactly as defined below.

You are validating an automated scenario change where:
- Original scenario: {original_scenario}
- Target scenario: {target_scenario}
- Purpose: {simulation_purpose}
- System mode: {system_mode}

Your primary question to answer is:
"Did the system correctly convert the simulation into a realistic, working target scenario simulation without breaking anything?"

----------------------------------------------------------------
GLOBAL RULES (NON-NEGOTIABLE)
----------------------------------------------------------------
1. Do NOT expose raw JSON in the main report.
2. Do NOT use technical jargon unless absolutely necessary.
3. Write for non-technical decision makers first.
4. Be concise, factual, and confidence-oriented.
5. Use tables wherever possible.
6. All conclusions must be directly supported by validation results.
7. If something fails, clearly say:
   - What failed
   - Why it matters
   - What to fix
   - Which agent should fix it
8. Assume the reader will NOT inspect logs unless explicitly pointed to do so.
9. Treat this output as a RELEASE GATE contract.

----------------------------------------------------------------
DEFINITIONS
----------------------------------------------------------------
- One "run" = the full end-to-end conversion of a simulation into a fully adapted, ideal target simulation,
  including all internal agent passes and compliance loops.
- A Critical check is non-negotiable. If it fails, that run fails.
- A Flagged check is informational and does NOT block release.
- System acceptance requires ≥{acceptance_threshold:.0%} of runs to pass all Critical checks.

----------------------------------------------------------------
TONE AND STYLE REQUIREMENTS
----------------------------------------------------------------
- Calm, confident, neutral
- No speculation
- No defensiveness
- No excessive detail
- No internal agent arguments
- Optimize for speed of understanding

----------------------------------------------------------------
SUCCESS CONDITION
----------------------------------------------------------------
If someone reads only this report:
- They should know whether to ship or not
- They should know exactly what must be fixed
- A prompt engineer should know which agent to adjust
- No human review should be required to proceed
"""


VALIDATION_AGENT_USER_PROMPT = """Based on the validation data below, produce the canonical validation report.

## VALIDATION DATA

### Run Statistics
- Total Runs: {total_runs}
- Runs Passing All Critical Checks: {runs_passing_critical}
- Critical Pass Rate: {critical_pass_rate:.1%}
- Acceptance Threshold: {acceptance_threshold:.0%}
- Meets Acceptance: {meets_acceptance}

### Critical Check Results
{critical_checks_summary}

### Flagged Check Results
{flagged_checks_summary}

### Failed Runs
{failed_runs_summary}

### Common Failure Patterns
{failure_patterns}

---

Produce the complete validation report now, following the exact format specified.
Include all 8 sections:
1. Canonical Header
2. Executive Decision Gate
3. Critical Checks Dashboard
4. Flagged Quality Checks
5. What Failed (if any)
6. Recommended Fixes
7. Binary System Decision
8. Final One-Line Summary
"""


class ValidationAgent:
    """
    LLM-powered validation agent.

    Can operate in two modes:
    1. Rule-based: Uses check_runner and report_formatter (fast, deterministic)
    2. LLM-based: Uses GPT for more nuanced analysis (slower, more flexible)
    """

    def __init__(
        self,
        original_scenario: str,
        target_scenario: str,
        simulation_purpose: str = "Training Simulation",
        system_mode: str = "Fully automated",
        acceptance_threshold: float = 0.95,
        use_llm: bool = False,
    ):
        self.original_scenario = original_scenario
        self.target_scenario = target_scenario
        self.simulation_purpose = simulation_purpose
        self.system_mode = system_mode
        self.acceptance_threshold = acceptance_threshold
        self.use_llm = use_llm

        # Initialize components
        self.report_generator = ValidationReportGenerator(
            original_scenario=original_scenario,
            target_scenario=target_scenario,
            simulation_purpose=simulation_purpose,
            system_mode=system_mode,
            acceptance_threshold=acceptance_threshold,
        )

        if use_llm:
            self.llm = ChatOpenAI(
                model=VALIDATION_AGENT_MODEL,
                temperature=0.1,
                api_key=os.getenv("OPENAI_API_KEY"),
            )

    @traceable(name="validation_agent")
    async def validate_and_report(
        self,
        runs_data: list[dict],
        output_format: str = "markdown",
    ) -> str | dict:
        """
        Validate multiple runs and generate report.

        Args:
            runs_data: List of run data dicts
            output_format: "markdown", "json", or "slack"

        Returns:
            Formatted validation report
        """
        # Generate report data
        report_data = self.report_generator.validate_and_report(runs_data)

        if self.use_llm:
            # Use LLM for report generation
            return await self._generate_llm_report(report_data)
        else:
            # Use rule-based formatter
            if output_format == "json":
                return format_json_report(report_data)
            elif output_format == "slack":
                from .report_formatter import format_slack_report
                return format_slack_report(report_data)
            else:
                return format_markdown_report(report_data)

    async def _generate_llm_report(
        self,
        data: ValidationReportData,
    ) -> str:
        """Generate report using LLM."""

        # Format critical checks summary
        critical_summary = []
        for check in CRITICAL_CHECKS:
            agg = data.aggregated_results.check_aggregations.get(check.id, {})
            passes = agg.get("passes", 0)
            total = agg.get("total", data.total_runs)
            critical_summary.append(
                f"- {check.id} ({check.name}): {passes}/{total} passed - {check.ensures}"
            )

        # Format flagged checks summary
        flagged_summary = []
        for check in FLAGGED_CHECKS:
            agg = data.aggregated_results.check_aggregations.get(check.id, {})
            avg_score = agg.get("avg_score", 1.0)
            flagged_summary.append(
                f"- {check.id} ({check.name}): avg {avg_score:.0%} - {check.ensures}"
            )

        # Format failed runs
        failed_runs_summary = []
        for run in data.aggregated_results.run_results:
            if not run.passed_all_critical:
                failed_runs_summary.append(
                    f"- Run {run.run_id}: Failed checks: {', '.join(run.critical_failures)}"
                )

        # Build prompts
        system_prompt = VALIDATION_AGENT_SYSTEM_PROMPT.format(
            original_scenario=self.original_scenario,
            target_scenario=self.target_scenario,
            simulation_purpose=self.simulation_purpose,
            system_mode=self.system_mode,
            acceptance_threshold=self.acceptance_threshold,
        )

        user_prompt = VALIDATION_AGENT_USER_PROMPT.format(
            total_runs=data.total_runs,
            runs_passing_critical=data.aggregated_results.runs_passing_critical,
            critical_pass_rate=data.critical_pass_rate,
            acceptance_threshold=self.acceptance_threshold,
            meets_acceptance="Yes" if data.meets_threshold else "No",
            critical_checks_summary="\n".join(critical_summary),
            flagged_checks_summary="\n".join(flagged_summary),
            failed_runs_summary="\n".join(failed_runs_summary) or "None",
            failure_patterns="\n".join(data.aggregated_results.common_failure_patterns) or "None",
        )

        # Call LLM
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", user_prompt),
        ])

        chain = chat_prompt | self.llm
        result = await chain.ainvoke({})

        return result.content

    def generate_quick_summary(
        self,
        runs_data: list[dict],
    ) -> dict:
        """
        Generate a quick summary without full report.

        Returns dict with key metrics for dashboards/APIs.
        """
        report_data = self.report_generator.validate_and_report(runs_data)

        return {
            "status": "approved" if report_data.can_ship_as_is else "requires_fixes",
            "pass_rate": report_data.critical_pass_rate,
            "total_runs": report_data.total_runs,
            "passed_runs": report_data.aggregated_results.runs_passing_critical,
            "failed_runs": len(report_data.aggregated_results.failed_runs),
            "meets_threshold": report_data.meets_threshold,
            "one_line": report_data.one_line_summary,
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def validate_adaptation_runs(
    runs_data: list[dict],
    original_scenario: str,
    target_scenario: str,
    output_format: str = "markdown",
    use_llm: bool = False,
) -> str | dict:
    """
    Convenience function to validate adaptation runs.

    Args:
        runs_data: List of run data dicts, each containing:
            - run_id: str
            - run_number: int
            - adapted_json: dict
            - validation_report: dict (optional)
            - alignment_report: dict (optional)
            - fix_results: dict (optional)
            - factsheet: dict (optional)
        original_scenario: Source scenario name
        target_scenario: Target scenario name
        output_format: "markdown", "json", or "slack"
        use_llm: Whether to use LLM for report generation

    Returns:
        Formatted validation report
    """
    agent = ValidationAgent(
        original_scenario=original_scenario,
        target_scenario=target_scenario,
        use_llm=use_llm,
    )

    return await agent.validate_and_report(runs_data, output_format)


def validate_single_adaptation(
    adapted_json: dict,
    factsheet: dict,
    validation_report: dict = None,
    alignment_report: dict = None,
    original_scenario: str = "Source",
    target_scenario: str = "Target",
) -> str:
    """
    Validate a single adaptation and return quick report.

    For use in the pipeline after adaptation completes.
    """
    agent = ValidationAgent(
        original_scenario=original_scenario,
        target_scenario=target_scenario,
    )

    runs_data = [{
        "run_id": "single-run",
        "run_number": 1,
        "adapted_json": adapted_json,
        "validation_report": validation_report or {},
        "alignment_report": alignment_report or {},
        "fix_results": {},
        "factsheet": factsheet,
    }]

    report_data = agent.report_generator.validate_and_report(runs_data)
    return format_markdown_report(report_data)
