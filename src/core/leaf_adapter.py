"""
Leaf Adapter - Coordinates the leaf-based adaptation workflow.

This is the main orchestrator that:
1. Extracts full context (company, KLOs, resources, industry) - Gemini
2. Indexes all leaves in the JSON
3. INDEX FOR RAG (optional) - Index leaves for retrieval
4. RETRIEVE RAG CONTEXT - Get similar examples for ICL
5. Pre-filters leaves (skip IDs, URLs, etc.)
6. Sends candidates to LLM with SMART PROMPTS + RAG (validation rules built-in) - Gemini
7. Validates adapted leaves (5 validators) - GPT 5.2
8. Runs repair loop (Validate -> Fix -> Re-validate) - GPT 5.2
9. Applies patches based on LLM decisions
10. Generates feedback report - GPT 5.2

Model Split:
- Gemini: Adaptation (context extraction, leaf decisions)
- GPT 5.2: Validation, Fixing, Feedback

RAG:
- Uses ChromaDB to index leaves by semantic group
- Retrieves similar examples for in-context learning (ICL)
- Helps LLM understand patterns from similar simulations

Usage:
    from src.core.leaf_adapter import adapt_json_with_leaves

    result = await adapt_json_with_leaves(
        input_json=my_json,
        factsheet=factsheet,
        target_scenario="New beverage company scenario...",
        use_rag=True,
    )
    adapted_json = result.adapted_json
    report = result.feedback_report
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging
import copy
import time
import re

from langsmith import traceable

from .indexer import index_leaves, get_leaf_stats
from .decider import LeafDecider, DecisionResult, get_decision_stats, get_changes_only
from .context import AdaptationContext, extract_adaptation_context
from .smart_prompts import check_poison_terms, build_targeted_retry_prompt
from .leaf_validators import LeafValidator, LeafValidationResult, validate_leaf_decisions
from .leaf_fixers import LeafFixer, apply_fixes_to_decisions
from .leaf_repair_loop import LeafRepairLoop, RepairLoopResult, run_repair_loop
from .feedback_agent import FeedbackAgent, FeedbackReport, generate_feedback_report
from .leaf_rag import LeafRAG, LeafRAGResult, get_rag_context_for_adaptation_parallel
from ..utils.patcher import PatchOp, get_patcher

logger = logging.getLogger(__name__)


# =============================================================================
# POST-PATCH INTEGRITY SCAN
# =============================================================================
# Critical learner-facing paths that should be scanned even if "kept"
LEARNER_FACING_PATHS = [
    r".*/guidelines/text$",
    r".*/guidelines/purpose$",
    r".*/taskEmail/body$",
    r".*/taskEmail/subject$",
    r".*/email/body$",
    r".*/secondaryTaskEmail/body$",
    r".*/resource/markdownText$",
    r".*/resource/title$",
    r".*/keyLearningOutcome$",
    r".*/scenarioOptions/.*/option$",
    r".*/workplaceScenario/.*",
    r".*/lessonInformation/lesson$",
    r"launchSettings/coverTab/overview$",
]
_LEARNER_FACING_COMPILED = [re.compile(p, re.IGNORECASE) for p in LEARNER_FACING_PATHS]


@dataclass
class IntegrityIssue:
    """An issue found during post-patch integrity scan."""
    path: str
    issue_type: str  # "tag_stub", "invalid_kpi", "placeholder", "poison_term"
    description: str
    severity: str = "blocker"  # "blocker" or "warning"


def scan_final_json_integrity(
    json_data: Dict[str, Any],
    context: AdaptationContext,
) -> List[IntegrityIssue]:
    """
    Scan the final JSON for learner-facing content issues.

    This catches issues that slipped through because:
    - A leaf was incorrectly "kept" instead of replaced
    - A leaf was skipped by pre-filter
    - Validation only ran on changed leaves

    Checks for:
    1. Tag stubs (<p>, <strong>, etc.)
    2. Invalid KPIs for target industry (CAC, ARR, etc.)
    3. Bracket placeholders ([Your Name], etc.)
    4. Poison terms (old company names, etc.)

    Args:
        json_data: The final patched JSON
        context: AdaptationContext with poison terms and invalid KPIs

    Returns:
        List of IntegrityIssue objects
    """
    from .indexer import index_leaves

    issues = []
    all_leaves = index_leaves(json_data)

    # Tag stubs to detect
    TAG_STUBS = {"<p>", "</p>", "<ol>", "</ol>", "<ul>", "</ul>",
                 "<li>", "</li>", "<strong>", "</strong>", "<em>", "</em>",
                 "<div>", "</div>", "<h1>", "</h1>", "<h2>", "</h2>", "<h3>", "</h3>"}

    # Bracket placeholder pattern
    PLACEHOLDER_PATTERN = re.compile(r'\[(?:Your|Insert|Enter|Add|Company|Name|Date|Title)[^\]]*\]', re.IGNORECASE)

    # Invalid KPIs from context
    invalid_kpis = set(kpi.lower() for kpi in (context.invalid_kpis or []))

    # Poison terms from context
    poison_terms = set(term.lower() for term in (context.poison_terms or []) if len(term) >= 4)

    for path, value in all_leaves:
        if not isinstance(value, str):
            continue

        value_stripped = value.strip()
        value_lower = value.lower()

        # Only scan learner-facing paths for detailed checks
        is_learner_facing = any(p.match(path) for p in _LEARNER_FACING_COMPILED)

        # Check 1: Tag stubs (scan all paths - these are always wrong)
        if value_stripped.lower() in {s.lower() for s in TAG_STUBS}:
            issues.append(IntegrityIssue(
                path=path,
                issue_type="tag_stub",
                description=f"Tag stub found: '{value_stripped}'",
                severity="blocker",
            ))
            continue

        # For remaining checks, only scan learner-facing paths
        if not is_learner_facing:
            continue

        # Check 2: Invalid KPIs
        for kpi in invalid_kpis:
            if kpi in value_lower:
                # Make sure it's a word boundary match
                if re.search(rf'\b{re.escape(kpi)}\b', value_lower):
                    issues.append(IntegrityIssue(
                        path=path,
                        issue_type="invalid_kpi",
                        description=f"Invalid KPI '{kpi}' found in learner-facing content",
                        severity="blocker",
                    ))
                    break

        # Check 3: Bracket placeholders
        placeholder_match = PLACEHOLDER_PATTERN.search(value)
        if placeholder_match:
            issues.append(IntegrityIssue(
                path=path,
                issue_type="placeholder",
                description=f"Placeholder found: '{placeholder_match.group()}'",
                severity="blocker",
            ))

        # Check 4: Poison terms (only for longer content to avoid false positives)
        if len(value) > 50:
            for term in poison_terms:
                if term in value_lower:
                    # Check word boundary
                    if re.search(rf'\b{re.escape(term)}\b', value_lower):
                        issues.append(IntegrityIssue(
                            path=path,
                            issue_type="poison_term",
                            description=f"Poison term '{term}' found in learner-facing content",
                            severity="blocker",
                        ))
                        break

    if issues:
        logger.warning(f"[INTEGRITY] Found {len(issues)} issues in final JSON")
        for issue in issues[:5]:
            logger.warning(f"  - [{issue.issue_type}] {issue.path}: {issue.description}")
        if len(issues) > 5:
            logger.warning(f"  ... and {len(issues) - 5} more issues")

    return issues


@dataclass
class AdaptationResult:
    """Result of leaf-based adaptation with full pipeline."""
    adapted_json: Dict[str, Any]
    total_leaves: int
    pre_filtered: int
    llm_evaluated: int
    changes_made: int
    kept_unchanged: int
    time_ms: int
    decisions: List[DecisionResult] = field(default_factory=list)

    # Validation results (GPT 5.2)
    validation_result: Optional[LeafValidationResult] = None
    blockers_found: int = 0
    warnings_found: int = 0

    # Repair loop results (GPT 5.2)
    repair_result: Optional[RepairLoopResult] = None
    repair_iterations: int = 0
    fixes_succeeded: int = 0

    # Feedback report (GPT 5.2)
    feedback_report: Optional[FeedbackReport] = None

    # Post-patch integrity scan
    integrity_issues: List[IntegrityIssue] = field(default_factory=list)

    # Final status
    passed: bool = True
    release_decision: str = "Approved"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_leaves": self.total_leaves,
            "pre_filtered": self.pre_filtered,
            "llm_evaluated": self.llm_evaluated,
            "changes_made": self.changes_made,
            "kept_unchanged": self.kept_unchanged,
            "time_ms": self.time_ms,
            "success_rate": self.changes_made / max(1, self.llm_evaluated) if self.llm_evaluated > 0 else 1.0,
            # Validation
            "blockers_found": self.blockers_found,
            "warnings_found": self.warnings_found,
            # Repair
            "repair_iterations": self.repair_iterations,
            "fixes_succeeded": self.fixes_succeeded,
            # Integrity
            "integrity_issues": len(self.integrity_issues),
            # Final
            "passed": self.passed,
            "release_decision": self.release_decision,
        }


class LeafAdapter:
    """
    Main adapter that coordinates leaf-based JSON adaptation.

    FULL PIPELINE:
    1. CONTEXT: Extract full adaptation context (Gemini)
    2. INDEX: Extract all leaf paths from JSON
    3. PRE-FILTER: Skip IDs, URLs, empty strings (no LLM)
    4. DECIDE: LLM decisions with SMART PROMPTS (Gemini)
    5. VALIDATE: Run 5 validators (GPT 5.2)
    6. REPAIR: Validate -> Fix -> Re-validate loop (GPT 5.2)
    7. PATCH: Apply all "replace" decisions
    8. FEEDBACK: Generate canonical report (GPT 5.2)

    Model Split:
    - Gemini: Adaptation (fast, cheap)
    - GPT 5.2: Validation, Fixing, Feedback (accurate)
    """

    def __init__(
        self,
        factsheet: Dict[str, Any],
        target_scenario: str,
        source_scenario: str = "",
        context: Optional[AdaptationContext] = None,
        max_repair_iterations: int = 3,
        generate_report: bool = True,
        use_rag: bool = True,
        simulation_id: str = "current",
        industry: str = "unknown",
    ):
        """
        Args:
            factsheet: Global factsheet with entity mappings
            target_scenario: Target scenario description
            source_scenario: Source scenario description
            context: Pre-extracted AdaptationContext (optional)
            max_repair_iterations: Maximum repair loop iterations
            generate_report: Whether to generate feedback report
            use_rag: Whether to use RAG for similar examples
            simulation_id: Unique ID for this simulation (for RAG)
            industry: Industry of the simulation (for RAG filtering)
        """
        self.factsheet = factsheet
        self.target_scenario = target_scenario
        self.source_scenario = source_scenario
        self.context = context
        self.max_repair_iterations = max_repair_iterations
        self.generate_report = generate_report
        self.use_rag = use_rag
        self.simulation_id = simulation_id
        self.industry = industry
        self.patcher = get_patcher()
        self.rag = LeafRAG() if use_rag else None

    @traceable(name="leaf_adapter_adapt")
    async def adapt(self, input_json: Dict[str, Any]) -> AdaptationResult:
        """
        Adapt JSON using leaf-based approach with full pipeline.

        Pipeline:
        1. Context extraction (Gemini)
        2. Leaf indexing
        2B. RAG index & retrieve
        3. LLM decisions (Gemini)
        4. Validation (GPT 5.2)
        5. Repair loop (GPT 5.2)
        6. Patch application
        7. Feedback report (GPT 5.2)

        Args:
            input_json: The JSON to adapt

        Returns:
            AdaptationResult with adapted JSON, validation, and report
        """
        start_time = time.time()

        # Make a deep copy to avoid modifying original
        working_json = copy.deepcopy(input_json)

        # =====================================================================
        # STEP 1: CONTEXT EXTRACTION (Gemini)
        # =====================================================================
        logger.info("=" * 60)
        logger.info("STEP 1: Extracting adaptation context (Gemini)...")
        if self.context is None:
            self.context = await extract_adaptation_context(
                input_json=input_json,
                target_scenario=self.target_scenario,
                source_scenario=self.source_scenario,
            )

        # Merge factsheet into context (adds canonical names, invalid KPIs, etc.)
        self.context.apply_factsheet(self.factsheet)

        logger.info(f"Context: {len(self.context.poison_terms)} poison terms, "
                   f"{len(self.context.klo_terms)} KLO terms, "
                   f"{len(self.context.invalid_kpis)} invalid KPIs")

        # =====================================================================
        # STEP 2: INDEX LEAVES
        # =====================================================================
        logger.info("=" * 60)
        logger.info("STEP 2: Indexing leaves...")
        all_leaves = index_leaves(working_json)
        leaf_stats = get_leaf_stats(all_leaves)
        logger.info(f"Indexed {len(all_leaves)} leaves: {leaf_stats}")

        # =====================================================================
        # STEP 2B: RAG - INDEX & RETRIEVE (Optional)
        # =====================================================================
        rag_context = {}
        if self.use_rag and self.rag and self.rag.available:
            logger.info("=" * 60)
            logger.info("STEP 2B: RAG - Indexing leaves and retrieving similar examples...")

            # Index current leaves for future retrieval (PARALLEL - much faster!)
            rag_result = await self.rag.index_leaves_parallel(
                leaves=all_leaves,
                simulation_id=self.simulation_id,
                industry=self.industry,
                clear_existing=True,
            )
            logger.info(f"RAG indexed: {rag_result.leaves_indexed} leaves (parallel)")

            # Retrieve similar examples for ICL (PARALLEL - much faster!)
            rag_context = await get_rag_context_for_adaptation_parallel(
                target_scenario=self.target_scenario,
                groups=["questions", "resources", "rubrics", "scenarios", "klos"],
                n_per_group=2,
                industry=self.industry,
            )
            logger.info(f"RAG retrieved context for {len(rag_context)} groups (parallel)")
        else:
            logger.info("STEP 2B: RAG skipped (disabled or unavailable)")

        # =====================================================================
        # STEP 3: LLM DECISIONS (Gemini)
        # =====================================================================
        logger.info("=" * 60)
        logger.info("STEP 3: Getting LLM decisions with smart prompts (Gemini)...")
        decider = LeafDecider(context=self.context, rag_context=rag_context)
        decisions = await decider.decide_all(all_leaves)

        stats = get_decision_stats(decisions)
        logger.info(f"Decisions: {stats}")

        # =====================================================================
        # STEP 4: VALIDATION (GPT 5.2)
        # =====================================================================
        logger.info("=" * 60)
        logger.info("STEP 4: Running validators (GPT 5.2)...")
        validation_result = await validate_leaf_decisions(decisions, self.context)
        logger.info(f"Validation: {validation_result.blockers} blockers, "
                   f"{validation_result.warnings} warnings")

        # =====================================================================
        # STEP 5: REPAIR LOOP (GPT 5.2)
        # =====================================================================
        repair_result = None
        if validation_result.blockers > 0 and self.max_repair_iterations > 0:
            logger.info("=" * 60)
            logger.info(f"STEP 5: Running repair loop (GPT 5.2, max {self.max_repair_iterations} iterations)...")
            repair_result = await run_repair_loop(
                decisions=decisions,
                context=self.context,
                max_iterations=self.max_repair_iterations,
            )
            decisions = repair_result.decisions  # Updated decisions
            logger.info(f"Repair: {repair_result.initial_blockers} -> {repair_result.final_blockers} blockers, "
                       f"{repair_result.total_fixes_succeeded} fixes succeeded")
        else:
            logger.info("STEP 5: Skipping repair loop (no blockers)")

        # =====================================================================
        # STEP 6: APPLY PATCHES
        # =====================================================================
        logger.info("=" * 60)
        logger.info("STEP 6: Applying patches...")
        changes = get_changes_only(decisions)

        patches = []
        for decision in changes:
            patch = PatchOp(
                op="replace",
                path=decision.path,
                value=decision.new_value,
                old_value=decision.old_value,
                reason=decision.reason,
            )
            patches.append(patch)

        if patches:
            patch_result = self.patcher.apply_patches(
                working_json,
                patches,
                validate_first=True,
                stop_on_error=False,
            )
            working_json = patch_result.patched_data
            applied_count = len(patch_result.applied_patches)

            if patch_result.failed_patches:
                logger.warning(f"{len(patch_result.failed_patches)} patches failed")
        else:
            applied_count = 0

        logger.info(f"Applied {applied_count} patches")

        # =====================================================================
        # STEP 6.5: POST-PATCH INTEGRITY SCAN
        # =====================================================================
        logger.info("=" * 60)
        logger.info("STEP 6.5: Running post-patch integrity scan...")
        integrity_issues = scan_final_json_integrity(working_json, self.context)
        logger.info(f"Integrity scan: {len(integrity_issues)} issues found")

        # If integrity issues found, they affect the pass status
        integrity_blockers = [i for i in integrity_issues if i.severity == "blocker"]

        # =====================================================================
        # STEP 7: FEEDBACK REPORT (GPT 5.2)
        # =====================================================================
        time_ms = int((time.time() - start_time) * 1000)
        feedback_report = None

        if self.generate_report:
            logger.info("=" * 60)
            logger.info("STEP 7: Generating feedback report (GPT 5.2)...")

            # Get final validation state
            final_validation = validation_result
            if repair_result:
                # Re-validate after repairs
                final_validation = await validate_leaf_decisions(decisions, self.context)

            feedback_report = await generate_feedback_report(
                decisions=decisions,
                context=self.context,
                validation_result=final_validation,
                repair_result=repair_result,
                total_leaves=len(all_leaves),
                time_ms=time_ms,
            )
            logger.info(f"Report: {feedback_report.release_decision}")

        # =====================================================================
        # FINAL RESULT
        # =====================================================================
        time_ms = int((time.time() - start_time) * 1000)

        # Determine final status
        passed = validation_result.passed
        if repair_result:
            passed = repair_result.passed

        # Integrity blockers also fail the pass
        if integrity_blockers:
            passed = False
            logger.warning(f"[INTEGRITY] {len(integrity_blockers)} integrity blockers found - marking as failed")

        release_decision = "Approved" if passed else "Fix Required"
        if feedback_report:
            release_decision = feedback_report.release_decision

        # Override feedback decision if integrity blockers found
        if integrity_blockers and release_decision == "Approved":
            release_decision = "Fix Required"

        logger.info("=" * 60)
        logger.info(f"COMPLETE: {time_ms}ms, {applied_count} changes, "
                   f"{len(integrity_issues)} integrity issues, {release_decision}")
        logger.info("=" * 60)

        return AdaptationResult(
            adapted_json=working_json,
            total_leaves=len(all_leaves),
            pre_filtered=stats["pre_filtered"],
            llm_evaluated=stats["llm_decided"],
            changes_made=applied_count,
            kept_unchanged=stats["keep"],
            time_ms=time_ms,
            decisions=decisions,
            # Validation
            validation_result=validation_result,
            blockers_found=validation_result.blockers,
            warnings_found=validation_result.warnings,
            # Repair
            repair_result=repair_result,
            repair_iterations=repair_result.total_iterations if repair_result else 0,
            fixes_succeeded=repair_result.total_fixes_succeeded if repair_result else 0,
            # Feedback
            feedback_report=feedback_report,
            # Integrity
            integrity_issues=integrity_issues,
            # Final
            passed=passed,
            release_decision=release_decision,
        )


async def adapt_json_with_leaves(
    input_json: Dict[str, Any],
    factsheet: Dict[str, Any],
    target_scenario: str,
    source_scenario: str = "",
    context: Optional[AdaptationContext] = None,
    max_repair_iterations: int = 3,
    generate_report: bool = True,
    use_rag: bool = True,
    simulation_id: str = "current",
    industry: str = "unknown",
) -> AdaptationResult:
    """
    Convenience function to adapt JSON using leaf-based approach with full pipeline.

    Pipeline:
    1. Context extraction (Gemini)
    2. Leaf indexing
    2B. RAG - Index & retrieve similar examples (optional)
    3. Leaf decisions with RAG context (Gemini)
    4. Validation (GPT 5.2)
    5. Repair loop (GPT 5.2)
    6. Patch application
    7. Feedback report (GPT 5.2)

    Args:
        input_json: The JSON to adapt
        factsheet: Global factsheet with entity mappings
        target_scenario: Target scenario description
        source_scenario: Source scenario description
        context: Pre-extracted AdaptationContext (optional)
        max_repair_iterations: Maximum repair loop iterations
        generate_report: Whether to generate feedback report
        use_rag: Whether to use RAG for similar examples
        simulation_id: Unique ID for RAG indexing
        industry: Industry for RAG filtering

    Returns:
        AdaptationResult with adapted JSON, validation, and report
    """
    adapter = LeafAdapter(
        factsheet=factsheet,
        target_scenario=target_scenario,
        source_scenario=source_scenario,
        context=context,
        max_repair_iterations=max_repair_iterations,
        generate_report=generate_report,
        use_rag=use_rag,
        simulation_id=simulation_id,
        industry=industry,
    )
    return await adapter.adapt(input_json)


def get_adaptation_diff(result: AdaptationResult) -> str:
    """
    Generate a human-readable diff of changes.

    Args:
        result: AdaptationResult from adaptation

    Returns:
        Formatted diff string
    """
    lines = []
    lines.append("=" * 70)
    lines.append("LEAF-BASED ADAPTATION SUMMARY")
    lines.append("=" * 70)
    lines.append(f"Total leaves indexed:  {result.total_leaves}")
    lines.append(f"Pre-filtered (skip):   {result.pre_filtered}")
    lines.append(f"LLM evaluated:         {result.llm_evaluated}")
    lines.append(f"Changes made:          {result.changes_made}")
    lines.append(f"Kept unchanged:        {result.kept_unchanged}")
    lines.append(f"Time:                  {result.time_ms}ms")
    lines.append("")

    # Show changes
    changes = get_changes_only(result.decisions)
    if changes:
        lines.append("=" * 70)
        lines.append(f"CHANGES MADE ({len(changes)})")
        lines.append("=" * 70)

        for i, decision in enumerate(changes, 1):
            lines.append(f"\n[{i}] {decision.path}")
            lines.append(f"    Reason: {decision.reason}")
            old_preview = decision.old_value[:80] + "..." if len(decision.old_value) > 80 else decision.old_value
            new_preview = decision.new_value[:80] + "..." if decision.new_value and len(decision.new_value) > 80 else decision.new_value
            lines.append(f"    Old: {old_preview}")
            lines.append(f"    New: {new_preview}")
    else:
        lines.append("\nNo changes made.")

    return "\n".join(lines)
