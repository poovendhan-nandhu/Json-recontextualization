"""
Check Definitions for Cartedo Validation Agent

Defines all Critical (blocking) and Flagged (non-blocking) checks.
Each check has a clear, plain-English description for non-technical audiences.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Any


class CheckTier(Enum):
    """Tier of validation check."""
    CRITICAL = "critical"  # Non-negotiable, blocking
    FLAGGED = "flagged"    # Non-blocking, review recommended


class CheckStatus(Enum):
    """Status of a validation check."""
    PASS = "Pass"
    FAIL = "Fail"
    WARNING = "Warning"


@dataclass
class CheckDefinition:
    """Definition of a validation check."""
    id: str                          # e.g., "C1", "F2"
    name: str                        # Short name
    ensures: str                     # What it ensures (plain English)
    tier: CheckTier                  # CRITICAL or FLAGGED
    threshold: str                   # Human-readable threshold (e.g., "100%", "≥0.85")
    threshold_value: float           # Numeric threshold (0.0-1.0)
    why_it_matters: str              # Impact if it fails
    fix_agent: str                   # Which agent should fix it
    detection_stage: str             # Which stage detects it


# =============================================================================
# CRITICAL CHECKS (Non-Negotiable, Blocking)
# =============================================================================

CRITICAL_CHECKS = [
    CheckDefinition(
        id="C1",
        name="Entity Removal",
        ensures="No original scenario references remain (company names, locations, industry terms)",
        tier=CheckTier.CRITICAL,
        threshold="100%",
        threshold_value=1.0,
        why_it_matters="Stale references break immersion and confuse learners",
        fix_agent="Semantic Fixer",
        detection_stage="Unified Checker + Global Guard",
    ),
    CheckDefinition(
        id="C2",
        name="KPI Alignment",
        ensures="Industry KPIs correctly updated (e.g., 'order accuracy' -> 'on-time departure')",
        tier=CheckTier.CRITICAL,
        threshold="100%",
        threshold_value=1.0,
        why_it_matters="Wrong KPIs make the simulation irrelevant to the target industry",
        fix_agent="Semantic Fixer",
        detection_stage="Unified Checker",
    ),
    CheckDefinition(
        id="C3",
        name="Schema Validity",
        ensures="Output JSON conforms to required simulation schema",
        tier=CheckTier.CRITICAL,
        threshold="100%",
        threshold_value=1.0,
        why_it_matters="Invalid schema breaks simulation execution",
        fix_agent="Structural Fixer",
        detection_stage="Sharder + Unified Checker",
    ),
    CheckDefinition(
        id="C4",
        name="Rubric Integrity",
        ensures="Rubric levels, scoring logic, and evaluation criteria preserved",
        tier=CheckTier.CRITICAL,
        threshold="100%",
        threshold_value=1.0,
        why_it_matters="Broken rubrics cause incorrect learner assessment",
        fix_agent="Structural Fixer",
        detection_stage="Unified Checker",
    ),
    CheckDefinition(
        id="C5",
        name="End-to-End Executability",
        ensures="Simulation executes from start to finish without missing references",
        tier=CheckTier.CRITICAL,
        threshold="100%",
        threshold_value=1.0,
        why_it_matters="Non-executable simulations cannot be deployed",
        fix_agent="Finisher + Global Guard",
        detection_stage="Finisher",
    ),
    CheckDefinition(
        id="C6",
        name="Barrier Compliance",
        ensures="Locked structural elements were never modified after barrier lock",
        tier=CheckTier.CRITICAL,
        threshold="100%",
        threshold_value=1.0,
        why_it_matters="Barrier violations can corrupt simulation structure",
        fix_agent="Structural Fixer",
        detection_stage="Structural Fixer + Finisher",
    ),
    # NOTE: C7 is now consolidated into UnifiedKLOValidator (src/validators/klo_validator.py)
    # The unified validator checks: preservation, questions, resources, tasks alignment
    # Keeping definition for documentation and backward compatibility
    CheckDefinition(
        id="C7",
        name="KLO Preservation (Unified)",
        ensures="Key Learning Outcomes preserved and mapped to activities, questions, and resources",
        tier=CheckTier.CRITICAL,
        threshold="≥95%",
        threshold_value=0.95,
        why_it_matters="Lost KLOs defeat the educational purpose of the simulation",
        fix_agent="Alignment Fixer + UnifiedKLOValidator",
        detection_stage="UnifiedKLOValidator",
    ),
    CheckDefinition(
        id="C8",
        name="Resource Completeness",
        ensures="All referenced resources exist and contain valid content",
        tier=CheckTier.CRITICAL,
        threshold="100%",
        threshold_value=1.0,
        why_it_matters="Missing resources leave learners without required information",
        fix_agent="Semantic Fixer",
        detection_stage="Unified Checker",
    ),
]


# =============================================================================
# FLAGGED CHECKS (Non-Blocking, Quality Signals)
# =============================================================================

FLAGGED_CHECKS = [
    CheckDefinition(
        id="F1",
        name="Persona Realism",
        ensures="Character personas feel authentic to target industry",
        tier=CheckTier.FLAGGED,
        threshold="≥0.85",
        threshold_value=0.85,
        why_it_matters="Unrealistic personas reduce learner engagement",
        fix_agent="Semantic Fixer",
        detection_stage="Unified Checker",
    ),
    CheckDefinition(
        id="F2",
        name="Resource Authenticity",
        ensures="Resources resemble real-world artifacts for scenario",
        tier=CheckTier.FLAGGED,
        threshold="≥0.85",
        threshold_value=0.85,
        why_it_matters="Unrealistic resources reduce training transfer",
        fix_agent="Semantic Fixer",
        detection_stage="Unified Checker",
    ),
    CheckDefinition(
        id="F3",
        name="Narrative Coherence",
        ensures="Story flow reads naturally after recontextualization",
        tier=CheckTier.FLAGGED,
        threshold="≥0.90",
        threshold_value=0.90,
        why_it_matters="Jarring narrative breaks immersion",
        fix_agent="Semantic Fixer",
        detection_stage="Alignment Checker",
    ),
    CheckDefinition(
        id="F4",
        name="Tone Consistency",
        ensures="Professional tone maintained throughout",
        tier=CheckTier.FLAGGED,
        threshold="≥0.90",
        threshold_value=0.90,
        why_it_matters="Inconsistent tone feels unprofessional",
        fix_agent="Semantic Fixer",
        detection_stage="Unified Checker",
    ),
    CheckDefinition(
        id="F5",
        name="Data Realism",
        ensures="Numbers, dates, and statistics are plausible for scenario",
        tier=CheckTier.FLAGGED,
        threshold="≥0.85",
        threshold_value=0.85,
        why_it_matters="Unrealistic data undermines credibility",
        fix_agent="Semantic Fixer",
        detection_stage="Unified Checker",
    ),
    CheckDefinition(
        id="F6",
        name="Industry Terminology",
        ensures="Correct industry jargon used consistently",
        tier=CheckTier.FLAGGED,
        threshold="≥0.90",
        threshold_value=0.90,
        why_it_matters="Wrong terminology signals inauthenticity",
        fix_agent="Semantic Fixer",
        detection_stage="Unified Checker",
    ),
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_check_by_id(check_id: str) -> Optional[CheckDefinition]:
    """Get a check definition by ID."""
    all_checks = CRITICAL_CHECKS + FLAGGED_CHECKS
    for check in all_checks:
        if check.id == check_id:
            return check
    return None


def get_critical_check_ids() -> list[str]:
    """Get list of critical check IDs."""
    return [c.id for c in CRITICAL_CHECKS]


def get_flagged_check_ids() -> list[str]:
    """Get list of flagged check IDs."""
    return [c.id for c in FLAGGED_CHECKS]
