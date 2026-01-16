"""
Adaptation & Validation Rules - Single Source of Truth

These rules define ALL requirements for simulation adaptation.
Prompts and validators are generated FROM these rules, not hardcoded.

Rule Structure:
- id: Unique identifier
- name: Human-readable name
- severity: blocker | major | minor | advisory | systemic
- impact: cross_slice | leaf_only | structural | read_only | system
- description: What the rule checks
- validation_criteria: Specific checks to perform
- depends_on: Rules that must pass first
"""

from typing import Optional
from dataclasses import dataclass, field


@dataclass
class ValidationCriteria:
    """Criteria for validating a rule."""
    checks: dict[str, str] = field(default_factory=dict)


@dataclass
class Rule:
    """Single validation/adaptation rule."""
    id: str
    name: str
    severity: str  # blocker, major, minor, advisory, systemic
    impact: str    # cross_slice, leaf_only, structural, read_only, system
    description: str
    validation_criteria: dict[str, str] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)
    soft_depends_on: list[str] = field(default_factory=list)
    role: Optional[str] = None


# =============================================================================
# MASTER RULES DEFINITION
# =============================================================================

RULES = [
    # -------------------------------------------------------------------------
    # CORE ENTITY & CONSISTENCY RULES
    # -------------------------------------------------------------------------
    Rule(
        id="R1",
        name="Entity Remap",
        severity="blocker",
        impact="cross_slice",
        description="Ensure all entity references are consistently mapped across the entire simulation. Zero legacy entity references after transform (org names, emails, image URLs with org names, manager names, simulation title, etc.)",
        validation_criteria={
            "leakage_scan": "No legacy entity references from original domain should remain after transformation",
            "referential_integrity": "All names/titles used in emails, activities, rubrics, helper prompts, and instructions refer to the same new org/competitor across the file"
        }
    ),
    Rule(
        id="R2",
        name="Narrative & Intent",
        severity="blocker",
        impact="read_only",
        description="Maintain narrative consistency and learning objectives throughout the simulation. Keep pedagogical structure intact while contextualizing wording and examples",
        validation_criteria={
            "instructional_consistency": "Prompts, emails, and guidelines all ask for the same deliverable in the new context",
            "pedagogical_preservation": "KLOs, rubric dimensions, reflection items remain intact"
        }
    ),
    Rule(
        id="R3",
        name="Domain Metrics Translation",
        severity="blocker",
        impact="leaf_only",
        depends_on=["R6.1", "R6.2"],
        soft_depends_on=["R6.3"],
        description="Translate domain metrics faithfully to match target industry. Ensure metrics and jargon match the target industry",
        validation_criteria={
            "domain_metrics": "At least two domain-correct metrics appear in each: (a) impact analysis, (b) KPIs section, (c) rubric examples",
            "metric_plausibility": "Magnitudes feel realistic for the domain (e.g., RevPAR shifts for hotels; load factor changes for airlines; churn delta for SaaS)"
        }
    ),
    Rule(
        id="R4",
        name="Options Schema",
        severity="blocker",
        impact="structural",
        description="Maintain exactly 4 strategic options with proper structure. Each option must include Estimated Cost, Projected Financial Impact, Benefits, Risks, Operational Considerations, Brand Alignment",
        validation_criteria={
            "option_completeness": "All four options present with cost, impact, benefits, risks, ops, brand alignment fields populated",
            "option_fidelity": "Options are meaningful analogs (value/innovation/loyalty/brand) customized to the domain's levers"
        }
    ),
    Rule(
        id="R5",
        name="Personas & Comms",
        severity="major",
        impact="leaf_only",
        description="Ensure personas and communication styles match the target domain. Replace manager persona with plausible equivalents for the new org",
        validation_criteria={
            "persona_consistency": "Simulation name, workplace scenario, emails, and persona details refer to the same organization and competitor",
            "role_appropriateness": "Roles/titles updated to industry-appropriate equivalents"
        }
    ),

    # -------------------------------------------------------------------------
    # RUBRIC RULES (R6.x)
    # -------------------------------------------------------------------------
    Rule(
        id="R6.1",
        name="Rubric Structure",
        severity="blocker",
        impact="structural",
        description="Maintain proper rubric structure with required fields"
    ),
    Rule(
        id="R6.2",
        name="Full Star Ladder",
        severity="blocker",
        impact="structural",
        description="Ensure all rubrics have complete 1-5 star rating scales"
    ),
    Rule(
        id="R6.3",
        name="Star Depth",
        severity="major",
        impact="structural_light",
        description="Maintain appropriate depth and detail in star rating descriptions"
    ),
    Rule(
        id="R6.4",
        name="Rubric Contextualization",
        severity="blocker",
        impact="leaf_only",
        depends_on=["R6.1", "R6.2"],
        description="Ensure rubric criteria are contextualized to the specific domain and scenario. Examples and evidence prompts inside the rubric reference the new scenario's data and levers",
        validation_criteria={
            "rubric_alignment": "Rubric questions, helper prompts, and resource titles all reference the new entities and domain-correct metrics",
            "evidence_specificity": "Rubric items contain concrete, scenario-specific evidence requests (named metrics, segments, figures)"
        }
    ),
    Rule(
        id="R6.5",
        name="Resource Alignment",
        severity="major",
        impact="leaf_only",
        depends_on=["R6.1", "R6.2"],
        description="Ensure resources align with rubric requirements and learning objectives. Rename resource packs to match new domain"
    ),
    Rule(
        id="R6.6",
        name="Tone & Pedagogy",
        severity="minor",
        impact="leaf_only",
        description="Maintain appropriate educational tone and pedagogical approach. Use executive, concise, domain-specific vocabulary"
    ),

    # -------------------------------------------------------------------------
    # FORMATTING & TECHNICAL RULES
    # -------------------------------------------------------------------------
    Rule(
        id="R7",
        name="Formatting/Links",
        severity="blocker",
        impact="read_only",
        role="trailing_guard",
        description="Ensure proper formatting, links, and final presentation quality. JSON must remain valid; URLs/emails should be syntactically valid",
        validation_criteria={
            "json_validity": "Parses; required keys present; no nulls where strings expected; URLs/emails well-formed",
            "html_rendering": "HTML bodies render correctly; no malformed tags"
        }
    ),
    Rule(
        id="R8",
        name="Industry Fit Validation",
        severity="blocker",
        impact="cross_slice",
        description="Validate that all content matches the target industry domain. Ensure no industry-inappropriate terms or concepts remain",
        validation_criteria={
            "domain_keywords": "Industry-specific terminology used consistently throughout",
            "concept_alignment": "Business concepts and strategies appropriate for target industry"
        }
    ),
    Rule(
        id="R9",
        name="Brand Values Alignment",
        severity="major",
        impact="leaf_only",
        description="Ensure the new org's positioning/values appear coherently in the background, options, and recommended KPIs",
        validation_criteria={
            "brand_consistency": "Brand positioning coherently expressed throughout simulation",
            "values_integration": "Company values reflected in strategic options and recommendations"
        }
    ),

    # -------------------------------------------------------------------------
    # QUALITY RULES
    # -------------------------------------------------------------------------
    Rule(
        id="Q1",
        name="Data Plausibility",
        severity="advisory",
        impact="read_only",
        description="Verify that all data points and statistics are realistic and plausible for the target domain"
    ),
    Rule(
        id="Q2",
        name="Golden Master Preservation",
        severity="blocker",
        impact="cross_slice",
        description="Ensure future adaptations perfectly preserve the approved simulation's format and pedagogy while fully contextualizing entities, metrics, and strategy levers",
        validation_criteria={
            "structure_preservation": "Keys, order, and format of the JSON preserved",
            "pedagogical_integrity": "Learning outcomes, activity flow, and rubric structure intact",
            "contextual_adaptation": "Only entities, domain language, metrics, options' industry analogs, personas, and resources changed"
        }
    ),

    # -------------------------------------------------------------------------
    # RESOURCE-SPECIFIC RULES (RSC.x)
    # -------------------------------------------------------------------------
    Rule(
        id="RSC.1",
        name="Entity Hygiene in Resources",
        severity="blocker",
        impact="cross_slice",
        description="No legacy names or links from old scenario in resources",
        validation_criteria={
            "legacy_entity_scan": "No old scenario entity references in resource content",
            "url_consistency": "All URLs and links point to new domain entities"
        }
    ),
    Rule(
        id="RSC.2",
        name="Domain-Specific KPIs",
        severity="blocker",
        impact="leaf_only",
        description="Metrics and figures match new industry domain",
        validation_criteria={
            "kpi_accuracy": "All KPIs reflect industry-appropriate metrics",
            "metric_consistency": "Financial figures realistic for target industry"
        }
    ),
    Rule(
        id="RSC.3",
        name="Narrative Alignment",
        severity="blocker",
        impact="cross_slice",
        description="Resource storyline aligns with scenario challenge",
        validation_criteria={
            "narrative_coherence": "Resource content supports the main scenario challenge",
            "storyline_consistency": "Resource narrative matches overall simulation story"
        }
    ),
    Rule(
        id="RSC.4",
        name="Structural Consistency",
        severity="blocker",
        impact="structural",
        description="Resource follows same structure as golden master",
        validation_criteria={
            "schema_compliance": "Resource structure matches expected format",
            "field_completeness": "All required fields present and properly formatted"
        }
    ),
    Rule(
        id="RSC.5",
        name="Rubric-Resource Consistency",
        severity="major",
        impact="cross_slice",
        description="Rubric and resource KPIs stay synchronized",
        validation_criteria={
            "kpi_alignment": "Resource KPIs match rubric evaluation criteria",
            "metric_synchronization": "Consistent metrics between rubric and resources"
        }
    ),
    Rule(
        id="RSC.6",
        name="Tone & Style",
        severity="minor",
        impact="leaf_only",
        description="Professional, executive brief tone",
        validation_criteria={
            "tone_consistency": "Professional executive tone maintained",
            "style_appropriateness": "Language appropriate for executive audience"
        }
    ),
    Rule(
        id="RSC.7",
        name="Formatting & Technical Validity",
        severity="blocker",
        impact="read_only",
        description="JSON + HTML clean, no syntax errors",
        validation_criteria={
            "json_validity": "Valid JSON structure with no syntax errors",
            "html_rendering": "HTML content renders correctly without malformed tags"
        }
    ),
    Rule(
        id="RSC.8",
        name="Completeness of Analysis",
        severity="blocker",
        impact="structural",
        description="Each resource option covers Cost, Impact, Benefits, Risks, Ops, Brand",
        validation_criteria={
            "analysis_completeness": "All six analysis dimensions present for each option",
            "content_depth": "Each dimension contains substantive content"
        }
    ),
    Rule(
        id="RSC.9",
        name="Data Plausibility",
        severity="advisory",
        impact="read_only",
        description="Numbers realistic for the new industry",
        validation_criteria={
            "realistic_magnitudes": "Financial figures and metrics within industry norms",
            "plausible_scenarios": "Business scenarios and outcomes are believable"
        }
    ),

    # -------------------------------------------------------------------------
    # SYSTEM/DAG RULES (Rule 0.x)
    # -------------------------------------------------------------------------
    Rule(
        id="Rule 0.1",
        name="DAG Dependency Integrity",
        severity="systemic",
        impact="system",
        description="Structural → Semantic → Global sequence enforced",
        validation_criteria={
            "execution_order": "Rules executed in proper dependency order",
            "barrier_respect": "Structural fixes complete before semantic fixes"
        }
    ),
    Rule(
        id="Rule 0.2",
        name="Write Contracts",
        severity="systemic",
        impact="system",
        description="Each agent edits only allowed JSON zones",
        validation_criteria={
            "edit_permissions": "Agents only modify their designated JSON sections",
            "zone_compliance": "No unauthorized edits outside assigned zones"
        }
    ),
    Rule(
        id="Rule 0.3",
        name="CAS Integrity",
        severity="systemic",
        impact="system",
        description="Fixers respect hash-matching; no overwrites",
        validation_criteria={
            "hash_validation": "Edits only applied to current hash-matched content",
            "no_collisions": "No concurrent edits to same content slice"
        }
    ),
    Rule(
        id="Rule 0.4",
        name="Targeted Rechecks",
        severity="systemic",
        impact="system",
        description="Dependencies correctly trigger revalidation after fixes",
        validation_criteria={
            "recheck_triggers": "Dependent rules rechecked after changes",
            "validation_completeness": "All affected rules revalidated"
        }
    ),
    Rule(
        id="Rule 0.5",
        name="Anti-Oscillation",
        severity="systemic",
        impact="system",
        description="No infinite fix/revert loops detected",
        validation_criteria={
            "loop_detection": "No endless fix-revert cycles",
            "convergence": "System stabilizes within maximum iterations"
        }
    ),
    Rule(
        id="Rule 0.6",
        name="Global Guard Compliance",
        severity="systemic",
        impact="system",
        description="Final run passes critical blockers",
        validation_criteria={
            "blocker_compliance": "All blocker rules pass at 100%",
            "final_validation": "System ready for human approval"
        }
    ),
]


# =============================================================================
# RULE ACCESS HELPERS
# =============================================================================

def get_rules_by_severity(severity: str) -> list[Rule]:
    """Get all rules of a specific severity."""
    return [r for r in RULES if r.severity == severity]


def get_blocker_rules() -> list[Rule]:
    """Get all blocker rules."""
    return get_rules_by_severity("blocker")


def get_rules_by_impact(impact: str) -> list[Rule]:
    """Get all rules with a specific impact type."""
    return [r for r in RULES if r.impact == impact]


def get_rule_by_id(rule_id: str) -> Optional[Rule]:
    """Get a rule by its ID."""
    for r in RULES:
        if r.id == rule_id:
            return r
    return None


def get_rules_for_shard(shard_id: str) -> list[Rule]:
    """Get rules relevant to a specific shard type."""
    shard_lower = shard_id.lower()

    # Map shard types to relevant rule categories
    if "resource" in shard_lower:
        return [r for r in RULES if r.id.startswith("RSC") or r.id in ["R1", "R3", "R8"]]
    elif "rubric" in shard_lower:
        return [r for r in RULES if r.id.startswith("R6") or r.id in ["R1", "R3"]]
    elif "email" in shard_lower:
        return [r for r in RULES if r.id in ["R1", "R5", "R7"]]
    elif "simulation_flow" in shard_lower:
        return [r for r in RULES if r.id in ["R1", "R2", "R4", "R6.4"]]
    elif "scenario" in shard_lower:
        return [r for r in RULES if r.id in ["R1", "R2", "R5", "R8", "R9"]]
    else:
        # Default: core rules
        return [r for r in RULES if r.id in ["R1", "R2", "R7", "R8"]]


def format_rules_for_prompt(rules: list[Rule], include_criteria: bool = True) -> str:
    """Format rules into prompt-ready text."""
    lines = []

    for rule in rules:
        severity_icon = {
            "blocker": "⛔",
            "major": "⚠️",
            "minor": "📝",
            "advisory": "💡",
            "systemic": "⚙️"
        }.get(rule.severity, "•")

        lines.append(f"\n### {severity_icon} {rule.id}: {rule.name}")
        lines.append(f"**Severity:** {rule.severity.upper()}")
        lines.append(f"**Description:** {rule.description}")

        if include_criteria and rule.validation_criteria:
            lines.append("**Validation Criteria:**")
            for key, value in rule.validation_criteria.items():
                lines.append(f"  - {key}: {value}")

    return "\n".join(lines)


def get_blocker_summary() -> str:
    """Get a summary of all blocker rules for prompts."""
    blockers = get_blocker_rules()
    lines = ["## ⛔ BLOCKER RULES — VIOLATIONS = AUTOMATIC REJECTION\n"]

    for rule in blockers:
        lines.append(f"**{rule.id} - {rule.name}:** {rule.description}")
        if rule.validation_criteria:
            for key, value in rule.validation_criteria.items():
                lines.append(f"  • {value}")
        lines.append("")

    return "\n".join(lines)


def to_json() -> dict:
    """Export all rules as JSON."""
    return {
        "rules": [
            {
                "id": r.id,
                "name": r.name,
                "severity": r.severity,
                "impact": r.impact,
                "description": r.description,
                "validation_criteria": r.validation_criteria,
                "depends_on": r.depends_on,
                "soft_depends_on": r.soft_depends_on,
                "role": r.role
            }
            for r in RULES
        ]
    }
