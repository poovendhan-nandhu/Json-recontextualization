"""
Skeleton Extractor - Extract JSON structure, clear text values, keep IDs.

This module extracts the "bones" of a simulation JSON:
- Keeps all IDs, types, and structural metadata
- Replaces text content with "__GENERATE__" placeholders
- Preserves URLs, images, and system fields

The skeleton is used for GENERATION (not adaptation) - the LLM fills in
the placeholders without seeing the original content.
"""

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


# Keys that should be preserved as-is (not replaced with __GENERATE__)
PRESERVE_KEYS = {
    # IDs and references
    'id', 'type', 'email_type', 'flow_properties',
    'workspace', 'selected_workspace_id',

    # Structural metadata
    'is_default', 'visible', 'max_word_count',
    'template', 'show_in_tour', 'show_in_flow',
    'level', 'duration',  # lesson_information constants

    # Counts and indices
    'order', 'index', 'position',

    # Boolean flags
    'is_locked', 'is_required', 'is_optional',
    'enabled', 'disabled', 'active',
}

# Keys that contain URLs/media - preserve these
URL_KEYS = {
    'avatar_url', 'image', 'url', 'video_url', 'image_url',
    'thumbnail', 'thumbnail_url', 'media_url', 'src',
    'organization_image', 'simulation_image',
}

# Keys that end with these suffixes should be preserved
PRESERVE_SUFFIXES = ('_id', '_ids', '_type', '_key')


def extract_skeleton(obj: Any, preserve_keys: set = None) -> Any:
    """
    Extract JSON structure, clear text values, keep IDs.

    Args:
        obj: Any JSON-serializable object
        preserve_keys: Optional set of additional keys to preserve

    Returns:
        Skeleton with same structure but text replaced with placeholders
    """
    all_preserve_keys = PRESERVE_KEYS.copy()
    if preserve_keys:
        all_preserve_keys.update(preserve_keys)

    return _extract_recursive(obj, all_preserve_keys)


def _extract_recursive(obj: Any, preserve_keys: set) -> Any:
    """Recursive helper for skeleton extraction."""

    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            # Check if key should be preserved
            if _should_preserve_key(k, preserve_keys):
                result[k] = v  # Keep as-is
            elif k in URL_KEYS:
                result[k] = v  # Keep URLs
            else:
                result[k] = _extract_recursive(v, preserve_keys)
        return result

    elif isinstance(obj, list):
        return [_extract_recursive(item, preserve_keys) for item in obj]

    elif isinstance(obj, str):
        # Replace non-empty strings with placeholder
        if obj.strip():
            return "__GENERATE__"
        return ""  # Keep empty strings as empty

    elif isinstance(obj, (int, float)):
        # Keep numbers that look like IDs or counts
        if isinstance(obj, int) and obj < 100:
            return obj  # Likely a count or index
        return "__GENERATE_NUMBER__"

    elif isinstance(obj, bool) or obj is None:
        return obj  # Keep booleans and None as-is

    else:
        return obj


def _should_preserve_key(key: str, preserve_keys: set) -> bool:
    """Check if a key should be preserved."""
    if key in preserve_keys:
        return True

    # Check suffixes
    for suffix in PRESERVE_SUFFIXES:
        if key.endswith(suffix):
            return True

    return False


def extract_structure_summary(source_json: dict) -> dict:
    """
    Extract structure counts and IDs from source JSON (no content).

    This is used as input to Stage 0 - tells the LLM how many
    KLOs, questions, rubrics, etc. to generate.

    Args:
        source_json: The source simulation JSON

    Returns:
        dict with counts and IDs for all major components
    """
    # Handle both wrapped and unwrapped formats
    topic_data = source_json.get("topicWizardData", source_json)

    # Extract KLOs
    klos = (
        topic_data.get("assessment_criterion") or
        topic_data.get("assessmentCriterion") or
        topic_data.get("selected_assessment_criterion") or
        topic_data.get("selectedAssessmentCriterion") or
        []
    )

    klo_ids = [k.get('id', '') for k in klos if isinstance(k, dict)]
    criteria_per_klo = [
        len(k.get('criterion', k.get('criteria', [])))
        for k in klos if isinstance(k, dict)
    ]

    # Extract questions from simulation_flow
    questions = _extract_questions(topic_data)
    question_ids = [q.get('id', '') for q in questions if isinstance(q, dict)]

    # Extract rubrics
    rubrics = _extract_rubrics(topic_data)
    rubric_ids = [r.get('id', '') for r in rubrics if isinstance(r, dict)]

    # Extract scenario options
    scenario_options = (
        topic_data.get("scenario_options") or
        topic_data.get("scenarioOptions") or
        []
    )

    # Extract scope of work
    scope_of_work = _extract_scope_of_work(topic_data)

    # Extract resource options
    resource_options = _extract_resource_options(topic_data)

    summary = {
        "klo_count": len(klos),
        "klo_ids": klo_ids,
        "criteria_per_klo": criteria_per_klo,
        "question_count": len(questions),
        "question_ids": question_ids,
        "rubric_count": len(rubrics),
        "rubric_ids": rubric_ids,
        "scenario_option_count": len(scenario_options),
        "resource_option_count": len(resource_options),
        "scope_of_work_count": len(scope_of_work),
    }

    logger.info(f"[SKELETON] Structure summary: {summary['klo_count']} KLOs, "
                f"{summary['question_count']} questions, {summary['rubric_count']} rubrics")

    return summary


def _extract_questions(topic_data: dict) -> list:
    """Extract all questions from simulation_flow."""
    questions = []

    sim_flow = (
        topic_data.get("simulation_flow") or
        topic_data.get("simulationFlow") or
        []
    )

    for stage in sim_flow:
        if not isinstance(stage, dict):
            continue

        data = stage.get("data", {})
        if not isinstance(data, dict):
            continue

        # Check for questions in activity data
        activity_data = data.get("activityData", data.get("activity_data", {}))
        if isinstance(activity_data, dict):
            selected = activity_data.get("selectedValue", activity_data.get("selected_value", {}))
            if isinstance(selected, dict):
                qs = selected.get("questions", [])
                if isinstance(qs, list):
                    questions.extend(qs)

        # Check for questions directly in stage children
        children = stage.get("children", [])
        for child in children:
            if isinstance(child, dict):
                child_data = child.get("data", {})
                if isinstance(child_data, dict) and "questions" in child_data:
                    questions.extend(child_data.get("questions", []))

    return questions


def _extract_rubrics(topic_data: dict) -> list:
    """Extract all rubric items from the JSON."""
    rubrics = []

    # Check top-level rubric
    top_rubric = topic_data.get("rubric", [])
    if isinstance(top_rubric, list):
        rubrics.extend(top_rubric)

    # Check simulation_flow for review rubrics
    sim_flow = (
        topic_data.get("simulation_flow") or
        topic_data.get("simulationFlow") or
        []
    )

    for stage in sim_flow:
        if not isinstance(stage, dict):
            continue

        data = stage.get("data", {})
        if not isinstance(data, dict):
            continue

        review = data.get("review", {})
        if isinstance(review, dict):
            stage_rubric = review.get("rubric", [])
            if isinstance(stage_rubric, list):
                rubrics.extend(stage_rubric)

    return rubrics


def _extract_scope_of_work(topic_data: dict) -> list:
    """Extract scope of work items from workplace_scenario."""
    ws = (
        topic_data.get("workplace_scenario") or
        topic_data.get("workplaceScenario") or
        {}
    )

    if not isinstance(ws, dict):
        return []

    lr_rm = ws.get("learner_role_reporting_manager", ws.get("learnerRoleReportingManager", {}))
    if not isinstance(lr_rm, dict):
        return []

    lr = lr_rm.get("learner_role", lr_rm.get("learnerRole", {}))
    if not isinstance(lr, dict):
        return []

    scope = lr.get("scope_of_work", lr.get("scopeOfWork", []))
    return scope if isinstance(scope, list) else []


def _extract_resource_options(topic_data: dict) -> list:
    """Extract resource options from simulation_flow."""
    options = []

    sim_flow = (
        topic_data.get("simulation_flow") or
        topic_data.get("simulationFlow") or
        []
    )

    for stage in sim_flow:
        if not isinstance(stage, dict):
            continue

        data = stage.get("data", {})
        if not isinstance(data, dict):
            continue

        resource_opts = data.get("resource_options", data.get("resourceOptions", []))
        if isinstance(resource_opts, list):
            options.extend(resource_opts)

    return options


# =============================================================================
# Utility: Get shard skeleton from full skeleton
# =============================================================================

def get_shard_skeleton(full_skeleton: dict, shard_id: str) -> dict:
    """
    Extract the skeleton for a specific shard.

    Args:
        full_skeleton: The complete skeleton with all shards
        shard_id: The shard identifier (e.g., "workplace_scenario", "simulation_flow")

    Returns:
        The skeleton for just that shard
    """
    # Handle both wrapped and unwrapped formats
    topic_data = full_skeleton.get("topicWizardData", full_skeleton)

    # Map shard_id to JSON paths
    shard_paths = {
        "lesson_information": ["lesson_information", "lessonInformation"],
        "assessment_criteria": ["assessment_criterion", "assessmentCriterion",
                               "selected_assessment_criterion", "selectedAssessmentCriterion"],
        "workplace_scenario": ["workplace_scenario", "workplaceScenario"],
        "selected_scenario": ["selected_scenario_option", "selectedScenarioOption",
                             "scenario_description", "scenarioDescription"],
        "simulation_flow": ["simulation_flow", "simulationFlow"],
        "emails": ["emails"],
        "rubrics": ["rubric"],
        "resources": ["simulation_flow", "simulationFlow"],  # Resources are inside simulation_flow
        "launch_settings": ["launch_settings", "launchSettings", "simulation_name", "overview"],
        "chat_history": ["chat_history", "chatHistory"],
    }

    paths = shard_paths.get(shard_id, [shard_id])

    for path in paths:
        if path in topic_data:
            return topic_data[path]

    logger.warning(f"[SKELETON] Shard '{shard_id}' not found in skeleton")
    return {}
