"""
Semantic grouping logic for JSON Contextualizer Agent.

This module groups leaves by their semantic context for parallel processing.
"""

from typing import List, Tuple, Dict, Any
import logging
import re

logger = logging.getLogger(__name__)


# Semantic group definitions based on JSON paths
# These map to the main sections of the simulation JSON
SEMANTIC_GROUPS = {
    "lesson_content": [
        r"^/topicWizardData/lessonInformation/.*",
        r"^/lessonInformation/.*",
    ],
    "assessment_criteria": [
        r"^/topicWizardData/assessmentCriterion/.*",
        r"^/topicWizardData/selectedAssessmentCriterion/.*",
        r"^/assessmentCriterion/.*",
        r"^/selectedAssessmentCriterion/.*",
    ],
    "workplace_scenario": [
        r"^/topicWizardData/workplaceScenario/.*",
        r"^/topicWizardData/simulationName$",
        r"^/topicWizardData/launchSettings/coverTab/overview$",
        r"^/workplaceScenario/.*",
    ],
    "simulation_flow": [
        r"^/topicWizardData/simulationFlow/.*",
        r"^/simulationFlow/.*",
    ],
    "industry_activities": [
        r"^/topicWizardData/industryAlignedActivities/.*",
        r"^/topicWizardData/selectedIndustryAlignedActivities/.*",
        r"^/industryAlignedActivities/.*",
    ],
    "rubrics": [
        r"^/topicWizardData/rubric/.*",
        r"^/rubric/.*",
    ],
    "resources": [
        r"^/topicWizardData/resources/.*",
        r"^/resources/.*",
    ],
}


def match_path_to_group(path: str) -> str:
    """
    Match a JSON pointer path to its semantic group.

    Args:
        path: JSON Pointer path (e.g., "/topicWizardData/lessonInformation/lesson")

    Returns:
        Group name or "other" if no match
    """
    for group_name, patterns in SEMANTIC_GROUPS.items():
        for pattern in patterns:
            if re.match(pattern, path):
                return group_name

    return "other"


def group_leaves_by_semantic_context(
    modifiable_leaves: List[Tuple[str, str]]
) -> Dict[str, List[Tuple[str, str]]]:
    """
    Group leaves by semantic context for parallel processing.

    This function groups leaves into semantic categories based on their
    JSON paths, enabling parallel processing of related content.

    Args:
        modifiable_leaves: List of (path, value) tuples for modifiable leaves

    Returns:
        Dictionary mapping group names to lists of (path, value) tuples
    """
    groups: Dict[str, List[Tuple[str, str]]] = {
        "lesson_content": [],
        "assessment_criteria": [],
        "workplace_scenario": [],
        "simulation_flow": [],
        "industry_activities": [],
        "rubrics": [],
        "resources": [],
        "other": [],
    }

    # Group each leaf by its semantic context
    for path, value in modifiable_leaves:
        group = match_path_to_group(path)
        groups[group].append((path, value))

    # Log grouping statistics
    for group_name, leaves in groups.items():
        if leaves:
            logger.info(f"Group '{group_name}': {len(leaves)} leaves")

    # Remove empty groups
    groups = {k: v for k, v in groups.items() if v}

    return groups


def get_group_summary(grouped_leaves: Dict[str, List[Tuple[str, str]]]) -> Dict[str, Any]:
    """
    Get summary statistics for grouped leaves.

    Args:
        grouped_leaves: Dictionary of grouped leaves

    Returns:
        Summary statistics including counts and sample paths
    """
    summary = {}

    for group_name, leaves in grouped_leaves.items():
        if leaves:
            summary[group_name] = {
                "count": len(leaves),
                "sample_paths": [path for path, _ in leaves[:3]],
                "total_chars": sum(len(value) for _, value in leaves),
            }

    return summary


def redistribute_large_groups(
    grouped_leaves: Dict[str, List[Tuple[str, str]]],
    max_group_size: int = 50
) -> Dict[str, List[Tuple[str, str]]]:
    """
    Redistribute large groups into smaller sub-groups for better parallelization.

    Args:
        grouped_leaves: Dictionary of grouped leaves
        max_group_size: Maximum number of leaves per group

    Returns:
        Dictionary with potentially split groups
    """
    redistributed = {}

    for group_name, leaves in grouped_leaves.items():
        if len(leaves) <= max_group_size:
            redistributed[group_name] = leaves
        else:
            # Split into sub-groups
            num_subgroups = (len(leaves) + max_group_size - 1) // max_group_size
            for i in range(num_subgroups):
                start_idx = i * max_group_size
                end_idx = min((i + 1) * max_group_size, len(leaves))
                subgroup_name = f"{group_name}_part{i+1}"
                redistributed[subgroup_name] = leaves[start_idx:end_idx]
                logger.info(f"Split '{group_name}' into '{subgroup_name}' with {end_idx - start_idx} leaves")

    return redistributed
