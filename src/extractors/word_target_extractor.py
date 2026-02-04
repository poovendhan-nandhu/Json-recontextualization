"""
Word Target Extractor - Measure word counts from source JSON.

This module measures the actual word counts in source content and
produces target ranges (±20%) for generation.

Instead of hardcoding "about_organization: 50-80 words", we measure
the source and derive dynamic targets that match the source structure.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def count_words(text: Any) -> int:
    """Count words in text. Returns 0 for non-strings."""
    if not isinstance(text, str):
        return 0
    return len(text.split())


def make_range(count: int, tolerance: float = 0.2) -> list[int]:
    """
    Create a word count range from a measured count.

    Args:
        count: Measured word count
        tolerance: Percentage tolerance (default 20%)

    Returns:
        [min_words, max_words] range
    """
    if count <= 0:
        return [50, 100]  # Default range for empty/missing content

    min_words = max(10, int(count * (1 - tolerance)))
    max_words = max(min_words + 10, int(count * (1 + tolerance)))  # Ensure max > min

    return [min_words, max_words]


def measure_word_targets(source_json: dict) -> dict:
    """
    Measure source content and return target word count ranges (±20%).

    Args:
        source_json: The source simulation JSON

    Returns:
        dict mapping field names to [min, max] word count ranges
    """
    # Handle both wrapped and unwrapped formats
    topic_data = source_json.get("topicWizardData", source_json)

    # Extract major sections
    lesson_info = (
        topic_data.get("lesson_information") or
        topic_data.get("lessonInformation") or
        {}
    )

    ws = (
        topic_data.get("workplace_scenario") or
        topic_data.get("workplaceScenario") or
        {}
    )

    bg = ws.get("background", {})
    ch = ws.get("challenge", {})

    lr_rm = ws.get("learner_role_reporting_manager", ws.get("learnerRoleReportingManager", {}))
    lr = lr_rm.get("learner_role", lr_rm.get("learnerRole", {}))
    rm = lr_rm.get("reporting_manager", lr_rm.get("reportingManager", {}))

    # Extract activity data (emails, resources, guidelines)
    activity_data = extract_activity_data(topic_data)

    # Extract KLOs for measuring
    klos = (
        topic_data.get("assessment_criterion") or
        topic_data.get("assessmentCriterion") or
        topic_data.get("selected_assessment_criterion") or
        topic_data.get("selectedAssessmentCriterion") or
        []
    )

    # Measure KLO word counts
    klo_word_counts = []
    criteria_word_counts = []
    for klo in klos:
        if isinstance(klo, dict):
            outcome = klo.get("keyLearningOutcome", klo.get("key_learning_outcome", ""))
            klo_word_counts.append(count_words(outcome))

            criteria = klo.get("criterion", klo.get("criteria", []))
            for c in criteria:
                if isinstance(c, dict):
                    criteria_word_counts.append(count_words(c.get("criteria", "")))

    avg_klo_words = sum(klo_word_counts) // len(klo_word_counts) if klo_word_counts else 25
    avg_criteria_words = sum(criteria_word_counts) // len(criteria_word_counts) if criteria_word_counts else 15

    # Measure questions
    questions = _extract_all_questions(topic_data)
    question_word_counts = [count_words(q.get("name", "")) for q in questions if isinstance(q, dict)]
    avg_question_words = sum(question_word_counts) // len(question_word_counts) if question_word_counts else 20

    # Measure rubrics
    rubrics = _extract_all_rubrics(topic_data)
    rubric_word_counts = []
    star_5_word_counts = []
    for r in rubrics:
        if isinstance(r, dict):
            rubric_word_counts.append(count_words(r.get("question", "")))
            # Get 5-star description
            descriptions = r.get("description", [])
            if isinstance(descriptions, list) and len(descriptions) >= 5:
                star_5_word_counts.append(count_words(descriptions[4] if len(descriptions) > 4 else ""))

    avg_rubric_words = sum(rubric_word_counts) // len(rubric_word_counts) if rubric_word_counts else 30
    avg_star_5_words = sum(star_5_word_counts) // len(star_5_word_counts) if star_5_word_counts else 50

    # Build word targets dictionary
    targets = {
        # Lesson information
        "lesson": make_range(count_words(lesson_info.get("lesson", ""))),

        # Workplace scenario
        "scenario": make_range(count_words(ws.get("scenario", ""))),
        "about_organization": make_range(count_words(bg.get("about_organization", bg.get("aboutOrganization", "")))),
        "industry_context": make_range(count_words(bg.get("industry_context", bg.get("industryContext", "")))),
        "current_issue": make_range(count_words(ch.get("current_issue", ch.get("currentIssue", "")))),

        # Learner role
        "role_description": make_range(count_words(lr.get("role_description", lr.get("roleDescription", "")))),
        "scope_of_work_item": make_range(50),  # Each scope item

        # Manager
        "manager_message": make_range(count_words(rm.get("message", ""))),

        # Emails
        "task_email_subject": make_range(count_words(activity_data.get("task_email", {}).get("subject", ""))),
        "task_email_body": make_range(count_words(activity_data.get("task_email", {}).get("body", ""))),
        "secondary_email_subject": make_range(count_words(activity_data.get("secondary_task_email", {}).get("subject", ""))),
        "secondary_email_body": make_range(count_words(activity_data.get("secondary_task_email", {}).get("body", ""))),

        # Resource
        "resource_markdown": make_range(count_words(activity_data.get("resource", {}).get("markdown_text", ""))),
        "resource_title": make_range(count_words(activity_data.get("resource", {}).get("title", ""))),

        # Resource options
        "resource_option_title": make_range(count_words(activity_data.get("resource_option_sample", {}).get("title", ""))),
        "resource_option_description": make_range(count_words(activity_data.get("resource_option_sample", {}).get("description", ""))),

        # Guidelines
        "guidelines": make_range(count_words(activity_data.get("guidelines", {}).get("text", ""))),
        "guidelines_purpose": make_range(count_words(activity_data.get("guidelines", {}).get("purpose", ""))),

        # KLOs and criteria (from averages)
        "klo": make_range(avg_klo_words),
        "criteria": make_range(avg_criteria_words),

        # Questions
        "question": make_range(avg_question_words),

        # Rubrics
        "rubric_question": make_range(avg_rubric_words),
        "star_5_description": make_range(avg_star_5_words),
        "star_4_description": make_range(int(avg_star_5_words * 0.7)),
        "star_3_description": make_range(int(avg_star_5_words * 0.6)),
        "star_2_description": make_range(int(avg_star_5_words * 0.5)),
        "star_1_description": make_range(int(avg_star_5_words * 0.4)),
    }

    # Log summary
    logger.info(f"[WORD TARGETS] Measured targets from source:")
    logger.info(f"  - lesson: {targets['lesson']}")
    logger.info(f"  - about_organization: {targets['about_organization']}")
    logger.info(f"  - resource_markdown: {targets['resource_markdown']}")
    logger.info(f"  - task_email_body: {targets['task_email_body']}")
    logger.info(f"  - klo: {targets['klo']}")
    logger.info(f"  - question: {targets['question']}")

    return targets


def extract_activity_data(topic_data: dict) -> dict:
    """
    Extract activity-related data from simulation_flow.

    Returns dict with:
    - task_email: Primary task email content
    - secondary_task_email: Peer tips email content
    - resource: Main resource content
    - resource_option_sample: Sample resource option (for measuring)
    - guidelines: Guidelines content
    """
    result = {
        "task_email": {},
        "secondary_task_email": {},
        "resource": {},
        "resource_option_sample": {},
        "guidelines": {},
    }

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

        # Task email
        task_email = data.get("task_email", data.get("taskEmail", {}))
        if isinstance(task_email, dict) and task_email:
            if not result["task_email"]:
                result["task_email"] = task_email

        # Secondary task email
        secondary = data.get("secondary_task_email", data.get("secondaryTaskEmail", {}))
        if isinstance(secondary, dict) and secondary:
            if not result["secondary_task_email"]:
                result["secondary_task_email"] = secondary

        # Resource
        resource = data.get("resource", {})
        if isinstance(resource, dict) and resource:
            if not result["resource"]:
                result["resource"] = resource

        # Resource options (get first one as sample)
        resource_opts = data.get("resource_options", data.get("resourceOptions", []))
        if isinstance(resource_opts, list) and resource_opts:
            if not result["resource_option_sample"] and isinstance(resource_opts[0], dict):
                result["resource_option_sample"] = resource_opts[0]

        # Guidelines
        guidelines = data.get("guidelines", {})
        if isinstance(guidelines, dict) and guidelines:
            if not result["guidelines"]:
                result["guidelines"] = guidelines

        # Also check activity data
        activity_data = data.get("activityData", data.get("activity_data", {}))
        if isinstance(activity_data, dict):
            selected = activity_data.get("selectedValue", activity_data.get("selected_value", {}))
            if isinstance(selected, dict):
                # Guidelines might be here
                if "guidelines" in selected and not result["guidelines"]:
                    result["guidelines"] = selected["guidelines"]

    return result


def _extract_all_questions(topic_data: dict) -> list:
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

        # Check activity data
        activity_data = data.get("activityData", data.get("activity_data", {}))
        if isinstance(activity_data, dict):
            selected = activity_data.get("selectedValue", activity_data.get("selected_value", {}))
            if isinstance(selected, dict):
                qs = selected.get("questions", [])
                if isinstance(qs, list):
                    questions.extend(qs)

        # Check children
        children = stage.get("children", [])
        for child in children:
            if isinstance(child, dict):
                child_data = child.get("data", {})
                if isinstance(child_data, dict) and "questions" in child_data:
                    questions.extend(child_data.get("questions", []))

    return questions


def _extract_all_rubrics(topic_data: dict) -> list:
    """Extract all rubric items."""
    rubrics = []

    # Top-level rubric
    top_rubric = topic_data.get("rubric", [])
    if isinstance(top_rubric, list):
        rubrics.extend(top_rubric)

    # Rubrics in simulation_flow
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


def get_word_target_for_field(word_targets: dict, field_path: str) -> list[int]:
    """
    Get the word target range for a specific field.

    Args:
        word_targets: The measured word targets dict
        field_path: Dot-separated path like "workplace_scenario.background.about_organization"

    Returns:
        [min_words, max_words] range
    """
    # Try direct lookup
    if field_path in word_targets:
        return word_targets[field_path]

    # Try last part of path
    parts = field_path.split(".")
    last_part = parts[-1]
    if last_part in word_targets:
        return word_targets[last_part]

    # Default range
    return [50, 200]
