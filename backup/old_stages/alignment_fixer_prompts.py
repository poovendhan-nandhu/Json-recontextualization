"""
Prompt templates for the Alignment Fixer.

These prompts are specifically designed to fix ALIGNMENT issues
(not validation issues). Each prompt targets a specific alignment rule.

Key difference from validation fixer prompts:
- Focus on improving alignment SCORE
- Target KLO mapping, resource alignment, coherence
- More specific about what constitutes "aligned"
"""

# KLO-Question Alignment Fix Prompt
KLO_QUESTION_FIX_PROMPT = """You are an instructional designer fixing KLO-Question alignment.

## CONTEXT
Company: {company_name}
Industry: {industry}

## CURRENT KLOs (Key Learning Outcomes)
{klos}

## CURRENT QUESTIONS
{questions}

## ALIGNMENT ISSUES FOUND
{issues}

## YOUR TASK
For each KLO that is NOT properly assessed by a question:
1. Find the closest question that COULD assess this KLO
2. Rewrite the question to DIRECTLY assess the KLO
3. Use the SAME action verb as the KLO (analyze, evaluate, develop, etc.)
4. Reference the specific concept from the KLO

## ALIGNMENT CRITERIA (What Makes a Question "Aligned"):
- Uses the SAME action verb as the KLO
- Tests the SAME concept/skill as the KLO
- Requires the learner to demonstrate the KLO
- NOT just tangentially related

## OUTPUT FORMAT (JSON only):
{{
    "fixes": [
        {{
            "klo": "The KLO text",
            "question_id": "ID of question to rewrite",
            "original_question": "Original question text",
            "fixed_question": "Rewritten question that DIRECTLY assesses the KLO",
            "reason": "How this question now assesses the KLO"
        }}
    ]
}}

IMPORTANT: Only return JSON, no other text."""


# KLO-Resource Alignment Fix Prompt
KLO_RESOURCE_FIX_PROMPT = """You are a learning resource specialist fixing KLO-Resource alignment.

## CONTEXT
Company: {company_name}
Industry: {industry}

## CURRENT KLOs
{klos}

## CURRENT RESOURCES
{resources}

## ALIGNMENT ISSUES FOUND
{issues}

## YOUR TASK
For each KLO that is NOT supported by resources:
1. Identify which resource SHOULD support it
2. Generate SPECIFIC content to add to that resource
3. The content must enable learners to achieve the KLO
4. Include concrete facts, data, examples (NOT placeholders)

## WHAT "SUPPORTED BY RESOURCE" MEANS:
- Resource contains information needed to achieve the KLO
- Learner can answer questions about the KLO using resource data
- Resource provides examples, frameworks, or knowledge for the KLO
- NOT just mentioning related topics

## OUTPUT FORMAT (JSON only):
{{
    "fixes": [
        {{
            "klo": "The KLO text",
            "resource_id": "ID of resource to update",
            "content_to_add": "SPECIFIC content with facts, data, examples that supports this KLO",
            "reason": "How this content enables learners to achieve the KLO"
        }}
    ]
}}

IMPORTANT: Content must be SPECIFIC to {company_name} and {industry}. No generic placeholders."""


# Scenario-Resource Alignment Fix Prompt
SCENARIO_RESOURCE_FIX_PROMPT = """You are a scenario designer fixing Scenario-Resource alignment.

## CONTEXT
Company: {company_name}
Industry: {industry}

## SCENARIO
{scenario}

## WORKPLACE CONTEXT
{workplace_context}

## CURRENT RESOURCES
{resources}

## ALIGNMENT ISSUES FOUND
{issues}

## YOUR TASK
Ensure resources contain data needed to solve the scenario challenge.
For each missing piece of information:
1. Identify which resource should contain it
2. Add specific, relevant content

## WHAT "SCENARIO-ALIGNED RESOURCE" MEANS:
- Resource provides data needed to address the scenario challenge
- Questions about the scenario can be answered using resource content
- Resource reflects the company's actual situation described in scenario

## OUTPUT FORMAT (JSON only):
{{
    "fixes": [
        {{
            "resource_id": "ID of resource to update",
            "content_to_add": "Content that provides scenario-specific data",
            "reason": "How this helps solve the scenario challenge"
        }}
    ]
}}"""


# Role-Task Alignment Fix Prompt
ROLE_TASK_FIX_PROMPT = """You are a scenario designer fixing Role-Task alignment.

## CONTEXT
Company: {company_name}
Industry: {industry}

## LEARNER ROLE
{role}

## CURRENT TASKS
{tasks}

## ALIGNMENT ISSUES FOUND
{issues}

## YOUR TASK
Ensure each task is appropriate for the learner's role.
For each misaligned task:
1. Rewrite the description to match role responsibilities
2. Make tasks actionable for someone in this role

## WHAT "ROLE-ALIGNED TASK" MEANS:
- Task is something the role would actually do
- Task uses skills the role would have
- Task contributes to role's goals
- NOT a task for a different role

## OUTPUT FORMAT (JSON only):
{{
    "fixes": [
        {{
            "task_name": "Name of task to fix",
            "new_description": "Updated description aligned with the role",
            "reason": "How this task is now appropriate for the role"
        }}
    ]
}}"""


# Scenario Coherence Fix Prompt
SCENARIO_COHERENCE_FIX_PROMPT = """You are a scenario designer fixing internal coherence.

## CONTEXT
Company: {company_name}
Industry: {industry}

## WORKPLACE SCENARIO
{workplace_scenario}

## SELECTED SCENARIO
{selected_scenario}

## COHERENCE ISSUES FOUND
{issues}

## YOUR TASK
Fix internal inconsistencies in the scenario:
1. Dates and timelines must be consistent
2. Names and roles must match throughout
3. Facts and figures must align
4. Challenge must connect to context

## WHAT "COHERENT" MEANS:
- No contradictions between sections
- Timeline makes logical sense
- All references to same entity use same name
- Challenge follows from the background

## OUTPUT FORMAT (JSON only):
{{
    "fixes": [
        {{
            "path": "JSON path to fix (e.g., /topicWizardData/workplaceScenario/background)",
            "original_value": "What it currently says",
            "new_value": "Corrected value",
            "reason": "What inconsistency this fixes"
        }}
    ]
}}"""
