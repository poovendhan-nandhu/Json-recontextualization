"""
Smart Prompt Templates for Leaf-Based Adaptation.

FULLY DYNAMIC APPROACH - No hardcoded mappings!
The LLM understands context and performs semantic transformation.

Principles:
1. Give LLM the SOURCE and TARGET context
2. Let LLM UNDERSTAND the educational purpose
3. LLM decides HOW to transform (not us)
4. Validate OUTPUT quality, not specific patterns
"""

from typing import Dict, List
from .context import AdaptationContext


def build_smart_decision_prompt(
    context: AdaptationContext,
    group_name: str,
    leaves: List[tuple],
    rag_context: str = "",
) -> str:
    """
    Build a DYNAMIC prompt for semantic transformation.

    NO HARDCODED MAPPINGS - LLM figures out the transformation
    based on understanding source and target contexts.

    Args:
        context: AdaptationContext with source/target info
        group_name: Semantic group being processed
        leaves: List of (path, value) tuples
        rag_context: Optional RAG-retrieved examples for ICL
    """
    leaves_text = _build_leaves_text(leaves)

    # Build RAG examples section if available
    rag_section = ""
    if rag_context:
        rag_section = f"""
<similar_examples>
Here are examples of similar content transformations from other simulations.
Use these as reference for style and approach, NOT as templates to copy:

{rag_context}
</similar_examples>
"""

    prompt = f"""<role>
You are an expert Educational Content Transformation Specialist. You transform
simulation-based learning content from one business context to another while
preserving pedagogical integrity.

Your key skill: Understanding the EDUCATIONAL PURPOSE of content and rewriting
it to achieve the SAME PURPOSE in a different business context.
</role>

<task>
Transform content from the SOURCE context to the TARGET context using SEMANTIC
TRANSFORMATION - understand meaning, preserve purpose, change domain.
</task>

<source_context>
<description>{context.source_scenario[:600] if context.source_scenario else "The original business simulation context"}</description>
<industry>{context.source_industry or "Original industry"}</industry>
<company>{context.old_company_names[0] if context.old_company_names else "Original company"}</company>
</source_context>

<target_context>
<description>{context.target_scenario[:800] if context.target_scenario else "Target business scenario"}</description>
<industry>{context.target_industry or "Target industry"}</industry>
<company>{context.new_company_name or "Target company"}</company>
</target_context>

<transformation_principles>

**PRINCIPLE 1: UNDERSTAND BEFORE TRANSFORMING**
Before changing anything, ask yourself:
- What is this content trying to TEACH?
- What SKILL is being assessed or developed?
- What EDUCATIONAL PURPOSE does this serve?

**PRINCIPLE 2: PRESERVE THE PEDAGOGY**
The LEARNING OBJECTIVE stays the same, only the CONTEXT changes:
- If it teaches analysis -> it should still teach analysis
- If it assesses decision-making -> it should still assess decision-making
- If it develops critical thinking -> it should still develop critical thinking

**PRINCIPLE 3: SEMANTIC TRANSFORMATION (NOT WORD REPLACEMENT)**
❌ WRONG: Find "interview" -> Replace with "market analysis"
✅ RIGHT: Understand the content teaches "structured evaluation" -> Rewrite to teach "structured evaluation" in the new domain

**PRINCIPLE 4: COHERENCE CHECK**
Your output must:
- Make grammatical sense
- Be understandable WITHOUT knowing the source
- Fit naturally in the target industry
- Sound like it was written FOR the target scenario (not adapted from something else)

</transformation_principles>

<critical_constraints>

**MUST REMOVE (Poison Terms):**
These terms from the source context must NOT appear in your output:
{_build_poison_terms(context)}

**MUST USE:**
- Company name: {context.new_company_name or "the target company"}
- Industry terminology appropriate for: {context.target_industry or "the target industry"}

</critical_constraints>
{rag_section}
<content_group type="{group_name}">
{_get_group_guidance(group_name, leaves)}
</content_group>

<content_to_transform>
{leaves_text}
</content_to_transform>

<thinking_framework>
For EACH piece of content, think through:

1. PURPOSE ANALYSIS
   - What does this content do? (inform, assess, guide, provide data, etc.)
   - What skill/knowledge does it develop or test?

2. DOMAIN MAPPING
   - What is the EQUIVALENT purpose in the target scenario?
   - What activities/concepts in {context.target_industry or "the target industry"} serve this purpose?

3. REWRITE
   - Write NEW content that achieves the same educational purpose
   - Use terminology natural to {context.target_industry or "the target industry"}
   - Ensure it's coherent and professional

4. QUALITY CHECK
   - Does it make sense on its own?
   - Is it free of source context references?
   - Would someone in {context.target_industry or "the target industry"} recognize this as relevant?
</thinking_framework>

<response_format>
Return JSON with a decision for each leaf:

{{
  "decisions": [
    {{
      "index": 0,
      "action": "replace",
      "new_value": "Your semantically transformed content for target industry...",
      "reason": "MUST_REPLACE path - transformed [source concept] to [target concept]"
    }},
    {{
      "index": 1,
      "action": "keep",
      "new_value": null,
      "reason": "Generic UI label - no domain-specific content"
    }}
  ]
}}

RULES:
- **CRITICAL: Leaves marked with MUST_REPLACE="true" MUST have action="replace"**
  - These include: keyLearningOutcome, workplaceScenario, guidelines, emails, roles, resources
  - You CANNOT use "keep" for these paths - they are ALWAYS domain-specific!
- "keep" = ONLY for truly generic content (UI labels, button text, generic instructions)
- "replace" = content is specific to source context and needs transformation
- EVERY leaf needs a decision (indices 0 to {len(leaves) - 1})
- New content must be COMPLETE and COHERENT - no truncation!
- For HTML content: preserve tag structure, transform ALL text content inside tags
</response_format>
"""
    return prompt


def _get_group_guidance(group_name: str, leaves: List[tuple] = None) -> str:
    """Get dynamic guidance for content group - NO hardcoded mappings.

    Also checks leaf paths for specific content types (e.g., guidelines in simulation_flow).
    """
    group_lower = group_name.lower()

    # Check if any leaf paths contain specific content types
    # This catches guidelines/emails that are nested in simulation_flow
    has_guidelines = False
    has_email = False
    has_resource = False

    if leaves:
        for path, _ in leaves:
            path_lower = path.lower()
            if "/guidelines/" in path_lower:
                has_guidelines = True
            if "/email/" in path_lower or "taskmail" in path_lower:
                has_email = True
            if "/resource/" in path_lower or "markdowntext" in path_lower:
                has_resource = True

    # Path-based detection takes priority (for nested content)
    if has_guidelines:
        return """
This is GUIDELINES/SCAFFOLDING content (HTML blocks with grading tips, instructions).

**CRITICAL: Guidelines content is ALWAYS scenario-specific and MUST be FULLY transformed!**

Guidelines provide detailed instructions to students or evaluators. When transforming:
- COMPLETELY rewrite ALL scenario references (company names, roles, activities)
- Update ALL example content to match target industry
- Transform ALL rubric criteria examples to target context
- Ensure NO source scenario terms remain in ANY part of the HTML
- Keep the same structure (headings, lists, tips) but change ALL content

**IMPORTANT**: This content often contains embedded examples, criteria, and specific scenario
details throughout. You MUST read and transform the ENTIRE content, not just the beginning.
Every mention of the source company, industry, or scenario activities must be changed.
"""

    if has_email:
        return """
This is EMAIL content (communications, instructions).

Emails provide context and instructions to students. When transforming:
- Update SENDER to appropriate role in target organization
- Update CONTENT to reflect target business activities
- Keep the TONE and PURPOSE of the communication
- Ensure tasks/requests are relevant to target scenario
"""

    if has_resource:
        return """
This is RESOURCE content (data, reports, documents).

Resources provide information for students to analyze. When transforming:
- Keep the TYPE of information (financial data stays financial, qualitative stays qualitative)
- Keep the COMPLEXITY level
- Change SPECIFICS to fit the target industry and scenario
- Ensure data is REALISTIC for the target industry
- Ensure resources SUPPORT the questions students will answer
"""

    if "question" in group_lower or "submission" in group_lower:
        return """
This is ASSESSMENT content (questions, submissions).

Assessment content tests student learning. When transforming:
- Keep the COGNITIVE LEVEL (if it asks to analyze, new version should ask to analyze)
- Keep the ASSESSMENT STRUCTURE (if open-ended, stay open-ended; if specific criteria, keep criteria)
- Change the SUBJECT MATTER to fit the target scenario
- Ensure questions are ANSWERABLE based on resources in the target scenario
"""

    elif "resource" in group_lower:
        return """
This is RESOURCE content (data, reports, documents).

Resources provide information for students to analyze. When transforming:
- Keep the TYPE of information (financial data stays financial, qualitative stays qualitative)
- Keep the COMPLEXITY level
- Change SPECIFICS to fit the target industry and scenario
- Ensure data is REALISTIC for the target industry
- Ensure resources SUPPORT the questions students will answer
"""

    elif "rubric" in group_lower or "review" in group_lower:
        return """
This is RUBRIC content (evaluation criteria, scoring guides).

Rubrics define how student work is evaluated. When transforming:
- Keep the SCORING STRUCTURE (levels, point values)
- Keep the EVALUATION DIMENSIONS (quality criteria categories)
- Change the SUBJECT of evaluation to fit target scenario
- Ensure criteria are MEASURABLE and CLEAR for the target context
"""

    elif "workplace" in group_lower or "scenario" in group_lower or "background" in group_lower:
        return """
This is SCENARIO content (background, context, setting).

Scenario content establishes the business context. When transforming:
- COMPLETELY replace with target company/industry context
- Update ALL roles, departments, and organizational references
- Ensure background is CONSISTENT with target scenario
- Make it feel like the simulation was BUILT for the target context
"""

    elif "email" in group_lower:
        return """
This is EMAIL content (communications, instructions).

Emails provide context and instructions to students. When transforming:
- Update SENDER to appropriate role in target organization
- Update CONTENT to reflect target business activities
- Keep the TONE and PURPOSE of the communication
- Ensure tasks/requests are relevant to target scenario
"""

    elif "klo" in group_lower or "criterion" in group_lower or "criteria" in group_lower:
        return """
This is KLO content (Key Learning Outcomes, assessment criteria, keyLearningOutcome fields).

**CRITICAL: keyLearningOutcome fields are ALWAYS domain-specific and MUST be transformed!**

KLOs describe what students learn using DOMAIN-SPECIFIC activities and concepts.
The learning SKILL stays the same, but the DOMAIN SUBJECT must change completely.

When transforming KLOs:
1. IDENTIFY the learning skill (analyze, evaluate, develop, demonstrate, apply, etc.)
2. IDENTIFY the domain-specific subject being learned about
3. COMPLETELY REWRITE the subject for the target industry
4. Keep the same cognitive complexity and learning depth

**DO NOT** keep source industry terminology in KLOs - these fields define what students
learn, so they MUST reflect the target scenario's industry and activities.

**EVERY keyLearningOutcome NEEDS TRANSFORMATION** - there is no such thing as a
"generic" KLO. If it describes learning, it describes learning about SOMETHING specific.
"""

    elif "guidelines" in group_lower or ("text" in group_lower and group_lower != "text"):
        return """
This is GUIDELINES/SCAFFOLDING content (HTML blocks with grading tips, instructions).

**CRITICAL: Guidelines content is ALWAYS scenario-specific and MUST be FULLY transformed!**

Guidelines provide detailed instructions to students or evaluators. When transforming:
- COMPLETELY rewrite ALL scenario references (company names, roles, activities)
- Update ALL example content to match target industry
- Transform ALL rubric criteria examples to target context
- Ensure NO source scenario terms remain in ANY part of the HTML
- Keep the same structure (headings, lists, tips) but change ALL content

**IMPORTANT**: This content often contains embedded examples, criteria, and specific scenario
details throughout. You MUST read and transform the ENTIRE content, not just the beginning.
Every mention of the source company, industry, or scenario activities must be changed.
"""

    else:
        return """
This is general simulation content. When transforming:
- Understand its PURPOSE in the learning experience
- Preserve that purpose while changing the context
- Ensure coherence with target scenario
"""


def _build_poison_terms(context: AdaptationContext) -> str:
    """Build poison terms list from context."""
    if not context.poison_terms:
        return "(Any references to the source company, industry, or scenario-specific terms)"

    terms = context.poison_terms[:50]
    return ", ".join(f'"{t}"' for t in terms)


def _build_leaves_text(leaves: List[tuple]) -> str:
    """Build leaves text with clear structure and force_replace markers."""
    # Import here to avoid circular imports
    from .decider import is_force_replace_path

    lines = []
    force_replace_indices = []

    for i, (path, value) in enumerate(leaves):
        str_value = str(value)
        path_lower = path.lower()

        # Check if this is a force_replace path
        is_forced = is_force_replace_path(path)
        if is_forced:
            force_replace_indices.append(i)

        # Longer limit for HTML/guidelines content to ensure proper transformation
        # IMPORTANT: Don't truncate force_replace content - it MUST be fully transformed
        if is_forced and len(str_value) > 3000:
            # For force_replace, show more content to ensure proper transformation
            max_chars = 6000
        elif ("guidelines" in path_lower or "text" in path_lower) and "<" in str_value:
            max_chars = 5000  # HTML content needs more context for full transformation
        elif len(str_value) > 1500:
            max_chars = 3000  # Long content gets more context
        else:
            max_chars = 1500  # Increased for better context

        display_value = str_value[:max_chars]
        if len(str_value) > max_chars:
            display_value += f"... [truncated, full length: {len(str_value)} chars - TRANSFORM THE FULL CONTENT]"

        display_value = display_value.replace('\n', ' ').replace('\r', '')

        # Mark force_replace leaves clearly
        force_marker = ' MUST_REPLACE="true"' if is_forced else ''

        lines.append(f"""
<leaf index="{i}"{force_marker}>
<path>{path}</path>
<content>{display_value}</content>
</leaf>""")

    # Add summary of force_replace indices
    if force_replace_indices:
        lines.insert(0, f"""
<CRITICAL_NOTICE>
The following leaf indices MUST be replaced (action="replace") - keeping them is NOT allowed:
Indices: {force_replace_indices}
These paths contain domain-specific content that ALWAYS needs transformation.
</CRITICAL_NOTICE>""")

    return "\n".join(lines)


# =============================================================================
# REFERENCE CHECK PROMPT (Dynamic)
# =============================================================================

def build_reference_check_prompt(
    context: AdaptationContext,
    text: str,
) -> str:
    """Build prompt to check for leaked old references - DYNAMIC."""
    return f"""<task>
Check if this text contains references to the SOURCE context that should have been removed.
</task>

<source_context_to_detect>
<company_names>{', '.join(context.old_company_names) if context.old_company_names else 'Unknown source company'}</company_names>
<industry>{context.source_industry or 'Source industry'}</industry>
<poison_terms>{', '.join(context.poison_terms[:30]) if context.poison_terms else 'None specified'}</poison_terms>
</source_context_to_detect>

<target_context>
<company>{context.new_company_name}</company>
<industry>{context.target_industry}</industry>
</target_context>

<text_to_check>
{text[:2000]}
</text_to_check>

<check_for>
1. Direct mentions of source company/industry names
2. Industry-specific terms that belong to source, not target
3. HYBRID content that mixes source and target terminology
4. Content that doesn't make sense in the target context
</check_for>

<response_format>
{{
  "has_source_references": true/false,
  "found_issues": ["description of each issue"],
  "is_coherent": true/false,
  "coherence_issues": ["any content that doesn't make sense"]
}}
</response_format>
"""


# =============================================================================
# TARGETED RETRY PROMPT (Dynamic)
# =============================================================================

def build_targeted_retry_prompt(
    context: AdaptationContext,
    path: str,
    previous_value: str,
    original_value: str,
    failures: List[str],
) -> str:
    """Build retry prompt - focuses on UNDERSTANDING, not specific fixes."""
    failures_text = "\n".join(f"- {f}" for f in failures)

    return f"""<failure_notice>
Your previous transformation FAILED validation.
</failure_notice>

<what_failed>
<path>{path}</path>
<original_content>{original_value[:500]}</original_content>
<your_attempt>{previous_value[:500]}</your_attempt>
<problems>{failures_text}</problems>
</what_failed>

<target_context>
<company>{context.new_company_name}</company>
<industry>{context.target_industry}</industry>
<scenario>{context.target_scenario[:400] if context.target_scenario else "See target context"}</scenario>
</target_context>

<retry_instructions>
The previous attempt likely failed because of one of these issues:

1. INCOMPLETE TRANSFORMATION
   - Some source context terms remained
   - Fix: Rewrite completely for target context

2. INCOHERENT CONTENT
   - The output doesn't make sense on its own
   - Fix: Write content that a reader would understand without knowing the source

3. HYBRID CONTENT
   - Mixed source and target terminology
   - Fix: Use ONLY target context terminology

START FRESH:
1. Read the ORIGINAL content
2. Understand its EDUCATIONAL PURPOSE
3. Write NEW content that achieves that purpose in the TARGET context
4. Verify it makes complete sense and contains no source references
</retry_instructions>

<response_format>
{{
  "analysis": "What the original content's purpose is",
  "transformation_approach": "How you're achieving that purpose in target context",
  "new_value": "Your complete, coherent, transformed content",
  "verification": "Confirmation that it's free of source references and makes sense"
}}
</response_format>
"""


# =============================================================================
# VALIDATION HELPERS (Dynamic - No Hardcoded Patterns)
# =============================================================================

def check_poison_terms(text: str, poison_terms: List[str]) -> List[str]:
    """
    Check for poison terms in text.
    Returns list of found terms.

    This is a simple check - the LLM-based validators do deeper semantic checking.
    """
    found = []
    text_lower = text.lower()

    for term in poison_terms:
        # Only check terms that are meaningful (3+ chars)
        if len(term) >= 3 and term.lower() in text_lower:
            found.append(term)

    return found


def check_klo_alignment(question_text: str, klo_terms: Dict[str, str]) -> bool:
    """
    Check if question contains at least one KLO term.
    Returns True if aligned.

    This is a heuristic - the LLM validators do deeper semantic alignment checking.
    """
    text_lower = question_text.lower()

    for term in klo_terms.values():
        if term and len(term) >= 3 and term.lower() in text_lower:
            return True

    return False
