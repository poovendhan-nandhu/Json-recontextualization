"""
LLM Decider for JSON Contextualizer Agent.

This module handles the LLM decision-making process:
1. Pre-filter leaves (skip IDs, URLs, empty strings, etc.)
2. Group by semantic type for batching
3. Send batch to LLM with full context + validation rules built-in
4. LLM returns {action, new_value, reason} for each leaf
5. QUICK VALIDATE the output - check for obvious issues
6. RETRY with feedback if validation fails (up to MAX_RETRIES)

The LLM decides: "keep" or "replace" for each leaf.
Uses SMART PROMPTS with validation rules so LLM gets it right first time.
If first attempt fails validation, RETRIES with specific feedback.
"""

from typing import List, Tuple, Dict, Any, Optional, Literal
from dataclasses import dataclass
from pydantic import BaseModel, Field
import logging
import asyncio
import json
import re

from langsmith import traceable

from .grouper import group_leaves_by_semantic_context
from .context import AdaptationContext
from .smart_prompts import build_smart_decision_prompt, check_poison_terms

logger = logging.getLogger(__name__)

# =============================================================================
# RETRY CONFIGURATION
# =============================================================================

MAX_RETRIES = 1  # Reduced from 3 - single pass for faster completion
RETRY_TEMPERATURE_INCREMENT = 0.1  # Increase temperature on retry


# =============================================================================
# QUICK VALIDATION (Fast checks before accepting output)
# =============================================================================

@dataclass
class QuickValidationResult:
    """Result of quick validation on decider output."""
    passed: bool
    issues: List[str]
    failed_indices: List[int]  # Which decisions failed

    @property
    def feedback(self) -> str:
        """Generate feedback for retry prompt."""
        if self.passed:
            return ""
        return "\n".join([f"- {issue}" for issue in self.issues])


def quick_validate_decisions(
    decisions: List["DecisionResult"],
    context: "AdaptationContext",
) -> QuickValidationResult:
    """
    Quick validation of decider output BEFORE accepting it.

    Checks:
    1. FORCE_REPLACE leaves were actually replaced (not kept!)
    2. No POISON TERMS in new_value (old company names + context.poison_terms)
    3. No nonsensical content (gibberish detection)
    4. No literal replacement patterns
    5. HTML content is sane (no broken tags, truncation)
    6. Tag stubs are not kept (must be replaced or merged)

    This is FAST - not full LLM validation.
    """
    issues = []
    failed_indices = []

    # Combine old company names with dynamically extracted poison terms
    old_names = context.old_company_names or []
    poison_terms = context.poison_terms or []

    # Tag stubs that should NEVER be kept as standalone values
    TAG_STUBS = {
        "<p>", "</p>", "<p/>",
        "<ol>", "</ol>",
        "<ul>", "</ul>",
        "<li>", "</li>",
        "<div>", "</div>",
        "<strong>", "</strong>",
        "<em>", "</em>",
        "<span>", "</span>",
        "<h1>", "</h1>", "<h2>", "</h2>", "<h3>", "</h3>",
        "<h4>", "</h4>", "<h5>", "</h5>", "<h6>", "</h6>",
    }

    # Truly generic English words that should NEVER be treated as poison
    # NOTE: Removed HR-specific terms like "interview", "candidate", "KSAO"
    # Those will be caught by context.poison_terms when changing FROM HR domain
    SAFE_WORDS = {
        "reliability", "objectivity", "validity",
        "event", "events", "venue", "venues",
        "rating", "scale", "process", "evaluation", "criteria",
        "questions", "question", "analysis", "market", "data", "research",
        "performance", "quality", "feedback", "development", "training", "review",
        "score", "scoring", "rubric", "skill", "skills",
    }

    for i, decision in enumerate(decisions):
        path_lower = decision.path.lower()

        # =================================================================
        # CHECK 0: FORCE_REPLACE paths MUST have action="replace"
        # =================================================================
        if is_force_replace_path(decision.path):
            if decision.action != "replace":
                logger.error(f"[FORCE_REPLACE VIOLATION] Index {i}: LLM chose 'keep' for forced path!")
                logger.error(f"  Path: {decision.path}")
                logger.error(f"  Old value (first 100 chars): {decision.old_value[:100]}...")
                issues.append(
                    f"Index {i}: FORCE_REPLACE path '{decision.path}' was kept instead of replaced!"
                )
                failed_indices.append(i)
                continue  # This is a critical failure

            if not decision.new_value or not decision.new_value.strip():
                logger.error(f"[FORCE_REPLACE VIOLATION] Index {i}: Empty new_value for forced path!")
                logger.error(f"  Path: {decision.path}")
                issues.append(
                    f"Index {i}: FORCE_REPLACE path has empty new_value"
                )
                failed_indices.append(i)
                continue

            # For force_replace, also check it's actually different
            if decision.new_value.strip() == decision.old_value.strip():
                logger.error(f"[FORCE_REPLACE VIOLATION] Index {i}: Identical old/new for forced path!")
                logger.error(f"  Path: {decision.path}")
                logger.error(f"  Value: {decision.old_value[:100]}...")
                issues.append(
                    f"Index {i}: FORCE_REPLACE path has identical old/new value - must transform!"
                )
                failed_indices.append(i)
                continue

            # Success - log that force_replace worked
            logger.debug(f"[FORCE_REPLACE OK] Index {i}: {decision.path} transformed successfully")

        # =================================================================
        # CHECK 0.5: Tag stubs should NOT be kept (they need transformation)
        # =================================================================
        old_stripped = decision.old_value.strip() if decision.old_value else ""
        if decision.action == "keep" and old_stripped.lower() in {s.lower() for s in TAG_STUBS}:
            logger.warning(f"[TAG_STUB VIOLATION] Index {i}: Kept a tag stub '{old_stripped}'")
            issues.append(f"Index {i}: Tag stub '{old_stripped}' was kept - should be replaced or merged")
            failed_indices.append(i)
            continue

        # Skip further checks if action is "keep"
        if decision.action != "replace" or not decision.new_value:
            continue

        new_val = decision.new_value.lower()
        old_val = decision.old_value.lower() if decision.old_value else ""

        # =================================================================
        # CHECK 1: Old company names remain (ONLY specific names, not generic words)
        # =================================================================
        for old_name in old_names:
            old_name_lower = old_name.lower().strip()
            # Skip if it's a common word (not a company name)
            if old_name_lower in SAFE_WORDS:
                continue
            # Skip single common words
            if len(old_name_lower.split()) == 1 and len(old_name_lower) < 8:
                continue
            # Check for actual company name
            if old_name_lower in new_val:
                issues.append(f"Index {i}: Old company name '{old_name}' still in output")
                failed_indices.append(i)
                break

        # =================================================================
        # CHECK 1.5: Poison terms from context (dynamically extracted)
        # =================================================================
        for term in poison_terms:
            term_lower = term.lower().strip()
            # Skip if it's a safe common word
            if term_lower in SAFE_WORDS:
                continue
            # Skip very short terms (likely false positives)
            if len(term_lower) < 4:
                continue
            # Check for poison term in new value
            if term_lower in new_val:
                issues.append(f"Index {i}: Poison term '{term}' found in output")
                failed_indices.append(i)
                break

        # =================================================================
        # CHECK 2: Nonsensical content detection - ONLY for company name fields
        # =================================================================
        is_company_field = any(x in path_lower for x in ["companyname", "company_name", "organizationname"])
        if is_company_field:
            # Company names shouldn't be full sentences
            if len(decision.new_value.split()) > 5:  # Company name > 5 words is suspicious
                issues.append(f"Index {i}: Company name too long: '{decision.new_value[:50]}...'")
                failed_indices.append(i)

        # =================================================================
        # CHECK 3: Empty or whitespace-only replacement
        # =================================================================
        if decision.new_value and not decision.new_value.strip():
            issues.append(f"Index {i}: Empty replacement value")
            failed_indices.append(i)

        # =================================================================
        # CHECK 4: Replacement same as original (pointless) - for non-force paths
        # =================================================================
        if not is_force_replace_path(decision.path):
            if decision.new_value == decision.old_value:
                issues.append(f"Index {i}: Replacement identical to original")
                failed_indices.append(i)

        # =================================================================
        # CHECK 5: HTML sanity for HTML content
        # =================================================================
        if "<" in decision.new_value and ">" in decision.new_value:
            html_valid, html_issues = check_html_sanity(decision.new_value)
            if not html_valid:
                for html_issue in html_issues[:2]:  # Limit to 2 issues per leaf
                    issues.append(f"Index {i}: HTML issue - {html_issue}")
                failed_indices.append(i)

    passed = len(issues) == 0

    if not passed:
        logger.warning(f"Quick validation FAILED with {len(issues)} issues")
        for issue in issues[:5]:  # Log first 5
            logger.warning(f"  {issue}")

    return QuickValidationResult(
        passed=passed,
        issues=issues,
        failed_indices=list(set(failed_indices)),
    )


# =============================================================================
# RESPONSE MODELS
# =============================================================================

class LeafDecision(BaseModel):
    """LLM's decision for a single leaf."""
    index: int = Field(description="Index of the leaf in the batch")
    action: Literal["keep", "replace"] = Field(description="Whether to keep or replace")
    new_value: Optional[str] = Field(None, description="New value if action is replace")
    reason: str = Field(description="Brief reason for the decision")


class BatchDecisionResponse(BaseModel):
    """LLM's response for a batch of leaves."""
    decisions: List[LeafDecision] = Field(description="Decisions for each leaf")


@dataclass
class DecisionResult:
    """Result of a decision for a leaf."""
    path: str
    old_value: str
    action: str  # "keep" or "replace"
    new_value: Optional[str]
    reason: str

    @property
    def should_change(self) -> bool:
        return self.action == "replace" and self.new_value is not None


# =============================================================================
# PRE-FILTER (No LLM needed)
# =============================================================================

# Paths that should NEVER be sent to LLM (only truly immutable fields)
# IMPORTANT: IDs must be preserved per Rachit's requirements!
SKIP_PATH_PATTERNS = [
    r".*/id$",           # Keep IDs locked!
    r".*/uuid$",
    r".*/guid$",
    r".*/_id$",
    r".*/url$",
    r".*/href$",
    r".*/src$",
    r".*/key$",
    r".*/token$",
    r".*/workspace$",
    r".*/builderType$",
    r".*/type$",
    # REMOVED: r".*/scenarioOptions/.*" - content SHOULD be transformed!
    r".*/duration$",
    r".*/level$",
]

# Keywords in path that indicate skip (must be exact segment match, not substring)
# NOTE: We check for "/keyword" or "/keyword/" to avoid false matches like "keyLearningOutcome"
SKIP_PATH_KEYWORDS = [
    "/url", "/id/", "/key/", "/token", "/hash", "/uuid", "/guid",
    "/workspace", "/builderType",
]


# =============================================================================
# FORCE REPLACE PATHS (These MUST be transformed - no "keep" option!)
# =============================================================================
# These paths contain domain-specific content that ALWAYS needs transformation.
# The LLM cannot choose "keep" for these - they are ALWAYS "replace".

FORCE_REPLACE_PATTERNS = [
    # KLOs - ALWAYS domain-specific, MUST transform
    r".*/keyLearningOutcome$",
    r".*/keyLearningOutcome/.*",

    # Scenario/Background - ALWAYS context-specific
    r".*/workplaceScenario/.*",
    r".*/aboutOrganization$",
    r".*/organizationName$",
    r".*/currentIssue$",
    r".*/scenarioOptions/.*/option$",
    r".*/scenarioDescription$",

    # Lesson content - describes what students do
    r".*/lessonInformation/lesson$",

    # Guidelines - detailed instructions, ALWAYS scenario-specific
    r".*/guidelines/text$",
    r".*/guidelines/purpose$",

    # Emails - communications with scenario-specific content
    r".*/email/body$",
    r".*/email/subject$",
    r".*/secondaryTaskEmail/body$",
    r".*/secondaryTaskEmail/subject$",
    r".*/reportingManager/message$",

    # Task emails in activities (different schema paths)
    r".*/taskEmail/body$",
    r".*/taskEmail/subject$",
    r".*/task-email/body$",
    r".*/task-email/subject$",
    r".*/managerEmail/body$",
    r".*/managerEmail/subject$",

    # Role descriptions - ALWAYS tied to scenario
    r".*/learnerRole/roleDescription$",
    r".*/learnerRole/role$",
    r".*/reportingManager/role$",
    r".*/scopeOfWork/.*/task$",
    r".*/scopeOfWork/.*/description$",

    # Resource content - ALWAYS domain-specific
    r".*/resource/markdownText$",
    r".*/resource/title$",
    r".*/resourceOptions/.*/title$",
    r".*/resourceOptions/.*/description$",
    r".*/resourceOptions/.*/relevance$",

    # Review helper prompts - carry grading assumptions
    r".*/review/helperPrompt$",
    r".*/helperPrompt$",
]

# Compile patterns for efficiency
_FORCE_REPLACE_COMPILED = [re.compile(p, re.IGNORECASE) for p in FORCE_REPLACE_PATTERNS]


def is_force_replace_path(path: str) -> bool:
    """Check if path MUST be replaced (no keep option)."""
    for pattern in _FORCE_REPLACE_COMPILED:
        if pattern.match(path):
            return True
    return False


def classify_leaf(path: str, value: Any) -> str:
    """
    Classify a leaf into one of three categories:
    - "skip": Don't send to LLM at all (IDs, URLs, etc.)
    - "force_replace": MUST be transformed (KLOs, scenario, etc.)
    - "llm_decides": LLM can choose keep or replace

    Returns: "skip", "force_replace", or "llm_decides"
    """
    # First check if it should be skipped entirely
    should_send, reason = should_send_to_llm(path, value)
    if not should_send:
        return "skip"

    # Check if it's a force-replace path
    if is_force_replace_path(path):
        logger.debug(f"[FORCE_REPLACE] Path classified as force_replace: {path}")
        return "force_replace"

    # Otherwise, LLM decides
    return "llm_decides"


def log_force_replace_summary(leaves: List[Tuple[str, Any]]) -> Dict[str, int]:
    """
    Log a summary of leaf classifications.
    Returns counts by classification type.
    """
    counts = {"skip": 0, "force_replace": 0, "llm_decides": 0}
    force_replace_paths = []

    for path, value in leaves:
        classification = classify_leaf(path, value)
        counts[classification] += 1
        if classification == "force_replace":
            force_replace_paths.append(path)

    logger.info(f"[CLASSIFICATION] Leaf breakdown: {counts}")
    if force_replace_paths:
        logger.info(f"[FORCE_REPLACE] {len(force_replace_paths)} leaves MUST be replaced:")
        for p in force_replace_paths[:10]:  # Show first 10
            logger.info(f"  -> {p}")
        if len(force_replace_paths) > 10:
            logger.info(f"  ... and {len(force_replace_paths) - 10} more")

    return counts


# =============================================================================
# HTML PROCESSING (Chunking + Sanity Checks)
# =============================================================================

def chunk_html_content(html: str, max_chunk_size: int = 3000) -> List[str]:
    """
    Split HTML content into logical chunks for processing.

    Splits on block-level elements (headings, paragraphs, lists) to maintain
    structural integrity. Each chunk is a complete, valid HTML fragment.
    """
    if len(html) <= max_chunk_size:
        return [html]

    # Split on block-level boundaries
    # Priority: h1-h6, then hr, then p/div/ul/ol/table
    block_patterns = [
        r'(<h[1-6][^>]*>)',  # Headings
        r'(<hr[^>]*>)',       # Horizontal rules
        r'(<(?:p|div|ul|ol|table|blockquote)[^>]*>)',  # Block elements
    ]

    chunks = []
    current_chunk = ""

    # Simple approach: split by </p>, </ul>, </ol>, </div>, </h1-6>
    # and recombine into chunks under max size
    split_pattern = r'(</(?:p|ul|ol|div|h[1-6]|table|blockquote)>)'
    parts = re.split(split_pattern, html)

    for i in range(0, len(parts), 2):
        segment = parts[i]
        if i + 1 < len(parts):
            segment += parts[i + 1]  # Include the closing tag

        if len(current_chunk) + len(segment) <= max_chunk_size:
            current_chunk += segment
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = segment

    if current_chunk:
        chunks.append(current_chunk)

    # Fallback: if we couldn't split nicely, just chunk by size
    if not chunks:
        for i in range(0, len(html), max_chunk_size):
            chunks.append(html[i:i + max_chunk_size])

    return chunks


def check_html_sanity(html: str) -> Tuple[bool, List[str]]:
    """
    Check HTML for common issues that indicate broken transformation.

    Returns: (is_valid, list_of_issues)
    """
    issues = []

    # Check 0: CRITICAL - Value is ONLY a tag stub (blocker!)
    tag_stub_patterns = [
        r'^<p>$', r'^</p>$', r'^<p/>$',
        r'^<ol>$', r'^</ol>$',
        r'^<ul>$', r'^</ul>$',
        r'^<li>$', r'^</li>$',
        r'^<div>$', r'^</div>$',
        r'^<h[1-6]>$', r'^</h[1-6]>$',
    ]
    html_stripped = html.strip()
    for pattern in tag_stub_patterns:
        if re.match(pattern, html_stripped, re.IGNORECASE):
            issues.append(f"BLOCKER: Value is only a tag stub: '{html_stripped}'")
            return False, issues  # Immediate fail

    # Check 1: Standalone placeholder tags
    placeholder_patterns = [
        r'^<p>\s*$',           # Empty <p> tag alone
        r'^<ol>\s*$',          # Empty <ol> tag alone
        r'^<ul>\s*$',          # Empty <ul> tag alone
        r'^<div>\s*$',         # Empty <div> tag alone
        r'^\s*</p>\s*$',       # Orphan closing tag
        r'^\s*</ol>\s*$',
        r'^\s*</ul>\s*$',
    ]
    for pattern in placeholder_patterns:
        if re.match(pattern, html.strip()):
            issues.append(f"Placeholder/orphan tag detected: {html[:50]}")

    # Check 2: Truncated sentences (ends with incomplete word)
    # Look for patterns like "market-s" or "strateg" at the end
    truncation_pattern = r'\b[a-zA-Z]{2,}-[a-zA-Z]{1,3}\.{0,3}$'
    if re.search(truncation_pattern, html.strip()):
        issues.append(f"Possible truncation detected at end")

    # Check 3: Unbalanced critical tags
    critical_tags = ['p', 'ul', 'ol', 'h1', 'h2', 'h3', 'h4', 'li', 'table', 'tr', 'td']
    for tag in critical_tags:
        open_count = len(re.findall(f'<{tag}[^>]*>', html, re.IGNORECASE))
        close_count = len(re.findall(f'</{tag}>', html, re.IGNORECASE))
        # Allow self-closing and some flexibility
        if abs(open_count - close_count) > 2:
            issues.append(f"Unbalanced <{tag}> tags: {open_count} open, {close_count} close")

    # Check 4: Content too short for HTML (likely stripped/broken)
    text_only = re.sub(r'<[^>]+>', '', html).strip()
    if len(html) > 100 and len(text_only) < 20:
        issues.append(f"HTML has tags but almost no text content")

    # Check 5: Sentence ends abruptly (mid-word truncation)
    text_end = text_only[-50:] if len(text_only) > 50 else text_only
    if text_end and not re.search(r'[.!?"\'\)]\s*$', text_end):
        # Ends without punctuation - might be truncated
        last_word = text_end.split()[-1] if text_end.split() else ""
        if len(last_word) > 2 and last_word[-1].isalpha():
            # Check if it looks like a truncated word
            if not last_word[0].isupper():  # Not an acronym
                issues.append(f"Content may be truncated: ends with '{last_word}'")

    return len(issues) == 0, issues


def repair_html_issues(html: str) -> str:
    """
    Attempt basic repairs on HTML issues.

    This is a best-effort cleanup, not a full fix.
    """
    result = html

    # Remove orphan closing tags at start
    result = re.sub(r'^\s*</[a-z]+>\s*', '', result, flags=re.IGNORECASE)

    # Remove empty placeholder tags
    result = re.sub(r'<(p|div|span)>\s*</\1>', '', result, flags=re.IGNORECASE)

    # Ensure doesn't start with just a closing tag
    result = result.strip()

    return result


def should_send_to_llm(path: str, value: Any) -> Tuple[bool, str]:
    """
    Pre-filter: Decide if a leaf should be sent to LLM.

    Returns:
        (should_send, reason)
    """
    # Skip non-strings
    if not isinstance(value, str):
        return False, "non-string value"

    # Skip empty strings
    if not value.strip():
        return False, "empty string"

    # Skip very short values (likely labels/codes)
    if len(value) < 3:
        return False, "too short"

    # Skip HTML tag stubs - these are structural elements with no content to transform
    # They appear as artifacts of leaf-based JSON decomposition
    TAG_STUB_PATTERNS = {
        "<p>", "</p>", "<p/>", "<p />",
        "<ol>", "</ol>", "<ul>", "</ul>",
        "<li>", "</li>",
        "<div>", "</div>",
        "<strong>", "</strong>",
        "<em>", "</em>",
        "<span>", "</span>",
        "<br>", "<br/>", "<br />",
        "<hr>", "<hr/>", "<hr />",
        "<h1>", "</h1>", "<h2>", "</h2>", "<h3>", "</h3>",
        "<h4>", "</h4>", "<h5>", "</h5>", "<h6>", "</h6>",
        "<table>", "</table>", "<tr>", "</tr>", "<td>", "</td>", "<th>", "</th>",
        "<tbody>", "</tbody>", "<thead>", "</thead>",
    }
    if value.strip().lower() in {s.lower() for s in TAG_STUB_PATTERNS}:
        return False, "HTML tag stub (structural element)"

    # Skip if path matches skip patterns
    for pattern in SKIP_PATH_PATTERNS:
        if re.match(pattern, path, re.IGNORECASE):
            return False, f"skip pattern: {pattern}"

    # Skip if path contains skip keywords
    path_lower = path.lower()
    for keyword in SKIP_PATH_KEYWORDS:
        if keyword in path_lower:
            return False, f"skip keyword: {keyword}"

    # Skip URLs
    if value.startswith(("http://", "https://", "www.")):
        return False, "URL value"

    # Skip UUIDs/GUIDs
    uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
    if re.match(uuid_pattern, value, re.IGNORECASE):
        return False, "UUID value"

    # Skip base64 or encoded data
    if len(value) > 100 and not " " in value:
        return False, "likely encoded data"

    return True, "candidate for LLM"


def pre_filter_leaves(
    leaves: List[Tuple[str, Any]]
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, Any, str]]]:
    """
    Pre-filter leaves before sending to LLM.

    Returns:
        (candidates_for_llm, skipped_with_reasons)
    """
    candidates = []
    skipped = []

    for path, value in leaves:
        should_send, reason = should_send_to_llm(path, value)
        if should_send:
            candidates.append((path, value))
        else:
            skipped.append((path, value, reason))

    logger.info(f"Pre-filter: {len(candidates)} candidates, {len(skipped)} skipped")
    return candidates, skipped


# =============================================================================
# LLM DECIDER
# =============================================================================

class LeafDecider:
    """
    LLM-based decision maker for leaf modifications.

    Flow:
    1. Pre-filter leaves (no LLM)
    2. Group by semantic type
    3. Process leaves in parallel with concurrency control
    4. LLM returns {action, new_value, reason} for each

    Uses SMART PROMPTS with ALL validation rules built-in:
    - Entity removal (poison terms)
    - KLO-Question alignment
    - Resource answerability
    - Domain fidelity (valid/invalid KPIs)
    """

    def __init__(
        self,
        context: AdaptationContext,
        max_leaves_per_batch: int = 50,
        max_concurrent_calls: int = 6,
        rag_context: Dict[str, str] = None,
    ):
        """
        Args:
            context: AdaptationContext with all extracted context
            max_leaves_per_batch: Maximum leaves per LLM call
            max_concurrent_calls: Maximum concurrent LLM calls
            rag_context: RAG retrieved examples by group (optional)
        """
        self.context = context
        self.max_leaves_per_batch = max_leaves_per_batch
        self.rag_context = rag_context or {}
        self.max_concurrent_calls = max_concurrent_calls

        # Semaphore for rate limiting
        self._semaphore = None

    @classmethod
    def from_factsheet(
        cls,
        factsheet: Dict[str, Any],
        target_scenario: str,
        source_scenario: str = "",
        max_leaves_per_batch: int = 50,
        max_concurrent_calls: int = 6,
    ) -> "LeafDecider":
        """
        Create LeafDecider from legacy factsheet format.

        Args:
            factsheet: Global factsheet with entity/context info
            target_scenario: Target scenario description
            source_scenario: Source scenario description
            max_leaves_per_batch: Maximum leaves per LLM call
            max_concurrent_calls: Maximum concurrent LLM calls
        """
        # Convert factsheet to AdaptationContext
        context = AdaptationContext()
        context.target_scenario = target_scenario
        context.source_scenario = source_scenario

        # Extract company info
        company = factsheet.get("company", {})
        if isinstance(company, dict):
            context.new_company_name = company.get("name") or company.get("new_name", "")
            old_name = company.get("old_name") or company.get("source_name", "")
            if old_name:
                context.old_company_names = [old_name]
                context.poison_terms.append(old_name)
            context.target_industry = company.get("industry", "")

        # Extract manager mapping
        manager = factsheet.get("manager", {})
        if isinstance(manager, dict):
            old = manager.get("old_name") or manager.get("source_name")
            new = manager.get("name") or manager.get("new_name")
            if old and new:
                context.entity_map[old] = new

        # Extract additional mappings
        extra = factsheet.get("entity_mappings", {})
        if isinstance(extra, dict):
            context.entity_map.update(extra)

        return cls(
            context=context,
            max_leaves_per_batch=max_leaves_per_batch,
            max_concurrent_calls=max_concurrent_calls,
        )

    @traceable(name="leaf_decider_decide_all")
    async def decide_all(
        self,
        leaves: List[Tuple[str, Any]],
    ) -> List[DecisionResult]:
        """
        Make decisions for all leaves with parallel processing and rate limiting.

        Args:
            leaves: List of (path, value) tuples

        Returns:
            List of DecisionResult objects
        """
        # Initialize semaphore for this run
        self._semaphore = asyncio.Semaphore(self.max_concurrent_calls)

        # Step 1: Pre-filter
        candidates, skipped = pre_filter_leaves(leaves)

        # Create results for skipped leaves (action=keep)
        results = []
        for path, value, reason in skipped:
            results.append(DecisionResult(
                path=path,
                old_value=str(value) if value is not None else "",
                action="keep",
                new_value=None,
                reason=f"pre-filter: {reason}",
            ))

        if not candidates:
            logger.info("No candidates for LLM after pre-filter")
            return results

        # Log force_replace classification summary
        log_force_replace_summary(candidates)

        # Step 2: Group by semantic type
        grouped = group_leaves_by_semantic_context(candidates)
        logger.info(f"Grouped into {len(grouped)} semantic groups")

        # Step 3: Collect ALL batches from ALL groups
        # IMPORTANT: Long HTML leaves get their own batch for focused processing
        HTML_THRESHOLD = 4000  # Leaves longer than this get individual processing
        all_batches = []
        long_html_count = 0

        for group_name, group_leaves in grouped.items():
            if not group_leaves:
                continue

            # Separate long HTML leaves from normal leaves
            normal_leaves = []
            long_html_leaves = []

            for path, value in group_leaves:
                str_val = str(value)
                is_long_html = (
                    len(str_val) > HTML_THRESHOLD and
                    "<" in str_val and ">" in str_val
                )
                if is_long_html:
                    long_html_leaves.append((path, value))
                else:
                    normal_leaves.append((path, value))

            # Each long HTML leaf gets its own batch for focused processing
            for leaf in long_html_leaves:
                all_batches.append((group_name, [leaf]))
                long_html_count += 1

            # Normal leaves get batched together
            for i in range(0, len(normal_leaves), self.max_leaves_per_batch):
                batch = normal_leaves[i:i + self.max_leaves_per_batch]
                all_batches.append((group_name, batch))

        if long_html_count > 0:
            logger.info(f"[HTML] Separated {long_html_count} long HTML leaves for individual processing")

        logger.info(f"Processing {len(all_batches)} batches in parallel (max {self.max_concurrent_calls} concurrent)")

        # Step 4: Process ALL batches in parallel with semaphore
        async def process_with_limit(group_name: str, batch: List[Tuple[str, str]]):
            async with self._semaphore:
                return await self._process_batch(group_name, batch)

        tasks = [process_with_limit(gn, batch) for gn, batch in all_batches]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Flatten results
        for result in batch_results:
            if isinstance(result, Exception):
                logger.error(f"Batch processing failed: {result}")
            elif isinstance(result, list):
                results.extend(result)

        logger.info(f"Total decisions: {len(results)}")
        return results

    @traceable(name="leaf_decider_process_group")
    async def _process_group(
        self,
        group_name: str,
        leaves: List[Tuple[str, str]],
    ) -> List[DecisionResult]:
        """Process a semantic group of leaves."""
        logger.info(f"Processing group '{group_name}' with {len(leaves)} leaves")

        # Split into batches and process in parallel
        batches = []
        for i in range(0, len(leaves), self.max_leaves_per_batch):
            batch = leaves[i:i + self.max_leaves_per_batch]
            batches.append(batch)

        # Process all batches in parallel
        tasks = [self._process_batch(group_name, batch) for batch in batches]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Flatten results
        results = []
        for result in batch_results:
            if isinstance(result, Exception):
                logger.error(f"Batch processing failed: {result}")
            elif isinstance(result, list):
                results.extend(result)

        return results

    @traceable(name="leaf_decider_process_batch")
    async def _process_batch(
        self,
        group_name: str,
        batch: List[Tuple[str, str]],
    ) -> List[DecisionResult]:
        """
        Process a single batch with LLM + SMART RETRY LOOP.

        Flow:
        1. Build prompt and call LLM
        2. Parse response into DecisionResults
        3. Quick validate the results
        4. If validation fails, ONLY RETRY FAILED LEAVES (not entire batch!)
        5. Merge fixed results with passing results
        6. Return best results

        SMART RETRY: Only re-processes failed leaves, not entire batch!
        """
        attempt = 0
        temperature = 0.2  # Start with low temperature
        last_issues = []  # Track issues for retry feedback

        # Track results: index -> DecisionResult
        final_results: Dict[int, DecisionResult] = {}

        # Track which indices still need processing
        pending_indices = set(range(len(batch)))
        current_batch = batch  # Start with full batch
        index_map = {i: i for i in range(len(batch))}  # Maps current batch index to original index

        while attempt < MAX_RETRIES and pending_indices:
            attempt += 1
            logger.info(f"[DECIDER] Batch attempt {attempt}/{MAX_RETRIES} for group '{group_name}' ({len(pending_indices)} pending)")

            try:
                # Build prompt
                if attempt == 1:
                    prompt = self._build_prompt(group_name, current_batch)
                else:
                    # Build retry prompt with feedback about failures
                    failed_results = [final_results.get(i) for i in pending_indices if i in final_results]
                    prompt = self._build_retry_prompt(
                        group_name, current_batch, failed_results, last_issues
                    )

                # Call LLM with current temperature
                response = await self._call_llm(prompt, temperature=temperature)

                # Parse response
                results = self._parse_response(current_batch, response)

                # Quick validate
                validation = quick_validate_decisions(results, self.context)

                if validation.passed:
                    # All passed - store results and we're done
                    for i, result in enumerate(results):
                        orig_idx = index_map[i]
                        final_results[orig_idx] = result
                    pending_indices.clear()
                    logger.info(f"[DECIDER] Batch passed validation on attempt {attempt}")
                    break

                # Some failed - separate passing from failing
                failed_set = set(validation.failed_indices)
                last_issues = validation.issues

                logger.warning(f"[DECIDER] Batch attempt {attempt}: {len(failed_set)} failed, {len(results) - len(failed_set)} passed")

                # Store passing results
                for i, result in enumerate(results):
                    orig_idx = index_map[i]
                    if i not in failed_set:
                        final_results[orig_idx] = result
                        pending_indices.discard(orig_idx)
                    else:
                        # Store failed result temporarily (might be used if retries exhausted)
                        final_results[orig_idx] = result

                # Build new batch with ONLY failed leaves for retry
                if pending_indices:
                    new_batch = []
                    new_index_map = {}
                    for new_i, orig_i in enumerate(sorted(pending_indices)):
                        new_batch.append(batch[orig_i])
                        new_index_map[new_i] = orig_i

                    current_batch = new_batch
                    index_map = new_index_map

                    logger.info(f"[DECIDER] Retrying {len(current_batch)} failed leaves only")

                # Increase temperature slightly for retry
                temperature = min(0.7, temperature + RETRY_TEMPERATURE_INCREMENT)

            except Exception as e:
                logger.error(f"[DECIDER] Batch attempt {attempt} failed with error: {e}")
                last_issues = [str(e)]
                # Continue to retry

        # Return results in original order
        if pending_indices:
            logger.warning(f"[DECIDER] Exhausted {MAX_RETRIES} retries, {len(pending_indices)} leaves still failing")

        # Build final result list in original order
        result_list = []
        for i in range(len(batch)):
            if i in final_results:
                result_list.append(final_results[i])
            else:
                # Should not happen, but fallback to keep
                path, value = batch[i]
                result_list.append(DecisionResult(
                    path=path,
                    old_value=value,
                    action="keep",
                    new_value=None,
                    reason=f"Failed after {MAX_RETRIES} retries",
                ))

        return result_list

    def _build_retry_prompt(
        self,
        group_name: str,
        batch: List[Tuple[str, str]],
        previous_results: List[DecisionResult],
        issues: List[str],
    ) -> str:
        """
        Build a retry prompt that includes feedback about what went wrong.

        This tells the LLM:
        1. What it tried before
        2. Why it was wrong
        3. How to fix it
        """
        # Get the base prompt
        base_prompt = self._build_prompt(group_name, batch)

        # Build feedback section
        feedback_lines = [
            "\n\n" + "="*60,
            "⚠️  RETRY - YOUR PREVIOUS ATTEMPT HAD ISSUES",
            "="*60,
            "",
            "Your previous output had the following problems:",
            ""
        ]

        for issue in issues[:10]:  # Limit to 10 issues
            feedback_lines.append(f"  ❌ {issue}")

        feedback_lines.extend([
            "",
            "IMPORTANT CORRECTIONS:",
            "1. Do NOT include any poison terms (old company names, old industry terms)",
            "2. Company names should be SHORT (1-4 words), not sentences",
            "3. Transform content SEMANTICALLY for the new industry",
            "4. Every replacement must be DIFFERENT from the original",
            "5. Use terminology appropriate for: " + (self.context.target_industry or "the target industry"),
            "",
            "Please provide CORRECTED decisions:",
            "="*60,
        ])

        feedback = "\n".join(feedback_lines)

        # Also show what they tried before for specific failed indices
        if previous_results:
            failed_examples = []
            for issue in issues[:5]:
                # Extract index from issue message
                if "Index " in issue:
                    try:
                        idx = int(issue.split("Index ")[1].split(":")[0])
                        if idx < len(previous_results):
                            prev = previous_results[idx]
                            failed_examples.append(
                                f"  Index {idx}: You wrote '{prev.new_value[:50]}...' - THIS WAS WRONG"
                            )
                    except:
                        pass

            if failed_examples:
                feedback += "\n\nYour WRONG outputs (do not repeat these):\n"
                feedback += "\n".join(failed_examples)

        return base_prompt + feedback

    def _build_prompt(
        self,
        group_name: str,
        batch: List[Tuple[str, str]],
    ) -> str:
        """Build the SMART prompt with all validation rules built-in."""
        # Get RAG context for this group (if available)
        rag_for_group = ""
        if self.rag_context:
            rag_for_group = self.rag_context.get(group_name, "")
            if rag_for_group:
                logger.debug(f"[RAG] Using RAG context for group '{group_name}' ({len(rag_for_group)} chars)")

        # Use the smart prompt builder with full context + RAG
        return build_smart_decision_prompt(
            context=self.context,
            group_name=group_name,
            leaves=batch,
            rag_context=rag_for_group,
        )

    @traceable(name="leaf_decider_call_llm")
    async def _call_llm(self, prompt: str, temperature: float = 0.2) -> dict:
        """
        Call the LLM with the prompt. Returns parsed JSON dict.

        Args:
            prompt: The prompt to send
            temperature: Temperature for generation (higher = more creative)
        """
        try:
            # Use Gemini client - returns already-parsed JSON
            from ..utils.gemini_client import call_gemini
            response = await call_gemini(
                prompt=prompt,
                temperature=temperature,
            )
            # call_gemini returns parsed dict directly
            if isinstance(response, dict):
                return response
            # Fallback: parse if string
            return json.loads(str(response))
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise

    def _parse_response(
        self,
        batch: List[Tuple[str, str]],
        response: dict,
    ) -> List[DecisionResult]:
        """Parse LLM response dict into DecisionResult objects."""
        results = []

        try:
            # Response is already a parsed dict from Gemini
            decisions = response.get("decisions", [])

            # Map decisions to results
            decision_map = {d["index"]: d for d in decisions}

            for i, (path, value) in enumerate(batch):
                if i in decision_map:
                    d = decision_map[i]
                    results.append(DecisionResult(
                        path=path,
                        old_value=value,
                        action=d.get("action", "keep"),
                        new_value=d.get("new_value"),
                        reason=d.get("reason", ""),
                    ))
                else:
                    # Missing decision - default to keep
                    results.append(DecisionResult(
                        path=path,
                        old_value=value,
                        action="keep",
                        new_value=None,
                        reason="no decision in response",
                    ))

        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            logger.debug(f"Response was: {response}")
            # Return keep for all on parse error
            for path, value in batch:
                results.append(DecisionResult(
                    path=path,
                    old_value=value,
                    action="keep",
                    new_value=None,
                    reason=f"parse error: {e}",
                ))

        return results


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def decide_leaf_changes(
    leaves: List[Tuple[str, Any]],
    factsheet: Optional[Dict[str, Any]] = None,
    target_scenario: str = "",
    source_scenario: str = "",
    context: Optional[AdaptationContext] = None,
) -> List[DecisionResult]:
    """
    Decide which leaves to change using LLM with smart prompts.

    Args:
        leaves: List of (path, value) tuples
        factsheet: Global factsheet (legacy, use context instead)
        target_scenario: Target scenario description
        source_scenario: Source scenario description
        context: AdaptationContext with full extracted context (preferred)

    Returns:
        List of DecisionResult objects
    """
    if context is not None:
        # Use provided AdaptationContext
        decider = LeafDecider(context=context)
    elif factsheet is not None:
        # Legacy: convert factsheet to context
        decider = LeafDecider.from_factsheet(
            factsheet=factsheet,
            target_scenario=target_scenario,
            source_scenario=source_scenario,
        )
    else:
        raise ValueError("Either context or factsheet must be provided")

    return await decider.decide_all(leaves)


def get_changes_only(decisions: List[DecisionResult]) -> List[DecisionResult]:
    """Filter to only decisions that result in changes."""
    return [d for d in decisions if d.should_change]


def get_decision_stats(decisions: List[DecisionResult]) -> Dict[str, Any]:
    """Get statistics about decisions."""
    stats = {
        "total": len(decisions),
        "keep": 0,
        "replace": 0,
        "pre_filtered": 0,
        "llm_decided": 0,
    }

    for d in decisions:
        if d.action == "keep":
            stats["keep"] += 1
        else:
            stats["replace"] += 1

        if "pre-filter" in d.reason:
            stats["pre_filtered"] += 1
        else:
            stats["llm_decided"] += 1

    return stats
