"""
Stage 4: Scoped Validators

Validates EACH SHARD independently (not full JSON).

Validators:
1. StructureIntegrityValidator - All required fields present
2. IDPreservationValidator - All IDs unchanged from base
3. DomainFidelityValidator - KPIs valid for industry (RAG-assisted)
4. EntityRemovalValidator - Old company names removed
5. ContentCompletenessValidator - No empty required fields
6. ToneValidator - Professional language

Each shard validated ONLY against its own rules.
NO reprocessing of full JSON.
"""
import os
import json
import logging
import asyncio
import re
from typing import Any, Optional
from dataclasses import dataclass, field

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel
from langsmith import traceable

from .base import (
    BaseValidator,
    ValidationResult,
    ValidationIssue,
    ValidationSeverity,
)

logger = logging.getLogger(__name__)

# GPT model for LLM-based validation
VALIDATION_MODEL = os.getenv("VALIDATION_MODEL", "gpt-5.2-2025-12-11")

# Semaphore for controlling concurrent LLM calls (prevents rate limiting)
MAX_CONCURRENT_VALIDATION_CALLS = int(os.getenv("MAX_CONCURRENT_LLM_CALLS", "10"))
_validation_semaphore = None  # Lazy initialization


def _get_validation_semaphore():
    """Get or create validation semaphore (must be called in async context)."""
    global _validation_semaphore
    if _validation_semaphore is None:
        _validation_semaphore = asyncio.Semaphore(MAX_CONCURRENT_VALIDATION_CALLS)
    return _validation_semaphore


def _get_validation_llm():
    """Get OpenAI client for validation."""
    import httpx

    # Create custom httpx client with longer timeout
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(600.0, connect=60.0)  # 10 min read, 1 min connect
    )

    return ChatOpenAI(
        model=VALIDATION_MODEL,
        temperature=0.1,
        max_retries=2,
        request_timeout=600,  # 10 min
        http_async_client=http_client,
        api_key=os.getenv("OPENAI_API_KEY"),
    )


# =============================================================================
# PYDANTIC MODELS FOR LLM OUTPUT
# =============================================================================

class StructureCheckResponse(BaseModel):
    """Response from structure check."""
    is_valid: bool
    missing_fields: list[str]
    extra_fields: list[str]
    type_mismatches: list[str]
    confidence: float


class IDCheckResponse(BaseModel):
    """Response from ID preservation check."""
    all_preserved: bool
    changed_ids: list[str]
    missing_ids: list[str]


class DomainCheckResponse(BaseModel):
    """Response from domain fidelity check."""
    is_valid: bool
    invalid_terms: list[str]
    invalid_kpis: list[str]
    suggestions: list[str]
    confidence: float


class EntityCheckResponse(BaseModel):
    """Response from entity removal check."""
    all_removed: bool
    found_old_entities: list[str]
    locations: list[str]


class CompletenessCheckResponse(BaseModel):
    """Response from completeness check."""
    is_complete: bool
    empty_fields: list[str]
    placeholder_fields: list[str]


class ToneCheckResponse(BaseModel):
    """Response from tone check."""
    is_professional: bool
    issues: list[str]
    suggestions: list[str]


class BatchedCheckIssue(BaseModel):
    """Single issue from batched check."""
    check_type: str  # domain_fidelity, context_fidelity, resource_self_contained, data_consistency, realism
    severity: str  # blocker, warning, info
    message: str
    location: str  # JSON Pointer path
    current_value: str = ""
    suggestion: str = ""


class BatchedCheckFix(BaseModel):
    """Single fix from batched check."""
    path: str  # JSON Pointer path (e.g., "/topicWizardData/simulationFlow/0/data/name")
    old_value: Any = None  # Current value at this path
    new_value: Any = None  # New value to set (optional - LLM sometimes omits)
    reason: str = ""  # Why this fix is needed (optional)
    check_type: str = "unknown"  # Which check (optional with default)


class BatchedCheckResponse(BaseModel):
    """Response from batched shard checker."""
    issues: list[BatchedCheckIssue] = []
    fixes: list[BatchedCheckFix] = []
    summary: str = "No issues found"  # Default when LLM returns minimal response
    overall_score: float = 1.0  # 0.0 to 1.0, default to perfect if no issues


# =============================================================================
# VALIDATORS
# =============================================================================

class StructureIntegrityValidator(BaseValidator):
    """
    Check that shard structure matches expected schema.

    Validates:
    - All required fields present
    - Correct field types (array vs object)
    - No unexpected extra fields
    """

    name = "StructureIntegrity"
    description = "Validates shard structure matches schema"
    is_blocker = False  # Downgraded: structure changes are warnings, not blockers

    async def validate(self, shard: Any, context: dict) -> ValidationResult:
        issues = []

        # Get expected structure from base shard if available
        base_shard = context.get("base_shard")

        if not base_shard:
            # No base to compare, just check for basic structure
            return self._create_result(
                shard_id=shard.id if hasattr(shard, 'id') else "unknown",
                passed=True,
                score=1.0,
                details={"note": "No base shard for comparison"}
            )

        # Compare structures
        base_content = base_shard.content if hasattr(base_shard, 'content') else base_shard
        current_content = shard.content if hasattr(shard, 'content') else shard

        # Check for missing keys
        missing = self._find_missing_keys(base_content, current_content)
        for key in missing:
            issues.append(self._create_issue(
                message=f"Missing required field: {key}",
                location=key,
                severity=ValidationSeverity.BLOCKER,
                expected_value="present",
                current_value="missing"
            ))

        # Check for type mismatches
        mismatches = self._find_type_mismatches(base_content, current_content)
        for key, (expected, actual) in mismatches.items():
            issues.append(self._create_issue(
                message=f"Type mismatch for {key}: expected {expected}, got {actual}",
                location=key,
                severity=ValidationSeverity.BLOCKER,
                expected_value=expected,
                current_value=actual
            ))

        passed = len([i for i in issues if i.severity == ValidationSeverity.BLOCKER]) == 0
        score = 1.0 - (len(issues) * 0.1)

        return self._create_result(
            shard_id=shard.id if hasattr(shard, 'id') else "unknown",
            passed=passed,
            score=max(0.0, score),
            issues=issues,
            details={"missing_count": len(missing), "mismatch_count": len(mismatches)}
        )

    def _find_missing_keys(self, base: Any, current: Any, path: str = "") -> list[str]:
        """Find keys in base that are missing from current."""
        missing = []

        if isinstance(base, dict) and isinstance(current, dict):
            for key in base:
                new_path = f"{path}.{key}" if path else key
                if key not in current:
                    missing.append(new_path)
                else:
                    missing.extend(self._find_missing_keys(base[key], current[key], new_path))
        elif isinstance(base, list) and isinstance(current, list):
            # Check first item structure if both have items
            if base and current:
                missing.extend(self._find_missing_keys(base[0], current[0], f"{path}[0]"))

        return missing

    def _find_type_mismatches(self, base: Any, current: Any, path: str = "") -> dict:
        """Find type mismatches between base and current."""
        mismatches = {}

        if type(base) != type(current):
            if path:
                mismatches[path] = (type(base).__name__, type(current).__name__)
            return mismatches

        if isinstance(base, dict) and isinstance(current, dict):
            for key in base:
                if key in current:
                    new_path = f"{path}.{key}" if path else key
                    mismatches.update(self._find_type_mismatches(base[key], current[key], new_path))

        return mismatches


class IDPreservationValidator(BaseValidator):
    """
    Check that all IDs are preserved from base shard.

    IDs should NEVER change during adaptation.
    """

    name = "IDPreservation"
    description = "Validates all IDs are preserved"
    is_blocker = True

    async def validate(self, shard: Any, context: dict) -> ValidationResult:
        issues = []

        # Get base IDs
        base_ids = context.get("base_ids", set())
        if hasattr(shard, 'extracted_ids'):
            base_ids = shard.extracted_ids

        # Extract current IDs
        current_content = shard.content if hasattr(shard, 'content') else shard
        current_ids = self._extract_ids(current_content)

        # Check for changed/missing IDs
        base_id_set = set(str(id) for id in base_ids)
        current_id_set = set(str(id) for id in current_ids)

        missing_ids = base_id_set - current_id_set

        for id_val in missing_ids:
            issues.append(self._create_issue(
                message=f"ID missing or changed: {id_val}",
                location="id_field",
                severity=ValidationSeverity.BLOCKER,
                expected_value=id_val,
                current_value="missing",
                suggestion="IDs must never change during adaptation"
            ))

        passed = len(missing_ids) == 0
        score = 1.0 if passed else max(0.0, 1.0 - (len(missing_ids) / max(len(base_id_set), 1)))

        return self._create_result(
            shard_id=shard.id if hasattr(shard, 'id') else "unknown",
            passed=passed,
            score=score,
            issues=issues,
            details={
                "base_id_count": len(base_id_set),
                "current_id_count": len(current_id_set),
                "missing_count": len(missing_ids)
            }
        )

    def _extract_ids(self, content: Any, ids: set = None) -> set:
        """Recursively extract all ID values from content."""
        if ids is None:
            ids = set()

        if isinstance(content, dict):
            for key, value in content.items():
                if key.lower() in ('id', 'uid', '_id', 'uuid'):
                    ids.add(str(value))
                elif key.lower().endswith('id') or key.lower().endswith('Id'):
                    ids.add(str(value))
                else:
                    self._extract_ids(value, ids)
        elif isinstance(content, list):
            for item in content:
                self._extract_ids(item, ids)

        return ids


class DomainFidelityValidator(BaseValidator):
    """
    Check that KPIs and terminology match the target industry.

    Uses RAG for industry-specific validation.
    """

    name = "DomainFidelity"
    description = "Validates KPIs and terminology for industry"
    is_blocker = True
    applicable_shards = ["resources", "simulation_flow", "workplace_scenario"]

    def __init__(self):
        super().__init__()
        self.llm = _get_validation_llm()

    @traceable
    async def validate(self, shard: Any, context: dict) -> ValidationResult:
        issues = []

        industry = context.get("industry", "unknown")
        content = shard.content if hasattr(shard, 'content') else shard

        # Use RAG to get valid KPIs for industry
        try:
            from ..rag import get_industry_context, is_valid_kpi_for_industry
            industry_ctx = get_industry_context(industry)
            valid_kpis = industry_ctx.kpis
        except ImportError:
            valid_kpis = []

        # Extract KPIs mentioned in content
        content_text = json.dumps(content) if isinstance(content, (dict, list)) else str(content)
        mentioned_kpis = self._extract_kpis(content_text)

        # Check each KPI
        invalid_kpis = []
        for kpi in mentioned_kpis:
            try:
                if not is_valid_kpi_for_industry(kpi, industry):
                    invalid_kpis.append(kpi)
            except Exception:
                pass  # Skip if RAG not available

        for kpi in invalid_kpis:
            issues.append(self._create_issue(
                message=f"KPI '{kpi}' may not be appropriate for {industry} industry",
                location="content",
                severity=ValidationSeverity.WARNING,
                current_value=kpi,
                suggestion=f"Consider using industry-specific KPIs: {', '.join(valid_kpis[:5])}"
            ))

        passed = len([i for i in issues if i.severity == ValidationSeverity.BLOCKER]) == 0
        score = 1.0 - (len(invalid_kpis) * 0.1)

        return self._create_result(
            shard_id=shard.id if hasattr(shard, 'id') else "unknown",
            passed=passed,
            score=max(0.0, score),
            issues=issues,
            details={
                "industry": industry,
                "kpis_checked": len(mentioned_kpis),
                "invalid_kpis": len(invalid_kpis)
            }
        )

    def _extract_kpis(self, text: str) -> list[str]:
        """Extract potential KPI mentions from text."""
        # Common KPI patterns
        kpi_patterns = [
            r'(?:occupancy|load factor|yield|ADR|RevPAR|CASK|RASK)',
            r'(?:conversion rate|churn rate|NPS|satisfaction)',
            r'(?:revenue|cost|margin|profit)',
            r'\b[A-Z]{2,5}\b',  # Acronyms
        ]

        found = set()
        for pattern in kpi_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            found.update(m.lower() for m in matches)

        return list(found)[:20]  # Limit


class EntityRemovalValidator(BaseValidator):
    """
    Check that old scenario entities are removed.

    Uses poison list from factsheet.
    """

    name = "EntityRemoval"
    description = "Validates old entity names are removed"
    is_blocker = True

    @traceable
    async def validate(self, shard: Any, context: dict) -> ValidationResult:
        issues = []

        # Get poison list from factsheet
        factsheet = context.get("global_factsheet", {})
        poison_list = factsheet.get("poison_list", [])
        source_scenario = context.get("source_scenario", "")

        # Extract source company name if not in poison list
        if source_scenario and not poison_list:
            # Try to extract company name from source
            match = re.search(r'([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)', source_scenario)
            if match:
                poison_list = [match.group(1)]

        content = shard.content if hasattr(shard, 'content') else shard
        content_text = json.dumps(content) if isinstance(content, (dict, list)) else str(content)
        content_lower = content_text.lower()

        # Check for poison terms
        found_terms = []
        for term in poison_list:
            if term.lower() in content_lower:
                found_terms.append(term)
                issues.append(self._create_issue(
                    message=f"Old entity '{term}' still present in content",
                    location="content",
                    severity=ValidationSeverity.BLOCKER,
                    current_value=term,
                    expected_value="removed",
                    suggestion=f"Replace '{term}' with target scenario equivalent"
                ))

        passed = len(found_terms) == 0
        score = 1.0 if passed else max(0.0, 1.0 - (len(found_terms) * 0.2))

        return self._create_result(
            shard_id=shard.id if hasattr(shard, 'id') else "unknown",
            passed=passed,
            score=score,
            issues=issues,
            details={
                "poison_terms_checked": len(poison_list),
                "found_count": len(found_terms)
            }
        )


class ContentCompletenessValidator(BaseValidator):
    """
    Check that no required fields are empty or placeholders.

    IMPORTANT: Only flags fields that became empty DURING adaptation.
    Pre-existing empty fields in the base JSON are IGNORED.

    Also checks for:
    - HTML tag placeholders (<p>, <ol>, <ul> with no content)
    - Truncated sentences (ending mid-word)
    """

    name = "ContentCompleteness"
    description = "Validates no empty or placeholder content"
    is_blocker = True  # Changed to blocker - these are critical issues

    PLACEHOLDER_PATTERNS = [
        r'\[.*?\]',  # [PLACEHOLDER]
        r'\{.*?\}',  # {placeholder}
        r'TODO',
        r'TBD',
        r'FIXME',
        r'XXX',
        r'lorem ipsum',
    ]

    # CRITICAL BLOCKER patterns - these indicate completely broken content
    BLOCKER_PLACEHOLDER_PATTERNS = [
        r'\[industry-specific[^\]]*\]',   # [industry-specific metric] - unresolved template
        r'\[insert[^\]]*\]',              # [insert X here]
        r'\[placeholder[^\]]*\]',         # [placeholder]
        r'\[TBD[^\]]*\]',                 # [TBD]
        r'\[TODO[^\]]*\]',                # [TODO]
        r'\[REPLACE[^\]]*\]',             # [REPLACE WITH...]
    ]

    # HTML tag placeholders - standalone tags with no content
    HTML_PLACEHOLDER_PATTERNS = [
        r'<p>\s*</p>',           # Empty paragraph
        r'<ol>\s*</ol>',         # Empty ordered list
        r'<ul>\s*</ul>',         # Empty unordered list
        r'<div>\s*</div>',       # Empty div
        r'<li>\s*</li>',         # Empty list item
        r'<span>\s*</span>',     # Empty span
        r'(?<![a-zA-Z])<p>(?!\s*[a-zA-Z<])',   # <p> not followed by content
        r'(?<![a-zA-Z])<ol>(?!\s*[a-zA-Z<])',  # <ol> not followed by content
        r'(?<![a-zA-Z])<ul>(?!\s*[a-zA-Z<])',  # <ul> not followed by content
    ]

    # Truncated sentence patterns - sentences ending mid-word
    TRUNCATED_PATTERNS = [
        r'\b[a-zA-Z]{1,3}\s*["\']?\s*$',      # Ends with 1-3 letter word at string end
        r'\b[a-zA-Z]+\-\s*$',                  # Ends with hyphen mid-word
        r'\b[a-zA-Z]+\.\.\.\s*$',              # Ends with ellipsis (might be intentional)
        r'\b(?:the|a|an|to|of|in|for|and|or|is|are|was|were)\s*$',  # Ends with article/preposition
    ]

    async def validate(self, shard: Any, context: dict) -> ValidationResult:
        issues = []

        content = shard.content if hasattr(shard, 'content') else shard

        # Get base shard to compare - only flag NEWLY empty fields
        base_shard = context.get("base_shard")
        base_content = None
        if base_shard:
            base_content = base_shard.content if hasattr(base_shard, 'content') else base_shard

        # Find base empty fields (to exclude from validation)
        base_empty_fields = set()
        if base_content:
            base_empty_fields = set(self._find_empty_fields(base_content))

        # Find empty fields in current content
        current_empty_fields = self._find_empty_fields(content)

        # Only flag fields that are NEWLY empty (weren't empty in base)
        newly_empty_fields = [f for f in current_empty_fields if f not in base_empty_fields]

        for field_path in newly_empty_fields:
            issues.append(self._create_issue(
                message=f"Empty required field: {field_path}",
                location=field_path,
                severity=ValidationSeverity.WARNING,
                current_value="empty",
                expected_value="non-empty content"
            ))

        # Find placeholder content
        content_text = json.dumps(content) if isinstance(content, (dict, list)) else str(content)

        # First check for CRITICAL blocker placeholders
        blocker_placeholders = self._find_blocker_placeholders(content_text)
        for bp in blocker_placeholders:
            issues.append(self._create_issue(
                message=f"CRITICAL: Unresolved template placeholder found: {bp}",
                location="content",
                severity=ValidationSeverity.BLOCKER,
                current_value=bp,
                suggestion="This placeholder was never resolved - content generation failed"
            ))

        # Then check regular placeholders
        placeholders = self._find_placeholders(content_text)
        for placeholder in placeholders:
            # Skip if already caught as blocker
            if any(bp in placeholder for bp in blocker_placeholders):
                continue
            issues.append(self._create_issue(
                message=f"Placeholder content found: {placeholder}",
                location="content",
                severity=ValidationSeverity.WARNING,
                current_value=placeholder,
                suggestion="Replace placeholder with actual content"
            ))

        # Find HTML tag placeholders (BLOCKER - broken content)
        html_placeholders = self._find_html_placeholders(content_text)
        for html_ph in html_placeholders:
            issues.append(self._create_issue(
                message=f"HTML placeholder/broken tag found: {html_ph}",
                location="content",
                severity=ValidationSeverity.BLOCKER,
                current_value=html_ph,
                suggestion="Remove empty HTML tags or add proper content inside them"
            ))

        # Find truncated sentences (BLOCKER - incomplete content)
        truncated = self._find_truncated_sentences(content)
        for path, text_preview in truncated:
            issues.append(self._create_issue(
                message=f"Truncated/incomplete sentence detected",
                location=path,
                severity=ValidationSeverity.BLOCKER,
                current_value=text_preview,
                suggestion="Complete the sentence - it appears to be cut off mid-word"
            ))

        # Check for sparse resource content (WARNING - content too thin for learning)
        shard_id = shard.id if hasattr(shard, 'id') else ""
        if "resource" in shard_id.lower():
            sparse_resources = self._find_sparse_resources(content)
            for path, word_count in sparse_resources:
                issues.append(self._create_issue(
                    message=f"Sparse resource content: only {word_count} words (minimum 300 recommended)",
                    location=path,
                    severity=ValidationSeverity.WARNING,
                    current_value=f"{word_count} words",
                    expected_value="300+ words with statistics and citations",
                    suggestion="Resources must be comprehensive with statistics, sources, and actionable content"
                ))

        passed = len([i for i in issues if i.severity == ValidationSeverity.BLOCKER]) == 0
        score = 1.0 - (len(issues) * 0.1)

        return self._create_result(
            shard_id=shard.id if hasattr(shard, 'id') else "unknown",
            passed=passed,
            score=max(0.0, score),
            issues=issues,
            details={
                "newly_empty_fields": len(newly_empty_fields),
                "pre_existing_empty_fields": len(base_empty_fields),
                "placeholders": len(placeholders),
                "note": "Only newly emptied fields are flagged; pre-existing empty fields are ignored"
            }
        )

    def _find_empty_fields(self, content: Any, path: str = "") -> list[str]:
        """Find empty string or null fields."""
        empty = []

        if isinstance(content, dict):
            for key, value in content.items():
                new_path = f"{path}.{key}" if path else key
                if value is None or value == "":
                    empty.append(new_path)
                elif isinstance(value, (dict, list)):
                    empty.extend(self._find_empty_fields(value, new_path))
        elif isinstance(content, list):
            for i, item in enumerate(content):
                empty.extend(self._find_empty_fields(item, f"{path}[{i}]"))

        return empty

    def _find_placeholders(self, text: str) -> list[str]:
        """Find placeholder patterns in text."""
        found = []
        for pattern in self.PLACEHOLDER_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            found.extend(matches[:5])  # Limit per pattern
        return found[:10]  # Total limit

    def _find_blocker_placeholders(self, text: str) -> list[str]:
        """Find CRITICAL blocker placeholders that indicate broken content."""
        found = []
        for pattern in self.BLOCKER_PLACEHOLDER_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            found.extend(matches)
        return list(set(found))[:10]  # Dedupe and limit

    def _find_html_placeholders(self, text: str) -> list[str]:
        """Find HTML tag placeholders (empty or broken tags)."""
        found = []
        for pattern in self.HTML_PLACEHOLDER_PATTERNS:
            try:
                matches = re.findall(pattern, text, re.IGNORECASE)
                found.extend(matches[:3])  # Limit per pattern
            except re.error:
                pass
        return list(set(found))[:5]  # Dedupe and limit

    def _find_truncated_sentences(self, content: Any, path: str = "") -> list[tuple[str, str]]:
        """Find truncated sentences in text fields."""
        truncated = []

        if isinstance(content, dict):
            for key, value in content.items():
                new_path = f"{path}.{key}" if path else key
                if isinstance(value, str) and len(value) > 50:
                    # Check if text ends abruptly
                    text = value.strip()
                    if text:
                        # Check for truncation patterns
                        last_50 = text[-50:]
                        for pattern in self.TRUNCATED_PATTERNS:
                            if re.search(pattern, text):
                                # Make sure it's not a normal sentence ending
                                if not text.endswith(('.', '!', '?', '"', "'", ')', ']')):
                                    truncated.append((new_path, text[-80:] if len(text) > 80 else text))
                                    break
                elif isinstance(value, (dict, list)):
                    truncated.extend(self._find_truncated_sentences(value, new_path))
        elif isinstance(content, list):
            for i, item in enumerate(content):
                truncated.extend(self._find_truncated_sentences(item, f"{path}[{i}]"))

        return truncated[:5]  # Limit total

    def _find_sparse_resources(self, content: Any, path: str = "") -> list[tuple[str, int]]:
        """
        Find resource entries with sparse content (under 300 words).

        Resources should be comprehensive for learners, including:
        - Data tables
        - Statistics with sources
        - Actionable content

        Returns list of (path, word_count) tuples for sparse resources.
        """
        sparse = []
        MIN_RESOURCE_WORDS = 500  # Minimum words for a useful resource (increased for KLO coverage)

        # Fields that should have substantial content in resources
        content_fields = ['markdownText', 'content', 'description', 'text']

        if isinstance(content, dict):
            for key, value in content.items():
                new_path = f"{path}.{key}" if path else key

                # Check content fields for minimum length
                if key.lower() in [f.lower() for f in content_fields]:
                    if isinstance(value, str):
                        word_count = len(value.split())
                        if word_count < MIN_RESOURCE_WORDS and word_count > 10:  # Ignore very short/empty
                            sparse.append((new_path, word_count))
                elif isinstance(value, (dict, list)):
                    sparse.extend(self._find_sparse_resources(value, new_path))

        elif isinstance(content, list):
            for i, item in enumerate(content):
                sparse.extend(self._find_sparse_resources(item, f"{path}[{i}]"))

        return sparse[:10]  # Limit to avoid noise


class ToneValidator(BaseValidator):
    """
    Check that content maintains professional, instructional tone.

    Uses LLM for tone analysis.
    """

    name = "ToneValidator"
    description = "Validates professional tone"
    is_blocker = False
    applicable_shards = ["emails", "workplace_scenario", "simulation_flow"]

    def __init__(self):
        super().__init__()
        self.llm = _get_validation_llm()

    @traceable
    async def validate(self, shard: Any, context: dict) -> ValidationResult:
        content = shard.content if hasattr(shard, 'content') else shard
        content_text = json.dumps(content) if isinstance(content, (dict, list)) else str(content)

        # Only check substantial content
        if len(content_text) < 100:
            return self._create_result(
                shard_id=shard.id if hasattr(shard, 'id') else "unknown",
                passed=True,
                score=1.0,
                details={"note": "Content too short for tone analysis"}
            )

        # Sample content for analysis (limit tokens)
        sample = content_text[:2000]

        prompt = f"""Analyze the TONE of this business simulation content.

CONTENT:
{sample}

Check for:
1. Professional language (not casual/slang)
2. Instructional clarity
3. Appropriate for educational context
4. No inappropriate content

Return JSON with:
- is_professional: boolean
- issues: list of specific tone issues found
- suggestions: list of improvement suggestions"""

        try:
            parser = PydanticOutputParser(pydantic_object=ToneCheckResponse)

            chat_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a tone validator for educational simulations.\n\n{format_instructions}"),
                ("human", "{input}"),
            ])

            chain = chat_prompt | self.llm | parser

            # Use semaphore to prevent rate limiting
            semaphore = _get_validation_semaphore()
            async with semaphore:
                result = await chain.ainvoke({
                    "input": prompt,
                    "format_instructions": parser.get_format_instructions(),
                })

            issues = []
            for issue in result.issues:
                issues.append(self._create_issue(
                    message=issue,
                    location="content",
                    severity=ValidationSeverity.WARNING,
                    suggestion=result.suggestions[0] if result.suggestions else None
                ))

            return self._create_result(
                shard_id=shard.id if hasattr(shard, 'id') else "unknown",
                passed=result.is_professional,
                score=1.0 if result.is_professional else 0.7,
                issues=issues
            )

        except Exception as e:
            logger.warning(f"Tone validation failed: {e}")
            return self._create_result(
                shard_id=shard.id if hasattr(shard, 'id') else "unknown",
                passed=True,
                score=0.8,
                details={"note": f"Tone check skipped: {e}"}
            )


# =============================================================================
# SENDER CONSISTENCY VALIDATOR
# =============================================================================

class SenderConsistencyValidator(BaseValidator):
    """
    Check that email signatures match sender metadata.

    Detects mismatches like:
    - Email signed by "Liam Rodriguez" but sender metadata says "Sophia Chen"
    - Body references different person than From field
    """

    name = "SenderConsistency"
    description = "Validates email sender/signature consistency"
    is_blocker = True
    applicable_shards = ["emails", "simulation_flow"]

    async def validate(self, shard: Any, context: dict) -> ValidationResult:
        issues = []
        shard_id = shard.id if hasattr(shard, 'id') else "unknown"

        content = shard.content if hasattr(shard, 'content') else shard

        # Find all emails in content
        emails = self._find_emails(content)

        for email_path, email_data in emails:
            # Extract sender from metadata
            sender_name = email_data.get("senderName", "")
            sender_email = email_data.get("senderEmail", "")
            from_field = email_data.get("from", "")

            # Extract signature from body
            body = email_data.get("body", "") or email_data.get("content", "")
            signature_names = self._extract_signature_names(body)

            # Check for mismatches
            if sender_name and signature_names:
                sender_first = sender_name.split()[0].lower() if sender_name else ""
                for sig_name in signature_names:
                    sig_first = sig_name.split()[0].lower() if sig_name else ""
                    if sig_first and sender_first and sig_first != sender_first:
                        # Mismatch detected
                        issues.append(self._create_issue(
                            message=f"Sender mismatch: email signed by '{sig_name}' but sender is '{sender_name}'",
                            location=email_path,
                            severity=ValidationSeverity.BLOCKER,
                            current_value=f"Signature: {sig_name}, Sender: {sender_name}",
                            suggestion=f"Update signature to match sender '{sender_name}' or change sender metadata"
                        ))

        passed = len(issues) == 0
        score = 1.0 if passed else max(0.0, 1.0 - (len(issues) * 0.3))

        return self._create_result(
            shard_id=shard_id,
            passed=passed,
            score=score,
            issues=issues,
            details={"emails_checked": len(emails)}
        )

    def _find_emails(self, content: Any, path: str = "") -> list[tuple[str, dict]]:
        """Find all email objects in content."""
        emails = []

        if isinstance(content, dict):
            # Check if this is an email object
            if any(k in content for k in ["body", "subject", "senderName", "senderEmail"]):
                emails.append((path, content))

            # Also check for nested emails
            for key, value in content.items():
                new_path = f"{path}.{key}" if path else key
                if key in ["email", "taskEmail", "secondaryTaskEmail"]:
                    if isinstance(value, dict):
                        emails.append((new_path, value))
                emails.extend(self._find_emails(value, new_path))

        elif isinstance(content, list):
            for i, item in enumerate(content):
                emails.extend(self._find_emails(item, f"{path}[{i}]"))

        return emails

    def _extract_signature_names(self, body: str) -> list[str]:
        """Extract potential signature names from email body."""
        import re

        names = []

        # Common signature patterns
        patterns = [
            r'(?:Best|Regards|Thanks|Sincerely|Cheers)[,\s]*\n+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',  # After salutation
            r'(?:^|\n)([A-Z][a-z]+\s+[A-Z][a-z]+)[,\s]*\n*(?:Senior|Director|Manager|Head|VP|CEO|CTO)',  # Name + Title
            r'(?:signed|from)[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',  # "signed by X"
        ]

        for pattern in patterns:
            matches = re.findall(pattern, body, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                if isinstance(match, str) and len(match) > 3:
                    names.append(match.strip())

        return list(set(names))


# =============================================================================
# NEW VALIDATORS (8-validator suite)
# =============================================================================

class ContextFidelityValidator(BaseValidator):
    """
    Validator #2: Context Fidelity

    Checks that KLO/criteria/question counts match the base.
    If counts don't match, something was lost or incorrectly added during adaptation.

    NO LLM needed - pure count comparison.
    """

    name = "ContextFidelity"
    description = "Validates KLO, criteria, and question counts match base"
    is_blocker = True
    applicable_shards = ["assessment_criteria", "rubrics", "simulation_flow"]

    async def validate(self, shard: Any, context: dict) -> ValidationResult:
        issues = []
        shard_id = shard.id if hasattr(shard, 'id') else "unknown"

        # Get base shard for comparison
        base_shard = context.get("base_shard")
        if not base_shard:
            return self._create_result(
                shard_id=shard_id,
                passed=True,
                score=1.0,
                details={"note": "No base shard for count comparison"}
            )

        content = shard.content if hasattr(shard, 'content') else shard
        base_content = base_shard.content if hasattr(base_shard, 'content') else base_shard

        # Count KLOs
        base_klo_count = self._count_items(base_content, "assessmentCriterion")
        current_klo_count = self._count_items(content, "assessmentCriterion")

        if current_klo_count != base_klo_count:
            issues.append(self._create_issue(
                message=f"KLO count mismatch: expected {base_klo_count}, got {current_klo_count}",
                location="assessmentCriterion",
                severity=ValidationSeverity.BLOCKER,
                expected_value=str(base_klo_count),
                current_value=str(current_klo_count),
                suggestion="Restore missing KLOs from base or remove extra ones"
            ))

        # Count rubric criteria
        base_criteria_count = self._count_nested(base_content, "rubric", "criteria")
        current_criteria_count = self._count_nested(content, "rubric", "criteria")

        if current_criteria_count != base_criteria_count:
            issues.append(self._create_issue(
                message=f"Rubric criteria count mismatch: expected {base_criteria_count}, got {current_criteria_count}",
                location="rubric.criteria",
                severity=ValidationSeverity.BLOCKER,
                expected_value=str(base_criteria_count),
                current_value=str(current_criteria_count),
                suggestion="Restore missing criteria from base or remove extra ones"
            ))

        # Count submission questions
        base_question_count = self._count_nested(base_content, "submission", "questions")
        current_question_count = self._count_nested(content, "submission", "questions")

        if current_question_count != base_question_count:
            issues.append(self._create_issue(
                message=f"Submission question count mismatch: expected {base_question_count}, got {current_question_count}",
                location="submission.questions",
                severity=ValidationSeverity.WARNING,
                expected_value=str(base_question_count),
                current_value=str(current_question_count),
                suggestion="Restore missing questions from base or remove extra ones"
            ))

        passed = len([i for i in issues if i.severity == ValidationSeverity.BLOCKER]) == 0
        score = 1.0 - (len(issues) * 0.2)

        return self._create_result(
            shard_id=shard_id,
            passed=passed,
            score=max(0.0, score),
            issues=issues,
            details={
                "base_klo_count": base_klo_count,
                "current_klo_count": current_klo_count,
                "base_criteria_count": base_criteria_count,
                "current_criteria_count": current_criteria_count,
            }
        )

    def _count_items(self, content: Any, key: str) -> int:
        """Count items in a list field."""
        if isinstance(content, dict):
            if key in content:
                val = content[key]
                return len(val) if isinstance(val, list) else 1
            # Recursively search
            for v in content.values():
                count = self._count_items(v, key)
                if count > 0:
                    return count
        elif isinstance(content, list):
            total = 0
            for item in content:
                total += self._count_items(item, key)
            return total
        return 0

    def _count_nested(self, content: Any, parent_key: str, child_key: str) -> int:
        """Count items in nested structure like rubric.criteria."""
        if isinstance(content, dict):
            if parent_key in content:
                parent = content[parent_key]
                if isinstance(parent, dict) and child_key in parent:
                    val = parent[child_key]
                    return len(val) if isinstance(val, list) else 1
                elif isinstance(parent, list):
                    total = 0
                    for item in parent:
                        if isinstance(item, dict) and child_key in item:
                            val = item[child_key]
                            total += len(val) if isinstance(val, list) else 1
                    return total
            # Recursively search
            for v in content.values():
                count = self._count_nested(v, parent_key, child_key)
                if count > 0:
                    return count
        elif isinstance(content, list):
            total = 0
            for item in content:
                total += self._count_nested(item, parent_key, child_key)
            return total
        return 0


class InferenceIntegrityValidator(BaseValidator):
    """
    Validator #5: Inference Integrity

    Checks that resources contain INPUTS ONLY - no ranges, placeholders, or conclusions.
    Students should infer conclusions themselves.

    NO LLM needed - regex pattern matching.
    """

    name = "InferenceIntegrity"
    description = "Validates resources have inputs only (no ranges, placeholders, conclusions)"
    is_blocker = True
    applicable_shards = ["resources", "simulation_flow"]

    # Import patterns from config
    def __init__(self):
        super().__init__()
        from ..utils.config import INFERENCE_INTEGRITY_PATTERNS
        self.patterns = INFERENCE_INTEGRITY_PATTERNS

    async def validate(self, shard: Any, context: dict) -> ValidationResult:
        issues = []
        shard_id = shard.id if hasattr(shard, 'id') else "unknown"

        content = shard.content if hasattr(shard, 'content') else shard
        content_text = json.dumps(content) if isinstance(content, (dict, list)) else str(content)

        # Check each pattern
        found_violations = []
        for pattern in self.patterns:
            matches = re.findall(pattern, content_text, re.IGNORECASE)
            if matches:
                found_violations.extend(matches[:3])  # Limit per pattern

        # Remove duplicates and limit total
        found_violations = list(set(found_violations))[:10]

        for violation in found_violations:
            # Determine severity based on type
            is_placeholder = any(p in violation.upper() for p in ['TBD', 'TBC', 'N/A', 'XX'])
            is_conclusion = any(w in violation.lower() for w in ['expected', 'projected', 'likely', 'will lead', 'should result'])

            severity = ValidationSeverity.BLOCKER if is_placeholder else ValidationSeverity.WARNING

            issues.append(self._create_issue(
                message=f"Inference integrity violation: '{violation}' - resources should contain inputs only",
                location="resource_content",
                severity=severity,
                current_value=violation,
                suggestion="Convert ranges to specific values, remove conclusions, replace placeholders with actual data"
            ))

        passed = len([i for i in issues if i.severity == ValidationSeverity.BLOCKER]) == 0
        score = 1.0 - (len(issues) * 0.1)

        return self._create_result(
            shard_id=shard_id,
            passed=passed,
            score=max(0.0, score),
            issues=issues,
            details={
                "violations_found": len(found_violations),
                "patterns_checked": len(self.patterns)
            }
        )


class WordCountValidator(BaseValidator):
    """
    Validator #6: Word Count

    Checks that section lengths are within bounds.

    NO LLM needed - simple counting.
    NOT a blocker - warning only.
    """

    name = "WordCount"
    description = "Validates section lengths within bounds"
    is_blocker = False  # Warning only
    applicable_shards = ["simulation_flow", "workplace_scenario", "emails", "resources"]

    def __init__(self):
        super().__init__()
        from ..utils.config import WORD_COUNT_LIMITS
        self.limits = WORD_COUNT_LIMITS

    async def validate(self, shard: Any, context: dict) -> ValidationResult:
        issues = []
        shard_id = shard.id if hasattr(shard, 'id') else "unknown"

        content = shard.content if hasattr(shard, 'content') else shard

        # Check different section types
        sections_checked = self._check_sections(content, issues)

        passed = True  # Word count is never a blocker
        score = 1.0 - (len(issues) * 0.05)

        return self._create_result(
            shard_id=shard_id,
            passed=passed,
            score=max(0.5, score),  # Min 0.5 for word count issues
            issues=issues,
            details={
                "sections_checked": sections_checked,
                "issues_found": len(issues)
            }
        )

    def _check_sections(self, content: Any, issues: list, path: str = "") -> int:
        """Recursively check word counts in sections."""
        count = 0

        if isinstance(content, dict):
            for key, value in content.items():
                new_path = f"{path}.{key}" if path else key

                # Check specific field types
                if key in ('body', 'content', 'text', 'description'):
                    if isinstance(value, str):
                        word_count = len(value.split())
                        section_type = self._detect_section_type(path, key)

                        if section_type in self.limits:
                            limits = self.limits[section_type]
                            if word_count < limits["min"]:
                                issues.append(self._create_issue(
                                    message=f"Section too short: {word_count} words (min: {limits['min']})",
                                    location=new_path,
                                    severity=ValidationSeverity.WARNING,
                                    current_value=str(word_count),
                                    expected_value=f">= {limits['min']}"
                                ))
                            elif word_count > limits["max"]:
                                issues.append(self._create_issue(
                                    message=f"Section too long: {word_count} words (max: {limits['max']})",
                                    location=new_path,
                                    severity=ValidationSeverity.WARNING,
                                    current_value=str(word_count),
                                    expected_value=f"<= {limits['max']}"
                                ))
                        count += 1
                else:
                    count += self._check_sections(value, issues, new_path)
        elif isinstance(content, list):
            for i, item in enumerate(content):
                count += self._check_sections(item, issues, f"{path}[{i}]")

        return count

    def _detect_section_type(self, path: str, field: str) -> str:
        """Detect section type from path for limit lookup."""
        path_lower = path.lower()

        if 'email' in path_lower:
            if 'intro' in path_lower:
                return "intro_email"
            return "task_email"
        elif 'resource' in path_lower:
            return "resource"
        elif 'rubric' in path_lower or 'criteria' in path_lower:
            return "rubric_criteria"
        elif 'workplace' in path_lower or 'scenario' in path_lower:
            return "workplace_scenario"
        elif 'klo' in path_lower or 'assessment' in path_lower:
            return "klo_description"

        return "unknown"


class EnhancedDomainFidelityValidator(BaseValidator):
    """
    Validator #1: Domain Fidelity (ENHANCED)

    Checks that industry terms and KPIs match the target industry.
    Detects WRONG terms (e.g., "CAC" in beverage scenario).

    Uses pattern matching first (fast), then LLM validates in batch.
    """

    name = "DomainFidelity"
    description = "Validates industry terms and KPIs match target"
    is_blocker = True
    applicable_shards = ["rubrics", "resources", "simulation_flow", "workplace_scenario"]

    def __init__(self):
        super().__init__()
        from ..utils.config import WRONG_INDUSTRY_TERMS, CORRECT_INDUSTRY_TERMS
        self.wrong_terms = WRONG_INDUSTRY_TERMS
        self.correct_terms = CORRECT_INDUSTRY_TERMS

    async def validate(self, shard: Any, context: dict) -> ValidationResult:
        issues = []
        shard_id = shard.id if hasattr(shard, 'id') else "unknown"

        industry = context.get("industry", "").lower()
        content = shard.content if hasattr(shard, 'content') else shard
        content_text = json.dumps(content) if isinstance(content, (dict, list)) else str(content)

        # Get wrong terms for this industry
        wrong_terms_list = self.wrong_terms.get(industry, [])

        # Check for wrong terms
        found_wrong_terms = []
        for term in wrong_terms_list:
            # Case-insensitive search with word boundary
            pattern = r'\b' + re.escape(term) + r'\b'
            if re.search(pattern, content_text, re.IGNORECASE):
                found_wrong_terms.append(term)

        # Create issues for wrong terms
        for term in found_wrong_terms:
            # Find correct replacement suggestions
            correct_terms_list = self.correct_terms.get(industry, [])
            suggestion = f"Replace with industry-appropriate term. Suggestions: {', '.join(correct_terms_list[:5])}"

            issues.append(self._create_issue(
                message=f"Wrong industry term '{term}' found - not appropriate for {industry} industry",
                location="content",
                severity=ValidationSeverity.BLOCKER,
                current_value=term,
                suggestion=suggestion
            ))

        passed = len(found_wrong_terms) == 0
        score = 1.0 - (len(found_wrong_terms) * 0.15)

        return self._create_result(
            shard_id=shard_id,
            passed=passed,
            score=max(0.0, score),
            issues=issues,
            details={
                "industry": industry,
                "wrong_terms_found": found_wrong_terms,
                "terms_checked": len(wrong_terms_list)
            }
        )


# =============================================================================
# NEW VALIDATORS (Shweta Requirements)
# =============================================================================

class ResourceAnswerabilityValidator(BaseValidator):
    """
    Validator: Resource Answerability (Shweta Requirement Dec 22)

    "does the resource contain all the information the student needs
    to answer the submission questions"

    Uses LLM to check if each question can be answered from resources ONLY.
    """

    name = "ResourceAnswerability"
    description = "Validates every question can be answered from resources alone"
    is_blocker = True
    applicable_shards = ["simulation_flow", "resources"]

    def __init__(self):
        super().__init__()
        self.llm = _get_validation_llm()

    @traceable(name="resource_answerability_check")
    async def validate(self, shard: Any, context: dict) -> ValidationResult:
        """Check if all questions are answerable from resources."""
        issues = []
        shard_id = shard.id if hasattr(shard, 'id') else "unknown"

        # Extract questions and resources from context
        questions = context.get("questions", [])
        resources = context.get("resources", [])

        # If not in context, try to extract from shard
        content = shard.content if hasattr(shard, 'content') else shard
        if not questions:
            questions = self._extract_questions(content)
        if not resources:
            resources = self._extract_resources(content)

        if not questions or not resources:
            return self._create_result(
                shard_id=shard_id,
                passed=True,
                score=1.0,
                details={"note": "No questions or resources to check"}
            )

        # Build prompt for LLM
        prompt = self._build_prompt(questions, resources)

        try:
            semaphore = _get_validation_semaphore()
            async with semaphore:
                result = await self.llm.ainvoke(prompt)

            # Parse response
            response_text = result.content if hasattr(result, 'content') else str(result)
            unanswerable = self._parse_unanswerable(response_text, questions)

            for q_idx, reason in unanswerable:
                issues.append(self._create_issue(
                    message=f"Question {q_idx + 1} cannot be answered from resources alone",
                    location=f"question_{q_idx}",
                    severity=ValidationSeverity.BLOCKER,
                    current_value=questions[q_idx][:100] if q_idx < len(questions) else "",
                    suggestion=f"Add required data to resources: {reason}"
                ))

            passed = len(issues) == 0
            score = 1.0 - (len(issues) * 0.2)

            return self._create_result(
                shard_id=shard_id,
                passed=passed,
                score=max(0.0, score),
                issues=issues,
                details={
                    "questions_checked": len(questions),
                    "unanswerable_count": len(unanswerable)
                }
            )

        except Exception as e:
            logger.warning(f"Resource answerability check failed: {e}")
            return self._create_result(
                shard_id=shard_id,
                passed=True,
                score=0.8,
                details={"note": f"Check skipped: {e}"}
            )

    def _extract_questions(self, content: Any) -> list[str]:
        """Extract submission questions from content."""
        questions = []

        def search(obj, path=""):
            if isinstance(obj, dict):
                # Look for submission questions
                if "questions" in obj and isinstance(obj["questions"], list):
                    for q in obj["questions"]:
                        if isinstance(q, dict):
                            text = q.get("text") or q.get("question") or q.get("content", "")
                            if text:
                                questions.append(text)
                        elif isinstance(q, str):
                            questions.append(q)
                for k, v in obj.items():
                    search(v, f"{path}.{k}")
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    search(item, f"{path}[{i}]")

        search(content)
        return questions

    def _extract_resources(self, content: Any) -> list[str]:
        """Extract resources from content."""
        resources = []

        def search(obj, path=""):
            if isinstance(obj, dict):
                # Look for resource content
                if "content" in obj and "resource" in path.lower():
                    resources.append(str(obj["content"]))
                elif "body" in obj and "resource" in path.lower():
                    resources.append(str(obj["body"]))
                elif "data" in obj and isinstance(obj["data"], str) and len(obj["data"]) > 100:
                    resources.append(obj["data"])
                for k, v in obj.items():
                    search(v, f"{path}.{k}")
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    search(item, f"{path}[{i}]")

        search(content)
        return resources

    def _build_prompt(self, questions: list[str], resources: list[str]) -> str:
        """Build prompt for answerability check."""
        q_text = "\n".join([f"{i+1}. {q[:300]}" for i, q in enumerate(questions[:10])])
        r_text = "\n---\n".join([r[:1000] for r in resources[:5]])

        return f"""You are validating a business simulation for educational purposes.

RESOURCES PROVIDED TO STUDENT:
{r_text}

QUESTIONS STUDENT MUST ANSWER:
{q_text}

For each question, determine if it can be FULLY answered using ONLY the resources above.
The student should NOT need any external knowledge or data not in the resources.

List any questions that CANNOT be answered from the resources alone.
For each unanswerable question, explain what data is missing.

Format your response as:
UNANSWERABLE:
- Question X: [reason what data is missing]
- Question Y: [reason what data is missing]

If ALL questions are answerable, respond with:
ALL QUESTIONS ANSWERABLE

Be strict - if a question requires specific numbers, dates, or facts not in the resources, it's unanswerable."""

    def _parse_unanswerable(self, response: str, questions: list[str]) -> list[tuple[int, str]]:
        """Parse LLM response to find unanswerable questions."""
        unanswerable = []

        if "ALL QUESTIONS ANSWERABLE" in response.upper():
            return []

        lines = response.split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("- Question") or line.startswith("-Question"):
                # Extract question number
                match = re.search(r'Question\s*(\d+)', line, re.IGNORECASE)
                if match:
                    q_num = int(match.group(1)) - 1  # Convert to 0-indexed
                    reason = line.split(":", 1)[1].strip() if ":" in line else "Missing data"
                    if 0 <= q_num < len(questions):
                        unanswerable.append((q_num, reason))

        return unanswerable


class AnswerLeakageValidator(BaseValidator):
    """
    Validator: Answer Leakage (Shweta Requirement Dec 22)

    "the resource does not have the answer... it should basically have
    all the dots for inference to connect the dots, but it doesn't
    really give the connected dots"

    Resources should provide DATA, not CONCLUSIONS.
    """

    name = "AnswerLeakage"
    description = "Validates resources provide data, not direct answers"
    is_blocker = True
    applicable_shards = ["resources", "simulation_flow"]

    # Patterns that indicate answer leakage
    LEAKAGE_PATTERNS = [
        r'\bthe best (?:option|choice|strategy|approach) is\b',
        r'\bshould choose\b',
        r'\bthe answer is\b',
        r'\bthe solution is\b',
        r'\bclearly (?:the|this) is\b',
        r'\bthe correct (?:answer|response|choice) is\b',
        r'\bthis means (?:that |we should)\b',
        r'\btherefore.{0,20}(?:should|must|recommend)\b',
        r'\bin conclusion.{0,30}(?:best|recommend|choose)\b',
        r'\bthe key takeaway is\b',
        r'\bthe main point is\b',
    ]

    def __init__(self):
        super().__init__()
        self.llm = _get_validation_llm()

    @traceable(name="answer_leakage_check")
    async def validate(self, shard: Any, context: dict) -> ValidationResult:
        """Check if resources leak answers."""
        issues = []
        shard_id = shard.id if hasattr(shard, 'id') else "unknown"

        content = shard.content if hasattr(shard, 'content') else shard
        content_text = json.dumps(content) if isinstance(content, (dict, list)) else str(content)

        # Phase 1: Fast pattern matching
        found_patterns = []
        for pattern in self.LEAKAGE_PATTERNS:
            matches = re.findall(pattern, content_text, re.IGNORECASE)
            if matches:
                found_patterns.extend(matches[:2])

        for pattern_match in found_patterns[:5]:
            issues.append(self._create_issue(
                message=f"Potential answer leakage detected: '{pattern_match}'",
                location="resource_content",
                severity=ValidationSeverity.WARNING,
                current_value=pattern_match,
                suggestion="Provide data/facts only, let students draw conclusions"
            ))

        # Phase 2: LLM check for semantic leakage (only if resources are substantial)
        if len(content_text) > 500:
            questions = context.get("questions", [])
            if questions:
                try:
                    semantic_leaks = await self._check_semantic_leakage(content_text, questions)
                    for leak in semantic_leaks:
                        issues.append(self._create_issue(
                            message=f"Resource directly answers question: {leak}",
                            location="resource_content",
                            severity=ValidationSeverity.BLOCKER,
                            suggestion="Remove direct answers, provide supporting data only"
                        ))
                except Exception as e:
                    logger.warning(f"Semantic leakage check failed: {e}")

        passed = len([i for i in issues if i.severity == ValidationSeverity.BLOCKER]) == 0
        score = 1.0 - (len(issues) * 0.15)

        return self._create_result(
            shard_id=shard_id,
            passed=passed,
            score=max(0.0, score),
            issues=issues,
            details={
                "pattern_matches": len(found_patterns),
                "issues_found": len(issues)
            }
        )

    async def _check_semantic_leakage(self, resource_text: str, questions: list[str]) -> list[str]:
        """Use LLM to check for semantic answer leakage."""
        q_text = "\n".join([f"{i+1}. {q[:200]}" for i, q in enumerate(questions[:5])])

        prompt = f"""You are checking if resources LEAK ANSWERS to student questions.

RESOURCE CONTENT (excerpt):
{resource_text[:2000]}

STUDENT QUESTIONS:
{q_text}

Resources should provide DATA and FACTS that students analyze.
Resources should NOT directly state conclusions or answers.

GOOD: "Option A has 20% ROI, Option B has 15% ROI, Option C has 25% ROI"
BAD: "Option C is clearly the best choice with 25% ROI"

Check if any resource text directly answers any question.
List any answer leakage found.

Format:
LEAKAGE FOUND:
- [description of what answer is leaked]

If no leakage:
NO LEAKAGE FOUND"""

        semaphore = _get_validation_semaphore()
        async with semaphore:
            result = await self.llm.ainvoke(prompt)

        response = result.content if hasattr(result, 'content') else str(result)

        if "NO LEAKAGE FOUND" in response.upper():
            return []

        leaks = []
        lines = response.split("\n")
        for line in lines:
            if line.strip().startswith("-"):
                leaks.append(line.strip()[1:].strip())

        return leaks[:3]  # Limit


class CrossShardAlignmentValidator(BaseValidator):
    """
    Validator: Cross-Shard Alignment

    Checks alignment ACROSS shards after individual processing:
    1. Every KLO has at least one question assessing it
    2. Every question maps to a KLO
    3. Company name is consistent across all shards
    4. Industry terms are consistent

    This runs AFTER all shards are processed.
    """

    name = "CrossShardAlignment"
    description = "Validates alignment across all shards (KLOs ↔ Questions ↔ Resources)"
    is_blocker = True

    @traceable(name="cross_shard_alignment_check")
    async def validate(self, shard: Any, context: dict) -> ValidationResult:
        """
        For cross-shard validation, 'shard' is actually the full adapted JSON.
        Context should contain 'all_shards' for cross-reference.
        """
        issues = []

        # This validator expects the full adapted content
        content = shard.content if hasattr(shard, 'content') else shard

        # Extract components
        klos = self._extract_klos(content)
        questions = self._extract_questions(content)
        resources = self._extract_resources(content)
        company_names = self._extract_company_names(content)

        # Check 1: Every KLO has at least one question
        klo_coverage = self._check_klo_coverage(klos, questions)
        for klo_id, klo_text in klo_coverage.get("uncovered", []):
            issues.append(self._create_issue(
                message=f"KLO not covered by any question: '{klo_text[:50]}...'",
                location=f"klo_{klo_id}",
                severity=ValidationSeverity.BLOCKER,
                suggestion="Add a question that assesses this KLO"
            ))

        # Check 2: Company name consistency
        if len(company_names) > 1:
            main_name = company_names[0]
            for name in company_names[1:]:
                if name.lower() != main_name.lower():
                    issues.append(self._create_issue(
                        message=f"Inconsistent company name: '{name}' vs '{main_name}'",
                        location="company_name",
                        severity=ValidationSeverity.BLOCKER,
                        current_value=name,
                        expected_value=main_name,
                        suggestion=f"Use consistent company name: {main_name}"
                    ))

        # Check 3: Resource count matches expected
        expected_resource_count = context.get("base_resource_count", 0)
        if expected_resource_count > 0 and len(resources) != expected_resource_count:
            issues.append(self._create_issue(
                message=f"Resource count mismatch: expected {expected_resource_count}, got {len(resources)}",
                location="resources",
                severity=ValidationSeverity.WARNING,
                suggestion="Ensure all resources are preserved"
            ))

        passed = len([i for i in issues if i.severity == ValidationSeverity.BLOCKER]) == 0
        score = 1.0 - (len(issues) * 0.15)

        return self._create_result(
            shard_id="cross_shard",
            passed=passed,
            score=max(0.0, score),
            issues=issues,
            details={
                "klo_count": len(klos),
                "question_count": len(questions),
                "resource_count": len(resources),
                "company_names_found": len(company_names)
            }
        )

    def _extract_klos(self, content: Any) -> list[tuple[str, str]]:
        """Extract KLOs (id, text) from content."""
        klos = []

        def search(obj):
            if isinstance(obj, dict):
                if "assessmentCriterion" in obj:
                    items = obj["assessmentCriterion"]
                    if isinstance(items, list):
                        for item in items:
                            if isinstance(item, dict):
                                klo_id = item.get("id", "")
                                text = item.get("text") or item.get("description", "")
                                if text:
                                    klos.append((klo_id, text))
                for v in obj.values():
                    search(v)
            elif isinstance(obj, list):
                for item in obj:
                    search(item)

        search(content)
        return klos

    def _extract_questions(self, content: Any) -> list[str]:
        """Extract questions from content."""
        questions = []

        def search(obj):
            if isinstance(obj, dict):
                if "questions" in obj and isinstance(obj["questions"], list):
                    for q in obj["questions"]:
                        if isinstance(q, dict):
                            text = q.get("text") or q.get("question", "")
                            if text:
                                questions.append(text)
                for v in obj.values():
                    search(v)
            elif isinstance(obj, list):
                for item in obj:
                    search(item)

        search(content)
        return questions

    def _extract_resources(self, content: Any) -> list[str]:
        """Extract resources from content."""
        resources = []

        def search(obj, path=""):
            if isinstance(obj, dict):
                if "resource" in path.lower() and ("content" in obj or "body" in obj):
                    text = obj.get("content") or obj.get("body", "")
                    if text:
                        resources.append(text)
                for k, v in obj.items():
                    search(v, f"{path}.{k}")
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    search(item, f"{path}[{i}]")

        search(content)
        return resources

    def _extract_company_names(self, content: Any) -> list[str]:
        """Extract company names from content."""
        names = []

        def search(obj):
            if isinstance(obj, dict):
                for key in ["companyName", "company_name", "organizationName", "name"]:
                    if key in obj and isinstance(obj[key], str) and len(obj[key]) > 2:
                        # Filter out generic names
                        name = obj[key]
                        if name.lower() not in ["the company", "company", "organization"]:
                            names.append(name)
                for v in obj.values():
                    search(v)
            elif isinstance(obj, list):
                for item in obj:
                    search(item)

        search(content)
        return list(set(names))

    def _check_klo_coverage(self, klos: list[tuple[str, str]], questions: list[str]) -> dict:
        """Check if each KLO is covered by at least one question."""
        uncovered = []

        for klo_id, klo_text in klos:
            # Simple heuristic: check if key words from KLO appear in any question
            klo_words = set(klo_text.lower().split())
            klo_words -= {"the", "a", "an", "to", "and", "or", "of", "in", "for", "is", "are", "will", "be"}

            covered = False
            for q in questions:
                q_words = set(q.lower().split())
                overlap = klo_words & q_words
                if len(overlap) >= 2:  # At least 2 meaningful words overlap
                    covered = True
                    break

            if not covered:
                uncovered.append((klo_id, klo_text))

        return {"uncovered": uncovered}


# =============================================================================
# BATCHED SHARD CHECKER (Key LLM class)
# =============================================================================

class BatchedShardChecker:
    """
    Single LLM call per shard that validates ALL semantic checks + returns fixes.

    Checks:
    - Domain Fidelity (enhanced)
    - Context Fidelity
    - Resource Self-Contained
    - Data Consistency
    - Realism

    Returns issues + JSON Pointer fixes in one response.
    """

    def __init__(self):
        self.llm = _get_validation_llm()

    @traceable(name="batched_shard_check")
    async def check_and_fix(
        self,
        shard: Any,
        base_shard: Any,
        context: dict,
    ) -> tuple[list[ValidationResult], list[dict]]:
        """
        Run all semantic checks on a shard in ONE LLM call.

        Returns:
            (list of ValidationResults, list of fix dicts with JSON Pointer paths)
        """
        shard_id = shard.id if hasattr(shard, 'id') else "unknown"

        content = shard.content if hasattr(shard, 'content') else shard
        base_content = base_shard.content if base_shard and hasattr(base_shard, 'content') else base_shard

        industry = context.get("industry", "unknown")
        company_name = context.get("company_name", "")

        # Build the batched prompt
        prompt = self._build_prompt(content, base_content, industry, company_name, shard_id)

        try:
            parser = PydanticOutputParser(pydantic_object=BatchedCheckResponse)

            chat_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a validation expert for educational business simulations.
Content is for undergraduate (UG) business students.

Your job is to:
1. Run ALL semantic validation checks
2. Identify ALL issues with precise JSON Pointer paths
3. Provide FIXES for each issue - EVERY fix must include ALL required fields

CRITICAL: Follow the output format EXACTLY. Every fix object MUST have:
- path (JSON Pointer)
- old_value (actual current value)
- new_value (exact replacement)
- reason (why)
- check_type (domain_fidelity/context_fidelity/resource_self_contained/data_consistency/realism)

{format_instructions}"""),
                ("human", "{input}"),
            ])

            chain = chat_prompt | self.llm | parser

            # Use semaphore to prevent rate limiting (12 shards hitting API at once)
            semaphore = _get_validation_semaphore()
            async with semaphore:
                result = await chain.ainvoke({
                    "input": prompt,
                    "format_instructions": parser.get_format_instructions(),
                })

            # Convert to ValidationResults
            validation_results = self._convert_to_results(result, shard_id)

            # Extract fixes
            fixes = [
                {
                    "path": fix.path,
                    "old_value": fix.old_value,
                    "new_value": fix.new_value,
                    "reason": fix.reason,
                    "check_type": fix.check_type,
                }
                for fix in result.fixes
            ]

            return validation_results, fixes

        except Exception as e:
            logger.error(f"Batched check failed for {shard_id}: {e}")
            # Return empty results on failure
            return [], []

    def _build_prompt(
        self,
        content: Any,
        base_content: Any,
        industry: str,
        company_name: str,
        shard_id: str,
    ) -> str:
        """Build the batched validation prompt."""

        # Count items from base for context fidelity
        base_klo_count = self._count_items(base_content, "assessmentCriterion") if base_content else "unknown"
        base_criteria_count = self._count_nested(base_content, "rubric", "criteria") if base_content else "unknown"
        base_question_count = self._count_nested(base_content, "submission", "questions") if base_content else "unknown"

        from ..utils.config import WRONG_INDUSTRY_TERMS, CORRECT_INDUSTRY_TERMS
        wrong_terms = WRONG_INDUSTRY_TERMS.get(industry.lower(), [])
        correct_terms = CORRECT_INDUSTRY_TERMS.get(industry.lower(), [])

        prompt = f"""## SHARD: {shard_id}
## TARGET INDUSTRY: {industry}
## COMPANY: {company_name}

---

## RUN ALL CHECKS:

### 1. DOMAIN FIDELITY
Check if ALL KPIs and terminology match {industry} industry.
WRONG terms for {industry} (should NOT appear): {', '.join(wrong_terms[:15])}
CORRECT terms for {industry}: {', '.join(correct_terms[:15])}
Flag ANY wrong terms found - these are BLOCKERS.

### 2. CONTEXT FIDELITY
Verify counts match base:
- Expected KLO count: {base_klo_count}
- Expected criteria count: {base_criteria_count}
- Expected question count: {base_question_count}
If counts don't match, identify what's missing or extra.

### 3. RESOURCE SELF-CONTAINED
Can EVERY submission question be answered using ONLY the resources provided?
If a question requires external knowledge not in resources, flag it.

### 4. DATA CONSISTENCY
Do numbers/data match across sections?
- Same revenue figures in resources and rubrics?
- Same market sizes mentioned consistently?
- Same percentages/growth rates?

### 5. REALISM
Are numbers and timelines plausible?
- Market sizes appropriate for company scale?
- Growth rates realistic for industry?
- Timelines reasonable?

---

## CURRENT CONTENT:
```json
{json.dumps(content, indent=2)[:8000]}
```

{f'''## BASE CONTENT (for count comparison):
```json
{json.dumps(base_content, indent=2)[:4000]}
```''' if base_content else ''}

---

## OUTPUT FORMAT (STRICT - follow exactly):

### Issues array - each issue MUST have:
```json
{{
  "check_type": "domain_fidelity",  // ONE OF: domain_fidelity, context_fidelity, resource_self_contained, data_consistency, realism
  "severity": "blocker",            // ONE OF: blocker, warning
  "message": "Wrong term found - not appropriate for {industry}",
  "location": "/topicWizardData/rubric/criteria/0/text",  // JSON Pointer path
  "current_value": "the wrong term",
  "suggestion": "Replace with {industry}-appropriate term"
}}
```

### Fixes array - each fix MUST have ALL these fields:
```json
{{
  "path": "/topicWizardData/rubric/criteria/0/text",  // JSON Pointer path (REQUIRED)
  "old_value": "the actual current text",              // Current value (REQUIRED)
  "new_value": "the corrected text for {industry}",   // Corrected value (REQUIRED)
  "reason": "Term not appropriate for {industry} industry",  // Why (REQUIRED)
  "check_type": "domain_fidelity"                     // Which check found this (REQUIRED)
}}
```

### CRITICAL RULES:
1. JSON Pointer paths use FORWARD SLASHES: /topicWizardData/field/0/subfield
2. Every fix MUST include check_type field
3. old_value must be the ACTUAL current value from the content, not a description
4. new_value must be the EXACT replacement text to use
5. Use {industry}-appropriate terminology: {', '.join(correct_terms[:5]) if correct_terms else 'industry standard terms'}
"""
        return prompt

    def _convert_to_results(self, response: BatchedCheckResponse, shard_id: str) -> list[ValidationResult]:
        """Convert batched response to ValidationResults."""
        results_by_type = {}

        for issue in response.issues:
            check_type = issue.check_type
            if check_type not in results_by_type:
                results_by_type[check_type] = []

            severity = ValidationSeverity.BLOCKER if issue.severity == "blocker" else ValidationSeverity.WARNING

            results_by_type[check_type].append(ValidationIssue(
                rule_id=check_type,
                message=issue.message,
                location=issue.location,
                severity=severity,
                current_value=issue.current_value,
                suggestion=issue.suggestion,
            ))

        # Create ValidationResult for each check type
        validation_results = []
        for check_type, issues in results_by_type.items():
            blocker_count = len([i for i in issues if i.severity == ValidationSeverity.BLOCKER])
            warning_count = len([i for i in issues if i.severity == ValidationSeverity.WARNING])

            validation_results.append(ValidationResult(
                validator_name=f"Batched_{check_type}",
                shard_id=shard_id,
                passed=blocker_count == 0,
                score=max(0.0, 1.0 - (blocker_count * 0.2) - (warning_count * 0.05)),
                issues=issues,
                details={"blocker_count": blocker_count, "warning_count": warning_count}
            ))

        # If no issues, create a passing result
        if not validation_results:
            validation_results.append(ValidationResult(
                validator_name="BatchedCheck",
                shard_id=shard_id,
                passed=True,
                score=response.overall_score,
                issues=[],
            ))

        return validation_results

    def _count_items(self, content: Any, key: str) -> int:
        """Count items in a list field."""
        if isinstance(content, dict):
            if key in content:
                val = content[key]
                return len(val) if isinstance(val, list) else 1
            for v in content.values():
                count = self._count_items(v, key)
                if count > 0:
                    return count
        elif isinstance(content, list):
            total = 0
            for item in content:
                total += self._count_items(item, key)
            return total
        return 0

    def _count_nested(self, content: Any, parent_key: str, child_key: str) -> int:
        """Count items in nested structure."""
        if isinstance(content, dict):
            if parent_key in content:
                parent = content[parent_key]
                if isinstance(parent, dict) and child_key in parent:
                    val = parent[child_key]
                    return len(val) if isinstance(val, list) else 1
                elif isinstance(parent, list):
                    total = 0
                    for item in parent:
                        if isinstance(item, dict) and child_key in item:
                            val = item[child_key]
                            total += len(val) if isinstance(val, list) else 1
                    return total
            for v in content.values():
                count = self._count_nested(v, parent_key, child_key)
                if count > 0:
                    return count
        elif isinstance(content, list):
            total = 0
            for item in content:
                total += self._count_nested(item, parent_key, child_key)
            return total
        return 0


# =============================================================================
# SCOPED VALIDATOR ORCHESTRATOR
# =============================================================================

@dataclass
class ScopedValidationReport:
    """Complete validation report for all shards."""
    overall_score: float
    passed: bool
    shard_results: dict[str, list[ValidationResult]] = field(default_factory=dict)
    blocker_count: int = 0
    warning_count: int = 0

    def to_dict(self) -> dict:
        return {
            "overall_score": round(self.overall_score, 4),
            "passed": self.passed,
            "blocker_count": self.blocker_count,
            "warning_count": self.warning_count,
            "shards": {
                shard_id: [r.to_dict() for r in results]
                for shard_id, results in self.shard_results.items()
            }
        }


class ScopedValidator:
    """
    Stage 4: Orchestrates scoped validation across all shards.

    Each shard is validated independently by applicable validators.

    Validators run in two phases:
    1. FAST validators (non-LLM): Structure, ID, Inference, WordCount, ContentCompleteness
    2. BATCHED LLM validators: Domain Fidelity, Context Fidelity, Resource Self-Contained, Data Consistency, Realism
    """

    def __init__(self, use_batched_checker: bool = True):
        # Initialize FAST validators (no LLM)
        self.fast_validators = [
            StructureIntegrityValidator(),
            IDPreservationValidator(),
            ContentCompletenessValidator(),
            ContextFidelityValidator(),
            InferenceIntegrityValidator(),
            WordCountValidator(),
            EnhancedDomainFidelityValidator(),  # Fast pattern matching
        ]

        # Legacy validators (kept for backward compatibility)
        self.validators = self.fast_validators + [
            EntityRemovalValidator(),
            ToneValidator(),
        ]

        # Batched checker for semantic validation (1 LLM call per shard)
        self.use_batched_checker = use_batched_checker
        self.batched_checker = BatchedShardChecker() if use_batched_checker else None

    async def validate_shard(
        self,
        shard: Any,
        context: dict,
        run_batched: bool = True,
    ) -> tuple[list[ValidationResult], list[dict]]:
        """
        Validate a single shard with all applicable validators.

        Two phases:
        1. FAST validators (parallel, no LLM)
        2. BATCHED LLM check (if enabled)

        Args:
            shard: Shard to validate
            context: Validation context
            run_batched: Whether to run batched LLM check

        Returns:
            (List of ValidationResults, List of fixes from batched checker)
        """
        shard_id = shard.id if hasattr(shard, 'id') else "unknown"

        # Add base shard to context for comparison
        base_shards = context.get("base_shards", {})
        shard_context = {**context}
        base_shard = None
        if shard_id in base_shards:
            base_shard = base_shards[shard_id]
            shard_context["base_shard"] = base_shard

        # Phase 1: Run FAST validators in parallel (no LLM)
        applicable_fast = [v for v in self.fast_validators if v.applies_to(shard_id)]
        fast_tasks = [v.validate(shard, shard_context) for v in applicable_fast]
        fast_results = await asyncio.gather(*fast_tasks, return_exceptions=True)

        # Handle exceptions from fast validators
        valid_results = []
        for i, result in enumerate(fast_results):
            if isinstance(result, Exception):
                logger.error(f"Validator {applicable_fast[i].name} failed: {result}")
                valid_results.append(ValidationResult(
                    validator_name=applicable_fast[i].name,
                    shard_id=shard_id,
                    passed=False,
                    score=0.0,
                    issues=[ValidationIssue(
                        rule_id="validator_error",
                        message=str(result),
                        location="validator",
                        severity=ValidationSeverity.WARNING
                    )]
                ))
            else:
                valid_results.append(result)

        # Phase 2: Run BATCHED LLM check (if enabled)
        fixes = []
        if run_batched and self.use_batched_checker and self.batched_checker:
            try:
                batched_results, batched_fixes = await self.batched_checker.check_and_fix(
                    shard, base_shard, shard_context
                )
                valid_results.extend(batched_results)
                fixes = batched_fixes
            except Exception as e:
                logger.error(f"Batched checker failed for {shard_id}: {e}")

        return valid_results, fixes

    async def validate_all(
        self,
        shards: list,
        context: dict,
        run_batched: bool = True,
    ) -> tuple[ScopedValidationReport, dict[str, list[dict]]]:
        """
        Validate all shards in PARALLEL.

        Args:
            shards: List of Shard objects
            context: Validation context (factsheet, scenarios, etc.)
            run_batched: Whether to run batched LLM checks

        Returns:
            (ScopedValidationReport, dict of shard_id -> fixes)
        """
        shard_results = {}
        all_fixes = {}
        all_scores = []
        total_blockers = 0
        total_warnings = 0

        # Filter to unlocked shards
        unlocked_shards = []
        for shard in shards:
            if hasattr(shard, 'lock_state'):
                if shard.lock_state.value == "FULLY_LOCKED":
                    continue
            unlocked_shards.append(shard)

        # Validate ALL shards in PARALLEL (key for low latency)
        async def validate_one(shard):
            shard_id = shard.id if hasattr(shard, 'id') else "unknown"
            results, fixes = await self.validate_shard(shard, context, run_batched)
            return shard_id, results, fixes

        tasks = [validate_one(shard) for shard in unlocked_shards]
        results_list = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for result in results_list:
            if isinstance(result, Exception):
                logger.error(f"Shard validation failed: {result}")
                continue

            shard_id, results, fixes = result
            shard_results[shard_id] = results
            if fixes:
                all_fixes[shard_id] = fixes

            for r in results:
                all_scores.append(r.score)
                total_blockers += r.blocker_count
                total_warnings += r.warning_count

        # Calculate overall score
        overall_score = sum(all_scores) / len(all_scores) if all_scores else 1.0
        passed = total_blockers == 0 and overall_score >= 0.95

        report = ScopedValidationReport(
            overall_score=overall_score,
            passed=passed,
            shard_results=shard_results,
            blocker_count=total_blockers,
            warning_count=total_warnings
        )

        return report, all_fixes


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

async def validate_shards(
    shards: list,
    context: dict,
    run_batched: bool = True,
) -> tuple[ScopedValidationReport, dict[str, list[dict]]]:
    """
    Run scoped validation on all shards with batched LLM checking.

    Args:
        shards: List of Shard objects
        context: Validation context
        run_batched: Whether to run batched LLM checks (default True)

    Returns:
        (ScopedValidationReport, dict of shard_id -> fixes)
    """
    validator = ScopedValidator(use_batched_checker=run_batched)
    return await validator.validate_all(shards, context, run_batched)


async def validate_shards_fast(
    shards: list,
    context: dict,
) -> ScopedValidationReport:
    """
    Run FAST validation only (no LLM calls).

    Use this for quick checks where latency is critical.

    Args:
        shards: List of Shard objects
        context: Validation context

    Returns:
        ScopedValidationReport (no fixes)
    """
    validator = ScopedValidator(use_batched_checker=False)
    report, _ = await validator.validate_all(shards, context, run_batched=False)
    return report
