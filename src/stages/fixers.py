"""
Stage 4B: Scoped Fixers (Hybrid Approach)

LLM identifies exact fields to fix -> outputs JSON Pointer paths
Patcher applies surgical 'replace' operations
Stores old values for rollback

Two types of fixers:
1. Structural Fixer - Fix shape only (missing keys, types)
2. Semantic Fixer - Fix meaning only (KPIs, terminology, entities)

Rules:
- Structural fixes happen FIRST, then lock structure
- Semantic fixes happen SECOND, cannot touch structure
- Max 3 attempts per shard, then flag for human
- All fixes via JSON Pointer patches (only 'replace' ops)
- Rollback capability for repair cycles
"""
import os
import json
import logging
import copy
import asyncio
from typing import Any, Optional
from dataclasses import dataclass, field
from enum import Enum

import httpx
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langsmith import traceable

# Global semaphore for controlling concurrent LLM calls
# This is the KEY to true parallelism - limits concurrent requests to avoid serialization
MAX_CONCURRENT_LLM_CALLS = int(os.getenv("MAX_CONCURRENT_LLM_CALLS", "10"))
_llm_semaphore = None  # Will be initialized lazily in async context


def _get_semaphore():
    """Get or create the semaphore (must be called in async context)."""
    global _llm_semaphore
    if _llm_semaphore is None:
        _llm_semaphore = asyncio.Semaphore(MAX_CONCURRENT_LLM_CALLS)
    return _llm_semaphore

from ..utils.patcher import (
    JSONPatcher,
    PatchOp,
    PatchResult,
    get_patcher,
    create_patch,
)

logger = logging.getLogger(__name__)

# GPT model for fixing - uses LLM to IDENTIFY fixes, not regenerate content
FIXER_MODEL = os.getenv("FIXER_MODEL", "gpt-5.2-2025-12-11")


def _get_fixer_llm():
    """
    Get OpenAI client for fixing.

    CRITICAL: Each call creates a NEW httpx.AsyncClient to enable TRUE parallel execution.
    Without this, all LLM calls share the same connection and run sequentially.
    """
    # Create NEW http client for each LLM instance - enables true parallelism
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(300.0, connect=30.0),  # 5 min read, 30s connect
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=50),  # Higher limits for parallelism
    )

    return ChatOpenAI(
        model=FIXER_MODEL,
        temperature=0.1,  # Low temp for precise field identification
        max_retries=3,
        request_timeout=300,  # 5 minutes for complex KLO alignment
        http_async_client=http_client,  # CRITICAL: Use separate HTTP client
        api_key=os.getenv("OPENAI_API_KEY"),
    )


async def _invoke_with_semaphore(llm, prompt: str) -> str:
    """
    Invoke LLM with semaphore for controlled parallelism.

    This is the KEY to true parallel execution:
    1. Semaphore limits concurrent requests (prevents serialization)
    2. Each call gets its own HTTP client (via _get_llm())
    3. asyncio.gather then runs them truly in parallel
    """
    semaphore = _get_semaphore()
    async with semaphore:
        logger.debug(f"[PARALLEL] Acquired semaphore, invoking LLM...")
        result = await llm.ainvoke(prompt)
        return result.content if hasattr(result, 'content') else str(result)


# =============================================================================
# PYDANTIC MODELS FOR LLM OUTPUT
# =============================================================================

class FieldFix(BaseModel):
    """A single field fix identified by LLM."""
    path: str = Field(description="JSON Pointer path to the field (e.g., '/topicWizardData/simulationName')")
    current_value: Any = Field(description="Current value at this path")
    new_value: Any = Field(description="New value to set")
    reason: str = Field(description="Why this fix is needed")
    fix_type: str = Field(description="Type: 'structural' or 'semantic'")


class FixIdentificationResponse(BaseModel):
    """LLM response identifying fields to fix."""
    fixes: list[FieldFix] = Field(description="List of fields to fix with JSON Pointer paths")
    summary: str = Field(description="Summary of what needs fixing")
    can_fix: bool = Field(description="Whether fixes can be applied automatically")


class StructuralFixResponse(BaseModel):
    """LLM response for structural fixes."""
    fixes: list[FieldFix] = Field(description="List of structural fixes with JSON Pointer paths")
    summary: str = Field(description="Summary of structural issues")


class SemanticFixResponse(BaseModel):
    """LLM response for semantic fixes."""
    fixes: list[FieldFix] = Field(description="List of semantic fixes with JSON Pointer paths")
    replacements: dict = Field(default_factory=dict, description="Entity replacements: old -> new")
    summary: str = Field(description="Summary of semantic issues")


# =============================================================================
# FIX RESULT
# =============================================================================

class FixType(Enum):
    """Type of fix applied."""
    STRUCTURAL = "structural"
    SEMANTIC = "semantic"
    NONE = "none"


@dataclass
class FixResult:
    """Result of fixing a shard."""
    shard_id: str
    fix_type: FixType
    success: bool
    fixed_content: dict = None
    patches_applied: list[PatchOp] = field(default_factory=list)  # For rollback
    changes_made: list[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "shard_id": self.shard_id,
            "fix_type": self.fix_type.value,
            "success": self.success,
            "patches_count": len(self.patches_applied),
            "patches": [p.to_dict() for p in self.patches_applied[:10]],  # Limit output
            "changes": self.changes_made[:10],
            "error": self.error,
        }

    def get_patches_for_rollback(self) -> list[PatchOp]:
        """Get patches that can be rolled back."""
        return [p for p in self.patches_applied if p.old_value is not None]


# =============================================================================
# STRUCTURAL FIXER (Hybrid: LLM identifies, Patcher applies)
# =============================================================================

class StructuralFixer:
    """
    Stage 4B-1: Structural Fixer (Hybrid Approach)

    LLM IDENTIFIES exact fields with structural issues.
    Patcher APPLIES surgical JSON Pointer patches.

    Fixes SHAPE only:
    - Missing required keys
    - Wrong types
    - Broken array structures

    Does NOT change:
    - Text content/wording
    - Values (unless type mismatch)

    Golden rule: "LLM points, Patcher fixes"
    """

    def __init__(self):
        # Don't create shared LLM - get fresh instance for each call
        self.patcher = get_patcher()

    def _get_llm(self):
        """Get fresh LLM instance for parallel execution."""
        return _get_fixer_llm()

    @traceable(name="structural_fixer_identify")
    async def identify_fixes(
        self,
        content: dict,
        issues: list,
        base_content: dict = None,
    ) -> list[FieldFix]:
        """
        Use LLM to identify exact fields that need structural fixes.

        Returns list of FieldFix with JSON Pointer paths.
        """
        # Build issue list for prompt - handle both dict and object formats
        def get_issue_info(issue):
            if isinstance(issue, dict):
                msg = issue.get('message', issue.get('description', str(issue)))
                loc = issue.get('location', 'unknown')
            else:
                msg = getattr(issue, 'message', str(issue))
                loc = getattr(issue, 'location', 'unknown')
            return f"- {msg} at {loc}"

        issue_text = "\n".join([get_issue_info(i) for i in issues])

        prompt = f"""Identify the EXACT fields that need STRUCTURAL fixes in this JSON.

## RULES:
1. Return JSON Pointer paths (RFC 6901) for each field to fix
2. Only structural issues (missing keys, wrong types, broken arrays)
3. DO NOT suggest content/text changes
4. Preserve ALL existing values where possible

## STRUCTURAL ISSUES FOUND:
{issue_text}

## CURRENT JSON:
```json
{json.dumps(content, indent=2)[:6000]}
```

{f'''## EXPECTED STRUCTURE (reference):
```json
{json.dumps(base_content, indent=2)[:3000]}
```''' if base_content else ''}

## OUTPUT:
For each field that needs fixing, provide:
- path: JSON Pointer path (e.g., "/topicWizardData/simulationFlow/0/data/name")
- current_value: What's there now (or null if missing)
- new_value: What it should be (structure fix only)
- reason: Why this fix is needed
- fix_type: "structural"

Only return fields that ACTUALLY need fixing."""

        try:
            parser = PydanticOutputParser(pydantic_object=StructuralFixResponse)

            chat_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a JSON structure analyzer.
Identify exact fields that need structural fixes using JSON Pointer paths.
Only suggest structural changes (shape, types), never content changes.

{format_instructions}"""),
                ("human", "{input}"),
            ])

            chain = chat_prompt | self._get_llm() | parser

            # Use semaphore for controlled parallelism
            semaphore = _get_semaphore()
            async with semaphore:
                try:
                    result = await chain.ainvoke({
                        "input": prompt,
                        "format_instructions": parser.get_format_instructions(),
                    })
                    return result.fixes
                except Exception as parse_error:
                    # LLM returned malformed JSON - skip this fixer
                    logger.warning(f"[STRUCTURAL FIXER]: LLM returned invalid format, skipping: {str(parse_error)[:200]}")
                    return []

        except Exception as e:
            logger.error(f"Failed to identify structural fixes: {e}")
            return []

    @traceable(name="structural_fixer_apply")
    async def fix(
        self,
        shard: Any,
        issues: list,
        context: dict,
    ) -> FixResult:
        """
        Fix structural issues in a shard using hybrid approach.

        1. LLM identifies exact fields (JSON Pointer paths)
        2. Patcher applies surgical 'replace' operations
        3. Stores old values for rollback
        """
        shard_id = shard.id if hasattr(shard, 'id') else "unknown"
        content = shard.content if hasattr(shard, 'content') else shard
        content = copy.deepcopy(content)  # Don't modify original

        # Filter to structural issues only
        structural_issues = [
            i for i in issues
            if getattr(i, 'rule_id', '') in (
                'structure_integrity', 'structureintegrity',
                'id_preservation', 'idpreservation',
                'content_completeness', 'contentcompleteness'
            )
        ]

        if not structural_issues:
            return FixResult(
                shard_id=shard_id,
                fix_type=FixType.NONE,
                success=True,
                fixed_content=content,
                changes_made=["No structural issues to fix"]
            )

        # Get base structure for reference
        base_shard = context.get("base_shard")
        base_content = base_shard.content if base_shard and hasattr(base_shard, 'content') else None

        # Step 1: LLM identifies exact fields to fix
        field_fixes = await self.identify_fixes(content, structural_issues, base_content)

        if not field_fixes:
            return FixResult(
                shard_id=shard_id,
                fix_type=FixType.STRUCTURAL,
                success=True,
                fixed_content=content,
                changes_made=["LLM found no structural fixes needed"]
            )

        # Step 2: Convert to PatchOps
        patches = []
        for fix in field_fixes:
            if fix.fix_type != "structural":
                continue  # Skip non-structural fixes

            # Skip invalid paths (empty, root, or None)
            if not fix.path or fix.path == "/" or fix.path.strip() == "":
                logger.debug(f"Skipping invalid path for structural fix: '{fix.path}'")
                continue

            # Get current value for rollback
            try:
                old_value = self.patcher.get_value(content, fix.path)
            except KeyError:
                old_value = None

            patch = PatchOp(
                op="replace",
                path=fix.path,
                value=fix.new_value,
                old_value=old_value,
                reason=fix.reason,
            )
            patches.append(patch)

        if not patches:
            return FixResult(
                shard_id=shard_id,
                fix_type=FixType.STRUCTURAL,
                success=True,
                fixed_content=content,
                changes_made=["No valid patches generated"]
            )

        # Step 3: Apply patches via Patcher
        patch_result = self.patcher.apply_patches(content, patches)

        changes_made = [
            f"{p.path}: {p.reason}"
            for p in patch_result.applied_patches
        ]

        return FixResult(
            shard_id=shard_id,
            fix_type=FixType.STRUCTURAL,
            success=patch_result.success,
            fixed_content=patch_result.patched_data,
            patches_applied=patch_result.applied_patches,
            changes_made=changes_made,
            error=None if patch_result.success else f"Failed patches: {len(patch_result.failed_patches)}"
        )


# =============================================================================
# SEMANTIC FIXER (Hybrid: LLM identifies, Patcher applies)
# =============================================================================

class SemanticFixer:
    """
    Stage 4B-2: Semantic Fixer (Hybrid Approach)

    LLM IDENTIFIES exact fields with semantic issues.
    Patcher APPLIES surgical JSON Pointer patches.

    Fixes MEANING only:
    - Entity names (old company -> new company)
    - KPI terminology
    - Industry-specific terms

    Does NOT change:
    - Structure
    - Key names
    - Array lengths

    Uses RAG for industry-appropriate replacements.
    """

    def __init__(self):
        # Don't create shared LLM - get fresh instance for each call
        self.patcher = get_patcher()

    def _get_llm(self):
        """Get fresh LLM instance for parallel execution."""
        return _get_fixer_llm()

    @traceable(name="semantic_fixer_identify")
    async def identify_fixes(
        self,
        content: dict,
        issues: list,
        context: dict,
    ) -> tuple[list[FieldFix], dict]:
        """
        Use LLM to identify exact fields that need semantic fixes.
        Uses SPECIALIZED PROMPTS based on shard type for better fixes.

        Returns (list of FieldFix, entity_replacements dict)
        """
        from .fixer_prompts import get_prompt_for_shard, get_prompt_type_for_shard

        # Get context for fixes
        factsheet = context.get("global_factsheet", {})
        industry = context.get("industry", "unknown")
        poison_list = factsheet.get("poison_list", [])
        replacement_hints = factsheet.get("replacement_hints", {})
        shard_id = context.get("shard_id", "unknown")

        # Get company info
        company_info = factsheet.get("company", {})
        company_name = company_info.get("name", "Unknown Company")

        # Get KLOs for alignment prompts
        klos = factsheet.get("klos", [])
        if not klos:
            # Try to extract from learner objectives
            klos = factsheet.get("learning_objectives", [])

        # Get learner role info
        learner_role = factsheet.get("learner_role", {})
        role_name = learner_role.get("role", "Analyst")
        challenge = factsheet.get("context", {}).get("challenge", "")

        # Build issue list - handle both dict and object formats
        def get_issue_message(issue):
            if isinstance(issue, dict):
                return issue.get('message', issue.get('description', str(issue)))
            return getattr(issue, 'message', str(issue))

        issue_text = "\n".join([
            f"- {get_issue_message(i)}"
            for i in issues
        ])

        # Add alignment issues from context
        alignment_feedback = context.get("alignment_feedback", {})
        alignment_issues = alignment_feedback.get("critical_issues", [])
        if alignment_issues:
            issue_text += "\n" + "\n".join([f"- {issue}" for issue in alignment_issues[:5]])

        # ⭐ Get SPECIALIZED PROMPT based on shard type
        prompt_type = get_prompt_type_for_shard(shard_id)
        prompt_template = get_prompt_for_shard(shard_id)

        print(f"[SEMANTIC FIXER] Shard {shard_id} -> using {prompt_type} prompt")

        # Format the specialized prompt
        prompt = prompt_template.format(
            industry=industry,
            company_name=company_name,
            content=json.dumps(content, indent=2)[:6000],
            issues=issue_text,
            poison_list=json.dumps(poison_list) if poison_list else "[]",
            replacements=json.dumps(replacement_hints) if replacement_hints else "{}",
            klos=json.dumps(klos[:5]) if klos else "[]",
            learner_role=role_name,
            challenge=challenge,
            entity_map=json.dumps(replacement_hints) if replacement_hints else "{}",
        )

        try:
            parser = PydanticOutputParser(pydantic_object=SemanticFixResponse)

            chat_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a specialized content fixer.
Return fixes as JSON with 'fixes' array and 'replacements' dict.

Each fix needs:
- path: JSON Pointer to the SPECIFIC field (e.g., "/topicWizardData/lessonInformation/title" - NEVER use "/" or empty path)
- current_value: The current value at that path
- new_value: The fixed value
- reason: Why this fix is needed
- fix_type: "semantic"

IMPORTANT: The path MUST point to a specific field, not the root. Example paths:
- /topicWizardData/simulationName
- /topicWizardData/workplaceScenario/background/organizationName
- /topicWizardData/simulationFlow/0/children/0/data/email/body

{format_instructions}"""),
                ("human", "{input}"),
            ])

            chain = chat_prompt | self._get_llm() | parser

            # Use semaphore for controlled parallelism
            semaphore = _get_semaphore()
            async with semaphore:
                try:
                    result = await chain.ainvoke({
                        "input": prompt,
                        "format_instructions": parser.get_format_instructions(),
                    })

                    # Filter out invalid fixes (empty paths, root paths)
                    valid_fixes = [
                        fix for fix in result.fixes
                        if fix.path and fix.path != "/" and fix.path.strip() != ""
                    ]
                    invalid_count = len(result.fixes) - len(valid_fixes)

                    if invalid_count > 0:
                        logger.debug(f"[SEMANTIC FIXER] {shard_id}: Filtered out {invalid_count} invalid fixes (empty/root paths)")

                    print(f"[SEMANTIC FIXER] {shard_id}: Found {len(valid_fixes)} valid fixes" +
                          (f" (filtered {invalid_count} invalid)" if invalid_count > 0 else ""))
                    return valid_fixes, result.replacements
                except Exception as parse_error:
                    # LLM returned malformed JSON - skip this fixer
                    logger.warning(f"[SEMANTIC FIXER] {shard_id}: LLM returned invalid format, skipping: {str(parse_error)[:200]}")
                    return [], {}

        except Exception as e:
            logger.error(f"Failed to identify semantic fixes for {shard_id}: {e}")
            return [], {}

    @traceable(name="semantic_fixer_apply")
    async def fix(
        self,
        shard: Any,
        issues: list,
        context: dict,
    ) -> FixResult:
        """
        Fix semantic issues in a shard using hybrid approach.

        1. LLM identifies exact fields (JSON Pointer paths)
        2. Patcher applies surgical 'replace' operations
        3. Stores old values for rollback
        """
        shard_id = shard.id if hasattr(shard, 'id') else "unknown"
        content = shard.content if hasattr(shard, 'content') else shard
        content = copy.deepcopy(content)  # Don't modify original

        # Filter to semantic issues only
        # Handle both dict and object formats
        def get_issue_rule_id(issue):
            if isinstance(issue, dict):
                return issue.get('rule_id', '')
            return getattr(issue, 'rule_id', '')

        SEMANTIC_RULE_IDS = {
            'entity_removal', 'entityremoval',
            'domain_fidelity', 'domainfidelity',
            'tone', 'tonevalidator',
            # Batched check rule_ids
            'context_fidelity',
            'resource_self_contained',
            'data_consistency',
            'realism',
            'inference_integrity',
            # ContentCompleteness issues
            'contentcompleteness',
        }

        semantic_issues = [
            i for i in issues
            if get_issue_rule_id(i) in SEMANTIC_RULE_IDS or get_issue_rule_id(i) not in {
                'structure_integrity', 'structureintegrity',
                'id_preservation', 'idpreservation',
            }
        ]

        print(f"[SEMANTIC FIXER] {shard_id}: {len(issues)} total issues, {len(semantic_issues)} semantic issues")

        if not semantic_issues:
            return FixResult(
                shard_id=shard_id,
                fix_type=FixType.NONE,
                success=True,
                fixed_content=content,
                changes_made=["No semantic issues to fix"]
            )

        # Add shard_id to context for specialized prompt selection
        fix_context = {**context, "shard_id": shard_id}

        # Step 1: LLM identifies exact fields to fix (uses specialized prompt)
        field_fixes, replacements = await self.identify_fixes(content, semantic_issues, fix_context)

        if not field_fixes:
            return FixResult(
                shard_id=shard_id,
                fix_type=FixType.SEMANTIC,
                success=True,
                fixed_content=content,
                changes_made=["LLM found no semantic fixes needed"]
            )

        # Step 2: Convert to PatchOps
        patches = []
        for fix in field_fixes:
            if fix.fix_type != "semantic":
                continue  # Skip non-semantic fixes

            # Skip invalid paths (empty, root, or None)
            if not fix.path or fix.path == "/" or fix.path.strip() == "":
                logger.debug(f"Skipping invalid path for semantic fix: '{fix.path}'")
                continue

            # Get current value for rollback
            try:
                old_value = self.patcher.get_value(content, fix.path)
            except (KeyError, IndexError, TypeError):
                # Expected - LLM sometimes generates invalid paths, skip silently
                logger.debug(f"Path not found for semantic fix: {fix.path}")
                continue

            # Only create patch if value actually changed
            if old_value == fix.new_value:
                continue

            patch = PatchOp(
                op="replace",
                path=fix.path,
                value=fix.new_value,
                old_value=old_value,
                reason=fix.reason,
            )
            patches.append(patch)

        if not patches:
            return FixResult(
                shard_id=shard_id,
                fix_type=FixType.SEMANTIC,
                success=True,
                fixed_content=content,
                changes_made=["No valid patches generated"]
            )

        # Step 3: Apply patches via Patcher
        patch_result = self.patcher.apply_patches(content, patches)

        # Log actual results
        applied_count = len(patch_result.applied_patches)
        failed_count = len(patch_result.failed_patches)
        if applied_count > 0 or failed_count > 0:
            print(f"[SEMANTIC FIXER] {shard_id}: Applied {applied_count}/{len(patches)} patches" +
                  (f" ({failed_count} failed)" if failed_count > 0 else ""))

        changes_made = [
            f"{p.path}: {p.reason}"
            for p in patch_result.applied_patches
        ]

        # Add replacement summary
        if replacements:
            changes_made.append(f"Replacements: {replacements}")

        return FixResult(
            shard_id=shard_id,
            fix_type=FixType.SEMANTIC,
            success=patch_result.success,
            fixed_content=patch_result.patched_data,
            patches_applied=patch_result.applied_patches,
            changes_made=changes_made,
            error=None if patch_result.success else f"Failed patches: {len(patch_result.failed_patches)}"
        )


# =============================================================================
# SCOPED FIXER ORCHESTRATOR
# =============================================================================

@dataclass
class ShardFixStatus:
    """Track fix status for a shard."""
    shard_id: str
    fix_attempts: int = 0
    structural_fixed: bool = False
    semantic_fixed: bool = False
    flagged_for_human: bool = False
    fix_history: list[FixResult] = field(default_factory=list)
    all_patches: list[PatchOp] = field(default_factory=list)  # For full rollback


class ScopedFixer:
    """
    Stage 4B: Orchestrates scoped fixes across shards.

    Hybrid approach:
    - LLM identifies exact fields (JSON Pointer paths)
    - Patcher applies surgical 'replace' operations
    - Stores all patches for rollback capability

    Order:
    1. Structural fixes (then lock structure)
    2. Semantic fixes

    Max 3 attempts per shard, then flag for human.
    """

    MAX_ATTEMPTS = 3

    def __init__(self):
        self.structural_fixer = StructuralFixer()
        self.semantic_fixer = SemanticFixer()
        self.patcher = get_patcher()
        self.shard_status: dict[str, ShardFixStatus] = {}

    def get_shard_status(self, shard_id: str) -> ShardFixStatus:
        """Get or create fix status for a shard."""
        if shard_id not in self.shard_status:
            self.shard_status[shard_id] = ShardFixStatus(shard_id=shard_id)
        return self.shard_status[shard_id]

    async def fix_shard(
        self,
        shard: Any,
        validation_results: list,
        context: dict,
    ) -> FixResult:
        """
        Fix a single shard based on validation results.

        Uses hybrid approach:
        1. LLM identifies exact fields to fix
        2. Patcher applies surgical patches
        3. Stores patches for rollback
        """
        shard_id = shard.id if hasattr(shard, 'id') else "unknown"
        status = self.get_shard_status(shard_id)

        # Check if max attempts reached
        if status.fix_attempts >= self.MAX_ATTEMPTS:
            status.flagged_for_human = True
            return FixResult(
                shard_id=shard_id,
                fix_type=FixType.NONE,
                success=False,
                error=f"Max fix attempts ({self.MAX_ATTEMPTS}) reached. Flagged for human review."
            )

        # Collect all issues
        # Handle both dict and object formats
        all_issues = []
        for result in validation_results:
            if isinstance(result, dict):
                issues = result.get('issues', [])
            else:
                issues = getattr(result, 'issues', [])
            all_issues.extend(issues)

        print(f"[FIXER DEBUG] Collected {len(all_issues)} issues for shard {shard_id}")

        if not all_issues:
            return FixResult(
                shard_id=shard_id,
                fix_type=FixType.NONE,
                success=True,
                fixed_content=shard.content if hasattr(shard, 'content') else shard,
                changes_made=["No issues to fix"]
            )

        # Determine fix type needed
        # Handle both dict and object formats for issues
        def get_rule_id(issue):
            if isinstance(issue, dict):
                return issue.get('rule_id', '')
            return getattr(issue, 'rule_id', '')

        # Structural issues: structure, IDs, content completeness
        STRUCTURAL_RULES = {
            'structure_integrity', 'structureintegrity',
            'id_preservation', 'idpreservation',
            'content_completeness', 'contentcompleteness'
        }

        # Semantic issues: ALL other issues that need LLM-based fixes
        SEMANTIC_RULES = {
            'entity_removal', 'entityremoval',
            'domain_fidelity', 'domainfidelity',
            'tone', 'tonevalidator',
            # Add batched check rule_ids
            'context_fidelity',
            'resource_self_contained',
            'data_consistency',
            'realism',
            'inference_integrity',
        }

        rule_ids_found = set(get_rule_id(i) for i in all_issues)
        print(f"[FIXER DEBUG] Shard {shard_id}: rule_ids found = {rule_ids_found}")

        has_structural = bool(rule_ids_found & STRUCTURAL_RULES)
        # Semantic = any semantic rules OR any unknown rules (catch-all)
        has_semantic = bool(rule_ids_found & SEMANTIC_RULES) or bool(rule_ids_found - STRUCTURAL_RULES)

        print(f"[FIXER DEBUG] Shard {shard_id}: has_structural={has_structural}, has_semantic={has_semantic}")

        status.fix_attempts += 1
        final_result = None
        all_patches = []

        # Step 1: Structural fix (if needed and not already done)
        if has_structural and not status.structural_fixed:
            result = await self.structural_fixer.fix(shard, all_issues, context)
            status.fix_history.append(result)

            if result.success:
                status.structural_fixed = True
                all_patches.extend(result.patches_applied)

                # Update shard content for semantic fix
                if hasattr(shard, 'content') and result.fixed_content:
                    shard.content = result.fixed_content

                final_result = result

        # Step 2: Semantic fix (if needed)
        if has_semantic and not status.semantic_fixed:
            result = await self.semantic_fixer.fix(shard, all_issues, context)
            status.fix_history.append(result)

            if result.success:
                status.semantic_fixed = True
                all_patches.extend(result.patches_applied)
                final_result = result

        # Store all patches for potential rollback
        status.all_patches.extend(all_patches)

        # Return final result
        if final_result:
            # Combine all patches from this fix cycle
            final_result.patches_applied = all_patches
            return final_result
        else:
            return FixResult(
                shard_id=shard_id,
                fix_type=FixType.NONE,
                success=True,
                fixed_content=shard.content if hasattr(shard, 'content') else shard,
                changes_made=["No applicable fixes needed"]
            )

    async def fix_all(
        self,
        shards: list,
        validation_report: Any,
        context: dict,
    ) -> dict[str, FixResult]:
        """
        Fix all shards that have validation issues.

        Uses hybrid approach for each shard.
        """
        results = {}

        # Debug: Print what we received
        print(f"[FIXER DEBUG] Number of shards: {len(shards)}")
        print(f"[FIXER DEBUG] validation_report has shard_results: {hasattr(validation_report, 'shard_results')}")
        if hasattr(validation_report, 'shard_results'):
            print(f"[FIXER DEBUG] shard_results keys: {list(validation_report.shard_results.keys())}")

        # Helper to check issue counts
        def get_count(r, key):
            if isinstance(r, dict):
                return r.get(key, 0)
            return getattr(r, key, 0)

        # Prepare tasks for parallel execution
        async def fix_one_shard(shard):
            shard_id = shard.id if hasattr(shard, 'id') else "unknown"

            # Get validation results for this shard
            shard_results = []
            if hasattr(validation_report, 'shard_results'):
                shard_results = validation_report.shard_results.get(shard_id, [])

            has_issues = any(
                get_count(r, 'blocker_count') > 0 or get_count(r, 'warning_count') > 0
                for r in shard_results
            )

            if not has_issues:
                return shard_id, FixResult(
                    shard_id=shard_id,
                    fix_type=FixType.NONE,
                    success=True,
                    fixed_content=shard.content if hasattr(shard, 'content') else shard,
                    changes_made=["No issues to fix"]
                )

            print(f"[FIXER] Shard {shard_id} has issues - fixing in parallel")
            result = await self.fix_shard(shard, shard_results, context)
            return shard_id, result

        # Run ALL shards in parallel using create_task for TRUE parallelism
        print(f"[FIXER] Running {len(shards)} shards in PARALLEL (semaphore limit: {MAX_CONCURRENT_LLM_CALLS})")
        coroutines = [fix_one_shard(shard) for shard in shards]
        # Create actual Task objects to force concurrent scheduling
        tasks = [asyncio.create_task(coro) for coro in coroutines]
        task_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for result in task_results:
            if isinstance(result, Exception):
                logger.error(f"Fixer failed: {result}")
                continue
            shard_id, fix_result = result
            results[shard_id] = fix_result

        print(f"[FIXER] Completed {len(results)} shards in parallel")
        return results

    def rollback_shard(self, shard: Any, shard_id: str = None) -> Optional[dict]:
        """
        Rollback all fixes for a shard.

        Args:
            shard: Shard with current content
            shard_id: Shard ID (uses shard.id if not provided)

        Returns:
            Original content before fixes, or None if no patches to rollback
        """
        shard_id = shard_id or (shard.id if hasattr(shard, 'id') else "unknown")
        status = self.shard_status.get(shard_id)

        if not status or not status.all_patches:
            logger.warning(f"No patches to rollback for shard {shard_id}")
            return None

        content = shard.content if hasattr(shard, 'content') else shard
        rolled_back = self.patcher.rollback_patches(content, status.all_patches)

        logger.info(f"Rolled back {len(status.all_patches)} patches for shard {shard_id}")

        return rolled_back

    def get_flagged_shards(self) -> list[str]:
        """Get list of shard IDs flagged for human review."""
        return [
            status.shard_id
            for status in self.shard_status.values()
            if status.flagged_for_human
        ]

    def get_all_patches(self, shard_id: str = None) -> list[PatchOp]:
        """Get all patches applied to a shard or all shards."""
        if shard_id:
            status = self.shard_status.get(shard_id)
            return status.all_patches if status else []
        else:
            all_patches = []
            for status in self.shard_status.values():
                all_patches.extend(status.all_patches)
            return all_patches

    def reset_status(self, shard_id: str = None):
        """Reset fix status for a shard or all shards."""
        if shard_id:
            self.shard_status.pop(shard_id, None)
        else:
            self.shard_status.clear()


# =============================================================================
# BATCHED SEMANTIC FIXER
# =============================================================================

class BatchedSemanticFixer:
    """
    Applies fixes from BatchedShardChecker using JSON Patcher.

    Takes the list of fixes (with JSON Pointer paths) and applies them surgically.
    No additional LLM calls needed - fixes already identified by BatchedShardChecker.
    """

    def __init__(self):
        self.patcher = get_patcher()

    async def apply_fixes(
        self,
        shard: Any,
        fixes: list[dict],
        context: dict,
    ) -> FixResult:
        """
        Apply fixes to a shard using JSON Patcher.

        Args:
            shard: Shard to fix
            fixes: List of fix dicts with path, old_value, new_value, reason
            context: Pipeline context

        Returns:
            FixResult with patched content
        """
        shard_id = shard.id if hasattr(shard, 'id') else "unknown"
        content = shard.content if hasattr(shard, 'content') else shard
        content = copy.deepcopy(content)  # Don't modify original

        if not fixes:
            return FixResult(
                shard_id=shard_id,
                fix_type=FixType.NONE,
                success=True,
                fixed_content=content,
                changes_made=["No fixes to apply"]
            )

        # Build set of paths already fixed by alignment fixer - don't overwrite
        alignment_fixes = context.get("alignment_fix_results", [])
        protected_paths = set()
        for fix_result in alignment_fixes:
            if isinstance(fix_result, dict):
                for change in fix_result.get("changes", []):
                    if isinstance(change, str):
                        # Extract path from change description like "Fixed at path/to/field"
                        if "at " in change:
                            protected_paths.add(change.split("at ")[-1].strip())
                        elif ":" in change:
                            protected_paths.add(change.split(":")[0].strip())
                # Also check for direct path field
                if fix_result.get("path"):
                    protected_paths.add(fix_result["path"])

        if protected_paths:
            logger.info(f"[SEMANTIC FIXER] Protecting {len(protected_paths)} paths from alignment fixer")

        # Convert fixes to PatchOps
        patches = []
        for fix in fixes:
            try:
                path = fix.get("path", "")

                # Skip paths protected by alignment fixer
                if any(protected_path in path for protected_path in protected_paths):
                    logger.debug(f"Skipping {path} - protected by alignment fixer")
                    continue

                # Skip invalid paths (empty, root, or None)
                if not path or path == "/" or path.strip() == "":
                    logger.debug(f"Skipping invalid path for batched fix: '{path}'")
                    continue

                # Validate path exists
                if not self.patcher.path_exists(content, path):
                    # Expected - LLM sometimes generates invalid paths
                    logger.debug(f"Path not found for fix: {path}")
                    continue

                patch = PatchOp(
                    op="replace",
                    path=path,
                    value=fix["new_value"],
                    old_value=fix.get("old_value"),
                    reason=fix.get("reason", "Batched semantic fix"),
                )
                patches.append(patch)
            except Exception as e:
                logger.warning(f"Failed to create patch for {fix.get('path')}: {e}")

        if not patches:
            return FixResult(
                shard_id=shard_id,
                fix_type=FixType.SEMANTIC,
                success=True,
                fixed_content=content,
                changes_made=["No valid patches generated"]
            )

        # Apply patches
        patch_result = self.patcher.apply_patches(content, patches, validate_first=True, stop_on_error=False)

        changes_made = [
            f"{p.path}: {p.reason}"
            for p in patch_result.applied_patches
        ]

        return FixResult(
            shard_id=shard_id,
            fix_type=FixType.SEMANTIC,
            success=len(patch_result.failed_patches) == 0,
            fixed_content=patch_result.patched_data,
            patches_applied=patch_result.applied_patches,
            changes_made=changes_made,
            error=None if not patch_result.failed_patches else f"Failed: {len(patch_result.failed_patches)} patches"
        )

    async def apply_all_fixes(
        self,
        shards: list,
        all_fixes: dict[str, list[dict]],
        context: dict,
    ) -> dict[str, FixResult]:
        """
        Apply fixes to all shards in parallel.

        Args:
            shards: List of Shard objects
            all_fixes: Dict of shard_id -> list of fixes
            context: Pipeline context

        Returns:
            Dict of shard_id -> FixResult
        """
        results = {}

        async def fix_one(shard):
            shard_id = shard.id if hasattr(shard, 'id') else "unknown"
            fixes = all_fixes.get(shard_id, [])
            return shard_id, await self.apply_fixes(shard, fixes, context)

        # Create actual Task objects for TRUE parallelism
        coroutines = [fix_one(shard) for shard in shards]
        tasks = [asyncio.create_task(coro) for coro in coroutines]
        fix_results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in fix_results:
            if isinstance(result, Exception):
                logger.error(f"Fix failed: {result}")
                continue
            shard_id, fix_result = result
            results[shard_id] = fix_result

            # Update shard content if fix succeeded
            if fix_result.success and fix_result.fixed_content:
                for shard in shards:
                    if (shard.id if hasattr(shard, 'id') else "unknown") == shard_id:
                        if hasattr(shard, 'content'):
                            shard.content = fix_result.fixed_content
                        break

        return results


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def fix_shard(
    shard: Any,
    validation_results: list,
    context: dict,
) -> FixResult:
    """Fix a single shard. See ScopedFixer.fix_shard."""
    fixer = ScopedFixer()
    return await fixer.fix_shard(shard, validation_results, context)


async def fix_all_shards(
    shards: list,
    validation_report: Any,
    context: dict,
) -> dict[str, FixResult]:
    """Fix all shards. See ScopedFixer.fix_all."""
    fixer = ScopedFixer()
    return await fixer.fix_all(shards, validation_report, context)


async def apply_batched_fixes(
    shards: list,
    all_fixes: dict[str, list[dict]],
    context: dict,
) -> dict[str, FixResult]:
    """
    Apply batched fixes from BatchedShardChecker.

    This is the preferred method for applying fixes after batched validation.

    Args:
        shards: List of Shard objects
        all_fixes: Dict of shard_id -> list of fixes (from validate_shards)
        context: Pipeline context

    Returns:
        Dict of shard_id -> FixResult
    """
    fixer = BatchedSemanticFixer()
    return await fixer.apply_all_fixes(shards, all_fixes, context)


# =============================================================================
# KLO ALIGNMENT FIXER (Post-Adaptation)
# =============================================================================

class KLOAlignmentFixResponse(BaseModel):
    """LLM response for KLO alignment fixes."""
    fixed_questions: list[dict] = Field(description="List of fixed questions with proper KLO mapping")
    fixed_activities: list[dict] = Field(description="List of fixed activities with proper KLO mapping")
    klo_mapping: dict = Field(description="Mapping of each KLO to its assessing questions/activities")
    summary: str = Field(description="Summary of alignment fixes made")


class KLOAlignmentFixer:
    """
    Post-adaptation fixer that ensures questions/activities properly assess KLOs.

    This runs AFTER adaptation but BEFORE alignment checking.
    It extracts the adapted KLOs and rewrites questions to ensure proper mapping.
    """

    def __init__(self):
        # Don't create shared LLM - get fresh instance for each call
        pass

    def _get_llm(self):
        """Get fresh LLM instance for parallel execution."""
        return _get_fixer_llm()

    def _extract_klos(self, topic_data: dict) -> list[dict]:
        """Extract KLOs from adapted JSON."""
        klos = []
        for criterion in topic_data.get("assessmentCriterion", []):
            klo_text = criterion.get("keyLearningOutcome", "")
            criteria_list = [c.get("criteria", "") for c in criterion.get("criterion", []) if c.get("criteria")]
            if klo_text:
                klos.append({
                    "id": criterion.get("id", ""),
                    "klo": klo_text,
                    "criteria": criteria_list
                })
        return klos

    def _extract_questions(self, topic_data: dict) -> list[dict]:
        """Extract all questions from simulation flow."""
        questions = []

        # From simulationFlow stages
        for stage in topic_data.get("simulationFlow", []):
            stage_data = stage.get("data", {})

            # Direct questions
            if "questions" in stage_data:
                for q in stage_data["questions"]:
                    questions.append({
                        "location": f"simulationFlow/{stage.get('name', 'unknown')}/questions",
                        "question": q
                    })

            # Submission questions
            if "submissionQuestions" in stage_data:
                for q in stage_data["submissionQuestions"]:
                    questions.append({
                        "location": f"simulationFlow/{stage.get('name', 'unknown')}/submissionQuestions",
                        "question": q
                    })

            # Activity data questions
            activity_data = stage_data.get("activityData", {})
            if isinstance(activity_data, dict) and "questions" in activity_data:
                for q in activity_data["questions"]:
                    questions.append({
                        "location": f"simulationFlow/{stage.get('name', 'unknown')}/activityData/questions",
                        "question": q
                    })

            # Review rubric questions
            review = stage_data.get("review", {})
            if isinstance(review, dict):
                for rubric in review.get("rubric", []):
                    if isinstance(rubric, dict):
                        if rubric.get("question"):
                            questions.append({
                                "location": f"simulationFlow/{stage.get('name', 'unknown')}/review/rubric",
                                "question": {"question": rubric["question"]}
                            })

            # Children
            for child in stage.get("children", []):
                child_data = child.get("data", {})
                if "questions" in child_data:
                    for q in child_data["questions"]:
                        questions.append({
                            "location": f"simulationFlow/{stage.get('name', 'unknown')}/children/{child.get('name', 'unknown')}/questions",
                            "question": q
                        })

        # Top-level submission questions
        for q in topic_data.get("submissionQuestions", []):
            questions.append({"location": "submissionQuestions", "question": q})
        for q in topic_data.get("selectedSubmissionQuestions", []):
            questions.append({"location": "selectedSubmissionQuestions", "question": q})

        return questions

    def _extract_activities(self, topic_data: dict) -> list[dict]:
        """Extract all activities from simulation flow."""
        activities = []

        for stage in topic_data.get("simulationFlow", []):
            stage_data = stage.get("data", {})

            # Stage itself as activity
            if stage_data.get("name") or stage_data.get("description"):
                activities.append({
                    "location": f"simulationFlow/{stage.get('name', 'unknown')}",
                    "name": stage_data.get("name", stage.get("name", "")),
                    "description": stage_data.get("description", ""),
                    "type": stage.get("type", "")
                })

            # Children activities
            for child in stage.get("children", []):
                child_data = child.get("data", {})
                if child_data.get("name") or child_data.get("description"):
                    activities.append({
                        "location": f"simulationFlow/{stage.get('name', 'unknown')}/children/{child.get('name', 'unknown')}",
                        "name": child_data.get("name", child.get("name", "")),
                        "description": child_data.get("description", ""),
                        "type": child.get("type", "")
                    })

        return activities

    async def _check_and_fix_klo(
        self,
        klo: dict,
        questions: list,
        company_name: str,
        industry: str,
    ) -> dict:
        """
        Check alignment for a single KLO and generate fix if needed.
        Runs in parallel with other KLO checks.

        Returns:
            {
                "klo_id": str,
                "is_aligned": bool,
                "matched_question_id": str or None,
                "fix": None or {
                    "question_id": str,
                    "original_text": str,
                    "new_text": str
                }
            }
        """
        klo_id = klo.get("id", "unknown")
        klo_text = klo.get("klo", "")

        # Format questions for prompt (only id and text)
        # Questions come wrapped: {"location": ..., "question": {...actual question...}}
        questions_simplified = []
        for i, q in enumerate(questions[:15]):
            # Unwrap if wrapped
            actual_q = q.get("question", q) if isinstance(q, dict) else q
            if isinstance(actual_q, dict):
                q_id = actual_q.get("id", f"q_{i}")
                q_text = actual_q.get("question", actual_q.get("text", ""))
                if q_text:
                    questions_simplified.append({"id": q_id, "question": q_text})

        prompt = f"""You are a KLO Alignment Expert for business simulations.

## STEP 1: ANALYZE THIS KLO
KLO: "{klo_text}"

Extract the KEY TERMS that MUST appear in an aligned question:
- What is the ACTION VERB? (e.g., analyze, develop, create, evaluate, identify)
- What is the MAIN CONCEPT? (e.g., market strategy, customer segments, pricing model)
- What is the CONTEXT/METHOD? (e.g., using data, for target market, with criteria)

## STEP 2: CHECK THESE QUESTIONS
{json.dumps(questions_simplified, indent=2)}

A question is ALIGNED if it contains the KEY TERMS from the KLO above.
- ALIGNED: Uses the SAME action verb + SAME main concept + SAME context
- NOT ALIGNED: Uses synonyms, is vague, or misses key terms

## STEP 3: IF NOT ALIGNED, REWRITE
Pick the closest question and rewrite it to include:
1. The EXACT key terms from THIS KLO (not generic terms)
2. Reference to {company_name} ({industry})
3. A measurable outcome

## CONTEXT
- Company: {company_name}
- Industry: {industry}
- This is an undergraduate business simulation

## OUTPUT (JSON only, no other text):
{{
  "key_terms_from_klo": ["list", "of", "key", "terms", "extracted"],
  "is_aligned": true/false,
  "matched_question_id": "id" or null,
  "fix": null or {{
    "question_id": "id of question to rewrite",
    "original_text": "the original question",
    "new_text": "rewritten question using the key_terms_from_klo"
  }}
}}"""

        try:
            # Use semaphore for controlled parallelism
            content = await _invoke_with_semaphore(self._get_llm(), prompt)

            # Extract JSON by finding balanced braces (handles nested objects)
            def extract_json_object(text):
                start = text.find('{')
                if start == -1:
                    return None
                depth = 0
                for i, char in enumerate(text[start:], start):
                    if char == '{':
                        depth += 1
                    elif char == '}':
                        depth -= 1
                        if depth == 0:
                            return text[start:i+1]
                return None

            json_str = extract_json_object(content)

            if json_str:
                try:
                    parsed = json.loads(json_str)
                    parsed["klo_id"] = klo_id
                    return parsed
                except json.JSONDecodeError:
                    # Try to repair common JSON issues
                    import re
                    # Remove trailing commas before }
                    json_str = re.sub(r',\s*}', '}', json_str)
                    json_str = re.sub(r',\s*]', ']', json_str)
                    parsed = json.loads(json_str)
                    parsed["klo_id"] = klo_id
                    return parsed

            logger.warning(f"Could not find JSON for KLO {klo_id}")
            return {"klo_id": klo_id, "is_aligned": True, "matched_question_id": None, "fix": None}

        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error for KLO {klo_id}: {e}")
            return {"klo_id": klo_id, "is_aligned": True, "matched_question_id": None, "fix": None}
        except Exception as e:
            logger.warning(f"KLO check failed for {klo_id}: {e}")
            return {"klo_id": klo_id, "is_aligned": True, "matched_question_id": None, "fix": None}

    @traceable(name="klo_alignment_fixer")
    async def fix(self, adapted_json: dict, context: dict) -> dict:
        """
        Fix KLO alignment in adapted JSON using BATCH + PARALLEL approach.

        Each KLO is checked separately in parallel for speed.
        If a KLO is misaligned, the question is rewritten to match KLO terminology.

        Args:
            adapted_json: The adapted JSON after adaptation stage
            context: Pipeline context with global_factsheet

        Returns:
            Fixed JSON with proper KLO-to-question mapping
        """
        topic_data = adapted_json.get("topicWizardData", {})

        # Extract components
        klos = self._extract_klos(topic_data)
        questions = self._extract_questions(topic_data)

        if not klos:
            logger.warning("No KLOs found in adapted JSON, skipping KLO alignment fix")
            return adapted_json

        if not questions:
            logger.warning("No questions found in adapted JSON, skipping KLO alignment fix")
            return adapted_json

        logger.info(f"KLO Alignment Fixer (PARALLEL): {len(klos)} KLOs, {len(questions)} questions")

        # Get context info
        factsheet = context.get("global_factsheet", {})
        company_name = factsheet.get("company", {}).get("name", "the company")
        industry = factsheet.get("company", {}).get("industry", "business")

        # Process each KLO in PARALLEL using create_task for TRUE parallelism
        logger.info(f"Processing {len(klos)} KLOs in PARALLEL (semaphore limit: {MAX_CONCURRENT_LLM_CALLS})")
        coroutines = [
            self._check_and_fix_klo(klo, questions, company_name, industry)
            for klo in klos
        ]
        # Create actual Task objects to force concurrent scheduling
        tasks = [asyncio.create_task(coro) for coro in coroutines]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect fixes
        fixes = []
        aligned_count = 0
        misaligned_count = 0

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"KLO {i} check failed: {result}")
                continue

            klo_id = result.get("klo_id", f"klo_{i}")

            if result.get("is_aligned", True):
                aligned_count += 1
                logger.debug(f"KLO {klo_id}: ALIGNED (matched: {result.get('matched_question_id')})")
            else:
                misaligned_count += 1
                if result.get("fix"):
                    fixes.append(result["fix"])
                    logger.info(f"KLO {klo_id}: MISALIGNED - will rewrite question {result['fix'].get('question_id')}")

        logger.info(f"KLO Alignment: {aligned_count} aligned, {misaligned_count} misaligned, {len(fixes)} fixes to apply")

        # Apply fixes to JSON
        if fixes:
            fixed_json = self._apply_question_fixes(adapted_json, fixes)
            logger.info(f"Applied {len(fixes)} question fixes")
            return fixed_json

        return adapted_json

    def _apply_question_fixes(self, adapted_json: dict, fixes: list) -> dict:
        """
        Apply question rewrites to the JSON.

        Args:
            adapted_json: The adapted JSON
            fixes: List of fixes, each with {question_id, original_text, new_text}

        Returns:
            Fixed JSON with rewritten questions
        """
        fixed_json = copy.deepcopy(adapted_json)
        topic_data = fixed_json.get("topicWizardData", {})

        for fix in fixes:
            question_id = fix.get("question_id")
            new_text = fix.get("new_text")

            if not question_id or not new_text:
                logger.warning(f"Invalid fix: {fix}")
                continue

            found = False

            # 1. Check submissionQuestions
            for q in topic_data.get("submissionQuestions", []):
                if q.get("id") == question_id:
                    old_text = q.get("question", "")[:50]
                    q["question"] = new_text
                    logger.info(f"Fixed submissionQuestion {question_id}: '{old_text}...' -> '{new_text[:50]}...'")
                    found = True
                    break

            # 2. Check simulationFlow stages
            if not found:
                for stage in topic_data.get("simulationFlow", []):
                    stage_data = stage.get("data", {})

                    # Check questions in stage data
                    for q in stage_data.get("questions", []):
                        if q.get("id") == question_id:
                            old_text = q.get("question", q.get("text", ""))[:50]
                            if "question" in q:
                                q["question"] = new_text
                            else:
                                q["text"] = new_text
                            logger.info(f"Fixed simulationFlow question {question_id}: '{old_text}...' -> '{new_text[:50]}...'")
                            found = True
                            break

                    # Check reflectionQuestions
                    for q in stage_data.get("reflectionQuestions", []):
                        if q.get("id") == question_id:
                            old_text = q.get("question", "")[:50]
                            q["question"] = new_text
                            logger.info(f"Fixed reflectionQuestion {question_id}: '{old_text}...' -> '{new_text[:50]}...'")
                            found = True
                            break

                    if found:
                        break

            if not found:
                # Expected - LLM sometimes generates invalid question IDs
                logger.debug(f"Could not find question {question_id} to fix")

        fixed_json["topicWizardData"] = topic_data
        return fixed_json


async def fix_klo_alignment(adapted_json: dict, context: dict) -> dict:
    """
    Convenience function to fix KLO alignment.

    Call this AFTER adaptation but BEFORE alignment checking.

    Args:
        adapted_json: The adapted JSON from adaptation stage
        context: Pipeline context with global_factsheet

    Returns:
        Fixed JSON with proper KLO-to-question mapping
    """
    fixer = KLOAlignmentFixer()
    return await fixer.fix(adapted_json, context)
