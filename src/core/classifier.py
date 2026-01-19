"""
Leaf classifier for JSON Contextualizer Agent.

This module classifies leaves into processing strategies:
- SKIP: Don't touch (IDs, numbers, locked fields)
- REPLACE: Simple find/replace using factsheet mappings
- REWRITE: Needs LLM to rewrite content
"""

from typing import List, Tuple, Dict, Any
from enum import Enum
from dataclasses import dataclass
import logging
import re

logger = logging.getLogger(__name__)


class LeafStrategy(Enum):
    """Processing strategy for a leaf."""
    SKIP = "skip"           # Don't modify
    REPLACE = "replace"     # Simple find/replace
    REWRITE = "rewrite"     # LLM rewrite needed


@dataclass
class ClassifiedLeaf:
    """A leaf with its classification."""
    path: str
    value: Any
    strategy: LeafStrategy
    reason: str
    replacement: str = None  # For REPLACE strategy, the new value


class LeafClassifier:
    """
    Classify leaves into processing strategies.

    Uses factsheet for entity mappings and poison terms.
    """

    def __init__(self, factsheet: Dict[str, Any]):
        """
        Args:
            factsheet: Global factsheet with entity mappings
        """
        self.factsheet = factsheet

        # Extract mappings from factsheet
        self.entity_map = self._build_entity_map()
        self.poison_terms = self._get_poison_terms()

        # Patterns for locked fields
        # ONLY lock truly immutable fields (IDs, system fields)
        # DO NOT lock content fields that need transformation!
        self.locked_patterns = [
            r".*/id$",
            r".*/uuid$",
            r".*/guid$",
            r".*/_id$",
            r".*/workspace$",
            r".*/builderType$",
            r".*/duration$",
            r".*/level$",
            # REMOVED: r".*/scenarioOptions/.*" - these SHOULD be transformed!
        ]

    def _build_entity_map(self) -> Dict[str, str]:
        """Build entity replacement map from factsheet."""
        entity_map = {}

        # Company name mapping
        company = self.factsheet.get("company", {})
        if isinstance(company, dict):
            old_name = company.get("old_name") or company.get("source_name")
            new_name = company.get("name") or company.get("new_name")
            if old_name and new_name:
                entity_map[old_name] = new_name

        # Manager name mapping
        manager = self.factsheet.get("manager", {})
        if isinstance(manager, dict):
            old_name = manager.get("old_name") or manager.get("source_name")
            new_name = manager.get("name") or manager.get("new_name")
            if old_name and new_name:
                entity_map[old_name] = new_name

        # Additional entity mappings
        mappings = self.factsheet.get("entity_mappings", {})
        if isinstance(mappings, dict):
            entity_map.update(mappings)

        # Replacements mapping
        replacements = self.factsheet.get("replacements", {})
        if isinstance(replacements, dict):
            entity_map.update(replacements)

        logger.info(f"Built entity map with {len(entity_map)} mappings")
        return entity_map

    def _get_poison_terms(self) -> List[str]:
        """Get poison terms from factsheet."""
        poison = self.factsheet.get("poison_list", [])
        if not isinstance(poison, list):
            poison = []

        # Also add old entity names as poison terms
        company = self.factsheet.get("company", {})
        if isinstance(company, dict):
            old_name = company.get("old_name") or company.get("source_name")
            if old_name and old_name not in poison:
                poison.append(old_name)

        logger.info(f"Found {len(poison)} poison terms")
        return poison

    def classify(self, leaves: List[Tuple[str, Any]]) -> List[ClassifiedLeaf]:
        """
        Classify all leaves into processing strategies.

        Args:
            leaves: List of (path, value) tuples

        Returns:
            List of ClassifiedLeaf objects
        """
        classified = []

        for path, value in leaves:
            clf = self._classify_single(path, value)
            classified.append(clf)

        # Log statistics
        stats = self._get_stats(classified)
        logger.info(f"Classification: {stats}")

        return classified

    def _classify_single(self, path: str, value: Any) -> ClassifiedLeaf:
        """Classify a single leaf."""

        # 1. Non-string values -> SKIP
        if not isinstance(value, str):
            return ClassifiedLeaf(
                path=path,
                value=value,
                strategy=LeafStrategy.SKIP,
                reason="non-string value"
            )

        # 2. Empty strings -> SKIP
        if not value.strip():
            return ClassifiedLeaf(
                path=path,
                value=value,
                strategy=LeafStrategy.SKIP,
                reason="empty string"
            )

        # 3. Locked fields -> SKIP
        for pattern in self.locked_patterns:
            if re.match(pattern, path):
                return ClassifiedLeaf(
                    path=path,
                    value=value,
                    strategy=LeafStrategy.SKIP,
                    reason=f"locked field: {pattern}"
                )

        # 4. Check for entity replacements -> REPLACE
        replacement, matched_term = self._find_replacement(value)
        if replacement:
            return ClassifiedLeaf(
                path=path,
                value=value,
                strategy=LeafStrategy.REPLACE,
                reason=f"contains entity: {matched_term}",
                replacement=replacement
            )

        # 5. Check for poison terms (needs rewrite even if no direct mapping)
        for poison in self.poison_terms:
            if poison.lower() in value.lower():
                return ClassifiedLeaf(
                    path=path,
                    value=value,
                    strategy=LeafStrategy.REWRITE,
                    reason=f"contains poison term: {poison}"
                )

        # 6. Long text content -> REWRITE (likely needs context adaptation)
        if len(value) > 100:
            return ClassifiedLeaf(
                path=path,
                value=value,
                strategy=LeafStrategy.REWRITE,
                reason="long text content needs context adaptation"
            )

        # 7. Short text without entities -> SKIP (labels, titles, etc.)
        return ClassifiedLeaf(
            path=path,
            value=value,
            strategy=LeafStrategy.SKIP,
            reason="short text without entities"
        )

    def _find_replacement(self, value: str) -> Tuple[str, str]:
        """
        Find if value contains any entity that needs replacement.

        Returns:
            (new_value, matched_term) or (None, None) if no match
        """
        new_value = value

        for old_term, new_term in self.entity_map.items():
            # Case-insensitive search
            pattern = re.compile(re.escape(old_term), re.IGNORECASE)
            if pattern.search(value):
                new_value = pattern.sub(new_term, new_value)
                return (new_value, old_term)

        return (None, None)

    def _get_stats(self, classified: List[ClassifiedLeaf]) -> Dict[str, int]:
        """Get classification statistics."""
        stats = {
            "total": len(classified),
            "skip": 0,
            "replace": 0,
            "rewrite": 0,
        }

        for clf in classified:
            stats[clf.strategy.value] += 1

        return stats


def classify_leaves(
    leaves: List[Tuple[str, Any]],
    factsheet: Dict[str, Any]
) -> List[ClassifiedLeaf]:
    """
    Classify leaves using factsheet.

    Args:
        leaves: List of (path, value) tuples
        factsheet: Global factsheet with entity mappings

    Returns:
        List of ClassifiedLeaf objects
    """
    classifier = LeafClassifier(factsheet)
    return classifier.classify(leaves)


def get_leaves_by_strategy(
    classified: List[ClassifiedLeaf],
    strategy: LeafStrategy
) -> List[ClassifiedLeaf]:
    """Filter classified leaves by strategy."""
    return [clf for clf in classified if clf.strategy == strategy]


def apply_replacements(
    classified: List[ClassifiedLeaf]
) -> List[Tuple[str, str, str]]:
    """
    Get replacement operations for REPLACE strategy leaves.

    Returns:
        List of (path, old_value, new_value) tuples
    """
    replacements = []

    for clf in classified:
        if clf.strategy == LeafStrategy.REPLACE and clf.replacement:
            replacements.append((clf.path, clf.value, clf.replacement))

    return replacements
