"""
Leaf indexer for JSON Contextualizer Agent.

This module provides functionality to enumerate all leaf paths
in a JSON document using JSON Pointer syntax.
"""

from typing import Any, List, Tuple
import logging

logger = logging.getLogger(__name__)


def index_leaves(json_obj: Any, base_path: str = "") -> List[Tuple[str, Any]]:
    """
    Recursively index all leaf nodes in a JSON object.

    Args:
        json_obj: The JSON object to index
        base_path: The base JSON Pointer path (used internally for recursion)

    Returns:
        List of tuples containing (json_pointer_path, current_value)
    """
    leaves = []

    if isinstance(json_obj, dict):
        for key, value in json_obj.items():
            # Escape special characters in the key (RFC 6901)
            escaped_key = key.replace("~", "~0").replace("/", "~1")
            current_path = f"{base_path}/{escaped_key}" if base_path else f"/{escaped_key}"

            if _is_leaf(value):
                leaves.append((current_path, value))
            else:
                leaves.extend(index_leaves(value, current_path))

    elif isinstance(json_obj, list):
        for i, value in enumerate(json_obj):
            current_path = f"{base_path}/{i}" if base_path else f"/{i}"

            if _is_leaf(value):
                leaves.append((current_path, value))
            else:
                leaves.extend(index_leaves(value, current_path))

    return leaves


def _is_leaf(value: Any) -> bool:
    """
    Check if a value is a leaf node (primitive type).

    Args:
        value: The value to check

    Returns:
        True if the value is a leaf (primitive), False otherwise
    """
    return isinstance(value, (str, int, float, bool, type(None)))


def is_string_leaf(value: Any) -> bool:
    """
    Check if a value is a string leaf (candidate for editing).

    Args:
        value: The value to check

    Returns:
        True if the value is a string, False otherwise
    """
    return isinstance(value, str)


def filter_string_leaves(leaves: List[Tuple[str, Any]]) -> List[Tuple[str, str]]:
    """
    Filter leaves to only include string values.

    Args:
        leaves: List of (path, value) tuples

    Returns:
        List of (path, string_value) tuples for string leaves only
    """
    return [(path, value) for path, value in leaves if is_string_leaf(value)]


def filter_modifiable_leaves(
    leaves: List[Tuple[str, Any]],
    locked_patterns: List[str] = None
) -> List[Tuple[str, str]]:
    """
    Filter leaves to only include modifiable string values.

    Excludes:
    - Non-string values
    - ID fields
    - Locked paths
    - Empty strings

    Args:
        leaves: List of (path, value) tuples
        locked_patterns: List of path patterns to exclude

    Returns:
        List of (path, string_value) tuples for modifiable leaves
    """
    import re

    locked_patterns = locked_patterns or []

    # Default locked patterns
    default_locked = [
        r".*/id$",           # ID fields
        r".*/uuid$",         # UUID fields
        r".*/guid$",         # GUID fields
        r".*/_id$",          # MongoDB-style IDs
        r".*/workspace$",    # Workspace field
        r".*/builderType$",  # Builder type
        r".*/duration$",     # Duration field
    ]

    all_locked = default_locked + locked_patterns

    modifiable = []
    for path, value in leaves:
        # Skip non-strings
        if not isinstance(value, str):
            continue

        # Skip empty strings
        if not value.strip():
            continue

        # Skip locked patterns
        is_locked = False
        for pattern in all_locked:
            if re.match(pattern, path):
                is_locked = True
                break

        if not is_locked:
            modifiable.append((path, value))

    logger.info(f"Found {len(modifiable)} modifiable leaves out of {len(leaves)} total")
    return modifiable


def find_leaves_containing(
    leaves: List[Tuple[str, Any]],
    search_terms: List[str],
    case_insensitive: bool = True
) -> List[Tuple[str, str, str]]:
    """
    Find leaves containing specific terms.

    Args:
        leaves: List of (path, value) tuples
        search_terms: List of terms to search for
        case_insensitive: Whether to ignore case

    Returns:
        List of (path, value, matched_term) tuples
    """
    results = []

    for path, value in leaves:
        if not isinstance(value, str):
            continue

        check_value = value.lower() if case_insensitive else value

        for term in search_terms:
            check_term = term.lower() if case_insensitive else term
            if check_term in check_value:
                results.append((path, value, term))
                break  # Only match once per leaf

    return results


def get_leaf_stats(leaves: List[Tuple[str, Any]]) -> dict:
    """
    Get statistics about indexed leaves.

    Args:
        leaves: List of (path, value) tuples

    Returns:
        Statistics dictionary
    """
    stats = {
        "total": len(leaves),
        "strings": 0,
        "numbers": 0,
        "booleans": 0,
        "nulls": 0,
        "empty_strings": 0,
        "avg_string_length": 0,
    }

    string_lengths = []

    for path, value in leaves:
        if value is None:
            stats["nulls"] += 1
        elif isinstance(value, bool):
            stats["booleans"] += 1
        elif isinstance(value, (int, float)):
            stats["numbers"] += 1
        elif isinstance(value, str):
            stats["strings"] += 1
            if not value.strip():
                stats["empty_strings"] += 1
            string_lengths.append(len(value))

    if string_lengths:
        stats["avg_string_length"] = sum(string_lengths) / len(string_lengths)

    return stats
