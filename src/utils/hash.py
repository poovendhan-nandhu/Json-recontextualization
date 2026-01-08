"""Hashing utilities for shard change detection."""
import hashlib
import json
from typing import Any


def compute_hash(content: Any) -> str:
    """
    Compute SHA-256 hash of content.

    Args:
        content: Any JSON-serializable content

    Returns:
        Hex string of SHA-256 hash
    """
    # Serialize with sorted keys for consistent hashing
    json_str = json.dumps(content, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(json_str.encode('utf-8')).hexdigest()


def compute_hash_for_paths(data: dict, paths: list[str]) -> str:
    """
    Compute hash for specific paths in a JSON structure.

    Args:
        data: Full JSON data
        paths: List of dot-notation paths to include

    Returns:
        Combined hash of all path contents
    """
    contents = []
    for path in sorted(paths):  # Sort for consistency
        value = get_nested_value(data, path)
        if value is not None:
            contents.append({path: value})

    return compute_hash(contents)


def get_nested_value(data: dict, path: str) -> Any:
    """
    Get a nested value from a dict using dot notation.

    Supports:
    - Simple paths: "topicWizardData.id"
    - Array wildcards: "topicWizardData.simulationFlow[*].id"
    - Specific indices: "topicWizardData.simulationFlow[0].name"

    Args:
        data: The dictionary to traverse
        path: Dot-notation path (e.g., "topicWizardData.lessonInformation")

    Returns:
        The value at that path, or None if not found
    """
    if not path:
        return data

    parts = parse_path(path)
    current = data

    for part in parts:
        if current is None:
            return None

        if isinstance(part, int):
            # Array index
            if isinstance(current, list) and 0 <= part < len(current):
                current = current[part]
            else:
                return None
        elif part == "*":
            # Wildcard - return list of all items
            if isinstance(current, list):
                return current
            return None
        else:
            # Dict key
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None

    return current


def set_nested_value(data: dict, path: str, value: Any) -> dict:
    """
    Set a nested value in a dict using dot notation.

    Args:
        data: The dictionary to modify (will be copied)
        path: Dot-notation path
        value: Value to set

    Returns:
        Modified copy of the dictionary
    """
    import copy
    result = copy.deepcopy(data)

    parts = parse_path(path)
    current = result

    for i, part in enumerate(parts[:-1]):
        if isinstance(part, int):
            if isinstance(current, list) and 0 <= part < len(current):
                current = current[part]
            else:
                return result  # Can't set, path doesn't exist
        else:
            # Must be a dict to access with string key
            if not isinstance(current, dict):
                return result  # Can't set, structure mismatch
            if part not in current:
                # Create missing intermediate dicts
                next_part = parts[i + 1]
                current[part] = [] if isinstance(next_part, int) else {}
            current = current[part]

    # Set the final value
    final_part = parts[-1]
    if isinstance(final_part, int):
        if isinstance(current, list) and 0 <= final_part < len(current):
            current[final_part] = value
    else:
        # Must be a dict to set with string key
        if isinstance(current, dict):
            current[final_part] = value

    return result


def parse_path(path: str) -> list:
    """
    Parse a dot-notation path into parts.

    Examples:
        "a.b.c" -> ["a", "b", "c"]
        "a[0].b" -> ["a", 0, "b"]
        "a[*].b" -> ["a", "*", "b"]

    Args:
        path: Dot-notation path string

    Returns:
        List of path parts (strings and ints)
    """
    parts = []
    current = ""

    i = 0
    while i < len(path):
        char = path[i]

        if char == ".":
            if current:
                parts.append(current)
                current = ""
        elif char == "[":
            if current:
                parts.append(current)
                current = ""
            # Find closing bracket
            j = i + 1
            while j < len(path) and path[j] != "]":
                j += 1
            bracket_content = path[i + 1:j]
            if bracket_content == "*":
                parts.append("*")
            else:
                parts.append(int(bracket_content))
            i = j  # Skip to closing bracket
        else:
            current += char

        i += 1

    if current:
        parts.append(current)

    return parts


def extract_all_ids(data: Any, id_keys: list[str] = None) -> list[str]:
    """
    Recursively extract all ID values from a JSON structure.

    Args:
        data: JSON data to search
        id_keys: List of key names to treat as IDs (default: ["id"])

    Returns:
        List of all ID values found
    """
    if id_keys is None:
        id_keys = ["id"]

    ids = []

    def _extract(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key in id_keys and isinstance(value, str) and value:
                    ids.append(value)
                _extract(value)
        elif isinstance(obj, list):
            for item in obj:
                _extract(item)

    _extract(data)
    return ids


def content_diff(original: dict, modified: dict) -> dict:
    """
    Compute a simple diff between two dicts.

    Returns:
        Dict with 'added', 'removed', 'changed' keys
    """
    diff = {
        "added": [],
        "removed": [],
        "changed": []
    }

    def _compare(orig, mod, path=""):
        if type(orig) != type(mod):
            diff["changed"].append({"path": path, "from": type(orig).__name__, "to": type(mod).__name__})
            return

        if isinstance(orig, dict):
            all_keys = set(orig.keys()) | set(mod.keys())
            for key in all_keys:
                new_path = f"{path}.{key}" if path else key
                if key not in orig:
                    diff["added"].append(new_path)
                elif key not in mod:
                    diff["removed"].append(new_path)
                else:
                    _compare(orig[key], mod[key], new_path)
        elif isinstance(orig, list):
            if len(orig) != len(mod):
                diff["changed"].append({"path": path, "from": f"len={len(orig)}", "to": f"len={len(mod)}"})
            for i, (o, m) in enumerate(zip(orig, mod)):
                _compare(o, m, f"{path}[{i}]")
        elif orig != mod:
            diff["changed"].append({"path": path, "from": orig, "to": mod})

    _compare(original, modified)
    return diff
