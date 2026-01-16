"""
Stage 2: Sharder

Splits simulation JSON into independent shards for parallel processing.
Each shard can be validated and fixed independently.
"""
import copy
import logging
import sys
import os
from typing import Any, Optional

# Handle both package and direct imports
try:
    from ..models.shard import (
        Shard,
        ShardCollection,
        LockState,
        ShardStatus,
    )
    from ..utils.config import SHARD_DEFINITIONS, config
    from ..utils.hash import (
        compute_hash,
        get_nested_value,
        set_nested_value,
        extract_all_ids,
    )
except ImportError:
    # Direct import for testing
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models.shard import (
        Shard,
        ShardCollection,
        LockState,
        ShardStatus,
    )
    from utils.config import SHARD_DEFINITIONS, config
    from utils.hash import (
        compute_hash,
        get_nested_value,
        set_nested_value,
        extract_all_ids,
    )

logger = logging.getLogger(__name__)


class Sharder:
    """
    Splits simulation JSON into independent shards.

    Each shard:
    - Contains a subset of the JSON based on defined paths
    - Has a hash for change detection
    - Has a lock state (locked shards cannot be modified)
    - Tracks which other shards it aligns with
    """

    def __init__(self, shard_definitions: dict = None):
        """
        Initialize sharder with shard definitions.

        Args:
            shard_definitions: Custom shard definitions (uses default if None)
        """
        self.shard_definitions = shard_definitions or SHARD_DEFINITIONS

    def shard(self, data: dict, scenario_prompt: str = "") -> ShardCollection:
        """
        Split JSON into shards.

        Args:
            data: Full simulation JSON
            scenario_prompt: Target scenario prompt for adaptation

        Returns:
            ShardCollection containing all shards
        """
        logger.info(f"Sharding JSON with {len(self.shard_definitions)} shard definitions")

        shards = []
        source_hash = compute_hash(data)

        for shard_id, definition in self.shard_definitions.items():
            shard = self._create_shard(shard_id, definition, data)
            shards.append(shard)
            logger.debug(f"Created shard '{shard_id}' with {len(shard.extracted_ids)} IDs")

        collection = ShardCollection(
            shards=shards,
            source_json_hash=source_hash,
            scenario_prompt=scenario_prompt,
        )

        logger.info(f"Created {len(shards)} shards, {len(collection.all_ids())} total IDs extracted")
        return collection

    def _create_shard(self, shard_id: str, definition: dict, data: dict) -> Shard:
        """
        Create a single shard from definition.

        Args:
            shard_id: Unique shard identifier
            definition: Shard definition from config
            data: Full JSON data

        Returns:
            Populated Shard instance
        """
        # Extract content for all paths
        content = {}
        for path in definition["paths"]:
            value = self._extract_path(data, path)
            if value is not None:
                content[path] = value

        # Compute hash
        content_hash = compute_hash(content)

        # Extract all IDs from content
        extracted_ids = extract_all_ids(content)

        # Determine lock state
        lock_state = LockState.FULLY_LOCKED if definition["locked"] else LockState.UNLOCKED

        return Shard(
            id=shard_id,
            name=definition["name"],
            paths=definition["paths"],
            content=content,
            original_hash=content_hash,
            current_hash=content_hash,
            lock_state=lock_state,
            status=ShardStatus.PENDING,
            is_blocker=definition["is_blocker"],
            aligns_with=definition.get("aligns_with", []),
            extracted_ids=extracted_ids,
        )

    def _extract_path(self, data: dict, path: str) -> Any:
        """
        Extract value at path, handling wildcards.

        Supports:
        - Simple paths: "topicWizardData.id"
        - Wildcards: "topicWizardData.simulationFlow[*].data.email"

        Args:
            data: Source data
            path: Path to extract

        Returns:
            Extracted value or None
        """
        if "[*]" in path:
            return self._extract_wildcard_path(data, path)
        return get_nested_value(data, path)

    def _extract_wildcard_path(self, data: dict, path: str) -> list[dict]:
        """
        Extract values from wildcard paths.

        Example: "topicWizardData.simulationFlow[*].data.email"
        Returns all email objects from all simulation flow items.

        Args:
            data: Source data
            path: Path with [*] wildcards

        Returns:
            List of extracted values with their indices
        """
        results = []
        parts = path.split("[*]")

        if len(parts) < 2:
            return results

        # Get the array
        array_path = parts[0].rstrip(".")
        array = get_nested_value(data, array_path)

        if not isinstance(array, list):
            return results

        # Extract from each array item
        remainder = parts[1].lstrip(".")
        for i, item in enumerate(array):
            if remainder:
                # Handle nested wildcards recursively
                if "[*]" in remainder:
                    nested = self._extract_wildcard_path(item, remainder)
                    for n in nested:
                        results.append({
                            "index": i,
                            "nested_index": n.get("index"),
                            "value": n.get("value"),
                            "path": f"{array_path}[{i}].{n.get('path', remainder)}"
                        })
                else:
                    value = get_nested_value(item, remainder)
                    if value is not None:
                        results.append({
                            "index": i,
                            "value": value,
                            "path": f"{array_path}[{i}].{remainder}"
                        })
            else:
                results.append({
                    "index": i,
                    "value": item,
                    "path": f"{array_path}[{i}]"
                })

        return results


def merge_shards(collection: ShardCollection, original_data: dict) -> dict:
    """
    Merge shards back into a complete JSON.

    Args:
        collection: ShardCollection with (possibly modified) shards
        original_data: Original JSON structure (for locked fields)

    Returns:
        Merged JSON
    """
    logger.info("Merging shards back into JSON")

    # Start with a deep copy of original
    result = copy.deepcopy(original_data)

    for shard in collection.shards:
        # Skip locked shards - they stay unchanged
        if shard.lock_state == LockState.FULLY_LOCKED:
            logger.debug(f"Skipping locked shard '{shard.id}'")
            continue

        # Apply modified content back to result
        for path, value in shard.content.items():
            if "[*]" in path:
                # Handle wildcard paths
                result = _merge_wildcard_content(result, path, value)
            else:
                result = set_nested_value(result, path, value)

        logger.debug(f"Merged shard '{shard.id}'")

    return result


def _merge_wildcard_content(data: dict, path: str, values: list[dict]) -> dict:
    """
    Merge wildcard path content back into data.

    Args:
        data: Target data structure
        path: Original wildcard path
        values: List of extracted values with indices

    Returns:
        Updated data
    """
    if not isinstance(values, list):
        return data

    for item in values:
        if not isinstance(item, dict):
            continue

        item_path = item.get("path")
        item_value = item.get("value")

        if item_path and item_value is not None:
            data = set_nested_value(data, item_path, item_value)

    return data


def cleanup_merged_json(data: dict) -> dict:
    """
    Post-processing cleanup for merged JSON.

    Fixes:
    - Trailing dots in simulationImage, simulationName
    - Trailing dots in flowProperties.type
    - Truncated URLs (avatarF2. → avatarF2.webp)
    - Placeholder text removal
    - Inconsistent template braces (normalizes to {{{var}}})

    Args:
        data: Merged JSON

    Returns:
        Cleaned JSON
    """
    import re
    from ..utils.content_fixer import fix_truncated_urls, remove_placeholders

    result = copy.deepcopy(data)
    tw = result.get("topicWizardData", {})

    # === CRITICAL: Fix truncated URLs BEFORE any other processing ===
    # This catches avatar URLs ending with . and fixes them to .webp
    url_fixes = fix_truncated_urls(tw)
    if url_fixes:
        logger.info(f"Fixed {len(url_fixes)} truncated URLs: {url_fixes[:3]}")

    # === Fix placeholders ===
    placeholder_fixes = remove_placeholders(tw)
    if placeholder_fixes:
        logger.info(f"Fixed {len(placeholder_fixes)} placeholders: {placeholder_fixes[:3]}")

    # Fix trailing dots in specific fields
    if tw.get("simulationImage", "").endswith("."):
        tw["simulationImage"] = tw["simulationImage"].rstrip(".")
        logger.debug("Removed trailing dot from simulationImage")

    if tw.get("simulationName", "").endswith("."):
        tw["simulationName"] = tw["simulationName"].rstrip(".")
        logger.debug("Removed trailing dot from simulationName")

    # Fix flowProperties.type trailing dots
    def fix_flow_properties(obj):
        if isinstance(obj, dict):
            if "flowProperties" in obj and isinstance(obj["flowProperties"], dict):
                fp_type = obj["flowProperties"].get("type", "")
                if fp_type.endswith("."):
                    obj["flowProperties"]["type"] = fp_type.rstrip(".")
            for v in obj.values():
                fix_flow_properties(v)
        elif isinstance(obj, list):
            for item in obj:
                fix_flow_properties(item)

    fix_flow_properties(tw)

    # Normalize template braces: {{var}} -> {{{var}}}
    def normalize_braces(text: str) -> str:
        # Convert {{var}} to {{{var}}} for consistency
        # But don't touch already triple-braced
        pattern = r'\{\{([a-zA-Z]+)\}\}(?!\})'
        return re.sub(pattern, r'{{{\1}}}', text)

    def fix_braces_in_obj(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, str):
                    obj[k] = normalize_braces(v)
                else:
                    fix_braces_in_obj(v)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                if isinstance(item, str):
                    obj[i] = normalize_braces(item)
                else:
                    fix_braces_in_obj(item)

    fix_braces_in_obj(tw)

    result["topicWizardData"] = tw
    logger.info("Post-processing cleanup complete")
    return result


def shard_json(data: dict, scenario_prompt: str = "") -> ShardCollection:
    """
    Convenience function to shard a JSON.

    Args:
        data: Simulation JSON
        scenario_prompt: Target scenario prompt

    Returns:
        ShardCollection
    """
    sharder = Sharder()
    return sharder.shard(data, scenario_prompt)


def get_shard_summary(collection: ShardCollection) -> dict:
    """
    Get a summary of the shard collection.

    Returns:
        Summary dict with counts and status
    """
    return {
        "total_shards": len(collection.shards),
        "locked_shards": len([s for s in collection.shards if s.lock_state == LockState.FULLY_LOCKED]),
        "unlocked_shards": len([s for s in collection.shards if s.lock_state != LockState.FULLY_LOCKED]),
        "blocker_shards": len([s for s in collection.shards if s.is_blocker]),
        "total_ids_extracted": len(collection.all_ids()),
        "shards": [
            {
                "id": s.id,
                "name": s.name,
                "locked": s.lock_state == LockState.FULLY_LOCKED,
                "is_blocker": s.is_blocker,
                "paths_count": len(s.paths),
                "ids_count": len(s.extracted_ids),
                "aligns_with": s.aligns_with,
            }
            for s in collection.shards
        ]
    }
