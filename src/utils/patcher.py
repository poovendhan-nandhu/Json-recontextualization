"""
JSON Pointer Patcher with Rollback Support

Surgical JSON patching using only 'replace' operations.
Supports rollback for repair cycles.

JSON Pointer spec: RFC 6901
"""
import logging
import copy
from typing import Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class PatchOp:
    """A single patch operation."""
    op: str  # Always "replace" - we don't use add/remove/move
    path: str  # JSON Pointer path (e.g., "/topicWizardData/simulationName")
    value: Any  # New value to set
    old_value: Any = None  # Original value (for rollback)
    reason: str = ""  # Why this fix was needed
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "op": self.op,
            "path": self.path,
            "value": self.value,
            "old_value": self.old_value,
            "reason": self.reason,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PatchOp":
        return cls(
            op=data["op"],
            path=data["path"],
            value=data["value"],
            old_value=data.get("old_value"),
            reason=data.get("reason", ""),
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else datetime.utcnow(),
        )


@dataclass
class PatchResult:
    """Result of applying patches."""
    success: bool
    patched_data: dict
    applied_patches: list[PatchOp] = field(default_factory=list)
    failed_patches: list[tuple[PatchOp, str]] = field(default_factory=list)  # (patch, error)

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "applied_count": len(self.applied_patches),
            "failed_count": len(self.failed_patches),
            "applied_patches": [p.to_dict() for p in self.applied_patches],
            "failed_patches": [(p.to_dict(), err) for p, err in self.failed_patches],
        }


class JSONPatcher:
    """
    JSON Patcher with rollback support.

    Uses JSON Pointer (RFC 6901) for precise field targeting.
    Only supports 'replace' operations for safety.
    """

    def __init__(self):
        self._patch_history: list[list[PatchOp]] = []  # Stack of patch batches for rollback

    # =========================================================================
    # JSON POINTER OPERATIONS
    # =========================================================================

    @staticmethod
    def parse_pointer(pointer: str) -> list[str]:
        """
        Parse JSON Pointer into path segments.

        "/topicWizardData/simulationFlow/0/data" -> ["topicWizardData", "simulationFlow", "0", "data"]
        """
        if not pointer:
            return []
        if not pointer.startswith("/"):
            raise ValueError(f"JSON Pointer must start with '/': {pointer}")

        # Split and unescape (~1 -> /, ~0 -> ~)
        parts = pointer[1:].split("/")
        return [p.replace("~1", "/").replace("~0", "~") for p in parts]

    @staticmethod
    def get_value(data: dict, pointer: str) -> Any:
        """
        Get value at JSON Pointer path.

        Args:
            data: The JSON object
            pointer: JSON Pointer path (e.g., "/topicWizardData/simulationName")

        Returns:
            Value at path, or raises KeyError if not found
        """
        if not pointer or pointer == "/":
            return data

        parts = JSONPatcher.parse_pointer(pointer)
        current = data

        for part in parts:
            if isinstance(current, dict):
                if part not in current:
                    raise KeyError(f"Key '{part}' not found at path '{pointer}'")
                current = current[part]
            elif isinstance(current, list):
                try:
                    idx = int(part)
                    current = current[idx]
                except (ValueError, IndexError) as e:
                    raise KeyError(f"Invalid array index '{part}' at path '{pointer}'") from e
            else:
                raise KeyError(f"Cannot traverse into {type(current)} at path '{pointer}'")

        return current

    @staticmethod
    def set_value(data: dict, pointer: str, value: Any) -> dict:
        """
        Set value at JSON Pointer path.

        Args:
            data: The JSON object (will be modified in place)
            pointer: JSON Pointer path
            value: New value to set

        Returns:
            Modified data dict
        """
        if not pointer or pointer == "/":
            raise ValueError("Cannot replace root object")

        parts = JSONPatcher.parse_pointer(pointer)
        current = data

        # Navigate to parent
        for part in parts[:-1]:
            if isinstance(current, dict):
                current = current[part]
            elif isinstance(current, list):
                current = current[int(part)]
            else:
                raise KeyError(f"Cannot traverse into {type(current)}")

        # Set the final value
        final_key = parts[-1]
        if isinstance(current, dict):
            current[final_key] = value
        elif isinstance(current, list):
            current[int(final_key)] = value
        else:
            raise KeyError(f"Cannot set value on {type(current)}")

        return data

    @staticmethod
    def path_exists(data: dict, pointer: str) -> bool:
        """Check if a JSON Pointer path exists."""
        try:
            JSONPatcher.get_value(data, pointer)
            return True
        except (KeyError, IndexError):
            return False

    # =========================================================================
    # PATCH OPERATIONS
    # =========================================================================

    def validate_patch(self, patch: PatchOp, data: dict) -> tuple[bool, Optional[str]]:
        """
        Validate a patch before applying.

        Returns:
            (is_valid, error_message)
        """
        # Only allow replace operations
        if patch.op != "replace":
            return False, f"Only 'replace' operation allowed, got '{patch.op}'"

        # Check path exists
        if not self.path_exists(data, patch.path):
            return False, f"Path does not exist: {patch.path}"

        # Type preservation check
        old_value = self.get_value(data, patch.path)
        if type(old_value) != type(patch.value):
            # Allow None -> value and value -> None
            if old_value is not None and patch.value is not None:
                return False, f"Type mismatch: {type(old_value).__name__} -> {type(patch.value).__name__}"

        return True, None

    def validate_patches(self, patches: list[PatchOp], data: dict) -> tuple[bool, list[str]]:
        """
        Validate all patches before applying.

        Returns:
            (all_valid, list_of_errors)
        """
        errors = []
        for patch in patches:
            is_valid, error = self.validate_patch(patch, data)
            if not is_valid:
                errors.append(f"{patch.path}: {error}")

        return len(errors) == 0, errors

    def apply_patch(self, data: dict, patch: PatchOp) -> dict:
        """
        Apply a single patch operation.

        Stores old value for rollback.
        """
        # Store old value if not already set
        if patch.old_value is None:
            patch.old_value = copy.deepcopy(self.get_value(data, patch.path))

        # Apply the replacement
        self.set_value(data, patch.path, patch.value)

        logger.debug(f"Applied patch: {patch.path} = {patch.value}")

        return data

    def apply_patches(
        self,
        data: dict,
        patches: list[PatchOp],
        validate_first: bool = True,
        stop_on_error: bool = True,
    ) -> PatchResult:
        """
        Apply multiple patches to data.

        Args:
            data: The JSON object to patch
            patches: List of patch operations
            validate_first: Validate all patches before applying
            stop_on_error: Stop on first error vs collect all errors

        Returns:
            PatchResult with success status and modified data
        """
        # Make a deep copy to avoid modifying original
        working_data = copy.deepcopy(data)
        applied = []
        failed = []

        # Pre-validation (log warnings but continue with valid patches)
        if validate_first:
            all_valid, errors = self.validate_patches(patches, working_data)
            if not all_valid:
                logger.warning(f"Some patches have invalid paths: {[e for e in errors if e]}")
                # Filter to only valid patches instead of failing completely
                valid_patches = [p for p, e in zip(patches, errors) if not e]
                invalid_patches = [(p, e) for p, e in zip(patches, errors) if e]
                failed.extend(invalid_patches)
                patches = valid_patches  # Continue with valid patches only
                if not patches:
                    logger.warning("No valid patches to apply")
                    return PatchResult(
                        success=False,
                        patched_data=data,
                        failed_patches=failed,
                    )

        # Apply patches
        for patch in patches:
            try:
                is_valid, error = self.validate_patch(patch, working_data)
                if not is_valid:
                    failed.append((patch, error))
                    if stop_on_error:
                        break
                    continue

                self.apply_patch(working_data, patch)
                applied.append(patch)

            except Exception as e:
                error_msg = str(e)
                failed.append((patch, error_msg))
                logger.error(f"Failed to apply patch {patch.path}: {error_msg}")
                if stop_on_error:
                    break

        # Store in history for rollback
        if applied:
            self._patch_history.append(applied)

        success = len(failed) == 0
        return PatchResult(
            success=success,
            patched_data=working_data if success else data,
            applied_patches=applied,
            failed_patches=failed,
        )

    # =========================================================================
    # ROLLBACK OPERATIONS
    # =========================================================================

    def rollback_patches(self, data: dict, patches: list[PatchOp]) -> dict:
        """
        Rollback patches by restoring old values.

        Args:
            data: Current data state
            patches: Patches to rollback (must have old_value set)

        Returns:
            Data with patches reversed
        """
        working_data = copy.deepcopy(data)

        # Rollback in reverse order
        for patch in reversed(patches):
            if patch.old_value is None:
                logger.warning(f"Cannot rollback {patch.path}: no old_value stored")
                continue

            try:
                self.set_value(working_data, patch.path, patch.old_value)
                logger.debug(f"Rolled back: {patch.path}")
            except Exception as e:
                logger.error(f"Failed to rollback {patch.path}: {e}")

        return working_data

    def rollback_last_batch(self, data: dict) -> Optional[dict]:
        """
        Rollback the last batch of patches.

        Returns:
            Data with last batch reversed, or None if no history
        """
        if not self._patch_history:
            logger.warning("No patch history to rollback")
            return None

        last_batch = self._patch_history.pop()
        return self.rollback_patches(data, last_batch)

    def get_patch_history(self) -> list[list[PatchOp]]:
        """Get all patch history batches."""
        return self._patch_history

    def clear_history(self):
        """Clear patch history."""
        self._patch_history = []

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    @staticmethod
    def create_patch(
        path: str,
        new_value: Any,
        reason: str = "",
        old_value: Any = None,
    ) -> PatchOp:
        """
        Create a replace patch operation.

        Args:
            path: JSON Pointer path
            new_value: Value to set
            reason: Explanation for the change
            old_value: Original value (optional, captured on apply)
        """
        return PatchOp(
            op="replace",
            path=path,
            value=new_value,
            old_value=old_value,
            reason=reason,
        )

    @staticmethod
    def diff_to_patches(original: dict, modified: dict, prefix: str = "") -> list[PatchOp]:
        """
        Generate patches by comparing two dicts.

        Only generates 'replace' operations for changed values.
        Does NOT generate add/remove operations.
        """
        patches = []

        for key in original:
            path = f"{prefix}/{key}"

            if key not in modified:
                # Key removed - we don't handle this (no remove ops)
                continue

            orig_val = original[key]
            mod_val = modified[key]

            if isinstance(orig_val, dict) and isinstance(mod_val, dict):
                # Recurse into nested dicts
                patches.extend(JSONPatcher.diff_to_patches(orig_val, mod_val, path))
            elif isinstance(orig_val, list) and isinstance(mod_val, list):
                # For lists, compare element by element
                for i in range(min(len(orig_val), len(mod_val))):
                    item_path = f"{path}/{i}"
                    if isinstance(orig_val[i], dict) and isinstance(mod_val[i], dict):
                        patches.extend(JSONPatcher.diff_to_patches(orig_val[i], mod_val[i], item_path))
                    elif orig_val[i] != mod_val[i]:
                        patches.append(PatchOp(
                            op="replace",
                            path=item_path,
                            value=mod_val[i],
                            old_value=orig_val[i],
                            reason="Value changed",
                        ))
            elif orig_val != mod_val:
                # Value changed
                patches.append(PatchOp(
                    op="replace",
                    path=path,
                    value=mod_val,
                    old_value=orig_val,
                    reason="Value changed",
                ))

        return patches


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_patcher: Optional[JSONPatcher] = None


def get_patcher() -> JSONPatcher:
    """Get singleton patcher instance."""
    global _patcher
    if _patcher is None:
        _patcher = JSONPatcher()
    return _patcher


def apply_patches(data: dict, patches: list[PatchOp]) -> PatchResult:
    """Apply patches to data. See JSONPatcher.apply_patches."""
    return get_patcher().apply_patches(data, patches)


def rollback_patches(data: dict, patches: list[PatchOp]) -> dict:
    """Rollback patches. See JSONPatcher.rollback_patches."""
    return get_patcher().rollback_patches(data, patches)


def create_patch(path: str, new_value: Any, reason: str = "") -> PatchOp:
    """Create a replace patch. See JSONPatcher.create_patch."""
    return JSONPatcher.create_patch(path, new_value, reason)
