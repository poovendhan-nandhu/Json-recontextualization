"""Utility helper functions."""
import hashlib
import json
import re
from typing import Any
from jsondiff import diff as json_diff
from datetime import datetime


def compute_sha256(data: Any) -> str:
    """Compute SHA-256 hash of JSON data."""
    if isinstance(data, dict) or isinstance(data, list):
        json_str = json.dumps(data, sort_keys=True, ensure_ascii=False)
    else:
        json_str = str(data)
    
    return hashlib.sha256(json_str.encode('utf-8')).hexdigest()


def get_by_path(obj: dict, path: str) -> Any:
    """Get value from nested dict using JSONPath-like notation."""
    if path == "":
        return obj
    
    keys = path.split('.')
    current = obj
    
    for key in keys:
        # Handle array notation like "simulationFlow[0]"
        if '[' in key:
            key_name = key.split('[')[0]
            indices = re.findall(r'\[(\d+)\]', key)
            current = current.get(key_name, {})
            for idx in indices:
                current = current[int(idx)]
        else:
            current = current.get(key, {})
    
    return current


def set_by_path(obj: dict, path: str, value: Any) -> None:
    """Set value in nested dict using JSONPath-like notation."""
    keys = path.split('.')
    current = obj
    
    for i, key in enumerate(keys[:-1]):
        if '[' in key:
            key_name = key.split('[')[0]
            indices = re.findall(r'\[(\d+)\]', key)
            current = current.setdefault(key_name, {})
            for idx in indices:
                current = current[int(idx)]
        else:
            current = current.setdefault(key, {})
    
    last_key = keys[-1]
    if '[' in last_key:
        key_name = last_key.split('[')[0]
        indices = re.findall(r'\[(\d+)\]', last_key)
        target = current.setdefault(key_name, [])
        for idx in indices[:-1]:
            target = target[int(idx)]
        target[int(indices[-1])] = value
    else:
        current[last_key] = value


def generate_json_diff(original: dict, transformed: dict) -> list[str]:
    """Generate list of changed JSONPaths between two dicts."""
    differences = json_diff(original, transformed)
    changed_paths = []
    
    def extract_paths(diff_obj, prefix=""):
        if isinstance(diff_obj, dict):
            for key, value in diff_obj.items():
                current_path = f"{prefix}.{key}" if prefix else str(key)
                if isinstance(value, (dict, list)):
                    extract_paths(value, current_path)
                else:
                    changed_paths.append(current_path)
        elif isinstance(diff_obj, list):
            for idx, item in enumerate(diff_obj):
                current_path = f"{prefix}[{idx}]"
                if isinstance(item, (dict, list)):
                    extract_paths(item, current_path)
                else:
                    changed_paths.append(current_path)
    
    extract_paths(differences)
    return changed_paths


def extract_all_text_values(obj: Any, exclude_paths: list[str] = None) -> list[str]:
    """Extract all text values from a nested structure."""
    exclude_paths = exclude_paths or []
    text_values = []
    
    def _extract(data, current_path=""):
        if current_path in exclude_paths:
            return
        
        if isinstance(data, dict):
            for key, value in data.items():
                new_path = f"{current_path}.{key}" if current_path else key
                _extract(value, new_path)
        elif isinstance(data, list):
            for idx, item in enumerate(data):
                new_path = f"{current_path}[{idx}]"
                _extract(item, new_path)
        elif isinstance(data, str) and len(data) > 0:
            text_values.append(data)
    
    _extract(obj)
    return text_values


def search_keywords(obj: Any, keywords: list[str], exclude_paths: list[str] = None) -> list[dict]:
    """Search for keywords in all text fields."""
    exclude_paths = exclude_paths or []
    findings = []
    
    def _search(data, current_path=""):
        # Check if current path should be excluded
        # Support both exact match and partial match (e.g., "scenarioOptions" excludes "topicWizardData.scenarioOptions[0]")
        for exclude in exclude_paths:
            if exclude in current_path:
                return
        
        if isinstance(data, dict):
            for key, value in data.items():
                new_path = f"{current_path}.{key}" if current_path else key
                _search(value, new_path)
        elif isinstance(data, list):
            for idx, item in enumerate(data):
                new_path = f"{current_path}[{idx}]"
                _search(item, new_path)
        elif isinstance(data, str):
            for keyword in keywords:
                if keyword.lower() in data.lower():
                    findings.append({
                        "path": current_path,
                        "keyword": keyword,
                        "context": data[:100] + "..." if len(data) > 100 else data
                    })
    
    _search(obj)
    return findings


def create_log_entry(node_name: str, status: str, duration_ms: int, **kwargs) -> dict:
    """Create a standardized log entry."""
    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "node": node_name,
        "status": status,
        "duration_ms": duration_ms,
        **kwargs
    }


def truncate_for_preview(text: str, max_length: int = 100) -> str:
    """Truncate text for preview display."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."
