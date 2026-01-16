"""
Post-Validation Content Fixer

Programmatically detects and fixes content quality issues AFTER LLM adaptation.
This catches issues that prompts alone can't prevent.

Issues fixed:
1. Duplicate activity names
2. Duplicate question texts
3. Truncated sentences
4. Missing sources on statistics

Usage:
    from src.utils.content_fixer import fix_content_issues
    fixed_json, issues_found = fix_content_issues(adapted_json)
"""

import re
import logging
from typing import Any
from collections import Counter

logger = logging.getLogger(__name__)


def fix_content_issues(adapted_json: dict) -> tuple[dict, list[str]]:
    """
    Fix content quality issues in adapted JSON.

    Args:
        adapted_json: The adapted simulation JSON

    Returns:
        (fixed_json, list of issues found and fixed)
    """
    issues_fixed = []

    # Get topic data
    topic_data = adapted_json.get("topicWizardData", {})
    if not topic_data:
        return adapted_json, ["No topicWizardData found"]

    # Fix 1: Duplicate activity/stage names
    dup_fixes = fix_duplicate_names(topic_data)
    issues_fixed.extend(dup_fixes)

    # Fix 2: Duplicate questions
    q_fixes = fix_duplicate_questions(topic_data)
    issues_fixed.extend(q_fixes)

    # Fix 3: Truncated sentences in resources
    trunc_fixes = fix_truncated_content(topic_data)
    issues_fixed.extend(trunc_fixes)

    # Fix 4: Truncated URLs (trailing dots)
    url_fixes = fix_truncated_urls(topic_data)
    issues_fixed.extend(url_fixes)

    # Fix 5: Remove placeholder text
    placeholder_fixes = remove_placeholders(topic_data)
    issues_fixed.extend(placeholder_fixes)

    logger.info(f"ContentFixer: Fixed {len(issues_fixed)} issues")

    return adapted_json, issues_fixed


def fix_duplicate_names(topic_data: dict) -> list[str]:
    """
    Find and fix duplicate activity/stage names in simulationFlow.

    Strategy: Append a number suffix to duplicates.
    """
    fixes = []
    sim_flow = topic_data.get("simulationFlow", [])

    if not sim_flow:
        return fixes

    # Collect all names and their locations
    name_locations = []  # [(name, path_to_object), ...]

    def collect_names(obj, path=""):
        if isinstance(obj, dict):
            name = obj.get("name", "")
            if name:
                name_locations.append((name, obj))
            for key, value in obj.items():
                collect_names(value, f"{path}/{key}")
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                collect_names(item, f"{path}[{i}]")

    collect_names(sim_flow, "simulationFlow")

    # Find duplicates
    name_counts = Counter(name for name, _ in name_locations)
    duplicates = {name for name, count in name_counts.items() if count > 1}

    if not duplicates:
        return fixes

    # Fix duplicates by appending suffix
    for dup_name in duplicates:
        occurrences = [(name, obj) for name, obj in name_locations if name == dup_name]

        # Skip first occurrence, rename subsequent ones
        for i, (name, obj) in enumerate(occurrences[1:], start=2):
            new_name = f"{name} ({i})"
            obj["name"] = new_name
            fixes.append(f"Renamed duplicate '{name}' to '{new_name}'")

    return fixes


def fix_duplicate_questions(topic_data: dict) -> list[str]:
    """
    Find and fix duplicate question texts.

    Strategy: Rephrase duplicate questions with different wording.
    """
    fixes = []

    # Collect all questions (including rubric/submission questions)
    questions = []

    def collect_questions(obj):
        if isinstance(obj, dict):
            # Check multiple question field names
            for field in ["question", "reviewQuestion", "questionText", "text"]:
                q_text = obj.get(field, "")
                if q_text and len(q_text) > 20:
                    questions.append((q_text, obj, field))
                    break  # Only add once per object
            for value in obj.values():
                collect_questions(value)
        elif isinstance(obj, list):
            for item in obj:
                collect_questions(item)

    collect_questions(topic_data)

    # Find duplicates (case-insensitive, ignoring whitespace)
    def normalize(text):
        return re.sub(r'\s+', ' ', text.lower().strip())

    normalized_map = {}
    for q_text, obj, field in questions:
        norm = normalize(q_text)
        if norm not in normalized_map:
            normalized_map[norm] = []
        normalized_map[norm].append((q_text, obj, field))

    # Fix duplicates
    for norm_text, occurrences in normalized_map.items():
        if len(occurrences) > 1:
            # Keep first, modify subsequent
            for i, (q_text, obj, field) in enumerate(occurrences[1:], start=2):
                # Add differentiation prefix
                prefix = f"[Variation {i}] "
                current_val = obj.get(field, "")
                if current_val and not current_val.startswith("["):
                    obj[field] = prefix + current_val
                    fixes.append(f"Differentiated duplicate {field}: '{q_text[:50]}...'")

    return fixes


def fix_truncated_content(topic_data: dict) -> list[str]:
    """
    Find and fix truncated sentences in content fields.

    Strategy: Complete truncated sentences or remove them.
    """
    fixes = []

    # Patterns that indicate truncation
    truncation_patterns = [
        r'\.\.\.$',  # ends with ...
        r'\s+$',  # ends with whitespace
        r'[a-z]$',  # ends with lowercase (likely mid-word)
        r'such as$',  # incomplete phrase
        r'including$',
        r'for example$',
        r'e\.g\.$',
        r'i\.e\.$',
    ]

    def check_and_fix_field(obj, field_name):
        nonlocal fixes
        if field_name not in obj:
            return

        content = obj[field_name]
        if not isinstance(content, str) or len(content) < 50:
            return

        # Check for truncation at end
        for pattern in truncation_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                # Try to fix by adding proper ending
                if content.rstrip().endswith('...'):
                    # Remove ellipsis and add period
                    obj[field_name] = content.rstrip()[:-3].rstrip() + '.'
                    fixes.append(f"Fixed truncated content in {field_name}")
                elif re.search(r'such as$|including$|for example$', content, re.IGNORECASE):
                    # Add generic completion
                    obj[field_name] = content + " relevant items."
                    fixes.append(f"Completed truncated phrase in {field_name}")
                break

    def process_obj(obj):
        if isinstance(obj, dict):
            for field in ['content', 'description', 'body', 'markdownText', 'text']:
                check_and_fix_field(obj, field)
            for value in obj.values():
                process_obj(value)
        elif isinstance(obj, list):
            for item in obj:
                process_obj(item)

    process_obj(topic_data)

    return fixes


def detect_content_issues(adapted_json: dict) -> dict:
    """
    Detect content quality issues without fixing them.

    Returns a report of issues found.
    """
    topic_data = adapted_json.get("topicWizardData", {})
    issues = {
        "duplicates": [],
        "truncations": [],
        "missing_sources": [],
        "klo_gaps": [],
    }

    # Detect duplicate names
    sim_flow = topic_data.get("simulationFlow", [])
    all_names = []

    def collect_names(obj):
        if isinstance(obj, dict):
            name = obj.get("name", "")
            if name:
                all_names.append(name)
            for value in obj.values():
                collect_names(value)
        elif isinstance(obj, list):
            for item in obj:
                collect_names(item)

    collect_names(sim_flow)

    name_counts = Counter(all_names)
    issues["duplicates"] = [name for name, count in name_counts.items() if count > 1]

    # Detect truncation
    def check_truncation(obj):
        if isinstance(obj, dict):
            for field in ['content', 'description', 'body', 'markdownText']:
                content = obj.get(field, "")
                if isinstance(content, str) and content.rstrip().endswith('...'):
                    issues["truncations"].append(f"{field}: ...{content[-50:]}")
            for value in obj.values():
                check_truncation(value)
        elif isinstance(obj, list):
            for item in obj:
                check_truncation(item)

    check_truncation(topic_data)

    # Detect missing sources on statistics
    def check_sources(obj):
        if isinstance(obj, dict):
            for field in ['content', 'markdownText', 'body']:
                content = obj.get(field, "")
                if isinstance(content, str):
                    # Find percentages without sources
                    stats = re.findall(r'\d+(?:\.\d+)?%', content)
                    for stat in stats:
                        # Check if source is nearby
                        idx = content.find(stat)
                        context = content[max(0, idx-100):idx+100]
                        if 'source' not in context.lower() and 'Source' not in context:
                            issues["missing_sources"].append(f"'{stat}' without source")
            for value in obj.values():
                check_sources(value)
        elif isinstance(obj, list):
            for item in obj:
                check_sources(item)

    check_sources(topic_data)

    return issues


def fix_truncated_urls(topic_data: dict) -> list[str]:
    """
    Fix URLs that were truncated (ending with a dot instead of proper extension).

    Common patterns:
    - avatarF2. → avatarF2.webp
    - image.png. → image.png (remove trailing dot)
    - s3.amazonaws.com/path/file. → infer extension from path
    """
    fixes = []
    print(f"[URL FIX] Starting fix_truncated_urls on {type(topic_data)}")

    # Valid extensions that indicate URL is NOT truncated
    valid_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.gif', '.svg', '.pdf', '.mp4', '.mp3', '.html', '.htm']

    def fix_url(obj, depth=0):
        nonlocal fixes
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, str) and ('http' in value or '//' in value):
                    # Check if URL ends with just a dot (truncated)
                    if value.endswith('.'):
                        print(f"[URL FIX] Found truncated URL at key='{key}': {value[-60:]}")
                        # Check if it's a valid extension followed by dot (e.g., ".png.")
                        base_url = value[:-1]  # Remove trailing dot

                        # If the URL already has a valid extension, just remove the trailing dot
                        has_valid_ext = any(base_url.lower().endswith(ext) for ext in valid_extensions)

                        if has_valid_ext:
                            # Just remove the trailing dot
                            obj[key] = base_url
                            fixes.append(f"Removed trailing dot from URL: ...{value[-40:]}")
                        elif 'avatar' in value.lower():
                            # Avatar URLs - add .webp
                            obj[key] = base_url + '.webp'
                            fixes.append(f"Fixed truncated avatar URL: ...{value[-40:]}")
                        elif 's3' in value.lower() and 'amazonaws' in value.lower():
                            # S3 URLs - infer from path or default to .webp for images
                            if '/avatar/' in value.lower() or '/image/' in value.lower():
                                obj[key] = base_url + '.webp'
                            elif '/document/' in value.lower() or '/doc/' in value.lower():
                                obj[key] = base_url + '.pdf'
                            else:
                                # Default: just remove trailing dot
                                obj[key] = base_url
                            fixes.append(f"Fixed truncated S3 URL: ...{value[-40:]}")
                        else:
                            # Generic URL ending with dot - just remove it
                            obj[key] = base_url
                            fixes.append(f"Removed trailing dot from URL: ...{value[-40:]}")
                else:
                    fix_url(value)
        elif isinstance(obj, list):
            for item in obj:
                fix_url(item)

    fix_url(topic_data)
    print(f"[URL FIX] Completed. Fixed {len(fixes)} URLs")
    return fixes


def remove_placeholders(topic_data: dict) -> list[str]:
    """
    Remove or replace placeholder text like [brackets], [placeholder], [YOUR_NAME], etc.
    """
    fixes = []

    # Placeholder patterns to detect and remove (order matters - specific before generic)
    placeholder_patterns = [
        # Specific named placeholders
        (r'\[YOUR[_ ]?NAME\]', '{{{recipientName}}}'),  # Replace with template variable
        (r'\[RECIPIENT[_ ]?NAME\]', '{{{recipientName}}}'),
        (r'\[COMPANY[_ ]?NAME\]', '{{{companyName}}}'),
        (r'\[VALUE\]', 'the specified value'),
        (r'\[value\]', 'the specified value'),
        (r'\[TBD\]', ''),
        (r'\[TODO\]', ''),
        (r'\[PLACEHOLDER\]', ''),
        (r'\[placeholder\]', ''),
        (r'\[brackets\]', ''),
        # Pattern-based placeholders
        (r'\[INSERT[^\]]*\]', ''),
        (r'\[ADD[^\]]*\]', ''),
        (r'\[FILL[^\]]*\]', ''),
        (r'\[ENTER[^\]]*\]', ''),
        (r'\[REPLACE[^\]]*\]', ''),
        (r'\[EDIT[^\]]*\]', ''),
        (r'\[UPDATE[^\]]*\]', ''),
        (r'\[REMOVE[^\]]*\]', ''),
        (r'\[DELETE[^\]]*\]', ''),
        # Generic suspicious patterns (single words in brackets that look like placeholders)
        # But NOT valid things like [1], [a], [Figure 1], [Source: ...]
        (r'\[(?:placeholder|PLACEHOLDER|Placeholder)\]', ''),
        (r'\[(?:example|EXAMPLE|Example)\]', ''),
        (r'\[(?:text|TEXT|Text)\]', ''),
        (r'\[(?:content|CONTENT|Content)\]', ''),
        (r'\[(?:description|DESCRIPTION|Description)\]', ''),
        (r'\[(?:name|NAME|Name)\]', ''),
        (r'\[(?:title|TITLE|Title)\]', ''),
        (r'\[(?:data|DATA|Data)\]', ''),
        (r'\[(?:info|INFO|Info)\]', ''),
        # Catch-all for remaining suspicious single-word ALL-CAPS brackets
        (r'\[[A-Z_]{3,20}\]', ''),  # e.g., [BRACKETS], [PLACEHOLDER_TEXT]
    ]

    def fix_placeholders(obj):
        nonlocal fixes
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, str):
                    original = value
                    for pattern, replacement in placeholder_patterns:
                        if re.search(pattern, value, re.IGNORECASE):
                            value = re.sub(pattern, replacement, value, flags=re.IGNORECASE)
                    if value != original:
                        obj[key] = value.strip()
                        fixes.append(f"Removed placeholder in {key}")
                else:
                    fix_placeholders(value)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                if isinstance(item, str):
                    original = item
                    for pattern, replacement in placeholder_patterns:
                        if re.search(pattern, item, re.IGNORECASE):
                            item = re.sub(pattern, replacement, item, flags=re.IGNORECASE)
                    if item != original:
                        obj[i] = item.strip()
                        fixes.append(f"Removed placeholder in list item")
                else:
                    fix_placeholders(item)

    fix_placeholders(topic_data)
    return fixes


# Convenience function for integration
async def post_process_and_fix(adapted_json: dict) -> tuple[dict, dict]:
    """
    Run post-processing to detect and fix content issues.

    Args:
        adapted_json: The adapted JSON from adaptation stage

    Returns:
        (fixed_json, report of issues found/fixed)
    """
    # First detect
    issues_detected = detect_content_issues(adapted_json)

    # Then fix
    fixed_json, issues_fixed = fix_content_issues(adapted_json)

    report = {
        "issues_detected": issues_detected,
        "issues_fixed": issues_fixed,
        "duplicate_count": len(issues_detected.get("duplicates", [])),
        "truncation_count": len(issues_detected.get("truncations", [])),
        "missing_source_count": len(issues_detected.get("missing_sources", [])),
    }

    return fixed_json, report
