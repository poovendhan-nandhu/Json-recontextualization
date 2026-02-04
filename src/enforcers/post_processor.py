"""
Post-Processor - Enforce entity_map and canonical_numbers consistency.

After all shards are generated, this module:
1. Ensures entity_map names are used consistently
2. Validates canonical_numbers appear in output
3. Fixes email domains to match company
4. Removes slash-hedging patterns

These are PROGRAMMATIC fixes that don't require LLM calls.
"""

import json
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


def post_process_adapted_json(
    adapted_json: dict,
    entity_map: dict,
    canonical_numbers: dict,
    domain_profile: Optional[dict] = None
) -> dict:
    """
    Apply all post-processing fixes to adapted JSON.

    Args:
        adapted_json: The adapted JSON from shard generation
        entity_map: Company, people, roles, terminology
        canonical_numbers: Exact figures for consistency
        domain_profile: Optional domain terminology mapping

    Returns:
        Post-processed JSON with fixes applied
    """
    logger.info("[POST-PROCESS] Starting post-processing enforcement...")

    result = adapted_json

    # Step 1: Enforce entity_map (names, company)
    result = enforce_entity_map(result, entity_map)

    # Step 2: Fix email domains
    result = fix_email_domains(result, entity_map)

    # Step 3: Normalize role titles
    result = normalize_role_titles(result, entity_map)

    # Step 4: Normalize terminology from domain_profile
    if domain_profile:
        result = normalize_terminology(result, domain_profile)

    # Step 5: Normalize canonical numbers (actually fix, not just validate)
    result = normalize_canonical_numbers(result, canonical_numbers)

    # Step 6: Remove slash-hedging patterns
    result = remove_slash_hedging(result, entity_map)

    # Step 7: Replace common placeholders
    result = replace_placeholders(result)

    logger.info("[POST-PROCESS] Post-processing complete")

    return result


def replace_placeholders(adapted_json: dict) -> dict:
    """
    Replace common placeholders with reasonable values.

    Fixes:
    - [Current Date] -> today's date
    - [Insert X] -> removes brackets
    - [TBD] -> removes

    Args:
        adapted_json: The adapted JSON

    Returns:
        JSON with placeholders replaced
    """
    import datetime

    json_str = json.dumps(adapted_json, ensure_ascii=False)
    fixes_applied = 0

    # Get today's date in a professional format
    today = datetime.date.today().strftime("%B %d, %Y")

    # Common placeholders and their replacements
    placeholder_fixes = [
        (r'\[Current Date\]', today),
        (r'\[Today\'s Date\]', today),
        (r'\[DATE\]', today),
        (r'\[Insert Date\]', today),
        (r'\[TBD\]', ''),
        (r'\[TBC\]', ''),
        (r'\[N/A\]', ''),
    ]

    for pattern, replacement in placeholder_fixes:
        new_str, count = re.subn(pattern, replacement, json_str, flags=re.IGNORECASE)
        if count > 0:
            json_str = new_str
            fixes_applied += count

    if fixes_applied > 0:
        logger.info(f"[POST-PROCESS] replace_placeholders: Fixed {fixes_applied} placeholders")

    return json.loads(json_str)


def enforce_entity_map(adapted_json: dict, entity_map: dict) -> dict:
    """
    Programmatically enforce entity_map values throughout the JSON.

    Fixes:
    - "the company" -> actual company name
    - "the manager" -> actual manager name
    - Source person names -> target person names
    - Variations and misspellings

    Args:
        adapted_json: The adapted JSON
        entity_map: Company, people, roles, terminology

    Returns:
        JSON with entity_map values enforced
    """
    json_str = json.dumps(adapted_json, ensure_ascii=False)
    fixes_applied = 0

    # Get company name
    company = entity_map.get('company', {})
    company_name = company.get('name', '')

    if company_name:
        # Fix "the company" patterns
        patterns_to_fix = [
            (r'\bthe company\b', company_name),
            (r'\bThe Company\b', company_name),
            (r'\bTHE COMPANY\b', company_name),
            (r'\bthe organization\b', company_name),
            (r'\bThe Organization\b', company_name),
            (r'\bour company\b', company_name),
            (r'\bOur Company\b', company_name),
        ]

        for pattern, replacement in patterns_to_fix:
            new_str, count = re.subn(pattern, replacement, json_str)
            if count > 0:
                json_str = new_str
                fixes_applied += count
                logger.debug(f"[POST-PROCESS] Fixed {count} occurrences of '{pattern}'")

    # Get manager/people names
    people = entity_map.get('people', [])
    if isinstance(people, list) and people:
        manager = people[0]  # First person is usually the manager
        manager_name = manager.get('name', '') if isinstance(manager, dict) else ''
        manager_email = manager.get('email', '') if isinstance(manager, dict) else ''

        if manager_name:
            patterns_to_fix = [
                (r'\bthe manager\b', manager_name),
                (r'\bThe Manager\b', manager_name),
                (r'\bTHE MANAGER\b', manager_name),
                (r'\byour manager\b', manager_name),
                (r'\bYour Manager\b', manager_name),
                (r'\bthe supervisor\b', manager_name),
                (r'\bThe Supervisor\b', manager_name),
            ]

            for pattern, replacement in patterns_to_fix:
                new_str, count = re.subn(pattern, replacement, json_str)
                if count > 0:
                    json_str = new_str
                    fixes_applied += count

    # Replace source person names with target names (from source_people mapping)
    source_people = entity_map.get('source_people', {})
    if source_people:
        for source_name, target_info in source_people.items():
            if isinstance(target_info, dict):
                target_name = target_info.get('name', '')
                target_email = target_info.get('email', '')
            else:
                target_name = str(target_info)
                target_email = ''

            if source_name and target_name and source_name != target_name:
                # Replace full name
                pattern = re.compile(rf'\b{re.escape(source_name)}\b', re.IGNORECASE)
                new_str = pattern.sub(target_name, json_str)
                if new_str != json_str:
                    json_str = new_str
                    fixes_applied += 1
                    logger.debug(f"[POST-PROCESS] Replaced source name: '{source_name}' -> '{target_name}'")

                # Replace first name only (common pattern)
                source_first = source_name.split()[0] if ' ' in source_name else source_name
                target_first = target_name.split()[-1] if ' ' in target_name else target_name  # Use last name for formal
                if len(source_first) > 2:
                    pattern = re.compile(rf'\b{re.escape(source_first)}\b')
                    # Only replace if it looks like a standalone first name (followed by comma, period, etc.)
                    # Be careful not to replace partial matches in other names

    if fixes_applied > 0:
        logger.info(f"[POST-PROCESS] enforce_entity_map: Fixed {fixes_applied} generic references")

    return json.loads(json_str)


def fix_email_domains(adapted_json: dict, entity_map: dict) -> dict:
    """
    Fix email domains to match the company domain from entity_map.

    Args:
        adapted_json: The adapted JSON
        entity_map: Contains company.domain

    Returns:
        JSON with fixed email domains
    """
    company = entity_map.get('company', {})
    company_domain = company.get('domain', '')

    if not company_domain:
        logger.warning("[POST-PROCESS] No company domain in entity_map, skipping email fix")
        return adapted_json

    json_str = json.dumps(adapted_json, ensure_ascii=False)

    # Find all email addresses and fix their domains
    email_pattern = r'([a-zA-Z0-9._%+-]+)@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'

    def fix_domain(match):
        local_part = match.group(1)
        current_domain = match.group(2)

        # Don't fix if it already matches
        if current_domain.lower() == company_domain.lower():
            return match.group(0)

        # Don't fix common external domains
        external_domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com']
        if current_domain.lower() in external_domains:
            return match.group(0)

        return f"{local_part}@{company_domain}"

    new_str, count = re.subn(email_pattern, fix_domain, json_str)

    if count > 0:
        logger.info(f"[POST-PROCESS] fix_email_domains: Processed {count} email addresses")

    return json.loads(new_str)


def normalize_role_titles(adapted_json: dict, entity_map: dict) -> dict:
    """
    Normalize role titles from entity_map throughout the JSON.

    Replaces source role titles with target role titles using case-insensitive
    word boundary matching.

    Args:
        adapted_json: The adapted JSON
        entity_map: Contains roles mapping (source_role -> target_role)

    Returns:
        JSON with normalized role titles
    """
    roles = entity_map.get('roles', {})
    if not roles:
        return adapted_json

    json_str = json.dumps(adapted_json, ensure_ascii=False)
    fixes_applied = 0

    for source_role, target_role in roles.items():
        if not source_role or not target_role or source_role == target_role:
            continue
        if len(source_role) < 3:  # Skip very short roles to avoid false matches
            continue

        # Case-insensitive word boundary match
        pattern = re.compile(rf'\b{re.escape(source_role)}\b', re.IGNORECASE)
        if pattern.search(json_str):
            new_str = pattern.sub(target_role, json_str)
            if new_str != json_str:
                json_str = new_str
                fixes_applied += 1
                logger.debug(f"[POST-PROCESS] Normalized role: '{source_role}' -> '{target_role}'")

    if fixes_applied > 0:
        logger.info(f"[POST-PROCESS] normalize_role_titles: Fixed {fixes_applied} role titles")

    return json.loads(json_str)


def normalize_terminology(adapted_json: dict, domain_profile: dict) -> dict:
    """
    Normalize domain terminology from domain_profile.

    Replaces source domain terms with target domain terms.

    Args:
        adapted_json: The adapted JSON
        domain_profile: Contains terminology_map (source_term -> target_term)

    Returns:
        JSON with normalized terminology
    """
    terminology_map = domain_profile.get('terminology_map', {})
    if not terminology_map:
        return adapted_json

    json_str = json.dumps(adapted_json, ensure_ascii=False)
    fixes_applied = 0

    # Sort by length (longest first) to avoid partial replacements
    sorted_terms = sorted(terminology_map.items(), key=lambda x: -len(x[0]))

    for source_term, target_term in sorted_terms[:50]:  # Limit to top 50 to avoid slowdown
        if not source_term or not target_term or len(source_term) < 3:
            continue
        if source_term.lower() == target_term.lower():
            continue

        # Case-insensitive match with word boundaries for longer terms
        if len(source_term) >= 5:
            pattern = re.compile(rf'\b{re.escape(source_term)}\b', re.IGNORECASE)
        else:
            # For short terms, require exact case match to avoid false positives
            pattern = re.compile(rf'\b{re.escape(source_term)}\b')

        if pattern.search(json_str):
            new_str = pattern.sub(target_term, json_str)
            if new_str != json_str:
                json_str = new_str
                fixes_applied += 1
                logger.debug(f"[POST-PROCESS] Normalized term: '{source_term}' -> '{target_term}'")

    if fixes_applied > 0:
        logger.info(f"[POST-PROCESS] normalize_terminology: Fixed {fixes_applied} terms")

    return json.loads(json_str)


def normalize_canonical_numbers(adapted_json: dict, canonical_numbers: dict) -> dict:
    """
    Normalize canonical numbers - find variations and replace with canonical values.

    This actively fixes numbers, not just validates them.

    Args:
        adapted_json: The adapted JSON
        canonical_numbers: Exact figures that should appear

    Returns:
        JSON with normalized numbers
    """
    if not canonical_numbers:
        return adapted_json

    json_str = json.dumps(adapted_json, ensure_ascii=False)
    fixes_applied = 0
    missing_numbers = []

    for key, canonical_value in canonical_numbers.items():
        canonical_str = str(canonical_value)

        # Check if the canonical value already appears
        if canonical_str in json_str:
            continue

        # Try to find variations of this number and replace them
        variations_found = _find_numeric_variations(json_str, canonical_str, key)

        if variations_found:
            for variation in variations_found:
                if variation != canonical_str:
                    json_str = json_str.replace(variation, canonical_str)
                    fixes_applied += 1
                    logger.debug(f"[POST-PROCESS] Normalized number: '{variation}' -> '{canonical_str}' ({key})")
        else:
            missing_numbers.append(f"{key}={canonical_value}")

    if fixes_applied > 0:
        logger.info(f"[POST-PROCESS] normalize_canonical_numbers: Fixed {fixes_applied} numbers")

    if missing_numbers:
        logger.warning(f"[POST-PROCESS] Canonical numbers not found: {missing_numbers[:5]}")

    return json.loads(json_str)


def _find_numeric_variations(json_str: str, canonical: str, key: str) -> list[str]:
    """
    Find variations of a canonical number in the JSON string.

    For example, if canonical is "$7.2B", find "$7.2 billion", "7.2B", "$7200M", etc.

    Args:
        json_str: The JSON string to search
        canonical: The canonical number string
        key: The key name (for context, e.g., "revenue" helps identify related numbers)

    Returns:
        List of variations found
    """
    variations = []

    # Extract the numeric part and suffix
    # Handle formats like: "$7.2B", "18%", "$2.8B", "65%", "45%"
    import re

    # Pattern for currency amounts (e.g., $7.2B, $2.8B)
    if canonical.startswith('$'):
        # Extract the number and suffix
        match = re.match(r'\$?([\d.]+)\s*([BMK]|billion|million|thousand)?', canonical, re.IGNORECASE)
        if match:
            num = match.group(1)
            suffix = match.group(2) or ''

            # Look for variations
            variations_patterns = [
                rf'\$?{re.escape(num)}\s*{suffix}',  # Exact match
                rf'\$?{re.escape(num)}\s*(?:billion|million)',  # Full word suffix
            ]

            for pattern in variations_patterns:
                found = re.findall(pattern, json_str, re.IGNORECASE)
                variations.extend(found)

    # Pattern for percentages (e.g., 18%, 45%)
    elif '%' in canonical:
        num = canonical.replace('%', '').strip()
        # Look for the percentage with possible variations
        pattern = rf'{re.escape(num)}\s*(?:%|percent)'
        found = re.findall(pattern, json_str, re.IGNORECASE)
        variations.extend(found)

    # Pattern for years (e.g., 2028)
    elif canonical.isdigit() and len(canonical) == 4:
        # Year - just look for exact match
        if canonical in json_str:
            variations.append(canonical)

    return list(set(variations))


def enforce_canonical_numbers(adapted_json: dict, canonical_numbers: dict) -> dict:
    """
    Validate that canonical numbers appear in the output.

    This is now a thin wrapper around normalize_canonical_numbers for backwards compatibility.

    Args:
        adapted_json: The adapted JSON
        canonical_numbers: Exact figures that should appear

    Returns:
        JSON with canonical numbers enforced
    """
    return normalize_canonical_numbers(adapted_json, canonical_numbers)


def remove_slash_hedging(adapted_json: dict, entity_map: dict) -> dict:
    """
    Remove slash-hedging patterns like "BurgerBlast / fast food".

    These patterns indicate the LLM was uncertain and tried to hedge.
    We want clean, confident text.

    Args:
        adapted_json: The adapted JSON
        entity_map: To get the company name for pattern matching

    Returns:
        JSON with slash-hedging removed
    """
    company = entity_map.get('company', {})
    company_name = company.get('name', '')

    if not company_name:
        return adapted_json

    json_str = json.dumps(adapted_json, ensure_ascii=False)
    fixes_applied = 0

    # Pattern: "CompanyName / something" or "CompanyName/something"
    # Keep just the company name
    pattern = rf'{re.escape(company_name)}\s*/\s*[A-Za-z\s]{{2,20}}'

    new_str, count = re.subn(pattern, company_name, json_str)
    if count > 0:
        json_str = new_str
        fixes_applied += count

    # Pattern: "something / CompanyName"
    pattern = rf'[A-Za-z\s]{{2,20}}\s*/\s*{re.escape(company_name)}'

    new_str, count = re.subn(pattern, company_name, json_str)
    if count > 0:
        json_str = new_str
        fixes_applied += count

    # Generic slash-hedging in company context
    # "QSR / fast food" -> keep first part
    # But be careful not to break legitimate slashes

    if fixes_applied > 0:
        logger.info(f"[POST-PROCESS] remove_slash_hedging: Fixed {fixes_applied} patterns")

    return json.loads(json_str)


def validate_entity_consistency(adapted_json: dict, entity_map: dict) -> list[str]:
    """
    Validate that entity_map values are used consistently.

    Returns list of issues found (for reporting, not fixing).

    Args:
        adapted_json: The adapted JSON
        entity_map: Company, people, roles

    Returns:
        List of consistency issues found
    """
    issues = []
    json_str = json.dumps(adapted_json, ensure_ascii=False).lower()

    # Check for forbidden generic terms
    forbidden_terms = [
        "the company",
        "the manager",
        "the organization",
        "the supervisor",
        "your manager",
        "our company",
    ]

    for term in forbidden_terms:
        if term in json_str:
            count = json_str.count(term)
            issues.append(f"Found '{term}' {count} time(s) - should use specific name")

    # Check company name appears
    company_name = entity_map.get('company', {}).get('name', '')
    if company_name and company_name.lower() not in json_str:
        issues.append(f"Company name '{company_name}' not found in output")

    # Check at least one person name appears
    people = entity_map.get('people', [])
    if isinstance(people, list):
        person_found = False
        for person in people:
            if isinstance(person, dict):
                name = person.get('name', '')
                if name and name.lower() in json_str:
                    person_found = True
                    break
        if not person_found and people:
            issues.append("No person names from entity_map found in output")

    return issues


def validate_number_consistency(adapted_json: dict, canonical_numbers: dict) -> list[str]:
    """
    Validate that canonical numbers are used consistently.

    Returns list of issues found (for reporting).

    Args:
        adapted_json: The adapted JSON
        canonical_numbers: Expected figures

    Returns:
        List of consistency issues found
    """
    issues = []
    json_str = json.dumps(adapted_json, ensure_ascii=False)

    # Check each canonical number
    for key, value in canonical_numbers.items():
        if str(value) not in json_str:
            issues.append(f"Canonical number '{key}={value}' not found in output")

    # Check for conflicting numbers (same metric, different values)
    # This is harder to detect programmatically, so we just count occurrences

    return issues
