"""
Content Processor - Comprehensive post-processing and entity management.

Features:
1. Entity extraction and mapping
2. Dynamic path resolver
3. Post-processing fixes (trailing periods, company names, consistency)
4. Feedback loop support for regeneration
"""
import re
import json
import logging
from typing import Any, Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# =============================================================================
# 1. ENTITY EXTRACTOR - Extract entities BEFORE adaptation
# =============================================================================

@dataclass
class EntityMap:
    """Container for entity mappings."""
    company_names: Dict[str, str] = field(default_factory=dict)
    person_names: Dict[str, str] = field(default_factory=dict)
    emails: Dict[str, str] = field(default_factory=dict)
    roles: Dict[str, str] = field(default_factory=dict)
    products: Dict[str, str] = field(default_factory=dict)
    locations: Dict[str, str] = field(default_factory=dict)

    def all_mappings(self) -> Dict[str, str]:
        """Get all mappings combined."""
        all_maps = {}
        all_maps.update(self.company_names)
        all_maps.update(self.person_names)
        all_maps.update(self.emails)
        all_maps.update(self.roles)
        all_maps.update(self.products)
        all_maps.update(self.locations)
        return all_maps


class EntityExtractor:
    """
    Extract entities from source JSON BEFORE adaptation.
    This ensures we know exactly what needs to be replaced.
    """

    # Fields that typically contain entity names
    NAME_FIELDS = {'name', 'fullName', 'organizationName', 'companyName', 'senderName'}
    EMAIL_FIELDS = {'email', 'senderEmail', 'recipientEmail'}
    ROLE_FIELDS = {'role', 'title', 'designation', 'jobTitle', 'position'}

    def extract_entities(self, content: Dict[str, Any]) -> EntityMap:
        """
        Extract all entities from content.

        Args:
            content: Source JSON content

        Returns:
            EntityMap with all extracted entities
        """
        entity_map = EntityMap()

        # Extract from known locations
        topic_data = content.get("topicWizardData", {})

        # Company name
        workplace = topic_data.get("workplaceScenario", {})
        background = workplace.get("background", {})
        org_name = background.get("organizationName", "")
        if org_name:
            entity_map.company_names[org_name] = ""  # Will be filled by factsheet

        # Manager info
        lrrm = workplace.get("learnerRoleReportingManager", {})
        manager = lrrm.get("reportingManager", {})
        if manager:
            if manager.get("name"):
                entity_map.person_names[manager["name"]] = ""
            if manager.get("email"):
                entity_map.emails[manager["email"]] = ""
            if manager.get("role"):
                entity_map.roles[manager["role"]] = ""

        # Recursively find all entities
        self._extract_recursive(content, entity_map)

        logger.info(f"Extracted entities: {len(entity_map.company_names)} companies, "
                   f"{len(entity_map.person_names)} people, {len(entity_map.emails)} emails")

        return entity_map

    def _extract_recursive(self, obj: Any, entity_map: EntityMap, path: str = ""):
        """Recursively extract entities from nested structure."""
        if isinstance(obj, dict):
            for key, value in obj.items():
                current_path = f"{path}.{key}" if path else key

                if isinstance(value, str) and value:
                    # Check field type
                    key_lower = key.lower()
                    if key_lower in {f.lower() for f in self.NAME_FIELDS}:
                        if '@' not in value:  # Not an email
                            entity_map.person_names[value] = ""
                    elif key_lower in {f.lower() for f in self.EMAIL_FIELDS}:
                        if '@' in value:
                            entity_map.emails[value] = ""
                    elif key_lower in {f.lower() for f in self.ROLE_FIELDS}:
                        entity_map.roles[value] = ""
                    elif key_lower == 'organizationname' or key_lower == 'companyname':
                        entity_map.company_names[value] = ""
                else:
                    self._extract_recursive(value, entity_map, current_path)

        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                self._extract_recursive(item, entity_map, f"{path}[{i}]")

    def build_mappings_from_factsheet(
        self,
        entity_map: EntityMap,
        factsheet: Dict[str, Any]
    ) -> EntityMap:
        """
        Build target mappings using the global factsheet.

        Args:
            entity_map: Extracted source entities
            factsheet: Global factsheet with target values

        Returns:
            Updated EntityMap with target values
        """
        # Get target values from factsheet
        company = factsheet.get("company", {})
        target_company = company.get("name", "")

        manager = factsheet.get("reporting_manager", {})
        target_manager_name = manager.get("name", "")
        target_manager_email = manager.get("email", "")
        target_manager_role = manager.get("role", "")

        # Map source company to target
        for source_company in entity_map.company_names:
            entity_map.company_names[source_company] = target_company

        # Map source manager to target (first one found)
        for source_name in list(entity_map.person_names.keys())[:1]:
            entity_map.person_names[source_name] = target_manager_name

        for source_email in list(entity_map.emails.keys())[:1]:
            entity_map.emails[source_email] = target_manager_email

        for source_role in list(entity_map.roles.keys())[:1]:
            entity_map.roles[source_role] = target_manager_role

        return entity_map


# =============================================================================
# 2. DYNAMIC PATH RESOLVER - Robust field access
# =============================================================================

class DynamicPathResolver:
    """
    Dynamically resolve paths in JSON structure.
    More robust than hardcoded paths.
    """

    def __init__(self, content: Dict[str, Any]):
        self.content = content
        self._cache = {}

    def find_all(self, field_name: str) -> List[Tuple[str, Any]]:
        """
        Find all occurrences of a field by name.

        Args:
            field_name: Field name to search for

        Returns:
            List of (path, value) tuples
        """
        if field_name in self._cache:
            return self._cache[field_name]

        results = []
        self._find_recursive(self.content, field_name, "", results)
        self._cache[field_name] = results
        return results

    def _find_recursive(
        self,
        obj: Any,
        field_name: str,
        path: str,
        results: List[Tuple[str, Any]]
    ):
        """Recursively search for field."""
        if isinstance(obj, dict):
            for key, value in obj.items():
                current_path = f"{path}.{key}" if path else key
                if key.lower() == field_name.lower():
                    results.append((current_path, value))
                self._find_recursive(value, field_name, current_path, results)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                self._find_recursive(item, field_name, f"{path}[{i}]", results)

    def get_by_path(self, path: str) -> Optional[Any]:
        """
        Get value at a specific path.

        Args:
            path: Dot-notation path (e.g., "topicWizardData.workplaceScenario.background")

        Returns:
            Value at path or None
        """
        parts = self._parse_path(path)
        current = self.content

        for part in parts:
            if isinstance(part, int):
                if isinstance(current, list) and part < len(current):
                    current = current[part]
                else:
                    return None
            elif isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None

        return current

    def set_by_path(self, path: str, value: Any) -> bool:
        """
        Set value at a specific path.

        Args:
            path: Dot-notation path
            value: Value to set

        Returns:
            True if successful
        """
        parts = self._parse_path(path)
        if not parts:
            return False

        current = self.content
        for part in parts[:-1]:
            if isinstance(part, int):
                if isinstance(current, list) and part < len(current):
                    current = current[part]
                else:
                    return False
            elif isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return False

        last_part = parts[-1]
        if isinstance(last_part, int) and isinstance(current, list):
            if last_part < len(current):
                current[last_part] = value
                return True
        elif isinstance(current, dict):
            current[last_part] = value
            return True

        return False

    def _parse_path(self, path: str) -> List:
        """Parse path string into parts."""
        parts = []
        current = ""
        i = 0

        while i < len(path):
            char = path[i]
            if char == '.':
                if current:
                    parts.append(current)
                    current = ""
            elif char == '[':
                if current:
                    parts.append(current)
                    current = ""
                # Find closing bracket
                j = i + 1
                while j < len(path) and path[j] != ']':
                    j += 1
                index_str = path[i+1:j]
                if index_str.isdigit():
                    parts.append(int(index_str))
                i = j
            else:
                current += char
            i += 1

        if current:
            parts.append(current)

        return parts


# =============================================================================
# 3. COMPREHENSIVE POST-PROCESSOR
# =============================================================================

class ContentPostProcessor:
    """
    Comprehensive post-processing for adapted content.
    Fixes all common LLM output issues.
    """

    # Industry-specific wrong terms to replace
    WRONG_TERMS_BY_INDUSTRY = {
        "retail": {
            "churn": "customer attrition",
            "churn rate": "customer retention rate",
            "MRR": "monthly revenue",
            "ARR": "annual revenue",
            "CAC": "customer acquisition cost",
            "SaaS": "retail",
            "subscription": "purchase",
            "activation rate": "conversion rate",
            "user onboarding": "customer onboarding",
        },
        "fashion": {
            "churn": "customer attrition",
            "churn rate": "customer retention rate",
            "MRR": "monthly sales",
            "ARR": "annual sales",
            "SaaS": "fashion retail",
            "subscription": "purchase",
            "activation rate": "conversion rate",
        },
        "apparel": {
            "churn": "customer attrition",
            "churn rate": "customer retention rate",
            "MRR": "monthly sales",
            "ARR": "annual sales",
            "SaaS": "apparel retail",
            "subscription": "purchase",
        },
        "hospitality": {
            "churn": "guest attrition",
            "MRR": "monthly revenue",
            "ARR": "annual revenue",
            "SaaS": "hospitality",
            "subscription": "booking",
            "CAC": "guest acquisition cost",
        },
        "beverage": {
            "churn": "brand switching",
            "MRR": "monthly sales",
            "ARR": "annual sales",
            "SaaS": "beverage industry",
            "subscription": "recurring purchase",
            "CAC": "customer acquisition cost",
            "activation rate": "trial rate",
        },
    }

    def __init__(
        self,
        company_name: str = None,
        manager_info: Dict[str, str] = None,
        industry: str = None,
        wrong_terms: List[str] = None
    ):
        """
        Args:
            company_name: Correct company name to enforce
            manager_info: Dict with name, email, role, gender for manager
            industry: Target industry for wrong term replacement
            wrong_terms: Custom list of wrong terms to flag
        """
        self.company_name = company_name
        self.manager_info = manager_info or {}
        self.industry = (industry or "").lower()
        self.wrong_terms = wrong_terms or []

    def process(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply all post-processing fixes.

        Args:
            content: Adapted content to fix

        Returns:
            Fixed content
        """
        if not isinstance(content, dict):
            return content

        # Apply fixes in order - entity corruption FIRST, then other fixes
        content = self._fix_entity_corruption(content)  # Fix manager name embedded in company name
        content = self._fix_company_names(content)
        content = self._fix_wrong_terms(content)  # Fix wrong industry terms
        content = self._fix_truncated_text(content, "")  # Fix truncated text
        content = self._fix_trailing_punctuation(content)  # Remove trailing periods
        content = self._fix_gender_casing(content)
        content = self._fix_manager_consistency(content)
        content = self._remove_duplicates(content)
        content = self._remove_empty_entries(content)  # Remove empty resource entries

        return content

    def _remove_empty_entries(self, obj: Any) -> Any:
        """Remove empty entries (blank title/content) from resources and other lists."""
        if isinstance(obj, dict):
            result = {}
            for key, value in obj.items():
                if isinstance(value, list) and value:
                    # Filter out empty entries
                    if isinstance(value[0], dict):
                        filtered = []
                        for item in value:
                            # Handle mixed lists - some items might be strings
                            if not isinstance(item, dict):
                                filtered.append(item)
                                continue

                            # Check if entry is essentially empty
                            title = item.get('title', '') or item.get('name', '') or item.get('resourceTitle', '') or ''
                            content = (item.get('markdownText', '') or item.get('content', '') or
                                      item.get('description', '') or item.get('text', '') or
                                      item.get('body', '') or item.get('resourceContent', '') or '')

                            # Keep if has meaningful content
                            if title.strip() or (content.strip() and len(content.strip()) > 10):
                                filtered.append(self._remove_empty_entries(item))
                            else:
                                logger.info(f"Removing empty entry in {key}: title='{title[:30] if title else ''}', content_len={len(content)}")

                        result[key] = filtered if filtered else value  # Keep original if all filtered
                    else:
                        result[key] = [self._remove_empty_entries(item) for item in value]
                else:
                    result[key] = self._remove_empty_entries(value)
            return result
        elif isinstance(obj, list):
            return [self._remove_empty_entries(item) for item in obj]
        return obj

    def _fix_wrong_terms(self, obj: Any) -> Any:
        """
        Replace wrong industry terms with appropriate equivalents.

        Priority:
        1. Use wrong_terms from factsheet (DYNAMIC - preferred)
        2. Fall back to hardcoded industry map only if factsheet empty
        """
        # PRIORITY 1: Use dynamic wrong_terms from factsheet
        if self.wrong_terms:
            return self._remove_wrong_terms_dynamic(obj, self.wrong_terms)

        # PRIORITY 2: Fallback to hardcoded map (only if no factsheet terms)
        replacements = {}
        for ind_key in self.WRONG_TERMS_BY_INDUSTRY:
            if ind_key in self.industry:
                replacements.update(self.WRONG_TERMS_BY_INDUSTRY[ind_key])
                break

        if not replacements:
            return obj

        if isinstance(obj, str):
            result = obj
            for wrong, correct in replacements.items():
                # Case-insensitive replacement
                pattern = rf'\b{re.escape(wrong)}\b'
                result = re.sub(pattern, correct, result, flags=re.IGNORECASE)
            return result
        elif isinstance(obj, dict):
            return {k: self._fix_wrong_terms(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._fix_wrong_terms(item) for item in obj]
        return obj

    def _remove_wrong_terms_dynamic(self, obj: Any, wrong_terms: List[str]) -> Any:
        """
        Remove/flag wrong terms using DYNAMIC list from factsheet.
        This is the preferred approach - no hardcoding!
        """
        if isinstance(obj, str):
            result = obj
            for term in wrong_terms:
                if not term or not isinstance(term, str):
                    continue
                # Remove the wrong term or replace with placeholder
                pattern = rf'\b{re.escape(term)}\b'
                # Check if term exists before replacing
                if re.search(pattern, result, flags=re.IGNORECASE):
                    # Log for debugging
                    logger.debug(f"Removing wrong term: {term}")
                    # Replace with generic equivalent based on context
                    result = re.sub(pattern, self._get_generic_replacement(term), result, flags=re.IGNORECASE)
            return result
        elif isinstance(obj, dict):
            return {k: self._remove_wrong_terms_dynamic(v, wrong_terms) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._remove_wrong_terms_dynamic(item, wrong_terms) for item in obj]
        return obj

    def _get_generic_replacement(self, term: str) -> str:
        """Get a generic replacement for a wrong term."""
        term_lower = term.lower()

        # Common SaaS/tech terms -> generic business terms
        replacements = {
            "churn": "customer attrition",
            "churn rate": "retention rate",
            "mrr": "monthly revenue",
            "arr": "annual revenue",
            "cac": "acquisition cost",
            "ltv": "lifetime value",
            "saas": "business",
            "subscription": "purchase",
            "activation": "conversion",
            "onboarding": "orientation",
        }

        # Return the replacement if found, otherwise return original term (never a placeholder!)
        return replacements.get(term_lower, term)

    def _fix_company_names(self, obj: Any) -> Any:
        """Fix company name typos and variations."""
        if not self.company_name:
            return obj

        if isinstance(obj, str):
            # Fix common typos - missing 's', extra spaces, etc.
            base_name = self.company_name.rstrip('s')  # "Verde Threads" -> "Verde Thread"

            # Pattern: "Verde Thread" (missing s) -> "Verde Threads"
            pattern = rf'\b{re.escape(base_name)}\b(?!s)'
            obj = re.sub(pattern, self.company_name, obj)

            # Also fix with case variations
            obj = re.sub(
                rf'\b{re.escape(base_name.lower())}\b(?!s)',
                self.company_name,
                obj,
                flags=re.IGNORECASE
            )

            return obj
        elif isinstance(obj, dict):
            return {k: self._fix_company_names(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._fix_company_names(item) for item in obj]
        return obj

    def _fix_trailing_punctuation(self, obj: Any) -> Any:
        """Remove trailing punctuation from names, emails, roles, URLs."""
        if isinstance(obj, dict):
            result = {}
            for key, value in obj.items():
                key_lower = key.lower()
                # Fields that should NOT have trailing periods
                if key_lower in ('name', 'fullname', 'email', 'senderemail', 'role',
                                'title', 'designation', 'jobtitle', 'avatarurl',
                                'imageurl', 'url', 'organizationname'):
                    if isinstance(value, str) and value.endswith('.'):
                        result[key] = value.rstrip('.')
                    elif isinstance(value, str):
                        result[key] = value
                    else:
                        # Still recurse for non-string values (dicts, lists)
                        result[key] = self._fix_trailing_punctuation(value)
                else:
                    result[key] = self._fix_trailing_punctuation(value)
            return result
        elif isinstance(obj, list):
            return [self._fix_trailing_punctuation(item) for item in obj]
        return obj

    def _fix_gender_casing(self, obj: Any) -> Any:
        """Normalize gender casing: 'female' -> 'Female'."""
        if isinstance(obj, dict):
            result = {}
            for key, value in obj.items():
                if key.lower() == 'gender' and isinstance(value, str):
                    # Capitalize first letter
                    if value.lower() in ('male', 'female', 'other'):
                        result[key] = value.capitalize()
                    else:
                        result[key] = value
                else:
                    result[key] = self._fix_gender_casing(value)
            return result
        elif isinstance(obj, list):
            return [self._fix_gender_casing(item) for item in obj]
        return obj

    def _fix_manager_consistency(self, obj: Any) -> Any:
        """Ensure manager info is consistent across all occurrences."""
        if not self.manager_info:
            return obj

        manager_name = self.manager_info.get('name', '')

        if isinstance(obj, dict):
            result = {}
            for key, value in obj.items():
                key_lower = key.lower()

                # Check if this is a manager-related dict
                if key_lower in ('reportingmanager', 'manager', 'sender'):
                    if isinstance(value, dict):
                        # Apply manager info
                        value = dict(value)  # Copy
                        if 'name' in value and self.manager_info.get('name'):
                            value['name'] = self.manager_info['name']
                        if 'email' in value and self.manager_info.get('email'):
                            value['email'] = self.manager_info['email']
                        if 'role' in value and self.manager_info.get('role'):
                            value['role'] = self.manager_info['role']
                        if 'gender' in value and self.manager_info.get('gender'):
                            value['gender'] = self.manager_info['gender'].capitalize()

                # Fix email body signatures that have corrupted manager names
                if key_lower in ('body', 'content', 'text', 'message') and isinstance(value, str) and manager_name:
                    value = self._fix_email_signature(value, manager_name)

                result[key] = self._fix_manager_consistency(value)
            return result
        elif isinstance(obj, list):
            return [self._fix_manager_consistency(item) for item in obj]
        return obj

    def _fix_email_signature(self, body: str, manager_name: str) -> str:
        """Fix corrupted email signatures to use the correct manager name."""
        import re

        if not manager_name or not body:
            return body

        # Pattern to match common closing phrases followed by a name
        # This catches things like "Regards,\nPractices Guide" or "Best,\nfor the"
        closing_patterns = [
            r'(Regards,?\s*\n\s*)([A-Za-z]+ [A-Za-z]+)',
            r'(Best,?\s*\n\s*)([A-Za-z]+ [A-Za-z]+)',
            r'(Sincerely,?\s*\n\s*)([A-Za-z]+ [A-Za-z]+)',
            r'(Thanks,?\s*\n\s*)([A-Za-z]+ [A-Za-z]+)',
            r'(Cheers,?\s*\n\s*)([A-Za-z]+ [A-Za-z]+)',
            r'(Warm regards,?\s*\n\s*)([A-Za-z]+ [A-Za-z]+)',
            r'(Kind regards,?\s*\n\s*)([A-Za-z]+ [A-Za-z]+)',
            r'(All the best,?\s*\n\s*)([A-Za-z]+ [A-Za-z]+)',
        ]

        for pattern in closing_patterns:
            match = re.search(pattern, body, re.IGNORECASE)
            if match:
                found_name = match.group(2)
                # Check if the found name is NOT the expected manager name
                if manager_name.lower() not in found_name.lower():
                    # Check if found name looks like garbage (not a real name)
                    # Real names usually have first letter caps: "Maya Sharma"
                    # Garbage looks like: "practices and", "for the", "Practices Guide"
                    words = found_name.split()
                    if len(words) >= 2:
                        # If the "name" contains common words that aren't names, replace it
                        garbage_indicators = ['and', 'the', 'for', 'with', 'from', 'guide', 'practices', 'approaches']
                        if any(w.lower() in garbage_indicators for w in words):
                            # Replace with correct manager name
                            body = body.replace(found_name, manager_name)
                            logger.info(f"Fixed corrupted email signature: '{found_name}' -> '{manager_name}'")

        return body

    def _fix_entity_corruption(self, obj: Any) -> Any:
        """
        Fix corrupted entity names where one entity is incorrectly embedded in another.
        E.g., "EcoChic TMaya Sharmaeads" -> "EcoChic Threads"

        Uses factsheet values dynamically - no hardcoding.
        """
        if not self.company_name or not self.manager_info:
            return obj

        manager_name = self.manager_info.get('name', '')
        if not manager_name:
            return obj

        if isinstance(obj, str):
            # Check if manager name appears INSIDE what should be company name
            # Pattern: company_prefix + manager_name + company_suffix
            if manager_name in obj and self.company_name not in obj:
                # Try to detect corrupted company name
                # Look for pattern where manager name is embedded incorrectly
                import re
                # Escape special regex chars in manager name
                escaped_manager = re.escape(manager_name)
                # Pattern: word chars + manager name + word chars (not at word boundary)
                pattern = rf'(\w+){escaped_manager}(\w+)'
                match = re.search(pattern, obj)
                if match:
                    # This looks like corruption - replace with company name
                    corrupted = match.group(0)
                    obj = obj.replace(corrupted, self.company_name)
            return obj
        elif isinstance(obj, dict):
            return {k: self._fix_entity_corruption(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._fix_entity_corruption(item) for item in obj]
        return obj

    def _remove_duplicates(self, obj: Any) -> Any:
        """Remove duplicate activities and questions."""
        if isinstance(obj, dict):
            result = {}
            for key, value in obj.items():
                if isinstance(value, list) and value:
                    # Check if this is a list of objects that might have duplicates
                    if isinstance(value[0], dict):
                        # Check multiple possible name fields for activities/questions/rubrics
                        name_fields = ['name', 'title', 'activityName', 'questionText', 'text', 'description',
                                      'question', 'reviewQuestion', 'keyLearningOutcome']
                        name_field = None
                        for field in name_fields:
                            if field in value[0]:
                                name_field = field
                                break

                        if name_field:
                            seen_names = set()
                            unique_items = []
                            for item in value:
                                # Handle mixed lists - some items might be strings
                                if not isinstance(item, dict):
                                    unique_items.append(item)
                                    continue

                                name = item.get(name_field, '').lower().strip()
                                # Also check description for near-duplicates
                                desc = item.get('description', '')[:50].lower().strip() if item.get('description') else ''
                                identifier = f"{name}|{desc}" if desc else name

                                if identifier and identifier not in seen_names:
                                    seen_names.add(identifier)
                                    unique_items.append(item)
                                elif not identifier:
                                    unique_items.append(item)
                                else:
                                    logger.debug(f"Removing duplicate: {name}")
                            result[key] = [self._remove_duplicates(item) for item in unique_items]
                        else:
                            result[key] = [self._remove_duplicates(item) for item in value]
                    else:
                        result[key] = [self._remove_duplicates(item) for item in value]
                else:
                    result[key] = self._remove_duplicates(value)
            return result
        elif isinstance(obj, list):
            return [self._remove_duplicates(item) for item in obj]
        return obj

    def _fix_truncated_text(self, obj: Any, parent_key: str = "") -> Any:
        """Fix text that was truncated mid-sentence. Skip name/email/role/URL fields."""
        if isinstance(obj, str):
            # Skip fields that shouldn't have periods added
            parent_lower = parent_key.lower()
            if parent_lower in ('name', 'fullname', 'email', 'senderemail', 'role',
                               'title', 'designation', 'jobtitle', 'avatarurl',
                               'imageurl', 'url', 'organizationname', 'gender',
                               'videourl', 'fileurl', 'attachmenturl', 'src', 'href'):
                return obj  # Don't modify these fields

            # CRITICAL: Skip URLs entirely - they should never get periods added
            if obj.startswith('http://') or obj.startswith('https://') or obj.startswith('//'):
                return obj  # Don't touch URLs

            if len(obj) < 10:
                return obj

            # Check if text ends abruptly - only for content/description fields
            if obj[-1] not in '.!?"\')':
                # Common truncation patterns
                if obj.endswith(' and') or obj.endswith(' or'):
                    obj = obj.rsplit(' ', 1)[0] + '.'
                elif obj.endswith(' the') or obj.endswith(' a') or obj.endswith(' an'):
                    obj = obj.rsplit(' ', 1)[0] + '.'
                elif obj[-1].isalnum() and len(obj) > 50:  # Only add period to longer text
                    obj += '.'

            return obj
        elif isinstance(obj, dict):
            return {k: self._fix_truncated_text(v, k) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._fix_truncated_text(item, parent_key) for item in obj]
        return obj


# =============================================================================
# 4. FEEDBACK LOOP - Regeneration support
# =============================================================================

@dataclass
class AlignmentFeedback:
    """Feedback from alignment checker for regeneration."""
    failed_rules: List[Dict[str, Any]] = field(default_factory=list)
    critical_issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    focus_shards: List[str] = field(default_factory=list)


class FeedbackAnalyzer:
    """
    Analyze alignment failures and generate feedback for regeneration.
    """

    def analyze(self, alignment_report: Dict[str, Any]) -> AlignmentFeedback:
        """
        Analyze alignment report and extract actionable feedback.

        Args:
            alignment_report: Report from alignment checker

        Returns:
            AlignmentFeedback with structured feedback
        """
        feedback = AlignmentFeedback()

        results = alignment_report.get("results", [])

        for result in results:
            if not result.get("passed", True):
                rule_id = result.get("rule_id", "unknown")
                rule_name = result.get("rule_name", "Unknown Rule")
                score = result.get("score", 0)
                issues = result.get("issues", [])

                # Add to failed rules
                feedback.failed_rules.append({
                    "rule_id": rule_id,
                    "rule_name": rule_name,
                    "score": score,
                    "issues": issues
                })

                # Extract critical issues and suggestions
                for issue in issues:
                    if issue.get("severity") == "blocker":
                        feedback.critical_issues.append(issue.get("description", ""))

                    if issue.get("suggestion"):
                        feedback.suggestions.append(issue.get("suggestion"))

                # ⭐ IMPROVED: Map rule_id to shards that need regeneration
                rule_to_shards = {
                    "klo_to_resources": ["resources", "simulation_flow"],
                    "scenario_to_resources": ["resources", "workplace_scenario"],
                    "klo_to_questions": ["simulation_flow"],
                    "role_to_tasks": ["simulation_flow", "workplace_scenario"],
                    "klo_task_alignment": ["simulation_flow"],
                    "scenario_coherence": ["workplace_scenario", "emails"],
                    "company_consistency": ["workplace_scenario", "emails", "resources"],
                    "reporting_manager_consistency": ["workplace_scenario", "emails"],
                }

                if rule_id in rule_to_shards:
                    feedback.focus_shards.extend(rule_to_shards[rule_id])

                # Also check location field for additional context
                for issue in issues:
                    location = issue.get("location", "")
                    if "simulationFlow" in location:
                        feedback.focus_shards.append("simulation_flow")
                    elif "workplaceScenario" in location:
                        feedback.focus_shards.append("workplace_scenario")
                    elif "chatHistory" in location:
                        feedback.focus_shards.append("scenario_chat_history")
                    elif "resource" in location.lower():
                        feedback.focus_shards.append("resources")

        # Deduplicate
        feedback.focus_shards = list(set(feedback.focus_shards))
        logger.info(f"Identified focus shards for regeneration: {feedback.focus_shards}")

        return feedback

    def generate_regeneration_prompt(self, feedback: AlignmentFeedback) -> str:
        """
        Generate additional prompt instructions based on feedback.

        Args:
            feedback: Analyzed feedback

        Returns:
            Additional prompt text for regeneration
        """
        if not feedback.critical_issues:
            return ""

        prompt_parts = [
            "\n## ⚠️ CRITICAL: FIX THESE ISSUES FROM PREVIOUS ATTEMPT:",
            ""
        ]

        for i, issue in enumerate(feedback.critical_issues[:5], 1):
            prompt_parts.append(f"{i}. {issue}")

        if feedback.suggestions:
            prompt_parts.append("\n## SUGGESTIONS:")
            for suggestion in feedback.suggestions[:3]:
                prompt_parts.append(f"- {suggestion}")

        prompt_parts.append("\nENSURE these issues are fixed in this adaptation.")

        return "\n".join(prompt_parts)


# =============================================================================
# 5. CONVENIENCE FUNCTIONS
# =============================================================================

def extract_and_map_entities(
    source_json: Dict[str, Any],
    factsheet: Dict[str, Any]
) -> EntityMap:
    """
    Extract entities from source and map to target using factsheet.

    Args:
        source_json: Source simulation JSON
        factsheet: Global factsheet with target values

    Returns:
        EntityMap with source->target mappings
    """
    extractor = EntityExtractor()
    entity_map = extractor.extract_entities(source_json)
    entity_map = extractor.build_mappings_from_factsheet(entity_map, factsheet)
    return entity_map


def post_process_content(
    content: Dict[str, Any],
    factsheet: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Apply comprehensive post-processing to adapted content.

    Args:
        content: Adapted content
        factsheet: Global factsheet for correct values

    Returns:
        Fixed content
    """
    # Extract correct values from factsheet
    company_info = factsheet.get("company", {})
    company_name = company_info.get("name", "") if isinstance(company_info, dict) else ""
    industry = company_info.get("industry", "") if isinstance(company_info, dict) else ""

    manager = factsheet.get("reporting_manager", {})
    manager_info = {
        "name": manager.get("name", ""),
        "email": manager.get("email", ""),
        "role": manager.get("role", ""),
        "gender": manager.get("gender", "Female")
    } if isinstance(manager, dict) else {}

    # Get wrong terms from factsheet
    industry_context = factsheet.get("industry_context", {})
    wrong_terms = industry_context.get("wrong_terms", []) if isinstance(industry_context, dict) else []

    processor = ContentPostProcessor(
        company_name=company_name,
        manager_info=manager_info,
        industry=industry,
        wrong_terms=wrong_terms
    )

    processed = processor.process(content)

    # Apply content quality fixes (duplicates, truncation)
    from .content_fixer import fix_content_issues
    fixed_content, fixes_applied = fix_content_issues(processed)

    if fixes_applied:
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Content fixer applied {len(fixes_applied)} fixes: {fixes_applied[:5]}")

    return fixed_content


def analyze_and_get_feedback(alignment_report: Dict[str, Any]) -> Tuple[AlignmentFeedback, str]:
    """
    Analyze alignment report and get feedback for regeneration.

    Args:
        alignment_report: Report from alignment checker

    Returns:
        (AlignmentFeedback, regeneration_prompt)
    """
    analyzer = FeedbackAnalyzer()
    feedback = analyzer.analyze(alignment_report)
    prompt = analyzer.generate_regeneration_prompt(feedback)
    return feedback, prompt
