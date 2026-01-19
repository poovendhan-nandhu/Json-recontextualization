"""
Unified KLO Validator - Single source of truth for ALL KLO alignment checks.

Consolidates:
- C7 (KLO Preservation) from check_definitions.py
- R4 (KLO-to-Questions) from alignment_checker.py
- R5 (KLO-to-Resources) from alignment_checker.py
- R8 (KLO-Task Alignment) from alignment_checker.py

Design Principle: One validator, one threshold, clear failure messages.
"""
import logging
from typing import Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class KLOCheckResult:
    """Result of a single KLO sub-check."""
    name: str
    score: float
    passed: bool
    issues: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)


@dataclass
class KLOValidationResult:
    """Overall KLO validation result."""
    overall_score: float
    passed: bool
    checks: dict[str, KLOCheckResult]
    summary: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "overall_score": self.overall_score,
            "passed": self.passed,
            "summary": self.summary,
            "checks": {
                name: {
                    "score": check.score,
                    "passed": check.passed,
                    "issues": check.issues,
                }
                for name, check in self.checks.items()
            }
        }


class UnifiedKLOValidator:
    """
    Single source of truth for ALL KLO alignment checks.

    Replaces:
    - C7 (KLO Preservation) from check_definitions.py
    - R4 (KLO-to-Questions) from alignment_checker.py
    - R5 (KLO-to-Resources) from alignment_checker.py
    - R8 (KLO-Task Alignment) from alignment_checker.py

    Usage:
        validator = UnifiedKLOValidator()
        result = validator.validate(adapted_json, factsheet)
        if result.passed:
            print("KLO alignment passed!")
        else:
            print(f"Failed: {result.summary}")
    """

    THRESHOLD = 0.95  # Single threshold for all KLO checks

    # Weights for different checks (higher = more important)
    WEIGHTS = {
        "preservation": 1.0,
        "questions": 1.5,  # Questions are critical
        "resources": 1.5,  # Resources are critical
        "tasks": 1.0,
    }

    def __init__(self, threshold: float = None):
        """
        Initialize validator.

        Args:
            threshold: Override default 95% threshold
        """
        if threshold is not None:
            self.THRESHOLD = threshold

    def validate(self, adapted_json: dict, factsheet: dict) -> KLOValidationResult:
        """
        Run all KLO alignment checks and return unified result.

        Args:
            adapted_json: The adapted simulation JSON
            factsheet: Global factsheet with KLOs

        Returns:
            KLOValidationResult with overall score and per-check details
        """
        topic_data = adapted_json.get("topicWizardData", {})

        # Extract KLOs - prefer adapted KLOs from topic_data
        klos = self._extract_klos(factsheet, topic_data)
        if not klos:
            logger.warning("[KLO VALIDATOR] No KLOs found in adapted JSON or factsheet")
            return KLOValidationResult(
                overall_score=0.0,
                passed=False,
                checks={},
                summary="No KLOs found in adapted JSON or factsheet"
            )

        # Run all checks
        checks = {
            "preservation": self._check_preservation(topic_data, klos),
            "questions": self._check_questions(topic_data, klos),
            "resources": self._check_resources(topic_data, klos),
            "tasks": self._check_tasks(topic_data, klos),
        }

        # Calculate weighted average score
        total_weight = sum(self.WEIGHTS.values())
        overall_score = sum(
            checks[name].score * self.WEIGHTS[name]
            for name in checks
        ) / total_weight

        passed = overall_score >= self.THRESHOLD

        # Build summary
        failed_checks = [name for name, check in checks.items() if not check.passed]
        if passed:
            summary = f"KLO alignment passed ({overall_score:.1%})"
        else:
            summary = f"KLO alignment failed ({overall_score:.1%}): issues in {', '.join(failed_checks)}"

        result = KLOValidationResult(
            overall_score=overall_score,
            passed=passed,
            checks=checks,
            summary=summary
        )

        logger.info(f"[KLO VALIDATOR] {summary}")
        return result

    def _extract_klos(self, factsheet: dict, topic_data: dict = None) -> list[dict]:
        """
        Extract KLOs - prefer adapted KLOs from topic_data, fallback to factsheet.

        Priority:
        1. topic_data.assessmentCriterion (adapted KLOs)
        2. topic_data.selectedAssessmentCriterion
        3. factsheet.klos (source KLOs - last resort)
        """
        normalized = []

        # Try to get adapted KLOs from assessmentCriterion first
        if topic_data:
            assessment = topic_data.get("assessmentCriterion", [])
            if not assessment:
                assessment = topic_data.get("selectedAssessmentCriterion", [])

            if assessment:
                for i, item in enumerate(assessment):
                    if isinstance(item, dict):
                        # Extract keyLearningOutcome text
                        klo_text = item.get("keyLearningOutcome", "")
                        if not klo_text:
                            # Try to build from criterion
                            criteria = item.get("criterion", [])
                            if criteria and isinstance(criteria, list):
                                klo_text = criteria[0].get("criteria", "") if isinstance(criteria[0], dict) else str(criteria[0])

                        if klo_text:
                            normalized.append({
                                "id": item.get("id", f"klo_{i+1}"),
                                "outcome": klo_text
                            })

                if normalized:
                    logger.info(f"[KLO VALIDATOR] Using {len(normalized)} adapted KLOs from assessmentCriterion")
                    return normalized

        # Fallback to factsheet KLOs (source scenario)
        klos = factsheet.get("klos", [])
        if not isinstance(klos, list):
            return []

        # Normalize KLOs to dict format
        for i, klo in enumerate(klos):
            if isinstance(klo, str):
                normalized.append({"id": f"klo_{i+1}", "outcome": klo})
            elif isinstance(klo, dict):
                normalized.append({
                    "id": klo.get("id", f"klo_{i+1}"),
                    "outcome": klo.get("outcome", klo.get("text", str(klo)))
                })

        if normalized:
            logger.info(f"[KLO VALIDATOR] Using {len(normalized)} source KLOs from factsheet (fallback)")
        return normalized

    def _check_preservation(self, topic_data: dict, klos: list[dict]) -> KLOCheckResult:
        """
        Check if KLOs are preserved in assessment_criterion.

        Replaces: C7 (KLO Preservation)
        """
        assessment = topic_data.get("assessmentCriterion", [])
        if not assessment:
            assessment = topic_data.get("selectedAssessmentCriterion", [])

        if not assessment:
            return KLOCheckResult(
                name="preservation",
                score=0.5,  # Partial pass if no assessment found
                passed=False,
                issues=["No assessment criteria found in adapted JSON"]
            )

        preserved = 0
        issues = []

        for klo in klos:
            klo_text = klo.get("outcome", "")
            # Check if KLO essence is preserved (first 50 chars as signature)
            klo_signature = klo_text.lower()[:50] if klo_text else ""

            found = any(
                klo_signature in str(a).lower()
                for a in assessment
            )
            if found:
                preserved += 1
            else:
                issues.append(f"KLO '{klo.get('id')}' not found in assessment criteria")

        score = preserved / len(klos) if klos else 0
        return KLOCheckResult(
            name="preservation",
            score=score,
            passed=score >= self.THRESHOLD,
            issues=issues,
            details={"preserved": preserved, "total": len(klos)}
        )

    def _check_questions(self, topic_data: dict, klos: list[dict]) -> KLOCheckResult:
        """
        Check if questions align with KLOs.

        Replaces: R4 (KLO-to-Questions)
        """
        questions = self._get_all_questions(topic_data)

        if not questions:
            return KLOCheckResult(
                name="questions",
                score=0.0,
                passed=False,
                issues=["No submission questions found"]
            )

        aligned = 0
        issues = []

        for klo in klos:
            klo_text = klo.get("outcome", "")
            klo_keywords = self._extract_keywords(klo_text)

            # Check if any question covers this KLO using flexible matching
            covered = False
            for q in questions:
                q_text = q.get("question", q.get("text", "")).lower()
                # Use flexible matching with synonyms
                matches = self._count_keyword_matches(klo_keywords, q_text)
                # Need at least 1 keyword match (flexible matching is more robust)
                if matches >= 1:
                    covered = True
                    break

            if covered:
                aligned += 1
            else:
                issues.append(f"KLO '{klo.get('id')}' has no aligned question")

        score = aligned / len(klos) if klos else 0
        return KLOCheckResult(
            name="questions",
            score=score,
            passed=score >= self.THRESHOLD,
            issues=issues,
            details={"aligned": aligned, "total": len(klos), "questions_found": len(questions)}
        )

    def _check_resources(self, topic_data: dict, klos: list[dict]) -> KLOCheckResult:
        """
        Check if resources support KLOs.

        Replaces: R5 (KLO-to-Resources)
        """
        resources = []

        # Check all possible resource locations
        # 1. Top-level resources array
        if topic_data.get("resources"):
            resources.extend(topic_data.get("resources", []))

        # 2. Inside simulationFlow stages
        for stage in topic_data.get("simulationFlow", []):
            # Main resource object
            stage_resource = stage.get("resource", {})
            if isinstance(stage_resource, dict) and stage_resource.get("markdownText"):
                resources.append(stage_resource)

            # Resource options array
            resource_opts = stage.get("resourceOptions", [])
            if isinstance(resource_opts, list):
                resources.extend(resource_opts)

            # Also check stage.data
            stage_data = stage.get("data", {})
            if stage_data.get("resource"):
                resources.append(stage_data.get("resource"))
            if stage_data.get("resourceOptions"):
                resources.extend(stage_data.get("resourceOptions", []))

        if not resources:
            return KLOCheckResult(
                name="resources",
                score=0.0,
                passed=False,
                issues=["No resources found"]
            )

        supported = 0
        issues = []

        for klo in klos:
            klo_text = klo.get("outcome", "")
            klo_keywords = self._extract_keywords(klo_text)

            # Check if any resource supports this KLO using flexible matching
            covered = False
            for r in resources:
                r_text = str(r.get("markdownText", "") + " " + r.get("title", "")).lower()
                matches = self._count_keyword_matches(klo_keywords, r_text)
                if matches >= 1:
                    covered = True
                    break

            if covered:
                supported += 1
            else:
                issues.append(f"KLO '{klo.get('id')}' has no supporting resource")

        score = supported / len(klos) if klos else 0
        return KLOCheckResult(
            name="resources",
            score=score,
            passed=score >= self.THRESHOLD,
            issues=issues,
            details={"supported": supported, "total": len(klos), "resources_found": len(resources)}
        )

    def _check_tasks(self, topic_data: dict, klos: list[dict]) -> KLOCheckResult:
        """
        Check if tasks/activities align with KLOs.

        Replaces: R8 (KLO-Task Alignment)
        """
        activities = []

        # Check all possible activity locations in the JSON
        # 1. Top-level activities (rarely used)
        if topic_data.get("activities"):
            activities.extend(topic_data.get("activities", []))

        # 2. industryAlignedActivities (common location)
        for ia in topic_data.get("industryAlignedActivities", []):
            if isinstance(ia, dict):
                activities.append(ia)

        # 3. selectedIndustryAlignedActivities
        for sia in topic_data.get("selectedIndustryAlignedActivities", []):
            if isinstance(sia, dict):
                activities.append(sia)

        # 4. chatHistory.industryAlignedActivities (nested messages)
        chat_history = topic_data.get("chatHistory", {})
        for ia in chat_history.get("industryAlignedActivities", []):
            if isinstance(ia, dict):
                # Activities are nested in message array
                messages = ia.get("message", [])
                if isinstance(messages, list):
                    for msg in messages:
                        if isinstance(msg, dict) and msg.get("name"):
                            activities.append(msg)

        # 5. simulationFlow stages
        for stage in topic_data.get("simulationFlow", []):
            stage_activities = stage.get("data", {}).get("activities", [])
            if stage_activities:
                activities.extend(stage_activities)

        if not activities:
            return KLOCheckResult(
                name="tasks",
                score=0.5,  # Partial pass - activities might be embedded elsewhere
                passed=False,
                issues=["No activities found (checked activities and simulationFlow)"]
            )

        aligned = 0
        issues = []

        for klo in klos:
            klo_text = klo.get("outcome", "")
            klo_keywords = self._extract_keywords(klo_text)

            covered = False
            for a in activities:
                a_text = str(a.get("description", "") + " " + a.get("name", "")).lower()
                matches = self._count_keyword_matches(klo_keywords, a_text)
                if matches >= 1:  # Activities need less strict matching
                    covered = True
                    break

            if covered:
                aligned += 1
            else:
                issues.append(f"KLO '{klo.get('id')}' has no aligned activity")

        score = aligned / len(klos) if klos else 0
        return KLOCheckResult(
            name="tasks",
            score=score,
            passed=score >= self.THRESHOLD,
            issues=issues,
            details={"aligned": aligned, "total": len(klos), "activities_found": len(activities)}
        )

    def _get_all_questions(self, topic_data: dict) -> list[dict]:
        """Get questions from all locations."""
        questions = []

        # Top-level submissionQuestions
        top_level = topic_data.get("submissionQuestions", [])
        if isinstance(top_level, list):
            questions.extend(top_level)

        # Inside simulationFlow stages
        for stage in topic_data.get("simulationFlow", []):
            # Direct submissionQuestions
            stage_qs = stage.get("submissionQuestions", [])
            if isinstance(stage_qs, list):
                questions.extend(stage_qs)

            # Inside stage.data
            stage_data = stage.get("data", {})
            data_qs = stage_data.get("submissionQuestions", [])
            if isinstance(data_qs, list):
                questions.extend(data_qs)

            # Inside activityData.questions (can be at stage level or stage.data level)
            activity_data = stage.get("activityData", {})
            activity_qs = activity_data.get("questions", [])
            if isinstance(activity_qs, list):
                questions.extend(activity_qs)

            # Also check stage.data.activityData.questions (common structure)
            data_activity = stage_data.get("activityData", {})
            data_activity_qs = data_activity.get("questions", [])
            if isinstance(data_activity_qs, list):
                questions.extend(data_activity_qs)

        # Also check chatHistory for questions
        chat_history = topic_data.get("chatHistory", {})
        for key in ["submissionQuestions", "questions"]:
            chat_qs = chat_history.get(key, [])
            if isinstance(chat_qs, list):
                questions.extend(chat_qs)

        return questions

    def _extract_keywords(self, text: str) -> list[str]:
        """
        Extract meaningful keywords from KLO text.

        Returns lowercase keywords > 4 chars, excluding common words.
        """
        if not text:
            return []

        # Common words to exclude
        stop_words = {
            "about", "above", "after", "again", "against", "being", "below",
            "between", "both", "could", "during", "each", "from", "further",
            "have", "having", "here", "itself", "just", "more", "most", "only",
            "other", "over", "same", "should", "some", "such", "than", "that",
            "their", "them", "then", "there", "these", "they", "this", "those",
            "through", "under", "until", "very", "what", "when", "where",
            "which", "while", "will", "with", "would", "your", "using", "based",
            "demonstrate", "ability", "specific", "learner", "outcome",
        }

        words = text.lower().split()
        keywords = [
            w.strip(".,;:!?()[]{}\"'")
            for w in words
            if len(w) > 4 and w.isalpha() and w.lower() not in stop_words
        ]

        # Return unique keywords, up to 8
        seen = set()
        unique = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique.append(kw)
        return unique[:8]

    def _keyword_matches(self, keyword: str, text: str) -> bool:
        """
        Check if a keyword matches in text using domain-agnostic matching.

        Uses stemming/prefix matching - NO hardcoded synonyms.

        Returns True if:
        1. Exact keyword found in text
        2. Keyword stem (first 4+ chars) matches any word in text
        3. Any word in text starts with keyword's stem
        """
        text_lower = text.lower()
        keyword_lower = keyword.lower().strip(".,;:!?()[]{}\"'")

        # Skip very short keywords
        if len(keyword_lower) < 4:
            return keyword_lower in text_lower

        # 1. Exact/substring match
        if keyword_lower in text_lower:
            return True

        # 2. Stem matching - domain agnostic
        # Use first 4 chars as stem for shorter words, 5 for longer
        stem_len = 4 if len(keyword_lower) < 7 else 5
        stem = keyword_lower[:stem_len]

        # Check if any word in text starts with the same stem
        text_words = text_lower.split()
        for word in text_words:
            word_clean = word.strip(".,;:!?()[]{}\"'")
            if len(word_clean) >= stem_len:
                if word_clean.startswith(stem) or keyword_lower.startswith(word_clean[:stem_len]):
                    return True

        # 3. Check for plural/singular variations
        # "competencies" -> "competency", "skills" -> "skill"
        if keyword_lower.endswith('ies'):
            singular = keyword_lower[:-3] + 'y'
            if singular in text_lower:
                return True
        elif keyword_lower.endswith('es'):
            singular = keyword_lower[:-2]
            if singular in text_lower:
                return True
        elif keyword_lower.endswith('s') and len(keyword_lower) > 4:
            singular = keyword_lower[:-1]
            if singular in text_lower:
                return True

        return False

    def _count_keyword_matches(self, keywords: list[str], text: str) -> int:
        """Count how many keywords match in text using flexible matching."""
        return sum(1 for kw in keywords if self._keyword_matches(kw, text))
