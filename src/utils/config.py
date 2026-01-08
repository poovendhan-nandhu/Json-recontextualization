"""Application configuration."""
import os
from dotenv import load_dotenv
from typing import TypedDict, Literal

load_dotenv()


class ShardDefinition(TypedDict):
    """Type definition for shard configuration."""
    name: str
    paths: list[str]
    locked: bool
    is_blocker: bool
    parallel: bool
    aligns_with: list[str]


# ============================================================================
# SHARD DEFINITIONS - Based on sample_main.json structure
# ============================================================================
# These define how the simulation JSON is split into independent chunks
# for parallel processing and scoped validation.

SHARD_DEFINITIONS: dict[str, ShardDefinition] = {
    # -------------------------------------------------------------------------
    # LOCKED SHARDS - Never modify these (only system IDs)
    # -------------------------------------------------------------------------
    "metadata": {
        "name": "Metadata & System IDs",
        "paths": [
            "topicWizardData.id",
            "topicWizardData.workspace",
            "topicWizardData.selectedWorkspaceId",
        ],
        "locked": True,  # System IDs only - never change
        "is_blocker": True,
        "parallel": False,
        "aligns_with": [],
    },
    "scenario_options": {
        "name": "Scenario Options",
        "paths": [
            "topicWizardData.scenarioOptions",
        ],
        "locked": True,  # Original options never change
        "is_blocker": True,
        "parallel": False,
        "aligns_with": [],
    },

    # -------------------------------------------------------------------------
    # UNLOCKED SHARDS - These get recontextualized (IDs preserved within)
    # -------------------------------------------------------------------------
    "lesson_information": {
        "name": "Lesson Information",
        "paths": [
            "topicWizardData.lessonInformation",
        ],
        "locked": False,  # ✅ NOW UNLOCKED - content changes, structure preserved
        "is_blocker": True,
        "parallel": True,
        "aligns_with": ["assessment_criteria", "workplace_scenario"],
    },
    "assessment_criteria": {
        "name": "Assessment Criteria (KLOs)",
        "paths": [
            "topicWizardData.assessmentCriterion",
            "topicWizardData.selectedAssessmentCriterion",
        ],
        "locked": False,  # ✅ NOW UNLOCKED - IDs preserved, content adapts
        "is_blocker": True,
        "parallel": True,
        "aligns_with": ["simulation_flow", "rubrics", "workplace_scenario"],
    },
    "industry_activities": {
        "name": "Industry Aligned Activities",
        "paths": [
            "topicWizardData.industryAlignedActivities",
            "topicWizardData.selectedIndustryAlignedActivities",  # ✅ ADDED - was missing!
            "topicWizardData.chatHistory.industryAlignedActivities",
        ],
        "locked": False,  # ✅ NOW UNLOCKED - content adapts
        "is_blocker": False,
        "parallel": True,
        "aligns_with": ["assessment_criteria", "workplace_scenario"],
    },
    "selected_scenario": {
        "name": "Selected Scenario",
        "paths": [
            "topicWizardData.selectedScenarioOption",
            "topicWizardData.scenarioDescription",
        ],
        "locked": False,
        "is_blocker": True,
        "parallel": True,
        "aligns_with": ["workplace_scenario", "simulation_flow"],
    },
    "workplace_scenario": {
        "name": "Workplace Scenario",
        "paths": [
            "topicWizardData.workplaceScenario",
        ],
        "locked": False,
        "is_blocker": True,
        "parallel": True,
        "aligns_with": ["selected_scenario", "simulation_flow", "emails"],
    },
    "scenario_chat_history": {
        "name": "Scenario Chat History",
        "paths": [
            "topicWizardData.chatHistory.scenarioDescription",
        ],
        "locked": False,  # Content changes, but structure/IDs preserved
        "is_blocker": False,
        "parallel": True,
        "aligns_with": ["workplace_scenario"],
    },
    "simulation_flow": {
        "name": "Simulation Flow (Stages)",
        "paths": [
            "topicWizardData.simulationFlow",
        ],
        "locked": False,  # Content changes, structure preserved
        "is_blocker": True,
        "parallel": True,
        "aligns_with": ["workplace_scenario", "emails", "rubrics", "resources"],
    },
    "emails": {
        "name": "Emails",
        "paths": [
            "topicWizardData.simulationFlow[*].children[*].data.email",
            "topicWizardData.simulationFlow[*].data.taskEmail",
        ],
        "locked": False,
        "is_blocker": False,
        "parallel": True,
        "aligns_with": ["workplace_scenario", "simulation_flow"],
    },
    "rubrics": {
        "name": "Rubrics & Review",
        "paths": [
            "topicWizardData.simulationFlow[*].data.review.rubric",
            "topicWizardData.simulationFlow[*].data.review.tedoAlign",
        ],
        "locked": False,  # Content changes, structure preserved
        "is_blocker": True,
        "parallel": True,
        "aligns_with": ["assessment_criteria", "simulation_flow"],
    },
    "resources": {
        "name": "Resources & Attachments",
        "paths": [
            "topicWizardData.simulationFlow[*].data.resource",
            "topicWizardData.simulationFlow[*].data.resourceOptions",
            "topicWizardData.simulationFlow[*].children[*].data.email.attachments",
        ],
        "locked": False,
        "is_blocker": False,
        "parallel": True,
        "aligns_with": ["simulation_flow", "workplace_scenario"],
    },
    "launch_settings": {
        "name": "Launch Settings",
        "paths": [
            "topicWizardData.launchSettings",
            "topicWizardData.simulationName",
            "topicWizardData.simulationImage",
            "topicWizardData.overview",
        ],
        "locked": False,
        "is_blocker": False,
        "parallel": True,
        "aligns_with": ["workplace_scenario"],
    },
    "videos": {
        "name": "Videos",
        "paths": [
            "topicWizardData.videos",
            "topicWizardData.simulationFlow[*].children[*].data.video",
        ],
        "locked": False,
        "is_blocker": False,
        "parallel": True,
        "aligns_with": ["workplace_scenario"],
    },
}


class Config:
    """Application configuration."""

    # OpenAI Settings
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-5.2-2025-12-11")
    OPENAI_TEMPERATURE: float = float(os.getenv("OPENAI_TEMPERATURE", "0"))
    OPENAI_SEED: int = int(os.getenv("OPENAI_SEED", "42"))
    OPENAI_TIMEOUT: int = 300

    # App Settings
    APP_NAME: str = os.getenv("APP_NAME", "Cartedo Simulation Adaptation API")
    APP_VERSION: str = os.getenv("APP_VERSION", "2.0.0")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Workflow Settings
    MAX_RETRIES: int = 3
    MAX_FIX_ATTEMPTS: int = 3
    CONSISTENCY_THRESHOLD: float = 0.85

    # Compliance Thresholds
    BLOCKER_PASS_RATE_REQUIRED: float = 1.0   # Blockers must pass 100%
    OVERALL_SCORE_REQUIRED: float = 0.98      # Overall must be >= 98%

    # Locked Fields (immutable) - Only system IDs now
    LOCKED_FIELDS: list[str] = [
        "id",  # System ID
        "scenarioOptions",  # Original options preserved
    ]

    # Shard Definitions
    SHARD_DEFINITIONS: dict = SHARD_DEFINITIONS

    # RAG Settings
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    @classmethod
    def get_locked_shard_ids(cls) -> list[str]:
        """Get IDs of all locked shards."""
        return [sid for sid, sdef in SHARD_DEFINITIONS.items() if sdef["locked"]]

    @classmethod
    def get_blocker_shard_ids(cls) -> list[str]:
        """Get IDs of all blocker shards."""
        return [sid for sid, sdef in SHARD_DEFINITIONS.items() if sdef["is_blocker"]]

    @classmethod
    def get_parallel_shard_ids(cls) -> list[str]:
        """Get IDs of shards that can be processed in parallel."""
        return [sid for sid, sdef in SHARD_DEFINITIONS.items() if sdef["parallel"]]


config = Config()


# =============================================================================
# VALIDATION PATTERNS (for 8 validators)
# =============================================================================

# Terms that are WRONG when used in the specified industry
# Key = target industry, Value = terms that should NOT appear
WRONG_INDUSTRY_TERMS: dict[str, list[str]] = {
    # Tech/SaaS terms - wrong in non-tech industries
    "beverage": [
        "CAC", "LTV", "churn", "churn rate", "MRR", "ARR", "subscription",
        "activation", "user acquisition", "freemium", "SaaS", "API",
        "onboarding", "retention rate", "DAU", "MAU", "product-led growth",
    ],
    "hospitality": [
        "CAC", "LTV", "churn", "MRR", "ARR", "subscription", "freemium",
        "SaaS", "API", "onboarding", "DAU", "MAU", "product-led growth",
    ],
    "retail": [
        "CAC", "churn", "MRR", "ARR", "subscription", "freemium", "SaaS",
        "API", "DAU", "MAU", "product-led growth",
    ],
    "manufacturing": [
        "CAC", "LTV", "churn", "MRR", "ARR", "subscription", "freemium",
        "SaaS", "API", "DAU", "MAU", "product-led growth",
    ],
    "healthcare": [
        "CAC", "churn", "MRR", "ARR", "subscription", "freemium", "SaaS",
        "DAU", "MAU", "product-led growth",
    ],
    # Empty for tech - these terms are correct
    "tech": [],
    "tech_saas": [],
    "software": [],
    "fintech": [],
}

# Correct KPIs/terms for each industry
CORRECT_INDUSTRY_TERMS: dict[str, list[str]] = {
    "beverage": [
        "market share", "distribution", "shelf space", "trial rate",
        "repeat purchase", "brand awareness", "retail penetration",
        "volume sales", "price per unit", "trade promotion", "FMCG",
    ],
    "hospitality": [
        "RevPAR", "ADR", "occupancy rate", "guest satisfaction", "NPS",
        "average daily rate", "room nights", "booking conversion",
        "loyalty program", "F&B revenue", "ancillary revenue",
    ],
    "retail": [
        "footfall", "basket size", "conversion rate", "same-store sales",
        "inventory turnover", "gross margin", "shrinkage", "SKU",
        "sell-through rate", "markdown", "private label",
    ],
    "manufacturing": [
        "OEE", "yield", "throughput", "cycle time", "defect rate",
        "capacity utilization", "lead time", "inventory days",
        "cost per unit", "scrap rate", "downtime",
    ],
    "tech_saas": [
        "CAC", "LTV", "churn", "MRR", "ARR", "NRR", "activation rate",
        "DAU", "MAU", "feature adoption", "time to value", "expansion revenue",
    ],
}

# Patterns for Inference Integrity validator (things that shouldn't be in resources)
INFERENCE_INTEGRITY_PATTERNS: list[str] = [
    r"\d+\s*-\s*\d+",           # Ranges like "10-15" or "10 - 15"
    r"\d+\s*to\s*\d+",          # Ranges like "10 to 15"
    r"approximately",           # Vague
    r"about\s+\d+",             # Vague numbers
    r"around\s+\d+",            # Vague numbers
    r"roughly",                 # Vague
    r"estimated",               # Conclusions
    r"projected",               # Conclusions
    r"expected to",             # Conclusions
    r"likely to",               # Conclusions
    r"should result in",        # Conclusions
    r"will lead to",            # Conclusions
    r"TBD",                     # Placeholder
    r"TBC",                     # Placeholder
    r"N/A",                     # Placeholder
    r"\[.*?\]",                 # Placeholders like [INSERT]
    r"XX+",                     # Placeholders like XXX
]

# Word count limits by section type
WORD_COUNT_LIMITS: dict[str, dict[str, int]] = {
    "intro_email": {"min": 50, "max": 300},
    "task_email": {"min": 100, "max": 500},
    "resource": {"min": 200, "max": 3000},
    "rubric_criteria": {"min": 20, "max": 200},
    "workplace_scenario": {"min": 100, "max": 800},
    "klo_description": {"min": 10, "max": 150},
}
