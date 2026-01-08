"""
Industry Knowledge for RAG-assisted semantic fixes.

Provides:
1. KPI mappings between industries
2. Industry-specific terminology
3. Realistic value ranges for validation
4. Auto-bootstrap from scenarioOptions

Used by:
- Semantic Fixers (Stage 5) - Replace KPIs with industry-appropriate ones
- Domain Fidelity Validator - Check if KPIs are valid for industry
"""
import json
import logging
import re
from typing import Any, Optional
from dataclasses import dataclass, field

from .vector_store import get_vector_store, VectorStore

logger = logging.getLogger(__name__)


# =============================================================================
# INDUSTRY KNOWLEDGE DATA
# =============================================================================

INDUSTRY_KPIS = {
    "beverage": {
        "kpis": [
            "market share", "distribution", "shelf space", "trial rate",
            "repeat purchase rate", "brand awareness", "retail penetration",
            "volume sales", "price per unit", "trade promotion ROI", "FMCG velocity",
            "gross margin", "net revenue per case", "market penetration",
            "category growth", "brand equity score", "consumer satisfaction"
        ],
        "units": {
            "market share": "%",
            "distribution": "%",
            "shelf space": "facings",
            "trial rate": "%",
            "repeat purchase rate": "%",
            "brand awareness": "%",
            "volume sales": "units",
            "price per unit": "$",
            "gross margin": "%"
        },
        "realistic_ranges": {
            "market share": (1, 35),
            "distribution": (40, 95),
            "trial rate": (5, 30),
            "repeat purchase rate": (20, 60),
            "brand awareness": (10, 90),
            "gross margin": (30, 60)
        }
    },
    "hospitality": {
        "kpis": [
            "occupancy rate", "ADR", "RevPAR", "guest satisfaction score",
            "average length of stay", "booking lead time", "repeat guest rate",
            "room revenue", "F&B revenue", "ancillary revenue", "GOP margin",
            "cost per occupied room", "labor cost percentage", "energy cost per room"
        ],
        "units": {
            "occupancy rate": "%",
            "ADR": "$",
            "RevPAR": "$",
            "guest satisfaction score": "out of 10",
            "average length of stay": "nights",
            "booking lead time": "days"
        },
        "realistic_ranges": {
            "occupancy rate": (60, 90),
            "ADR": (100, 500),
            "RevPAR": (60, 450),
            "guest satisfaction score": (7, 9.5),
            "average length of stay": (1.5, 4),
            "GOP margin": (25, 45)
        }
    },
    "airline": {
        "kpis": [
            "load factor", "yield", "RASK", "CASK", "on-time performance",
            "passenger revenue", "ancillary revenue", "fuel cost per ASM",
            "customer satisfaction", "NPS", "booking conversion rate",
            "average fare", "revenue per passenger", "seat turnover"
        ],
        "units": {
            "load factor": "%",
            "yield": "cents/mile",
            "RASK": "cents",
            "CASK": "cents",
            "on-time performance": "%"
        },
        "realistic_ranges": {
            "load factor": (75, 90),
            "yield": (10, 20),
            "on-time performance": (70, 95),
            "NPS": (20, 70)
        }
    },
    "restaurant": {
        "kpis": [
            "foot traffic", "average check size", "table turnover",
            "food cost percentage", "labor cost percentage", "prime cost",
            "revenue per seat hour", "covers per day", "customer satisfaction",
            "repeat customer rate", "online order percentage", "delivery revenue"
        ],
        "units": {
            "foot traffic": "customers/day",
            "average check size": "$",
            "table turnover": "turns/day",
            "food cost percentage": "%",
            "labor cost percentage": "%"
        },
        "realistic_ranges": {
            "foot traffic": (100, 500),
            "average check size": (15, 75),
            "table turnover": (2, 5),
            "food cost percentage": (25, 35),
            "labor cost percentage": (25, 35)
        }
    },
    "retail": {
        "kpis": [
            "sales per square foot", "inventory turnover", "conversion rate",
            "average transaction value", "gross margin", "sell-through rate",
            "customer acquisition cost", "customer lifetime value", "NPS",
            "return rate", "basket size", "foot traffic", "online vs in-store ratio"
        ],
        "units": {
            "sales per square foot": "$",
            "inventory turnover": "times/year",
            "conversion rate": "%",
            "average transaction value": "$"
        },
        "realistic_ranges": {
            "sales per square foot": (200, 800),
            "inventory turnover": (4, 12),
            "conversion rate": (20, 40),
            "gross margin": (30, 60)
        }
    },
    "technology": {
        "kpis": [
            "MRR", "ARR", "churn rate", "CAC", "LTV", "LTV:CAC ratio",
            "NRR", "DAU", "MAU", "activation rate", "feature adoption rate",
            "support ticket volume", "NPS", "ARPU", "burn rate", "runway"
        ],
        "units": {
            "MRR": "$",
            "ARR": "$",
            "churn rate": "%",
            "CAC": "$",
            "LTV": "$"
        },
        "realistic_ranges": {
            "churn rate": (2, 10),
            "LTV:CAC ratio": (3, 8),
            "NRR": (100, 140),
            "NPS": (30, 70)
        }
    },
    "healthcare": {
        "kpis": [
            "patient satisfaction", "average wait time", "bed occupancy rate",
            "readmission rate", "patient volume", "revenue per patient",
            "cost per patient", "staff-to-patient ratio", "appointment no-show rate",
            "average length of stay", "mortality rate", "infection rate"
        ],
        "units": {
            "patient satisfaction": "out of 10",
            "average wait time": "minutes",
            "bed occupancy rate": "%",
            "readmission rate": "%"
        },
        "realistic_ranges": {
            "patient satisfaction": (7, 9),
            "average wait time": (10, 45),
            "bed occupancy rate": (70, 95),
            "readmission rate": (5, 15)
        }
    },
    "finance": {
        "kpis": [
            "AUM", "net interest margin", "cost-to-income ratio", "ROE",
            "customer acquisition cost", "customer lifetime value", "NPS",
            "loan default rate", "deposit growth", "fee income ratio",
            "digital adoption rate", "transaction volume", "fraud rate"
        ],
        "units": {
            "AUM": "$M",
            "net interest margin": "%",
            "cost-to-income ratio": "%",
            "ROE": "%"
        },
        "realistic_ranges": {
            "net interest margin": (2, 5),
            "cost-to-income ratio": (40, 70),
            "ROE": (8, 20),
            "loan default rate": (1, 5)
        }
    },
    "manufacturing": {
        "kpis": [
            "OEE", "yield rate", "cycle time", "defect rate", "scrap rate",
            "inventory turnover", "on-time delivery", "capacity utilization",
            "cost per unit", "labor productivity", "equipment downtime",
            "safety incident rate", "energy consumption per unit"
        ],
        "units": {
            "OEE": "%",
            "yield rate": "%",
            "cycle time": "minutes",
            "defect rate": "ppm"
        },
        "realistic_ranges": {
            "OEE": (60, 85),
            "yield rate": (90, 99),
            "on-time delivery": (85, 99),
            "capacity utilization": (70, 95)
        }
    },
    "education": {
        "kpis": [
            "enrollment rate", "retention rate", "graduation rate",
            "student satisfaction", "course completion rate", "student-to-faculty ratio",
            "cost per student", "placement rate", "average class size",
            "research output", "grant funding", "alumni engagement"
        ],
        "units": {
            "enrollment rate": "%",
            "retention rate": "%",
            "graduation rate": "%",
            "student satisfaction": "out of 5"
        },
        "realistic_ranges": {
            "retention rate": (70, 95),
            "graduation rate": (60, 95),
            "student satisfaction": (3.5, 4.8),
            "placement rate": (70, 95)
        }
    }
}

# KPI mapping between industries (for semantic fixes)
KPI_MAPPINGS = {
    ("restaurant", "hospitality"): {
        "foot traffic": "occupancy rate",
        "table turnover": "room turnover",
        "average check size": "ADR",
        "covers per day": "rooms sold per day",
        "food cost percentage": "cost per occupied room"
    },
    ("hospitality", "restaurant"): {
        "occupancy rate": "table occupancy rate",
        "ADR": "average check size",
        "RevPAR": "revenue per seat hour",
        "rooms sold per day": "covers per day"
    },
    ("retail", "hospitality"): {
        "foot traffic": "walk-in guests",
        "conversion rate": "booking conversion rate",
        "average transaction value": "average daily rate",
        "sales per square foot": "RevPAR"
    },
    ("hospitality", "airline"): {
        "occupancy rate": "load factor",
        "ADR": "average fare",
        "RevPAR": "RASK",
        "guest satisfaction score": "customer satisfaction"
    },
    ("airline", "hospitality"): {
        "load factor": "occupancy rate",
        "average fare": "ADR",
        "RASK": "RevPAR",
        "yield": "revenue yield"
    },
    ("technology", "hospitality"): {
        "MRR": "monthly room revenue",
        "churn rate": "guest attrition rate",
        "DAU": "daily active bookings",
        "NPS": "guest NPS"
    }
}

# Industry terminology (for domain fidelity checks)
INDUSTRY_TERMINOLOGY = {
    "beverage": [
        "consumer", "brand", "product", "SKU", "retailer", "distributor",
        "shelf", "category", "market", "launch", "formulation", "packaging",
        "nutrition", "ingredient", "flavor", "beverage", "drink", "bottle",
        "can", "serving", "calories", "sugar", "protein", "wellness"
    ],
    "hospitality": [
        "guest", "room", "suite", "amenities", "concierge", "front desk",
        "housekeeping", "check-in", "check-out", "reservation", "booking",
        "stay", "property", "hotel", "resort", "accommodation"
    ],
    "airline": [
        "passenger", "flight", "seat", "cabin", "boarding", "departure",
        "arrival", "gate", "terminal", "crew", "pilot", "route", "hub",
        "connection", "fare class", "frequent flyer"
    ],
    "restaurant": [
        "guest", "table", "reservation", "menu", "chef", "server",
        "kitchen", "dining", "meal", "cuisine", "dish", "course",
        "appetizer", "entree", "dessert"
    ],
    "retail": [
        "customer", "shopper", "store", "merchandise", "inventory",
        "checkout", "display", "aisle", "shelf", "SKU", "vendor",
        "supplier", "promotion", "discount"
    ],
    "technology": [
        "user", "subscriber", "platform", "feature", "product", "app",
        "software", "deployment", "release", "sprint", "backlog",
        "integration", "API", "dashboard"
    ]
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class KPIInfo:
    """Information about a KPI."""
    name: str
    industry: str
    unit: str = ""
    min_value: float = None
    max_value: float = None
    description: str = ""


@dataclass
class IndustryContext:
    """Full context for an industry."""
    name: str
    kpis: list[str] = field(default_factory=list)
    terminology: list[str] = field(default_factory=list)
    units: dict = field(default_factory=dict)
    realistic_ranges: dict = field(default_factory=dict)


# =============================================================================
# INDUSTRY KNOWLEDGE RETRIEVER
# =============================================================================

class IndustryKnowledgeRetriever:
    """
    Retrieves industry-specific knowledge for semantic fixes.

    Uses both in-memory data and ChromaDB for extended knowledge.
    """

    def __init__(self, vector_store: VectorStore = None):
        self.store = vector_store or get_vector_store()
        self._indexed = False

    def get_industry_context(self, industry: str) -> IndustryContext:
        """
        Get full context for an industry.

        Args:
            industry: Industry name (lowercase)

        Returns:
            IndustryContext with KPIs, terminology, etc.
        """
        industry = industry.lower()

        if industry not in INDUSTRY_KPIS:
            # Try to find closest match
            industry = self._find_closest_industry(industry)

        data = INDUSTRY_KPIS.get(industry, {})

        return IndustryContext(
            name=industry,
            kpis=data.get("kpis", []),
            terminology=INDUSTRY_TERMINOLOGY.get(industry, []),
            units=data.get("units", {}),
            realistic_ranges=data.get("realistic_ranges", {})
        )

    def get_kpi_mapping(
        self,
        source_industry: str,
        target_industry: str,
        kpi: str
    ) -> Optional[str]:
        """
        Get the equivalent KPI in target industry.

        Args:
            source_industry: Original industry
            target_industry: Target industry
            kpi: KPI to map

        Returns:
            Mapped KPI name or None if no mapping
        """
        source = source_industry.lower()
        target = target_industry.lower()
        kpi_lower = kpi.lower()

        # Direct mapping exists
        mapping_key = (source, target)
        if mapping_key in KPI_MAPPINGS:
            mappings = KPI_MAPPINGS[mapping_key]
            for src_kpi, tgt_kpi in mappings.items():
                if src_kpi.lower() == kpi_lower:
                    return tgt_kpi

        # Check if KPI exists in target industry (no change needed)
        target_kpis = [k.lower() for k in INDUSTRY_KPIS.get(target, {}).get("kpis", [])]
        if kpi_lower in target_kpis:
            return kpi  # Already valid for target

        # Try to find similar KPI by name matching
        for target_kpi in INDUSTRY_KPIS.get(target, {}).get("kpis", []):
            # Check if words overlap
            kpi_words = set(kpi_lower.split())
            target_words = set(target_kpi.lower().split())
            if kpi_words & target_words:  # Has common words
                return target_kpi

        return None

    def get_all_kpi_mappings(
        self,
        source_industry: str,
        target_industry: str
    ) -> dict[str, str]:
        """
        Get all KPI mappings between two industries.

        Args:
            source_industry: Original industry
            target_industry: Target industry

        Returns:
            Dict of source_kpi -> target_kpi mappings
        """
        source = source_industry.lower()
        target = target_industry.lower()

        mapping_key = (source, target)
        return KPI_MAPPINGS.get(mapping_key, {}).copy()

    def is_valid_kpi(self, kpi: str, industry: str) -> bool:
        """
        Check if a KPI is valid for an industry.

        Args:
            kpi: KPI name
            industry: Industry name

        Returns:
            True if KPI is valid for this industry
        """
        industry = industry.lower()

        if industry not in INDUSTRY_KPIS:
            return True  # Unknown industry, assume valid

        valid_kpis = [k.lower() for k in INDUSTRY_KPIS[industry]["kpis"]]
        return kpi.lower() in valid_kpis

    def get_invalid_kpis(self, kpis: list[str], industry: str) -> list[str]:
        """
        Find KPIs that are invalid for an industry.

        Args:
            kpis: List of KPI names
            industry: Industry name

        Returns:
            List of invalid KPIs
        """
        return [kpi for kpi in kpis if not self.is_valid_kpi(kpi, industry)]

    def is_value_realistic(
        self,
        kpi: str,
        value: float,
        industry: str
    ) -> tuple[bool, str]:
        """
        Check if a KPI value is realistic for the industry.

        Args:
            kpi: KPI name
            value: KPI value
            industry: Industry name

        Returns:
            (is_realistic, reason)
        """
        industry = industry.lower()

        if industry not in INDUSTRY_KPIS:
            return True, "Unknown industry"

        ranges = INDUSTRY_KPIS[industry].get("realistic_ranges", {})

        kpi_lower = kpi.lower()
        for range_kpi, (min_val, max_val) in ranges.items():
            if range_kpi.lower() == kpi_lower:
                if value < min_val:
                    return False, f"Value {value} below typical range ({min_val}-{max_val})"
                if value > max_val:
                    return False, f"Value {value} above typical range ({min_val}-{max_val})"
                return True, f"Within typical range ({min_val}-{max_val})"

        return True, "No range defined for this KPI"

    def get_industry_terminology(self, industry: str) -> list[str]:
        """Get terminology specific to an industry."""
        return INDUSTRY_TERMINOLOGY.get(industry.lower(), [])

    def detect_industry(self, text: str) -> str:
        """
        Detect industry from text content.

        Args:
            text: Text to analyze

        Returns:
            Detected industry name
        """
        text_lower = text.lower()

        # Score each industry by terminology matches
        scores = {}
        for industry, terms in INDUSTRY_TERMINOLOGY.items():
            score = sum(1 for term in terms if term.lower() in text_lower)
            scores[industry] = score

        # Also check for KPI mentions
        for industry, data in INDUSTRY_KPIS.items():
            kpi_score = sum(1 for kpi in data["kpis"] if kpi.lower() in text_lower)
            scores[industry] = scores.get(industry, 0) + kpi_score * 2  # KPIs weighted higher

        if not scores or max(scores.values()) == 0:
            return "unknown"

        return max(scores, key=scores.get)

    def _find_closest_industry(self, industry: str) -> str:
        """Find closest matching industry name."""
        industry_lower = industry.lower()

        # Direct match
        if industry_lower in INDUSTRY_KPIS:
            return industry_lower

        # Partial match
        for known_industry in INDUSTRY_KPIS:
            if known_industry in industry_lower or industry_lower in known_industry:
                return known_industry

        # Common aliases
        aliases = {
            "hotel": "hospitality",
            "hotels": "hospitality",
            "travel": "hospitality",
            "aviation": "airline",
            "airlines": "airline",
            "food": "restaurant",
            "dining": "restaurant",
            "f&b": "restaurant",
            "tech": "technology",
            "software": "technology",
            "saas": "technology",
            "ecommerce": "retail",
            "e-commerce": "retail",
            "store": "retail",
            "medical": "healthcare",
            "hospital": "healthcare",
            "banking": "finance",
            "fintech": "finance",
            "production": "manufacturing",
            "factory": "manufacturing"
        }

        return aliases.get(industry_lower, "unknown")

    # =========================================================================
    # VECTOR STORE INTEGRATION
    # =========================================================================

    def index_industry_knowledge(self) -> dict:
        """
        Index industry knowledge into ChromaDB for extended retrieval.

        Returns:
            Indexing summary
        """
        documents = []
        metadatas = []
        ids = []

        for industry, data in INDUSTRY_KPIS.items():
            # Create document for each industry
            doc_parts = [
                f"Industry: {industry}",
                f"KPIs: {', '.join(data['kpis'])}",
                f"Terminology: {', '.join(INDUSTRY_TERMINOLOGY.get(industry, []))}",
            ]

            # Add units
            if data.get("units"):
                units_text = ", ".join(f"{k}: {v}" for k, v in data["units"].items())
                doc_parts.append(f"Units: {units_text}")

            # Add ranges
            if data.get("realistic_ranges"):
                ranges_text = ", ".join(
                    f"{k}: {v[0]}-{v[1]}"
                    for k, v in data["realistic_ranges"].items()
                )
                doc_parts.append(f"Typical ranges: {ranges_text}")

            doc_text = "\n".join(doc_parts)

            documents.append(doc_text)
            metadatas.append({
                "industry": industry,
                "type": "industry_overview",
                "kpi_count": len(data["kpis"])
            })
            ids.append(f"industry_{industry}")

        # Index KPI mappings
        for (source, target), mappings in KPI_MAPPINGS.items():
            doc_text = f"KPI mappings from {source} to {target}:\n"
            doc_text += "\n".join(f"- {src} -> {tgt}" for src, tgt in mappings.items())

            documents.append(doc_text)
            metadatas.append({
                "source_industry": source,
                "target_industry": target,
                "type": "kpi_mapping"
            })
            ids.append(f"mapping_{source}_{target}")

        # Add to vector store
        count = self.store.add_documents(
            collection_name="industry_knowledge",
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

        self._indexed = True
        logger.info(f"Indexed {count} industry knowledge documents")

        return {
            "indexed": count,
            "industries": list(INDUSTRY_KPIS.keys()),
            "mappings": len(KPI_MAPPINGS)
        }

    def query_industry_knowledge(
        self,
        query: str,
        industry: str = None,
        n_results: int = 5
    ) -> list[dict]:
        """
        Query industry knowledge from vector store.

        Args:
            query: Search query
            industry: Filter to specific industry
            n_results: Number of results

        Returns:
            List of relevant knowledge documents
        """
        where = {"industry": industry} if industry else None

        results = self.store.query(
            collection_name="industry_knowledge",
            query_texts=[query],
            n_results=n_results,
            where=where
        )

        knowledge = []
        if results.get("documents") and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                metadata = results["metadatas"][0][i] if results.get("metadatas") else {}
                knowledge.append({
                    "content": doc,
                    "industry": metadata.get("industry"),
                    "type": metadata.get("type")
                })

        return knowledge


# =============================================================================
# AUTO-BOOTSTRAP FROM SCENARIO OPTIONS
# =============================================================================

def extract_industry_from_scenario(scenario_text: str) -> dict:
    """
    Extract industry context from a scenario description.

    Args:
        scenario_text: Scenario text from scenarioOptions

    Returns:
        Dict with detected industry and key terms
    """
    retriever = IndustryKnowledgeRetriever()

    industry = retriever.detect_industry(scenario_text)

    # Extract potential company name (often first capitalized phrase)
    company_match = re.search(r'([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)', scenario_text)
    company = company_match.group(1) if company_match else None

    # Extract KPIs mentioned
    mentioned_kpis = []
    for ind_data in INDUSTRY_KPIS.values():
        for kpi in ind_data["kpis"]:
            if kpi.lower() in scenario_text.lower():
                mentioned_kpis.append(kpi)

    return {
        "industry": industry,
        "company": company,
        "mentioned_kpis": mentioned_kpis,
        "text_preview": scenario_text[:200]
    }


def bootstrap_from_scenario_options(scenario_options: list[dict]) -> list[dict]:
    """
    Bootstrap industry knowledge from scenarioOptions in simulation JSON.

    Args:
        scenario_options: List of scenario options from topicWizardData.scenarioOptions

    Returns:
        List of extracted industry contexts
    """
    contexts = []

    for i, option in enumerate(scenario_options):
        # Handle different option structures
        if isinstance(option, str):
            text = option
        elif isinstance(option, dict):
            text = option.get("description", "") or option.get("text", "") or str(option)
        else:
            continue

        context = extract_industry_from_scenario(text)
        context["scenario_index"] = i
        contexts.append(context)

    return contexts


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

# Singleton instance
_retriever: Optional[IndustryKnowledgeRetriever] = None


def get_industry_retriever() -> IndustryKnowledgeRetriever:
    """Get singleton IndustryKnowledgeRetriever instance."""
    global _retriever
    if _retriever is None:
        _retriever = IndustryKnowledgeRetriever()
    return _retriever


def get_kpi_mapping(source_industry: str, target_industry: str, kpi: str) -> Optional[str]:
    """Get mapped KPI for target industry."""
    return get_industry_retriever().get_kpi_mapping(source_industry, target_industry, kpi)


def get_industry_context(industry: str) -> IndustryContext:
    """Get full context for an industry."""
    return get_industry_retriever().get_industry_context(industry)


def is_valid_kpi_for_industry(kpi: str, industry: str) -> bool:
    """Check if KPI is valid for industry."""
    return get_industry_retriever().is_valid_kpi(kpi, industry)


def detect_industry(text: str) -> str:
    """Detect industry from text."""
    return get_industry_retriever().detect_industry(text)
