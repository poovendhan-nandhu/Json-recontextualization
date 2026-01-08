"""
RAG (Retrieval-Augmented Generation) Module

Provides vector storage and retrieval for:
1. Base simulation chunks - for parallel generation context
2. Industry knowledge - for semantic fixes and validation
3. Entity mappings - for consistent recontextualization

Uses ChromaDB for local persistent vector storage.
"""

from .vector_store import VectorStore, get_vector_store
from .embeddings import get_embeddings, embed_text, embed_texts
from .retriever import (
    SimulationRetriever,
    index_simulation,
    retrieve_context,
    retrieve_similar_chunks,
)
from .industry_knowledge import (
    IndustryKnowledgeRetriever,
    IndustryContext,
    KPIInfo,
    get_industry_retriever,
    get_kpi_mapping,
    get_industry_context,
    is_valid_kpi_for_industry,
    detect_industry,
    bootstrap_from_scenario_options,
    INDUSTRY_KPIS,
    KPI_MAPPINGS,
    INDUSTRY_TERMINOLOGY,
)

__all__ = [
    # Vector Store
    "VectorStore",
    "get_vector_store",
    # Embeddings
    "get_embeddings",
    "embed_text",
    "embed_texts",
    # Simulation Retriever
    "SimulationRetriever",
    "index_simulation",
    "retrieve_context",
    "retrieve_similar_chunks",
    # Industry Knowledge (NEW)
    "IndustryKnowledgeRetriever",
    "IndustryContext",
    "KPIInfo",
    "get_industry_retriever",
    "get_kpi_mapping",
    "get_industry_context",
    "is_valid_kpi_for_industry",
    "detect_industry",
    "bootstrap_from_scenario_options",
    "INDUSTRY_KPIS",
    "KPI_MAPPINGS",
    "INDUSTRY_TERMINOLOGY",
]
