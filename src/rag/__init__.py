"""
RAG (Retrieval-Augmented Generation) Module

Provides vector storage and retrieval for:
1. Base simulation chunks - for parallel generation context
2. Industry knowledge - for semantic fixes and validation
3. Entity mappings - for consistent recontextualization
4. Per-shard-type collections - for similar examples during parallel generation

Uses ChromaDB for local persistent vector storage.

KEY FEATURES:
- Per-shard-type collections (scenarios, klos, resources, emails, activities, rubrics)
- Similar examples retrieval for RAG-assisted generation
- Industry knowledge for semantic validation
"""

from .vector_store import VectorStore, get_vector_store
from .embeddings import get_embeddings, embed_text, embed_texts
from .retriever import (
    SimulationRetriever,
    RetrievedContext,
    SimilarExample,
    # Legacy functions
    index_simulation,
    retrieve_context,
    retrieve_similar_chunks,
    # NEW: Per-shard-type functions
    index_simulation_by_shard_type,
    retrieve_similar_examples,
    retrieve_all_shard_examples,
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
    # Simulation Retriever (legacy)
    "SimulationRetriever",
    "RetrievedContext",
    "index_simulation",
    "retrieve_context",
    "retrieve_similar_chunks",
    # Per-shard-type RAG (NEW)
    "SimilarExample",
    "index_simulation_by_shard_type",
    "retrieve_similar_examples",
    "retrieve_all_shard_examples",
    # Industry Knowledge
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
