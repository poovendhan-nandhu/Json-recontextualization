"""
ChromaDB Vector Store for RAG.

Provides persistent local storage for:
- Simulation shards (for parallel generation context)
- Industry knowledge (for semantic fixes)
"""
import os
import logging
from typing import Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)

# Lazy imports to avoid startup issues if chromadb not installed
_chromadb = None
_chroma_client = None


def _get_chromadb():
    """Lazy import chromadb."""
    global _chromadb
    if _chromadb is None:
        try:
            import chromadb
            _chromadb = chromadb
        except Exception as e:
            logger.warning(f"ChromaDB not available: {e}. RAG will be disabled.")
            return None
    return _chromadb


class VectorStore:
    """
    ChromaDB-based vector store for simulation context.

    Collections:
    - Per-shard-type collections for RAG-assisted generation:
      - scenarios: Scenario/background content
      - klos: Key Learning Outcomes and assessment criteria
      - resources: Resource documents and data
      - emails: Email templates and content
      - activities: Activities and simulation flow
      - rubrics: Rubric criteria and review content
    - Generic collections:
      - simulations: All indexed simulation shards (legacy)
      - industry_knowledge: Industry-specific KPIs and terminology
      - entity_mappings: Entity name mappings for recontextualization
    """

    # Per-shard-type collections (for similar examples retrieval)
    SHARD_TYPE_COLLECTIONS = {
        "scenarios": "Scenario backgrounds, workplace context, selected scenarios",
        "klos": "Key Learning Outcomes, assessment criteria, learning objectives",
        "resources": "Resource documents, data tables, attachments",
        "emails": "Email templates, intro emails, task emails",
        "activities": "Simulation flow, stages, activities",
        "rubrics": "Rubric criteria, review guidelines, grading",
    }

    # Generic collections
    GENERIC_COLLECTIONS = {
        "simulations": "All indexed simulation shards (legacy/fallback)",
        "industry_knowledge": "Industry-specific KPIs and terminology",
        "entity_mappings": "Entity name mappings for recontextualization",
    }

    # Combined for backwards compatibility
    COLLECTIONS = {**SHARD_TYPE_COLLECTIONS, **GENERIC_COLLECTIONS}

    # Map shard names to collection types
    SHARD_TO_COLLECTION = {
        # Scenario-related shards
        "selected_scenario": "scenarios",
        "workplace_scenario": "scenarios",
        "scenario_chat_history": "scenarios",
        # KLO-related shards
        "assessment_criteria": "klos",
        "lesson_information": "klos",
        # Resource shards
        "resources": "resources",
        # Email shards
        "emails": "emails",
        # Activity/flow shards
        "simulation_flow": "activities",
        "industry_activities": "activities",
        # Rubric shards
        "rubrics": "rubrics",
        # Settings (goes to scenarios as context)
        "launch_settings": "scenarios",
        "videos": "activities",
    }

    def __init__(self, persist_dir: str = "./chroma_db"):
        """
        Initialize ChromaDB client.

        Args:
            persist_dir: Directory for persistent storage
        """
        self.persist_dir = persist_dir
        self._client = None
        self._collections = {}

    @property
    def client(self):
        """Lazy initialization of ChromaDB client."""
        if self._client is None:
            chromadb = _get_chromadb()

            if chromadb is None:
                logger.warning("ChromaDB unavailable - RAG disabled")
                return None

            # Ensure directory exists
            Path(self.persist_dir).mkdir(parents=True, exist_ok=True)

            try:
                # Create persistent client
                self._client = chromadb.PersistentClient(path=self.persist_dir)
                logger.info(f"ChromaDB initialized at {self.persist_dir}")
            except Exception as e:
                logger.warning(f"Failed to initialize ChromaDB: {e}")
                return None

        return self._client

    def get_collection(self, name: str):
        """
        Get or create a collection.

        Args:
            name: Collection name (must be in COLLECTIONS)

        Returns:
            ChromaDB collection or None if unavailable
        """
        if self.client is None:
            return None

        if name not in self.COLLECTIONS:
            raise ValueError(f"Unknown collection: {name}. Valid: {list(self.COLLECTIONS.keys())}")

        if name not in self._collections:
            self._collections[name] = self.client.get_or_create_collection(
                name=name,
                metadata={"description": self.COLLECTIONS[name]}
            )
            logger.debug(f"Collection '{name}' ready")

        return self._collections[name]

    @classmethod
    def get_collection_for_shard(cls, shard_name: str) -> str:
        """
        Get the appropriate collection name for a shard type.

        Args:
            shard_name: Name of the shard (e.g., "workplace_scenario", "assessment_criteria")

        Returns:
            Collection name (e.g., "scenarios", "klos")
        """
        return cls.SHARD_TO_COLLECTION.get(shard_name, "simulations")

    @classmethod
    def get_shard_type_collections(cls) -> list[str]:
        """Get list of per-shard-type collection names."""
        return list(cls.SHARD_TYPE_COLLECTIONS.keys())

    def add_documents(
        self,
        collection_name: str,
        documents: list[str],
        metadatas: list[dict],
        ids: list[str],
        embeddings: Optional[list[list[float]]] = None,
    ) -> int:
        """
        Add documents to a collection.

        Args:
            collection_name: Target collection
            documents: Text content to store
            metadatas: Metadata for each document
            ids: Unique IDs for each document
            embeddings: Pre-computed embeddings (optional)

        Returns:
            Number of documents added
        """
        collection = self.get_collection(collection_name)

        if collection is None:
            logger.warning(f"Cannot add documents - ChromaDB unavailable")
            return 0

        # Add in batches to avoid memory issues
        batch_size = 100
        total_added = 0

        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_metas = metadatas[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]
            batch_embeds = embeddings[i:i + batch_size] if embeddings else None

            if batch_embeds:
                collection.add(
                    documents=batch_docs,
                    metadatas=batch_metas,
                    ids=batch_ids,
                    embeddings=batch_embeds,
                )
            else:
                collection.add(
                    documents=batch_docs,
                    metadatas=batch_metas,
                    ids=batch_ids,
                )

            total_added += len(batch_docs)
            logger.debug(f"Added batch {i//batch_size + 1}, total: {total_added}")

        return total_added

    def query(
        self,
        collection_name: str,
        query_texts: list[str],
        n_results: int = 5,
        where: Optional[dict] = None,
        where_document: Optional[dict] = None,
    ) -> dict:
        """
        Query a collection for similar documents.

        Args:
            collection_name: Collection to query
            query_texts: Text queries
            n_results: Number of results per query
            where: Metadata filter
            where_document: Document content filter

        Returns:
            Query results with documents, metadatas, distances
        """
        collection = self.get_collection(collection_name)

        if collection is None:
            return {"documents": [], "metadatas": [], "distances": []}

        results = collection.query(
            query_texts=query_texts,
            n_results=n_results,
            where=where,
            where_document=where_document,
        )

        return results

    def get_by_id(self, collection_name: str, ids: list[str]) -> dict:
        """Get documents by their IDs."""
        collection = self.get_collection(collection_name)
        if collection is None:
            return {"documents": [], "metadatas": [], "ids": []}
        return collection.get(ids=ids)

    def delete(self, collection_name: str, ids: list[str]) -> None:
        """Delete documents by IDs."""
        collection = self.get_collection(collection_name)
        if collection is None:
            return
        collection.delete(ids=ids)

    def count(self, collection_name: str) -> int:
        """Get document count in collection."""
        collection = self.get_collection(collection_name)
        if collection is None:
            return 0
        return collection.count()

    def clear_collection(self, collection_name: str) -> None:
        """Delete and recreate a collection."""
        if self.client is None:
            return
        try:
            self.client.delete_collection(collection_name)
            logger.info(f"Cleared collection: {collection_name}")
        except Exception:
            pass  # Collection might not exist

        # Remove from cache
        self._collections.pop(collection_name, None)

    def list_collections(self) -> list[dict]:
        """List all collections with counts."""
        collections = []
        for name in self.COLLECTIONS:
            try:
                count = self.count(name)
                collections.append({
                    "name": name,
                    "description": self.COLLECTIONS[name],
                    "count": count
                })
            except Exception:
                collections.append({
                    "name": name,
                    "description": self.COLLECTIONS[name],
                    "count": 0
                })
        return collections


# Singleton instance
_vector_store: Optional[VectorStore] = None


def get_vector_store(persist_dir: str = None) -> VectorStore:
    """
    Get singleton VectorStore instance.

    Args:
        persist_dir: Override default persist directory

    Returns:
        VectorStore instance
    """
    global _vector_store

    if _vector_store is None:
        # Import config here to avoid circular imports
        try:
            from ..utils.config import config
            default_dir = config.CHROMA_PERSIST_DIR
        except ImportError:
            default_dir = "./chroma_db"

        _vector_store = VectorStore(persist_dir or default_dir)

    return _vector_store
