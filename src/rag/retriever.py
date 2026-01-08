"""
Simulation Retriever for RAG.

Indexes simulation shards and retrieves context for:
- Parallel generation (context from related shards)
- Alignment checking (finding related content)
- Semantic fixing (industry-appropriate replacements)
"""
import json
import logging
from typing import Any, Optional
from dataclasses import dataclass

from .vector_store import get_vector_store, VectorStore
from .embeddings import embed_text, embed_texts

logger = logging.getLogger(__name__)


@dataclass
class RetrievedContext:
    """Context retrieved from vector store."""
    document: str
    metadata: dict
    score: float  # Lower is more similar (distance)
    shard_id: str
    path: str


class SimulationRetriever:
    """
    Indexes and retrieves simulation context.

    Workflow:
    1. Index base simulation shards into ChromaDB
    2. During generation, retrieve relevant context
    3. During validation, find similar content for comparison
    """

    def __init__(self, vector_store: VectorStore = None):
        """
        Initialize retriever.

        Args:
            vector_store: VectorStore instance (uses singleton if None)
        """
        self.store = vector_store or get_vector_store()

    def index_simulation(
        self,
        simulation_id: str,
        shards: list,  # List of Shard objects from sharder
        clear_existing: bool = False,
    ) -> dict:
        """
        Index simulation shards into vector store.

        Creates searchable embeddings for each shard's content.

        Args:
            simulation_id: Unique simulation identifier
            shards: List of Shard objects from sharder
            clear_existing: Clear existing docs for this simulation first

        Returns:
            Indexing summary
        """
        if clear_existing:
            # Delete existing docs for this simulation
            try:
                existing = self.store.query(
                    "simulations",
                    query_texts=[""],
                    n_results=1000,
                    where={"simulation_id": simulation_id}
                )
                if existing.get("ids") and existing["ids"][0]:
                    self.store.delete("simulations", existing["ids"][0])
                    logger.info(f"Cleared {len(existing['ids'][0])} existing docs")
            except Exception as e:
                logger.warning(f"Could not clear existing: {e}")

        documents = []
        metadatas = []
        ids = []

        for shard in shards:
            # Create document from shard content
            doc_text = self._shard_to_text(shard)

            if not doc_text.strip():
                continue

            doc_id = f"{simulation_id}_{shard.id}"

            documents.append(doc_text)
            metadatas.append({
                "simulation_id": simulation_id,
                "shard_id": shard.id,
                "shard_name": shard.name,
                "is_locked": shard.lock_state.value == "FULLY_LOCKED",
                "is_blocker": shard.is_blocker,
                "paths": json.dumps(shard.paths),
                "aligns_with": json.dumps(shard.aligns_with),
            })
            ids.append(doc_id)

        if not documents:
            return {"indexed": 0, "simulation_id": simulation_id}

        # Add to vector store
        count = self.store.add_documents(
            collection_name="simulations",
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )

        logger.info(f"Indexed {count} shards for simulation {simulation_id}")

        return {
            "indexed": count,
            "simulation_id": simulation_id,
            "shards": [s.id for s in shards if self._shard_to_text(s).strip()],
        }

    def _shard_to_text(self, shard) -> str:
        """
        Convert shard content to searchable text.

        Args:
            shard: Shard object

        Returns:
            Text representation for embedding
        """
        parts = [f"Shard: {shard.name}"]

        for path, content in shard.content.items():
            if isinstance(content, str):
                parts.append(f"{path}: {content}")
            elif isinstance(content, dict):
                # Flatten dict to key points
                text = self._flatten_dict(content)
                parts.append(f"{path}: {text}")
            elif isinstance(content, list):
                # Handle list of extracted items
                for item in content[:10]:  # Limit items
                    if isinstance(item, dict):
                        if "value" in item:
                            parts.append(f"{item.get('path', path)}: {self._flatten_dict(item['value'])}")
                        else:
                            parts.append(self._flatten_dict(item))
                    else:
                        parts.append(str(item))

        return "\n".join(parts)[:10000]  # Limit total length

    def _flatten_dict(self, d, max_depth: int = 3) -> str:
        """Flatten dict/list to key points for embedding."""
        if max_depth <= 0:
            return str(d)[:200]

        # Handle list
        if isinstance(d, list):
            items = []
            for item in d[:10]:  # Limit items
                if isinstance(item, dict):
                    items.append(self._flatten_dict(item, max_depth - 1))
                else:
                    items.append(str(item)[:100])
            return f"[{', '.join(items)}]"

        # Handle non-dict
        if not isinstance(d, dict):
            return str(d)[:200]

        # Handle dict
        parts = []
        for key, value in list(d.items())[:20]:  # Limit keys
            if isinstance(value, str):
                parts.append(f"{key}: {value[:200]}")
            elif isinstance(value, dict):
                nested = self._flatten_dict(value, max_depth - 1)
                parts.append(f"{key}: {{{nested}}}")
            elif isinstance(value, list):
                if len(value) > 0 and isinstance(value[0], dict):
                    nested = self._flatten_dict(value[0], max_depth - 1)
                    parts.append(f"{key}: [{len(value)} items, first: {{{nested}}}]")
                else:
                    parts.append(f"{key}: [{len(value)} items]")
            else:
                parts.append(f"{key}: {value}")

        return ", ".join(parts)

    def retrieve_context(
        self,
        query: str,
        simulation_id: str = None,
        shard_ids: list[str] = None,
        n_results: int = 5,
    ) -> list[RetrievedContext]:
        """
        Retrieve relevant context for a query.

        Args:
            query: Search query (e.g., "organization background")
            simulation_id: Filter to specific simulation
            shard_ids: Filter to specific shards
            n_results: Number of results

        Returns:
            List of RetrievedContext objects
        """
        # Build filter
        where = {}
        if simulation_id:
            where["simulation_id"] = simulation_id
        if shard_ids:
            where["shard_id"] = {"$in": shard_ids}

        results = self.store.query(
            collection_name="simulations",
            query_texts=[query],
            n_results=n_results,
            where=where if where else None,
        )

        contexts = []
        if results.get("documents") and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                metadata = results["metadatas"][0][i] if results.get("metadatas") else {}
                distance = results["distances"][0][i] if results.get("distances") else 0.0

                contexts.append(RetrievedContext(
                    document=doc,
                    metadata=metadata,
                    score=distance,
                    shard_id=metadata.get("shard_id", "unknown"),
                    path=metadata.get("paths", "[]"),
                ))

        return contexts

    def retrieve_aligned_context(
        self,
        shard_id: str,
        simulation_id: str,
        n_results: int = 3,
    ) -> list[RetrievedContext]:
        """
        Retrieve context from shards that align with given shard.

        Used during generation to ensure consistency.

        Args:
            shard_id: Source shard ID
            simulation_id: Simulation to search
            n_results: Results per aligned shard

        Returns:
            Context from aligned shards
        """
        # Get the shard's alignment info
        shard_doc = self.store.get_by_id("simulations", [f"{simulation_id}_{shard_id}"])

        if not shard_doc.get("metadatas"):
            return []

        aligns_with = json.loads(shard_doc["metadatas"][0].get("aligns_with", "[]"))

        if not aligns_with:
            return []

        # Retrieve from aligned shards
        all_contexts = []
        for aligned_id in aligns_with:
            contexts = self.retrieve_context(
                query=f"content from {aligned_id}",
                simulation_id=simulation_id,
                shard_ids=[aligned_id],
                n_results=n_results,
            )
            all_contexts.extend(contexts)

        return all_contexts

    def find_similar_chunks(
        self,
        text: str,
        n_results: int = 5,
        exclude_simulation: str = None,
    ) -> list[RetrievedContext]:
        """
        Find similar chunks across all indexed simulations.

        Useful for finding patterns and examples.

        Args:
            text: Text to find similar chunks for
            n_results: Number of results
            exclude_simulation: Simulation ID to exclude

        Returns:
            Similar chunks from other simulations
        """
        results = self.store.query(
            collection_name="simulations",
            query_texts=[text],
            n_results=n_results * 2,  # Get more to filter
        )

        contexts = []
        if results.get("documents") and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                metadata = results["metadatas"][0][i] if results.get("metadatas") else {}

                # Skip excluded simulation
                if exclude_simulation and metadata.get("simulation_id") == exclude_simulation:
                    continue

                distance = results["distances"][0][i] if results.get("distances") else 0.0

                contexts.append(RetrievedContext(
                    document=doc,
                    metadata=metadata,
                    score=distance,
                    shard_id=metadata.get("shard_id", "unknown"),
                    path=metadata.get("paths", "[]"),
                ))

                if len(contexts) >= n_results:
                    break

        return contexts


# Convenience functions
def index_simulation(simulation_id: str, shards: list, clear_existing: bool = False) -> dict:
    """Index simulation shards. See SimulationRetriever.index_simulation."""
    retriever = SimulationRetriever()
    return retriever.index_simulation(simulation_id, shards, clear_existing)


def retrieve_context(query: str, simulation_id: str = None, n_results: int = 5) -> list[RetrievedContext]:
    """Retrieve context. See SimulationRetriever.retrieve_context."""
    retriever = SimulationRetriever()
    return retriever.retrieve_context(query, simulation_id, n_results=n_results)


def retrieve_similar_chunks(text: str, n_results: int = 5) -> list[RetrievedContext]:
    """Find similar chunks. See SimulationRetriever.find_similar_chunks."""
    retriever = SimulationRetriever()
    return retriever.find_similar_chunks(text, n_results)
