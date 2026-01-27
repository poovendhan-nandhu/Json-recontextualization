"""
Embedding-based Fact Retriever for RAG.

Uses ChromaDB's built-in embeddings (all-MiniLM-L6-v2) for semantic similarity.
NO hardcoded keywords or category mappings - pure embedding-based retrieval.

Usage:
    retriever = FactRetriever()
    retriever.index_facts(facts_list)  # One time
    relevant = await retriever.get_relevant_facts(scenario, questions)
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Collection name for facts
FACTS_COLLECTION = "industry_facts"


class FactRetriever:
    """
    Embedding-based fact retrieval using ChromaDB.

    No hardcoded mappings - uses semantic similarity to find relevant facts.
    """

    def __init__(self, persist_dir: str = "./chroma_db"):
        self.persist_dir = persist_dir
        self._client = None
        self._collection = None

    @property
    def client(self):
        """Lazy init ChromaDB client."""
        if self._client is None:
            try:
                import chromadb
                Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
                self._client = chromadb.PersistentClient(path=self.persist_dir)
                logger.info(f"[RAG] ChromaDB initialized at {self.persist_dir}")
            except Exception as e:
                logger.error(f"[RAG] ChromaDB init failed: {e}")
                return None
        return self._client

    @property
    def collection(self):
        """Get or create facts collection."""
        if self._collection is None and self.client:
            self._collection = self.client.get_or_create_collection(
                name=FACTS_COLLECTION,
                metadata={"description": "Industry facts for RAG injection"}
            )
        return self._collection

    def index_facts(self, facts: list[str], source: str = "manual") -> int:
        """
        Index facts into vector store.

        Args:
            facts: List of fact strings to index
            source: Source identifier for metadata

        Returns:
            Number of facts indexed
        """
        if not self.collection:
            logger.warning("[RAG] Cannot index - ChromaDB unavailable")
            return 0

        # Prepare documents
        documents = []
        metadatas = []
        ids = []

        for i, fact in enumerate(facts):
            if not fact or not isinstance(fact, str):
                continue
            fact = fact.strip()
            if len(fact) < 10:
                continue

            documents.append(fact)
            metadatas.append({"source": source, "length": len(fact)})
            ids.append(f"{source}_{i}")

        if not documents:
            return 0

        # ChromaDB auto-embeds using all-MiniLM-L6-v2
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

        logger.info(f"[RAG] Indexed {len(documents)} facts from {source}")
        return len(documents)

    def index_facts_from_file(self, file_path: str) -> int:
        """
        Index facts from a text file (one fact per line) or JSON file.

        Args:
            file_path: Path to facts file

        Returns:
            Number of facts indexed
        """
        path = Path(file_path)
        if not path.exists():
            logger.warning(f"[RAG] Facts file not found: {file_path}")
            return 0

        facts = []

        if path.suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Handle both list and dict formats
                if isinstance(data, list):
                    facts = data
                elif isinstance(data, dict):
                    # Flatten all values
                    for v in data.values():
                        if isinstance(v, list):
                            facts.extend(v)
                        elif isinstance(v, str):
                            facts.append(v)
        else:
            # Plain text, one fact per line
            with open(path, "r", encoding="utf-8") as f:
                facts = [line.strip() for line in f if line.strip()]

        return self.index_facts(facts, source=path.stem)

    async def get_relevant_facts(
        self,
        scenario: str,
        questions: list[str] = None,
        klos: list[dict] = None,
        max_facts: int = 12
    ) -> str:
        """
        Get relevant facts based on semantic similarity.

        Args:
            scenario: Target scenario description
            questions: Questions that need data
            klos: KLOs with 'outcome' key
            max_facts: Maximum facts to return

        Returns:
            Formatted string of relevant facts
        """
        if not self.collection:
            return ""

        # Check if we have any facts indexed
        count = self.collection.count()
        if count == 0:
            logger.warning("[RAG] No facts indexed - run index_facts first")
            return ""

        # Build query from scenario + questions + KLOs
        query_parts = [scenario[:500]] if scenario else []

        if questions:
            query_parts.extend([q[:200] for q in questions[:5]])

        if klos:
            query_parts.extend([k.get("outcome", "")[:200] for k in klos[:3]])

        query_text = " ".join(query_parts)

        if not query_text.strip():
            return ""

        # Query by semantic similarity (ChromaDB uses embeddings)
        results = self.collection.query(
            query_texts=[query_text],
            n_results=max_facts
        )

        if not results or not results.get("documents"):
            return ""

        facts = results["documents"][0]  # First query's results

        if not facts:
            return ""

        # Format as bullet points
        formatted = "\n".join([f"- {fact}" for fact in facts])

        logger.info(f"[RAG] Retrieved {len(facts)} relevant facts (from {count} total)")

        return formatted

    def clear(self):
        """Clear all indexed facts."""
        if self.client:
            try:
                self.client.delete_collection(FACTS_COLLECTION)
                self._collection = None
                logger.info("[RAG] Cleared facts collection")
            except Exception:
                pass

    def count(self) -> int:
        """Get number of indexed facts."""
        if not self.collection:
            return 0
        return self.collection.count()


# Singleton instance
_retriever: Optional[FactRetriever] = None


def get_fact_retriever() -> FactRetriever:
    """Get singleton FactRetriever."""
    global _retriever
    if _retriever is None:
        _retriever = FactRetriever()
    return _retriever


async def get_relevant_facts(
    scenario: str,
    questions: list[str] = None,
    klos: list[dict] = None,
    max_facts: int = 12
) -> str:
    """
    Convenience function to get relevant facts.

    Args:
        scenario: Target scenario
        questions: Questions needing data
        klos: KLOs to support
        max_facts: Maximum facts

    Returns:
        Formatted facts string
    """
    return await get_fact_retriever().get_relevant_facts(
        scenario, questions, klos, max_facts
    )


# Quick test
if __name__ == "__main__":
    import asyncio

    async def test():
        retriever = FactRetriever(persist_dir="./test_chroma")

        # Index some test facts
        test_facts = [
            "The US quick-service restaurant market is valued at $350 billion in 2024.",
            "Digital ordering accounts for 40% of QSR orders, up from 28% in 2021.",
            "Value menu items have 15-25% margins, lower than regular items at 30-40%.",
            "$1 menu promotions increase foot traffic by 12-18% in the first two weeks.",
            "Drive-thru represents 70% of QSR revenue with 3-4 minute average service time.",
            "Loyalty program members visit 2.5x more frequently than non-members.",
            "56% of QSR visits are primarily motivated by value/price considerations.",
            "Gen Z and Millennials account for 55% of QSR digital orders.",
        ]

        retriever.clear()
        indexed = retriever.index_facts(test_facts, source="test")
        print(f"Indexed {indexed} facts")

        # Query
        scenario = "BurgerBlitz responding to competitor's $1 value menu"
        questions = [
            "What is the impact on market share?",
            "How does digital ordering affect revenue?"
        ]

        facts = await retriever.get_relevant_facts(scenario, questions)
        print("\nRetrieved facts:")
        print(facts)

        # Cleanup
        retriever.clear()

    asyncio.run(test())
