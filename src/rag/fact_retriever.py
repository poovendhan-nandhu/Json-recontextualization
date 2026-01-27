"""
Embedding-based Fact Retriever for RAG.

Supports multiple backends:
- ChromaDB (local, default)
- Pinecone (cloud, for Heroku/production) - uses Pinecone's built-in embeddings

Usage:
    retriever = get_fact_retriever()  # Auto-selects based on env
    retriever.index_facts(facts_list)
    relevant = await retriever.get_relevant_facts(scenario, questions)
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# Collection/index name for facts
FACTS_COLLECTION = "industry_facts"


class BaseFactRetriever(ABC):
    """Abstract base for fact retrievers."""

    @abstractmethod
    def index_facts(self, facts: list[str], source: str = "manual") -> int:
        pass

    @abstractmethod
    async def get_relevant_facts(self, scenario: str, questions: list[str] = None, klos: list[dict] = None, max_facts: int = 12) -> str:
        pass

    @abstractmethod
    def clear(self):
        pass

    @abstractmethod
    def count(self) -> int:
        pass

    def index_facts_from_file(self, file_path: str) -> int:
        """Index facts from a text file or JSON file."""
        path = Path(file_path)
        if not path.exists():
            logger.warning(f"[RAG] Facts file not found: {file_path}")
            return 0

        facts = []

        if path.suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    facts = data
                elif isinstance(data, dict):
                    for v in data.values():
                        if isinstance(v, list):
                            facts.extend(v)
                        elif isinstance(v, str):
                            facts.append(v)
        else:
            with open(path, "r", encoding="utf-8") as f:
                facts = [line.strip() for line in f if line.strip()]

        return self.index_facts(facts, source=path.stem)


# =============================================================================
# CHROMADB BACKEND (Local)
# =============================================================================

class ChromaFactRetriever(BaseFactRetriever):
    """ChromaDB-based retriever (local storage)."""

    def __init__(self, persist_dir: str = "./chroma_db"):
        self.persist_dir = persist_dir
        self._client = None
        self._collection = None

    @property
    def client(self):
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
        if self._collection is None and self.client:
            self._collection = self.client.get_or_create_collection(
                name=FACTS_COLLECTION,
                metadata={"description": "Industry facts for RAG"}
            )
        return self._collection

    def index_facts(self, facts: list[str], source: str = "manual") -> int:
        if not self.collection:
            return 0

        documents, metadatas, ids = [], [], []
        for i, fact in enumerate(facts):
            if not fact or not isinstance(fact, str) or len(fact.strip()) < 10:
                continue
            documents.append(fact.strip())
            metadatas.append({"source": source})
            ids.append(f"{source}_{i}")

        if not documents:
            return 0

        self.collection.add(documents=documents, metadatas=metadatas, ids=ids)
        logger.info(f"[RAG] ChromaDB indexed {len(documents)} facts")
        return len(documents)

    async def get_relevant_facts(self, scenario: str, questions: list[str] = None, klos: list[dict] = None, max_facts: int = 12) -> str:
        if not self.collection or self.collection.count() == 0:
            return ""

        query_parts = [scenario[:500]] if scenario else []
        if questions:
            query_parts.extend([q[:200] for q in questions[:5]])
        if klos:
            query_parts.extend([k.get("outcome", "")[:200] for k in klos[:3]])

        query_text = " ".join(query_parts)
        if not query_text.strip():
            return ""

        results = self.collection.query(query_texts=[query_text], n_results=max_facts)
        facts = results.get("documents", [[]])[0]

        logger.info(f"[RAG] Retrieved {len(facts)} facts from ChromaDB")
        return "\n".join([f"- {fact}" for fact in facts]) if facts else ""

    def clear(self):
        if self.client:
            try:
                self.client.delete_collection(FACTS_COLLECTION)
                self._collection = None
            except Exception:
                pass

    def count(self) -> int:
        return self.collection.count() if self.collection else 0


# =============================================================================
# PINECONE BACKEND (Cloud - for Heroku/production)
# Uses Pinecone's built-in inference API (llama-text-embed-v2)
# =============================================================================

class PineconeFactRetriever(BaseFactRetriever):
    """Pinecone-based retriever using Pinecone's inference embeddings."""

    def __init__(self):
        self._pc = None
        self._index = None
        self.api_key = os.getenv("PINECONE_API_KEY")
        self.index_name = os.getenv("PINECONE_INDEX", "simulation")
        self.host = os.getenv("PINECONE_HOST")
        self.embed_model = "llama-text-embed-v2"  # Pinecone's hosted model

    @property
    def pc(self):
        """Lazy init Pinecone client."""
        if self._pc is None and self.api_key:
            try:
                from pinecone import Pinecone
                self._pc = Pinecone(api_key=self.api_key)
                logger.info("[RAG] Pinecone client initialized")
            except Exception as e:
                logger.error(f"[RAG] Pinecone client init failed: {e}")
        return self._pc

    @property
    def index(self):
        """Lazy init Pinecone index."""
        if self._index is None and self.pc:
            try:
                if self.host:
                    self._index = self.pc.Index(host=self.host)
                else:
                    self._index = self.pc.Index(self.index_name)
                logger.info(f"[RAG] Pinecone connected to index: {self.index_name}")
            except Exception as e:
                logger.error(f"[RAG] Pinecone index init failed: {e}")
        return self._index

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using Pinecone's inference API."""
        if not self.pc:
            return []
        try:
            # Use Pinecone's inference API
            embeddings = self.pc.inference.embed(
                model=self.embed_model,
                inputs=texts,
                parameters={"input_type": "passage"}
            )
            return [e.values for e in embeddings.data]
        except Exception as e:
            logger.error(f"[RAG] Embedding failed: {e}")
            return []

    def _embed_query(self, text: str) -> list[float]:
        """Generate query embedding."""
        if not self.pc:
            return []
        try:
            embeddings = self.pc.inference.embed(
                model=self.embed_model,
                inputs=[text],
                parameters={"input_type": "query"}
            )
            return embeddings.data[0].values
        except Exception as e:
            logger.error(f"[RAG] Query embedding failed: {e}")
            return []

    def index_facts(self, facts: list[str], source: str = "manual") -> int:
        if not self.index:
            return 0

        # Filter valid facts
        valid_facts = [f.strip() for f in facts if f and isinstance(f, str) and len(f.strip()) >= 10]
        if not valid_facts:
            return 0

        # Generate embeddings using Pinecone inference
        embeddings = self._embed(valid_facts)
        if not embeddings or len(embeddings) != len(valid_facts):
            logger.error(f"[RAG] Embedding count mismatch: {len(embeddings)} vs {len(valid_facts)}")
            return 0

        # Prepare vectors for upsert
        vectors = []
        for i, (fact, embedding) in enumerate(zip(valid_facts, embeddings)):
            vectors.append({
                "id": f"{source}_{i}",
                "values": embedding,
                "metadata": {"text": fact, "source": source}
            })

        # Upsert in batches of 100
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch)

        logger.info(f"[RAG] Pinecone indexed {len(vectors)} facts")
        return len(vectors)

    async def get_relevant_facts(self, scenario: str, questions: list[str] = None, klos: list[dict] = None, max_facts: int = 12) -> str:
        if not self.index:
            return ""

        # Build query
        query_parts = [scenario[:500]] if scenario else []
        if questions:
            query_parts.extend([q[:200] for q in questions[:5]])
        if klos:
            query_parts.extend([k.get("outcome", "")[:200] for k in klos[:3]])

        query_text = " ".join(query_parts)
        if not query_text.strip():
            return ""

        # Embed query using Pinecone inference
        query_embedding = self._embed_query(query_text)
        if not query_embedding:
            return ""

        # Query Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=max_facts,
            include_metadata=True
        )

        facts = []
        for match in results.get("matches", []):
            text = match.get("metadata", {}).get("text", "")
            if text:
                facts.append(text)

        logger.info(f"[RAG] Retrieved {len(facts)} facts from Pinecone")
        return "\n".join([f"- {fact}" for fact in facts]) if facts else ""

    def clear(self):
        if self.index:
            try:
                self.index.delete(delete_all=True)
                logger.info("[RAG] Pinecone index cleared")
            except Exception as e:
                logger.warning(f"[RAG] Failed to clear Pinecone: {e}")

    def count(self) -> int:
        if not self.index:
            return 0
        try:
            stats = self.index.describe_index_stats()
            return stats.get("total_vector_count", 0)
        except Exception:
            return 0


# =============================================================================
# FACTORY & SINGLETON
# =============================================================================

_retriever: Optional[BaseFactRetriever] = None


def get_fact_retriever() -> BaseFactRetriever:
    """
    Get singleton FactRetriever.

    Auto-selects backend:
    - Pinecone if PINECONE_API_KEY is set
    - ChromaDB otherwise (local)
    """
    global _retriever
    if _retriever is None:
        if os.getenv("PINECONE_API_KEY"):
            logger.info("[RAG] Using Pinecone backend (cloud)")
            _retriever = PineconeFactRetriever()
        else:
            logger.info("[RAG] Using ChromaDB backend (local)")
            _retriever = ChromaFactRetriever()
    return _retriever


async def get_relevant_facts(
    scenario: str,
    questions: list[str] = None,
    klos: list[dict] = None,
    max_facts: int = 12
) -> str:
    """Convenience function to get relevant facts."""
    return await get_fact_retriever().get_relevant_facts(
        scenario, questions, klos, max_facts
    )


# =============================================================================
# CLI for indexing
# =============================================================================

if __name__ == "__main__":
    import asyncio
    import sys
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file

    async def main():
        retriever = get_fact_retriever()

        if len(sys.argv) > 1:
            cmd = sys.argv[1]

            if cmd == "index" and len(sys.argv) > 2:
                file_path = sys.argv[2]
                count = retriever.index_facts_from_file(file_path)
                print(f"Indexed {count} facts from {file_path}")

            elif cmd == "clear":
                retriever.clear()
                print("Cleared all facts")

            elif cmd == "count":
                print(f"Facts indexed: {retriever.count()}")

            elif cmd == "query" and len(sys.argv) > 2:
                query = " ".join(sys.argv[2:])
                facts = await retriever.get_relevant_facts(query)
                print(f"Relevant facts:\n{facts}")

            else:
                print("Usage:")
                print("  python fact_retriever.py index <file>  - Index facts from file")
                print("  python fact_retriever.py clear         - Clear all facts")
                print("  python fact_retriever.py count         - Count indexed facts")
                print("  python fact_retriever.py query <text>  - Query for relevant facts")
        else:
            # Default: index from business_facts.txt
            facts_file = Path(__file__).parent.parent.parent / "data" / "business_facts.txt"
            if facts_file.exists():
                count = retriever.index_facts_from_file(str(facts_file))
                print(f"Indexed {count} facts from {facts_file}")
            else:
                print(f"Facts file not found: {facts_file}")

    asyncio.run(main())
