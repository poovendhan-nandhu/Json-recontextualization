"""
Leaf RAG - Retrieval Augmented Generation for leaf-based adaptation.

Indexes leaves by semantic group and retrieves similar examples for ICL.

COLLECTIONS:
- leaf_questions: Questions and submissions
- leaf_resources: Resource content
- leaf_rubrics: Rubric criteria
- leaf_scenarios: Scenario/workplace content
- leaf_klos: KLO descriptions

USAGE:
    from src.core.leaf_rag import LeafRAG

    rag = LeafRAG()

    # Index input for RAG
    rag.index_leaves(leaves, simulation_id="sim_123", industry="beverage")

    # Retrieve similar examples
    examples = rag.retrieve_similar_leaves(
        group="questions",
        query="profit margin analysis",
        n_results=3,
    )
"""

import json
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from langsmith import traceable

logger = logging.getLogger(__name__)


# =============================================================================
# LEAF COLLECTIONS
# =============================================================================

# Valid collections in the vector store:
# 'scenarios', 'klos', 'resources', 'emails', 'activities',
# 'rubrics', 'simulations', 'industry_knowledge', 'entity_mappings'

LEAF_COLLECTIONS = {
    "activities": "Questions, submissions, and assessments",
    "resources": "Resource documents, data tables, reports",
    "rubrics": "Rubric criteria, review guidelines",
    "scenarios": "Scenario descriptions, workplace background",
    "klos": "Key Learning Outcomes, objectives",
    "emails": "Email content, communications",
    "simulations": "Other content",
}

# Map semantic groups to collections
GROUP_TO_COLLECTION = {
    "questions": "activities",
    "submissions": "activities",
    "resources": "resources",
    "rubrics": "rubrics",
    "review": "rubrics",
    "scenario": "scenarios",
    "workplace": "scenarios",
    "klos": "klos",
    "criteria": "klos",
    "emails": "emails",
    "general": "simulations",
    "activities": "activities",
}


@dataclass
class LeafExample:
    """A similar leaf example retrieved for ICL."""
    path: str
    value: str
    group: str
    simulation_id: str
    industry: str
    score: float
    metadata: dict = field(default_factory=dict)


@dataclass
class LeafRAGResult:
    """Result of leaf indexing."""
    simulation_id: str
    leaves_indexed: int
    by_collection: Dict[str, int]
    industry: str


# =============================================================================
# LEAF RAG
# =============================================================================

class LeafRAG:
    """
    RAG for leaf-based adaptation.

    Indexes leaves into semantic collections and retrieves similar examples.
    """

    def __init__(self, vector_store=None):
        """
        Initialize LeafRAG.

        Args:
            vector_store: VectorStore instance (creates one if None)
        """
        try:
            from ..rag.vector_store import get_vector_store
            self.store = vector_store or get_vector_store()
            self.available = True
        except Exception as e:
            logger.warning(f"LeafRAG: Vector store not available: {e}")
            self.store = None
            self.available = False

    def _get_collection_for_group(self, group: str) -> str:
        """Get collection name for a semantic group."""
        group_lower = group.lower()

        for key, collection in GROUP_TO_COLLECTION.items():
            if key in group_lower:
                return collection

        return "simulations"  # Fallback to simulations collection

    def _classify_leaf_group(self, path: str) -> str:
        """Classify a leaf path into a semantic group."""
        path_lower = path.lower()

        if "question" in path_lower or "submission" in path_lower:
            return "questions"
        elif "resource" in path_lower or "document" in path_lower:
            return "resources"
        elif "rubric" in path_lower or "review" in path_lower:
            return "rubrics"
        elif "scenario" in path_lower or "workplace" in path_lower or "background" in path_lower:
            return "scenarios"
        elif "klo" in path_lower or "criterion" in path_lower or "criteria" in path_lower:
            return "klos"
        elif "email" in path_lower:
            return "emails"
        else:
            return "general"

    @traceable(name="leaf_rag_index")
    def index_leaves(
        self,
        leaves: List[tuple],  # List of (path, value) tuples
        simulation_id: str,
        industry: str = "unknown",
        clear_existing: bool = False,
    ) -> LeafRAGResult:
        """
        Index leaves into semantic collections.

        Args:
            leaves: List of (path, value) tuples from indexer
            simulation_id: Unique identifier for this simulation
            industry: Industry of the simulation
            clear_existing: Clear existing docs for this simulation

        Returns:
            LeafRAGResult with indexing stats
        """
        if not self.available:
            logger.warning("LeafRAG not available - skipping indexing")
            return LeafRAGResult(
                simulation_id=simulation_id,
                leaves_indexed=0,
                by_collection={},
                industry=industry,
            )

        by_collection = {}
        total_indexed = 0

        # Group leaves by collection
        collection_docs: Dict[str, List[Dict]] = {}

        for path, value in leaves:
            # Skip non-string values
            if not isinstance(value, str) or len(value.strip()) < 10:
                continue

            # Classify and get collection
            group = self._classify_leaf_group(path)
            collection = self._get_collection_for_group(group)

            if collection not in collection_docs:
                collection_docs[collection] = []

            doc_id = f"{simulation_id}_{hash(path)}"

            collection_docs[collection].append({
                "id": doc_id,
                "document": value[:5000],  # Limit length
                "metadata": {
                    "simulation_id": simulation_id,
                    "path": path,
                    "group": group,
                    "industry": industry,
                    "value_length": len(value),
                },
            })

        # Index each collection
        for collection, docs in collection_docs.items():
            if not docs:
                continue

            try:
                # Clear existing if requested
                if clear_existing:
                    try:
                        existing_ids = [d["id"] for d in docs]
                        self.store.delete(collection, existing_ids)
                    except Exception:
                        pass

                # Add documents
                self.store.add_documents(
                    collection_name=collection,
                    documents=[d["document"] for d in docs],
                    metadatas=[d["metadata"] for d in docs],
                    ids=[d["id"] for d in docs],
                )

                by_collection[collection] = len(docs)
                total_indexed += len(docs)

            except Exception as e:
                logger.warning(f"Failed to index to {collection}: {e}")

        logger.info(f"Indexed {total_indexed} leaves for {simulation_id}: {by_collection}")

        return LeafRAGResult(
            simulation_id=simulation_id,
            leaves_indexed=total_indexed,
            by_collection=by_collection,
            industry=industry,
        )

    @traceable(name="leaf_rag_index_parallel")
    async def index_leaves_parallel(
        self,
        leaves: List[tuple],
        simulation_id: str,
        industry: str = "unknown",
        clear_existing: bool = False,
    ) -> LeafRAGResult:
        """
        Index leaves into semantic collections in PARALLEL.

        Much faster than sequential - indexes all collections concurrently.
        """
        if not self.available:
            logger.warning("LeafRAG not available - skipping indexing")
            return LeafRAGResult(
                simulation_id=simulation_id,
                leaves_indexed=0,
                by_collection={},
                industry=industry,
            )

        # Group leaves by collection (this is fast, no I/O)
        collection_docs: Dict[str, List[Dict]] = {}

        for path, value in leaves:
            if not isinstance(value, str) or len(value.strip()) < 10:
                continue

            group = self._classify_leaf_group(path)
            collection = self._get_collection_for_group(group)

            if collection not in collection_docs:
                collection_docs[collection] = []

            doc_id = f"{simulation_id}_{hash(path)}"

            collection_docs[collection].append({
                "id": doc_id,
                "document": value[:5000],
                "metadata": {
                    "simulation_id": simulation_id,
                    "path": path,
                    "group": group,
                    "industry": industry,
                    "value_length": len(value),
                },
            })

        # Index all collections in PARALLEL
        loop = asyncio.get_event_loop()
        by_collection = {}

        async def index_collection(collection: str, docs: List[Dict]) -> tuple:
            """Index one collection in thread pool."""
            def _index():
                try:
                    if clear_existing:
                        try:
                            existing_ids = [d["id"] for d in docs]
                            self.store.delete(collection, existing_ids)
                        except Exception:
                            pass

                    self.store.add_documents(
                        collection_name=collection,
                        documents=[d["document"] for d in docs],
                        metadatas=[d["metadata"] for d in docs],
                        ids=[d["id"] for d in docs],
                    )
                    return (collection, len(docs))
                except Exception as e:
                    logger.warning(f"Failed to index to {collection}: {e}")
                    return (collection, 0)

            with ThreadPoolExecutor(max_workers=1) as executor:
                result = await loop.run_in_executor(executor, _index)
            return result

        # Run all indexing in parallel
        tasks = [
            index_collection(coll, docs)
            for coll, docs in collection_docs.items()
            if docs
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect results
        total_indexed = 0
        for result in results:
            if isinstance(result, tuple):
                coll, count = result
                by_collection[coll] = count
                total_indexed += count

        logger.info(f"Parallel indexed {total_indexed} leaves for {simulation_id}: {by_collection}")

        return LeafRAGResult(
            simulation_id=simulation_id,
            leaves_indexed=total_indexed,
            by_collection=by_collection,
            industry=industry,
        )

    @traceable(name="leaf_rag_retrieve")
    def retrieve_similar_leaves(
        self,
        group: str,
        query: str,
        n_results: int = 3,
        exclude_simulation: str = None,
        industry_filter: str = None,
    ) -> List[LeafExample]:
        """
        Retrieve similar leaf examples for ICL.

        Args:
            group: Semantic group (questions, resources, rubrics, etc.)
            query: Search query (target scenario or content)
            n_results: Number of results
            exclude_simulation: Simulation to exclude
            industry_filter: Prioritize this industry

        Returns:
            List of LeafExample for ICL
        """
        if not self.available:
            return []

        collection = self._get_collection_for_group(group)

        try:
            results = self.store.query(
                collection_name=collection,
                query_texts=[query],
                n_results=n_results * 2,  # Get more to filter
            )
        except Exception as e:
            logger.warning(f"Failed to query {collection}: {e}")
            return []

        examples = []

        if results.get("documents") and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                metadata = results["metadatas"][0][i] if results.get("metadatas") else {}
                score = results["distances"][0][i] if results.get("distances") else 0.0

                sim_id = metadata.get("simulation_id", "")

                # Skip excluded simulation
                if exclude_simulation and sim_id == exclude_simulation:
                    continue

                examples.append(LeafExample(
                    path=metadata.get("path", ""),
                    value=doc,
                    group=metadata.get("group", group),
                    simulation_id=sim_id,
                    industry=metadata.get("industry", "unknown"),
                    score=score,
                    metadata=metadata,
                ))

                if len(examples) >= n_results:
                    break

        # Sort by industry match if filter provided
        if industry_filter:
            examples.sort(key=lambda x: (
                0 if x.industry.lower() == industry_filter.lower() else 1,
                x.score
            ))

        return examples[:n_results]

    def retrieve_examples_for_groups(
        self,
        groups: List[str],
        query: str,
        n_per_group: int = 2,
        exclude_simulation: str = None,
        industry: str = None,
    ) -> Dict[str, List[LeafExample]]:
        """
        Retrieve examples for multiple groups at once (sequential).

        Args:
            groups: List of semantic groups
            query: Search query
            n_per_group: Examples per group
            exclude_simulation: Simulation to exclude
            industry: Industry to prioritize

        Returns:
            Dict mapping group names to examples
        """
        all_examples = {}

        for group in groups:
            examples = self.retrieve_similar_leaves(
                group=group,
                query=query,
                n_results=n_per_group,
                exclude_simulation=exclude_simulation,
                industry_filter=industry,
            )
            all_examples[group] = examples

        return all_examples

    @traceable(name="leaf_rag_retrieve_parallel")
    async def retrieve_examples_for_groups_parallel(
        self,
        groups: List[str],
        query: str,
        n_per_group: int = 2,
        exclude_simulation: str = None,
        industry: str = None,
    ) -> Dict[str, List[LeafExample]]:
        """
        Retrieve examples for multiple groups in PARALLEL.

        Uses ThreadPoolExecutor to run vector store queries concurrently.
        Much faster than sequential when querying multiple groups.

        Args:
            groups: List of semantic groups
            query: Search query
            n_per_group: Examples per group
            exclude_simulation: Simulation to exclude
            industry: Industry to prioritize

        Returns:
            Dict mapping group names to examples
        """
        if not self.available:
            return {group: [] for group in groups}

        # Create tasks for parallel execution
        loop = asyncio.get_event_loop()

        async def fetch_group(group: str) -> tuple:
            """Fetch examples for one group in thread pool."""
            def _fetch():
                return self.retrieve_similar_leaves(
                    group=group,
                    query=query,
                    n_results=n_per_group,
                    exclude_simulation=exclude_simulation,
                    industry_filter=industry,
                )
            # Run sync operation in thread pool
            with ThreadPoolExecutor(max_workers=1) as executor:
                examples = await loop.run_in_executor(executor, _fetch)
            return (group, examples)

        # Run all queries in parallel
        tasks = [fetch_group(group) for group in groups]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Build result dict
        all_examples = {}
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Parallel RAG fetch failed: {result}")
            elif isinstance(result, tuple):
                group, examples = result
                all_examples[group] = examples

        # Fill in any missing groups with empty list
        for group in groups:
            if group not in all_examples:
                all_examples[group] = []

        logger.info(f"Parallel RAG fetched {len(groups)} groups: {[len(v) for v in all_examples.values()]} examples")
        return all_examples

    def format_examples_for_prompt(
        self,
        examples: List[LeafExample],
        max_chars: int = 3000,
    ) -> str:
        """
        Format examples for inclusion in LLM prompt.

        Args:
            examples: List of LeafExample
            max_chars: Maximum characters

        Returns:
            Formatted string for prompt
        """
        if not examples:
            return ""

        parts = ["SIMILAR EXAMPLES (for reference):\n"]
        total_chars = len(parts[0])

        for i, ex in enumerate(examples, 1):
            header = f"\n[Example {i}] Industry: {ex.industry}\n"
            header += f"Path: {ex.path}\n"

            content = ex.value
            remaining = max_chars - total_chars - len(header) - 50
            if len(content) > remaining:
                content = content[:remaining] + "..."

            example_text = header + f"Value: {content}\n"
            total_chars += len(example_text)

            if total_chars > max_chars:
                break

            parts.append(example_text)

        return "".join(parts)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def index_leaves_for_rag(
    leaves: List[tuple],
    simulation_id: str,
    industry: str = "unknown",
    clear_existing: bool = False,
) -> LeafRAGResult:
    """
    Index leaves for RAG retrieval.

    Args:
        leaves: List of (path, value) tuples
        simulation_id: Simulation identifier
        industry: Industry of simulation
        clear_existing: Clear existing first

    Returns:
        LeafRAGResult
    """
    rag = LeafRAG()
    return rag.index_leaves(leaves, simulation_id, industry, clear_existing)


def retrieve_leaf_examples(
    group: str,
    query: str,
    n_results: int = 3,
    industry: str = None,
) -> List[LeafExample]:
    """
    Retrieve similar leaf examples.

    Args:
        group: Semantic group
        query: Search query
        n_results: Number of results
        industry: Industry filter

    Returns:
        List of LeafExample
    """
    rag = LeafRAG()
    return rag.retrieve_similar_leaves(group, query, n_results, industry_filter=industry)


def get_rag_context_for_adaptation(
    target_scenario: str,
    groups: List[str] = None,
    n_per_group: int = 2,
    industry: str = None,
) -> Dict[str, str]:
    """
    Get RAG context formatted for adaptation prompts (sequential).

    Args:
        target_scenario: Target scenario description
        groups: Groups to retrieve (defaults to all)
        n_per_group: Examples per group
        industry: Industry filter

    Returns:
        Dict mapping group names to formatted context strings
    """
    if groups is None:
        groups = ["questions", "resources", "rubrics", "scenarios", "klos"]

    rag = LeafRAG()
    all_examples = rag.retrieve_examples_for_groups(
        groups=groups,
        query=target_scenario,
        n_per_group=n_per_group,
        industry=industry,
    )

    context = {}
    for group, examples in all_examples.items():
        context[group] = rag.format_examples_for_prompt(examples)

    return context


async def get_rag_context_for_adaptation_parallel(
    target_scenario: str,
    groups: List[str] = None,
    n_per_group: int = 2,
    industry: str = None,
) -> Dict[str, str]:
    """
    Get RAG context formatted for adaptation prompts (PARALLEL).

    Much faster than sequential - queries all groups concurrently.

    Args:
        target_scenario: Target scenario description
        groups: Groups to retrieve (defaults to all)
        n_per_group: Examples per group
        industry: Industry filter

    Returns:
        Dict mapping group names to formatted context strings
    """
    if groups is None:
        groups = ["questions", "resources", "rubrics", "scenarios", "klos"]

    rag = LeafRAG()
    all_examples = await rag.retrieve_examples_for_groups_parallel(
        groups=groups,
        query=target_scenario,
        n_per_group=n_per_group,
        industry=industry,
    )

    context = {}
    for group, examples in all_examples.items():
        context[group] = rag.format_examples_for_prompt(examples)

    return context
