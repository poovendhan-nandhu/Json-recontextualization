"""
OpenAI Embeddings for RAG.

Uses text-embedding-3-small for efficient, high-quality embeddings.
"""
import os
import logging
from typing import Optional
from functools import lru_cache

logger = logging.getLogger(__name__)

# Lazy imports
_openai_client = None


def _get_openai():
    """Lazy import and initialize OpenAI client."""
    global _openai_client
    if _openai_client is None:
        try:
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            _openai_client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("openai is required. Install with: pip install openai")
    return _openai_client


class EmbeddingModel:
    """
    OpenAI embedding model wrapper.

    Uses text-embedding-3-small by default (fast, cheap, good quality).
    """

    def __init__(self, model: str = "text-embedding-3-small"):
        """
        Initialize embedding model.

        Args:
            model: OpenAI embedding model name
                - text-embedding-3-small: Fast, cheap, 1536 dims
                - text-embedding-3-large: Better quality, 3072 dims
                - text-embedding-ada-002: Legacy, 1536 dims
        """
        self.model = model
        self._dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }

    @property
    def dimensions(self) -> int:
        """Get embedding dimensions for this model."""
        return self._dimensions.get(self.model, 1536)

    def embed(self, text: str) -> list[float]:
        """
        Embed a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        client = _get_openai()

        # Truncate if too long (max ~8k tokens for embedding models)
        if len(text) > 30000:
            text = text[:30000]
            logger.warning("Text truncated to 30k chars for embedding")

        response = client.embeddings.create(
            model=self.model,
            input=text,
        )

        return response.data[0].embedding

    def embed_batch(self, texts: list[str], batch_size: int = 100) -> list[list[float]]:
        """
        Embed multiple texts in batches.

        Args:
            texts: List of texts to embed
            batch_size: Texts per API call

        Returns:
            List of embedding vectors
        """
        client = _get_openai()
        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            # Truncate long texts
            batch = [t[:30000] if len(t) > 30000 else t for t in batch]

            response = client.embeddings.create(
                model=self.model,
                input=batch,
            )

            # Extract embeddings in order
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

            logger.debug(f"Embedded batch {i//batch_size + 1}, total: {len(all_embeddings)}")

        return all_embeddings


# Singleton instance
_embedding_model: Optional[EmbeddingModel] = None


def get_embeddings(model: str = None) -> EmbeddingModel:
    """
    Get singleton embedding model.

    Args:
        model: Override default model

    Returns:
        EmbeddingModel instance
    """
    global _embedding_model

    if _embedding_model is None:
        # Import config here to avoid circular imports
        try:
            from ..utils.config import config
            default_model = config.EMBEDDING_MODEL
        except ImportError:
            default_model = "text-embedding-3-small"

        _embedding_model = EmbeddingModel(model or default_model)

    return _embedding_model


def embed_text(text: str) -> list[float]:
    """
    Convenience function to embed single text.

    Args:
        text: Text to embed

    Returns:
        Embedding vector
    """
    return get_embeddings().embed(text)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Convenience function to embed multiple texts.

    Args:
        texts: List of texts to embed

    Returns:
        List of embedding vectors
    """
    return get_embeddings().embed_batch(texts)
