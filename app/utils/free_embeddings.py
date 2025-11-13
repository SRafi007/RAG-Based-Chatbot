# app/utils/free_embeddings.py

"""
Free local embeddings using sentence-transformers.
NO API LIMITS, NO QUOTA ISSUES!
"""

from sentence_transformers import SentenceTransformer
from typing import List, Union
import logging

logger = logging.getLogger(__name__)


class FreeEmbeddingClient:
    """
    Client for generating embeddings using free local models via sentence-transformers.

    Supported models (all FREE and unlimited):
    - all-mpnet-base-v2: 768 dimensions (RECOMMENDED - best balance)
    - all-MiniLM-L6-v2: 384 dimensions (fastest, smaller)
    - all-MiniLM-L12-v2: 384 dimensions (good balance)
    - paraphrase-multilingual-mpnet-base-v2: 768 dimensions (multilingual)
    """

    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        """
        Initialize the embedding client.

        Args:
            model_name: Name of the sentence-transformers model to use
        """
        self.model_name = model_name
        logger.info(f"Loading sentence-transformers model: {model_name}")

        # Load the model (will download on first use, then cache locally)
        self.model = SentenceTransformer(model_name)

        # Get embedding dimension
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded! Embedding dimension: {self.dimension}")

    def get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Input text to embed

        Returns:
            List of floats representing the embedding vector
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return [0.0] * self.dimension

        # Generate embedding
        embedding = self.model.encode(text, convert_to_numpy=True)

        return embedding.tolist()

    def get_embeddings_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generate embeddings for multiple texts efficiently.

        Args:
            texts: List of input texts to embed
            batch_size: Number of texts to process at once

        Returns:
            List of embedding vectors
        """
        if not texts:
            logger.warning("Empty text list provided for batch embedding")
            return []

        # Filter out empty texts and track indices
        valid_texts = []
        valid_indices = []
        for i, text in enumerate(texts):
            if text and text.strip():
                valid_texts.append(text)
                valid_indices.append(i)

        if not valid_texts:
            logger.warning("No valid texts in batch")
            return [[0.0] * self.dimension] * len(texts)

        # Generate embeddings in batches
        embeddings = self.model.encode(
            valid_texts,
            batch_size=batch_size,
            show_progress_bar=len(valid_texts) > 10,
            convert_to_numpy=True
        )

        # Reconstruct full list with empty embeddings for invalid texts
        result = []
        valid_idx = 0
        for i in range(len(texts)):
            if i in valid_indices:
                result.append(embeddings[valid_idx].tolist())
                valid_idx += 1
            else:
                result.append([0.0] * self.dimension)

        return result

    def get_query_embedding(self, query: str) -> List[float]:
        """
        Generate embedding for a query (alias for get_embedding for compatibility).

        Args:
            query: Query text to embed

        Returns:
            Embedding vector
        """
        return self.get_embedding(query)

    def get_document_embedding(self, document: str) -> List[float]:
        """
        Generate embedding for a document (alias for get_embedding for compatibility).

        Args:
            document: Document text to embed

        Returns:
            Embedding vector
        """
        return self.get_embedding(document)


def get_free_embeddings(model_name: str = "all-mpnet-base-v2") -> FreeEmbeddingClient:
    """
    Factory function to create a FreeEmbeddingClient instance.

    Args:
        model_name: Name of the sentence-transformers model to use

    Returns:
        Configured FreeEmbeddingClient instance
    """
    return FreeEmbeddingClient(model_name=model_name)


# For backward compatibility with gemini_embeddings.py
def get_gemini_embeddings():
    """
    DEPRECATED: Use get_free_embeddings() instead.
    This function now returns local embeddings to avoid quota issues.
    """
    logger.warning("get_gemini_embeddings() is deprecated. Using local embeddings instead to avoid quota issues.")
    return get_free_embeddings()
