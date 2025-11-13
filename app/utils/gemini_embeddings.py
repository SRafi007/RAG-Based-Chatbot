# app/utils/gemini_embeddings.py

"""
Gemini Embeddings Client - FREE alternative to OpenAI
Uses Google's embedding-001 model (768 dimensions)
"""

from typing import List, Union
import google.generativeai as genai
from app.config.settings import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class GeminiEmbeddingClient:
    """
    Gemini embedding client using models/embedding-001.

    Features:
    - FREE (included with Gemini API)
    - 768 dimensions
    - High quality embeddings
    - Good for semantic search
    """

    def __init__(self):
        """Initialize Gemini embedding client."""
        try:
            # Configure Gemini API
            genai.configure(api_key=settings.gemini_api_key)
            self.model_name = settings.embedding_model
            self.dimension = settings.embedding_dimension

            logger.info(
                f"Gemini embeddings initialized: {self.model_name} ({self.dimension}D)"
            )

        except Exception as e:
            logger.error(f"Failed to initialize Gemini embeddings: {e}")
            raise

    def get_embedding(self, text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Input text
            task_type: Task type for embedding
                - "RETRIEVAL_DOCUMENT": For documents to be retrieved
                - "RETRIEVAL_QUERY": For search queries
                - "SEMANTIC_SIMILARITY": For similarity comparison
                - "CLASSIFICATION": For classification tasks

        Returns:
            Embedding vector (768 dimensions)
        """
        try:
            result = genai.embed_content(
                model=self.model_name,
                content=text,
                task_type=task_type
            )
            return result['embedding']

        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return []

    def get_embeddings_batch(
        self,
        texts: List[str],
        task_type: str = "RETRIEVAL_DOCUMENT"
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of input texts
            task_type: Task type for embeddings

        Returns:
            List of embedding vectors
        """
        try:
            # Gemini supports batch embedding
            result = genai.embed_content(
                model=self.model_name,
                content=texts,
                task_type=task_type
            )

            # Extract embeddings
            if isinstance(result['embedding'][0], list):
                # Multiple embeddings returned
                return result['embedding']
            else:
                # Single embedding returned as flat list
                return [result['embedding']]

        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            # Fallback to individual processing
            logger.info("Falling back to individual embedding generation")
            embeddings = []
            for text in texts:
                emb = self.get_embedding(text, task_type)
                if emb:
                    embeddings.append(emb)
            return embeddings

    def get_query_embedding(self, query: str) -> List[float]:
        """
        Generate embedding specifically for search queries.

        Args:
            query: Search query

        Returns:
            Query embedding vector
        """
        return self.get_embedding(query, task_type="RETRIEVAL_QUERY")

    def get_document_embedding(self, document: str) -> List[float]:
        """
        Generate embedding specifically for documents.

        Args:
            document: Document text

        Returns:
            Document embedding vector
        """
        return self.get_embedding(document, task_type="RETRIEVAL_DOCUMENT")


# Singleton instance
gemini_embeddings = None


def get_gemini_embeddings() -> GeminiEmbeddingClient:
    """Get or create Gemini embedding client singleton."""
    global gemini_embeddings
    if gemini_embeddings is None:
        gemini_embeddings = GeminiEmbeddingClient()
    return gemini_embeddings


if __name__ == "__main__":
    # Test Gemini embeddings
    print("\n" + "="*70)
    print("Testing Gemini Embeddings (FREE!)")
    print("="*70 + "\n")

    try:
        client = GeminiEmbeddingClient()

        # Test single embedding
        text = "What is the company's remote work policy?"
        embedding = client.get_document_embedding(text)

        print(f"✅ Model: {client.model_name}")
        print(f"✅ Dimensions: {client.dimension}")
        print(f"✅ Text: {text}")
        print(f"✅ Embedding (first 5 values): {embedding[:5]}")
        print(f"✅ Embedding length: {len(embedding)}")

        # Test query embedding
        query = "remote work policy"
        query_emb = client.get_query_embedding(query)
        print(f"\n✅ Query embedding generated: {len(query_emb)} dimensions")

        # Test batch embeddings
        texts = [
            "Vacation policy for employees",
            "Sick leave policy",
            "Remote work guidelines"
        ]
        print(f"\n✅ Batch processing {len(texts)} texts...")
        embeddings = client.get_embeddings_batch(texts)
        print(f"✅ Generated {len(embeddings)} embeddings")

        print("\n" + "="*70)
        print("🎉 All tests passed! Gemini embeddings are ready.")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check GEMINI_API_KEY is set in .env")
        print("2. Verify API key is active")
        print("3. Check internet connection")
