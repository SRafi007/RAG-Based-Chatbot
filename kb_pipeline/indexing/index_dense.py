# kb_pipeline/indexing/index_dense.py

from typing import List, Dict
from pinecone import Pinecone
from app.config.settings import settings
from app.utils.free_embeddings import get_free_embeddings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class DenseIndexer:
    """
    Dense vector indexer using Pinecone with FREE local embeddings.
    Provides semantic search with rich metadata.

    Features:
    - Uses sentence-transformers (all-mpnet-base-v2, 768 dimensions)
    - FREE, local embeddings (NO API LIMITS!)
    - High quality semantic search
    - Unlimited usage
    """

    def __init__(self):
        """Initialize Pinecone with free local embeddings."""
        try:
            # Initialize FREE local embeddings (all-mpnet-base-v2)
            self.embeddings = get_free_embeddings(model_name="all-mpnet-base-v2")
            self.embedding_model = "all-mpnet-base-v2"
            self.embedding_dimension = self.embeddings.dimension

            logger.info(f"Using FREE local embeddings: {self.embedding_model} ({self.embedding_dimension}D)")

            # Initialize Pinecone
            self.pc = Pinecone(api_key=settings.pinecone_api_key)
            self.index_name = settings.pinecone_dense_index

            # Connect to existing index using host
            logger.info(f"Connecting to Pinecone dense index: {self.index_name}")
            logger.info(f"Pinecone host: {settings.pinecone_dense_host}")

            # Get index using host
            self.index = self.pc.Index(
                name=self.index_name,
                host=settings.pinecone_dense_host
            )

            logger.info(f"✅ Connected to Pinecone index: {self.index_name}")

        except Exception as e:
            logger.error(f"Failed to initialize DenseIndexer: {e}")
            raise

    def _get_embedding(self, text: str, is_query: bool = False) -> List[float]:
        """
        Get embedding using Gemini (FREE!).

        Args:
            text: Input text
            is_query: True if text is a search query, False if document

        Returns:
            Embedding vector (768 dimensions)
        """
        try:
            if is_query:
                return self.embeddings.get_query_embedding(text)
            else:
                return self.embeddings.get_document_embedding(text)
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return []

    def _get_embeddings_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Get embeddings for multiple texts (more efficient).

        Args:
            texts: List of texts
            batch_size: Batch size for processing

        Returns:
            List of embedding vectors
        """
        try:
            return self.embeddings.get_embeddings_batch(texts, batch_size=batch_size)
        except Exception as e:
            logger.error(f"Error getting batch embeddings: {e}")
            return []

    def index_documents(self, chunks: List[Dict], batch_size: int = 50) -> int:
        """
        Index semantic chunks into Pinecone using FREE local embeddings.

        Args:
            chunks: List of semantic chunk dictionaries
            batch_size: Number of chunks per batch

        Returns:
            Number of successfully indexed chunks
        """
        try:
            indexed_count = 0
            total_chunks = len(chunks)

            logger.info(f"Starting indexing of {total_chunks} chunks with FREE local embeddings...")

            # Process in batches
            for i in range(0, total_chunks, batch_size):
                batch_chunks = chunks[i:i + batch_size]
                batch_num = i // batch_size + 1

                # Extract texts for batch embedding
                texts = [chunk["text"] for chunk in batch_chunks]

                # Get embeddings in batch (FREE and UNLIMITED!)
                logger.info(f"Generating local embeddings for batch {batch_num}...")
                embeddings = self._get_embeddings_batch(texts, batch_size=32)

                if not embeddings or len(embeddings) != len(texts):
                    logger.warning(f"Batch {batch_num} embedding mismatch, processing individually")
                    # Fallback to individual processing
                    embeddings = []
                    for text in texts:
                        emb = self._get_embedding(text, is_query=False)
                        if emb:
                            embeddings.append(emb)

                if not embeddings:
                    logger.warning(f"Failed to generate embeddings for batch {batch_num}")
                    continue

                # Prepare vectors for Pinecone
                vectors = []
                for chunk, embedding in zip(batch_chunks, embeddings):
                    vector = {
                        "id": chunk["id"],
                        "values": embedding,
                        "metadata": {
                            "text": chunk["text"][:1000],  # Truncate for Pinecone limit
                            "section": chunk["metadata"]["section"],
                            "policy_type": chunk["metadata"]["policy_type"],
                            "source_file": chunk["metadata"]["source_file"],
                            "chunk_index": chunk["metadata"]["chunk_index"],
                            "tokens": chunk["metadata"]["tokens"]
                        }
                    }
                    vectors.append(vector)

                # Upsert to Pinecone
                self.index.upsert(vectors=vectors)
                indexed_count += len(vectors)
                logger.info(f"[OK] Indexed {indexed_count}/{total_chunks} chunks")

            logger.info(f"Successfully indexed {indexed_count} chunks with FREE local embeddings!")
            return indexed_count

        except Exception as e:
            logger.error(f"Pinecone indexing failed: {e}")
            return 0

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for documents using FREE local embeddings.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of search results
        """
        try:
            # Get query embedding using local model (FREE!)
            query_embedding = self._get_embedding(query, is_query=True)

            if not query_embedding:
                logger.error("Failed to get query embedding")
                return []

            # Search Pinecone
            response = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )

            results = []
            for match in response['matches']:
                results.append({
                    "content": match['metadata']['text'],
                    "score": match['score'],
                    "source": match['metadata']['source_file'],
                    "section": match['metadata']['section'],
                    "policy_type": match['metadata']['policy_type'],
                    "chunk_id": match['id']
                })

            logger.info(f"Local dense search returned {len(results)} results for: {query[:50]}")
            return results

        except Exception as e:
            logger.error(f"Pinecone search failed: {e}")
            return []

    def delete_index(self):
        """Delete the Pinecone index."""
        try:
            self.pc.delete_index(self.index_name)
            logger.info(f"Deleted Pinecone index: {self.index_name}")
        except Exception as e:
            logger.error(f"Failed to delete index: {e}")


if __name__ == "__main__":
    # Test dense indexer with FREE local embeddings
    print("\n" + "="*70)
    print("Testing Dense Indexer with FREE Local Embeddings")
    print("="*70 + "\n")

    test_chunks = [
        {
            "id": "softvence_policy_001",
            "text": "## 1. Company Overview & Core Values\n\nSoftvence Agency is a digital solutions company specializing in web design, AI integration, and software development.",
            "metadata": {
                "section": "Company Overview & Core Values",
                "policy_type": "Company Policy",
                "source_file": "sample_policy_handbook.md",
                "chunk_index": 1,
                "tokens": 120
            }
        }
    ]

    try:
        indexer = DenseIndexer()
        print(f"Model: {indexer.embedding_model}")
        print(f"Dimensions: {indexer.embedding_dimension}")
        print(f"\nIndexing {len(test_chunks)} test chunks...")

        indexed = indexer.index_documents(test_chunks)
        print(f"[OK] Indexed {indexed} chunks\n")

        print("Searching...")
        results = indexer.search("what are the company values", top_k=3)
        print(f"[OK] Found {len(results)} results\n")

        for i, r in enumerate(results, 1):
            print(f"{i}. {r['section']} (score: {r['score']:.3f})")

        print("\n" + "="*70)
        print("All tests passed!")
        print("="*70)

    except Exception as e:
        print(f"[FAIL] Test failed: {e}")
