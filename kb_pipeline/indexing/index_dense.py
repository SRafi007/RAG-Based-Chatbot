# kb_pipeline/indexing/index_dense.py

from typing import List, Dict
import pinecone
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from app.config.settings import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class DenseIndexer:
    """
    Dense vector indexer using Pinecone with OpenAI embeddings.
    Provides semantic search capabilities.
    """

    def __init__(self):
        """Initialize Pinecone and OpenAI clients."""
        try:
            # Initialize Pinecone
            self.pc = Pinecone(api_key=settings.pinecone_api_key)
            self.index_name = settings.pinecone_index

            # Initialize OpenAI for embeddings
            self.openai_client = OpenAI(api_key=settings.openai_api_key)
            self.embedding_model = settings.embedding_model
            self.embedding_dimension = 1536  # For text-embedding-3-small

            # Create index if it doesn't exist
            if self.index_name not in self.pc.list_indexes().names():
                self._create_index()
                logger.info(f"Created Pinecone index: {self.index_name}")
            else:
                logger.info(f"Using existing Pinecone index: {self.index_name}")

            # Get index
            self.index = self.pc.Index(self.index_name)

        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            raise

    def _create_index(self):
        """Create Pinecone index."""
        self.pc.create_index(
            name=self.index_name,
            dimension=self.embedding_dimension,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region=settings.pinecone_env
            )
        )

    def _get_embedding(self, text: str) -> List[float]:
        """
        Get dense embedding for text using OpenAI.

        Args:
            text: Input text

        Returns:
            Dense embedding vector
        """
        try:
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return []

    def index_documents(self, chunks: List[Dict[str, str]], batch_size: int = 100) -> int:
        """
        Index document chunks into Pinecone.

        Args:
            chunks: List of preprocessed chunk dictionaries
            batch_size: Number of chunks to process in each batch

        Returns:
            Number of successfully indexed chunks
        """
        try:
            vectors = []
            indexed_count = 0

            for i, chunk in enumerate(chunks):
                # Get embedding
                embedding = self._get_embedding(chunk["content"])

                if not embedding:
                    logger.warning(f"Skipping chunk {chunk['chunk_id']} - no embedding")
                    continue

                # Prepare vector for upsert
                vector = {
                    "id": chunk["chunk_id"],
                    "values": embedding,
                    "metadata": {
                        "content": chunk["content"][:1000],  # Pinecone metadata limit
                        "source": chunk["source"],
                        "chunk_index": chunk["chunk_index"],
                        "file_type": chunk.get("file_type", "unknown")
                    }
                }
                vectors.append(vector)

                # Upsert in batches
                if len(vectors) >= batch_size or i == len(chunks) - 1:
                    self.index.upsert(vectors=vectors)
                    indexed_count += len(vectors)
                    logger.info(f"Indexed {indexed_count}/{len(chunks)} chunks to Pinecone")
                    vectors = []

            logger.info(f"Successfully indexed {indexed_count} chunks to Pinecone")
            return indexed_count

        except Exception as e:
            logger.error(f"Error indexing documents: {e}")
            return 0

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for documents using dense vector similarity.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of search results with scores
        """
        try:
            # Get query embedding
            query_embedding = self._get_embedding(query)

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
                    "content": match['metadata']['content'],
                    "source": match['metadata']['source'],
                    "chunk_id": match['id'],
                    "score": match['score'],
                    "retrieval_method": "dense"
                })

            logger.info(f"Dense search returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Error searching Pinecone: {e}")
            return []

    def delete_index(self):
        """Delete the Pinecone index."""
        try:
            self.pc.delete_index(self.index_name)
            logger.info(f"Deleted index: {self.index_name}")
        except Exception as e:
            logger.error(f"Error deleting index: {e}")


if __name__ == "__main__":
    # Test dense indexer
    indexer = DenseIndexer()

    # Test data
    test_chunks = [
        {
            "content": "Employees can work remotely up to 3 days per week with manager approval. "
                       "Full-time remote work requires VP approval.",
            "source": "hr_policy.pdf",
            "chunk_id": "hr_policy_chunk_0",
            "chunk_index": 0,
            "total_chunks": 1,
            "file_type": "pdf"
        }
    ]

    # Index
    indexed = indexer.index_documents(test_chunks)
    print(f"Indexed {indexed} chunks")

    # Search
    results = indexer.search("Can I work from home?", top_k=3)
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result['score']:.4f}")
        print(f"   Source: {result['source']}")
        print(f"   Content: {result['content'][:100]}...")
