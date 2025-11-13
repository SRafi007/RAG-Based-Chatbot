# kb_pipeline/indexing/index_dense.py

from typing import List, Dict
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from app.config.settings import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class DenseIndexer:
    """
    Dense vector indexer using Pinecone with OpenAI embeddings.
    Provides semantic search with rich metadata.
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

    def index_documents(self, chunks: List[Dict], batch_size: int = 100) -> int:
        """
        Index semantic chunks into Pinecone.

        Args:
            chunks: List of semantic chunk dictionaries with 'id', 'text', and 'metadata'
            batch_size: Number of chunks to process in each batch

        Returns:
            Number of successfully indexed chunks
        """
        try:
            vectors = []
            indexed_count = 0

            for i, chunk in enumerate(chunks):
                # Get embedding from text
                embedding = self._get_embedding(chunk["text"])

                if not embedding:
                    logger.warning(f"Skipping chunk {chunk['id']} - no embedding")
                    continue

                # Prepare vector with metadata
                # Pinecone metadata size limit: keep text truncated in metadata
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

                # Batch upsert
                if len(vectors) >= batch_size:
                    self.index.upsert(vectors=vectors)
                    indexed_count += len(vectors)
                    logger.info(f"Indexed batch: {indexed_count} chunks so far")
                    vectors = []

            # Upsert remaining vectors
            if vectors:
                self.index.upsert(vectors=vectors)
                indexed_count += len(vectors)

            logger.info(f"Successfully indexed {indexed_count} chunks to Pinecone")
            return indexed_count

        except Exception as e:
            logger.error(f"Pinecone indexing failed: {e}")
            return 0

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for documents using semantic similarity.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of search results with content, score, and metadata
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
                    "content": match['metadata']['text'],
                    "score": match['score'],
                    "source": match['metadata']['source_file'],
                    "section": match['metadata']['section'],
                    "policy_type": match['metadata']['policy_type'],
                    "chunk_id": match['id']
                })

            logger.info(f"Dense search returned {len(results)} results for query: {query[:50]}")
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
    # Test dense indexer
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

    indexer = DenseIndexer()
    indexed = indexer.index_documents(test_chunks)
    print(f"Indexed {indexed} chunks")

    results = indexer.search("what are the company values", top_k=3)
    print(f"\nFound {len(results)} results")
    for r in results:
        print(f"- {r['section']} (score: {r['score']:.2f})")
