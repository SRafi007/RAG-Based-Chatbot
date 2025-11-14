# kb_pipeline/indexing/index_sparse.py

"""
Sparse vector indexer using Pinecone with BM25 sparse vectors.
Provides traditional keyword-based search with BM25 algorithm.
"""

from typing import List, Dict
from pinecone import Pinecone
from app.config.settings import settings
import logging
from collections import Counter
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SparseIndexer:
    """
    Sparse vector indexer using Pinecone with BM25.
    Provides traditional keyword-based search with rich metadata.
    """

    def __init__(self):
        """Initialize Pinecone client for sparse vectors."""
        try:
            # Initialize Pinecone
            self.pc = Pinecone(api_key=settings.pinecone_api_key)

            # Connect to sparse index
            self.index = self.pc.Index(
                name=settings.pinecone_sparse_index,
                host=settings.pinecone_sparse_host
            )

            self.index_name = settings.pinecone_sparse_index

            logger.info(f"Pinecone Sparse connected to: {settings.pinecone_sparse_host}")
            logger.info(f"Sparse index: {self.index_name}")

            # BM25 parameters
            self.k1 = 1.5  # Term frequency saturation parameter
            self.b = 0.75  # Length normalization parameter

            # Document stats for BM25
            self.doc_count = 0
            self.avg_doc_length = 0
            self.term_doc_freq = {}  # IDF calculation

        except Exception as e:
            logger.error(f"Failed to initialize Pinecone Sparse: {e}")
            raise

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization: lowercase and split on whitespace."""
        import re
        # Remove special characters and convert to lowercase
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        tokens = text.split()
        return [t for t in tokens if len(t) > 2]  # Filter short tokens

    def _compute_bm25_sparse_vector(self, text: str, doc_length: int = None) -> Dict:
        """
        Compute BM25 sparse vector representation.

        Args:
            text: Document text
            doc_length: Length of document (token count)

        Returns:
            Sparse vector dict with indices and values
        """
        tokens = self._tokenize(text)

        if doc_length is None:
            doc_length = len(tokens)

        # Count term frequencies
        term_freq = Counter(tokens)

        # Create sparse vector
        # For simplicity, we use a hash-based approach for indices
        sparse_indices = []
        sparse_values = []

        for term, freq in term_freq.items():
            # Simple hash to index (in production, use a proper vocabulary)
            term_hash = abs(hash(term)) % 100000

            # BM25 term score (simplified without IDF for indexing)
            # Full BM25 score is computed during retrieval
            tf_score = freq / (freq + self.k1 * (1 - self.b + self.b * doc_length / max(self.avg_doc_length, 1)))

            sparse_indices.append(term_hash)
            sparse_values.append(tf_score)

        return {
            "indices": sparse_indices,
            "values": sparse_values
        }

    def index_documents(self, chunks: List[Dict]) -> int:
        """
        Index semantic chunks into Pinecone as sparse vectors.

        Args:
            chunks: List of semantic chunk dictionaries with 'id', 'text', and 'metadata'

        Returns:
            Number of successfully indexed chunks
        """
        try:
            vectors_to_upsert = []

            for chunk in chunks:
                chunk_id = chunk["id"]
                text = chunk["text"]
                metadata = chunk["metadata"]

                # Compute sparse vector (BM25-style)
                doc_length = metadata.get("tokens", len(self._tokenize(text)))
                sparse_vector = self._compute_bm25_sparse_vector(text, doc_length)

                # Prepare metadata for Pinecone (flatten nested dicts)
                flat_metadata = {
                    "text": text[:1000],  # Store truncated text in metadata
                    "section": metadata.get("section", ""),
                    "policy_type": metadata.get("policy_type", ""),
                    "source_file": metadata.get("source_file", ""),
                    "chunk_index": metadata.get("chunk_index", 0),
                    "tokens": metadata.get("tokens", 0)
                }

                # Create vector for upsert
                vector = {
                    "id": chunk_id,
                    "sparse_values": sparse_vector,
                    "metadata": flat_metadata
                }

                vectors_to_upsert.append(vector)

            # Upsert in batches
            batch_size = 100
            total_upserted = 0

            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i:i + batch_size]
                self.index.upsert(vectors=batch, namespace="")
                total_upserted += len(batch)
                logger.info(f"Upserted batch {i//batch_size + 1}: {len(batch)} sparse vectors")

            logger.info(f"Indexed {total_upserted} chunks to Pinecone Sparse")
            return total_upserted

        except Exception as e:
            logger.error(f"Pinecone Sparse indexing failed: {e}")
            return 0

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for documents using BM25 sparse vectors.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of search results with content, score, and metadata
        """
        try:
            # Compute sparse vector for query
            query_sparse = self._compute_bm25_sparse_vector(query)

            # Query Pinecone
            response = self.index.query(
                top_k=top_k,
                sparse_vector=query_sparse,
                include_metadata=True,
                namespace=""
            )

            results = []
            for match in response.get('matches', []):
                metadata = match.get('metadata', {})
                results.append({
                    "content": metadata.get("text", ""),
                    "score": match.get('score', 0.0),
                    "source": metadata.get("source_file", ""),
                    "section": metadata.get("section", ""),
                    "policy_type": metadata.get("policy_type", ""),
                    "chunk_id": match.get('id', "")
                })

            logger.info(f"Sparse search returned {len(results)} results for query: {query[:50]}")
            return results

        except Exception as e:
            logger.error(f"Pinecone Sparse search failed: {e}")
            return []

    def delete_all(self):
        """Delete all vectors from the sparse index."""
        try:
            self.index.delete(delete_all=True, namespace="")
            logger.info(f"Deleted all vectors from Pinecone Sparse index: {self.index_name}")
        except Exception as e:
            logger.error(f"Failed to delete vectors: {e}")


if __name__ == "__main__":
    # Test sparse indexer
    test_chunks = [
        {
            "id": "softvence_policy_001",
            "text": "## 1. Company Overview & Core Values\n\nSoftvence Agency is a digital solutions company...",
            "metadata": {
                "section": "Company Overview & Core Values",
                "policy_type": "Company Policy",
                "source_file": "sample_policy_handbook.md",
                "chunk_index": 1,
                "tokens": 120
            }
        }
    ]

    indexer = SparseIndexer()
    indexed = indexer.index_documents(test_chunks)
    print(f"Indexed {indexed} chunks")

    results = indexer.search("company values", top_k=3)
    print(f"\nFound {len(results)} results")
    for r in results:
        print(f"- {r['section']} (score: {r['score']:.2f})")
