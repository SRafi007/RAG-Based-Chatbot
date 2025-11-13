# kb_pipeline/indexing/index_sparse.py

from typing import List, Dict
from elasticsearch import Elasticsearch, helpers
from app.config.settings import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class SparseIndexer:
    """
    Sparse vector indexer using Elasticsearch with BM25.
    Provides traditional keyword-based search.
    """

    def __init__(self):
        """Initialize Elasticsearch client."""
        try:
            self.client = Elasticsearch(
                [settings.elastic_url],
                verify_certs=True
            )
            self.index_name = settings.elastic_index

            # Create index if it doesn't exist
            if not self.client.indices.exists(index=self.index_name):
                self._create_index()
                logger.info(f"Created Elasticsearch index: {self.index_name}")
            else:
                logger.info(f"Using existing Elasticsearch index: {self.index_name}")

        except Exception as e:
            logger.error(f"Failed to initialize Elasticsearch: {e}")
            raise

    def _create_index(self):
        """Create Elasticsearch index with proper mappings."""
        index_mapping = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "analysis": {
                    "analyzer": {
                        "custom_analyzer": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": ["lowercase", "stop", "snowball"]
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "content": {
                        "type": "text",
                        "analyzer": "custom_analyzer"
                    },
                    "source": {"type": "keyword"},
                    "chunk_id": {"type": "keyword"},
                    "chunk_index": {"type": "integer"},
                    "total_chunks": {"type": "integer"},
                    "file_type": {"type": "keyword"}
                }
            }
        }

        self.client.indices.create(index=self.index_name, body=index_mapping)

    def index_documents(self, chunks: List[Dict[str, str]]) -> int:
        """
        Index document chunks into Elasticsearch.

        Args:
            chunks: List of preprocessed chunk dictionaries

        Returns:
            Number of successfully indexed chunks
        """
        try:
            # Prepare bulk index operations
            actions = [
                {
                    "_index": self.index_name,
                    "_id": chunk["chunk_id"],
                    "_source": {
                        "content": chunk["content"],
                        "source": chunk["source"],
                        "chunk_id": chunk["chunk_id"],
                        "chunk_index": chunk["chunk_index"],
                        "total_chunks": chunk["total_chunks"],
                        "file_type": chunk.get("file_type", "unknown")
                    }
                }
                for chunk in chunks
            ]

            # Bulk index
            success, failed = helpers.bulk(
                self.client,
                actions,
                raise_on_error=False
            )

            logger.info(f"Indexed {success} documents to Elasticsearch")
            if failed:
                logger.warning(f"Failed to index {len(failed)} documents")

            return success

        except Exception as e:
            logger.error(f"Error indexing documents: {e}")
            return 0

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for documents using BM25 (sparse retrieval).

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of search results with scores
        """
        try:
            response = self.client.search(
                index=self.index_name,
                body={
                    "query": {
                        "match": {
                            "content": {
                                "query": query,
                                "fuzziness": "AUTO"
                            }
                        }
                    },
                    "size": top_k,
                    "_source": ["content", "source", "chunk_id", "chunk_index"]
                }
            )

            results = []
            for hit in response['hits']['hits']:
                results.append({
                    "content": hit['_source']['content'],
                    "source": hit['_source']['source'],
                    "chunk_id": hit['_source']['chunk_id'],
                    "score": hit['_score'],
                    "retrieval_method": "sparse"
                })

            logger.info(f"Sparse search returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Error searching Elasticsearch: {e}")
            return []

    def delete_index(self):
        """Delete the Elasticsearch index."""
        try:
            self.client.indices.delete(index=self.index_name)
            logger.info(f"Deleted index: {self.index_name}")
        except Exception as e:
            logger.error(f"Error deleting index: {e}")


if __name__ == "__main__":
    # Test sparse indexer
    indexer = SparseIndexer()

    # Test data
    test_chunks = [
        {
            "content": "Remote work policy allows 3 days per week with manager approval.",
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
    results = indexer.search("remote work policy", top_k=3)
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result['score']:.4f}")
        print(f"   Source: {result['source']}")
        print(f"   Content: {result['content'][:100]}...")
