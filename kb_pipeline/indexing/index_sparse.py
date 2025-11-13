# kb_pipeline/indexing/index_sparse.py

from typing import List, Dict
from elasticsearch import Elasticsearch, helpers
from app.config.settings import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class SparseIndexer:
    """
    Sparse vector indexer using Elasticsearch with BM25.
    Provides traditional keyword-based search with rich metadata.
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
        """Create Elasticsearch index with proper mappings for semantic chunks."""
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
                    "text": {
                        "type": "text",
                        "analyzer": "custom_analyzer"
                    },
                    "metadata": {
                        "properties": {
                            "section": {"type": "text"},
                            "policy_type": {"type": "keyword"},
                            "source_file": {"type": "keyword"},
                            "chunk_index": {"type": "integer"},
                            "tokens": {"type": "integer"}
                        }
                    }
                }
            }
        }

        self.client.indices.create(index=self.index_name, body=index_mapping)

    def index_documents(self, chunks: List[Dict]) -> int:
        """
        Index semantic chunks into Elasticsearch.

        Args:
            chunks: List of semantic chunk dictionaries with 'id', 'text', and 'metadata'

        Returns:
            Number of successfully indexed chunks
        """
        try:
            # Prepare bulk actions
            actions = []
            for chunk in chunks:
                action = {
                    "_index": self.index_name,
                    "_id": chunk["id"],
                    "_source": {
                        "text": chunk["text"],
                        "metadata": chunk["metadata"]
                    }
                }
                actions.append(action)

            # Bulk index
            success, failed = helpers.bulk(
                self.client,
                actions,
                raise_on_error=False
            )

            logger.info(f"Indexed {success} chunks to Elasticsearch (failed: {len(failed)})")
            return success

        except Exception as e:
            logger.error(f"Elasticsearch indexing failed: {e}")
            return 0

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for documents using BM25.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of search results with content, score, and metadata
        """
        try:
            response = self.client.search(
                index=self.index_name,
                body={
                    "query": {
                        "multi_match": {
                            "query": query,
                            "fields": ["text^2", "metadata.section"],
                            "fuzziness": "AUTO"
                        }
                    },
                    "size": top_k
                }
            )

            results = []
            for hit in response['hits']['hits']:
                results.append({
                    "content": hit['_source']['text'],
                    "score": hit['_score'],
                    "source": hit['_source']['metadata']['source_file'],
                    "section": hit['_source']['metadata']['section'],
                    "policy_type": hit['_source']['metadata']['policy_type'],
                    "chunk_id": hit['_id']
                })

            logger.info(f"Sparse search returned {len(results)} results for query: {query[:50]}")
            return results

        except Exception as e:
            logger.error(f"Elasticsearch search failed: {e}")
            return []

    def delete_index(self):
        """Delete the Elasticsearch index."""
        try:
            if self.client.indices.exists(index=self.index_name):
                self.client.indices.delete(index=self.index_name)
                logger.info(f"Deleted Elasticsearch index: {self.index_name}")
        except Exception as e:
            logger.error(f"Failed to delete index: {e}")


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
