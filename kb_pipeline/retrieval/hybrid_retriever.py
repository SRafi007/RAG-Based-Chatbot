# kb_pipeline/retrieval/hybrid_retriever.py

from typing import List, Dict
from kb_pipeline.indexing.index_sparse import SparseIndexer
from kb_pipeline.indexing.index_dense import DenseIndexer
from app.utils.logger import get_logger

logger = get_logger(__name__)


class HybridRetriever:
    """
    Hybrid retriever combining sparse (BM25) and dense (semantic) search.
    Uses weighted fusion to combine results from both methods.
    """

    def __init__(
        self,
        sparse_weight: float = 0.5,
        dense_weight: float = 0.5
    ):
        """
        Initialize hybrid retriever.

        Args:
            sparse_weight: Weight for sparse (BM25) results
            dense_weight: Weight for dense (semantic) results
        """
        self.sparse_indexer = SparseIndexer()
        self.dense_indexer = DenseIndexer()
        self.sparse_weight = sparse_weight
        self.dense_weight = dense_weight

        logger.info(
            f"HybridRetriever initialized with weights: "
            f"sparse={sparse_weight}, dense={dense_weight}"
        )

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve documents using hybrid search.

        Args:
            query: Search query
            top_k: Number of final results to return

        Returns:
            List of retrieved documents with scores
        """
        # Get results from both methods (retrieve more for fusion)
        retrieve_k = top_k * 2

        # Sparse retrieval (BM25)
        sparse_results = self.sparse_indexer.search(query, top_k=retrieve_k)

        # Dense retrieval (semantic)
        dense_results = self.dense_indexer.search(query, top_k=retrieve_k)

        # Combine results using weighted fusion
        combined_results = self._fuse_results(sparse_results, dense_results)

        # Return top_k results
        return combined_results[:top_k]

    def _fuse_results(
        self,
        sparse_results: List[Dict],
        dense_results: List[Dict]
    ) -> List[Dict]:
        """
        Fuse sparse and dense results using weighted reciprocal rank fusion.

        Args:
            sparse_results: Results from sparse retrieval
            dense_results: Results from dense retrieval

        Returns:
            Fused and ranked results
        """
        # Normalize scores and create result map
        result_map = {}

        # Process sparse results
        sparse_scores = self._normalize_scores([r['score'] for r in sparse_results])
        for i, result in enumerate(sparse_results):
            chunk_id = result['chunk_id']
            result_map[chunk_id] = {
                **result,
                "sparse_score": sparse_scores[i] * self.sparse_weight,
                "dense_score": 0.0,
                "final_score": sparse_scores[i] * self.sparse_weight,
                "retrieval_method": "sparse"
            }

        # Process dense results
        dense_scores = self._normalize_scores([r['score'] for r in dense_results])
        for i, result in enumerate(dense_results):
            chunk_id = result['chunk_id']
            if chunk_id in result_map:
                # Chunk found in both - combine scores
                result_map[chunk_id]["dense_score"] = dense_scores[i] * self.dense_weight
                result_map[chunk_id]["final_score"] += dense_scores[i] * self.dense_weight
                result_map[chunk_id]["retrieval_method"] = "hybrid"
            else:
                # Chunk only in dense results
                result_map[chunk_id] = {
                    **result,
                    "sparse_score": 0.0,
                    "dense_score": dense_scores[i] * self.dense_weight,
                    "final_score": dense_scores[i] * self.dense_weight,
                    "retrieval_method": "dense"
                }

        # Sort by final score
        fused_results = sorted(
            result_map.values(),
            key=lambda x: x['final_score'],
            reverse=True
        )

        logger.info(
            f"Fused {len(sparse_results)} sparse + {len(dense_results)} dense "
            f"into {len(fused_results)} unique results"
        )

        return fused_results

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """
        Normalize scores to [0, 1] range using min-max normalization.

        Args:
            scores: List of scores

        Returns:
            Normalized scores
        """
        if not scores:
            return []

        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            return [1.0] * len(scores)

        return [
            (score - min_score) / (max_score - min_score)
            for score in scores
        ]


if __name__ == "__main__":
    # Test hybrid retriever
    retriever = HybridRetriever(sparse_weight=0.5, dense_weight=0.5)

    # Test query
    query = "What is the remote work policy?"
    results = retriever.retrieve(query, top_k=3)

    print(f"\nHybrid search results for: '{query}'\n")
    for i, result in enumerate(results, 1):
        print(f"{i}. Retrieval Method: {result['retrieval_method']}")
        print(f"   Final Score: {result['final_score']:.4f}")
        print(f"   Sparse Score: {result['sparse_score']:.4f}")
        print(f"   Dense Score: {result['dense_score']:.4f}")
        print(f"   Source: {result['source']}")
        print(f"   Content: {result['content'][:100]}...")
        print()
