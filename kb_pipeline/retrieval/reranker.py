# kb_pipeline/retrieval/reranker.py

from typing import List, Dict
from app.utils.llm_client import gemini_client
from app.utils.logger import get_logger

logger = get_logger(__name__)


class Reranker:
    """
    Rerank retrieved documents using LLM-based relevance scoring.
    Provides a second-stage ranking to improve retrieval quality.
    """

    def __init__(self, use_llm: bool = True):
        """
        Initialize reranker.

        Args:
            use_llm: Whether to use LLM for reranking (otherwise use simple heuristics)
        """
        self.use_llm = use_llm
        logger.info(f"Reranker initialized (LLM-based: {use_llm})")

    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = 5
    ) -> List[Dict]:
        """
        Rerank documents based on relevance to query.

        Args:
            query: User query
            documents: List of retrieved documents
            top_k: Number of top documents to return

        Returns:
            Reranked documents
        """
        if not documents:
            return []

        if self.use_llm:
            reranked = self._llm_rerank(query, documents)
        else:
            reranked = self._heuristic_rerank(query, documents)

        return reranked[:top_k]

    def _llm_rerank(self, query: str, documents: List[Dict]) -> List[Dict]:
        """
        Rerank using LLM to score relevance.

        Args:
            query: User query
            documents: List of documents

        Returns:
            Reranked documents with LLM scores
        """
        try:
            reranked_docs = []

            for doc in documents:
                # Create prompt for relevance scoring
                prompt = f"""Rate the relevance of this document to the query on a scale of 0-10.
Only respond with a number.

Query: {query}

Document: {doc['content'][:500]}

Relevance score (0-10):"""

                # Get LLM score
                response = gemini_client.generate(prompt)

                try:
                    llm_score = float(response.strip())
                    llm_score = max(0.0, min(10.0, llm_score)) / 10.0  # Normalize to [0, 1]
                except ValueError:
                    logger.warning(f"Invalid LLM score: {response}, using default 0.5")
                    llm_score = 0.5

                # Combine with original score
                original_score = doc.get('final_score', doc.get('score', 0.0))
                combined_score = (original_score + llm_score) / 2.0

                reranked_docs.append({
                    **doc,
                    "llm_score": llm_score,
                    "original_score": original_score,
                    "rerank_score": combined_score
                })

            # Sort by rerank score
            reranked_docs.sort(key=lambda x: x['rerank_score'], reverse=True)

            logger.info(f"LLM reranking complete for {len(documents)} documents")
            return reranked_docs

        except Exception as e:
            logger.error(f"Error in LLM reranking: {e}")
            return documents

    def _heuristic_rerank(self, query: str, documents: List[Dict]) -> List[Dict]:
        """
        Rerank using simple heuristics (query term matching, length, etc.).

        Args:
            query: User query
            documents: List of documents

        Returns:
            Reranked documents
        """
        query_terms = set(query.lower().split())

        for doc in documents:
            content_lower = doc['content'].lower()

            # Count query term matches
            term_matches = sum(1 for term in query_terms if term in content_lower)
            term_coverage = term_matches / len(query_terms) if query_terms else 0

            # Consider document length (prefer moderate length)
            content_length = len(doc['content'])
            optimal_length = 500
            length_score = 1.0 - abs(content_length - optimal_length) / optimal_length
            length_score = max(0.0, min(1.0, length_score))

            # Combine scores
            original_score = doc.get('final_score', doc.get('score', 0.0))
            heuristic_score = (term_coverage * 0.7 + length_score * 0.3)
            combined_score = (original_score + heuristic_score) / 2.0

            doc['heuristic_score'] = heuristic_score
            doc['original_score'] = original_score
            doc['rerank_score'] = combined_score

        # Sort by rerank score
        documents.sort(key=lambda x: x['rerank_score'], reverse=True)

        logger.info(f"Heuristic reranking complete for {len(documents)} documents")
        return documents


if __name__ == "__main__":
    # Test reranker
    test_docs = [
        {
            "content": "Remote work policy allows 3 days per week with approval.",
            "source": "hr_policy.pdf",
            "chunk_id": "chunk_1",
            "score": 0.85,
            "final_score": 0.85
        },
        {
            "content": "Vacation policy states 15 days per year for full-time employees.",
            "source": "benefits.pdf",
            "chunk_id": "chunk_2",
            "score": 0.60,
            "final_score": 0.60
        }
    ]

    # Test heuristic reranking
    print("Testing Heuristic Reranking:")
    reranker = Reranker(use_llm=False)
    results = reranker.rerank("remote work policy", test_docs, top_k=5)

    for i, doc in enumerate(results, 1):
        print(f"\n{i}. Rerank Score: {doc['rerank_score']:.4f}")
        print(f"   Original Score: {doc['original_score']:.4f}")
        print(f"   Heuristic Score: {doc['heuristic_score']:.4f}")
        print(f"   Source: {doc['source']}")
        print(f"   Content: {doc['content'][:80]}...")
