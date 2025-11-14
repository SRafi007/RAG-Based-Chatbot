# app/orchestrator/agents/retriever_agent.py

from typing import List, Dict, Optional
from app.orchestrator.state import AgentState
from app.utils.logger import get_logger

logger = get_logger(__name__)


class RetrieverAgent:
    """
    Retriever Agent - Hybrid RAG Document Retriever

    LangGraph Node: Retrieves relevant company policy documents using hybrid retrieval.
    Combines sparse (BM25) and dense (semantic) search for optimal results.
    """

    def __init__(self, use_hybrid: bool = True):
        """
        Initialize the Retriever Agent.

        Args:
            use_hybrid: Whether to use hybrid retrieval (True) or placeholder (False)
        """
        self.use_hybrid = use_hybrid
        self.retriever = None
        self.reranker = None

        # Initialize hybrid retriever if enabled
        if use_hybrid:
            try:
                from kb_pipeline.retrieval import HybridRetriever, Reranker

                self.retriever = HybridRetriever(sparse_weight=0.5, dense_weight=0.5)
                self.reranker = Reranker(use_llm=True)
                logger.info("RetrieverAgent initialized with HybridRetriever")
            except Exception as e:
                logger.warning(
                    f"Failed to initialize HybridRetriever: {e}. Using placeholder mode."
                )
                self.use_hybrid = False
        else:
            logger.info("RetrieverAgent initialized in placeholder mode")

    def __call__(self, state: AgentState) -> AgentState:
        """
        LangGraph node execution.

        Args:
            state: Current agent state

        Returns:
            Updated state with retrieved documents and formatted context
        """
        try:
            query = state["message"]
            logger.info(f"RetrieverAgent: Retrieving documents for query: {query[:50]}...")

            # Retrieve documents
            retrieved_docs = self._retrieve(query, top_k=3)

            # Format context
            context = self._format_context(retrieved_docs)

            # Update state
            state["retrieved_docs"] = retrieved_docs
            state["context"] = context

            logger.info(f"RetrieverAgent: Retrieved {len(retrieved_docs)} documents")
            return state

        except Exception as e:
            logger.error(f"RetrieverAgent error: {e}")
            state["retrieved_docs"] = []
            state["context"] = "No relevant policy documents found."
            return state

    def _retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, str]]:
        """
        Retrieve relevant documents for the given query using hybrid search.

        Args:
            query: User's question
            top_k: Number of top relevant documents to retrieve

        Returns:
            List of retrieved document chunks with metadata
        """
        if self.use_hybrid and self.retriever:
            try:
                # Hybrid retrieval: Get more results for reranking
                results = self.retriever.retrieve(query, top_k=top_k * 2)

                # Rerank results using LLM
                if self.reranker and results:
                    results = self.reranker.rerank(query, results, top_k=top_k)

                logger.info(f"Hybrid retrieval returned {len(results)} documents")
                return results

            except Exception as e:
                logger.error(f"Hybrid retrieval failed: {e}. Using placeholder.")

        # Fallback: Placeholder implementation
        logger.warning("Using placeholder retrieval")
        return [
            {
                "content": "Company Remote Work Policy: Employees may work remotely up to 3 days per week with manager approval. Full-time remote work requires VP approval.",
                "source": "HR_Policy_Handbook_2024.pdf",
                "score": 0.95,
            },
            {
                "content": "Vacation Policy: Full-time employees receive 15 days of paid vacation per year. Part-time employees receive prorated vacation days.",
                "source": "Employee_Benefits_Guide.pdf",
                "score": 0.88,
            },
            {
                "content": "Sick Leave Policy: Employees are entitled to 10 days of sick leave annually. Medical documentation required for absences exceeding 3 consecutive days.",
                "source": "HR_Policy_Handbook_2024.pdf",
                "score": 0.82,
            },
        ]

    def _format_context(self, retrieved_docs: List[Dict[str, str]]) -> str:
        """
        Format retrieved documents into a context string for the LLM.

        Args:
            retrieved_docs: List of retrieved documents

        Returns:
            Formatted context string
        """
        if not retrieved_docs:
            return "No relevant policy documents found."

        context_parts = []
        for idx, doc in enumerate(retrieved_docs, 1):
            context_parts.append(
                f"[Document {idx}] (Source: {doc.get('source', 'Unknown')}, Relevance: {doc.get('score', 0):.2f})\n{doc['content']}\n"
            )

        return "\n---\n".join(context_parts)

    def enable_hybrid_mode(self, sparse_weight: float = 0.5, dense_weight: float = 0.5):
        """
        Enable or reconfigure hybrid retrieval mode.

        Args:
            sparse_weight: Weight for sparse (BM25) results
            dense_weight: Weight for dense (semantic) results
        """
        try:
            from kb_pipeline.retrieval import HybridRetriever, Reranker

            self.retriever = HybridRetriever(
                sparse_weight=sparse_weight, dense_weight=dense_weight
            )
            self.reranker = Reranker(use_llm=True)
            self.use_hybrid = True
            logger.info(
                f"Hybrid mode enabled (sparse: {sparse_weight}, dense: {dense_weight})"
            )
        except Exception as e:
            logger.error(f"Failed to enable hybrid mode: {e}")
            self.use_hybrid = False

    def disable_hybrid_mode(self):
        """Disable hybrid retrieval and use placeholder mode."""
        self.use_hybrid = False
        self.retriever = None
        self.reranker = None
        logger.info("Hybrid mode disabled, using placeholder retrieval")


# Singleton instance - hybrid mode enabled (Pinecone sparse + dense indexed)
retriever_agent = RetrieverAgent(use_hybrid=True)
