# kb_pipeline/pipeline.py

"""
Main pipeline orchestrator for the knowledge base.

Usage:
    # Index documents
    python -m kb_pipeline.pipeline --mode index --data_dir data/raw

    # Search
    python -m kb_pipeline.pipeline --mode search --query "What is the remote work policy?"
"""

import argparse
from pathlib import Path
from kb_pipeline.data.ingest import DocumentIngester
from kb_pipeline.data.preprocess import DocumentPreprocessor
from kb_pipeline.indexing.index_sparse import SparseIndexer
from kb_pipeline.indexing.index_dense import DenseIndexer
from kb_pipeline.retrieval.hybrid_retriever import HybridRetriever
from kb_pipeline.retrieval.reranker import Reranker
from app.utils.logger import get_logger

logger = get_logger(__name__)


class KnowledgeBasePipeline:
    """
    Complete pipeline for building and querying the knowledge base.
    """

    def __init__(self):
        """Initialize pipeline components."""
        logger.info("Initializing Knowledge Base Pipeline")

        # Data processing
        self.ingester = DocumentIngester()
        self.preprocessor = DocumentPreprocessor(
            chunk_size=512,
            chunk_overlap=128,
            min_chunk_size=100
        )

        # Indexing
        self.sparse_indexer = SparseIndexer()
        self.dense_indexer = DenseIndexer()

        # Retrieval
        self.retriever = HybridRetriever(
            sparse_weight=0.5,
            dense_weight=0.5
        )
        self.reranker = Reranker(use_llm=True)

        logger.info("Pipeline initialized successfully")

    def build_index(self, data_dir: str = "data/raw"):
        """
        Build the knowledge base index from documents.

        Args:
            data_dir: Directory containing raw documents

        Returns:
            Number of indexed chunks
        """
        logger.info(f"Starting index build from {data_dir}")

        # Step 1: Ingest documents
        logger.info("Step 1/4: Ingesting documents...")
        self.ingester.data_dir = Path(data_dir)
        documents = self.ingester.ingest_all()

        if not documents:
            logger.error("No documents found to index!")
            return 0

        # Step 2: Preprocess documents
        logger.info("Step 2/4: Preprocessing documents...")
        chunks = self.preprocessor.preprocess(documents)

        if not chunks:
            logger.error("No chunks created after preprocessing!")
            return 0

        # Step 3: Index into Elasticsearch (sparse)
        logger.info("Step 3/4: Indexing into Elasticsearch (sparse)...")
        sparse_indexed = self.sparse_indexer.index_documents(chunks)

        # Step 4: Index into Pinecone (dense)
        logger.info("Step 4/4: Indexing into Pinecone (dense)...")
        dense_indexed = self.dense_indexer.index_documents(chunks)

        logger.info(
            f"Index build complete! "
            f"Sparse: {sparse_indexed} chunks, Dense: {dense_indexed} chunks"
        )

        return min(sparse_indexed, dense_indexed)

    def search(
        self,
        query: str,
        top_k: int = 5,
        use_reranking: bool = True
    ):
        """
        Search the knowledge base.

        Args:
            query: Search query
            top_k: Number of results to return
            use_reranking: Whether to use LLM reranking

        Returns:
            List of search results
        """
        logger.info(f"Searching for: {query}")

        # Retrieve with hybrid search
        results = self.retriever.retrieve(query, top_k=top_k * 2)

        if not results:
            logger.warning("No results found")
            return []

        # Rerank if enabled
        if use_reranking:
            results = self.reranker.rerank(query, results, top_k=top_k)

        logger.info(f"Returning {len(results)} results")
        return results

    def format_results(self, results: list) -> str:
        """
        Format search results for display.

        Args:
            results: List of search results

        Returns:
            Formatted string
        """
        if not results:
            return "No results found."

        output = []
        for i, result in enumerate(results, 1):
            output.append(f"\n{'='*70}")
            output.append(f"Result {i}")
            output.append(f"{'='*70}")
            output.append(f"Source: {result['source']}")
            output.append(f"Retrieval Method: {result.get('retrieval_method', 'unknown')}")

            # Scores
            if 'rerank_score' in result:
                output.append(f"Rerank Score: {result['rerank_score']:.4f}")
            if 'final_score' in result:
                output.append(f"Hybrid Score: {result['final_score']:.4f}")

            output.append(f"\nContent:")
            output.append(f"{result['content']}")

        return "\n".join(output)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Knowledge Base Pipeline")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["index", "search"],
        required=True,
        help="Mode: index or search"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/raw",
        help="Directory containing documents (for index mode)"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Search query (for search mode)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of results to return"
    )
    parser.add_argument(
        "--no_rerank",
        action="store_true",
        help="Disable reranking"
    )

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = KnowledgeBasePipeline()

    if args.mode == "index":
        # Build index
        indexed = pipeline.build_index(args.data_dir)
        print(f"\n✅ Indexed {indexed} chunks successfully!")

    elif args.mode == "search":
        if not args.query:
            print("Error: --query is required for search mode")
            return

        # Search
        results = pipeline.search(
            args.query,
            top_k=args.top_k,
            use_reranking=not args.no_rerank
        )

        # Display results
        print(pipeline.format_results(results))


if __name__ == "__main__":
    main()
