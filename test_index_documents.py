# test_index_documents.py

"""
Simple script to index documents directly without the pipeline orchestrator.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from kb_pipeline.data.ingest import DocumentIngester
from kb_pipeline.preprocessor.preprocess import DocumentPreprocessor
from kb_pipeline.indexing.index_sparse import SparseIndexer
from kb_pipeline.indexing.index_dense import DenseIndexer


def main():
    print("="*70)
    print("KB Pipeline - Document Indexing")
    print("="*70)
    print()

    # Step 1: Ingest documents
    print("Step 1/4: Ingesting documents...")
    ingester = DocumentIngester(data_dir="kb_pipeline/data/raw")
    documents = ingester.ingest_all()

    if not documents:
        print("[FAIL] No documents found!")
        return

    print(f"[OK] Ingested {len(documents)} documents\n")

    # Step 2: Preprocess (semantic chunking)
    print("Step 2/4: Preprocessing with semantic chunking...")
    preprocessor = DocumentPreprocessor(
        target_tokens=350,
        max_tokens=450,
        overlap_tokens=50,
        min_tokens=50
    )

    chunks = preprocessor.preprocess(documents)

    if not chunks:
        print("[FAIL] No chunks created!")
        return

    print(f"[OK] Created {len(chunks)} semantic chunks\n")

    # Step 3: Index to Elasticsearch (sparse)
    print("Step 3/4: Indexing to Elasticsearch (BM25)...")
    try:
        sparse_indexer = SparseIndexer()
        sparse_count = sparse_indexer.index_documents(chunks)
        print(f"[OK] Indexed {sparse_count} chunks to Elasticsearch\n")
    except Exception as e:
        print(f"[FAIL] Elasticsearch indexing failed: {e}\n")
        sparse_count = 0

    # Step 4: Index to Pinecone (dense with all-mpnet-base-v2)
    print("Step 4/4: Indexing to Pinecone (all-mpnet-base-v2, 768D)...")
    try:
        dense_indexer = DenseIndexer()
        print(f"Model: {dense_indexer.embedding_model}")
        print(f"Dimensions: {dense_indexer.embedding_dimension}")
        dense_count = dense_indexer.index_documents(chunks, batch_size=10)
        print(f"[OK] Indexed {dense_count} chunks to Pinecone\n")
    except Exception as e:
        print(f"[FAIL] Pinecone indexing failed: {e}\n")
        dense_count = 0

    # Summary
    print("="*70)
    print("Indexing Complete!")
    print("="*70)
    print(f"Documents processed: {len(documents)}")
    print(f"Semantic chunks created: {len(chunks)}")
    print(f"Elasticsearch (sparse): {sparse_count} chunks")
    print(f"Pinecone (dense): {dense_count} chunks")
    print("="*70)


if __name__ == "__main__":
    main()
