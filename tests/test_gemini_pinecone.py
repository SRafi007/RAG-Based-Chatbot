"""
Test Gemini Embeddings with Pinecone
Verifies the FREE embedding setup is working correctly
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def test_gemini_embeddings():
    """Test Gemini embedding generation"""
    print("\n" + "=" * 70)
    print("Testing Gemini Embeddings (FREE!)")
    print("=" * 70)

    try:
        from app.utils.gemini_embeddings import GeminiEmbeddingClient

        client = GeminiEmbeddingClient()

        # Test document embedding
        doc_text = "Employees can work remotely up to 3 days per week"
        doc_embedding = client.get_document_embedding(doc_text)

        print(f"[OK] Model: {client.model_name}")
        print(f"[OK] Dimensions: {client.dimension}")
        print(f"[OK] Document embedding generated: {len(doc_embedding)} dimensions")

        # Test query embedding
        query_text = "remote work policy"
        query_embedding = client.get_query_embedding(query_text)
        print(f"[OK] Query embedding generated: {len(query_embedding)} dimensions")

        # Test batch embeddings
        texts = ["Policy 1", "Policy 2", "Policy 3"]
        batch_embeddings = client.get_embeddings_batch(texts)
        print(f"[OK] Batch embeddings: {len(batch_embeddings)} embeddings")

        return True

    except Exception as e:
        print(f"[FAIL] Gemini embeddings failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check GEMINI_API_KEY in .env")
        print("2. Verify EMBEDDING_MODEL=models/embedding-001")
        print("3. Check internet connection")
        return False


def test_pinecone_connection():
    """Test Pinecone connection"""
    print("\n" + "=" * 70)
    print("Testing Pinecone Connection")
    print("=" * 70)

    try:
        from pinecone import Pinecone

        api_key = os.getenv("PINECONE_API_KEY")
        host = os.getenv("PINECONE_HOST")
        index_name = os.getenv("PINECONE_INDEX")

        if not api_key:
            print("[FAIL] PINECONE_API_KEY not found in .env")
            return False

        if not host:
            print("[FAIL] PINECONE_HOST not found in .env")
            return False

        print(f"Connecting to Pinecone...")
        print(f"  Index: {index_name}")
        print(f"  Host: {host}")

        pc = Pinecone(api_key=api_key)

        # Connect to index using host
        index = pc.Index(name=index_name, host=host)

        # Get index stats
        stats = index.describe_index_stats()

        print(f"[OK] Connected to Pinecone!")
        print(f"[OK] Total vectors: {stats.get('total_vector_count', 0)}")
        print(f"[OK] Dimension: {stats.get('dimension', 'N/A')}")

        return True

    except Exception as e:
        print(f"[FAIL] Pinecone connection failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check PINECONE_API_KEY in .env")
        print("2. Verify PINECONE_HOST includes full URL")
        print("3. Check PINECONE_INDEX name matches your cloud index")
        return False


def test_full_pipeline():
    """Test full pipeline: Embed + Index + Search"""
    print("\n" + "=" * 70)
    print("Testing Full Pipeline (Gemini + Pinecone)")
    print("=" * 70)

    try:
        from kb_pipeline.indexing.index_dense import DenseIndexer

        # Create test chunk
        test_chunk = {
            "id": "test_gemini_001",
            "text": "Our remote work policy allows employees to work from home up to 3 days per week with manager approval.",
            "metadata": {
                "section": "Remote Work Policy",
                "policy_type": "HR Policy",
                "source_file": "test_policy.md",
                "chunk_index": 0,
                "tokens": 25,
            },
        }

        print("Initializing DenseIndexer...")
        indexer = DenseIndexer()

        print(f"[OK] Model: {indexer.embedding_model}")
        print(f"[OK] Dimensions: {indexer.embedding_dimension}")

        print("\nIndexing test document...")
        indexed = indexer.index_documents([test_chunk])
        print(f"[OK] Indexed {indexed} document(s)")

        print("\nSearching for 'remote work'...")
        results = indexer.search("remote work policy", top_k=3)
        print(f"[OK] Found {len(results)} result(s)")

        if results:
            for i, result in enumerate(results, 1):
                print(f"\n  Result {i}:")
                print(f"    Section: {result['section']}")
                print(f"    Score: {result['score']:.4f}")
                print(f"    Content: {result['content'][:80]}...")

        return True

    except Exception as e:
        print(f"[FAIL] Full pipeline test failed: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure Gemini embeddings test passed")
        print("2. Make sure Pinecone connection test passed")
        print("3. Check logs for detailed error messages")
        return False


def main():
    """Run all tests"""
    print("\n")
    print("+" + "=" * 68 + "+")
    print("|" + " " * 10 + "GEMINI EMBEDDINGS + PINECONE TEST SUITE" + " " * 18 + "|")
    print("+" + "=" * 68 + "+")

    # Check .env file
    if not os.path.exists(".env"):
        print("\n[FAIL] .env file not found!")
        print("Please create .env file with required credentials")
        return

    results = {
        "Gemini Embeddings": test_gemini_embeddings(),
        "Pinecone Connection": test_pinecone_connection(),
    }

    # Only run full pipeline if both tests passed
    if all(results.values()):
        results["Full Pipeline"] = test_full_pipeline()

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    for test_name, passed in results.items():
        status = "[OK] PASSED" if passed else "[FAIL] FAILED"
        print(f"{test_name:<30} {status}")

    all_passed = all(results.values())

    print("\n" + "=" * 70)
    if all_passed:
        print("All tests passed! Gemini embeddings + Pinecone ready!")
        print("\nNext steps:")
        print("1. Add documents to kb_pipeline/data/raw/")
        print("2. Run: python -m kb_pipeline.pipeline --mode index")
        print("3. Test search: python -m kb_pipeline.pipeline --mode search")
    else:
        print("WARNING: Some tests failed. Fix issues above before proceeding.")
        print("\nRefer to:")
        print("- .env file configuration")
        print("- CLOUD_SETUP_GUIDE.md for setup help")
        print("- FREE_EMBEDDING_ALTERNATIVES.md for embedding info")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
