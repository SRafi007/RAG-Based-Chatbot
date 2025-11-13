"""
Test Elasticsearch Connection
Verifies Elasticsearch setup with API key
"""

import os
from dotenv import load_dotenv

# Load environment variables (force override any cached values)
load_dotenv(override=True)


def test_elasticsearch_connection():
    """Test Elasticsearch connection with API key"""
    print("\n" + "=" * 70)
    print("Testing Elasticsearch Connection")
    print("=" * 70)

    try:
        from elasticsearch import Elasticsearch

        elastic_url = os.getenv("ELASTIC_URL")
        elastic_api_key = os.getenv("ELASTIC_API_KEY")
        elastic_index = os.getenv("ELASTIC_INDEX", "company_policies")

        if not elastic_url:
            print("[FAIL] ELASTIC_URL not found in .env")
            return False

        if not elastic_api_key:
            print("[FAIL] ELASTIC_API_KEY not found in .env")
            return False

        print(f"Connecting to: {elastic_url}")
        print(f"Index: {elastic_index}")

        # Connect with API key
        client = Elasticsearch(
            [elastic_url], api_key=elastic_api_key, verify_certs=True
        )

        # Test connection
        info = client.info()

        print("\n[OK] Elasticsearch connected successfully!")
        print(f"[OK] Cluster name: {info['cluster_name']}")
        print(f"[OK] Version: {info['version']['number']}")

        # Check if index exists
        if client.indices.exists(index=elastic_index):
            print(f"[OK] Index '{elastic_index}' exists")

            # Get index stats
            stats = client.count(index=elastic_index)
            print(f"[OK] Document count: {stats['count']}")
        else:
            print(
                f"[INFO] Index '{elastic_index}' does not exist yet (will be created)"
            )

        return True

    except Exception as e:
        print(f"[FAIL] Elasticsearch connection failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check ELASTIC_URL in .env")
        print("2. Verify ELASTIC_API_KEY is correct")
        print("3. Check internet connection")
        print("4. Verify API key has proper permissions")
        return False


def test_sparse_indexer():
    """Test SparseIndexer class"""
    print("\n" + "=" * 70)
    print("Testing SparseIndexer Class")
    print("=" * 70)

    try:
        from kb_pipeline.indexing.index_sparse import SparseIndexer

        print("Initializing SparseIndexer...")
        indexer = SparseIndexer()

        print(f"[OK] SparseIndexer initialized")
        print(f"[OK] Index name: {indexer.index_name}")
        print(f"[OK] Connected to Elasticsearch")

        # Test indexing a document
        test_chunk = {
            "id": "test_elastic_001",
            "text": "## Test Policy\n\nThis is a test policy document for Elasticsearch.",
            "metadata": {
                "section": "Test Policy",
                "policy_type": "Test",
                "source_file": "test.md",
                "chunk_index": 0,
                "tokens": 15,
            },
        }

        print("\nIndexing test document...")
        indexed = indexer.index_documents([test_chunk])
        print(f"[OK] Indexed {indexed} document(s)")

        # Test search
        print("\nSearching for 'test policy'...")
        results = indexer.search("test policy", top_k=3)
        print(f"[OK] Found {len(results)} result(s)")

        if results:
            for i, result in enumerate(results, 1):
                print(f"\n  Result {i}:")
                print(f"    Section: {result['section']}")
                print(f"    Score: {result['score']:.4f}")
                print(f"    Content: {result['content'][:60]}...")

        return True

    except Exception as e:
        print(f"[FAIL] SparseIndexer test failed: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure Elasticsearch connection test passed")
        print("2. Check API key has index/write permissions")
        print("3. Review logs for detailed errors")
        return False


def main():
    """Run all Elasticsearch tests"""
    print("\n")
    print("+" + "=" * 68 + "+")
    print("|" + " " * 15 + "ELASTICSEARCH CONNECTION TEST" + " " * 23 + "|")
    print("+" + "=" * 68 + "+")

    # Check .env file
    if not os.path.exists(".env"):
        print("\n[FAIL] .env file not found!")
        print("Please create .env file with Elasticsearch credentials")
        return

    results = {
        "Elasticsearch Connection": test_elasticsearch_connection(),
    }

    # Only test SparseIndexer if connection worked
    if results["Elasticsearch Connection"]:
        results["SparseIndexer"] = test_sparse_indexer()

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    for test_name, passed in results.items():
        status = "[OK] PASSED" if passed else "[FAIL] FAILED"
        print(f"{test_name:<35} {status}")

    all_passed = all(results.values())

    print("\n" + "=" * 70)
    if all_passed:
        print("All tests passed! Elasticsearch is ready!")
        print("\nYour setup:")
        print("  - Elasticsearch: Connected with API key")
        print("  - Index: company_policies")
        print("  - Ready for: BM25 keyword search (sparse retrieval)")
        print("\nNext: Set up Pinecone for dense retrieval")
    else:
        print("WARNING: Some tests failed. Fix issues above before proceeding.")
        print("\nRefer to:")
        print("- .env file configuration")
        print("- CLOUD_SETUP_GUIDE.md for help")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
