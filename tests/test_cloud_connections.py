"""
Test script to verify Elasticsearch, Pinecone, and OpenAI connections
Run this after setting up cloud accounts to verify everything works
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def test_elasticsearch():
    """Test Elasticsearch connection"""
    print("\n" + "=" * 70)
    print("Testing Elasticsearch Connection")
    print("=" * 70)

    try:
        from elasticsearch import Elasticsearch

        elastic_url = os.getenv("ELASTIC_URL")

        if not elastic_url:
            print("❌ ELASTIC_URL not found in .env file")
            return False

        print(f"Connecting to: {elastic_url}")

        client = Elasticsearch([elastic_url], verify_certs=True)

        # Get cluster info
        info = client.info()

        print("✅ Elasticsearch connected successfully!")
        print(f"   Cluster name: {info['cluster_name']}")
        print(f"   Version: {info['version']['number']}")
        print(f"   Cluster UUID: {info['cluster_uuid'][:8]}...")

        return True

    except ImportError:
        print("❌ elasticsearch package not installed")
        print("   Run: pip install elasticsearch")
        return False
    except Exception as e:
        print(f"❌ Elasticsearch connection failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check ELASTIC_URL in .env includes https:// and port :9243")
        print("2. Verify deployment is 'Healthy' in Elastic Cloud dashboard")
        print("3. Check firewall settings")
        return False


def test_pinecone():
    """Test Pinecone connection"""
    print("\n" + "=" * 70)
    print("Testing Pinecone Connection")
    print("=" * 70)

    try:
        from pinecone import Pinecone

        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        pinecone_index = os.getenv("PINECONE_INDEX")

        if not pinecone_api_key:
            print("❌ PINECONE_API_KEY not found in .env file")
            return False

        if not pinecone_index:
            print("❌ PINECONE_INDEX not found in .env file")
            return False

        print(f"Connecting to Pinecone...")

        pc = Pinecone(api_key=pinecone_api_key)

        # List indexes
        indexes = pc.list_indexes()
        index_names = indexes.names()

        print("✅ Pinecone connected successfully!")
        print(f"   Available indexes: {index_names}")

        # Check if our index exists
        if pinecone_index in index_names:
            print(f"   ✅ Index '{pinecone_index}' found")

            # Get index stats
            index = pc.Index(pinecone_index)
            stats = index.describe_index_stats()
            print(f"   Vector count: {stats.get('total_vector_count', 0)}")
            print(f"   Dimension: {stats.get('dimension', 'N/A')}")
        else:
            print(f"   ⚠️  Index '{pinecone_index}' not found")
            print(f"   Available: {index_names}")
            print("\nYou need to create the index in Pinecone dashboard:")
            print("   Name: company-policies")
            print("   Dimensions: 1536")
            print("   Metric: cosine")

        return True

    except ImportError:
        print("❌ pinecone-client package not installed")
        print("   Run: pip install pinecone-client")
        return False
    except Exception as e:
        print(f"❌ Pinecone connection failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check PINECONE_API_KEY is correct")
        print("2. Verify API key is active in Pinecone dashboard")
        print("3. Ensure no extra spaces in .env file")
        return False


def test_openai():
    """Test OpenAI connection and embeddings"""
    print("\n" + "=" * 70)
    print("Testing OpenAI Connection")
    print("=" * 70)

    try:
        from openai import OpenAI

        openai_api_key = os.getenv("OPENAI_API_KEY")
        embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

        if not openai_api_key:
            print("❌ OPENAI_API_KEY not found in .env file")
            return False

        print(f"Connecting to OpenAI...")
        print(f"Using model: {embedding_model}")

        client = OpenAI(api_key=openai_api_key)

        # Test embedding generation
        response = client.embeddings.create(
            model=embedding_model, input="Test connection to OpenAI embeddings API"
        )

        embedding = response.data[0].embedding

        print("✅ OpenAI connected successfully!")
        print(f"   Model: {embedding_model}")
        print(f"   Embedding dimension: {len(embedding)}")
        print(f"   Sample values: [{embedding[0]:.4f}, {embedding[1]:.4f}, ...]")

        if len(embedding) != 1536:
            print(f"\n⚠️  WARNING: Expected 1536 dimensions, got {len(embedding)}")
            print("   Make sure EMBEDDING_MODEL=text-embedding-3-small")

        return True

    except ImportError:
        print("❌ openai package not installed")
        print("   Run: pip install openai")
        return False
    except Exception as e:
        print(f"❌ OpenAI connection failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check OPENAI_API_KEY is correct (starts with sk-)")
        print("2. Verify payment method is added to OpenAI account")
        print("3. Check API key has not been revoked")
        print("4. Ensure sufficient credits/quota available")
        return False


def test_all():
    """Run all connection tests"""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "CLOUD CONNECTIONS TEST SUITE" + " " * 25 + "║")
    print("╚" + "=" * 68 + "╝")

    # Check if .env file exists
    if not os.path.exists(".env"):
        print("\n❌ .env file not found!")
        print("Please create .env file with your credentials")
        return

    results = {
        "Elasticsearch": test_elasticsearch(),
        "Pinecone": test_pinecone(),
        "OpenAI": test_openai(),
    }

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    for service, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{service:<20} {status}")

    all_passed = all(results.values())

    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 All tests passed! Your cloud services are ready.")
        print("\nNext steps:")
        print("1. Add policy documents to kb_pipeline/data/raw/")
        print(
            "2. Run: python -m kb_pipeline.pipeline --mode index --data_dir kb_pipeline/data/raw"
        )
        print(
            "3. Test search: python -m kb_pipeline.pipeline --mode search --query 'your query'"
        )
    else:
        print("⚠️  Some tests failed. Please fix the issues above.")
        print("\nRefer to CLOUD_SETUP_GUIDE.md for detailed setup instructions")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    test_all()
