# tests/test_llm.py

import asyncio
from app.utils.llm_client import gemini_client


async def test_stream():
    """Test streaming generation"""
    print("\n🔄 Testing stream_generate()...")
    print("Prompt: Tell me a short joke\n")

    full_response = ""
    async for chunk in gemini_client.stream_generate("Tell me a short joke"):
        print(chunk, end="", flush=True)
        full_response += chunk

    print("\n\n✅ Streaming test completed")
    print(f"Full response length: {len(full_response)} characters")


def test_generate():
    """Test standard generation"""
    print("\n🧩 Testing generate()...")
    print("Prompt: What is Python?\n")

    result = gemini_client.generate("What is Python?")
    print(f"Response: {result[:200]}...")
    print(f"\n✅ Generate test completed")
    print(f"Response length: {len(result)} characters")


async def main():
    print("=" * 60)
    print("🧠 GeminiClient Test Suite")
    print("=" * 60)

    # Test 1: Standard generation
    test_generate()

    # Test 2: Streaming generation
    await test_stream()

    print("\n" + "=" * 60)
    print("✅ All tests completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
