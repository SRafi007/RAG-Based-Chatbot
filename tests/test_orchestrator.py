# tests/test_orchestrator.py

import asyncio
from app.orchestrator.orchestrator import orchestrator


async def test_policy_query():
    """Test a policy-related query"""
    print("\n" + "=" * 60)
    print("TEST 1: Policy-Related Query")
    print("=" * 60)

    query = "What is the remote work policy?"
    print(f"\nQuery: {query}")

    result = await orchestrator.process(
        user_id="test_user", message=query, session_id="test_session_1", history=[]
    )

    print(f"\nClassification: {result['classification']}")
    print(f"Reply:\n{result['reply']}")
    print(f"Success: {result['success']}")


async def test_off_topic_query():
    """Test an off-topic query"""
    print("\n" + "=" * 60)
    print("TEST 2: Off-Topic Query")
    print("=" * 60)

    query = "Tell me a joke"
    print(f"\nQuery: {query}")

    result = await orchestrator.process(
        user_id="test_user", message=query, session_id="test_session_2", history=[]
    )

    print(f"\nClassification: {result['classification']}")
    print(f"Reply:\n{result['reply']}")
    print(f"Success: {result['success']}")


async def test_streaming():
    """Test streaming response"""
    print("\n" + "=" * 60)
    print("TEST 3: Streaming Response")
    print("=" * 60)

    query = "How many vacation days do I get?"
    print(f"\nQuery: {query}")
    print("\nStreaming response:")
    print("-" * 60)

    async for chunk in orchestrator.stream_process(
        user_id="test_user", message=query, session_id="test_session_3", history=[]
    ):
        print(chunk, end="", flush=True)

    print("\n" + "-" * 60)


async def main():
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  RAG-based Chatbot - Orchestrator Test Suite".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")

    try:
        # Test 1: Policy-related query
        await test_policy_query()

        # Test 2: Off-topic query
        await test_off_topic_query()

        # Test 3: Streaming
        await test_streaming()

        print("\n" + "=" * 60)
        print("✅ All tests completed successfully!")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
