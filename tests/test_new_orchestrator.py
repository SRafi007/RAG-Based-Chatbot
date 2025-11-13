# tests/test_new_orchestrator.py

import asyncio
from app.orchestrator import orchestrator


async def test_policy_query():
    """Test a policy-related query"""
    print("\n" + "=" * 70)
    print("TEST 1: Policy-Related Query")
    print("=" * 70)

    query = "What is the remote work policy?"
    print(f"\nQuery: {query}")

    result = await orchestrator.process(
        user_id="test_user",
        message=query,
        session_id="test_session_1",
    )

    print(f"\n{'Classification:':<20} {result['classification']}")
    print(f"{'Retrieved Docs:':<20} {result.get('retrieved_docs', 0)}")
    print(f"{'Success:':<20} {result['success']}")
    print(f"\n{'Reply:'}")
    print("-" * 70)
    print(result['reply'])
    print("-" * 70)


async def test_off_topic_query():
    """Test an off-topic query"""
    print("\n" + "=" * 70)
    print("TEST 2: Off-Topic Query")
    print("=" * 70)

    query = "Tell me a joke about programming"
    print(f"\nQuery: {query}")

    result = await orchestrator.process(
        user_id="test_user",
        message=query,
        session_id="test_session_2",
    )

    print(f"\n{'Classification:':<20} {result['classification']}")
    print(f"{'Success:':<20} {result['success']}")
    print(f"\n{'Reply:'}")
    print("-" * 70)
    print(result['reply'])
    print("-" * 70)


async def test_streaming():
    """Test streaming response"""
    print("\n" + "=" * 70)
    print("TEST 3: Streaming Response")
    print("=" * 70)

    query = "How many vacation days do I get?"
    print(f"\nQuery: {query}")
    print("\nStreaming response:")
    print("-" * 70)

    async for chunk in orchestrator.stream_process(
        user_id="test_user",
        message=query,
        session_id="test_session_3",
    ):
        print(chunk, end="", flush=True)

    print("\n" + "-" * 70)


async def test_multi_turn_conversation():
    """Test multi-turn conversation with context"""
    print("\n" + "=" * 70)
    print("TEST 4: Multi-Turn Conversation")
    print("=" * 70)

    session_id = "test_session_4"
    queries = [
        "What is the sick leave policy?",
        "How do I apply for it?",
        "What documentation is required?",
    ]

    for i, query in enumerate(queries, 1):
        print(f"\nTurn {i}: {query}")

        result = await orchestrator.process(
            user_id="test_user",
            message=query,
            session_id=session_id,
        )

        print(f"Reply: {result['reply'][:150]}...")
        print()


async def main():
    print("\n")
    print("=" * 68)
    print(" RAG-based Chatbot - LangGraph Orchestrator Test Suite".center(68))
    print("=" * 68)

    try:
        # Test 1: Policy-related query
        await test_policy_query()

        # Test 2: Off-topic query
        await test_off_topic_query()

        # Test 3: Streaming
        await test_streaming()

        # Test 4: Multi-turn conversation
        await test_multi_turn_conversation()

        print("\n" + "=" * 70)
        print("ALL TESTS COMPLETED SUCCESSFULLY!".center(70))
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"\nTEST FAILED with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
