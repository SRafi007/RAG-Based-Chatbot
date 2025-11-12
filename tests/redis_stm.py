#tests\redis_stm.py
import asyncio
from app.memory.short_term_memory import create_session, append_message, get_recent_messages

async def test():
    session_id = "test_session_1"
    await create_session(session_id, user_id="user007")
    await append_message(session_id, "user", "Hello, Redis Cloud!")
    await append_message(session_id, "assistant", "Hi there 👋 I'm connected successfully!")
    msgs = await get_recent_messages(session_id, limit=10)
    print(msgs)

asyncio.run(test())
