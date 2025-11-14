# app/memory/short_term_memory.py

import json
import time
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from app.utils.redis_client import get_redis
from app.config.settings import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Config values
MAX_MESSAGES = settings.max_session_messages
SESSION_TTL_SECONDS = settings.session_ttl_days * 24 * 3600

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _now_ts() -> float:
    return time.time()

def _messages_key(session_id: str) -> str:
    return f"session:{session_id}:messages"

def _meta_key(session_id: str) -> str:
    return f"session:{session_id}:meta"

ACTIVE_SESSIONS_KEY = "sessions:active"


async def create_session(session_id: str, user_id: str):
    """Initialize a new Redis chat session."""
    r = get_redis()
    now_iso = _now_iso()
    now_ts = _now_ts()
    pipe = r.pipeline()

    pipe.hset(_meta_key(session_id), mapping={
        "user_id": user_id,
        "created_at": now_iso,
        "last_activity": str(now_ts),
        "nb_messages": "0"
    })
    pipe.zadd(ACTIVE_SESSIONS_KEY, {session_id: now_ts})
    pipe.expire(_meta_key(session_id), SESSION_TTL_SECONDS)
    pipe.expire(ACTIVE_SESSIONS_KEY, SESSION_TTL_SECONDS)
    await pipe.execute()
    logger.info(f"[Memory] Created session {session_id} for user {user_id}")


async def append_message(session_id: str, role: str, text: str):
    """Store a new chat message."""
    r = get_redis()
    now_iso = _now_iso()
    now_ts = _now_ts()

    msg = json.dumps({"role": role, "text": text, "ts": now_iso})

    pipe = r.pipeline()
    pipe.lpush(_messages_key(session_id), msg)
    pipe.ltrim(_messages_key(session_id), 0, MAX_MESSAGES - 1)
    pipe.hincrby(_meta_key(session_id), "nb_messages", 1)
    pipe.hset(_meta_key(session_id), "last_activity", str(now_ts))
    pipe.zadd(ACTIVE_SESSIONS_KEY, {session_id: now_ts})
    pipe.expire(_messages_key(session_id), SESSION_TTL_SECONDS)
    pipe.expire(_meta_key(session_id), SESSION_TTL_SECONDS)
    await pipe.execute()


async def get_recent_messages(session_id: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Fetch the most recent N messages."""
    r = get_redis()
    data = await r.lrange(_messages_key(session_id), 0, limit - 1)
    messages = [json.loads(m) for m in reversed(data)]
    return messages


async def get_session_meta(session_id: str) -> Dict[str, Any]:
    """Return metadata for a given session."""
    r = get_redis()
    return await r.hgetall(_meta_key(session_id))


async def delete_session(session_id: str):
    """Delete all Redis keys for a given session."""
    r = get_redis()
    pipe = r.pipeline()
    pipe.delete(_messages_key(session_id))
    pipe.delete(_meta_key(session_id))
    pipe.zrem(ACTIVE_SESSIONS_KEY, session_id)
    await pipe.execute()
    logger.info(f"[Memory] Deleted session {session_id}")
