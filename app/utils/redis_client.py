# app/utils/redis_client.py

import redis.asyncio as redis
from app.config.settings import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_redis_client: redis.Redis | None = None


def get_redis() -> redis.Redis:
    global _redis_client
    if _redis_client is None:
        ssl_required = str(settings.redis_use_tls).lower() in ("1", "true", "yes")
        logger.info(
            f"Connecting to Redis Cloud at {settings.redis_host}:{settings.redis_port} (TLS={ssl_required})"
        )
        _redis_client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            username=settings.redis_username,
            password=settings.redis_password,
            db=settings.redis_db,
            decode_responses=True,
            ssl=ssl_required,
        )
    return _redis_client


async def close_redis():
    global _redis_client
    if _redis_client:
        try:
            await _redis_client.close()
            await _redis_client.connection_pool.disconnect()
        except Exception as e:
            logger.warning(f"Error closing Redis client: {e}")
        _redis_client = None
