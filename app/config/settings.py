# app/config/settings.py

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    # App
    app_name: str = Field(default="RAG-based Chatbot")
    app_env: str = Field(default="development")
    app_port: int = Field(default=8000)

    # PostgreSQL (Supabase)
    database_url: str

    # Gemini API
    gemini_api_key: str
    gemini_model_name: str = "gemini-2.5-flash"
    gemini_thinking_budget: int = -1
    gemini_image_size: str = "1K"

    # Logging
    log_level: str = Field(default="INFO")
    log_file: str = Field(default="logs/app_logs.log")

    # Redis Cloud
    redis_host: str
    redis_port: int
    redis_username: str
    redis_password: str
    redis_db: int = Field(default=0)
    redis_use_tls: bool = Field(default=False)

    # Memory config
    max_session_messages: int = 200
    session_ttl_days: int = 30

    # Pinecone Configuration
    pinecone_api_key: str

    # Pinecone Dense (Semantic Vector)
    pinecone_dense_host: str
    pinecone_dense_index: str = Field(default="company-policies")

    # Pinecone Sparse (BM25 Keyword)
    pinecone_sparse_host: str
    pinecone_sparse_index: str = Field(default="company-policies-sparse")

    # Local Embeddings (sentence-transformers - FREE & UNLIMITED!)
    embedding_model: str = Field(default="all-mpnet-base-v2")
    embedding_dimension: int = Field(default=768)

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


# Singleton instance
settings = Settings()
