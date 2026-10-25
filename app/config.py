from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime configuration comes from environment variables (or a local .env file)."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "banking-loyalty-assistant"
    environment: str = "local"
    log_level: str = "INFO"

    # Postgres (asyncpg driver). Alembic reuses the same URL.
    database_url: str = "postgresql+asyncpg://assistant:assistant@localhost:5432/assistant"
    db_pool_size: int = 10
    db_max_overflow: int = 10

    redis_url: str = "redis://localhost:6379/0"
    # Query embeddings are cached in Redis; 0 disables the cache.
    embedding_cache_ttl_seconds: int = 7 * 24 * 3600

    # Embeddings. The vector column is fixed at embedding_dim in the migration, so changing
    # the model to one with a different dimension needs a new migration plus re-ingestion.
    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_dim: int = 384
    embedding_batch_size: int = 32
    embedding_device: str = "cpu"

    # Chunking (characters, not tokens: e5-small truncates at 512 tokens, ~1500+ chars).
    chunk_size: int = 800
    chunk_overlap: int = 120

    # Hybrid search
    search_candidates: int = 40  # rows pulled from each retriever before fusion
    rrf_k: int = 60  # reciprocal rank fusion constant
    search_top_k: int = 5

    max_upload_mb: int = 20
    default_tenant: str = "demo-bank"


@lru_cache
def get_settings() -> Settings:
    return Settings()
