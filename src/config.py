"""Application configuration loaded from environment variables."""
from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Environment ────────────────────────────────────────
    environment: str = Field(default="development", alias="ENVIRONMENT")
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    log_format: str = Field(default="json", alias="LOG_FORMAT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # ── Required secrets ──────────────────────────────────
    inception_api_key: Optional[str] = Field(default=None, alias="INCEPTION_API_KEY")
    jwt_secret_key: Optional[str] = Field(default=None, alias="JWT_SECRET_KEY")
    database_url: Optional[str] = Field(default=None, alias="DATABASE_URL")

    # ── LLM ───────────────────────────────────────────────
    llm_api_key: Optional[str] = Field(default=None, alias="LLM_API_KEY")
    llm_base_url: Optional[str] = Field(default=None, alias="LLM_BASE_URL")
    llm_model: str = Field(default="gpt-4o-mini", alias="LLM_MODEL")
    inception_api_key: Optional[str] = Field(default=None, alias="INCEPTION_API_KEY")
    inception_base_url: str = Field(
        default="https://api.inceptionlabs.ai/v1", alias="INCEPTION_BASE_URL"
    )
    mercury_model: str = Field(default="mercury-2", alias="MERCURY_MODEL")
    llm_max_tokens: int = Field(default=2048, alias="LLM_MAX_TOKENS")
    llm_temperature: float = Field(default=0.0, alias="LLM_TEMPERATURE")
    skip_llm_prewarm: bool = Field(default=False, alias="SKIP_LLM_PREWARM")

    # ── Cloudinary ────────────────────────────────────────
    cloudinary_cloud_name: Optional[str] = Field(default=None, alias="CLOUDINARY_CLOUD_NAME")
    cloudinary_api_key: Optional[str] = Field(default=None, alias="CLOUDINARY_API_KEY")
    cloudinary_api_secret: Optional[str] = Field(default=None, alias="CLOUDINARY_API_SECRET")

    # ── Auth ──────────────────────────────────────────────
    jwt_expire_minutes: int = Field(default=60, alias="JWT_EXPIRE_MINUTES")
    jwt_refresh_threshold: int = Field(default=1440, alias="JWT_REFRESH_THRESHOLD")
    github_client_id: str = Field(default="", alias="GITHUB_CLIENT_ID")
    github_client_secret: str = Field(default="", alias="GITHUB_CLIENT_SECRET")

    # ── CORS ──────────────────────────────────────────────
    allowed_origins: str = Field(
        default=(
            "http://localhost:3000,http://localhost:3001,http://localhost:5173,"
            "http://localhost:8000,http://127.0.0.1:3000,http://127.0.0.1:3001,"
            "http://127.0.0.1:5173,http://127.0.0.1:8000"
        ),
        alias="ALLOWED_ORIGINS",
    )

    # ── Optional integrations ─────────────────────────────
    redis_url: str = Field(default="", alias="REDIS_URL")
    pii_enabled: bool = Field(default=False, alias="PII_ENABLED")
    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    qdrant_api_key: Optional[str] = Field(default=None, alias="QDRANT_API_KEY")

    # ── RAG pipeline ──────────────────────────────────────
    max_document_hops: int = Field(default=3, alias="MAX_DOCUMENT_HOPS")
    max_citations_per_doc: int = Field(default=6, alias="MAX_CITATIONS_PER_DOC")
    max_citations_total: int = Field(default=20, alias="MAX_CITATIONS_TOTAL")
    retrieve_k_per_query: int = Field(default=10, alias="RETRIEVE_K_PER_QUERY")

    # ── Database ──────────────────────────────────────────
    db_pool_size: int = Field(default=5, alias="DB_POOL_SIZE")
    db_max_overflow: int = Field(default=5, alias="DB_MAX_OVERFLOW")

    # ── Background jobs ───────────────────────────────────
    job_processor_timeout: float = Field(default=300.0, alias="JOB_PROCESSOR_TIMEOUT")
    job_poll_interval: float = Field(default=1.0, alias="JOB_POLL_INTERVAL")

    # ── Shutdown ──────────────────────────────────────────
    shutdown_drain_timeout: float = Field(default=30.0, alias="SHUTDOWN_DRAIN_TIMEOUT")
    disable_signal_handlers: bool = Field(default=False, alias="DISABLE_SIGNAL_HANDLERS")

    # ── Cache TTLs ────────────────────────────────────────
    cache_ttl_llm: int = Field(default=3600, alias="CACHE_TTL_LLM")
    cache_ttl_embedding: int = Field(default=86400, alias="CACHE_TTL_EMBEDDING")
    cache_ttl_document: int = Field(default=300, alias="CACHE_TTL_DOCUMENT")
    cache_ttl_query: int = Field(default=600, alias="CACHE_TTL_QUERY")
    cache_ttl_user: int = Field(default=60, alias="CACHE_TTL_USER")
    cache_max_entries: int = Field(default=10000, alias="CACHE_MAX_ENTRIES")

    # ── Derived ───────────────────────────────────────────
    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "production"

    @property
    def origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
