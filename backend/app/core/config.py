from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def parse_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]

    if isinstance(value, str):
        value = value.strip()

        if not value:
            return []

        if value.startswith("["):
            parsed = json.loads(value)

            if not isinstance(parsed, list):
                raise ValueError("Expected a JSON list.")

            return [
                str(item).strip()
                for item in parsed
                if str(item).strip()
            ]

        return [
            item.strip()
            for item in value.split(",")
            if item.strip()
        ]

    raise ValueError("Expected a list or comma-separated string.")


class Settings(BaseSettings):

    # ==========================================================
    # APPLICATION
    # ==========================================================

    API_VERSION: str = "v1"
    API_V1_STR: str = "/api/v1"

    PROJECT_NAME: str = "Enterprise Document Intelligence Platform"

    ENVIRONMENT: str = "development"
    DEBUG: bool = False
    TESTING: bool = False

    AUTO_CREATE_TABLES: bool = True

    # ==========================================================
    # SECURITY
    # ==========================================================

    SECRET_KEY_ACCESS_API: str

    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24

    FIRST_SUPERUSER: str
    FIRST_SUPERUSER_PASSWORD: str

    # ==========================================================
    # DATABASE
    # ==========================================================

    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "ragdb"
    DB_USER: str = "postgres"
    DB_PASS: str = "postgres"

    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20
    DB_POOL_TIMEOUT: int = 30
    DB_POOL_RECYCLE: int = 1800

    # ==========================================================
    # REDIS
    # ==========================================================

    REDIS_URL: str = "redis://localhost:6379/0"

    REDIS_RATE_LIMIT_ENABLED: bool = True

    QA_RATE_LIMIT: int = 20
    QA_RATE_LIMIT_WINDOW_SECONDS: int = 60

    SEARCH_RATE_LIMIT: int = 60
    SEARCH_RATE_LIMIT_WINDOW_SECONDS: int = 60

    LOGIN_RATE_LIMIT: int = 5
    REGISTER_RATE_LIMIT: int = 5

    AUTH_RATE_LIMIT_WINDOW_SECONDS: int = 60

    # ==========================================================
    # DOCUMENTS
    # ==========================================================

    DOCUMENT_STORAGE_PATH: str = "storage/documents"

    MAX_UPLOAD_SIZE: int = 50 * 1024 * 1024

    ALLOWED_DOCUMENT_TYPES_RAW: str = Field(
        default="application/pdf",
        alias="ALLOWED_DOCUMENT_TYPES",
    )

    # ==========================================================
    # EMBEDDINGS
    # ==========================================================

    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"

    EMBEDDING_DEVICE: str = "cpu"

    EMBEDDING_NORMALIZE: bool = True

    VECTOR_COLLECTION_NAME: str = "docs"

    # ==========================================================
    # RETRIEVAL
    # ==========================================================

    RETRIEVAL_TOP_K: int = 4

    RETRIEVAL_FETCH_K: int = 20

    RETRIEVAL_SEARCH_TYPE: str = "mmr"

    RETRIEVAL_MMR_LAMBDA: float = 0.65

    RETRIEVAL_KEYWORD_WEIGHT: float = 0.15

    RETRIEVAL_RELEVANCE_THRESHOLD: float = 0.20

    # ==========================================================
    # CHUNKING
    # ==========================================================

    CHUNK_SIZE: int = 900

    CHUNK_OVERLAP: int = 120

    # ==========================================================
    # RAG CONTEXT
    # ==========================================================

    MAX_CONTEXT_CHARS: int = 9000

    MAX_HISTORY_MESSAGES: int = 4

    MAX_CONTEXT_DOCUMENTS: int = 4

    # ==========================================================
    # LLM
    # ==========================================================
    # LLM_PROVIDER selects which backend answers questions:
    #   "ollama" (default) -- self-hosted local inference, used for
    #       local development and by anyone running this project
    #       open-source on their own machine/GPU.
    #   "groq"  -- hosted inference via Groq's free API, used for the
    #       publicly deployed demo so it stays fast and always-on
    #       without needing a self-hosted GPU/VM.
    # Both code paths are always available; this just picks which one
    # LLMService builds at runtime. See README.md for setup notes.

    LLM_PROVIDER: str = "ollama"

    OLLAMA_BASE_URL: str = "http://localhost:11434"

    OLLAMA_MODEL: str = "qwen2.5:7b"

    OLLAMA_TEMPERATURE: float = 0.0

    OLLAMA_MAX_TOKENS: int = 512

    OLLAMA_NUM_CTX: int = 2048

    OLLAMA_KEEP_ALIVE: str = "10m"

    # IMPORTANT:
    # This is GPU LAYERS, not GPU number.
    #
    # -1 = allow Ollama to determine the maximum
    # that fits.
    #
    # Do NOT use 1 expecting "GPU 1".
    OLLAMA_NUM_GPU: int = -1

    OLLAMA_NUM_THREAD: int = 6

    OLLAMA_TIMEOUT_SECONDS: int = 180

    # Number of concurrent LLM generations.
    #
    # Your 4GB GPU should not be hit by many simultaneous
    # 7B generations.
    OLLAMA_MAX_CONCURRENCY: int = 1

    # ==========================================================
    # GROQ (used only when LLM_PROVIDER=groq)
    # ==========================================================

    GROQ_API_KEY: str = ""

    GROQ_MODEL: str = "llama-3.3-70b-versatile"

    GROQ_TIMEOUT_SECONDS: int = 30

    # ==========================================================
    # HTTP
    # ==========================================================

    HTTP_TIMEOUT_SECONDS: int = 30

    # ==========================================================
    # CORS
    # ==========================================================

    CORS_ORIGINS_RAW: str = Field(
        default="http://localhost:3000,http://localhost:5173",
        alias="CORS_ORIGINS",
    )

    CORS_ALLOW_CREDENTIALS: bool = True

    # ==========================================================
    # LOGGING
    # ==========================================================

    LOGGING_LEVEL: str = "INFO"

    # ==========================================================
    # LANGSMITH / TRACING
    # ==========================================================
    # Optional. If LANGSMITH_API_KEY is set, tracing is enabled and the
    # standard LANGCHAIN_* env vars LangChain reads at import time are
    # populated from these settings. See README.md "LangSmith setup".

    LANGSMITH_TRACING: bool = False
    LANGSMITH_API_KEY: str = ""
    LANGSMITH_PROJECT: str = "enterprise-rag-api"
    LANGSMITH_ENDPOINT: str = "https://api.smith.langchain.com"

    @property
    def CORS_ORIGINS(self) -> list[str]:
        return parse_list(self.CORS_ORIGINS_RAW)

    @property
    def ALLOWED_DOCUMENT_TYPES(self) -> list[str]:
        return parse_list(self.ALLOWED_DOCUMENT_TYPES_RAW)

    @property
    def DOCUMENT_STORAGE_ABS(self) -> Path:
        path = Path(self.DOCUMENT_STORAGE_PATH)
        if path.is_absolute():
            return path
        return Path(__file__).resolve().parents[2] / path

    @property
    def ASYNC_DATABASE_URI(self) -> str:
        return (
            f"postgresql+asyncpg://"
            f"{self.DB_USER}:{self.DB_PASS}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    @property
    def SYNC_DATABASE_URI(self) -> str:
        return (
            f"postgresql+psycopg2://"
            f"{self.DB_USER}:{self.DB_PASS}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        case_sensitive=True,
        populate_by_name=True,
        extra="ignore",
    )


class LogConfig:

    FORMAT = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "{message}"
    )

    @staticmethod
    def configure() -> None:
        logger.remove()

        logger.add(
            sys.stderr,
            format=LogConfig.FORMAT,
            level="INFO",
            enqueue=True,
            backtrace=False,
            diagnose=False,
        )


LogConfig.configure()

settings = Settings()

if settings.LANGSMITH_TRACING and settings.LANGSMITH_API_KEY:
    import os

    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_API_KEY", settings.LANGSMITH_API_KEY)
    os.environ.setdefault("LANGCHAIN_PROJECT", settings.LANGSMITH_PROJECT)
    os.environ.setdefault("LANGCHAIN_ENDPOINT", settings.LANGSMITH_ENDPOINT)

logger.remove()

logger.add(
    sys.stderr,
    format=LogConfig.FORMAT,
    level=settings.LOGGING_LEVEL,
    enqueue=True,
    backtrace=False,
    diagnose=settings.DEBUG,
)
