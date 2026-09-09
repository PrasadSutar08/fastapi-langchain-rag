from __future__ import annotations

import time

from langchain_huggingface import HuggingFaceEmbeddings

from app.core.config import logger, settings


class EmbeddingService:
    """Process-local singleton for the embedding model."""

    _embeddings: HuggingFaceEmbeddings | None = None

    def get_embeddings(self) -> HuggingFaceEmbeddings:
        if self.__class__._embeddings is None:
            started = time.perf_counter()
            logger.info(
                "Loading embedding model "
                f"| model={settings.EMBEDDING_MODEL} "
                f"| device={settings.EMBEDDING_DEVICE}"
            )
            self.__class__._embeddings = HuggingFaceEmbeddings(
                model_name=settings.EMBEDDING_MODEL,
                model_kwargs={"device": settings.EMBEDDING_DEVICE},
                encode_kwargs={"normalize_embeddings": settings.EMBEDDING_NORMALIZE},
            )
            logger.success(
                "Embedding model loaded "
                f"| duration={(time.perf_counter() - started) * 1000:.2f} ms"
            )
        return self.__class__._embeddings

    @classmethod
    def reset(cls) -> None:
        cls._embeddings = None


embedding_service = EmbeddingService()


def get_embedding_model():
    return embedding_service.get_embeddings()
