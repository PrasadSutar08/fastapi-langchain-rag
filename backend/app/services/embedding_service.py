from __future__ import annotations

import time

from langchain_core.embeddings import Embeddings

from app.core.config import logger, settings


class EmbeddingService:
    """
    Process-local singleton for the embedding model.

    EMBEDDING_PROVIDER switches which backend computes embeddings:
      - "local" (default): self-hosted sentence-transformers, run on
        this process's own CPU. Used for local development and by
        anyone running this project open-source on their own machine.
        Requires the packages in requirements-local-embeddings.txt
        (torch, sentence-transformers, langchain-huggingface), which
        are installed by default in Docker builds but deliberately
        left out of the deployed image (see backend/Dockerfile's
        INSTALL_LOCAL_EMBEDDINGS build arg) to fit free-host memory
        limits.
      - "cohere": hosted embeddings via Cohere's free Embed API (no
        credit card, no local model weights, near-zero memory
        footprint). Used for the publicly deployed demo.

    Both code paths always exist in this file; only which one runs
    depends on config. The heavy "local" import is done lazily inside
    _build_local() rather than at module level, specifically so this
    module can still be imported (and the "cohere" path used) in a
    deployed image that never installed langchain_huggingface at all.
    """

    _embeddings: Embeddings | None = None

    def get_embeddings(self) -> Embeddings:
        if self.__class__._embeddings is None:
            provider = settings.EMBEDDING_PROVIDER.strip().lower()
            started = time.perf_counter()

            if provider == "cohere":
                self.__class__._embeddings = self._build_cohere()
            else:
                if provider != "local":
                    logger.warning(
                        "Unrecognized EMBEDDING_PROVIDER="
                        f"{settings.EMBEDDING_PROVIDER!r}, falling back "
                        "to 'local'."
                    )
                self.__class__._embeddings = self._build_local()

            logger.success(
                "Embedding model ready "
                f"| provider={provider} "
                f"| duration={(time.perf_counter() - started) * 1000:.2f} ms"
            )
        return self.__class__._embeddings

    @staticmethod
    def _build_local() -> Embeddings:
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError as exc:
            raise RuntimeError(
                "EMBEDDING_PROVIDER=local requires the optional local-"
                "embedding packages (requirements-local-embeddings.txt), "
                "which aren't installed in this image. Either rebuild "
                "with --build-arg INSTALL_LOCAL_EMBEDDINGS=true, or set "
                "EMBEDDING_PROVIDER=cohere instead."
            ) from exc

        logger.info(
            "Loading local embedding model "
            f"| model={settings.EMBEDDING_MODEL} "
            f"| device={settings.EMBEDDING_DEVICE}"
        )

        return HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={"device": settings.EMBEDDING_DEVICE},
            encode_kwargs={"normalize_embeddings": settings.EMBEDDING_NORMALIZE},
        )

    @staticmethod
    def _build_cohere() -> Embeddings:
        if not settings.COHERE_API_KEY:
            raise RuntimeError(
                "EMBEDDING_PROVIDER=cohere but COHERE_API_KEY is not "
                "set. Get a free key (no credit card) at "
                "https://dashboard.cohere.com/api-keys and set it in "
                ".env."
            )

        from langchain_cohere import CohereEmbeddings

        logger.info(
            f"Loading Cohere embeddings | model={settings.COHERE_EMBED_MODEL}"
        )

        # CohereEmbeddings reads the COHERE_API_KEY environment
        # variable automatically. Docker's env_file already provides
        # it under this exact name, but pydantic-settings reading
        # .env does NOT itself populate os.environ (only settings.*),
        # so set it explicitly here too -- covers running via plain
        # `uvicorn` outside Docker.
        import os

        os.environ.setdefault("COHERE_API_KEY", settings.COHERE_API_KEY)

        return CohereEmbeddings(model=settings.COHERE_EMBED_MODEL)

    @classmethod
    def reset(cls) -> None:
        cls._embeddings = None


embedding_service = EmbeddingService()


def get_embedding_model():
    return embedding_service.get_embeddings()
