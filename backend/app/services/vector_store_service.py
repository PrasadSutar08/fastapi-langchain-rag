from __future__ import annotations

from typing import Any

from langchain_core.documents import Document as LangChainDocument
from langchain_community.vectorstores.pgvector import PGVector
from sqlalchemy import create_engine, text

from app.core.config import logger, settings


class VectorStoreService:
    COLLECTION_NAME = settings.VECTOR_COLLECTION_NAME

    _vector_store: PGVector | None = None
    _engine = None

    def __init__(self, embedding_model) -> None:
        if self.__class__._vector_store is None:
            logger.info(
                f"Initializing PGVector | collection={self.COLLECTION_NAME}"
            )
            self.__class__._vector_store = PGVector(
                collection_name=self.COLLECTION_NAME,
                connection_string=settings.SYNC_DATABASE_URI,
                embedding_function=embedding_model,
            )
            self.__class__._engine = create_engine(
                settings.SYNC_DATABASE_URI,
                pool_pre_ping=True,
                pool_size=settings.DB_POOL_SIZE,
                max_overflow=settings.DB_MAX_OVERFLOW,
            )
            logger.success("PGVector initialized.")

    def get_vector_store(self) -> PGVector:
        if self.__class__._vector_store is None:
            raise RuntimeError("PGVector is not initialized.")
        return self.__class__._vector_store

    def add_documents(self, documents: list[LangChainDocument]) -> None:
        if not documents:
            return
        self.get_vector_store().add_documents(documents)

    def delete_document_embeddings(self, document_id: int) -> int:
        if self.__class__._engine is None:
            raise RuntimeError("PGVector database engine is not initialized.")

        query = text(
            """
            DELETE FROM langchain_pg_embedding
            WHERE cmetadata->>'document_id' = :document_id
            """
        )
        with self.__class__._engine.begin() as connection:
            result = connection.execute(
                query, {"document_id": str(document_id)}
            )
            return int(result.rowcount or 0)

    @staticmethod
    def build_user_filter(
        user_id: int,
        is_superuser: bool = False,
        document_id: int | None = None,
        filename: str | None = None,
    ) -> dict[str, Any] | None:
        filters: dict[str, Any] = {}
        if not is_superuser:
            filters["uploaded_by"] = str(user_id)
        if document_id is not None:
            filters["document_id"] = str(document_id)
        if filename is not None:
            filters["filename"] = filename
        return filters or None
