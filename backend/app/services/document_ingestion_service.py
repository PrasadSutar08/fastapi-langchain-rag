from __future__ import annotations

from datetime import datetime

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from sqlmodel import Session

from app.core.config import logger, settings
from app.crud import document_crud
from app.models.document_model import Document, DocumentStatus
from app.services.embedding_service import embedding_service
from app.services.vector_store_service import VectorStoreService
from app.services.metrics_service import metrics_service


class DocumentIngestionService:
    def __init__(self) -> None:
        self._embedding_model = None
        self._vector_store = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", "; ", ": ", " ", ""],
            length_function=len,
        )

    @property
    def embedding_model(self):
        if self._embedding_model is None:
            self._embedding_model = embedding_service.get_embeddings()
        return self._embedding_model

    @property
    def vector_store(self) -> VectorStoreService:
        if self._vector_store is None:
            self._vector_store = VectorStoreService(self.embedding_model)
        return self._vector_store

    def index_document(self, session: Session, document: Document) -> Document:
        try:
            document.status = DocumentStatus.PROCESSING
            document_crud.update_document(session, document)

            docs = self._load_document(document)
            chunks = self.text_splitter.split_documents(docs)
            if not chunks:
                raise ValueError("Document produced zero chunks.")

            total = len(chunks)
            for index, chunk in enumerate(chunks, start=1):
                page = chunk.metadata.get("page")
                page_number = page + 1 if isinstance(page, int) else page
                chunk.metadata.update({
                    "document_id": str(document.id),
                    "filename": document.filename,
                    "uploaded_by": str(document.uploaded_by),
                    "mime_type": document.mime_type,
                    "page": page_number,
                    "source": document.storage_path,
                    "chunk_index": index,
                    "chunk_count": total,
                })

            if document.id is not None:
                self.vector_store.delete_document_embeddings(document.id)
            self.vector_store.add_documents(chunks)

            document.status = DocumentStatus.INDEXED
            document.chunk_count = total
            document.last_indexed = datetime.utcnow()
            document.embedding_model = settings.EMBEDDING_MODEL
            document = document_crud.update_document(session, document)

            metrics_service.increment("document_indexed_total")
            metrics_service.increment("document_chunks_indexed_total", total)

            logger.success(
                f"DOCUMENT INDEXED | document_id={document.id} | chunks={total}"
            )
            return document
        except Exception:
            logger.exception(
                f"Document indexing failed | document_id={document.id}"
            )
            document.status = DocumentStatus.FAILED
            try:
                document_crud.update_document(session, document)
            except Exception:
                logger.exception("Failed to persist FAILED document status.")
            raise

    @staticmethod
    def _load_document(document: Document):
        if (document.mime_type or "").lower() != "application/pdf":
            raise ValueError(f"Unsupported document type: {document.mime_type}")
        docs = PyPDFLoader(document.storage_path).load()
        if not docs:
            raise ValueError("PDF contains no pages.")
        if not any((doc.page_content or "").strip() for doc in docs):
            raise ValueError("PDF contains no extractable text. OCR is required.")
        return docs


document_ingestion_service = DocumentIngestionService()
