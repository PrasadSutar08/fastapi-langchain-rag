from __future__ import annotations

import hashlib
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException, UploadFile, status
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session

from app.core.config import logger, settings
from app.crud import document_crud
from app.models.document_model import Document, DocumentStatus
from app.services.document_ingestion_service import document_ingestion_service
from app.services.vector_store_service import VectorStoreService


class DocumentService:
    def __init__(self) -> None:
        self.storage_path = Path(settings.DOCUMENT_STORAGE_PATH)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._vector_store_service: VectorStoreService | None = None

    @property
    def vector_store_service(self) -> VectorStoreService:
        if self._vector_store_service is None:
            self._vector_store_service = VectorStoreService(
                document_ingestion_service.embedding_model
            )
        return self._vector_store_service

    async def upload_document(
        self,
        session: Session,
        uploaded_file: UploadFile,
        uploaded_by: int,
    ) -> Document:
        filename = Path(uploaded_file.filename or "").name.strip()
        if not filename:
            raise HTTPException(400, "A valid filename is required.")

        mime_type = (uploaded_file.content_type or "").lower()
        allowed = {item.lower() for item in settings.ALLOWED_DOCUMENT_TYPES}
        if mime_type not in allowed:
            raise HTTPException(
                status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                f"Unsupported document type: {mime_type}.",
            )

        if mime_type == "application/pdf":
            signature = await uploaded_file.read(5)
            if signature != b"%PDF-":
                raise HTTPException(400, "The uploaded file is not a valid PDF.")
            await uploaded_file.seek(0)

        destination = self.storage_path / f"{uuid4().hex}_{filename}"
        digest = hashlib.sha256()
        size = 0

        try:
            with destination.open("wb") as output:
                while True:
                    chunk = await uploaded_file.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > settings.MAX_UPLOAD_SIZE:
                        raise HTTPException(
                            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            f"Uploaded file exceeds {settings.MAX_UPLOAD_SIZE} bytes.",
                        )
                    digest.update(chunk)
                    output.write(chunk)

            if size == 0:
                raise HTTPException(400, "Uploaded file is empty.")

            file_hash = digest.hexdigest()
            if document_crud.get_document_by_hash(session, file_hash, uploaded_by):
                raise HTTPException(
                    status.HTTP_409_CONFLICT,
                    "This document has already been uploaded by this user.",
                )

            document = Document(
                filename=filename,
                storage_path=str(destination),
                file_hash=file_hash,
                mime_type=mime_type,
                file_size=size,
                uploaded_by=uploaded_by,
            )

            try:
                document = document_crud.create_document(session, document)
            except IntegrityError as exc:
                session.rollback()
                raise HTTPException(
                    status.HTTP_409_CONFLICT,
                    "This document has already been uploaded by this user.",
                ) from exc

            try:
                return document_ingestion_service.index_document(session, document)
            except Exception as exc:
                document.status = DocumentStatus.FAILED
                try:
                    document_crud.update_document(session, document)
                except Exception:
                    session.rollback()
                raise HTTPException(
                    status.HTTP_500_INTERNAL_SERVER_ERROR,
                    "Document upload or indexing failed.",
                ) from exc

        except HTTPException:
            self._remove_file(destination)
            raise
        except Exception as exc:
            self._remove_file(destination)
            logger.exception(f"Document upload failed | error={exc}")
            raise HTTPException(
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                "Failed to process uploaded document.",
            ) from exc

    def list_documents(self, session, user_id, is_superuser, skip, limit):
        if is_superuser:
            return document_crud.list_documents(session, skip, limit)
        return document_crud.list_documents_for_user(session, user_id, skip, limit)

    def get_document(self, session, document_id, user_id, is_superuser):
        document = document_crud.get_document_by_id(session, document_id)
        if document is None or (
            not is_superuser and document.uploaded_by != user_id
        ):
            raise HTTPException(404, "Document not found.")
        return document

    def delete_document(self, session, document_id, user_id, is_superuser):
        document = self.get_document(session, document_id, user_id, is_superuser)
        if document.id is None:
            raise HTTPException(404, "Document not found.")

        # Vector deletion is the first destructive step. If it fails, the
        # database record remains and can be retried.
        self.vector_store_service.delete_document_embeddings(document.id)
        self._remove_file(Path(document.storage_path))
        document_crud.delete_document(session, document)

    @staticmethod
    def _remove_file(path: Path) -> None:
        try:
            if path.exists():
                path.unlink()
        except OSError as exc:
            logger.warning(f"Failed to remove document file | path={path} | error={exc}")


document_service = DocumentService()
