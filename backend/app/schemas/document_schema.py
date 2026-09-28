from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.models.document_model import DocumentStatus


class DocumentResponse(BaseModel):
    id: int
    filename: str
    mime_type: str
    file_size: int
    status: DocumentStatus
    uploaded_by: int
    uploaded_at: datetime
    last_indexed: datetime | None = None
    chunk_count: int
    embedding_model: str | None = None
    document_metadata: dict = Field(default_factory=dict)

    model_config = ConfigDict(from_attributes=True)


class DocumentUploadResponse(BaseModel):
    id: int
    filename: str
    status: DocumentStatus
    message: str


class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]
    total: int
