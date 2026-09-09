from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from sqlalchemy import Column, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, SQLModel


class DocumentStatus(str, Enum):
    UPLOADED = "UPLOADED"
    PROCESSING = "PROCESSING"
    INDEXED = "INDEXED"
    FAILED = "FAILED"
    DELETED = "DELETED"


class Document(SQLModel, table=True):
    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint(
            "uploaded_by",
            "file_hash",
            name="uq_documents_uploaded_by_file_hash",
        ),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    filename: str = Field(index=True, max_length=512)
    storage_path: str
    file_hash: str = Field(index=True, max_length=64)
    mime_type: str = Field(max_length=128)
    file_size: int
    status: DocumentStatus = Field(default=DocumentStatus.UPLOADED, index=True)
    uploaded_by: int = Field(foreign_key="users.id", index=True)
    uploaded_at: datetime = Field(
        default_factory=datetime.utcnow
    )
    last_indexed: Optional[datetime] = None
    chunk_count: int = 0
    embedding_model: Optional[str] = None
    document_metadata: dict = Field(
        default_factory=dict,
        sa_column=Column(JSONB),
    )
