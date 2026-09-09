from __future__ import annotations

from sqlmodel import Session, select

from app.models.document_model import Document


def create_document(session: Session, document: Document) -> Document:
    session.add(document)
    session.commit()
    session.refresh(document)
    return document


def get_document_by_id(session: Session, document_id: int) -> Document | None:
    return session.get(Document, document_id)


def get_document_by_hash(
    session: Session,
    file_hash: str,
    uploaded_by: int,
) -> Document | None:
    return session.exec(
        select(Document).where(
            Document.file_hash == file_hash,
            Document.uploaded_by == uploaded_by,
        )
    ).first()


def list_documents(session: Session, skip: int = 0, limit: int = 100) -> list[Document]:
    return list(
        session.exec(
            select(Document)
            .offset(skip)
            .limit(limit)
            .order_by(Document.uploaded_at.desc())
        ).all()
    )


def list_documents_for_user(
    session: Session,
    user_id: int,
    skip: int = 0,
    limit: int = 100,
) -> list[Document]:
    return list(
        session.exec(
            select(Document)
            .where(Document.uploaded_by == user_id)
            .offset(skip)
            .limit(limit)
            .order_by(Document.uploaded_at.desc())
        ).all()
    )


def update_document(session: Session, document: Document) -> Document:
    session.add(document)
    session.commit()
    session.refresh(document)
    return document


def delete_document(session: Session, document: Document) -> None:
    session.delete(document)
    session.commit()
