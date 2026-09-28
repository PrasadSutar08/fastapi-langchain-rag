from fastapi import APIRouter, File, Query, Response, UploadFile

from app.api.deps import CurrentUser, SyncSessionDep
from app.schemas.document_schema import (
    DocumentListResponse,
    DocumentResponse,
    DocumentUploadResponse,
)
from app.services.document_service import document_service

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post("/upload", response_model=DocumentUploadResponse, status_code=201)
async def upload_document(
    session: SyncSessionDep,
    current_user: CurrentUser,
    file: UploadFile = File(...),
):
    document = await document_service.upload_document(
        session=session,
        uploaded_file=file,
        uploaded_by=current_user.id,
    )
    return DocumentUploadResponse(
        id=document.id,
        filename=document.filename,
        status=document.status,
        message="Document uploaded and indexed successfully.",
    )


@router.get("", response_model=DocumentListResponse)
def list_documents(
    session: SyncSessionDep,
    current_user: CurrentUser,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
):
    documents = document_service.list_documents(
        session,
        current_user.id,
        current_user.is_superuser,
        skip,
        limit,
    )
    return DocumentListResponse(
        documents=documents,
        total=len(documents),
    )


@router.get("/{document_id}", response_model=DocumentResponse)
def get_document(
    document_id: int,
    session: SyncSessionDep,
    current_user: CurrentUser,
):
    return document_service.get_document(
        session,
        document_id,
        current_user.id,
        current_user.is_superuser,
    )


@router.delete("/{document_id}", status_code=204)
def delete_document(
    document_id: int,
    session: SyncSessionDep,
    current_user: CurrentUser,
):
    document_service.delete_document(
        session,
        document_id,
        current_user.id,
        current_user.is_superuser,
    )
    return Response(status_code=204)
