import uuid

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from sqlalchemy import delete, select

from app.api.deps import SessionDep, TenantDep
from app.config import get_settings
from app.models import Document
from app.schemas import DocumentOut, IngestOut
from app.services.ingestion import ingest_document
from app.services.parsing import UnsupportedDocumentError

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", response_model=IngestOut, status_code=status.HTTP_201_CREATED)
async def upload_document(
    session: SessionDep,
    tenant_id: TenantDep,
    file: UploadFile = File(...),
    title: str | None = Form(default=None),
) -> IngestOut:
    max_bytes = get_settings().max_upload_mb * 1024 * 1024
    data = await file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, f"File exceeds {get_settings().max_upload_mb} MB")
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Empty file")
    try:
        result = await ingest_document(session, tenant_id, file.filename or "upload.txt", data, title)
    except UnsupportedDocumentError as exc:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, str(exc)) from exc
    return IngestOut(document=DocumentOut.model_validate(result.document), created=result.created)


@router.get("", response_model=list[DocumentOut])
async def list_documents(session: SessionDep, tenant_id: TenantDep) -> list[Document]:
    rows = await session.scalars(
        select(Document).where(Document.tenant_id == tenant_id).order_by(Document.created_at.desc())
    )
    return list(rows)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(document_id: uuid.UUID, session: SessionDep, tenant_id: TenantDep) -> None:
    result = await session.execute(
        delete(Document).where(Document.id == document_id, Document.tenant_id == tenant_id)
    )
    if result.rowcount == 0:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
    await session.commit()
