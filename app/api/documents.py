"""Document upload is asynchronous: the API validates the file, records a job and returns 202; a
RabbitMQ worker parses, chunks and embeds it. Poll GET /documents/jobs/{id} for the result."""

import hashlib
import logging
import uuid
from pathlib import PurePath

from fastapi import APIRouter, File, Form, HTTPException, Request, Response, UploadFile, status
from sqlalchemy import delete, select

from app.api.deps import PrincipalDep, RedisDep, SessionDep, TenantDep, require_roles
from app.config import get_settings
from app.models import Document, Job
from app.schemas import DocumentOut, JobOut, UploadOut
from app.services.parsing import SUPPORTED_EXTENSIONS
from app.services.semantic_cache import bump_kb_version

router = APIRouter(prefix="/documents", tags=["documents"])
logger = logging.getLogger(__name__)

AdminOnly = require_roles("admin")


@router.post(
    "",
    response_model=UploadOut,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[AdminOnly],
    responses={200: {"model": UploadOut, "description": "Identical file already ingested"}},
)
async def upload_document(
    request: Request,
    response: Response,
    session: SessionDep,
    tenant_id: TenantDep,
    file: UploadFile = File(...),
    title: str | None = Form(default=None, description="Optional; defaults to the Markdown heading or filename"),
) -> UploadOut:
    title = title.strip() if title else None
    filename = file.filename or "upload.txt"
    # Cheap checks stay synchronous so the client hears about a bad file now, not from a failed job.
    if PurePath(filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, f"Unsupported file type; use one of {sorted(SUPPORTED_EXTENSIONS)}"
        )
    max_bytes = get_settings().max_upload_mb * 1024 * 1024
    data = await file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, f"File exceeds {get_settings().max_upload_mb} MB")
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Empty file")

    existing = await session.scalar(
        select(Document).where(Document.tenant_id == tenant_id, Document.content_sha256 == hashlib.sha256(data).hexdigest())
    )
    if existing is not None:
        response.status_code = status.HTTP_200_OK
        return UploadOut(status="duplicate", document=DocumentOut.model_validate(existing))

    job = Job(
        tenant_id=tenant_id,
        kind="ingest_document",
        status="queued",
        attempts=0,
        params={"filename": filename, "title": title},
        payload=data,
    )
    session.add(job)
    await session.commit()  # commit before publishing: a worker must be able to find the row
    await session.refresh(job, ["created_at", "updated_at"])  # not the payload again
    try:
        await request.app.state.job_publisher.publish(job.id)
    except Exception:
        # The job is safe in Postgres; the worker's sweeper will publish it once RabbitMQ is back.
        logger.warning("Could not publish job %s; left queued for the sweeper", job.id, exc_info=True)
    return UploadOut(status="queued", job=JobOut.model_validate(job))


@router.get("/jobs/{job_id}", response_model=JobOut, dependencies=[AdminOnly])
async def get_job(job_id: uuid.UUID, session: SessionDep, tenant_id: TenantDep) -> Job:
    job = await session.scalar(select(Job).where(Job.id == job_id, Job.tenant_id == tenant_id))
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Job not found")
    return job


@router.get("", response_model=list[DocumentOut])
async def list_documents(session: SessionDep, tenant_id: TenantDep, _: PrincipalDep) -> list[Document]:
    rows = await session.scalars(
        select(Document).where(Document.tenant_id == tenant_id).order_by(Document.created_at.desc())
    )
    return list(rows)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT, dependencies=[AdminOnly])
async def delete_document(document_id: uuid.UUID, session: SessionDep, tenant_id: TenantDep, redis: RedisDep) -> None:
    result = await session.execute(
        delete(Document).where(Document.id == document_id, Document.tenant_id == tenant_id)
    )
    if result.rowcount == 0:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
    await session.commit()
    await bump_kb_version(redis, tenant_id)
