"""Document ingestion: parse -> chunk -> embed -> store, in one transaction.

Runs inline in the request for now; Phase 3 moves this onto a RabbitMQ worker.
"""

import hashlib
import logging
import uuid
from dataclasses import dataclass
from pathlib import PurePath

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models import Chunk, Document
from app.services.chunking import chunk_sections
from app.services.embeddings import embed_passages
from app.services.parsing import mime_type_for, parse_document
from app.services.text import lexical_text, phonetic_text

logger = logging.getLogger(__name__)


def _title_from(filename: str, first_text: str) -> str:
    """A leading Markdown '# Heading' if there is one, otherwise the filename stem."""
    first_line = first_text.lstrip().split("\n", 1)[0]
    if first_line.startswith("# "):
        return first_line[2:].strip()[:300]
    return PurePath(filename).stem.replace("_", " ").replace("-", " ").strip()[:300]


@dataclass
class IngestResult:
    document: Document
    created: bool  # False when an identical file was already ingested for this tenant


async def ingest_document(
    session: AsyncSession,
    tenant_id: uuid.UUID,
    filename: str,
    data: bytes,
    title: str | None = None,
) -> IngestResult:
    settings = get_settings()
    sha = hashlib.sha256(data).hexdigest()

    existing = await session.scalar(
        select(Document).where(Document.tenant_id == tenant_id, Document.content_sha256 == sha)
    )
    if existing is not None:
        return IngestResult(document=existing, created=False)

    sections = parse_document(filename, data)
    pieces = chunk_sections(sections, settings.chunk_size, settings.chunk_overlap)
    title = title or _title_from(filename, sections[0].text)

    # Embed the title with each chunk so short chunks ("Fee: Rs 500") keep their context.
    vectors = await embed_passages([f"{title}\n{p.content}" for p in pieces])

    document = Document(
        tenant_id=tenant_id,
        title=title,
        source_filename=filename,
        mime_type=mime_type_for(filename),
        content_sha256=sha,
        chunk_count=len(pieces),
    )
    document.chunks = [
        Chunk(
            tenant_id=tenant_id,
            chunk_index=p.index,
            content=p.content,
            lexical_text=lexical_text(f"{title} {p.content}"),
            phonetic_text=phonetic_text(f"{title} {p.content}"),
            embedding=vector,
            meta=p.meta,
        )
        for p, vector in zip(pieces, vectors, strict=True)
    ]
    session.add(document)
    try:
        await session.commit()
    except IntegrityError:
        # The same file was uploaded concurrently and the other request won the unique constraint.
        await session.rollback()
        existing = await session.scalar(
            select(Document).where(Document.tenant_id == tenant_id, Document.content_sha256 == sha)
        )
        if existing is None:
            raise
        return IngestResult(document=existing, created=False)
    logger.info("Ingested %s for tenant %s: %d chunks", filename, tenant_id, len(pieces))
    return IngestResult(document=document, created=True)
