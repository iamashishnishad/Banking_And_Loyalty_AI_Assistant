"""Recompute lexical_text and phonetic_text for every chunk from its stored content.

Run after changing app/services/text.py or language.py (stopwords, spelling variants, phonetic
rules). Embeddings are untouched, so this is fast and needs no model.

    docker compose exec api python -m scripts.reindex_text
"""

import asyncio
import logging

from sqlalchemy import select, update

from app.db import dispose_engine, get_sessionmaker
from app.models import Chunk, Document
from app.services.text import lexical_text, phonetic_text

logger = logging.getLogger("reindex")
BATCH = 500


async def reindex() -> None:
    total = 0
    async with get_sessionmaker()() as session:
        rows = await session.execute(
            select(Chunk.id, Chunk.content, Document.title).join(Document, Document.id == Chunk.document_id)
        )
        for chunk_id, content, title in rows.all():
            # Same inputs as ingestion: the title is indexed with every chunk.
            source = f"{title} {content}"
            await session.execute(
                update(Chunk)
                .where(Chunk.id == chunk_id)
                .values(lexical_text=lexical_text(source), phonetic_text=phonetic_text(source))
            )
            total += 1
            if total % BATCH == 0:
                await session.commit()
        await session.commit()
    logger.info("Reindexed %d chunks", total)
    await dispose_engine()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    asyncio.run(reindex())
