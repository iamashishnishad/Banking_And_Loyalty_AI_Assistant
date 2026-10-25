"""Hybrid retrieval: Postgres full-text search + pgvector, fused with Reciprocal Rank Fusion.

Both retrievers run in one SQL statement. RRF only uses ranks, so the two very different score
scales (ts_rank_cd vs cosine distance) never need to be calibrated against each other:

    score(chunk) = sum over retrievers r of 1 / (rrf_k + rank_r(chunk))
"""

import time
import uuid
from dataclasses import dataclass
from typing import Literal

import numpy as np
from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.services.embeddings import embed_query
from app.services.text import to_or_tsquery

SearchMode = Literal["hybrid", "vector", "lexical"]


@dataclass
class SearchHit:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_title: str
    chunk_index: int
    content: str
    meta: dict
    score: float
    vector_rank: int | None
    lexical_rank: int | None


@dataclass
class SearchResult:
    hits: list[SearchHit]
    mode: SearchMode
    took_ms: float


# :tsquery is NULL when the query is only stopwords; to_tsquery(NULL) matches nothing, so the
# result degrades to pure vector search. Candidates are fetched per retriever, fused, then cut to
# :top_k. FULL OUTER JOIN keeps chunks that only one retriever found.
_HYBRID_SQL = text(
    """
    WITH vec AS (
        -- Inner ORDER BY ... LIMIT is the shape the HNSW index can serve; rank afterwards.
        SELECT id, row_number() OVER (ORDER BY dist) AS rnk
        FROM (
            SELECT id, embedding <=> CAST(:embedding AS vector) AS dist
            FROM chunks
            WHERE tenant_id = :tenant_id AND :use_vector
            ORDER BY dist
            LIMIT :candidates
        ) v
    ),
    lex AS (
        SELECT id, row_number() OVER (ORDER BY rank DESC) AS rnk
        FROM (
            -- Normalisation flag 1 divides by 1 + log(length) so long chunks don't win on volume.
            SELECT id, ts_rank_cd(tsv, q, 1) AS rank
            FROM chunks, to_tsquery('simple', :tsquery) AS q
            WHERE tenant_id = :tenant_id AND :use_lexical AND tsv @@ q
            ORDER BY rank DESC
            LIMIT :candidates
        ) l
    ),
    fused AS (
        SELECT COALESCE(vec.id, lex.id) AS id,
               vec.rnk AS vector_rank,
               lex.rnk AS lexical_rank,
               COALESCE(1.0 / (:rrf_k + vec.rnk), 0) + COALESCE(1.0 / (:rrf_k + lex.rnk), 0) AS score
        FROM vec FULL OUTER JOIN lex ON vec.id = lex.id
    )
    SELECT c.id, c.document_id, d.title, c.chunk_index, c.content, c.metadata,
           f.score, f.vector_rank, f.lexical_rank
    FROM fused f
    JOIN chunks c ON c.id = f.id
    JOIN documents d ON d.id = c.document_id
    ORDER BY f.score DESC, c.id
    LIMIT :top_k
    """
)


def _vector_literal(vector: np.ndarray) -> str:
    return "[" + ",".join(f"{x:.7f}" for x in vector.tolist()) + "]"


async def hybrid_search(
    session: AsyncSession,
    tenant_id: uuid.UUID,
    query: str,
    *,
    top_k: int | None = None,
    mode: SearchMode = "hybrid",
    redis: Redis | None = None,
) -> SearchResult:
    settings = get_settings()
    started = time.perf_counter()

    use_vector = mode in ("hybrid", "vector")
    use_lexical = mode in ("hybrid", "lexical")
    embedding = await embed_query(query, redis) if use_vector else np.zeros(settings.embedding_dim, np.float32)

    # HNSW + a WHERE filter can return fewer than LIMIT rows when most neighbours belong to other
    # tenants; iterative scans (pgvector >= 0.8) keep searching until enough rows pass the filter.
    await session.execute(text("SET LOCAL hnsw.iterative_scan = relaxed_order"))
    rows = await session.execute(
        _HYBRID_SQL,
        {
            "embedding": _vector_literal(embedding),
            "tsquery": to_or_tsquery(query),
            "tenant_id": tenant_id,
            "use_vector": use_vector,
            "use_lexical": use_lexical,
            "candidates": settings.search_candidates,
            "rrf_k": settings.rrf_k,
            "top_k": top_k or settings.search_top_k,
        },
    )
    hits = [
        SearchHit(
            chunk_id=r.id,
            document_id=r.document_id,
            document_title=r.title,
            chunk_index=r.chunk_index,
            content=r.content,
            meta=r.metadata or {},
            score=float(r.score),
            vector_rank=r.vector_rank,
            lexical_rank=r.lexical_rank,
        )
        for r in rows
    ]
    return SearchResult(hits=hits, mode=mode, took_ms=(time.perf_counter() - started) * 1000)
