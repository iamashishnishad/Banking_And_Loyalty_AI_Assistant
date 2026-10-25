"""Multilingual sentence embeddings (multilingual-e5 by default).

e5 models are trained with "query: " / "passage: " prefixes and lose noticeable accuracy without
them, so callers choose embed_queries or embed_passages rather than passing raw text.
"""

import asyncio
import hashlib
import logging
import threading

import numpy as np
from redis.asyncio import Redis

from app.config import get_settings

logger = logging.getLogger(__name__)


class Embedder:
    def __init__(self, model_name: str, device: str, batch_size: int, expected_dim: int) -> None:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model %s on %s", model_name, device)
        self.model = SentenceTransformer(model_name, device=device)
        self.model_name = model_name
        self.batch_size = batch_size
        # Older sentence-transformers names it get_sentence_embedding_dimension.
        get_dim = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        self.dim = get_dim()
        if self.dim != expected_dim:
            raise RuntimeError(
                f"{model_name} produces {self.dim}-d vectors but EMBEDDING_DIM={expected_dim}. "
                "Changing the dimension needs a new migration for chunks.embedding and re-ingestion."
            )
        self._is_e5 = "e5" in model_name.lower()
        # encode() is not documented as thread-safe; serialise calls from the threadpool.
        self._lock = threading.Lock()

    def _encode(self, texts: list[str], prefix: str) -> np.ndarray:
        if self._is_e5:
            texts = [prefix + t for t in texts]
        with self._lock:
            return self.model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,  # unit vectors, so cosine distance == 1 - dot product
                convert_to_numpy=True,
                show_progress_bar=False,
            ).astype(np.float32)

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, "passage: ")

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, "query: ")


_embedder: Embedder | None = None
_embedder_lock = threading.Lock()


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        with _embedder_lock:
            if _embedder is None:
                s = get_settings()
                _embedder = Embedder(s.embedding_model, s.embedding_device, s.embedding_batch_size, s.embedding_dim)
    return _embedder


async def embed_passages(texts: list[str]) -> np.ndarray:
    return await asyncio.to_thread(get_embedder().embed_passages, texts)


async def embed_query(text: str, redis: Redis | None = None) -> np.ndarray:
    """Embed one search query, using Redis as a cache keyed on model + exact query text.
    Cache failures are logged and ignored: search must keep working if Redis is down."""
    settings = get_settings()
    ttl = settings.embedding_cache_ttl_seconds
    key = None
    if redis is not None and ttl > 0:
        digest = hashlib.sha256(f"{settings.embedding_model}|{text}".encode()).hexdigest()
        key = f"emb:q:{digest}"
        try:
            cached = await redis.get(key)
            if cached is not None:
                return np.frombuffer(cached, dtype=np.float32)
        except Exception:
            logger.warning("Embedding cache read failed", exc_info=True)

    vector = (await asyncio.to_thread(get_embedder().embed_queries, [text]))[0]

    if key is not None:
        try:
            await redis.set(key, vector.tobytes(), ex=ttl)
        except Exception:
            logger.warning("Embedding cache write failed", exc_info=True)
    return vector
