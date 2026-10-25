import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from redis.asyncio import Redis

from app.api import chat, documents, health, loyalty, search
from app.config import get_settings
from app.db import dispose_engine, get_engine
from app.services.embeddings import get_embedder


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    get_engine()
    app.state.redis = Redis.from_url(settings.redis_url)
    # Load the model before accepting traffic so the first request doesn't pay ~10s of load time.
    await asyncio.to_thread(get_embedder)
    yield
    await app.state.redis.aclose()
    await dispose_engine()


app = FastAPI(title="Banking & Loyalty AI Assistant", version="0.1.0", lifespan=lifespan)
app.include_router(health.router)
app.include_router(documents.router)
app.include_router(search.router)
app.include_router(loyalty.router)
app.include_router(chat.router)
