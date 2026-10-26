import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from redis.asyncio import Redis

from app.agent.llm import LLMNotConfiguredError, get_llm
from app.api import chat, chat_baseline, documents, health, loyalty, search, voice
from app.api.deps import RateLimitDep
from app.config import get_settings
from app.db import dispose_engine, get_engine
from app.services.auth import check_secret
from app.services.embeddings import get_embedder
from app.services.jobs import JobPublisher
from app.services.ratelimit import RateLimiter
from app.services.speech import get_transcriber

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    check_secret()
    get_engine()
    app.state.redis = Redis.from_url(
        settings.redis_url,
        socket_timeout=settings.redis_socket_timeout_seconds,
        socket_connect_timeout=settings.redis_socket_timeout_seconds,
    )
    app.state.rate_limiter = RateLimiter(app.state.redis)
    app.state.job_publisher = JobPublisher(settings.rabbitmq_url)
    # Load models before accepting traffic so the first request doesn't pay their load time.
    await asyncio.to_thread(get_embedder)
    if settings.voice_enabled:
        await asyncio.to_thread(get_transcriber)
    try:
        llm = get_llm()
        logger.info("LLM provider: %s, model %s", settings.llm_provider, llm.model)
    except LLMNotConfiguredError as exc:
        logger.warning("%s /chat and /voice/chat will return 503; /chat/baseline still works", exc)
    yield
    await app.state.job_publisher.close()
    await app.state.redis.aclose()
    await dispose_engine()


app = FastAPI(title="Banking & Loyalty AI Assistant", version="0.3.0", lifespan=lifespan)
app.include_router(health.router)
# Everything except health needs a JWT (the routers' dependencies) and is rate limited per user + tenant.
for router in (documents.router, search.router, loyalty.router, chat.router, chat_baseline.router, voice.router):
    app.include_router(router, dependencies=[RateLimitDep])


@app.exception_handler(LLMNotConfiguredError)
async def llm_not_configured(_: Request, exc: LLMNotConfiguredError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc)})
