from fastapi import APIRouter, Request, Response, status
from sqlalchemy import text

from app.api.deps import RedisDep, SessionDep

router = APIRouter(tags=["health"])


@router.get("/health/live")
async def live() -> dict:
    return {"status": "ok"}


@router.get("/health/ready")
async def ready(request: Request, session: SessionDep, redis: RedisDep, response: Response) -> dict:
    checks = {}
    try:
        await session.execute(text("SELECT 1"))
        checks["postgres"] = "ok"
    except Exception as exc:
        checks["postgres"] = f"error: {type(exc).__name__}"
    try:
        await redis.ping()
        checks["redis"] = "ok"
    except Exception as exc:
        checks["redis"] = f"error: {type(exc).__name__}"
    ok = all(v == "ok" for v in checks.values())
    if not ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    # Reported but not required: uploads are queued in Postgres while RabbitMQ is down.
    rabbitmq = "ok" if await request.app.state.job_publisher.ping() else "unreachable"
    return {"status": "ok" if ok else "degraded", "checks": checks, "optional": {"rabbitmq": rabbitmq}}
