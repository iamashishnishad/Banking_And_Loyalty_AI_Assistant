import logging
import uuid
from typing import Annotated

from fastapi import Depends, HTTPException, Request, Response, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from redis.asyncio import Redis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db import get_session
from app.models import Tenant
from app.services.auth import AuthError, Principal, Role, decode_token
from app.services.ratelimit import RateLimiter

logger = logging.getLogger(__name__)

SessionDep = Annotated[AsyncSession, Depends(get_session)]

_bearer = HTTPBearer(auto_error=False, description="JWT from scripts/issue_token.py")


def get_redis(request: Request) -> Redis:
    return request.app.state.redis


RedisDep = Annotated[Redis, Depends(get_redis)]


def get_principal(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> Principal:
    if credentials is None:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, "Missing bearer token", headers={"WWW-Authenticate": "Bearer"}
        )
    try:
        return decode_token(credentials.credentials)
    except AuthError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(exc), headers={"WWW-Authenticate": "Bearer"}) from exc


PrincipalDep = Annotated[Principal, Depends(get_principal)]

# Tenant slugs never change id, so resolve each one once per process.
_tenant_ids: dict[str, uuid.UUID] = {}


async def get_tenant_id(session: SessionDep, principal: PrincipalDep) -> uuid.UUID:
    """The tenant is the one in the signed token, so a caller can't reach another bank's data."""
    slug = principal.tenant
    if slug not in _tenant_ids:
        tenant_id = await session.scalar(select(Tenant.id).where(Tenant.slug == slug))
        if tenant_id is None:
            raise HTTPException(status.HTTP_403_FORBIDDEN, f"Unknown tenant '{slug}'")
        _tenant_ids[slug] = tenant_id
    return _tenant_ids[slug]


TenantDep = Annotated[uuid.UUID, Depends(get_tenant_id)]


def require_roles(*roles: Role):
    def check(principal: PrincipalDep) -> Principal:
        if principal.role not in roles:
            raise HTTPException(status.HTTP_403_FORBIDDEN, f"Requires role: {' or '.join(roles)}")
        return principal

    return Depends(check)


def authorize_customer(principal: Principal, customer_ref: str) -> None:
    """Customers may only act on themselves; staff and admins on anyone in their tenant."""
    if not principal.can_access_customer(customer_ref):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Not allowed for this customer")


def check_customer_path(customer_ref: str, principal: PrincipalDep) -> None:
    """For routes with {customer_ref} in the path."""
    authorize_customer(principal, customer_ref)


async def rate_limit(request: Request, response: Response, principal: PrincipalDep, redis: RedisDep) -> None:
    if not get_settings().rate_limit_enabled:
        return
    limiter = getattr(request.app.state, "rate_limiter", None) or RateLimiter(redis)
    try:
        decision = await limiter.check(principal.tenant, principal.subject)
    except Exception:
        # Fail open: an unavailable limiter shouldn't take the whole API down with it.
        logger.warning("Rate limiter unavailable; allowing request", exc_info=True)
        return
    response.headers["X-RateLimit-Remaining"] = str(decision.remaining)
    if not decision.allowed:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            f"Rate limit exceeded ({decision.scope}); retry in {decision.retry_after}s",
            headers={"Retry-After": str(decision.retry_after)},
        )


RateLimitDep = Depends(rate_limit)
