import uuid
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request
from redis.asyncio import Redis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db import get_session
from app.models import Tenant

SessionDep = Annotated[AsyncSession, Depends(get_session)]

# Tenant slugs never change id, so resolve each one once per process.
_tenant_ids: dict[str, uuid.UUID] = {}


async def get_tenant_id(
    session: SessionDep,
    x_tenant: Annotated[str | None, Header(description="Tenant slug, e.g. demo-bank")] = None,
) -> uuid.UUID:
    """Phase 1 takes the tenant from the X-Tenant header. Phase 3 takes it from the JWT instead,
    so callers can no longer pick another bank's tenant."""
    slug = x_tenant or get_settings().default_tenant
    if slug not in _tenant_ids:
        tenant_id = await session.scalar(select(Tenant.id).where(Tenant.slug == slug))
        if tenant_id is None:
            raise HTTPException(status_code=404, detail=f"Unknown tenant '{slug}'")
        _tenant_ids[slug] = tenant_id
    return _tenant_ids[slug]


def get_redis(request: Request) -> Redis:
    return request.app.state.redis


TenantDep = Annotated[uuid.UUID, Depends(get_tenant_id)]
RedisDep = Annotated[Redis, Depends(get_redis)]
