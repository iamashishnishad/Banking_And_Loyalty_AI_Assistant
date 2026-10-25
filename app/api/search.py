from dataclasses import asdict

from fastapi import APIRouter

from app.api.deps import RedisDep, SessionDep, TenantDep
from app.schemas import SearchHitOut, SearchIn, SearchOut
from app.services.search import hybrid_search

router = APIRouter(tags=["search"])


@router.post("/search", response_model=SearchOut)
async def search(body: SearchIn, session: SessionDep, tenant_id: TenantDep, redis: RedisDep) -> SearchOut:
    result = await hybrid_search(session, tenant_id, body.query, top_k=body.top_k, mode=body.mode, redis=redis)
    return SearchOut(
        query=body.query,
        mode=result.mode,
        took_ms=round(result.took_ms, 2),
        hits=[SearchHitOut(**asdict(h)) for h in result.hits],
    )
