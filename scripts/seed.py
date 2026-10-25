"""Load demo tenants, customers, points and documents. Safe to run repeatedly.

    docker compose exec api python -m scripts.seed
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from app.db import dispose_engine, get_sessionmaker
from app.models import Customer, PointsLedger, Tenant
from app.services.ingestion import ingest_document
from app.services.parsing import SUPPORTED_EXTENSIONS

SEED_DIR = Path(__file__).resolve().parent.parent / "data" / "seed"
logger = logging.getLogger("seed")


async def seed() -> None:
    config = json.loads((SEED_DIR / "tenants.json").read_text())
    now = datetime.now(timezone.utc)

    async with get_sessionmaker()() as session:
        for t in config["tenants"]:
            await session.execute(
                insert(Tenant).values(slug=t["slug"], name=t["name"]).on_conflict_do_nothing(index_elements=["slug"])
            )
            tenant_id = await session.scalar(select(Tenant.id).where(Tenant.slug == t["slug"]))

            for c in t["customers"]:
                await session.execute(
                    insert(Customer)
                    .values(
                        tenant_id=tenant_id,
                        external_ref=c["external_ref"],
                        name=c["name"],
                        preferred_language=c["preferred_language"],
                    )
                    .on_conflict_do_nothing(constraint="uq_customers_tenant_ref")
                )
                customer_id = await session.scalar(
                    select(Customer.id).where(Customer.tenant_id == tenant_id, Customer.external_ref == c["external_ref"])
                )
                for entry in c["ledger"]:
                    days = entry.get("expires_in_days")
                    await session.execute(
                        insert(PointsLedger)
                        .values(
                            tenant_id=tenant_id,
                            customer_id=customer_id,
                            delta=entry["delta"],
                            reason=entry["reason"],
                            reference=entry["reference"],
                            expires_at=now + timedelta(days=days) if days is not None else None,
                        )
                        .on_conflict_do_nothing(constraint="uq_points_ledger_tenant_reference")
                    )
            await session.commit()

            doc_dir = SEED_DIR / t["slug"]
            for path in sorted(doc_dir.glob("*")) if doc_dir.exists() else []:
                if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                    continue
                result = await ingest_document(session, tenant_id, path.name, path.read_bytes())
                status = "ingested" if result.created else "already present"
                logger.info("[%s] %s: %s (%d chunks)", t["slug"], path.name, status, result.document.chunk_count)

    await dispose_engine()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    asyncio.run(seed())
