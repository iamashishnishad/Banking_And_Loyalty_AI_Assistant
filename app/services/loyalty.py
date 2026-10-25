"""Points balance and redemption on the append-only points ledger."""

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Customer, PointsLedger


class CustomerNotFoundError(LookupError):
    pass


class InsufficientPointsError(ValueError):
    def __init__(self, balance: int, requested: int) -> None:
        super().__init__(f"Insufficient points: balance {balance}, requested {requested}")
        self.balance = balance
        self.requested = requested


class DuplicateRedemptionError(ValueError):
    pass


@dataclass
class Balance:
    customer_ref: str
    customer_name: str
    points: int
    expiring_within_30_days: int


async def _get_customer(session: AsyncSession, tenant_id: uuid.UUID, customer_ref: str, lock: bool = False) -> Customer:
    stmt = select(Customer).where(Customer.tenant_id == tenant_id, Customer.external_ref == customer_ref)
    if lock:
        # Serialises redemptions per customer so two concurrent requests can't both spend the same points.
        stmt = stmt.with_for_update()
    customer = await session.scalar(stmt)
    if customer is None:
        raise CustomerNotFoundError(customer_ref)
    return customer


async def _points(session: AsyncSession, customer_id: uuid.UUID) -> int:
    total = await session.scalar(
        select(func.coalesce(func.sum(PointsLedger.delta), 0)).where(PointsLedger.customer_id == customer_id)
    )
    return int(total)


async def get_balance(session: AsyncSession, tenant_id: uuid.UUID, customer_ref: str) -> Balance:
    customer = await _get_customer(session, tenant_id, customer_ref)
    now = datetime.now(timezone.utc)
    # Informational only: earned batches with an expiry in the next 30 days. Exact FIFO expiry
    # accounting (which redemptions consumed which batch) is not modelled yet.
    expiring = await session.scalar(
        select(func.coalesce(func.sum(PointsLedger.delta), 0)).where(
            PointsLedger.customer_id == customer.id,
            PointsLedger.delta > 0,
            PointsLedger.expires_at > now,
            PointsLedger.expires_at <= now + timedelta(days=30),
        )
    )
    return Balance(
        customer_ref=customer.external_ref,
        customer_name=customer.name,
        points=await _points(session, customer.id),
        expiring_within_30_days=int(expiring),
    )


async def redeem_points(
    session: AsyncSession,
    tenant_id: uuid.UUID,
    customer_ref: str,
    points: int,
    reason: str,
    reference: str,
) -> int:
    """Debit points and return the new balance. `reference` is an idempotency key."""
    if points <= 0:
        raise ValueError("points must be positive")
    customer = await _get_customer(session, tenant_id, customer_ref, lock=True)
    balance = await _points(session, customer.id)
    if points > balance:
        await session.rollback()
        raise InsufficientPointsError(balance, points)
    session.add(
        PointsLedger(tenant_id=tenant_id, customer_id=customer.id, delta=-points, reason=reason, reference=reference)
    )
    try:
        await session.commit()
    except IntegrityError as exc:
        await session.rollback()
        raise DuplicateRedemptionError(reference) from exc
    return balance - points
