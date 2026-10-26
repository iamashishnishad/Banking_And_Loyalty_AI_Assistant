from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import SessionDep, TenantDep, check_customer_path
from app.schemas import BalanceOut, RedeemIn, RedeemOut
from app.services.loyalty import (
    CustomerNotFoundError,
    DuplicateRedemptionError,
    InsufficientPointsError,
    get_balance,
    redeem_points,
)

# Customers may only see and spend their own points; staff and admins any customer in the tenant.
router = APIRouter(prefix="/customers/{customer_ref}", tags=["loyalty"], dependencies=[Depends(check_customer_path)])


@router.get("/points", response_model=BalanceOut)
async def points_balance(customer_ref: str, session: SessionDep, tenant_id: TenantDep) -> BalanceOut:
    try:
        return BalanceOut(**asdict(await get_balance(session, tenant_id, customer_ref)))
    except CustomerNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Customer not found") from exc


@router.post("/redeem", response_model=RedeemOut)
async def redeem(customer_ref: str, body: RedeemIn, session: SessionDep, tenant_id: TenantDep) -> RedeemOut:
    try:
        new_balance = await redeem_points(session, tenant_id, customer_ref, body.points, body.reason, body.reference)
    except CustomerNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Customer not found") from exc
    except InsufficientPointsError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    except DuplicateRedemptionError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, f"Redemption '{body.reference}' already processed") from exc
    return RedeemOut(customer_ref=customer_ref, redeemed=body.points, new_balance=new_balance)
