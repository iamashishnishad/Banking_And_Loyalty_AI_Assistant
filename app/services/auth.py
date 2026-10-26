"""JWT bearer tokens: who is calling, for which bank (tenant), and in what role.

In production an identity provider issues these; here scripts/issue_token.py mints them with the
shared HS256 secret. Roles:

    customer  only their own customer_ref (chat, voice, points, redeem, memory) + search
    staff     any customer of their tenant (e.g. a call-centre agent) + search
    admin     staff rights + document upload/delete and job status
"""

import time
import uuid
from dataclasses import dataclass
from typing import Literal, get_args

import jwt

from app.config import get_settings

Role = Literal["customer", "staff", "admin"]
_ROLES = set(get_args(Role))
_ALGORITHM = "HS256"
_MIN_SECRET_LENGTH = 32


class AuthError(Exception):
    pass


@dataclass(frozen=True)
class Principal:
    subject: str
    tenant: str  # slug
    role: Role
    customer_ref: str | None  # always set for customers

    def can_access_customer(self, customer_ref: str) -> bool:
        return self.role != "customer" or self.customer_ref == customer_ref


def check_secret() -> None:
    """Called at startup so a missing or weak secret fails loudly instead of on the first request."""
    if len(get_settings().jwt_secret) < _MIN_SECRET_LENGTH:
        raise RuntimeError(f"JWT_SECRET must be set to at least {_MIN_SECRET_LENGTH} characters.")


def issue_token(
    *,
    tenant: str,
    role: Role,
    customer_ref: str | None = None,
    subject: str | None = None,
    ttl_seconds: int | None = None,
) -> str:
    if role not in _ROLES:
        raise ValueError(f"Unknown role '{role}'")
    if role == "customer" and not customer_ref:
        raise ValueError("A customer token needs customer_ref")
    s = get_settings()
    now = int(time.time())
    claims = {
        "sub": subject or (f"customer:{customer_ref}" if role == "customer" else f"{role}:{uuid.uuid4().hex[:8]}"),
        "tenant": tenant,
        "role": role,
        "iss": s.jwt_issuer,
        "aud": s.jwt_audience,
        "iat": now,
        "exp": now + (ttl_seconds if ttl_seconds is not None else s.jwt_ttl_seconds),
    }
    if customer_ref:
        claims["customer_ref"] = customer_ref
    return jwt.encode(claims, s.jwt_secret, algorithm=_ALGORITHM)


def decode_token(token: str) -> Principal:
    s = get_settings()
    try:
        claims = jwt.decode(
            token,
            s.jwt_secret,
            algorithms=[_ALGORITHM],  # pinned: never let the token pick its own algorithm
            audience=s.jwt_audience,
            issuer=s.jwt_issuer,
            leeway=30,
            options={"require": ["exp", "iat", "sub", "iss", "aud"]},
        )
    except jwt.ExpiredSignatureError as exc:
        raise AuthError("Token expired") from exc
    except jwt.InvalidTokenError as exc:
        raise AuthError("Invalid token") from exc

    role, tenant, customer_ref = claims.get("role"), claims.get("tenant"), claims.get("customer_ref")
    if role not in _ROLES or not isinstance(tenant, str) or not tenant:
        raise AuthError("Invalid token claims")
    if role == "customer" and not customer_ref:
        raise AuthError("Customer token without customer_ref")
    return Principal(subject=claims["sub"], tenant=tenant, role=role, customer_ref=customer_ref)
