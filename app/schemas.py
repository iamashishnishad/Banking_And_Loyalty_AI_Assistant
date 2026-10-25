import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class DocumentOut(BaseModel):
    id: uuid.UUID
    title: str
    source_filename: str
    mime_type: str
    chunk_count: int
    created_at: datetime

    model_config = {"from_attributes": True}


class IngestOut(BaseModel):
    document: DocumentOut
    created: bool


class SearchIn(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=50)
    mode: Literal["hybrid", "vector", "lexical"] = "hybrid"


class SearchHitOut(BaseModel):
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_title: str
    chunk_index: int
    content: str
    meta: dict
    score: float
    vector_rank: int | None
    lexical_rank: int | None


class SearchOut(BaseModel):
    query: str
    mode: str
    took_ms: float
    hits: list[SearchHitOut]


class BalanceOut(BaseModel):
    customer_ref: str
    customer_name: str
    points: int
    expiring_within_30_days: int


class RedeemIn(BaseModel):
    points: int = Field(gt=0, le=1_000_000)
    reason: str = Field(default="Reward redemption", max_length=200)
    reference: str = Field(min_length=1, max_length=100, description="Idempotency key for this redemption")


class RedeemOut(BaseModel):
    customer_ref: str
    redeemed: int
    new_balance: int


class ChatIn(BaseModel):
    customer_ref: str = Field(min_length=1, max_length=100)
    message: str = Field(min_length=1, max_length=2000)


class ChatOut(BaseModel):
    intent: str
    answer: str
    sources: list[SearchHitOut] = []
