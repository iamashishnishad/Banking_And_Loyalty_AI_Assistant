"""Initial schema: tenants, documents, chunks (pgvector + full-text), customers, points ledger.

Revision ID: 0001
Revises:
Create Date: 2026-10-26
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Fixed here on purpose: a migration must produce the same schema whatever the current env says.
# multilingual-e5-small = 384. Switching to bge-m3 (1024) means a new migration + re-ingestion.
EMBEDDING_DIM = 384


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "tenants",
        sa.Column("id", sa.UUID(), primary_key=True),
        sa.Column("slug", sa.String(64), nullable=False, unique=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    op.create_table(
        "documents",
        sa.Column("id", sa.UUID(), primary_key=True),
        sa.Column("tenant_id", sa.UUID(), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False),
        sa.Column("title", sa.String(300), nullable=False),
        sa.Column("source_filename", sa.String(300), nullable=False),
        sa.Column("mime_type", sa.String(100), nullable=False),
        sa.Column("content_sha256", sa.String(64), nullable=False),
        sa.Column("chunk_count", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("tenant_id", "content_sha256", name="uq_documents_tenant_sha"),
    )
    op.create_index("ix_documents_tenant_id", "documents", ["tenant_id"])

    op.create_table(
        "chunks",
        sa.Column("id", sa.UUID(), primary_key=True),
        sa.Column("document_id", sa.UUID(), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("tenant_id", sa.UUID(), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("lexical_text", sa.Text(), nullable=False),
        sa.Column(
            "tsv",
            postgresql.TSVECTOR(),
            sa.Computed("to_tsvector('simple', lexical_text)", persisted=True),
        ),
        sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=False),
        sa.Column("metadata", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_chunks_document_id", "chunks", ["document_id"])
    op.create_index("ix_chunks_tenant_id", "chunks", ["tenant_id"])
    op.create_index("ix_chunks_tsv", "chunks", ["tsv"], postgresql_using="gin")
    op.create_index(
        "ix_chunks_embedding_hnsw",
        "chunks",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )

    op.create_table(
        "customers",
        sa.Column("id", sa.UUID(), primary_key=True),
        sa.Column("tenant_id", sa.UUID(), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False),
        sa.Column("external_ref", sa.String(100), nullable=False),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("preferred_language", sa.String(16), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("tenant_id", "external_ref", name="uq_customers_tenant_ref"),
    )
    op.create_index("ix_customers_tenant_id", "customers", ["tenant_id"])

    op.create_table(
        "points_ledger",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("tenant_id", sa.UUID(), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False),
        sa.Column("customer_id", sa.UUID(), sa.ForeignKey("customers.id", ondelete="CASCADE"), nullable=False),
        sa.Column("delta", sa.Integer(), nullable=False),
        sa.Column("reason", sa.String(200), nullable=False),
        sa.Column("reference", sa.String(100), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("tenant_id", "reference", name="uq_points_ledger_tenant_reference"),
    )
    op.create_index("ix_points_ledger_customer_created", "points_ledger", ["customer_id", "created_at"])


def downgrade() -> None:
    op.drop_table("points_ledger")
    op.drop_table("customers")
    op.drop_table("chunks")
    op.drop_table("documents")
    op.drop_table("tenants")
