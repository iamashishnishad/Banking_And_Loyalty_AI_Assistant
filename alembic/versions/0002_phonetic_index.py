"""Phonetic full-text index on chunks for Hinglish spelling variants and cross-script matching.

Existing rows get an empty phonetic_text; run `python -m scripts.reindex_text` to backfill.
The key-building code lives in the app, so the migration doesn't import it.

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-26
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("chunks", sa.Column("phonetic_text", sa.Text(), nullable=False, server_default=""))
    op.add_column(
        "chunks",
        sa.Column(
            "tsv_phonetic",
            postgresql.TSVECTOR(),
            sa.Computed("to_tsvector('simple', phonetic_text)", persisted=True),
        ),
    )
    op.create_index("ix_chunks_tsv_phonetic", "chunks", ["tsv_phonetic"], postgresql_using="gin")


def downgrade() -> None:
    op.drop_index("ix_chunks_tsv_phonetic", table_name="chunks")
    op.drop_column("chunks", "tsv_phonetic")
    op.drop_column("chunks", "phonetic_text")
