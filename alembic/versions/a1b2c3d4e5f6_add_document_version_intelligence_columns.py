"""add document version intelligence columns

Adds documents.supersedes_id and documents.version, backing the Document
Version Intelligence feature (app/rag/version_intelligence.py). These were
added to app/core/models.py's Document model, but nothing previously
migrated existing `documents` tables to match — app/main.py only ran
Base.metadata.create_all(), which creates missing TABLES, not missing
COLUMNS on tables that already existed. That mismatch is what caused
"psycopg2.errors.UndefinedColumn: column 'supersedes_id' of relation
'documents' does not exist" in test_keyword_search_integration.py and
test_metadata.py.

Guarded with an existence check so this is a no-op on a fresh database
where create_all() already created the table with both columns (e.g. after
`docker compose down -v`), and additive on any pre-existing database that's
missing them.

Revision ID: a1b2c3d4e5f6
Revises:
Create Date: 2026-08-05

"""
from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_columns = {c["name"] for c in inspector.get_columns("documents")}

    if "supersedes_id" not in existing_columns:
        op.add_column(
            "documents",
            sa.Column("supersedes_id", postgresql.UUID(as_uuid=False), nullable=True),
        )
        op.create_foreign_key(
            "fk_documents_supersedes_id_documents",
            "documents",
            "documents",
            ["supersedes_id"],
            ["id"],
        )

    if "version" not in existing_columns:
        op.add_column(
            "documents",
            sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        )
        # Drop the server_default after backfilling existing rows so the
        # column matches the model exactly (default=1 is enforced at the
        # ORM level for new inserts, not as a DB-level server_default).
        op.alter_column("documents", "version", server_default=None)


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_columns = {c["name"] for c in inspector.get_columns("documents")}

    if "version" in existing_columns:
        op.drop_column("documents", "version")

    if "supersedes_id" in existing_columns:
        op.drop_constraint(
            "fk_documents_supersedes_id_documents", "documents", type_="foreignkey"
        )
        op.drop_column("documents", "supersedes_id")
