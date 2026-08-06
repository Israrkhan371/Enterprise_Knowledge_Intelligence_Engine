"""add usage_logs answer-review columns

Adds usage_logs.sources, usage_logs.citation_verified, usage_logs.citation_flags,
usage_logs.flagged_for_review, and usage_logs.reviewed, backing the AI-answer
review admin feature (app/admin/routes.py's GET /admin/answers,
GET /admin/answers/{id}, GET /admin/answers/{id}/review-history,
POST /admin/answers/{id}/review). These were added to UsageLog in
app/core/models.py, but nothing migrated existing `usage_logs` tables to
match -- the exact same create_all()-only-creates-missing-tables gap that
a1b2c3d4e5f6 already fixed once for documents.supersedes_id/version, just
missed here. Caused "psycopg2.errors.UndefinedColumn: column
usage_logs.sources does not exist" on GET /admin/answers (and the other
three answer-review endpoints, which query the same table/columns).

Notably, POST /ask's INSERT into these same columns can still succeed on a
DB missing them if that INSERT runs against a session/connection that
hasn't yet re-read the table's catalog metadata, or if it ran before this
drift was introduced in a given environment -- SELECTs immediately surface
it because they always reflect live catalog state. Either way, the fix is
the same: migrate the table to match the model, don't rely on ordering
coincidences.

Guarded with an existence check so this is a no-op on a fresh database
where create_all() already created the table with all five columns (e.g.
after `docker compose down -v`), and additive on any pre-existing database
that's missing some or all of them.

Revision ID: b7c8d9e0f1a2
Revises: a1b2c3d4e5f6
Create Date: 2026-08-06

"""
from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b7c8d9e0f1a2"
down_revision: Union[str, None] = "a1b2c3d4e5f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_columns = {c["name"] for c in inspector.get_columns("usage_logs")}

    if "sources" not in existing_columns:
        op.add_column("usage_logs", sa.Column("sources", sa.JSON(), nullable=True))

    if "citation_verified" not in existing_columns:
        op.add_column("usage_logs", sa.Column("citation_verified", sa.Boolean(), nullable=True))

    if "citation_flags" not in existing_columns:
        op.add_column("usage_logs", sa.Column("citation_flags", sa.JSON(), nullable=True))

    if "flagged_for_review" not in existing_columns:
        op.add_column(
            "usage_logs",
            sa.Column("flagged_for_review", sa.Boolean(), nullable=False, server_default="false"),
        )
        # Drop the server_default after backfilling existing rows so the
        # column matches the model exactly (default=False is enforced at
        # the ORM level for new inserts, not as a DB-level server_default),
        # same pattern as a1b2c3d4e5f6's documents.version.
        op.alter_column("usage_logs", "flagged_for_review", server_default=None)

    if "reviewed" not in existing_columns:
        op.add_column(
            "usage_logs",
            sa.Column("reviewed", sa.Boolean(), nullable=False, server_default="false"),
        )
        op.alter_column("usage_logs", "reviewed", server_default=None)


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_columns = {c["name"] for c in inspector.get_columns("usage_logs")}

    for col in ("reviewed", "flagged_for_review", "citation_flags", "citation_verified", "sources"):
        if col in existing_columns:
            op.drop_column("usage_logs", col)
