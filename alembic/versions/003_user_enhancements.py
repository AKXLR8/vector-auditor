"""003 user enhancements + document/dlq/message refactor.

Revision ID: 003
Revises: 002
Create Date: 2026-06-07
"""
from alembic import op
import sqlalchemy as sa


revision = "003"
down_revision = "002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # User enhancements
    with op.batch_alter_table("users") as batch:
        batch.add_column(sa.Column("roles", sa.Text, nullable=False, server_default='["user"]'))
        batch.add_column(sa.Column("first_name", sa.String(120), nullable=True))
        batch.add_column(sa.Column("last_name", sa.String(120), nullable=True))
        batch.add_column(sa.Column("mfa_enabled", sa.Boolean, nullable=False, server_default=sa.false()))
        batch.add_column(sa.Column("mfa_secret", sa.String(64), nullable=True))
        batch.alter_column("role", server_default="user")

    # Document enhancements
    with op.batch_alter_table("documents") as batch:
        batch.add_column(sa.Column("cloudinary_url", sa.Text, nullable=True))
        batch.add_column(sa.Column("sha256", sa.String(64), nullable=False, server_default=""))
        batch.add_column(sa.Column("has_pii", sa.Boolean, nullable=False, server_default=sa.false()))

    # Message enhancements
    with op.batch_alter_table("chat_messages") as batch:
        batch.add_column(sa.Column("reasoning_path", sa.Text, nullable=True))
        batch.add_column(sa.Column("tokens_used", sa.Integer, nullable=True))
        batch.add_column(sa.Column("cost_usd", sa.Float, nullable=True))
        batch.add_column(sa.Column("query_id", sa.String(64), nullable=True))
        batch.add_column(sa.Column("feedback", sa.String(16), nullable=True))
        batch.add_column(sa.Column("verification", sa.Text, nullable=True))
    op.create_index("ix_chat_messages_query_id", "chat_messages", ["query_id"])

    # Query: use string id (uuid) for new shape; backfill from auto-increment int.
    # Add new columns first; old `id` column will be replaced in a follow-up.
    with op.batch_alter_table("queries") as batch:
        batch.add_column(sa.Column("reasoning_path", sa.Text, nullable=True))
        batch.add_column(sa.Column("tokens_used", sa.Integer, nullable=True))
        batch.add_column(sa.Column("verification", sa.Text, nullable=True))

    # Feedback: rating → thumbs_up, query_id → string
    with op.batch_alter_table("feedback") as batch:
        batch.add_column(sa.Column("thumbs_up", sa.Boolean, nullable=False, server_default=sa.true()))
        batch.add_column(sa.Column("query_id_str", sa.String(64), nullable=True))
    # Backfill: if old `query_id` is set, copy as string. Then drop & rename.
    op.execute("UPDATE feedback SET query_id_str = CAST(query_id AS VARCHAR(64)) WHERE query_id IS NOT NULL")
    with op.batch_alter_table("feedback") as batch:
        batch.drop_column("query_id")
        batch.drop_column("rating")
        batch.alter_column("query_id_str", new_column_name="query_id", nullable=False)
    op.create_index("ix_feedback_query_id_str", "feedback", ["query_id"])

    # DeadLetter: source → task, id → string, created_at → failed_at
    # Add new columns, backfill, drop, rename
    op.execute("ALTER TABLE dead_letter_queue ADD COLUMN task VARCHAR(120)")
    op.execute("UPDATE dead_letter_queue SET task = source WHERE task IS NULL")
    op.execute("ALTER TABLE dead_letter_queue ALTER COLUMN task SET NOT NULL")
    op.execute("ALTER TABLE dead_letter_queue ADD COLUMN id_str VARCHAR(64)")
    op.execute("UPDATE dead_letter_queue SET id_str = CAST(id AS VARCHAR(64)) WHERE id_str IS NULL")
    op.execute("ALTER TABLE dead_letter_queue ADD COLUMN failed_at TIMESTAMPTZ")
    op.execute("UPDATE dead_letter_queue SET failed_at = created_at WHERE failed_at IS NULL")
    with op.batch_alter_table("dead_letter_queue") as batch:
        batch.drop_column("id")
        batch.drop_column("source")
        batch.drop_column("created_at")
        batch.alter_column("id_str", new_column_name="id", nullable=False)
        batch.alter_column("failed_at", nullable=False, server_default=sa.func.now())

    # UploadJob: state → stage, enqueued_at → created_at, completed_at → updated_at
    op.execute("ALTER TABLE upload_jobs ADD COLUMN stage VARCHAR(20)")
    op.execute("UPDATE upload_jobs SET stage = state WHERE stage IS NULL")
    op.execute("ALTER TABLE upload_jobs ALTER COLUMN stage SET NOT NULL")
    op.execute("ALTER TABLE upload_jobs ADD COLUMN progress INTEGER NOT NULL DEFAULT 0")
    op.execute("ALTER TABLE upload_jobs ADD COLUMN updated_at TIMESTAMPTZ")
    op.execute("UPDATE upload_jobs SET updated_at = completed_at WHERE updated_at IS NULL AND completed_at IS NOT NULL")
    op.execute("UPDATE upload_jobs SET updated_at = started_at WHERE updated_at IS NULL AND started_at IS NOT NULL")
    op.execute("UPDATE upload_jobs SET updated_at = created_at WHERE updated_at IS NULL")
    op.execute("ALTER TABLE upload_jobs ADD COLUMN created_at_new TIMESTAMPTZ")
    op.execute("UPDATE upload_jobs SET created_at_new = COALESCE(enqueued_at, CURRENT_TIMESTAMP)")
    with op.batch_alter_table("upload_jobs") as batch:
        batch.drop_column("state")
        batch.drop_column("last_error")
        batch.drop_column("enqueued_at")
        batch.drop_column("completed_at")
        batch.alter_column("created_at_new", new_column_name="created_at", nullable=False, server_default=sa.func.now())
        batch.alter_column("updated_at", nullable=True)


def downgrade() -> None:
    # Conservative downgrade: nothing safe to undo. Leave new columns in place.
    pass
