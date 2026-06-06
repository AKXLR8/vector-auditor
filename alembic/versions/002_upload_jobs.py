"""upload jobs

Revision ID: 002
Revises: 001
Create Date: 2026-06-06
"""
from alembic import op
import sqlalchemy as sa

revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "upload_jobs",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("user_id", sa.String(64), nullable=False),
        sa.Column("document_id", sa.String(64), nullable=False),
        sa.Column("filename", sa.String(512), nullable=False),
        sa.Column("content_path", sa.Text, nullable=False),
        sa.Column("state", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("attempts", sa.Integer, nullable=False, server_default="0"),
        sa.Column("max_attempts", sa.Integer, nullable=False, server_default="3"),
        sa.Column("last_error", sa.Text),
        sa.Column("enqueued_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.Column("next_run_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_upload_jobs_user_id", "upload_jobs", ["user_id"])
    op.create_index("ix_upload_jobs_document_id", "upload_jobs", ["document_id"])
    op.create_index("ix_upload_jobs_state", "upload_jobs", ["state"])
    op.create_index("ix_upload_jobs_state_next", "upload_jobs", ["state", "next_run_at"])


def downgrade() -> None:
    op.drop_table("upload_jobs")
