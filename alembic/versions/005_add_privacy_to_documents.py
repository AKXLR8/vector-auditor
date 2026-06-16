"""005 add privacy column to documents.

Revision ID: 005
Revises: 004
Create Date: 2026-06-16
"""
from alembic import op
import sqlalchemy as sa


revision = "005"
down_revision = "004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("documents") as batch:
        batch.add_column(sa.Column("privacy", sa.Boolean, nullable=False, server_default=sa.false()))


def downgrade() -> None:
    with op.batch_alter_table("documents") as batch:
        batch.drop_column("privacy")
