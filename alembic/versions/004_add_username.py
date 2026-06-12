"""004 add username to users table.

Revision ID: 004
Revises: 003
Create Date: 2026-06-12
"""
from alembic import op
import sqlalchemy as sa


revision = "004"
down_revision = "003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("users") as batch:
        batch.add_column(sa.Column("username", sa.String(128), nullable=True, unique=True))


def downgrade() -> None:
    with op.batch_alter_table("users") as batch:
        batch.drop_column("username")
