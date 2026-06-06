"""initial schema

Revision ID: 001
Revises:
Create Date: 2026-06-06
"""
from alembic import op
import sqlalchemy as sa

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("email", sa.String(320), unique=True),
        sa.Column("username", sa.String(120)),
        sa.Column("password_hash", sa.String(255)),
        sa.Column("role", sa.String(20), nullable=False, server_default="viewer"),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)
    op.create_index("ix_users_username", "users", ["username"])

    op.create_table(
        "token_blacklist",
        sa.Column("jti", sa.String(64), primary_key=True),
        sa.Column("user_id", sa.String(64), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_token_blacklist_user_id", "token_blacklist", ["user_id"])

    op.create_table(
        "documents",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("user_id", sa.String(64), nullable=False),
        sa.Column("filename", sa.String(512), nullable=False),
        sa.Column("file_size", sa.BigInteger, nullable=False, server_default="0"),
        sa.Column("mime_type", sa.String(120)),
        sa.Column("storage_url", sa.Text),
        sa.Column("status", sa.String(32), nullable=False, server_default="processing"),
        sa.Column("chunk_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("error", sa.Text),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_documents_user_id", "documents", ["user_id"])
    op.create_index("ix_documents_status", "documents", ["status"])
    op.create_index("ix_documents_user_created", "documents", ["user_id", "created_at"])

    op.create_table(
        "chat_sessions",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("user_id", sa.String(64), nullable=False),
        sa.Column("title", sa.String(255), nullable=False, server_default="New chat"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_chat_sessions_user_id", "chat_sessions", ["user_id"])

    op.create_table(
        "chat_messages",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("session_id", sa.String(64), sa.ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("role", sa.String(20), nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("citations", sa.Text),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_chat_messages_session_id", "chat_messages", ["session_id"])

    op.create_table(
        "queries",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.String(64), nullable=False),
        sa.Column("question", sa.Text, nullable=False),
        sa.Column("answer", sa.Text),
        sa.Column("mode", sa.String(20), nullable=False, server_default="white_box"),
        sa.Column("latency_ms", sa.Integer),
        sa.Column("cost_usd", sa.Float),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_queries_user_id", "queries", ["user_id"])

    op.create_table(
        "feedback",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.String(64), nullable=False),
        sa.Column("query_id", sa.BigInteger, nullable=False),
        sa.Column("rating", sa.Integer, nullable=False),
        sa.Column("comment", sa.Text),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_feedback_user_id", "feedback", ["user_id"])
    op.create_index("ix_feedback_query_id", "feedback", ["query_id"])

    op.create_table(
        "dead_letter_queue",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("source", sa.String(120), nullable=False),
        sa.Column("payload", sa.Text, nullable=False),
        sa.Column("error", sa.Text, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_dead_letter_source", "dead_letter_queue", ["source"])


def downgrade() -> None:
    op.drop_table("dead_letter_queue")
    op.drop_table("feedback")
    op.drop_table("queries")
    op.drop_table("chat_messages")
    op.drop_table("chat_sessions")
    op.drop_table("documents")
    op.drop_table("token_blacklist")
    op.drop_table("users")
