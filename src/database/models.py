"""SQLAlchemy ORM models — matches actual Neon schema."""
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    email: Mapped[str] = mapped_column(String(320), nullable=False, unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(256), nullable=False)
    roles: Mapped[str] = mapped_column(Text, default='["user"]', nullable=False)
    mfa_enabled: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    mfa_secret: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    display_name: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)


class BlacklistedToken(Base):
    __tablename__ = "blacklisted_tokens"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    token_jti: Mapped[str] = mapped_column(String(512), nullable=False, index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    has_pii: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    sha256: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    cloudinary_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    cloudinary_public_id: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="success", nullable=False, index=True)
    uploaded_by: Mapped[Optional[str]] = mapped_column(String(256), nullable=True, index=True)
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True, index=True)


class Session(Base):
    __tablename__ = "chat_sessions"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    title: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    user_id: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class Message(Base):
    __tablename__ = "chat_messages"

    # Neon uses string (uuid hex) for id, not bigint autoincrement
    id: Mapped[str] = mapped_column(String, primary_key=True)
    session_id: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    citations: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    reasoning_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    tokens_used: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    cost_usd: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    query_id: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    feedback: Mapped[Optional[str]] = mapped_column(String(8), nullable=True)
    verification: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class Query(Base):
    __tablename__ = "queries"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    citations: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    reasoning_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    tokens_used: Mapped[Optional[int]] = mapped_column(Integer, default=0, nullable=True)
    cost_usd: Mapped[Optional[float]] = mapped_column(Float, default=0, nullable=True)
    user_id: Mapped[Optional[str]] = mapped_column(String(256), nullable=True, index=True)
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    query_id: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    thumbs_up: Mapped[bool] = mapped_column(Boolean, nullable=False)
    comment: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    user_id: Mapped[Optional[str]] = mapped_column(String(256), nullable=True, index=True)
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class DeadLetter(Base):
    __tablename__ = "dead_letter_queue"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    source: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    filename: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    payload: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    retry_count: Mapped[Optional[int]] = mapped_column(Integer, default=0, nullable=True)
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class UploadJob(Base):
    """Tracks per-document upload pipeline progress.

    Note: in the live Neon schema, this table is named `upload_progress` (not
    `upload_jobs`). Field names match: `stage`, `progress`, `document_id`,
    `user_id`, plus `error` and timestamps.
    """
    __tablename__ = "upload_progress"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    stage: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    progress: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    document_id: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    user_id: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


Index("ix_documents_uploaded_by_created", Document.uploaded_by, Document.created_at.desc())
