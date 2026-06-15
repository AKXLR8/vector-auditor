"""Pydantic request/response schemas — matches frontend BACKEND_API.md spec."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated, Optional

from pydantic import BaseModel, BeforeValidator, EmailStr, Field


def _dt_to_str(v):
    if v is None:
        return None
    if isinstance(v, datetime):
        return v.isoformat()
    return str(v)


DatetimeOrStr = Annotated[Optional[str], BeforeValidator(_dt_to_str)]


# ── Enums ──────────────────────────────────────────────────────────────────

class Mode(str, Enum):
    white_box = "white_box"
    black_box = "black_box"


class Role(str, Enum):
    user = "user"
    admin = "admin"


# ── Auth ───────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=200)
    username: Optional[str] = Field(default=None, max_length=128)
    first_name: Optional[str] = Field(default=None, max_length=120)
    last_name: Optional[str] = Field(default=None, max_length=120)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class LoginResponse(BaseModel):
    access_token: str
    user_id: str
    email: str = ""
    username: Optional[str] = None
    display_name: Optional[str] = None
    roles: list[str] = []


class RefreshResponse(BaseModel):
    access_token: str


class MFALoginRequest(BaseModel):
    code: str


class MFASetupResponse(BaseModel):
    secret: str
    uri: str
    qr_code_url: str


class MFAVerifyRequest(BaseModel):
    code: str


class OAuthConfigResponse(BaseModel):
    github_client_id: str


class GitHubOAuthRequest(BaseModel):
    code: str


# ── User ───────────────────────────────────────────────────────────────────

class UserResponse(BaseModel):
    id: str
    email: Optional[str] = None
    username: Optional[str] = None
    display_name: Optional[str] = None
    roles: list[str] = []
    mfa_enabled: bool = False
    created_at: DatetimeOrStr = None


# ── Health ─────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str  # "ok" | "degraded" | "down"
    version: str
    timestamp: str
    checks: dict[str, str] = {}


class ReadyResponse(BaseModel):
    ready: bool
    reason: Optional[str] = None


# ── Query ──────────────────────────────────────────────────────────────────

class MessageHistory(BaseModel):
    role: str
    content: str


class Citation(BaseModel):
    quote: str
    source: str
    location: str
    page: Optional[int] = None
    document_id: Optional[str] = None
    bboxes: Optional[list[list[float]]] = None
    page_width: Optional[float] = None
    page_height: Optional[float] = None


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    document_ids: Optional[list[str]] = None
    conversation_history: Optional[list[MessageHistory]] = None
    mode: Mode = Mode.white_box
    max_citations: Optional[int] = Field(default=None, ge=1, le=100)
    model: Optional[str] = Field(default=None, description="LLM profile key (mercury, gpt5-mini, gpt5, claude-haiku, claude-opus, openrouter)")


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation] = []
    reasoning_path: list[str] = []
    tokens_used: int = 0
    cost_usd: float = 0.0
    query_id: str
    timestamp: str
    verification: Optional[str] = None
    mode: Optional[Mode] = None


# ── Analysis ───────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    question: Optional[str] = None
    document_ids: Optional[list[str]] = None
    max_citations: Optional[int] = Field(default=None, ge=1, le=100)
    model: Optional[str] = Field(default=None, description="LLM profile key")


class CrossDocComparison(BaseModel):
    common_themes: list[str] = []
    differences: list[str] = []
    complementary_insights: list[str] = []


class DocumentAnalysis(BaseModel):
    summary: str
    key_findings: list[str] = []
    methodology: str = ""
    research_gaps: list[str] = []
    contradictions: list[str] = []
    open_questions: list[str] = []
    limitations: str = ""
    confidence: str = "moderate"  # "high" | "moderate" | "low"
    citations: list[Citation] = []
    documents_analyzed: list[str] = []
    cross_document_comparison: Optional[CrossDocComparison] = None
    per_document_summary: Optional[dict[str, str]] = None


# ── Documents ──────────────────────────────────────────────────────────────

class DocumentResponse(BaseModel):
    id: str
    document_id: Optional[str] = None
    filename: str
    status: str  # "processing" | "ready" | "failed" | "duplicate" | "skipped" | "stuck"
    has_pii: bool = False
    sha256: str = ""
    cloudinary_url: Optional[str] = None
    uploaded_by: str
    created_at: DatetimeOrStr = None


class UploadedDocument(BaseModel):
    upload_id: str
    document_id: str
    filename: str
    status: str


class UploadResponse(BaseModel):
    uploaded_documents: list[UploadedDocument]


class UploadStatusResponse(BaseModel):
    id: str
    filename: str
    stage: str  # "uploading" | "extracting" | "chunking" | "embedding" | "indexing" | "completed" | "failed" | "duplicate" | "skipped" | "stuck"
    progress: int = 0
    error: Optional[str] = None
    document_id: Optional[str] = None
    user_id: str
    created_at: DatetimeOrStr = None
    updated_at: DatetimeOrStr = None


# ── Sessions ───────────────────────────────────────────────────────────────

class CreateSessionRequest(BaseModel):
    title: Optional[str] = "New chat"
    id: Optional[str] = None


class UpdateSessionRequest(BaseModel):
    title: str = Field(min_length=1, max_length=255)


class AddMessageRequest(BaseModel):
    role: str  # "user" | "assistant"
    content: str
    citations: Optional[list[Citation]] = None
    reasoning_path: Optional[list[str]] = None
    tokens_used: Optional[int] = None
    cost_usd: Optional[float] = None
    query_id: Optional[str] = None
    verification: Optional[str] = None


class SessionResponse(BaseModel):
    id: str
    title: Optional[str] = None
    user_id: str
    created_at: DatetimeOrStr = None
    updated_at: DatetimeOrStr = None


class MessageResponse(BaseModel):
    id: str
    session_id: str
    role: str
    content: str
    citations: Optional[list[Citation]] = None
    reasoning_path: Optional[list[str]] = None
    tokens_used: Optional[int] = None
    cost_usd: Optional[float] = None
    query_id: Optional[str] = None
    feedback: Optional[str] = None
    verification: Optional[str] = None
    created_at: DatetimeOrStr = None


class SessionDetailResponse(SessionResponse):
    messages: list[MessageResponse] = []


class SessionsListResponse(BaseModel):
    sessions: list[SessionResponse]


class MessagesListResponse(BaseModel):
    messages: list[MessageResponse]


# ── Feedback ───────────────────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    query_id: str  # string, not int
    thumbs_up: bool
    comment: Optional[str] = Field(default=None, max_length=2000)


# ── DLQ ────────────────────────────────────────────────────────────────────

class DLQItem(BaseModel):
    id: str
    task: str
    payload: str
    error: str
    failed_at: Optional[str] = None


class DLQResponse(BaseModel):
    dead_letter_queue: list[DLQItem] = []


# ── NexAGI ────────────────────────────────────────────────────────────────

class NexAGIMessage(BaseModel):
    role: str
    content: str | None = None
    reasoning_details: Optional[list[dict]] = None

class NexAGIRequest(BaseModel):
    messages: list[NexAGIMessage]
    model: str = "nex-agi/nex-n2-pro:free"
    reasoning: bool = True

class NexAGIResponse(BaseModel):
    content: str | None = None
    reasoning_details: Optional[dict] = None
    model: str = ""
    usage: Optional[dict] = None
