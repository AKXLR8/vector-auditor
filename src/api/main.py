"""FastAPI app — all routes (matches frontend BACKEND_API.md spec)."""
import asyncio
import hashlib
import json
import logging
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

_PATH = Path(__file__).resolve().parent.parent.parent / ".env"
if _PATH.exists():
    load_dotenv(_PATH, override=False)

import httpx
from fastapi import (
    Depends,
    FastAPI,
    File,
    HTTPException,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from ..agents.document_agent import DocumentAgent
from ..database.repository import (
    add_feedback,
    add_message,
    blacklist_token,
    create_document,
    create_session,
    create_user,
    delete_document,
    delete_session,
    dlq_size,
    get_document,
    get_document_by_sha256,
    get_session,
    get_user_by_email,
    get_user_by_id,
    is_token_blacklisted,
    list_documents,
    list_dlq,
    list_messages,
    list_sessions,
    push_dlq,
    update_document,
    update_session_title,
)
from ..database.session import get_session_factory
from ..job_queue import (
    JobRecord,
    STAGE_CHUNKING,
    STAGE_EMBEDDING,
    STAGE_EXTRACTING,
    STAGE_FAILED,
    STAGE_INDEXING,
    STAGE_UPLOADING,
    get_worker,
)
from ..models.schemas import (
    AddMessageRequest,
    AnalyzeRequest,
    CreateSessionRequest,
    DLQItem,
    DLQResponse,
    DocumentAnalysis,
    DocumentResponse,
    FeedbackRequest,
    GitHubOAuthRequest,
    HealthResponse,
    LoginRequest,
    LoginResponse,
    MessageResponse,
    MessagesListResponse,
    MFALoginRequest,
    MFASetupResponse,
    MFAVerifyRequest,
    Mode,
    OAuthConfigResponse,
    QueryRequest,
    QueryResponse,
    ReadyResponse,
    RefreshResponse,
    RegisterRequest,
    SessionDetailResponse,
    SessionResponse,
    SessionsListResponse,
    UpdateSessionRequest,
    UploadResponse,
    UploadStatusResponse,
    UploadedDocument,
    UserResponse,
)
from ..observability import get_metrics, metrics_response, setup_observability
from ..services.cache import get_cache
from ..services.cloudinary import get_cloudinary
from ..services.document_parser import parse_document
from ..services.guardrails import get_guardrails
from ..vectorstore.Qdrant import get_vector_store
from ..version import DESCRIPTION, TITLE, VERSION
from .auth import (
    bearer_scheme,
    create_access_token,
    current_user,
    decode_token,
    hash_password,
    require_role,
    verify_password,
)
from .middleware import (
    MetricsMiddleware,
    RequestContextMiddleware,
    RequestLoggingMiddleware,
    SecurityHeadersMiddleware,
    ShutdownGateMiddleware,
)

logger = logging.getLogger("rga_auditor")

limiter = Limiter(key_func=get_remote_address, default_limits=["200/minute"])

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
MAX_FILE_SIZE = 50 * 1024 * 1024

_UNSAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_filename(name: Optional[str]) -> str:
    if not name:
        return ""
    base = os.path.basename(name.replace("\\", "/"))
    if not base or base in {".", ".."}:
        return ""
    return _UNSAFE_FILENAME_RE.sub("_", base).strip("._")[:255]


def _parse_roles(value) -> list[str]:
    """Resilient roles[] parser — handles list, JSON string, or legacy single value."""
    if not value:
        return ["user"]
    if isinstance(value, list):
        return [str(r) for r in value if r]
    if isinstance(value, str):
        try:
            v = json.loads(value)
            if isinstance(v, list):
                return [str(r) for r in v if r]
            return [str(v)] if v else ["user"]
        except Exception:
            return [value] if value else ["user"]
    return [str(value)]


def _display_name(u: dict) -> Optional[str]:
    if u.get("first_name") or u.get("last_name"):
        return " ".join(p for p in [u.get("first_name"), u.get("last_name")] if p).strip() or None
    if u.get("username"):
        return u["username"]
    if u.get("email"):
        return u["email"].split("@", 1)[0]
    return None


def _user_response(u: dict) -> UserResponse:
    return UserResponse(
        id=u["id"],
        email=u.get("email"),
        username=u.get("username"),
        display_name=u.get("display_name") or _display_name(u),
        roles=_parse_roles(u.get("roles") or "user"),
        mfa_enabled=u.get("mfa_enabled", False),
        created_at=u.get("created_at"),
    )


def _doc_response(d: dict, base_url: str = "") -> DocumentResponse:
    cu: Optional[str] = d.get("cloudinary_url")
    if cu and base_url:
        cu = f"{base_url.rstrip('/')}/documents/{d['id']}/pdf"
    return DocumentResponse(
        id=d["id"],
        document_id=d.get("id"),
        filename=d.get("filename", ""),
        status=d.get("status", "processing"),
        has_pii=d.get("has_pii", False),
        sha256=d.get("sha256") or "",
        cloudinary_url=cu,
        uploaded_by=d.get("uploaded_by", ""),
        created_at=d.get("created_at"),
    )


def create_app() -> FastAPI:
    setup_observability()

    settings_origins = os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:3000,http://localhost:3001,http://localhost:5173,"
        "http://127.0.0.1:3000,http://127.0.0.1:3001,http://127.0.0.1:5173",
    )
    origins = [o.strip() for o in settings_origins.split(",") if o.strip()]

    app = FastAPI(title=TITLE, description=DESCRIPTION, version=VERSION)

    app.state.limiter = limiter
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(MetricsMiddleware)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(ShutdownGateMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(RateLimitExceeded)
    async def _rate_limited(request: Request, exc: RateLimitExceeded):
        return JSONResponse({"detail": f"rate limit: {exc.detail}"}, status_code=429)

    # ── Health ────────────────────────────────────────────────────────────

    @app.get("/health", response_model=HealthResponse)
    async def health():
        checks: dict[str, str] = {}
        circuit_breaker_info: dict[str, dict] = {}

        try:
            get_cache()
            checks["cache"] = "ok"
        except Exception as e:
            checks["cache"] = f"error: {e}"

        try:
            get_vector_store()
            checks["vector_store"] = "ok"
        except Exception as e:
            checks["vector_store"] = f"error: {e}"

        try:
            vs = get_vector_store()
            qs = getattr(vs, "_search_cb", None)
            qi = getattr(vs, "_index_cb", None)
            if qs is not None:
                circuit_breaker_info["qdrant_search"] = {
                    "state": qs.state,
                    "failures": qs.failure_count,
                    "available": qs.is_available(),
                }
            if qi is not None:
                circuit_breaker_info["qdrant_index"] = {
                    "state": qi.state,
                    "failures": qi.failure_count,
                    "available": qi.is_available(),
                }
        except Exception:
            pass

        try:
            checks["database"] = "ok" if get_session_factory() else "degraded: in-memory fallback"
        except Exception as e:
            checks["database"] = f"error: {e}"

        try:
            get_cloudinary()
            checks["object_store"] = "ok"
        except Exception as e:
            checks["object_store"] = f"error: {e}"

        try:
            from ..services.llm import get_llm
            llm = get_llm()
            cb = getattr(llm, "_cb", None)
            if cb is not None:
                circuit_breaker_info["llm"] = {
                    "state": cb.state,
                    "failures": cb.failure_count,
                    "available": cb.is_available(),
                }
            checks["llm_provider"] = "ok" if llm.is_available() else "degraded: circuit open"
        except Exception as e:
            checks["llm_provider"] = f"error: {e}"

        try:
            from ..shutdown import get_shutdown_manager
            sm = get_shutdown_manager()
            checks["shutdown_gate"] = "ok" if not sm.is_shutting_down else "shutting_down"
        except Exception:
            checks["shutdown_gate"] = "ok"

        cb_status = "ok"
        for name, info in circuit_breaker_info.items():
            if not info.get("available", True):
                cb_status = "degraded"
                checks[f"circuit_breaker/{name}"] = info["state"]
                break
        if cb_status == "ok":
            checks["circuit_breakers"] = "ok"

        overall = "ok" if all(v == "ok" for v in checks.values()) else "degraded"
        if any(v.startswith("error") for v in checks.values()):
            overall = "down"

        return HealthResponse(
            status=overall,
            version=VERSION,
            timestamp=datetime.utcnow().isoformat() + "Z",
            checks=checks,
        )

    @app.get("/readyz", response_model=ReadyResponse)
    async def readyz():
        from ..shutdown import get_shutdown_manager
        if get_shutdown_manager().is_shutting_down:
            return JSONResponse({"ready": False, "reason": "shutting down"}, status_code=503)
        try:
            get_vector_store()
            get_cache()
        except Exception as e:
            return JSONResponse({"ready": False, "reason": str(e)}, status_code=503)
        return ReadyResponse(ready=True)

    @app.get("/metrics")
    async def metrics():
        body, ctype = metrics_response()
        return Response(content=body, media_type=ctype)

    # ── Auth ──────────────────────────────────────────────────────────────

    @app.post("/auth/register", response_model=UserResponse, status_code=201)
    @limiter.limit("5/minute")
    async def register(request: Request, body: RegisterRequest):
        sf = get_session_factory()
        async with _maybe_session(sf) as s:
            existing = await get_user_by_email(s, body.email)
            if existing:
                raise HTTPException(status_code=409, detail="email already registered")
            # Build display_name from first_name+last_name, fall back to email-local
            display = " ".join(p for p in [body.first_name, body.last_name] if p).strip() or None
            user = await create_user(
                s,
                email=body.email,
                hashed_password=hash_password(body.password),
                username=body.username,
                display_name=display,
                roles=["user"],
            )
        return _user_response(user)

    @app.post("/auth/login", response_model=LoginResponse)
    @limiter.limit("10/minute")
    async def login(request: Request, body: LoginRequest):
        sf = get_session_factory()
        async with _maybe_session(sf) as s:
            user = await get_user_by_email(s, body.email)
        if not user or not user.get("hashed_password") or not verify_password(body.password, user["hashed_password"]):
            raise HTTPException(status_code=401, detail="invalid credentials")
        roles = _parse_roles(user.get("roles") or "user")
        if user.get("mfa_enabled"):
            raise HTTPException(status_code=401, detail="mfa required", headers={"X-MFA-Required": "true"})
        token, _expires, _jti = create_access_token(user["id"], roles=roles)
        return LoginResponse(
            access_token=token,
            user_id=user["id"],
            email=user.get("email", ""),
            username=user.get("username"),
            display_name=user.get("display_name"),
            roles=roles,
        )

    @app.post("/auth/login/mfa", response_model=LoginResponse)
    @limiter.limit("10/minute")
    async def login_mfa(request: Request, body: MFALoginRequest):
        raise HTTPException(status_code=501, detail="MFA login not yet implemented")

    @app.post("/auth/logout", status_code=204)
    async def logout(request: Request, creds=Depends(bearer_scheme), user=Depends(current_user)):
        if creds is None:
            raise HTTPException(status_code=401, detail="missing token")
        payload = decode_token(creds.credentials)
        sf = get_session_factory()
        async with _maybe_session(sf) as s:
            await blacklist_token(s, payload.get("jti", ""), user["id"])
        return Response(status_code=204)

    @app.get("/auth/token/refresh", response_model=RefreshResponse)
    async def refresh_token(user=Depends(current_user)):
        roles = _parse_roles(user.get("roles") or user.get("role") or "user")
        token, _expires, _jti = create_access_token(user["id"], roles=roles)
        return RefreshResponse(access_token=token)

    @app.post("/auth/mfa/setup", response_model=MFASetupResponse)
    async def mfa_setup(user=Depends(current_user)):
        secret = uuid.uuid4().hex[:16].upper()
        uri = f"otpauth://totp/VectorAuditor:{user['id']}?secret={secret}&issuer=VectorAuditor"
        qr_code_url = f"https://api.qrserver.com/v1/create-qr-code/?data={uri}"
        return MFASetupResponse(secret=secret, uri=uri, qr_code_url=qr_code_url)

    @app.post("/auth/mfa/verify", response_model=UserResponse)
    async def mfa_verify(request: Request, body: MFAVerifyRequest, user=Depends(current_user)):
        raise HTTPException(status_code=501, detail="MFA verify not yet implemented")

    @app.get("/auth/oauth/config", response_model=OAuthConfigResponse)
    async def oauth_config():
        return OAuthConfigResponse(github_client_id=os.getenv("GITHUB_CLIENT_ID", ""))

    @app.post("/auth/oauth/github", response_model=LoginResponse)
    @limiter.limit("10/minute")
    async def oauth_github(request: Request, body: GitHubOAuthRequest):
        client_id = os.getenv("GITHUB_CLIENT_ID", "")
        client_secret = os.getenv("GITHUB_CLIENT_SECRET", "")
        if not (client_id and client_secret):
            raise HTTPException(status_code=503, detail="GitHub OAuth not configured")
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.post(
                "https://github.com/login/oauth/access_token",
                json={"client_id": client_id, "client_secret": client_secret, "code": body.code},
                headers={"Accept": "application/json"},
            )
            if r.status_code != 200:
                raise HTTPException(status_code=502, detail="github token exchange failed")
            gh = r.json()
            gh_token = gh.get("access_token")
            if not gh_token:
                raise HTTPException(status_code=401, detail=gh.get("error_description", "no token"))
            me = await client.get("https://api.github.com/user", headers={"Authorization": f"Bearer {gh_token}"})
            gh_user = me.json()
        email = gh_user.get("email") or f"{gh_user['id']}@users.noreply.github.com"
        gh_login = gh_user.get("login") or ""
        display = gh_user.get("name") or gh_login or email
        sf = get_session_factory()
        async with _maybe_session(sf) as s:
            user = await get_user_by_email(s, email)
            if not user:
                user = await create_user(s, email=email, hashed_password="", username=gh_login, display_name=display, roles=["user"])
        roles = _parse_roles(user.get("roles") or "user")
        token, _expires, _jti = create_access_token(user["id"], roles=roles)
        return LoginResponse(
            access_token=token,
            user_id=user["id"],
            email=user.get("email", ""),
            username=user.get("username"),
            display_name=user.get("display_name"),
            roles=roles,
        )

    # ── Query ─────────────────────────────────────────────────────────────

    @app.post("/query", response_model=QueryResponse)
    @limiter.limit("20/minute")
    async def query(request: Request, body: QueryRequest, user=Depends(current_user)):
        agent = _get_agent()
        allowed, refusal, pii = await get_guardrails().check_input(body.question)
        if pii:
            logger.info("PII entities in query: %s", [{k: v for k, v in e.items() if k != "text"} for e in pii])
        if not allowed:
            raise HTTPException(status_code=400, detail=refusal or "blocked by guardrails")
        result = await agent.query(user["id"], body)
        if result:
            gr = get_guardrails()
            loop = asyncio.get_running_loop()
            if result.answer:
                result.answer = await loop.run_in_executor(None, gr.anonymize, result.answer)
            for cit in result.citations:
                if cit.quote:
                    cit.quote = await loop.run_in_executor(None, gr.anonymize, cit.quote)
        return result

    @app.post("/query/stream")
    @limiter.limit("20/minute")
    async def query_stream(request: Request, body: QueryRequest, user=Depends(current_user)):
        agent = _get_agent()
        allowed, refusal, pii = await get_guardrails().check_input(body.question)
        if pii:
            logger.info("PII entities in query: %s", [{k: v for k, v in e.items() if k != "text"} for e in pii])
        if not allowed:
            raise HTTPException(status_code=400, detail=refusal or "blocked by guardrails")

        async def event_gen():
            try:
                async for event in agent.stream_query(user["id"], body):
                    # Anonymize PII in text chunks
                    text = event.get("text", "")
                    if text:
                        loop = asyncio.get_running_loop()
                        event["text"] = await loop.run_in_executor(None, get_guardrails().anonymize, text)
                    yield f"data: {json.dumps(event)}\n\n"
            except Exception as e:
                err = json.dumps({"type": "error", "detail": str(e)[:500]})
                yield f"data: {err}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_gen(), media_type="text/event-stream")

    @app.post("/analyze", response_model=DocumentAnalysis)
    @limiter.limit("10/minute")
    async def analyze(request: Request, body: AnalyzeRequest, user=Depends(current_user)):
        from ..services.llm import LLMError
        allowed, refusal, pii = await get_guardrails().check_input(body.question)
        if pii:
            logger.info("PII entities in analyze: %s", [{k: v for k, v in e.items() if k != "text"} for e in pii])
        if not allowed:
            raise HTTPException(status_code=400, detail=refusal or "blocked by guardrails")
        logger.info("ANALYZE request: user=%s question=%r doc_ids=%s max_cit=%s",
                     user["id"], body.question, body.document_ids, body.max_citations)
        agent = _get_agent()
        try:
            result = await agent.analyze_document(user["id"], body.question, body.document_ids, body.max_citations, model=body.model)
            # Anonymize PII in all text fields
            gr = get_guardrails()
            loop = asyncio.get_running_loop()
            result.summary = await loop.run_in_executor(None, gr.anonymize, result.summary)
            result.key_findings = [await loop.run_in_executor(None, gr.anonymize, f) for f in result.key_findings]
            result.methodology = await loop.run_in_executor(None, gr.anonymize, result.methodology)
            result.research_gaps = [await loop.run_in_executor(None, gr.anonymize, g) for g in result.research_gaps]
            result.contradictions = [await loop.run_in_executor(None, gr.anonymize, c) for c in result.contradictions]
            result.open_questions = [await loop.run_in_executor(None, gr.anonymize, q) for q in result.open_questions]
            result.limitations = await loop.run_in_executor(None, gr.anonymize, result.limitations)
            for cit in result.citations:
                if cit.quote:
                    cit.quote = await loop.run_in_executor(None, gr.anonymize, cit.quote)
            if result.per_document_summary:
                result.per_document_summary = {k: await loop.run_in_executor(None, gr.anonymize, v) for k, v in result.per_document_summary.items()}
            logger.info("ANALYZE response: summary=%.200s doc_ids=%s len(citations)=%d",
                         result.summary, result.documents_analyzed, len(result.citations))
            return result
        except LLMError as e:
            logger.error("ANALYZE LLMError: %s", e)
            raise HTTPException(status_code=502, detail=str(e))
        except Exception as e:
            logger.exception("ANALYZE unhandled error")
            raise HTTPException(status_code=500, detail=str(e))

    # ── Documents ─────────────────────────────────────────────────────────

    @app.post("/documents", response_model=UploadResponse, status_code=201)
    @limiter.limit("10/minute")
    async def upload_documents(request: Request, files: list[UploadFile] = File(...), user=Depends(current_user)):
        sf = get_session_factory()

        async def _process_one(f: UploadFile) -> UploadedDocument:
            # 1. Read content
            content = await f.read()
            if len(content) > MAX_FILE_SIZE:
                raise HTTPException(status_code=413, detail=f"{f.filename} too large")
            safe = sanitize_filename(f.filename) or "document"
            doc_id = uuid.uuid4().hex
            upload_id = uuid.uuid4().hex
            target = UPLOAD_DIR / f"{doc_id}_{safe}"

            # 2. Write to disk + compute SHA256 (PII runs in job worker)
            async def _write_and_hash() -> str:
                target.write_bytes(content)
                return hashlib.sha256(content).hexdigest()

            digest = await _write_and_hash()

            # 3. Dedup check
            is_duplicate = False
            if digest and sf is not None:
                async with sf() as s:
                    existing = await get_document_by_sha256(s, digest)
                    if existing and existing.get("uploaded_by") == user["id"]:
                        is_duplicate = True
                        doc_id = existing["id"]
                        safe = existing["filename"]

            # 4. DB insert + enqueue
            status = "duplicate" if is_duplicate else "processing"
            async with _maybe_session(sf) as s:
                await create_document(
                    s,
                    id=doc_id,
                    uploaded_by=user["id"],
                    filename=safe,
                    status=status,
                    sha256=digest,
                )
            if not is_duplicate:
                record = JobRecord(
                    id=upload_id,
                    user_id=user["id"],
                    document_id=doc_id,
                    filename=safe,
                    content_path=str(target),
                )
                await get_worker().enqueue(record)
            return UploadedDocument(upload_id=upload_id, document_id=doc_id, filename=safe, status=status)

        results = await asyncio.gather(*[_process_one(f) for f in files], return_exceptions=True)
        uploaded: list[UploadedDocument] = []
        for r in results:
            if isinstance(r, Exception):
                logger.error("Upload sub-task failed: %s", r)
                continue
            uploaded.append(r)
        return UploadResponse(uploaded_documents=uploaded)

    @app.get("/documents", response_model=list[DocumentResponse])
    async def list_docs(request: Request, user=Depends(current_user)):
        sf = get_session_factory()
        async with _maybe_session(sf) as s:
            docs = await list_documents(s, user_id=user["id"])
        base = str(request.base_url)
        return [_doc_response(d, base_url=base) for d in docs]

    @app.get("/documents/{doc_id}", response_model=DocumentResponse)
    async def get_doc(doc_id: str, request: Request, user=Depends(current_user)):
        sf = get_session_factory()
        async with _maybe_session(sf) as s:
            d = await get_document(s, doc_id)
        if not d or d.get("uploaded_by") != user["id"]:
            raise HTTPException(status_code=404, detail="not found")
        return _doc_response(d, base_url=str(request.base_url))

    @app.delete("/documents/{doc_id}", status_code=204)
    async def del_doc(doc_id: str, user=Depends(current_user)):
        sf = get_session_factory()
        async with _maybe_session(sf) as s:
            d = await get_document(s, doc_id)
            if not d or d.get("uploaded_by") != user["id"]:
                raise HTTPException(status_code=404, detail="not found")
            await delete_document(s, doc_id)
        try:
            get_vector_store().delete_document(user["id"], doc_id)
        except Exception as e:
            logger.warning("qdrant delete failed: %s", e)
        return Response(status_code=204)

    @app.get("/documents/{doc_id}/pdf")
    async def stream_document_pdf(doc_id: str):
        sf = get_session_factory()
        async with _maybe_session(sf) as s:
            d = await get_document(s, doc_id)
            if not d:
                docs = await list_documents(s)
                match = [x for x in docs if x.get("filename") == doc_id]
                d = match[0] if match else None
        if not d:
            raise HTTPException(status_code=404, detail="not found")
        cu = d.get("cloudinary_url")
        filename = d.get("filename", "document.pdf")
        print(f"PDF-DEBUG: doc_id={doc_id} cu={cu}")
        if not cu:
            raise HTTPException(status_code=404, detail="no cloudinary URL")
        try:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                resp = await client.get(cu)
                print(f"PDF-DEBUG: Cloudinary status={resp.status_code} type={resp.headers.get('content-type')}")
                if resp.status_code != 200:
                    raise HTTPException(status_code=502, detail="Cloudinary returned " + str(resp.status_code))
                return StreamingResponse(
                    resp.iter_bytes(),
                    media_type="application/pdf",
                    headers={
                        "Content-Disposition": f'inline; filename="{filename}"',
                        "Access-Control-Allow-Origin": "*",
                        "X-Content-Type-Options": "nosniff",
                    },
                )
        except HTTPException:
            raise
        except Exception as e:
            print(f"PDF-DEBUG: EXCEPTION={e}")
            raise HTTPException(status_code=502, detail="failed to fetch PDF")

    @app.post("/documents/backfill")
    async def backfill_documents(request: Request, user=Depends(current_user)):
        sf = get_session_factory()
        async with _maybe_session(sf) as s:
            docs = await list_documents(s, user_id=user["id"])
        updated = 0
        for doc in docs:
            safe = doc.get("filename", "document")
            doc_id = doc["id"]
            local = UPLOAD_DIR / f"{doc_id}_{safe}"
            if not local.exists():
                alt = sorted(UPLOAD_DIR.glob(f"*_{safe}"))
                if alt:
                    local = alt[0]
                else:
                    continue
            content = local.read_bytes()
            try:
                cli = get_cloudinary()
                if cli:
                    res = cli.upload(content, public_id=f"{user['id']}/{doc_id}")
                    cu = res.get("secure_url") or res.get("url")
                    if cu:
                        async with _maybe_session(sf) as s:
                            await update_document(s, doc_id, cloudinary_url=cu)
                        updated += 1
            except Exception as e:
                logger.warning("backfill failed for %s: %s", doc_id, e)
        return {"backfilled": updated, "total": len(docs)}

    @app.get("/uploads/{upload_id}", response_model=UploadStatusResponse)
    async def upload_status(upload_id: str, user=Depends(current_user)):
        record = await get_worker().queue.get(upload_id) if hasattr(get_worker(), "queue") else None
        if record is None:
            raise HTTPException(status_code=404, detail="upload not found")
        if record.user_id != user["id"]:
            raise HTTPException(status_code=404, detail="not found")
        return UploadStatusResponse(**record.to_status_dict())

    # ── Sessions ──────────────────────────────────────────────────────────

    @app.get("/sessions", response_model=SessionsListResponse)
    async def list_sessions_endpoint(request: Request, user=Depends(current_user)):
        sf = get_session_factory()
        async with _maybe_session(sf) as s:
            rows = await list_sessions(s, user_id=user["id"])
        return SessionsListResponse(
            sessions=[
                SessionResponse(
                    id=r["id"],
                    title=r.get("title"),
                    user_id=r["user_id"],
                    created_at=r.get("created_at"),
                    updated_at=r.get("updated_at"),
                )
                for r in rows
            ]
        )

    @app.post("/sessions", response_model=SessionResponse, status_code=201)
    async def create_session_endpoint(request: Request, body: CreateSessionRequest, user=Depends(current_user)):
        sid = body.id or uuid.uuid4().hex
        title = body.title or "New chat"
        sf = get_session_factory()
        async with _maybe_session(sf) as s:
            existing = await get_session(s, sid, user_id=user["id"])
            if existing:
                return SessionResponse(
                    id=existing["id"],
                    title=existing.get("title"),
                    user_id=existing["user_id"],
                    created_at=existing.get("created_at"),
                    updated_at=existing.get("updated_at"),
                )
            row = await create_session(s, sid, user["id"], title)
        return SessionResponse(
            id=row["id"],
            title=row.get("title"),
            user_id=row["user_id"],
            created_at=row.get("created_at"),
            updated_at=row.get("updated_at"),
        )

    @app.get("/sessions/{sid}", response_model=SessionDetailResponse)
    async def get_session_endpoint(sid: str, user=Depends(current_user)):
        sf = get_session_factory()
        async with _maybe_session(sf) as s:
            sess = await get_session(s, sid, user_id=user["id"])
            if sess is None:
                raise HTTPException(status_code=404, detail="not found")
            msgs = await list_messages(s, sid, user_id=user["id"])
        return SessionDetailResponse(
            id=sess["id"],
            title=sess.get("title"),
            user_id=sess["user_id"],
            created_at=sess.get("created_at"),
            updated_at=sess.get("updated_at"),
            messages=[MessageResponse(**m) for m in msgs],
        )

    @app.get("/sessions/{sid}/messages", response_model=MessagesListResponse)
    async def get_session_messages(sid: str, user=Depends(current_user)):
        sf = get_session_factory()
        async with _maybe_session(sf) as s:
            sess = await get_session(s, sid, user_id=user["id"])
            if sess is None:
                raise HTTPException(status_code=404, detail="not found")
            msgs = await list_messages(s, sid, user_id=user["id"])
        return MessagesListResponse(messages=[MessageResponse(**m) for m in msgs])

    @app.put("/sessions/{sid}", response_model=SessionResponse)
    async def update_session_endpoint(sid: str, body: UpdateSessionRequest, user=Depends(current_user)):
        sf = get_session_factory()
        async with _maybe_session(sf) as s:
            sess = await get_session(s, sid, user_id=user["id"])
            if sess is None:
                raise HTTPException(status_code=404, detail="not found")
            row = await update_session_title(s, sid, user_id=user["id"], title=body.title)
        if row is None:
            raise HTTPException(status_code=404, detail="not found")
        return SessionResponse(
            id=row["id"],
            title=row.get("title"),
            user_id=row["user_id"],
            created_at=row.get("created_at") or sess.get("created_at"),
            updated_at=row.get("updated_at") or datetime.utcnow().isoformat() + "Z",
        )

    @app.delete("/sessions/{sid}", status_code=204)
    async def del_session_endpoint(sid: str, user=Depends(current_user)):
        sf = get_session_factory()
        async with _maybe_session(sf) as s:
            sess = await get_session(s, sid, user_id=user["id"])
            if sess is None:
                raise HTTPException(status_code=404, detail="not found")
            await delete_session(s, sid, user_id=user["id"])
        return Response(status_code=204)

    @app.post("/sessions/{sid}/messages", response_model=MessageResponse, status_code=201)
    async def add_message_endpoint(sid: str, body: AddMessageRequest, user=Depends(current_user)):
        sf = get_session_factory()
        async with _maybe_session(sf) as s:
            m = await add_message(
                s, sid, user_id=user["id"],
                role=body.role,
                content=body.content,
                citations=json.dumps([c.model_dump() for c in body.citations]) if body.citations is not None else None,
                reasoning_path=json.dumps(body.reasoning_path) if body.reasoning_path else None,
                tokens_used=body.tokens_used,
                cost_usd=body.cost_usd,
                query_id=body.query_id,
                verification=body.verification,
            )
        if m is None:
            raise HTTPException(status_code=404, detail="session not found")
        def _safe_json(val):
            if val is None:
                return None
            try:
                return json.loads(val)
            except (json.JSONDecodeError, TypeError, ValueError):
                return None
        return MessageResponse(
            id=m["id"],
            session_id=m.get("session_id", sid),
            role=m.get("role", "user"),
            content=m.get("content", ""),
            citations=_safe_json(m.get("citations")),
            reasoning_path=_safe_json(m.get("reasoning_path")),
            tokens_used=m.get("tokens_used"),
            cost_usd=m.get("cost_usd"),
            query_id=m.get("query_id"),
            verification=m.get("verification"),
            created_at=m.get("created_at"),
        )

    # ── Feedback ──────────────────────────────────────────────────────────

    @app.post("/feedback", status_code=204)
    async def feedback(request: Request, body: FeedbackRequest, user=Depends(current_user)):
        sf = get_session_factory()
        async with _maybe_session(sf) as s:
            await add_feedback(
                s,
                user_id=user["id"],
                query_id=body.query_id,
                thumbs_up=body.thumbs_up,
                comment=body.comment,
            )
        return Response(status_code=204)

    # ── Admin ─────────────────────────────────────────────────────────────

    @app.get("/admin/dlq", response_model=DLQResponse)
    async def admin_dlq(request: Request, user=Depends(require_role("admin"))):
        sf = get_session_factory()
        async with _maybe_session(sf) as s:
            items = await list_dlq(s)
        return DLQResponse(
            dead_letter_queue=[
                DLQItem(
                    id=i.get("id", ""),
                    task=i.get("source", "unknown"),
                    payload=i.get("payload", "")[:1000],
                    error=i.get("error", ""),
                    failed_at=i.get("created_at"),
                )
                for i in items
            ]
        )

    @app.get("/llm/profiles")
    async def list_llm_profiles(request: Request, user=Depends(require_role("user"))):
        from ..services.llm import list_profiles
        return {"profiles": list_profiles()}

    @app.post("/cache/flush")
    async def cache_flush(request: Request, user=Depends(require_role("admin"))):
        await get_cache().flush_pattern("")
        return {"status": "cache_flushed"}

    # ── Wire shutdown manager ─────────────────────────────────────────────

    @app.on_event("startup")
    async def _on_startup():
        from ..shutdown import get_shutdown_manager
        try:
            loop = asyncio.get_running_loop()
            get_shutdown_manager().install_signal_handlers(loop)
        except Exception as e:
            logger.debug("signal handler install skipped: %s", e)

        from ..database.session import init_engine
        try:
            await init_engine()
            logger.info("DB engine initialized")
        except Exception as e:
            logger.warning("DB init failed: %s — running with in-memory fallback", e)

        from ..vectorstore.Qdrant import get_vector_store
        get_vector_store()
        logger.info("Vector store ready (model loaded)")

        from ..job_queue import get_worker
        set_upload_processor()
        await get_worker().start()
        logger.info("Upload processor registered and worker started")

    return app


# ── Helpers ──────────────────────────────────────────────────────────────────

_agent: Optional[DocumentAgent] = None


def _get_agent() -> DocumentAgent:
    global _agent
    if _agent is None:
        _agent = DocumentAgent()
    return _agent


def set_upload_processor() -> None:
    async def _process(record) -> None:
        import time
        _t0 = time.time()
        logger.info("UPLOAD: starting processing for %s (file=%s)", record.id, record.filename)
        try:
            await get_worker().queue.update(record.id, stage="extracting", progress=20)
            content = Path(record.content_path).read_bytes()
            _t1 = time.time()
            text: str = ""
            page_ranges: Optional[list] = None
            has_pii: bool = False
            from ..services.document_parser import parse_document

            # Run PII detection concurrently with document parsing
            async def _pii_scan() -> bool:
                from ..services.pii_detector import get_pii_detector
                pii_detector = get_pii_detector()
                if not pii_detector or not getattr(pii_detector, "enabled", True):
                    return False
                try:
                    sample = content[:100_000].decode("utf-8", errors="ignore")
                    result = bool(pii_detector.detect(sample))
                    logger.info("PII: scan result for %s = %s (sample=%d chars)", record.filename, result, len(sample))
                    return result
                except Exception as e:
                    logger.warning("PII: scan failed for %s: %s", record.filename, e)
                    return False

            name_lower = (record.filename or "").lower()
            if name_lower.endswith(".pdf"):
                from ..services.document_parser import parse_pdf_with_pages
                results = await asyncio.gather(
                    asyncio.to_thread(parse_document, record.filename, content),
                    asyncio.to_thread(parse_pdf_with_pages, content),
                    _pii_scan(),
                )
                md_text, (pdf_text, pr), has_pii = results
                # Use pdfplumber text for chunking — page mapping is guaranteed accurate
                text = pdf_text if pdf_text.strip() else md_text
                page_ranges = pr
                logger.info("UPLOAD: parsed PDF %s → %d chars (MarkItDown) + %d chars (pdfplumber, used) + %d pages in %.2fs",
                             record.filename, len(md_text), len(pdf_text), len(page_ranges or []), time.time() - _t1)
            else:
                text, has_pii = await asyncio.gather(
                    asyncio.to_thread(parse_document, record.filename, content),
                    _pii_scan(),
                )
                logger.info("UPLOAD: parsed %s → %d chars in %.2fs", record.filename, len(text), time.time() - _t1)
            # Anonymize text + page texts if PII was detected (applies to both PDF and non-PDF paths)
            if has_pii and text:
                from ..services.pii_detector import get_pii_detector
                pii = get_pii_detector()
                if pii:
                    loop = asyncio.get_running_loop()
                    original_len = len(text)
                    text = await loop.run_in_executor(None, pii.anonymize, text)
                    if len(text) != original_len:
                        logger.info("PII: anonymized %s (%d chars → %d chars)", record.filename, original_len, len(text))
                    else:
                        logger.info("PII: scan flagged but no entities anonymized for %s", record.filename)
            await get_worker().queue.update(record.id, stage="chunking", progress=40)
            await get_worker().queue.update(record.id, stage="embedding", progress=60)
            _t2 = time.time()
            n_chunks = await get_vector_store().add_document(
                user_id=record.user_id, document_id=record.document_id, filename=record.filename, text=text, page_ranges=page_ranges
            )
            logger.info("UPLOAD: indexed %d chunks in %.2fs (total %.2fs)", n_chunks, time.time() - _t2, time.time() - _t0)
            await get_worker().queue.update(record.id, stage="indexing", progress=90)
            sf = get_session_factory()
            if sf is not None:
                async with sf() as s:
                    await update_document(s, record.document_id, status="success", has_pii=has_pii)
            # Fire-and-forget Cloudinary upload
            cli = get_cloudinary()
            if cli:
                try:
                    public_id = f"{record.user_id}/{record.document_id}"
                    res = cli.upload(content, public_id=public_id)
                    if isinstance(res, dict):
                        cu = res.get("secure_url") or res.get("url")
                        cpid = res.get("public_id")
                        if sf is not None:
                            async with sf() as s2:
                                await update_document(s2, record.document_id, cloudinary_url=cu, cloudinary_public_id=cpid)
                        logger.info("CLOUDINARY: uploaded %s → %s", record.filename, cu)
                except Exception as e:
                    logger.warning("CLOUDINARY: upload failed for %s: %s", record.filename, e)
            get_metrics().uploads_total.labels(status="ok").inc()
            logger.info("UPLOAD: completed %s in %.2fs", record.id, time.time() - _t0)
        except Exception as e:
            logger.exception("UPLOAD: failed for %s after %.2fs: %s", record.id, time.time() - _t0, e)
            get_metrics().uploads_total.labels(status="error").inc()
            sf = get_session_factory()
            if sf is not None:
                async with sf() as s:
                    await update_document(s, record.document_id, status="failed")
                    await push_dlq(s, source="upload", error=str(e)[:1000], payload=f"file={record.filename}", filename=record.filename)

    get_worker().set_processor(_process)


class _NullCtx:
    async def __aenter__(self):
        return None
    async def __aexit__(self, *a):
        return False


def _maybe_session(sf):
    """Return an async context manager yielding either an AsyncSession or None."""
    if sf is None:
        return _NullCtx()
    return sf()


app = create_app()
