# RAG Auditor

Production-grade document Q&A system. FastAPI + lite RAG agent (no LangChain runtime) + Qdrant + Postgres + Redis.

Designed to match the `vector-auditor-frontend` TypeScript spec at `BACKEND_API.md` (28 endpoints, 5 SSE event types, JWT with `roles[]`).

## Quick start (Windows / PowerShell)

```powershell
cd C:\Users\ajaym\OneDrive\Desktop\vector

# 1. Copy and edit env
Copy-Item .env.example .env
# Fill in: INCEPTION_API_KEY, JWT_SECRET_KEY, DATABASE_URL
# Optional: QDRANT_URL, QDRANT_API_KEY, REDIS_URL, CLOUDINARY_*, GITHUB_*, PII_ENABLED

# 2. Install + migrate
python -m pip install -r requirements.txt
alembic upgrade head

# 3. Run
python run.py
# -> http://localhost:8000  (binds 0.0.0.0, auto-reloads src/, alembic/, scripts/)
# -> API docs:             http://localhost:8000/docs
# -> Health:               http://localhost:8000/health
```

### Dev flags

| Flag | Effect |
|------|--------|
| `python run.py` | port 8000, reload on, watch `src/` `alembic/` `scripts/` |
| `python run.py --port 7860` | HF Spaces default port |
| `python run.py --no-reload` | disable auto-reload (use for production-like) |
| `python run.py --host 127.0.0.1` | bind localhost only |

### Production

```powershell
alembic upgrade head
gunicorn src.api.main:app --config gunicorn_conf.py --bind 0.0.0.0:%PORT%
```

`gunicorn_conf.py` reads `$PORT` (defaults to 8000), single uvicorn worker + multi-thread, graceful timeout 30s.

## Endpoints (28 spec + 3 ops)

### Auth (9)
- `POST /auth/register` · `POST /auth/login` · `POST /auth/login/mfa` · `POST /auth/logout`
- `GET  /auth/token/refresh` · `POST /auth/mfa/setup` · `POST /auth/mfa/verify`
- `GET  /auth/oauth/config` · `POST /auth/oauth/github`

### Query (3)
- `POST /query` · `POST /query/stream` (SSE: `citations` / `token` / `verification` / `gap_analysis` / `done` / `error`) · `POST /analyze`

### Documents (5)
- `POST   /documents` · `GET /documents` · `GET /documents/{id}` · `DELETE /documents/{id}` (204)
- `GET    /uploads/{id}` (poll stage+progress)

### Sessions (7)
- `GET    /sessions` (wrapped in `{sessions:[]}`) · `POST /sessions` (accepts client-supplied `id`)
- `GET    /sessions/{id}` · `PUT /sessions/{id}` · `DELETE /sessions/{id}` (204)
- `GET    /sessions/{id}/messages` (wrapped in `{messages:[]}`)
- `POST   /sessions/{id}/messages`

### Feedback / admin / ops (4)
- `POST /feedback` (204, body `{query_id, thumbs_up, comment?}`)
- `GET  /admin/dlq` (wrapped in `{dead_letter_queue:[]}`, admin only)
- `POST /cache/flush` (admin only)
- `GET  /health` · `GET /readyz` · `GET /metrics`

## Response shapes (frontend contract)

```jsonc
// Citation
{"quote": "...", "source": "paper.pdf", "location": "Section 3.2", "page": 7}

// QueryResponse
{"answer": "...", "citations": [...], "reasoning_path": ["Retrieved 10..."],
 "tokens_used": 842, "cost_usd": 0.0021, "query_id": "abc123...",
 "timestamp": "2026-06-07T00:00:00Z", "verification": "VERIFIED", "mode": "white_box"}

// HealthResponse
{"status": "ok|degraded|down", "version": "1.0.0", "timestamp": "2026-06-07T00:00:00Z",
 "checks": {"database": "ok", "vector_store": "ok", "cache": "ok", "object_store": "ok", "llm_provider": "ok"}}

// UserResponse
{"id": "u_1", "email": "alice@example.com", "display_name": "Alice Smith",
 "roles": ["user"], "mfa_enabled": false, "created_at": "2026-06-07T00:00:00Z"}

// DocumentResponse
{"id": "d_1", "document_id": "d_1", "filename": "x.pdf", "status": "ready",
 "has_pii": false, "sha256": "9f86d0...", "cloudinary_url": "https://...",
 "uploaded_by": "u_1", "created_at": "2026-06-07T00:00:00Z"}

// UploadStatusResponse
{"id": "up_1", "filename": "x.pdf", "stage": "embedding", "progress": 60,
 "error": null, "document_id": "d_1", "user_id": "u_1",
 "created_at": "2026-06-07T00:00:00Z", "updated_at": "2026-06-07T00:00:05Z"}

// SessionsListResponse
{"sessions": [{"id": "s_1", "title": "My chat", "user_id": "u_1", "created_at": "...", "updated_at": "..."}]}

// DLQResponse
{"dead_letter_queue": [{"id": "dlq_1", "task": "upload", "payload": "...", "error": "...", "failed_at": "..."}]}
```

JWT payload: `{sub, exp, iat, jti, roles: ["user"]}`. Default new-user role: `user`.

## Configuration

Copy `.env.example` → `.env` and fill in:

| Variable | Required | Notes |
|----------|----------|-------|
| `INCEPTION_API_KEY` | yes | LLM provider |
| `JWT_SECRET_KEY` | yes (prod) | hard-fails in `ENVIRONMENT=production`; dev gets a fallback |
| `DATABASE_URL` | no | falls back to in-memory JSONL stores (`.data/*.jsonl`) |
| `QDRANT_URL` + `QDRANT_API_KEY` | no | falls back to in-memory vector store |
| `REDIS_URL` | no | falls back to in-process TTLCache |
| `CLOUDINARY_*` | no | falls back to local `uploads/` |
| `GITHUB_CLIENT_ID` + `GITHUB_CLIENT_SECRET` | no | enables `/auth/oauth/github` |
| `PII_ENABLED` | no | enables Presidio PII detection at upload time |
| `ALLOWED_ORIGINS` | no | CSV, defaults to `localhost:3000,3001,5173` |

## Architecture

```
src/
├── api/              # FastAPI routes, middleware, auth
│   ├── main.py       # 28 endpoints + 3 ops, all 5 SSE event types
│   ├── auth.py       # JWT with roles[], refresh, MFA stubs, require_role
│   └── middleware.py # metrics, request context, security headers, shutdown gate
├── agents/
│   └── document_agent.py   # query() + stream_query() + analyze_document()
├── services/         # llm, cache, document_parser, guardrails, pii, cloudinary, token_counter, circuit_breaker, async_worker, secrets
├── database/         # SQLAlchemy models, session, repository (in-memory fallback)
├── vectorstore/      # Qdrant (user_id-isolated)
├── models/           # Pydantic schemas (matches BACKEND_API.md)
├── bootstrap.py      # Startup: load models, pre-warm, start worker
├── observability.py  # JSON logs + Prometheus
├── shutdown.py       # SIGTERM, in-flight tracker
├── job_queue.py      # Stage machine: uploading→extracting→chunking→embedding→indexing→completed
└── config.py         # pydantic-settings

alembic/versions/     # 001_initial → 002_upload_jobs → 003_user_enhancements
```

### Two retrieval modes

- **`white_box`** — full reasoning path: multi-hop retrieve → generate → verify → gap analysis. Streams `verification` and `gap_analysis` events. `reasoning_path` populated in response.
- **`black_box`** — `temperature=0`, terse cited answer. Omits `reasoning_path`, skips `verification`/`gap_analysis` events.

## Deploy

| Target | How |
|--------|-----|
| **Hugging Face Spaces** | `DEPLOY_HF_SPACES.md` (5 steps) — port 7860, Dockerfile, entrypoint runs `alembic upgrade head` then `gunicorn` |
| **Railway** | `railway.toml` (NIXPACKS) — `releaseCommand: alembic upgrade head`, `startCommand: gunicorn ...` |
| **Render** | `render.yaml` ready |
| **Heroku** | `Procfile` ready (`web: gunicorn ...`, `release: alembic upgrade head`) |

## Production checklist (from PRD)

- [x] FastAPI + Pydantic validation
- [x] LangGraph removed — lite agent (white_box/black_box)
- [x] In-memory job queue with crash recovery
- [x] DLQ with Postgres + JSONL fallback
- [x] Circuit breakers & retries (`tenacity`)
- [x] PII detection (Presidio, gated by `PII_ENABLED`)
- [x] NeMo guardrails (with lightweight fallback)
- [x] JSON logs + Prometheus metrics (`/metrics`)
- [ ] Golden dataset evals (Ragas) — TODO
- [ ] LangSmith cost monitoring — TODO
- [ ] AWS Secrets Manager — optional (env vars work)

## License

MIT
