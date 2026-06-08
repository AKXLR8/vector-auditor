---
title: Vector Auditor
emoji: 📊
colorFrom: indigo
colorTo: gray
sdk: docker
pinned: false
---

# Vector Auditor

**Agentic document intelligence** — upload PDFs, ask questions, get cited answers with page-level citations.

A FastAPI backend with a lite RAG pipeline (no LangChain), Qdrant vector store, Postgres persistence, Redis caching, and circuit-breaker resilience.

## Demo

```
POST /query  {"question": "What are the key findings?", "mode": "white_box"}
→ 200  {"answer": "...", "citations": [{"page": 3, "quote": "..."}], ...}
```

## Features

- **RAG with citations** — multi-hop retrieval → LLM generation → verification → gap analysis
- **Two modes**: `white_box` (full reasoning path) and `black_box` (temperature=0, terse)
- **Parallel uploads** — 5 concurrent jobs with SHA256 dedup
- **Page-level citations** — retrieved from pdfplumber, mapped to Qdrant chunks
- **Graceful degradation** — circuit breakers on LLM, Qdrant, embedding; fallback to raw context when LLM is down
- **Guardrails + PII detection** — NeMo Guardrails (with regex fallback) + Presidio PII (opt-in)
- **Dead letter queue** — failed uploads captured for replay
- **Streaming SSE** — `citations` / `token` / `verification` / `gap_analysis` / `done` / `error`
- **Multi-user** — JWT auth, document isolation per user
- **Feedback loop** — thumbs up/down per query
- **Observability** — JSON structured logs, Prometheus `/metrics`, health `/health`, readiness `/readyz`

## Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌──────────────┐
│  FastAPI     │────▶│  DocumentAgent   │────▶│  LLM (Incept)│
│  36 routes   │     │  (retrieve→gen→  │     │  + circuit   │
│  + middleware│     │   verify→gaps)   │     │  breaker     │
└──────┬───────┘     └────────┬─────────┘     └──────────────┘
       │                      │
       ▼                      ▼
┌──────────┐     ┌──────────────────────┐
│ Postgres │     │  Qdrant + all-MiniLM │
│ + Redis  │     │  (user-isolated)     │
│ fallback │     │  + circuit breaker   │
└──────────┘     └──────────────────────┘
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| API | FastAPI + Uvicorn + Pydantic v2 |
| Auth | JWT (python-jose) + bcrypt |
| Database | PostgreSQL (SQLAlchemy async) / in-memory JSONL fallback |
| Vector Store | Qdrant (Cloud / local / in-memory) |
| Embeddings | SentenceTransformers all-MiniLM-L6-v2 |
| LLM | OpenAI-compatible (Inception Labs mercury-2) |
| Cache | Redis / in-process TTLCache |
| File Store | Cloudinary (PDF serving) |
| PDF Parse | MarkItDown (text) + pdfplumber (page numbers) |
| Resilience | Circuit breakers + exponential backoff retry |
| Observability | JSON logs + Prometheus |
| Rate Limiting | slowapi (200/min default) |

## Quick Start

```bash
# Clone
git clone https://github.com/AKXLR8/vector-auditor.git
cd vector-auditor

# Environment
cp .env.example .env
# Edit .env — set at minimum: LLM_API_KEY, JWT_SECRET_KEY

# Install
python -m pip install -r requirements.txt

# Run
uvicorn src.api.main:app --reload --port 8000
```

Open http://localhost:8000/docs for interactive API docs.

## Configuration

| Variable | Required | Default | Notes |
|----------|----------|---------|-------|
| `LLM_API_KEY` | Yes | — | Inception Labs or OpenAI-compatible key |
| `JWT_SECRET_KEY` | Yes | — | `python -c "import secrets; print(secrets.token_urlsafe(32))"` |
| `DATABASE_URL` | No | in-memory | PostgreSQL with asyncpg |
| `QDRANT_URL` | No | in-memory | Qdrant Cloud URL |
| `QDRANT_API_KEY` | No | — | Qdrant Cloud API key |
| `REDIS_URL` | No | in-memory | Redis for cache |
| `CLOUDINARY_*` | No | local only | PDF file serving |
| `LOG_FORMAT` | No | `json` | `text` for human-readable |
| `JOB_MAX_CONCURRENT` | No | `5` | Parallel upload jobs |

## API Overview (28 endpoints)

### Auth `/auth/*`
`POST register` · `POST login` · `POST login/mfa` · `POST logout` · `GET token/refresh` · `POST mfa/setup` · `POST mfa/verify` · `GET oauth/config` · `POST oauth/github`

### Query `/query`
`POST /query` — single answer with citations · `POST /query/stream` — SSE streaming · `POST /analyze` — multi-document analysis

### Documents `/documents`
`POST /documents` — upload (multi-file) · `GET /documents` — list · `GET /documents/{id}` — detail · `DELETE /documents/{id}` — remove · `GET /documents/{id}/pdf` — stream PDF

### Sessions `/sessions`
`GET /sessions` — list · `POST /sessions` — create · `GET /sessions/{id}` — detail with messages · `PUT /sessions/{id}` — rename · `DELETE /sessions/{id}` · `GET /sessions/{id}/messages` · `POST /sessions/{id}/messages`

### Operations
`POST /feedback` · `GET /admin/dlq` · `POST /cache/flush` · `GET /health` · `GET /readyz` · `GET /metrics`

## Deploy

### Hugging Face Spaces

```bash
git remote add hf https://huggingface.co/spaces/akshayyy1/vector-auditor
git push hf main
```

Set secrets in HF Space Settings → Variables. See `DEPLOY_HF_SPACES.md`.

### Docker

```bash
docker build -t vector-auditor .
docker run -p 7860:7860 -e LLM_API_KEY=... -e JWT_SECRET_KEY=... vector-auditor
```

## Production Checklist

- [x] Circuit breakers (LLM, Qdrant, embedding)
- [x] Retry with exponential backoff
- [x] Graceful degradation when LLM is down
- [x] Health check + readiness probe
- [x] Rate limiting (200/min default)
- [x] Security headers (X-Frame-Options, X-Content-Type-Options, Referrer-Policy)
- [x] Request ID tracking
- [x] Shutdown gate (drain in-flight requests)
- [x] JSON structured logging
- [x] Prometheus metrics
- [x] Dead letter queue for failed uploads
- [x] PII detection (opt-in)
- [x] Guardrails against prompt injection
- [x] JWT auth with role-based access
- [x] Document isolation per user
- [x] SHA256 dedup on upload
- [x] Parallel upload processing
- [x] Multi-stage Docker build (slim image)
- [ ] Golden dataset evals
- [ ] LangSmith cost monitoring

## Project Structure

```
src/
├── api/                # FastAPI routes (main.py, auth.py, middleware.py)
├── agents/             # DocumentAgent (lite RAG pipeline)
├── services/           # LLM, cache, parsers, guardrails, PII, circuit_breaker
├── database/           # SQLAlchemy models, repository, session
├── vectorstore/        # Qdrant wrapper with user isolation
├── models/             # Pydantic schemas
├── observability.py    # JSON logs + Prometheus
├── shutdown.py         # Graceful shutdown
├── job_queue.py        # Upload pipeline orchestrator
└── config.py           # pydantic-settings

scripts/                # download_model.py
alembic/                # DB migrations
models/                 # embedding_model.pkl (gitignored, built at deploy)
```

## License

MIT
