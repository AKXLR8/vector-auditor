

# Vector Auditor 
https://vector-auditor-frontend.vercel.app

**Agentic document intelligence** — upload PDFs, ask questions, get cited answers with page-level citations.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.11"/>
  <img src="https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/PostgreSQL-17-4169E1?style=for-the-badge&logo=postgresql&logoColor=white" alt="PostgreSQL"/>
  <img src="https://img.shields.io/badge/Qdrant-Cloud-7B1FA2?style=for-the-badge&logo=qdrant&logoColor=white" alt="Qdrant"/>
  <img src="https://img.shields.io/badge/Redis-7-FF4438?style=for-the-badge&logo=redis&logoColor=white" alt="Redis"/>
  <img src="https://img.shields.io/badge/Docker-24-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker"/>
  <img src="https://img.shields.io/badge/HuggingFace-Spaces-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="HuggingFace Spaces"/>
  <img src="https://img.shields.io/badge/Prometheus-0.53-E6522C?style=for-the-badge&logo=prometheus&logoColor=white" alt="Prometheus"/>
  <img src="https://img.shields.io/badge/SentenceTransformers-MiniLM-FF6F00?style=for-the-badge&logo=huggingface&logoColor=white" alt="SentenceTransformers"/>
  <img src="https://img.shields.io/badge/Presidio-PII-00ACC1?style=for-the-badge&logo=microsoft&logoColor=white" alt="Presidio PII"/>
  <img src="https://img.shields.io/badge/Cloudinary-3448C5?style=for-the-badge&logo=cloudinary&logoColor=white" alt="Cloudinary"/>
  <img src="https://img.shields.io/badge/JWT-Auth-000000?style=for-the-badge&logo=jsonwebtokens&logoColor=white" alt="JWT Auth"/>
  <img src="https://img.shields.io/badge/GitHub-OAuth-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub OAuth"/>
</p>

A FastAPI backend with a lite RAG pipeline (no LangChain), Qdrant vector store, Postgres persistence, Redis caching, and circuit-breaker resilience.

## Demo

```
POST /query  {"question": "What are the key findings?", "mode": "white_box"}
→ 200  {"answer": "...", "citations": [{"page": 3, "quote": "..."}], ...}
```

## Features

- **Semantic Search** — `all-MiniLM-L6-v2` embeddings (384-d) for meaning-based retrieval
- **Cross-Encoder Reranker** — `BAAI/bge-reranker-v2-m3` re-scores 10 candidates → top 5 before LLM
- **Cited Grounding** — inline `[N]` markers with page-level citations from pdfplumber, click to jump
- **Section-Aware Chunking** — 1000-char windows with 200-char overlap, split by markdown headers
- **Multi-Document Q&A** — select any subset of uploaded PDFs, scoped answers
- **AI-Powered Answers** — Inception Labs Mercury-2 via OpenAI-compatible endpoint, strict source grounding
- **PII Redaction** — Presidio analyzer scans all uploads (enabled by default)
- **Multi-Hop Retrieval** — iterative search across 3 hops for comprehensive coverage
- **Two modes** — `white_box` (full reasoning + verification + gap analysis) and `black_box` (temperature=0)
- **Parallel uploads** — 5 concurrent jobs with SHA256 dedup & Cloudinary storage
- **Graceful degradation** — circuit breakers on LLM, Qdrant, embedding; raw-context fallback when LLM is down
- **LRU query cache** — repeated queries skip embedding + vector search (600s TTL)
- **Guardrails** — NeMo Guardrails with regex fallback for prompt injection detection
- **Dead letter queue** — failed uploads captured for replay
- **Streaming SSE** — `citations` / `token` / `verification` / `gap_analysis` / `done` / `error`
- **Multi-user** — JWT auth with GitHub OAuth, document isolation per user
- **Feedback loop** — thumbs up/down per query
- **Observability** — JSON structured logs, Prometheus `/metrics`, health `/health`, readiness `/readyz`


## Architecture
---
```mermaid
---
config:
  layout: elk
  theme: neo-dark
---
graph TB
  subgraph Clients
    User["User Browser"]
    FE["Frontend (separate repo)"]
  end

  subgraph "HF Spaces (Docker, 4 workers)"
    API["FastAPI App<br/>src/api/main.py"]
    MW["Middleware<br/>Logging · CORS · Auth"]
    Auth["Auth Service<br/>JWT · GitHub OAuth"]
    Rate["Rate Limiter<br/>slowapi"]
  end

  subgraph "Document Processing Pipeline"
    Parser["Document Parser<br/>MarkItDown + pdfplumber"]
    PII["PII Detection<br/>presidio-analyzer"]
    Cloud["Cloudinary<br/>raw file storage"]
    Chunker["Text Chunker<br/>RecursiveCharacterTextSplitter<br/>1000 chars · 200 overlap"]
  end

  subgraph "Vector Store"
    Qdrant["Qdrant Cloud<br/>Vector DB"]
    Embedder["Embedding Model<br/>all-MiniLM-L6-v2 (384d)"]
    CB_Q["Circuit Breaker<br/>search: 5/30s · index: 3/60s"]
  end

  subgraph "LLM / RAG"
    Agent["Document Agent<br/>src/agents/document_agent.py"]
    Reranker["Cross-Encoder Reranker<br/>BAAI/bge-reranker-v2-m3"]
    LLM["LLM Service<br/>Mercury-2 via Inception Labs"]
    CB_L["Circuit Breaker<br/>5 failures / 30s recovery"]
    Retry["Retry w/ Backoff<br/>0.5s → 1s → 2s"]
    Guard["Guardrails<br/>NeMo Guardrails"]
    Degrade["Graceful Degradation<br/>context-only fallback"]
  end

  subgraph "Infrastructure"
    PG[("PostgreSQL<br/>(Qdrant Cloud or in-memory fallback)")]
    Redis[("Redis<br/>(session cache)")]
    JobQ["Job Queue<br/>max_concurrent=5"]
    Metrics["Prometheus Metrics"]
    Cache["LRU Query Cache<br/>TTLCache 600s"]
    Shutdown["Graceful Shutdown"]
    TokenCounter["Token Counter"]
  end

  %% Connections
  User --> FE --> API
  API --> MW --> Auth
  MW --> Rate

  API --> JobQ --> Parser --> PII --> Cloud
  Parser --> Chunker --> Qdrant
  Qdrant --> Embedder
  Qdrant --> CB_Q

  API --> Agent
  Agent --> Qdrant
  Agent --> Reranker
  Reranker --> LLM
  Agent --> LLM
  LLM --> CB_L --> Degrade
  LLM --> Retry
  LLM --> Guard

  API --> PG
  API --> Redis
  API --> Metrics
  API --> Shutdown
  API --> TokenCounter

  %% Color Styling
  classDef client fill:#0f172a,stroke:#38bdf8,color:#f0f9ff
  classDef api fill:#0a2647,stroke:#60a5fa,color:#e0f2fe
  classDef process fill:#111827,stroke:#2dd4bf,color:#f0fdfa
  classDef vector fill:#22092C,stroke:#f59e0b,color:#fff7ed
  classDef llm fill:#1e1b4b,stroke:#a78bfa,color:#f5f3ff
  classDef infra fill:#1c1917,stroke:#d4d4d8,color:#f5f5f4

  class User,FE client
  class API,MW,Auth,Rate api
  class Parser,PII,Cloud,Chunker process
  class Qdrant,Embedder,CB_Q vector
  class Agent,Reranker,LLM,CB_L,Retry,Guard,Degrade llm
  class PG,Redis,JobQ,Metrics,Cache,Shutdown,TokenCounter infra
```


## Tech Stack

| Layer | Technology |
|-------|-----------|
| API | FastAPI + Uvicorn + Pydantic v2 |
| Auth | JWT (python-jose) + bcrypt |
| Database | PostgreSQL (SQLAlchemy async) / in-memory JSONL fallback |
| Vector Store | Qdrant (Cloud / local / in-memory, collection `documents`) |
| Reranker | Cross-Encoder BAAI/bge-reranker-v2-m3 |
| Embeddings | SentenceTransformers all-MiniLM-L6-v2 (384-d) |
| LLM | OpenAI-compatible (Inception Labs mercury-2) |
| Cache | Redis / in-process TTLCache |
| File Store | Cloudinary (PDF serving) |
| PDF Parse | MarkItDown (text) + pdfplumber (page numbers) |
| Resilience | Circuit breakers + exponential backoff retry + LRU query cache |
| Observability | JSON logs + Prometheus |
| Rate Limiting | slowapi (200/min default) |
| Workers | 4 uvicorn workers |

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
- [x] PII detection (enabled by default)
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
