# RAG Auditor — Production-Grade Document Intelligence

## Overview
A production-grade RAG system for auditing documents. Uses **vector retrieval** (Qdrant + sentence embeddings) with an **agentic reasoning** layer (LangGraph) to navigate documents, retrieve relevant chunks, and cite sources precisely.

## Architecture
- **Parsing**: LlamaParse (layout-aware) + Unstructured.io fallback
- **Reasoning**: LangGraph agentic layer
- **Database**: Document tree in PostgreSQL (or MongoDB)
- **Evaluation**: Ragas + golden dataset
- **Backend**: FastAPI + Pydantic validation
- **Monitoring**: LangSmith (cost per query, token usage)
- **Security**: PII detection guardrails, AWS Secrets Manager

## Source Types
- PDF, Markdown, DOCX, TXT
- Sources: Local filesystem, S3, Google Drive

## Query Requirements
- Every answer must include direct quote + page/section link
- Target latency: < 5 seconds for reasoning-based search

## Production Checklist
| Component | Status |
|-----------|--------|
| Parsing (layout-aware) | TODO |
| Dead Letter Queue (DLQ) | TODO |
| Agentic reasoning (LangGraph) | TODO |
| Guardrails (NeMo) | TODO |
| Golden dataset evals | TODO |
| PostgreSQL document DB | TODO |
| Circuit breakers & retries | TODO |
| PII detection | TODO |
| Feedback UI (thumbs up/down) | TODO |
| LangSmith monitoring | TODO |

## Key Files
- `PRD.md` — This document
- `src/` — Core application code
- `tests/` — Golden dataset and eval tests
- `docker/` — Containerization