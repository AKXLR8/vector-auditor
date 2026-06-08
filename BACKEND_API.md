# Backend API Contract — Frontend Reference

Base URL: `http://localhost:8000`

## Authentication

All endpoints except `/auth/*` and `/health`/`/ready` require:

```
Authorization: Bearer <access_token>
```

### Auth Endpoints

| Method | Path | Request | Response |
|--------|------|---------|----------|
| POST | `/auth/register` | `{ email, password, first_name?, last_name? }` | `UserResponse` |
| POST | `/auth/login` | `{ email, password }` | `{ access_token, user_id, roles }` |
| POST | `/auth/logout` | — | 204 |
| GET | `/auth/token/refresh` | — | `{ access_token }` |
| POST | `/auth/mfa/setup` | — | `{ secret, uri, qr_code_url }` |
| POST | `/auth/mfa/verify` | `{ code }` | `UserResponse` |
| GET | `/auth/oauth/config` | — | `{ github_client_id }` |
| POST | `/auth/oauth/github` | `{ code }` | `LoginResponse` |

## Documents

### Upload — `POST /documents`
```
Content-Type: multipart/form-data
Body: files[] = (binary PDF files)
```
Returns `{ uploaded_documents: [{ upload_id, document_id, filename, status }] }`

Documents process asynchronously. Poll `GET /uploads/{upload_id}` for status.

### List Documents — `GET /documents`
```
Response: [
  {
    id: string,
    filename: string,
    status: "processing" | "ready" | "failed",
    has_pii: bool,
    cloudinary_url: string | null,   // PDF viewing URL
    uploaded_by: string,
    created_at: string | null
  }
]
```

### Get Document — `GET /documents/{doc_id}`
Returns single `DocumentResponse` (same shape as list item).

### Delete Document — `DELETE /documents/{doc_id}`
204 No Content.

### Backfill Cloudinary — `POST /documents/backfill`
Re-uploads old docs to Cloudinary. Returns `{ backfilled: number, total: number }`.

### Upload Status — `GET /uploads/{upload_id}`
```
{
  id, filename, stage, progress, error?, document_id?, user_id, created_at?, updated_at?
}
```
Stages: `uploading → extracting → chunking → embedding → indexing → completed`

## Query

### Non-streaming — `POST /query`
```json
{
  "question": "what is attention?",
  "document_ids": ["uuid1", "uuid2"],   // optional, null = all docs
  "mode": "white_box",                   // "white_box" | "black_box"
  "max_citations": 20,                   // optional, 1-100
  "conversation_history": [              // optional, last 6 turns max
    { "role": "user", "content": "..." },
    { "role": "assistant", "content": "..." }
  ]
}
```
**Response:**
```json
{
  "answer": "The Transformer uses multi-head attention... [1][2]",
  "citations": [
    {
      "quote": "excerpt from the document",
      "source": "attention_paper.pdf",       // filename
      "location": "page 3 / chunk 5",
      "page": 3,
      "document_id": "uuid"
    }
  ],
  "reasoning_path": ["Retrieved 6 citations across 2 hop(s)", "VERIFIED"],
  "tokens_used": 450,
  "cost_usd": 0.0009,
  "query_id": "hex",
  "verification": "VERIFIED"               // white_box only, null for black_box
}
```

### Streaming — `POST /query/stream`
Same request body. Server-Sent Events (SSE):

```
data: {"type": "citations", "citations": [...], "query_id": "...", "reasoning_path": [...]}
data: {"type": "token", "content": "The"}
data: {"type": "token", "content": " Transformer"}
data: {"type": "token", "content": " uses..."}
data: {"type": "verification", "content": "VERIFIED"}       // white_box only
data: {"type": "gap_analysis", "content": "missing X"}       // white_box only, one per gap
data: {"type": "done", "tokens_used": 450, "cost_usd": 0.0009, "mode": "white_box", "query_id": "..."}
data: [DONE]
```

**Citation click:** Use `citation.document_id` → fetch `GET /documents/{document_id}` → read `cloudinary_url` → open in PDF viewer at that URL.

## Analyze

### `POST /analyze`
```json
{
  "question": "Compare their methodologies",    // optional
  "document_ids": ["uuid1", "uuid2"],           // optional, null = error
  "max_citations": 20
}
```

**Response:**
```json
{
  "summary": "Both papers use Transformer-based architectures...",
  "key_findings": ["Finding 1", "Finding 2", "..."],
  "methodology": "Description of methods...",
  "research_gaps": ["Gap 1", "Gap 2"],
  "contradictions": ["Contradiction 1"],
  "open_questions": ["Question 1"],
  "limitations": "Only 3 documents analyzed",
  "confidence": "high",
  "citations": [/* same Citation shape as query */],
  "documents_analyzed": ["doc-uuid-1", "doc-uuid-2"],
  "cross_document_comparison": {
    "common_themes": ["both use attention"],
    "differences": ["doc1 uses BERT, doc2 uses GPT"],
    "complementary_insights": ["doc1's findings on X support doc2's Y"]
  },                                      // null when single doc
  "per_document_summary": {
    "paper1.pdf": "Introduces Transformer...",
    "paper2.pdf": "Applies BERT to..."
  }                                       // null when single doc
}
```

## PDF Viewer Flow

1. Query returns citations with `document_id`
2. Frontend user clicks citation `[N]`
3. Frontend calls `GET /documents/{document_id}` → gets `cloudinary_url`
4. Opens Cloudinary URL in PDF viewer (iframe or new tab)

**Cloudinary URL format:** `https://res.cloudinary.com/da1nocxo0/image/upload/v.../{user_id}/{doc_id}.pdf`

The `cloudinary_url` is always set for new uploads. For old docs, run `POST /documents/backfill` once.

## Sessions (Chat History)

| Method | Path | Notes |
|--------|------|-------|
| GET | `/sessions` | List all sessions |
| POST | `/sessions` | Create new session |
| GET | `/sessions/{sid}` | Session with messages |
| PUT | `/sessions/{sid}` | Update title |
| DELETE | `/sessions/{sid}` | Delete session |
| GET | `/sessions/{sid}/messages` | List messages |
| POST | `/sessions/{sid}/messages` | Add message |
| POST | `/sessions/{sid}/feedback` | Thumbs up/down |

## Query Modes

### White Box (default)
- Temperature: default (some variation)
- Max tokens: 3048
- Streams: `verification` + `gap_analysis` events
- Multi-hop retrieval (up to 3 rounds)
- Reasoning path included
- Structured answer with sections

### Black Box
- Temperature: 0 (deterministic)
- Max tokens: default (2048)
- No verification, no gap analysis, no reasoning
- Single-pass retrieval only
- Minimal answer — just the cited response, no commentary

## Greeting Detection
All three endpoints (`/query`, `/query/stream`, `/analyze`) detect greetings and return a friendly message without calling the LLM or searching documents:
```
"Hi there! What would you like to know about your documents?"   // /query, /query/stream
"Hi there! Please select documents to analyze, or ask me a question about your documents."  // /analyze
```

## Common Error Codes

| Code | Meaning |
|------|---------|
| 401 | Missing/invalid token |
| 400 | Guardrails blocked the query |
| 413 | File too large (max 50MB) |
| 502 | LLM provider unreachable |
| 429 | Rate limited (20/min for query, 10/min for analyze) |
