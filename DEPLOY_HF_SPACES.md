# Deploying to Hugging Face Spaces

Free Docker container with 2 vCPU, 16 GB RAM, no sleep.

## Prerequisites

- A Hugging Face account
- A PostgreSQL database (optional — app falls back to in-memory JSONL)
- A Qdrant Cloud instance (optional — app falls back to in-memory Qdrant)
- An LLM API key (e.g., Inception Labs `mercury-2`)
- A JWT secret key

## Steps

### 1. Create Space

1. Go to https://huggingface.co/new-space
2. **Space name**: e.g., `rag-auditor`
3. **License**: MIT
4. **SDK**: **Docker** (not Gradio / Streamlit)
5. **Hardware**: **CPU basic — free** (or CPU upgrade if you need more)
6. Click **Create Space**

### 2. Push code

```bash
# Add your HF Space as a remote
git remote add hf https://huggingface.co/spaces/<username>/<space-name>

# Push
git push https://<username>:<hf-token>@huggingface.co/spaces/<username>/<space-name> main
```

Replace `<username>`, `<space-name>`, `<hf-token>` with your values.  
Generate a token at https://huggingface.co/settings/tokens

### 3. Set environment secrets

Go to **Space Settings → Variables and secrets** and add:

| Variable | Required | Notes |
|----------|----------|-------|
| `INCEPTION_API_KEY` | **yes** | LLM provider key |
| `JWT_SECRET_KEY` | **yes** | Generate: `python -c "import secrets; print(secrets.token_urlsafe(32))"` |
| `ENVIRONMENT` | no | Set to `production` for prod hardening |
| `ALLOWED_ORIGINS` | no | Comma-separated frontend URLs (default: localhost:3000,3001,5173) |

**External services (optional, app falls back to in-memory if unset):**

| Variable | Purpose | Free tier |
|----------|---------|-----------|
| `DATABASE_URL` | PostgreSQL (Neon, Supabase, etc.) | Neon free: 0.5 GB |
| `QDRANT_URL` + `QDRANT_API_KEY` | Persistent vector storage | Qdrant Cloud free: 1 GB |
| `REDIS_URL` | Faster caching | Upstash Redis free: 10 MB |
| `CLOUDINARY_CLOUD_NAME` | PDF file serving | Cloudinary free: 25 GB |
| `CLOUDINARY_API_KEY` | — | — |
| `CLOUDINARY_API_SECRET` | — | — |

**Important**: HF Spaces have ephemeral storage. Data written to `/app/uploads/` and `/app/.data/` will be lost on restart. Always use cloud services for persistence.

### 4. Wait for build

First build takes ~5-10 minutes (model download, dependency installation).  
Subsequent builds are faster (Docker layer caching).

### 5. Verify

```bash
curl https://<username>-<space-name>.hf.space/readyz
# → {"ready":true}

curl https://<username>-<space-name>.hf.space/health
# → {"status":"ok","version":"1.0.0","checks":{...}}
```

## Updating

```bash
git push hf main
```

HF detects the push and rebuilds automatically.

## Multi-worker config

The `Dockerfile` starts uvicorn with 2 workers. For the free tier this is fine.  
For CPU upgrade spaces, increase `--workers` to match vCPUs.

## Troubleshooting

- **Build fails on sentence-transformers**: The `pip install` step includes PyTorch (~900 MB). This is normal and happens once.
- **App crashes at startup**: Check HF Space logs (Settings → Logs). Common issues: missing `INCEPTION_API_KEY`, bad `DATABASE_URL`.
- **Memory limit hit**: The embedding model uses ~500 MB RAM. Free tier has 16 GB, so this should not be an issue.
- **Slow first query**: The model is cached in the Docker image. First query warms the connection pool.
