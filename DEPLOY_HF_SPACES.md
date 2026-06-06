# Deploying to Hugging Face Spaces

Free Docker container with 2 vCPU, 16 GB RAM, no sleep.

## Steps

1. **Create Space** at https://huggingface.co/new-space
   - SDK: **Docker**
   - Hardware: **CPU basic — free**
2. **Push code**:
   ```bash
   git remote add hf https://huggingface.co/spaces/<username>/<space-name>
   git push https://<username>:<token>@huggingface.co/spaces/<username>/<space-name> main
   ```
3. **Set env vars** in Space → Settings → Variables and secrets:
   - `INCEPTION_API_KEY`
   - `JWT_SECRET_KEY` — generate with `python -c "import secrets; print(secrets.token_urlsafe(32))"`
   - `QDRANT_URL`, `QDRANT_API_KEY`
   - `DATABASE_URL`
   - `REDIS_URL`
   - `ENVIRONMENT=production`
   - `ALLOWED_ORIGINS` — your frontend URL
4. **Wait** for build (~5-10 min first time)
5. **Verify**: `curl https://<username>-<space-name>.hf.space/readyz`

## What doesn't persist

HF Spaces have ephemeral storage. `/app/uploads/` and `/app/.data/` are wiped on restart. Vector embeddings (Qdrant Cloud) and DB rows (Postgres Cloud) survive.

For persistent file storage, use Fly.io + volume (~$2/mo).
