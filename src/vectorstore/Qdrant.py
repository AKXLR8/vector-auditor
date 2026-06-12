"""Qdrant vector store with user-isolated collections + circuit breaker + cross-encoder reranker."""
import asyncio
import hashlib
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Optional

from cachetools import TTLCache

from ..services.cache import CACHE_TTL, get_cache
from ..services.circuit_breaker import CircuitBreaker, retry_with_backoff

logger = logging.getLogger("rga_auditor.qdrant")

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"

DEFAULT_MAX_CITATIONS_PER_DOC = int(os.getenv("MAX_CITATIONS_PER_DOC", "6"))
DEFAULT_MAX_CITATIONS_TOTAL = int(os.getenv("MAX_CITATIONS_TOTAL", "20"))
DEFAULT_RETRIEVE_K = int(os.getenv("RETRIEVE_K_PER_QUERY", "10"))
DEFAULT_RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "5"))

COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "documents")


class VectorStore:
    def __init__(
        self,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        from qdrant_client import QdrantClient
        from sentence_transformers import SentenceTransformer
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        self.collection_name = collection_name
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.embedding_dim = 384
        self._search_cb = CircuitBreaker(name="qdrant_search", failure_threshold=5, recovery_timeout_s=30.0)
        self._index_cb = CircuitBreaker(name="qdrant_index", failure_threshold=3, recovery_timeout_s=60.0)
        self._reranker = None
        self._query_embed_cache = TTLCache(maxsize=128, ttl=300)

        url = qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333")
        key = qdrant_api_key or os.getenv("QDRANT_API_KEY") or None
        try:
            self.client = QdrantClient(url=url, api_key=key, timeout=30)
            self._create_collection()
            logger.info("Qdrant connected to %s", url)
        except Exception as e:
            logger.warning("Qdrant unavailable at %s (%s) — falling back to :memory:", url, e)
            self.client = QdrantClient(":memory:")
            self._create_collection()

        self._load_model()
        self._ensure_reranker()

    def _load_model(self) -> None:
        import time
        t0 = time.monotonic()
        from sentence_transformers import SentenceTransformer
        pkl = MODELS_DIR / "embedding_model.pkl"
        if pkl.exists():
            import joblib
            self._model = joblib.load(str(pkl))
            logger.info("VectorStore: loaded embedding model from %s in %.2fs", pkl, time.monotonic() - t0)
        else:
            logger.info("VectorStore: pickle not found at %s — downloading all-MiniLM-L6-v2", pkl)
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("VectorStore: model downloaded in %.2fs", time.monotonic() - t0)

    def _ensure_reranker(self):
        if self._reranker is not None:
            return
        import time
        t0 = time.monotonic()
        from sentence_transformers import CrossEncoder
        pkl = MODELS_DIR / "reranker.pkl"
        if pkl.exists():
            import joblib
            self._reranker = joblib.load(str(pkl))
            logger.info("VectorStore: loaded reranker from %s in %.2fs", pkl, time.monotonic() - t0)
        else:
            logger.info("VectorStore: pickle not found at %s — downloading BAAI/bge-reranker-v2-m3", pkl)
            self._reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")
            logger.info("VectorStore: reranker downloaded in %.2fs", time.monotonic() - t0)

    def _create_collection(self) -> None:
        from qdrant_client.http import models
        try:
            col = self.client.get_collection(self.collection_name)
            if col.config.params.vectors.size != self.embedding_dim:
                logger.warning("Collection %s has dim %d, need %d — recreating",
                               self.collection_name, col.config.params.vectors.size, self.embedding_dim)
                self.client.delete_collection(self.collection_name)
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(size=self.embedding_dim, distance=models.Distance.COSINE),
                )
        except Exception:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=self.embedding_dim, distance=models.Distance.COSINE),
            )
            self.client.create_payload_index(
                collection_name=self.collection_name, field_name="user_id", field_schema=models.PayloadSchemaType.KEYWORD
            )
            self.client.create_payload_index(
                collection_name=self.collection_name, field_name="document_id", field_schema=models.PayloadSchemaType.KEYWORD
            )
            logger.info("Created collection %s", self.collection_name)

    @staticmethod
    def _map_chunks_to_pages(chunks: list[str], text: str, page_ranges: list[dict]) -> list[int]:
        """Map each chunk to a page number using character-offset ranges.
        Uses incremental search to avoid O(n*m) scanning from start each time."""
        pages: list[int] = []
        search_start = 0
        for chunk in chunks:
            pos = text.find(chunk, search_start)
            if pos < 0:
                pos = text.find(chunk)
            pg = 0
            if pos >= 0:
                for pr in page_ranges:
                    if pr["start"] <= pos < pr["end"]:
                        pg = pr["page"]
                        break
                search_start = pos + 1
            pages.append(pg)
        return pages

    async def add_document(self, user_id: str, document_id: str, filename: str, text: str, page_ranges: Optional[list[dict]] = None) -> int:
        return await self._index_cb.call(self._do_add_document, user_id, document_id, filename, text, page_ranges)

    @retry_with_backoff(max_retries=2, base_delay_s=0.5, retryable_exceptions=(ConnectionError, TimeoutError, OSError))
    async def _do_add_document(self, user_id: str, document_id: str, filename: str, text: str, page_ranges: Optional[list[dict]] = None) -> int:
        import time
        from ..services.async_worker import run_sync
        from qdrant_client.http import models

        t0 = time.monotonic()
        chunks = self.splitter.split_text(text)
        logger.info("VectorStore.add_document: chunked %d chars → %d chunks in %.2fs", len(text), len(chunks), time.monotonic() - t0)

        if not chunks:
            logger.warning("VectorStore.add_document: no chunks for %s/%s", user_id, document_id)
            return 0

        chunk_pages: list[int] = []
        if page_ranges:
            chunk_pages = self._map_chunks_to_pages(chunks, text, page_ranges)

        t1 = time.monotonic()
        vectors = await _run_embedding(self._model, chunks)
        logger.info("VectorStore.add_document: encoded %d chunks in %.2fs", len(chunks), time.monotonic() - t1)

        points = [
            models.PointStruct(
                id=uuid.uuid4().int & ((1 << 64) - 1),
                vector=v,
                payload={
                    "user_id": user_id,
                    "document_id": document_id,
                    "filename": filename,
                    "chunk_index": i,
                    "page": chunk_pages[i] if chunk_pages else None,
                    "text": chunk,
                },
            )
            for i, (chunk, v) in enumerate(zip(chunks, vectors))
        ]

        t2 = time.monotonic()
        BATCH_SIZE = 2000
        total = len(points)
        tasks = []
        for i in range(0, total, BATCH_SIZE):
            batch = points[i:i+BATCH_SIZE]
            tasks.append(run_sync(self.client.upsert, collection_name=self.collection_name, points=batch, wait=False))
        if tasks:
            await asyncio.gather(*tasks)
        logger.info("VectorStore.add_document: upserted %d points in %d parallel batches in %.2fs (total %.2fs)",
                     total, len(tasks), time.monotonic() - t2, time.monotonic() - t0)
        return len(chunks)

    async def _do_search(
        self,
        user_id: str,
        query: str,
        k: int = 10,
        document_ids: Optional[list[str]] = None,
    ) -> list[dict]:
        from ..services.async_worker import run_sync
        from qdrant_client.http import models

        embed_key = hashlib.sha256(query.encode()).hexdigest()
        cached = self._query_embed_cache.get(embed_key)
        if cached is not None:
            vec = cached
        else:
            vec = await _run_embedding(self._model, [query])
            self._query_embed_cache[embed_key] = vec
        flt = models.Filter(must=[models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))])
        if document_ids:
            flt.must.append(models.FieldCondition(key="document_id", match=models.MatchAny(any=document_ids)))
        logger.info("Qdrant.search: user=%s query=%.80s k=%d doc_ids=%s", user_id, query, k, document_ids)
        resp = await run_sync(
            self.client.query_points, collection_name=self.collection_name, query=vec[0], limit=k, query_filter=flt
        )
        n = len(resp.points)
        logger.info("Qdrant.search: got %d points", n)
        if n > 0:
            logger.info("Qdrant.search: first point doc_id=%s filename=%s score=%.4f",
                         resp.points[0].payload.get("document_id"),
                         resp.points[0].payload.get("filename"),
                         resp.points[0].score)
        return [
            {
                "id": r.id,
                "score": float(r.score),
                "document_id": r.payload.get("document_id"),
                "filename": r.payload.get("filename"),
                "chunk_index": r.payload.get("chunk_index"),
                "page": r.payload.get("page"),
                "text": r.payload.get("text", ""),
            }
            for r in resp.points
        ]

    async def rerank(self, query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
        if not candidates:
            return candidates
        self._ensure_reranker()
        import time
        t0 = time.monotonic()
        pairs = [(query, c.get("text", "")) for c in candidates]
        from ..services.async_worker import run_sync
        scores = await run_sync(self._reranker.predict, pairs)
        for i, c in enumerate(candidates):
            c["rerank_score"] = float(scores[i])
        candidates.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
        logger.info("Qdrant.rerank: reranked %d candidates → top %d in %.2fs", len(candidates), top_k, time.monotonic() - t0)
        return candidates[:top_k]

    async def search(
        self,
        user_id: str,
        query: str,
        k: int = 10,
        document_ids: Optional[list[str]] = None,
    ) -> list[dict]:
        cache = get_cache()
        doc_ids_sorted = sorted(document_ids) if document_ids else []
        raw_key = json.dumps({"user_id": user_id, "query": query, "doc_ids": doc_ids_sorted, "k": k}, sort_keys=True)
        cache_key = "search:" + hashlib.sha256(raw_key.encode()).hexdigest()
        cached = await cache.get(cache_key)
        if cached is not None:
            logger.info("Qdrant.search: cache hit for query=%.80s", query)
            return cached
        results = await self._search_cb.call(self._do_search, user_id, query, k, document_ids)
        await cache.set(cache_key, results, CACHE_TTL.get("query_result", 600))
        return results

    def delete_document(self, user_id: str, document_id: str) -> None:
        from qdrant_client.http import models
        flt = models.Filter(
            must=[
                models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id)),
                models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id)),
            ]
        )
        self.client.delete(collection_name=self.collection_name, points_selector=models.FilterSelector(filter=flt))


_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store


async def _run_embedding(model, texts: list[str]) -> list:
    from ..services.async_worker import run_sync
    vec = await run_sync(model.encode, texts, batch_size=128, show_progress_bar=False)
    if hasattr(vec, "tolist"):
        vec = vec.tolist()
    return vec
