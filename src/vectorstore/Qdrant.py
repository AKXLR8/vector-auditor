"""Qdrant vector store with user-isolated collections."""
import asyncio
import logging
import os
import uuid
from pathlib import Path
from typing import Optional

import asyncio

logger = logging.getLogger("rga_auditor.qdrant")

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"

DEFAULT_MAX_CITATIONS_PER_DOC = int(os.getenv("MAX_CITATIONS_PER_DOC", "6"))
DEFAULT_MAX_CITATIONS_TOTAL = int(os.getenv("MAX_CITATIONS_TOTAL", "20"))
DEFAULT_RETRIEVE_K = int(os.getenv("RETRIEVE_K_PER_QUERY", "10"))


class VectorStore:
    def __init__(
        self,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        collection_name: str = "documents",
    ) -> None:
        from qdrant_client import QdrantClient
        from sentence_transformers import SentenceTransformer
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        url = qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333")
        key = qdrant_api_key or os.getenv("QDRANT_API_KEY") or None
        self.client = QdrantClient(url=url, api_key=key)
        self.collection_name = collection_name
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        self.embedding_dim = 384
        self._model: SentenceTransformer
        self._create_collection()
        self._load_model()

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

    def _create_collection(self) -> None:
        from qdrant_client.http import models
        try:
            self.client.get_collection(self.collection_name)
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

    async def add_document(self, user_id: str, document_id: str, filename: str, text: str) -> int:
        import time
        from ..services.async_worker import run_sync
        from qdrant_client.http import models

        t0 = time.monotonic()
        chunks = self.splitter.split_text(text)
        logger.info("VectorStore.add_document: chunked %d chars → %d chunks in %.2fs", len(text), len(chunks), time.monotonic() - t0)

        if not chunks:
            logger.warning("VectorStore.add_document: no chunks for %s/%s", user_id, document_id)
            return 0

        t1 = time.monotonic()
        vectors = await run_sync(self._model.encode, chunks)
        if hasattr(vectors, "tolist"):
            vectors = vectors.tolist()
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
                    "text": chunk,
                },
            )
            for i, (chunk, v) in enumerate(zip(chunks, vectors))
        ]

        t2 = time.monotonic()
        await run_sync(self.client.upsert, collection_name=self.collection_name, points=points, wait=True)
        logger.info("VectorStore.add_document: upserted %d points in %.2fs (total %.2fs)", len(points), time.monotonic() - t2, time.monotonic() - t0)
        return len(chunks)

    async def search(
        self,
        user_id: str,
        query: str,
        k: int = 10,
        document_ids: Optional[list[str]] = None,
    ) -> list[dict]:
        from ..services.async_worker import run_sync
        from qdrant_client.http import models

        vec = await run_sync(self._model.encode, [query])
        if hasattr(vec, "tolist"):
            vec = vec.tolist()
        flt = models.Filter(must=[models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))])
        if document_ids:
            flt.must.append(models.FieldCondition(key="document_id", match=models.MatchAny(any=document_ids)))
        results = await run_sync(
            self.client.search, collection_name=self.collection_name, query_vector=vec[0], limit=k, query_filter=flt
        )
        return [
            {
                "id": r.id,
                "score": float(r.score),
                "document_id": r.payload.get("document_id"),
                "filename": r.payload.get("filename"),
                "chunk_index": r.payload.get("chunk_index"),
                "text": r.payload.get("text", ""),
            }
            for r in results
        ]

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
