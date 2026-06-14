"""Lite RAG agent — no LangChain, no LangGraph.

Pipeline: understand → retrieve → evaluate (loop) → generate → verify → gap_analysis
Two modes: white_box (full reasoning), black_box (temperature=0, minimal prompt).
Graceful degradation when LLM is unavailable.
"""
import asyncio
import json
import logging
import os
import re
import time
import uuid
from datetime import datetime
from typing import Optional

from ..models.schemas import (
    Citation,
    DocumentAnalysis,
    MessageHistory,
    Mode,
    QueryRequest,
    QueryResponse,
)

from ..services.cache import CACHE_TTL, cache_key, get_cache
from ..services.circuit_breaker import CircuitBreakerOpenError
from ..services.llm import LLM, LLMError, get_llm
from ..services.token_counter import estimate_cost, estimate_tokens
from ..vectorstore.Qdrant import get_vector_store

logger = logging.getLogger("rga_auditor.agent")


def _sanitize_citations(answer: str, max_citation: int) -> str:
    """Remove citation references [N] where N > max_citation (hallucinated)."""
    import re
    def _replace(m):
        num = int(m.group(1))
        return m.group(0) if 1 <= num <= max_citation else m.group(0).replace(f"[{num}]", f"(ref {num})")
    return re.sub(r'\[(\d+)\]', _replace, answer)


SYSTEM_WHITE_BOX = (
    "You are a meticulous research analyst. Answer the user's question using ONLY "
    "the provided context. Cite sources using [n] notation matching the numbered "
    "context blocks. CRITICAL: Only valid citation numbers are the ones shown "
    "in the Context section below (e.g., [1], [2], [3], ...). Never use citation "
    "numbers outside this range. If the context is insufficient, say so explicitly. "
    "Structure your answer with clear sections: Summary, Key Findings, Analysis, "
    "and Caveats. Be precise and concise."
)

SYSTEM_BLACK_BOX = (
    "You are a precise question-answering system. Answer the question using ONLY the "
    "provided context. Cite sources using [n] notation. CRITICAL: Only valid citation "
    "numbers are the ones shown in the Context section below (e.g., [1], [2], [3], ...). "
    "Never use citation numbers outside this range. No commentary, no reasoning, "
    "no caveats — just the cited answer."
)


class DocumentAgent:
    def __init__(
        self,
        llm: Optional[LLM] = None,
        vector_store=None,
        max_hops: Optional[int] = None,
        max_citations_total: Optional[int] = None,
        retrieve_k: Optional[int] = None,
        rerank_top_k: Optional[int] = None,
    ) -> None:
        self.llm = llm or get_llm()
        self.vs = vector_store or get_vector_store()
        self.max_hops = max_hops or int(os.getenv("MAX_DOCUMENT_HOPS", "5"))
        self.max_citations_total = max_citations_total or int(os.getenv("MAX_CITATIONS_TOTAL", "10"))
        self.retrieve_k = retrieve_k or int(os.getenv("RETRIEVE_K_PER_QUERY", "15"))
        self.rerank_top_k = rerank_top_k or int(os.getenv("RERANK_TOP_K", "10"))

    def _truncate_citations(self, citations: list[Citation]) -> list[Citation]:
        out: list[Citation] = []
        for c in citations:
            if len(c.quote) < 50:
                continue
            out.append(c)
            if self.max_citations_total and len(out) >= self.max_citations_total:
                break
        return out

    async def _retrieve(self, user_id: str, question: str, document_ids: Optional[list[str]], k: int) -> list[Citation]:
        logger.info("_retrieve: user=%s question=%.100s doc_ids=%s k=%d", user_id, question, document_ids, k)
        results = await self.vs.search(user_id=user_id, query=question, k=k, document_ids=document_ids)
        logger.info("_retrieve: search returned %d results", len(results))
        citations: list[Citation] = []
        for r in results:
            location = r.get("location") or f"chunk {r.get('chunk_index', 0)}"
            citations.append(
                Citation(
                    quote=r.get("text", ""),
                    source=r.get("filename", ""),
                    location=location,
                    page=r.get("page"),
                    document_id=r.get("document_id"),
                )
            )
        logger.info("_retrieve: built %d citations", len(citations))
        return self._truncate_citations(citations)

    async def _generate(self, question: str, context: list[Citation], mode: Mode, history: Optional[list[MessageHistory]]) -> tuple[str, int, int]:
        sys = SYSTEM_WHITE_BOX if mode == Mode.white_box else SYSTEM_BLACK_BOX
        ctx_lines = [f"[{i+1}] (source={c.source}, location={c.location})\n{c.quote}" for i, c in enumerate(context)]
        ctx_block = "\n\n".join(ctx_lines) if ctx_lines else "(no context)"
        history_block = ""
        if history:
            turns = "\n".join(f"{m.role}: {m.content}" for m in history[-6:])
            history_block = f"\n\nConversation so far:\n{turns}\n"
        prompt = f"Context:\n{ctx_block}\n{history_block}\nQuestion: {question}\nAnswer:"
        prompt_tokens = estimate_tokens(prompt)
        key = cache_key("llm", mode.value, question, str([(c.source, c.location) for c in context]))
        cache = get_cache()
        cached = await cache.get(key)
        if cached:
            answer_tokens = estimate_tokens(cached)
            return cached, prompt_tokens, answer_tokens
        try:
            answer = await self.llm.chat(prompt, system=sys, mode=mode.value)
            # Strip hallucinated citation numbers outside the valid range
            answer = _sanitize_citations(answer, len(context))
            await cache.set(key, answer, CACHE_TTL["llm_response"])
            answer_tokens = estimate_tokens(answer)
            return answer, prompt_tokens, answer_tokens
        except (CircuitBreakerOpenError, LLMError) as e:
            logger.warning("LLM unavailable for _generate: %s — returning context-only fallback", e)
            fallback = self._build_fallback_answer(question, context)
            answer_tokens = estimate_tokens(fallback)
            return fallback, prompt_tokens, answer_tokens

    @staticmethod
    def _build_fallback_answer(question: str, context: list[Citation]) -> str:
        if not context:
            return f"**LLM temporarily unavailable.** I found no context to answer: {question}"
        lines = [f"**LLM temporarily unavailable.** Here are relevant excerpts from your documents for: {question}\n"]
        for i, c in enumerate(context[:15]):
            src = c.source or "unknown"
            loc = c.location or ""
            pg = f" (p. {c.page})" if c.page else ""
            lines.append(f"*[{i+1}]* **{src}**{pg} — {loc}:")
            lines.append(f"  > {c.quote[:500]}")
        lines.append("\n*Please retry shortly when the LLM service is restored for a synthesized answer.*")
        return "\n".join(lines)

    async def _rerank_citations(self, question: str, citations: list[Citation], top_k: int) -> list[Citation]:
        if not citations:
            return citations
        candidates = [{"text": c.quote} for c in citations]
        reranked = await self.vs.rerank(question, candidates, top_k)
        kept = {c["text"] for c in reranked}
        result = [c for c in citations if c.quote in kept]
        logger.info("_rerank_citations: %d → %d after reranking", len(citations), len(result))
        return result

    @staticmethod
    def _enrich_bboxes(citations: list[Citation], upload_dir: str = "uploads") -> list[Citation]:
        from ..services.bbox_extractor import extract_bboxes_with_dimensions
        for c in citations:
            if c.page is None or not c.document_id:
                continue
            bboxes, w, h = extract_bboxes_with_dimensions(c.document_id, c.page, c.quote, upload_dir)
            if bboxes:
                c.bboxes = bboxes
            if w and h:
                c.page_width = w
                c.page_height = h
        return citations

    async def _verify(self, question: str, answer: str, context: list[Citation]) -> str:
        if not context:
            return "no context to verify against"
        ctx = "\n".join(f"[{i+1}] {c.quote}" for i, c in enumerate(context))
        prompt = (
            f"Question: {question}\n\nProposed answer: {answer}\n\nContext:\n{ctx}\n\n"
            "Is every claim in the answer supported by the context? Reply with one line: "
            "'VERIFIED' or 'ISSUES: <specific unsupported claims>'."
        )
        try:
            return await self.llm.chat(prompt, system="You are a strict fact-checker.", mode="verify")
        except (CircuitBreakerOpenError, LLMError) as e:
            logger.warning("LLM unavailable for _verify: %s", e)
            return "VERIFICATION_SKIPPED: LLM unavailable"

    async def _gaps(self, question: str, context: list[Citation]) -> list[str]:
        if not context:
            return []
        ctx = "\n".join(f"[{i+1}] {c.quote}" for i, c in enumerate(context))
        prompt = (
            f"Question: {question}\n\nContext:\n{ctx}\n\n"
            "List up to 3 specific research gaps or missing information in the context. "
            "Return as a JSON array of strings, e.g. [\"gap 1\", \"gap 2\"]."
        )
        try:
            raw = await self.llm.chat(prompt, system="You identify research gaps concisely.", mode="gaps")
        except (CircuitBreakerOpenError, LLMError) as e:
            logger.warning("LLM unavailable for _gaps: %s", e)
            return []
        m = re.search(r"\[.*?\]", raw, re.DOTALL)
        if not m:
            return []
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return []

    @staticmethod
    def _is_greeting(text: str) -> bool:
        return bool(re.match(r"^(hi|hello|hey|greetings|good\s*(morning|afternoon|evening)|sup|howdy|yo)\b", text.strip(), re.I))

    def apply_profile(self, profile: Optional[str]) -> None:
        if profile:
            from ..services.llm import LLM
            self.llm = LLM(profile=profile)

    async def query(self, user_id: str, req: QueryRequest) -> QueryResponse:
        self.apply_profile(req.model)
        if self._is_greeting(req.question):
            return QueryResponse(
                answer="Hi there! What would you like to know about your documents?",
                citations=[], reasoning_path=[], tokens_used=0, cost_usd=0.0,
                query_id=uuid.uuid4().hex, timestamp=datetime.utcnow().isoformat() + "Z",
                mode=req.mode,
            )
        t0 = time.time()
        k = self.retrieve_k
        all_citations: list[Citation] = []
        seen: set[tuple[str, str]] = set()
        questions = [req.question]
        for hop in range(self.max_hops):
            citations = await self._retrieve(user_id, questions[-1], req.document_ids, k)
            for c in citations:
                key = (c.source, c.location)
                if key not in seen:
                    seen.add(key)
                    all_citations.append(c)
            if self.max_citations_total and len(all_citations) >= self.max_citations_total or not citations:
                break
            if req.mode == Mode.white_box and citations:
                topics = ", ".join(c.quote[:80] for c in citations[:2])
                questions.append(f"More context related to: {topics} — regarding {req.question}")
            else:
                break
        all_citations = self._truncate_citations(all_citations)
        all_citations = await self._rerank_citations(req.question, all_citations, self.rerank_top_k)
        all_citations = self._enrich_bboxes(all_citations)

        reasoning_path: list[str] = []
        if req.mode == Mode.white_box:
            reasoning_path.append("Retrieved %d citations across %d hop(s)" % (len(all_citations), min(len(questions), self.max_hops)))

        answer, prompt_tokens, answer_tokens = await self._generate(req.question, all_citations, req.mode, req.conversation_history)
        if req.mode == Mode.white_box:
            reasoning_path.append("Generated answer with cite-and-explain framing")

        verification: Optional[str] = None
        gaps: list[str] = []
        if req.mode == Mode.white_box:
            verification = await self._verify(req.question, answer, all_citations)
            gaps = await self._gaps(req.question, all_citations)
            reasoning_path.append("Verified answer against source citations")
            if gaps:
                reasoning_path.append("Identified %d research gap(s)" % len(gaps))
            if verification.startswith("VERIFICATION_SKIPPED"):
                reasoning_path.append("Verification skipped — LLM unavailable")
            else:
                reasoning_path.append("VERIFIED" if verification.startswith("VERIFIED") else f"ISSUES: {verification[:120]}")

        cost = estimate_cost(prompt_tokens, answer_tokens)
        query_id = uuid.uuid4().hex
        return QueryResponse(
            answer=answer,
            citations=all_citations,
            reasoning_path=reasoning_path,
            tokens_used=prompt_tokens + answer_tokens,
            cost_usd=round(cost, 6),
            query_id=query_id,
            timestamp=datetime.utcnow().isoformat() + "Z",
            verification=verification if req.mode == Mode.white_box else None,
            mode=req.mode,
        )

    async def stream_query(self, user_id: str, req: QueryRequest):
        self.apply_profile(req.model)
        # Async generator yielding SSE-ready dicts.
        # Events: citations, token, verification, gap_analysis, done, error
        query_id = uuid.uuid4().hex
        timestamp = datetime.utcnow().isoformat() + "Z"
        if self._is_greeting(req.question):
            yield {"type": "citations", "citations": [], "query_id": query_id, "reasoning_path": []}
            yield {"type": "token", "content": "Hi there! What would you like to know about your documents?"}
            yield {"type": "done", "tokens_used": 0, "cost_usd": 0.0, "mode": req.mode.value, "query_id": query_id, "timestamp": timestamp}
            return
        try:
            k = self.retrieve_k
            all_citations: list[Citation] = []
            seen: set[tuple[str, str]] = set()
            questions = [req.question]
            for hop in range(self.max_hops):
                citations = await self._retrieve(user_id, questions[-1], req.document_ids, k)
                for c in citations:
                    key = (c.source, c.location)
                    if key not in seen:
                        seen.add(key)
                        all_citations.append(c)
                if self.max_citations_total and len(all_citations) >= self.max_citations_total or not citations:
                    break
                if req.mode == Mode.white_box and citations:
                    topics = ", ".join(c.quote[:80] for c in citations[:2])
                    questions.append(f"More context related to: {topics} — regarding {req.question}")
                else:
                    break
            all_citations = self._truncate_citations(all_citations)
            all_citations = await self._rerank_citations(req.question, all_citations, self.rerank_top_k)
            all_citations = self._enrich_bboxes(all_citations)

            reasoning_path: list[str] = []
            if req.mode == Mode.white_box:
                reasoning_path.append(f"Retrieved {len(all_citations)} citations across {len(questions)} hop(s)")

            yield {
                "type": "citations",
                "citations": [c.model_dump() for c in all_citations],
                "query_id": query_id,
                "reasoning_path": reasoning_path,
            }

            yield {
                "type": "status",
                "content": "it may take a while to generate cuz it's using Max reasoning on CPU hardware",
                "query_id": query_id,
            }

            sys = SYSTEM_WHITE_BOX if req.mode == Mode.white_box else SYSTEM_BLACK_BOX
            ctx_lines = [f"[{i+1}] (source={c.source}, location={c.location})\n{c.quote}" for i, c in enumerate(all_citations)]
            ctx_block = "\n\n".join(ctx_lines) if ctx_lines else "(no context)"
            prompt = f"Context:\n{ctx_block}\n\nQuestion: {req.question}\nAnswer:"

            prompt_tokens = estimate_tokens(prompt)
            total_tokens = prompt_tokens
            answer_buf: list[str] = []
            async for chunk in self.llm.astream(prompt, system=sys, mode=req.mode.value):
                total_tokens += estimate_tokens(chunk)
                answer_buf.append(chunk)
                yield {"type": "token", "content": chunk}

            full_answer = "".join(answer_buf)
            # Sanitize hallucinated citations before verify/gaps
            full_answer = _sanitize_citations(full_answer, len(all_citations))
            answer_tokens = total_tokens - prompt_tokens
            cost = estimate_cost(prompt_tokens, answer_tokens)

            if req.mode == Mode.white_box:
                verification = await self._verify(req.question, full_answer, all_citations)
                yield {"type": "verification", "content": verification}
                reasoning_path.append("Verification: " + ("passed" if verification.startswith("VERIFIED") else "issues found"))
                gaps = await self._gaps(req.question, all_citations)
                for g in gaps:
                    yield {"type": "gap_analysis", "content": g}
                if gaps:
                    reasoning_path.append(f"Identified {len(gaps)} research gap(s)")

            yield {
                "type": "done",
                "tokens_used": total_tokens,
                "cost_usd": round(cost, 6),
                "mode": req.mode.value,
                "query_id": query_id,
                "timestamp": timestamp,
            }
        except Exception as e:
            logger.exception("stream_query failed")
            yield {"type": "error", "detail": str(e)[:500], "query_id": query_id}

    async def analyze_document(self, user_id: str, question: Optional[str], document_ids: Optional[list[str]],
                                 max_citations: Optional[int] = None, model: Optional[str] = None) -> DocumentAnalysis:
        self.apply_profile(model)
        req_doc_ids = document_ids or []
        logger.info("analyze_document: user=%s question=%r doc_ids=%s max_cit=%s",
                     user_id, question, req_doc_ids, max_citations)
        if question and re.match(r"^(hi|hello|hey|greetings|good\s*(morning|afternoon|evening)|sup|howdy|yo)\b", question.strip(), re.I):
            logger.info("analyze_document: greeting detected, returning early")
            return DocumentAnalysis(
                summary="Hi there! Please select documents to analyze, or ask me a question about your documents.",
                key_findings=[], methodology="",
                research_gaps=[], contradictions=[], open_questions=[],
                limitations="", confidence="high",
                citations=[], documents_analyzed=[],
            )
        if not req_doc_ids:
            logger.info("analyze_document: no doc_ids provided, fetching all user docs")
            from ..database.repository import list_documents
            from ..database.session import get_session_factory
            sf = get_session_factory()
            if sf:
                async with sf() as s:
                    all_docs = await list_documents(s, user_id=user_id)
            else:
                all_docs = await list_documents(None, user_id=user_id)
            ready = [d for d in all_docs if d.get("status") == "ready"]
            if not ready:
                logger.info("analyze_document: no ready docs found for user")
                return DocumentAnalysis(
                    summary="No processed documents found. Upload a document first.",
                    key_findings=[], methodology="",
                    research_gaps=[], contradictions=[], open_questions=[],
                    limitations="No documents available.", confidence="low",
                    citations=[], documents_analyzed=[],
                )
            req_doc_ids = [d["id"] for d in ready]
            logger.info("analyze_document: auto-selected %d ready docs: %s", len(req_doc_ids), req_doc_ids)
        if len(req_doc_ids) > 1:
            per_doc = max(3, 12 // len(req_doc_ids))
            logger.info("analyze_document: multi-doc mode (%d docs), per_doc=%d", len(req_doc_ids), per_doc)
            citations: list[Citation] = []
            for did in req_doc_ids:
                part = await self._retrieve(user_id, question or "key findings and methodology", [did], per_doc)
                logger.info("analyze_document: retrieved %d citations for doc=%s", len(part), did)
                citations.extend(part)
            if citations:
                dedup = {}
                for c in citations:
                    dedup.setdefault(c.quote, c)
                citations = list(dedup.values())[:max_citations or self.max_citations_total]
        else:
            logger.info("analyze_document: single-doc mode, doc=%s", req_doc_ids[0])
            citations = await self._retrieve(user_id, question or "key findings and methodology", req_doc_ids or None,
                                              self.max_citations_total)
        citations = self._truncate_citations(citations)
        citations = await self._rerank_citations(question or "key findings and methodology", citations, self.rerank_top_k)
        citations = self._enrich_bboxes(citations)
        logger.info("analyze_document: total citations after truncation+rerank=%d", len(citations))
        if not citations:
            logger.info("analyze_document: no citations found, returning empty")
            return DocumentAnalysis(
                summary="No content found.", key_findings=[], methodology="",
                research_gaps=[], contradictions=[], open_questions=[],
                limitations="No documents accessible.", confidence="low",
                citations=[], documents_analyzed=[],
            )
        ctx = "\n\n".join(f"[{i+1}] (source={c.source}) {c.quote}" for i, c in enumerate(citations))
        focus = question or "Provide a structured analysis comparing all documents."
        def _norm(s: str) -> str:
            return re.sub(r"[\s_-]+", "_", s.strip().lower())
        seen_docs: dict[str, str] = {}
        for c in citations:
            key = _norm(c.source)
            seen_docs.setdefault(key, c.source)
        docs_analyzed = sorted({c.source for c in citations})
        doc_ids_with_data = sorted({c.document_id for c in citations if c.document_id})
        doc_names = sorted({c.source for c in citations})
        is_multi = len(doc_names) > 1
        prompt = (
            f"Based on the following document excerpts, answer the user's question "
            f"in a detailed, structured, well-reasoned manner. "
            f"Do NOT include a summary section — directly address what was asked."
            f"\n\nDocuments analyzed ({len(doc_names)}): {', '.join(doc_names)}"
            f"\n\nQuestion/focus: {focus}\n\nContext:\n{ctx}\n\n"
            f"Answer:"
        )
        logger.info("analyze_document: calling LLM with context length=%d chars", len(ctx))
        try:
            raw = await self.llm.chat(prompt, system="You are a research analyst. Answer the user's question directly and thoroughly.", mode="analyze")
        except (CircuitBreakerOpenError, LLMError) as e:
            logger.warning("LLM unavailable for analyze_document: %s — returning raw-context analysis", e)
            raw_citations = "\n\n".join(f"[{i+1}] **{c.source}**" + (f" (p. {c.page})" if c.page else "") + f":\n  {c.quote[:300]}" for i, c in enumerate(citations[:10]))
            return DocumentAnalysis(
                summary=f"**LLM temporarily unavailable.** Raw context from {len(citations)} excerpts.",
                key_findings=[f"Found {len(citations)} relevant passages across {len(docs_analyzed)} documents."],
                methodology="LLM service unavailable — showing raw excerpts only.",
                research_gaps=[], contradictions=[], open_questions=[],
                limitations="Full analysis requires LLM service. Showing raw excerpts below:\n\n" + raw_citations,
                confidence="low",
                citations=citations,
                documents_analyzed=docs_analyzed if not is_multi else doc_ids_with_data or docs_analyzed,
            )
        logger.info("analyze_document: LLM raw response length=%d chars", len(raw))
        return DocumentAnalysis(
            summary=raw,
            citations=citations,
            documents_analyzed=doc_ids_with_data or docs_analyzed,
        )
