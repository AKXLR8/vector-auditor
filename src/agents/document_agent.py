"""Lite RAG agent — no LangChain, no LangGraph.

Pipeline: understand → retrieve → evaluate (loop) → generate → verify → gap_analysis
Two modes: white_box (full reasoning), black_box (temperature=0, minimal prompt).
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
from ..services.llm import LLM, get_llm
from ..services.token_counter import estimate_cost, estimate_tokens
from ..vectorstore.Qdrant import get_vector_store

logger = logging.getLogger("rga_auditor.agent")


SYSTEM_WHITE_BOX = (
    "You are a meticulous research analyst. Answer the user's question using ONLY "
    "the provided context. Cite sources using [n] notation matching the numbered "
    "context blocks. If the context is insufficient, say so explicitly. "
    "Structure your answer with clear sections: Summary, Key Findings, Analysis, "
    "and Caveats. Be precise and concise."
)

SYSTEM_BLACK_BOX = (
    "You are a precise question-answering system. Answer the question using ONLY the "
    "provided context. Cite sources using [n] notation. No commentary, no reasoning, "
    "no caveats — just the cited answer."
)


class DocumentAgent:
    def __init__(
        self,
        llm: Optional[LLM] = None,
        vector_store=None,
        max_hops: Optional[int] = None,
        max_citations_per_doc: Optional[int] = None,
        max_citations_total: Optional[int] = None,
        retrieve_k: Optional[int] = None,
    ) -> None:
        self.llm = llm or get_llm()
        self.vs = vector_store or get_vector_store()
        self.max_hops = max_hops or int(os.getenv("MAX_DOCUMENT_HOPS", "3"))
        self.max_citations_per_doc = max_citations_per_doc or int(os.getenv("MAX_CITATIONS_PER_DOC", "6"))
        self.max_citations_total = max_citations_total or int(os.getenv("MAX_CITATIONS_TOTAL", "20"))
        self.retrieve_k = retrieve_k or int(os.getenv("RETRIEVE_K_PER_QUERY", "10"))

    def _per_doc_caps(self, override: Optional[int]) -> int:
        return min(self.max_citations_per_doc, override or self.max_citations_per_doc)

    def _truncate_citations(self, citations: list[Citation]) -> list[Citation]:
        per_doc: dict[str, int] = {}
        out: list[Citation] = []
        for c in citations:
            doc_key = c.source or "unknown"
            n = per_doc.get(doc_key, 0)
            if n >= self.max_citations_per_doc:
                continue
            per_doc[doc_key] = n + 1
            out.append(c)
            if len(out) >= self.max_citations_total:
                break
        return out

    async def _retrieve(self, user_id: str, question: str, document_ids: Optional[list[str]], k: int) -> list[Citation]:
        results = await self.vs.search(user_id=user_id, query=question, k=k, document_ids=document_ids)
        citations: list[Citation] = []
        for r in results:
            location = r.get("location") or f"chunk {r.get('chunk_index', 0)}"
            citations.append(
                Citation(
                    quote=r.get("text", ""),
                    source=r.get("filename", ""),
                    location=location,
                    page=r.get("page"),
                )
            )
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
        temperature = 0.0 if mode == Mode.black_box else None
        max_tokens = 3048 if mode == Mode.white_box else None
        answer = await self.llm.chat(prompt, system=sys, temperature=temperature, max_tokens=max_tokens)
        await cache.set(key, answer, CACHE_TTL["llm_response"])
        answer_tokens = estimate_tokens(answer)
        return answer, prompt_tokens, answer_tokens

    async def _verify(self, question: str, answer: str, context: list[Citation]) -> str:
        if not context:
            return "no context to verify against"
        ctx = "\n".join(f"[{i+1}] {c.quote[:300]}" for i, c in enumerate(context[:5]))
        prompt = (
            f"Question: {question}\n\nProposed answer: {answer}\n\nContext:\n{ctx}\n\n"
            "Is every claim in the answer supported by the context? Reply with one line: "
            "'VERIFIED' or 'ISSUES: <specific unsupported claims>'."
        )
        return await self.llm.chat(prompt, system="You are a strict fact-checker.")

    async def _gaps(self, question: str, context: list[Citation]) -> list[str]:
        if not context:
            return []
        ctx = "\n".join(c.quote[:200] for c in context[:8])
        prompt = (
            f"Question: {question}\n\nContext:\n{ctx}\n\n"
            "List up to 3 specific research gaps or missing information in the context. "
            "Return as a JSON array of strings, e.g. [\"gap 1\", \"gap 2\"]."
        )
        raw = await self.llm.chat(prompt, system="You identify research gaps concisely.")
        m = re.search(r"\[.*?\]", raw, re.DOTALL)
        if not m:
            return []
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return []

    async def query(self, user_id: str, req: QueryRequest) -> QueryResponse:
        t0 = time.time()
        k = self._per_doc_caps(req.max_citations) * 2
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
            if len(all_citations) >= self.max_citations_total or not citations:
                break
            if req.mode == Mode.white_box and len(citations) >= self.max_citations_per_doc:
                topics = ", ".join(c.quote[:80] for c in citations[:2])
                questions.append(f"More context related to: {topics} — regarding {req.question}")
            else:
                break
        all_citations = self._truncate_citations(all_citations)

        reasoning_path: list[str] = []
        if req.mode == Mode.white_box:
            reasoning_path.append("Retrieved %d candidate citations across %d hop(s)" % (len(all_citations), min(len(questions), self.max_hops)))

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
        """Async generator yielding SSE-ready dicts for the spec event types.

        Event types yielded:
          {type: "citations", citations, query_id, reasoning_path}
          {type: "token", content} (per LLM chunk)
          {type: "verification", content} (white_box only)
          {type: "gap_analysis", content} (white_box only, one event per gap)
          {type: "done", tokens_used, mode, query_id}
          {type: "error", detail}  (on failure)
        """
        query_id = uuid.uuid4().hex
        timestamp = datetime.utcnow().isoformat() + "Z"
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
                if len(all_citations) >= self.max_citations_total or not citations:
                    break
                if req.mode == Mode.white_box and len(citations) >= self.max_citations_per_doc:
                    topics = ", ".join(c.quote[:80] for c in citations[:2])
                    questions.append(f"More context related to: {topics} — regarding {req.question}")
                else:
                    break
            all_citations = self._truncate_citations(all_citations)

            reasoning_path: list[str] = []
            if req.mode == Mode.white_box:
                reasoning_path.append(f"Retrieved {len(all_citations)} citations across {len(questions)} hop(s)")

            yield {
                "type": "citations",
                "citations": [c.model_dump() for c in all_citations],
                "query_id": query_id,
                "reasoning_path": reasoning_path,
            }

            sys = SYSTEM_WHITE_BOX if req.mode == Mode.white_box else SYSTEM_BLACK_BOX
            ctx_lines = [f"[{i+1}] (source={c.source}, location={c.location})\n{c.quote}" for i, c in enumerate(all_citations)]
            ctx_block = "\n\n".join(ctx_lines) if ctx_lines else "(no context)"
            prompt = f"Context:\n{ctx_block}\n\nQuestion: {req.question}\nAnswer:"

            prompt_tokens = estimate_tokens(prompt)
            total_tokens = prompt_tokens
            answer_buf: list[str] = []
            temperature = 0.0 if req.mode == Mode.black_box else None
            max_tokens = 3048 if req.mode == Mode.white_box else None
            async for chunk in self.llm.astream(prompt, system=sys, temperature=temperature, max_tokens=max_tokens):
                total_tokens += estimate_tokens(chunk)
                answer_buf.append(chunk)
                yield {"type": "token", "content": chunk}

            full_answer = "".join(answer_buf)
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
                                max_citations: Optional[int] = None) -> DocumentAnalysis:
        doc_ids = document_ids or []
        # Retrieve fairly from each document
        if len(doc_ids) > 1:
            per_doc = max(3, 12 // len(doc_ids))
            citations: list[Citation] = []
            for did in doc_ids:
                part = await self._retrieve(user_id, question or "key findings and methodology", [did], per_doc)
                citations.extend(part)
            if citations:
                dedup = {}
                for c in citations:
                    dedup.setdefault(c.quote, c)
                citations = list(dedup.values())[:max_citations or 20]
        else:
            citations = await self._retrieve(user_id, question or "key findings and methodology", doc_ids or None,
                                              min(self.retrieve_k * 2, 20))
        citations = self._truncate_citations(citations)
        if not citations:
            return DocumentAnalysis(
                summary="No content found.", key_findings=[], methodology="",
                research_gaps=[], contradictions=[], open_questions=[],
                limitations="No documents accessible.", confidence="low",
                citations=[], documents_analyzed=doc_ids,
            )
        ctx = "\n\n".join(f"[{i+1}] (source={c.source}) {c.quote}" for i, c in enumerate(citations))
        focus = question or "Provide a structured analysis comparing all documents."
        is_multi = len(doc_ids) > 1
        structure_instruction = (
            "summary, key_findings (array of 3-5), methodology, "
            "research_gaps (array of 2-4), contradictions (array), open_questions (array of 2-3), "
            "limitations, confidence ('high'|'moderate'|'low')"
        )
        if is_multi:
            structure_instruction += (
                ", cross_document_comparison (object with keys: common_themes, differences, "
                "complementary_insights), per_document_summary (object keyed by filename with short summary)"
            )
        prompt = (
            f"Based on the following document excerpts, produce a JSON object with keys: "
            f"{structure_instruction}."
            f"\n\nDocuments analyzed: {', '.join({c.source for c in citations})}"
            f"\n\nQuestion/focus: {focus}\n\nContext:\n{ctx}\n\n"
            f"Return ONLY a valid JSON object, no commentary."
        )
        raw = await self.llm.chat(prompt, system="You are a research analyst. Output JSON only.")
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        data: dict = {}
        if m:
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError:
                data = {}
        if not data:
            data = {"summary": raw[:500], "key_findings": [], "methodology": "",
                    "research_gaps": [], "contradictions": [], "open_questions": [],
                    "limitations": "Could not parse structured output.", "confidence": "low"}
        return DocumentAnalysis(
            summary=data.get("summary", ""),
            key_findings=data.get("key_findings", []),
            methodology=data.get("methodology", ""),
            research_gaps=data.get("research_gaps", []),
            contradictions=data.get("contradictions", []),
            open_questions=data.get("open_questions", []),
            limitations=data.get("limitations", ""),
            confidence=data.get("confidence", "moderate"),
            citations=citations,
            documents_analyzed=doc_ids or list({c.source for c in citations}),
        )
