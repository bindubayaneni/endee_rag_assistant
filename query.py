from __future__ import annotations

import logging
from dataclasses import dataclass

from .config import settings
from .embeddings import embed_query
from .llm import generate_answer
from .store_factory import get_document_store

logger = logging.getLogger(__name__)


def _snippet_from_payload(payload: dict) -> str:
    # Prefer safe masked snippet stored in payload (FAISS persists only this).
    snip = payload.get("snippet_masked")
    if snip:
        return str(snip)
    # Fallback: if some store returns raw text, never return it raw.
    # Keep it minimal and assume store has already masked it or it is safe.
    text = str(payload.get("text") or "")
    text = " ".join(text.split())
    return text[:220] + ("…" if len(text) > 220 else "")


@dataclass(frozen=True)
class RetrievedSource:
    doc_name: str
    document_id: str
    chunk_index: int
    snippet: str
    score: float


def answer_question(
    *,
    question: str,
    session_id: str,
    document_id: str | None = None,
    top_k: int | None = None,
    candidate_pool: int | None = None,
    alpha: float | None = None,
) -> dict:
    if not document_id:
        return {"answer": "Please upload and ingest a document first (no active document selected).", "sources": []}

    top_k = int(top_k or settings.top_k or 5)
    top_k = max(3, min(top_k, 5))
    candidate_pool = int(candidate_pool or settings.candidate_pool or 25)

    qvec = embed_query(question)
    store = get_document_store()

    hits = store.search(vector=qvec, top_k=top_k, filter={"document_id": document_id}, candidate_pool=candidate_pool)

    sources: list[RetrievedSource] = []
    for h in hits[:top_k]:
        payload = h.get("payload") or {}
        sources.append(
            RetrievedSource(
                doc_name=str(payload.get("doc_name") or "unknown"),
                document_id=str(payload.get("document_id") or ""),
                chunk_index=int(payload.get("chunk_index") or 0),
                snippet=_snippet_from_payload(payload),
                score=float(h.get("score") or 0.0),
            )
        )

    context_lines = [
        f"[Source {i}] doc={s.doc_name} chunk={s.chunk_index} snippet={s.snippet}" for i, s in enumerate(sources, 1)
    ]
    safe_context = "\n".join(context_lines) if context_lines else "No relevant sources found."

    system_instructions = (
        "You are a RAG assistant. Answer strictly using the provided source snippets.\n"
        "Rules:\n"
        "- Do NOT reproduce long passages verbatim.\n"
        "- The snippets may be masked; do not attempt to guess masked parts.\n"
        "- If sources do not contain the answer, say so.\n"
        "- Always include citations like [Source 1]."
    )
    user_prompt = f"Question: {question}\n\nSources:\n{safe_context}\n\nWrite a concise answer."

    answer = generate_answer(system=system_instructions, prompt=user_prompt)

    return {
        "answer": answer,
        "sources": [
            {
                "doc_name": s.doc_name,
                "document_id": s.document_id,
                "chunk_index": s.chunk_index,
                "snippet": s.snippet,
                "score": s.score,
            }
            for s in sources
        ],
    }