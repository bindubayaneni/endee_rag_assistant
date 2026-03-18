from __future__ import annotations

"""
Ingestion pipeline.

Responsibilities:
- Chunk input text
- Embed chunks (batched + cached)
- Upsert into the configured vector store
- Store document_id + doc_name in payload to enable strict isolation
"""

import logging
import re
import uuid
from dataclasses import dataclass

from .embeddings import embed_texts
from .store_factory import get_document_store

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    chunk_index: int


def _simple_tokenize(text: str) -> list[str]:
    # Basic whitespace tokenizer; chunking is approximate by "tokens".
    return re.findall(r"\S+", text)


def _chunk_text(text: str, *, chunk_size_tokens: int, overlap_tokens: int) -> list[Chunk]:
    tokens = _simple_tokenize(text)
    if not tokens:
        return []

    if chunk_size_tokens <= 0:
        chunk_size_tokens = 300
    if overlap_tokens < 0:
        overlap_tokens = 0
    if overlap_tokens >= chunk_size_tokens:
        overlap_tokens = max(0, chunk_size_tokens // 5)

    chunks: list[Chunk] = []
    start = 0
    chunk_index = 0
    while start < len(tokens):
        end = min(len(tokens), start + chunk_size_tokens)
        chunk_tokens = tokens[start:end]
        chunk_text = " ".join(chunk_tokens).strip()
        if chunk_text:
            chunks.append(Chunk(id=f"c{chunk_index}", text=chunk_text, chunk_index=chunk_index))
            chunk_index += 1
        start = end - overlap_tokens if overlap_tokens else end

    return chunks


def ingest_text(
    text: str,
    *,
    doc_name: str,
    document_id: str | None = None,
    chunk_size_tokens: int = 500,
    overlap_tokens: int = 100,
) -> dict:
    """
    Ingest text into the active document store with strict document isolation.
    Returns ingestion metadata including the assigned document_id.
    """
    document_id = document_id or f"doc_{uuid.uuid4().hex}"

    chunks = _chunk_text(text, chunk_size_tokens=chunk_size_tokens, overlap_tokens=overlap_tokens)
    if not chunks:
        return {"ok": True, "document_id": document_id, "doc_name": doc_name, "chunks": 0}

    logger.info("Embedding chunks", extra={"chunks": len(chunks), "document_id": document_id})
    emb = embed_texts([c.text for c in chunks], batch_size=64)

    store = get_document_store()

    points = []
    for i, c in enumerate(chunks):
        payload = {
            # IMPORTANT: do NOT store raw full doc text elsewhere; the chunk text is needed for retrieval,
            # but we will never return it raw to the UI in query.py (sanitization layer).
            "text": c.text,
            "doc_name": doc_name,
            "document_id": document_id,
            "chunk_index": c.chunk_index,
        }
        points.append(
            {
                "id": f"{document_id}:{c.id}",
                "vector": emb.vectors[i],
                "payload": payload,
            }
        )

    logger.info("Upserting chunks", extra={"chunks": len(points), "document_id": document_id})
    store.upsert_points(points=points)

    return {"ok": True, "document_id": document_id, "doc_name": doc_name, "chunks": len(points)}