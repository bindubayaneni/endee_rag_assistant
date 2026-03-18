from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, List

from .embeddings import embed_text
from .store_factory import get_memory_store

logger = logging.getLogger(__name__)


def _mask_memory(text: str) -> str:
    import re

    email_re = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
    number_re = re.compile(r"\b\d+(?:[\d,]*)(?:\.\d+)?\b")

    text = email_re.sub("[EMAIL]", text)
    text = number_re.sub("[NUMBER]", text)
    return text


def store_turn(*, user_text: str, assistant_text: str, session_id: str) -> None:
    """Store a conversation turn as a memory item in the active vector store (sanitized)."""

    combined = f"User: {user_text}\nAssistant: {assistant_text}".strip()
    vec = embed_text(combined)

    safe_text = _mask_memory(combined)
    payload = {
        "type": "memory_turn",
        "session_id": session_id,
        # Do NOT store raw user/assistant text to reduce sensitive retention
        "snippet_masked": safe_text[:400] + ("…" if len(safe_text) > 400 else ""),
        "created_at": int(time.time()),
    }

    point = {"id": str(uuid.uuid4()), "vector": vec, "payload": payload}

    store = get_memory_store()
    store.upsert_points([point])


def retrieve_memory(*, query: str, session_id: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """Retrieve conversation memory items relevant to this query."""
    store = get_memory_store()
    vec = embed_text(query)

    # Support both old and new store signatures by using keywords and dict hits.
    matches = store.search(vector=vec, top_k=top_k, filter={"session_id": session_id}, candidate_pool=max(10, top_k * 3))

    filtered: List[Dict[str, Any]] = []
    for m in matches:
        payload = m.get("payload") or {}
        if payload.get("session_id") == session_id:
            filtered.append({"id": m.get("id"), "score": m.get("score", 0.0), "payload": payload})

    logger.debug("Memory retrieval", extra={"session_id": session_id, "returned": len(filtered)})
    return filtered