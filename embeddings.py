from __future__ import annotations

"""
Embeddings utilities with caching.

Goals:
- Avoid recomputing embeddings for identical inputs (queries/chunks).
- Batch embedding calls for speed.
- Keep implementation simple and local-first.

Note: Cache is process-local (in-memory). For a multi-worker deployment,
use an external cache (Redis) or persist to disk.
"""

import hashlib
import logging
import threading
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EmbeddingResult:
    vectors: list[list[float]]
    dim: int


class _EmbeddingCache:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: dict[str, list[float]] = {}

    def get(self, key: str) -> list[float] | None:
        with self._lock:
            return self._data.get(key)

    def set(self, key: str, value: list[float]) -> None:
        with self._lock:
            self._data[key] = value


_MODEL: SentenceTransformer | None = None
_MODEL_LOCK = threading.Lock()
_CACHE = _EmbeddingCache()


def get_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> SentenceTransformer:
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    with _MODEL_LOCK:
        if _MODEL is None:
            logger.info("Loading SentenceTransformer model", extra={"model_name": model_name})
            _MODEL = SentenceTransformer(model_name)
    return _MODEL


def embed_texts(
    texts: list[str],
    *,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    normalize: bool = True,
    batch_size: int = 64,
) -> EmbeddingResult:
    """
    Embed a list of texts, using per-text cache, batching only the misses.
    Returns vectors aligned to 'texts' order.
    """
    if not texts:
        return EmbeddingResult(vectors=[], dim=0)

    model = get_model(model_name=model_name)

    keys = [_sha256(t) for t in texts]
    out: list[list[float] | None] = [_CACHE.get(k) for k in keys]

    misses_idx = [i for i, v in enumerate(out) if v is None]
    if misses_idx:
        misses = [texts[i] for i in misses_idx]
        logger.info("Embedding cache misses", extra={"count": len(misses_idx), "batch_size": batch_size})

        vecs = model.encode(
            misses,
            batch_size=batch_size,
            show_progress_bar=len(misses) >= 32,
            normalize_embeddings=normalize,
        )
        if isinstance(vecs, list):
            vecs = np.array(vecs, dtype=np.float32)

        for j, i in enumerate(misses_idx):
            v = vecs[j].astype(np.float32).tolist()
            out[i] = v
            _CACHE.set(keys[i], v)

    vectors = [v for v in out if v is not None]  # type: ignore[assignment]
    dim = len(vectors[0]) if vectors else 0
    return EmbeddingResult(vectors=vectors, dim=dim)


def embed_query(
    query: str,
    *,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> list[float]:
    """Embed a single query with caching."""
    res = embed_texts([query], model_name=model_name, normalize=True, batch_size=64)
    return res.vectors[0]


def embed_text(text: str) -> list[float]:
    """
    Backwards-compatible alias.

    Some modules (e.g. memory.py) import embed_text(). Keep it to avoid crashes.
    """
    return embed_query(text)