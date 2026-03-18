from __future__ import annotations

"""Vector store selection.

Modes:
- VECTOR_STORE=faiss  -> always use local FAISS
- VECTOR_STORE=endee  -> always use Endee (will error if endpoints unavailable)
- VECTOR_STORE=auto   -> try Endee healthcheck, else FAISS

Recruiter note:
- 'auto' mode is common in production with optional dependencies.
"""

import logging
import os

import requests

from .config import settings
from .stores.endee_store import EndeeVectorStore
from .stores.faiss_store import FaissVectorStore
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


def _endee_is_healthy(base_url: str) -> bool:
    try:
        r = requests.get(base_url.rstrip("/") + "/api/v1/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def get_document_store() -> VectorStore:
    mode = (os.getenv("VECTOR_STORE") or "auto").lower().strip()

    if mode == "faiss":
        logger.warning("VECTOR_STORE=faiss (local fallback)")
        return FaissVectorStore(dim=384, namespace="docs")

    if mode == "endee":
        logger.info("VECTOR_STORE=endee")
        return EndeeVectorStore(base_url=settings.endee_url, index_name=settings.docs_index)

    # auto
    if _endee_is_healthy(settings.endee_url):
        logger.info("Endee healthcheck OK; attempting Endee vector store")
        return EndeeVectorStore(base_url=settings.endee_url, index_name=settings.docs_index)

    logger.warning("Endee not healthy or not reachable; using FAISS local store")
    return FaissVectorStore(dim=384, namespace="docs")


def get_memory_store() -> VectorStore:
    mode = (os.getenv("VECTOR_STORE") or "auto").lower().strip()

    if mode == "faiss":
        return FaissVectorStore(dim=384, namespace="memory")

    if mode == "endee":
        return EndeeVectorStore(base_url=settings.endee_url, index_name=settings.memory_index)

    if _endee_is_healthy(settings.endee_url):
        return EndeeVectorStore(base_url=settings.endee_url, index_name=settings.memory_index)

    return FaissVectorStore(dim=384, namespace="memory")
