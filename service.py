from __future__ import annotations

"""Application service layer."""

import logging

from .config import settings
from .errors import ValidationError
from .ingest import ingest_text
from .memory import store_turn
from .query import answer_question

logger = logging.getLogger(__name__)


class RagService:
    def ingest_document(self, *, doc_name: str, text: str) -> dict:
        if not doc_name.strip():
            raise ValidationError("doc_name must not be empty")
        if not text.strip():
            raise ValidationError("text must not be empty")

        logger.info("Ingesting document", extra={"doc_name": doc_name})

        return ingest_text(
            text,
            doc_name=doc_name,
            document_id=None,
            chunk_size_tokens=settings.chunk_size_tokens,
            overlap_tokens=settings.overlap_tokens,
        )

    def ask(self, *, question: str, session_id: str, document_id: str) -> dict:
        if not question.strip():
            raise ValidationError("question must not be empty")
        if not document_id.strip():
            raise ValidationError("document_id must not be empty")

        logger.info("Answering question", extra={"session_id": session_id, "document_id": document_id})

        out = answer_question(
            question=question,
            session_id=session_id,
            document_id=document_id,
            top_k=settings.top_k,
            candidate_pool=settings.candidate_pool,
            alpha=settings.hybrid_alpha,
        )

        # Best-effort memory write (not document-isolated; it is session-isolated)
        try:
            store_turn(
                user_text=question,
                assistant_text=out["answer"],
                session_id=session_id,
            )
        except Exception:
            logger.exception("Failed to store memory turn")

        return out