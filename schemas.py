from __future__ import annotations

"""API schemas (Pydantic models)."""

import uuid
from typing import List, Optional

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    doc_name: str = Field(..., description="Human-friendly document name")
    text: str = Field(..., description="Raw text content of the document")


class IngestStats(BaseModel):
    ok: bool = True
    document_id: str
    doc_name: str
    chunks: int


class IngestResponse(BaseModel):
    ok: bool
    document_id: str
    doc_name: str
    chunks: int
    stats: dict


class AskRequest(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str
    document_id: str = Field(..., description="Active document_id to enforce strict isolation")


class SourceItem(BaseModel):
    doc_name: Optional[str] = None
    document_id: Optional[str] = None
    chunk_index: Optional[int] = None
    snippet: Optional[str] = None
    score: float = 0.0


class AskResponse(BaseModel):
    ok: bool
    session_id: str
    answer: str
    sources: List[SourceItem]