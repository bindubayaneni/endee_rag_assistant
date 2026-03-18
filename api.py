from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import settings
from .embeddings import get_model
from .errors import AppError, ExternalServiceError, ValidationError
from .logging_setup import setup_logging
from .schemas import AskRequest, AskResponse, IngestRequest, IngestResponse
from .service import RagService

load_dotenv()

setup_logging(settings.log_level)
logger = logging.getLogger(__name__)

service = RagService()

app = FastAPI(title="Endee Hybrid RAG Assistant", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup_warmup() -> None:
    # Warm up embedding model so first ingest doesn't appear hung.
    try:
        get_model()
        logger.info("Startup warmup complete: embeddings model loaded")
    except Exception:
        logger.exception("Startup warmup failed (embeddings model)")


@app.exception_handler(ValidationError)
def handle_validation_error(_, exc: ValidationError):
    return JSONResponse(status_code=400, content={"ok": False, "error": str(exc)})


@app.exception_handler(ExternalServiceError)
def handle_external_error(_, exc: ExternalServiceError):
    return JSONResponse(status_code=503, content={"ok": False, "error": str(exc)})


@app.exception_handler(AppError)
def handle_app_error(_, exc: AppError):
    return JSONResponse(status_code=500, content={"ok": False, "error": str(exc)})


@app.exception_handler(Exception)
def handle_unexpected_error(_, exc: Exception):
    logger.exception("Unhandled server error")
    return JSONResponse(status_code=500, content={"ok": False, "error": "Internal server error"})


@app.get("/health")
def health():
    vector_store = (os.getenv("VECTOR_STORE") or "auto").lower().strip()
    llm_provider = (settings.llm_provider or "openai").lower().strip()

    return {
        "ok": True,
        "endee_url": settings.endee_url,
        "vector_store": vector_store,
        "llm_provider": llm_provider,
        "ollama": {"base_url": settings.ollama_base_url, "model": settings.ollama_model} if llm_provider == "ollama" else None,
        "openai": {"model": settings.openai_model, "api_key_present": bool(settings.openai_api_key)} if llm_provider == "openai" else None,
    }


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest):
    stats = service.ingest_document(doc_name=req.doc_name, text=req.text)
    return {
        "ok": True,
        "document_id": stats.get("document_id"),
        "doc_name": stats.get("doc_name"),
        "chunks": stats.get("chunks", 0),
        "stats": stats,
    }


@app.post("/ingest_file", response_model=IngestResponse)
async def ingest_file(file: UploadFile = File(...)):
    if not file.filename:
        raise ValidationError("Uploaded file must have a filename")

    content = (await file.read()).decode("utf-8", errors="replace")
    stats = service.ingest_document(doc_name=file.filename, text=content)
    return {
        "ok": True,
        "document_id": stats.get("document_id"),
        "doc_name": stats.get("doc_name") or file.filename,
        "chunks": stats.get("chunks", 0),
        "stats": stats,
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    out = service.ask(question=req.question, session_id=req.session_id, document_id=req.document_id)
    return {"ok": True, "session_id": req.session_id, "answer": out["answer"], "sources": out["sources"]}