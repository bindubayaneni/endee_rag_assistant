from __future__ import annotations

"""Central configuration for the application.

Recruiter note:
- In production we keep *all* tunables in one place and read them from env.
- This makes the system portable across local/dev/staging/prod.
"""

from dataclasses import dataclass

from .utils import env


def _to_int(value: str | None, default: int) -> int:
    try:
        return int(value) if value is not None else default
    except ValueError:
        return default


def _to_float(value: str | None, default: float) -> float:
    try:
        return float(value) if value is not None else default
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    # Endee
    endee_url: str = env("ENDEE_URL", "http://localhost:8080") or "http://localhost:8080"
    docs_index: str = env("ENDEE_DOCS_INDEX", "rag_documents") or "rag_documents"
    memory_index: str = env("ENDEE_MEMORY_INDEX", "rag_memory") or "rag_memory"

    # Retrieval
    top_k: int = _to_int(env("RAG_TOP_K", "5"), 5)
    candidate_pool: int = _to_int(env("RAG_CANDIDATE_POOL", "25"), 25)
    hybrid_alpha: float = _to_float(env("RAG_HYBRID_ALPHA", "0.65"), 0.65)

    # Chunking
    chunk_size_tokens: int = _to_int(env("CHUNK_SIZE_TOKENS", "500"), 500)
    overlap_tokens: int = _to_int(env("CHUNK_OVERLAP_TOKENS", "100"), 100)

    # LLM
    # Options: openai | ollama | mock
    llm_provider: str = (env("LLM_PROVIDER", "openai") or "openai").lower()

    # OpenAI
    # Recruiter note: we do NOT hardcode keys. They come from environment variables / secret managers.
    openai_api_key: str | None = env("OPENAI_API_KEY", None)
    openai_model: str = env("OPENAI_MODEL", "gpt-4o-mini") or "gpt-4o-mini"

    # Ollama (local)
    ollama_base_url: str = env("OLLAMA_BASE_URL", "http://localhost:11434") or "http://localhost:11434"
    ollama_model: str = env("OLLAMA_MODEL", "llama3.1:8b") or "llama3.1:8b"

    # Observability
    log_level: str = env("LOG_LEVEL", "INFO") or "INFO"


settings = Settings()
