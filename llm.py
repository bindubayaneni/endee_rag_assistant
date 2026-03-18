from __future__ import annotations

"""LLM interface.

Recruiter note:
- This abstraction keeps vendor-specific code (OpenAI / Ollama) out of the core RAG logic.
- If we switch providers or add streaming, only this module changes.
"""

import logging
from typing import Any

import requests

from .config import settings
from .errors import ExternalServiceError

logger = logging.getLogger(__name__)


def _mock_answer() -> str:
    # Keep deterministic text for easy demos/tests.
    return (
        "(Mock answer — configure LLM_PROVIDER=ollama or set OPENAI_API_KEY to enable live LLM.)\n\n"
        "Based on the retrieved sources, here is a grounded summary:\n"
        "- RAG combines retrieval with generation to ground answers in external context.\n"
        "- Vector databases store embeddings and enable semantic similarity search.\n"
        "- Hybrid retrieval improves results by combining semantic + keyword scoring.\n\n"
        "Citations: [Source 1]"
    )


def _generate_with_openai(prompt: str) -> str:
    if not settings.openai_api_key:
        return _mock_answer()

    try:
        from openai import OpenAI

        client = OpenAI(api_key=settings.openai_api_key)
        resp = client.chat.completions.create(
            model=settings.openai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        # Common case: insufficient_quota / 429
        logger.exception("OpenAI call failed")
        raise ExternalServiceError(f"LLM generation failed (OpenAI): {e}") from e


def _generate_with_ollama(prompt: str) -> str:
    """Generate with local Ollama.

    Uses Ollama's chat endpoint: http://localhost:11434/api/chat
    """

    url = f"{settings.ollama_base_url.rstrip('/')}/api/chat"
    payload: dict[str, Any] = {
        "model": settings.ollama_model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }

    try:
        r = requests.post(url, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        # Ollama returns: { message: { role, content }, ... }
        msg = (data or {}).get("message") or {}
        return (msg.get("content") or "").strip()
    except requests.RequestException as e:
        logger.exception("Ollama request failed")
        raise ExternalServiceError(f"LLM generation failed (Ollama): {e}") from e
    except ValueError as e:
        logger.exception("Ollama returned invalid JSON")
        raise ExternalServiceError("LLM generation failed (Ollama): invalid JSON") from e


def generate_answer(prompt: str) -> str:
    """Generate a grounded answer from a prompt.

    Behavior:
    - LLM_PROVIDER=ollama => use local Ollama.
    - LLM_PROVIDER=mock => deterministic mock.
    - LLM_PROVIDER=openai (default) => use OpenAI if key exists; otherwise mock.

    Notes:
    - We intentionally *do not* crash the whole request if the provider is misconfigured.
      Instead we surface a clear error.
    """

    provider = (settings.llm_provider or "openai").lower()

    if provider == "mock":
        return _mock_answer()

    if provider == "ollama":
        return _generate_with_ollama(prompt)

    if provider == "openai":
        # If OpenAI is configured but fails due to quota/rate limiting, treat it as an external failure.
        return _generate_with_openai(prompt)

    raise ExternalServiceError(
        f"Unknown LLM_PROVIDER='{settings.llm_provider}'. Use one of: openai | ollama | mock."
    )
