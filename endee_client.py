from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List

import requests

from .errors import ExternalServiceError


class EndeeError(ExternalServiceError):
    """Raised when Endee (vector DB) is unavailable or returns invalid responses."""


@dataclass
class EndeeConfig:
    base_url: str = "http://localhost:8080"
    index_name: str = "rag_documents"
    timeout_s: float = 15.0


class EndeeClient:
    """Thin HTTP client for Endee.

    Recruiter note:
    - Real systems isolate external dependencies (vector DB, LLM APIs) behind a client.
    - This keeps the rest of the codebase testable and vendor-agnostic.

    IMPORTANT:
    Endee is an open-source vector DB; endpoint shapes vary by version.

    This client supports two common patterns:

    1) Qdrant-compatible style:
       - PUT/POST /collections/{collection}/points
       - POST /collections/{collection}/points/search

    2) Generic style (fallback):
       - POST /indexes/{index}/upsert
       - POST /indexes/{index}/search

    The client will try pattern (1) first and fall back to (2).

    If your Endee installation uses different endpoints, update this file only.
    """

    def __init__(self, cfg: EndeeConfig):
        self.cfg = cfg

    def _url(self, path: str) -> str:
        return self.cfg.base_url.rstrip("/") + path

    def _request(self, method: str, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make a JSON HTTP request with small retries.

        Why retries?
        - Local services (Endee in Docker) can have brief warm-up periods.
        - Retrying once or twice improves developer experience without masking real issues.
        """
        url = self._url(path)
        last_err: Exception | None = None

        for attempt in range(1, 3):
            try:
                r = requests.request(method, url, json=payload, timeout=self.cfg.timeout_s)
                if r.status_code >= 400:
                    raise EndeeError(f"Endee HTTP {r.status_code} at {path}: {r.text}")

                if not r.text:
                    return {}

                return r.json()
            except (requests.RequestException, json.JSONDecodeError) as e:
                last_err = e

        raise EndeeError(f"Endee request failed after retries at {path}: {last_err}")

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", path, payload)

    def _put(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("PUT", path, payload)

    def upsert_points(self, points: List[Dict[str, Any]]) -> None:
        """Upsert points with schema: {id, vector, payload}."""
        if not points:
            return

        # Try Qdrant-like
        try:
            self._put(
                f"/collections/{self.cfg.index_name}/points",
                {"points": points},
            )
            return
        except EndeeError:
            # Fallback to generic
            self._post(f"/indexes/{self.cfg.index_name}/upsert", {"points": points})

    def search(self, vector: List[float], *, top_k: int = 5) -> List[Dict[str, Any]]:
        """Vector search; returns list of matches with payload and score."""
        qdrant_payload = {"vector": vector, "limit": top_k, "with_payload": True}
        try:
            data = self._post(
                f"/collections/{self.cfg.index_name}/points/search",
                qdrant_payload,
            )
            return data.get("result", data.get("results", []))
        except EndeeError:
            data = self._post(
                f"/indexes/{self.cfg.index_name}/search",
                {"vector": vector, "top_k": top_k, "include_payload": True},
            )
            return data.get("result", data.get("results", data.get("matches", [])))
