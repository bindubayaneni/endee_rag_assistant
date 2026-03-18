from __future__ import annotations

"""Vector store interface.

Recruiter note:
- Storage behind an interface supports swapping FAISS/Endee without touching core logic.
- We also keep filtering in the contract to enable document-level isolation.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol


@dataclass(frozen=True)
class VectorMatch:
    id: str
    score: float
    payload: Dict[str, Any]


class VectorStore(Protocol):
    def upsert_points(self, points: List[Dict[str, Any]]) -> None:
        """Upsert points with schema: {id: str, vector: List[float], payload: dict}."""

    def search(
        self,
        *,
        vector: List[float],
        top_k: int,
        filter: Optional[Dict[str, Any]] = None,
        candidate_pool: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return matches as dicts: {id, score, payload}.
        - filter: payload-level equality filters, e.g. {"document_id": "..."}.
        - candidate_pool: retrieve more candidates internally then filter down.
        """