from __future__ import annotations

"""Logging setup.

Recruiter note:
- Production services should have structured, consistent logs.
- This module centralizes log formatting and level.
"""

import logging


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
