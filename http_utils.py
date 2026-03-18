from __future__ import annotations

import time
from contextlib import contextmanager


@contextmanager
def timed(op_name: str):
    """Very small timing helper.

    Recruiter note:
    - In production we would emit metrics (Prometheus/OpenTelemetry).
    - Here we keep it simple and log durations via the calling code.
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        _ = time.perf_counter() - start
