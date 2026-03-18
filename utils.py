from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import tiktoken


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    metadata: dict


def env(name: str, default: str | None = None) -> str | None:
    """Read environment variable with optional default."""
    value = os.getenv(name)
    if value is None:
        return default
    return value


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse too many newlines but keep paragraph boundaries
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def simple_token_count(text: str, model: str = "gpt-4o-mini") -> int:
    """Token estimate using tiktoken.

    Note: Even if you use a different LLM, this is good enough for chunk sizing.
    """
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))


def chunk_text(
    text: str,
    *,
    chunk_size_tokens: int = 500,
    overlap_tokens: int = 100,
    model_for_count: str = "gpt-4o-mini",
) -> List[str]:
    """Token-aware chunking with overlap.

    Strategy:
    - Split on paragraphs first to keep coherence
    - Then pack paragraphs into ~chunk_size_tokens
    - Apply overlap by sliding window on the packed text
    """

    text = normalize_text(text)
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    # Pack paragraphs into coarse chunks
    packed: List[str] = []
    current: List[str] = []
    cur_tokens = 0

    for p in paragraphs:
        p_tokens = simple_token_count(p, model_for_count)
        if current and cur_tokens + p_tokens > chunk_size_tokens:
            packed.append("\n\n".join(current))
            current = [p]
            cur_tokens = p_tokens
        else:
            current.append(p)
            cur_tokens += p_tokens

    if current:
        packed.append("\n\n".join(current))

    # Apply overlap by splitting packed chunks into token windows
    enc = tiktoken.encoding_for_model(model_for_count)
    results: List[str] = []

    for block in packed:
        tokens = enc.encode(block)
        if len(tokens) <= chunk_size_tokens:
            results.append(block)
            continue

        start = 0
        while start < len(tokens):
            end = min(start + chunk_size_tokens, len(tokens))
            window = enc.decode(tokens[start:end]).strip()
            if window:
                results.append(window)
            if end == len(tokens):
                break
            start = max(0, end - overlap_tokens)

    return results


def dedupe_keep_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def safe_filename(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9._-]+", "-", name)
    name = re.sub(r"-+", "-", name)
    return name.strip("-") or "document"
