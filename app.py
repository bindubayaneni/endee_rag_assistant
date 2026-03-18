from __future__ import annotations



import time
import uuid

import requests
import streamlit as st


def _safe_get_json(url: str, *, timeout_s: int = 10) -> dict | None:
    try:
        resp = requests.get(url, timeout=timeout_s)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def _safe_post_json(url: str, payload: dict, *, timeout_s: int = 90) -> dict:
    resp = requests.post(url, json=payload, timeout=timeout_s)
    resp.raise_for_status()
    return resp.json()


def _safe_post_file(url: str, file_bytes: bytes, filename: str, *, timeout_s: int = 180) -> dict:
    files = {"file": (filename, file_bytes)}
    resp = requests.post(url, files=files, timeout=timeout_s)
    resp.raise_for_status()
    return resp.json()


def _backend_url() -> str:
    url = st.session_state.get("backend_url")
    if url:
        return url
    default = "http://127.0.0.1:8001"
    try:
        return st.secrets.get("BACKEND_URL", default)  # type: ignore[attr-defined]
    except Exception:
        return default


def _ensure_session_state() -> None:
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"sess_{uuid.uuid4().hex}"
    if "chat" not in st.session_state:
        st.session_state.chat = []
    if "active_document_id" not in st.session_state:
        st.session_state.active_document_id = None
    if "active_doc_name" not in st.session_state:
        st.session_state.active_doc_name = None
    if "_last_uploaded_name" not in st.session_state:
        st.session_state._last_uploaded_name = None


def _render_runtime_box(backend: str) -> None:
    health = _safe_get_json(f"{backend}/health", timeout_s=5)
    if not health:
        st.sidebar.error("Backend unreachable")
        return

    llm = health.get("llm_provider", "unknown")
    store = health.get("vector_store", "unknown")
    st.sidebar.markdown("### Runtime")
    st.sidebar.code(f"llm_provider: {llm}\nvector_store: {store}", language="text")


_ensure_session_state()

st.sidebar.header("Settings")
st.sidebar.text_input("Backend URL", value=_backend_url(), key="backend_url")
backend = st.session_state.backend_url.rstrip("/")

_render_runtime_box(backend)

st.sidebar.markdown("---")
st.sidebar.markdown("### Active document")
if st.session_state.active_document_id:
    st.sidebar.success(f"{st.session_state.active_doc_name}\n\nid={st.session_state.active_document_id}")
else:
    st.sidebar.warning("None (upload a document)")

st.title("Hybrid RAG Knowledge Assistant (Endee Vector DB)")

# Upload section
st.subheader("Upload a document")
uploaded = st.file_uploader("Choose a text file", type=["txt"], accept_multiple_files=False)

col_ingest, col_reset = st.columns([1, 1])

with col_reset:
    if st.button("Reset chat", type="secondary"):
        st.session_state.chat = []
        st.rerun()

with col_ingest:
    if uploaded is not None:
        # If a new file is selected, reset active document context immediately.
        if st.session_state._last_uploaded_name != uploaded.name:
            st.session_state._last_uploaded_name = uploaded.name
            st.session_state.active_document_id = None
            st.session_state.active_doc_name = None
            st.session_state.chat = []

        if st.button("Ingest document", type="primary"):
            try:
                with st.spinner("Ingesting..."):
                    out = _safe_post_file(
                        f"{backend}/ingest_file",
                        uploaded.getvalue(),
                        uploaded.name,
                        timeout_s=300,
                    )

                st.session_state.active_document_id = out.get("document_id") or out.get("stats", {}).get("document_id")
                st.session_state.active_doc_name = (
                    out.get("doc_name") or out.get("stats", {}).get("doc_name") or uploaded.name
                )

                if not st.session_state.active_document_id:
                    st.warning("Ingest succeeded, but document_id was missing from response.")
                st.success(f"Ingested {out.get('chunks', out.get('stats', {}).get('chunks', 0))} chunks")
            except Exception as e:
                st.error(f"Backend request failed ({backend}/ingest_file): {e}")

st.markdown("---")

# Chat section
st.subheader("Chat")

for role, content in st.session_state.chat:
    with st.chat_message(role):
        st.markdown(content)

question = st.chat_input("Ask a question about the active document…")
if question:
    st.session_state.chat.append(("user", question))
    with st.chat_message("user"):
        st.markdown(question)

    if not st.session_state.active_document_id:
        msg = "No active document. Upload + ingest first."
        with st.chat_message("assistant"):
            st.error(msg)
        st.session_state.chat.append(("assistant", msg))
        st.stop()

    try:
        with st.chat_message("assistant"):
            with st.spinner("Searching…"):
                time.sleep(0.05)
                out = _safe_post_json(
                    f"{backend}/ask",
                    {
                        "question": question,
                        "session_id": st.session_state.session_id,
                        "document_id": st.session_state.active_document_id,
                    },
                    timeout_s=180,
                )

            answer = out.get("answer", "")
            st.markdown(answer)

            sources = out.get("sources") or []
            if sources:
                st.markdown("#### Sources used")
                for i, s in enumerate(sources, start=1):
                    score = s.get("score")
                    score_str = f"{float(score):.4f}" if score is not None else "n/a"
                    st.caption(
                        f"[Source {i}] {s.get('doc_name')} (chunk {s.get('chunk_index')}) — score={score_str}"
                    )
                    st.code(s.get("snippet", ""), language="text")

        st.session_state.chat.append(("assistant", answer))
    except Exception as e:
        st.error(f"Backend request failed ({backend}/ask): {e}")