# Endee RAG Assistant — Safe, Isolated, Fast

A production-minded Retrieval-Augmented Generation (RAG) demo that supports **strict document isolation**, **fast retrieval**, and **safe citations** (no raw text leakage). Runs locally with a FAISS fallback store and an Ollama/OpenAI/mock LLM provider.

> Built to showcase end-to-end engineering: ingestion → embedding → filtered retrieval → safe context → LLM answer → UI.

---

## Why this project

Most RAG demos “work”, but fail on the things that matter in real systems:

- **Document mixing**: new uploads contaminate results with old data  
- **Sensitive data leakage**: UIs show raw retrieved text verbatim  
- **Slow responses**: repeated embeddings + huge `top_k` + inefficient retrieval  

This repo addresses those issues directly.

---

## Key features

### Safety-first citations (no raw text leakage)
- Sources shown as **masked, short snippets** only (emails → `[EMAIL]`, numbers → `[NUMBER]`)
- LLM receives **sanitized snippets**, not full raw context  
- FAISS persistence stores **safe payload only** (no raw chunk text at rest)

### Strict document-level isolation
- Every ingestion produces a unique **`document_id`**
- Retrieval uses a **document_id filter** so answers only reference the active document
- Streamlit session tracks the active document id and resets chat on new upload

### Faster retrieval pipeline
- **Embedding cache** (in-memory) avoids recomputing embeddings for identical queries/chunks
- **Batched chunk embedding** during ingestion
- `top_k` capped to an optimal **3–5**
- Minimal vector store calls with candidate pooling + filtering

---

## Architecture (clean pipeline)

```text
Upload → Chunk → Batch Embed (cached) → Upsert (document_id payload)
Ask → Query Embed (cached) → Filtered Search (document_id) → Safe Context → LLM → Answer + Safe Sources
```

---

## Quickstart (local)

### 1) Create & activate venv
```bash
cd endee-rag-assistant
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Start backend (FastAPI)
```bash
export VECTOR_STORE=faiss
export LLM_PROVIDER=ollama           # or: mock, openai
export OLLAMA_BASE_URL=http://localhost:11434
export OLLAMA_MODEL=phi3:mini
unset OPENAI_API_KEY

python -m uvicorn backend.api:app --host 127.0.0.1 --port 8001 --reload
```

### 3) Start frontend (Streamlit)
```bash
source .venv/bin/activate
streamlit run frontend/app.py
```

---

## Runtime verification

### Health check (shows active provider/store)
```bash
curl -s http://127.0.0.1:8001/health
```

Expected includes:
- `llm_provider`: `ollama | openai | mock`
- `vector_store`: `faiss | endee | auto`

---

## Configuration

| Env var | Example | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | Select LLM provider (`ollama`, `openai`, `mock`) |
| `VECTOR_STORE` | `faiss` | Select vector store (`faiss`, `endee`, `auto`) |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `phi3:mini` | Ollama model |
| `OPENAI_API_KEY` | `...` | OpenAI key (if using `openai`) |

---

## API (minimal)

### `POST /ingest_file`
Uploads a file and returns a `document_id` for isolated retrieval.
- Response includes: `document_id`, `doc_name`, `chunks`

### `POST /ask`
Request:
```json
{
  "session_id": "sess_...",
  "question": "…",
  "document_id": "doc_..."
}
```

Response:
- `answer`
- `sources`: safe masked snippets + metadata (no raw text)


## Repo layout


backend/
  api.py            # FastAPI routes + /health
  ingest.py         # chunking + batch embeddings + doc_id payload
  query.py          # filtered retrieval + safe context + LLM
  embeddings.py     # caching + batching
  stores/faiss_store.py  # local store with safe persistence + filtering
frontend/
  app.py            # Streamlit UI with session-based active document




## Security & privacy notes

- This demo **prevents raw text exposure in the UI** by design.
- For FAISS local persistence, only masked snippets are persisted in `payload.json`.
- For real production deployments: add encryption-at-rest, auth, audit logging, and a hardened secret manager.

---

## Roadmap
- PDF ingestion (text extraction) + better chunking heuristics
- Proper ID-mapped FAISS index / deletions by document_id
- Observability (timings, CPU/memory metrics, tracing)
- Evaluation harness (RAGAS / golden QA sets)

<img width="1470" height="517" alt="Screenshot 2026-03-18 at 3 14 04 PM" src="https://github.com/user-attachments/assets/4a9d87e5-818b-4a10-9044-5d23cb173270" />


## License
MIT (or update
