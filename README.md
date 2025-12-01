# RAG Video Chatbot

This repository contains a Retrieval-Augmented Generation (RAG) demo focused on video transcripts and PDF documents. It extracts text from videos and PDFs, creates semantic embeddings, stores vectors in Qdrant, and serves an interactive FastAPI web UI and programmatic endpoints to ask questions.

This README documents repository layout, setup steps, key commands used during development, debugging helpers, and recommendations for running and troubleshooting embeddings and search.

---

## Project Structure (high level)

- `api/` - FastAPI web UI and endpoints. Main entry: `api/main.py`.
- `src/` - Core application code:
  - `src/config.py` - typed configuration (env var defaults).
  - `src/indexing/embeddings.py` - embedder implementations (SentenceTransformers primary, Ollama fallback) and factory `get_default_embedder()`.
  - `src/indexing/vector_store.py` - Qdrant integration and compatibility layer.
  - `src/ingestion/` - loaders and chunking (video/pdf ingestion).
  - `src/pipeline/rag_pipeline.py` - orchestration of ingestion, indexing, retrieval, and LLM generation.
  - `src/retrieval/` - retrievers and ranking strategies.
  - `src/models.py` - Pydantic models (uses `llm_model` field to avoid `model_` protected namespace warning).
- `scripts/` - helper scripts added for diagnostics and testing:
  - `scripts/test_embeddings.py` — tests sentence-transformers and default embedder.
  - `scripts/inspect_qdrant.py` — inspects Qdrant client methods and performs a raw `_perform_search`.
  - `scripts/test_index_and_search.py` — indexes a single test point then searches it to validate end-to-end index/search flow.
- `requirements.txt` - Python dependencies.
- `main.py` - CLI entrypoint for indexing and queries.

Logs and caches will be created under `logs/` and `~/.cache/huggingface` (for model downloads).

---

## Key design decisions

- SentenceTransformers: the default local embedder is `all-MiniLM-L6-v2` (384 dims) via `sentence-transformers` for fast, CPU-friendly embeddings for both PDF paragraphs and video transcripts.
- Ollama: kept as an optional fallback for local embedding/LLM calls. The embedder sends `{"model":..., "input": [...]}` payloads to Ollama and handles several response shapes.
- Qdrant compatibility: `src/indexing/vector_store.py` probes several client method names to support multiple `qdrant-client` versions. It also normalizes returned `score`/`distance` into a similarity-like `score` for ranking.
- Pydantic v2: code uses `.model_dump()` for serialization and avoids Pydantic `model_` protected namespace warnings by using `llm_model` in `RAGResponse`.

---

## Environment & Setup

Recommended: create and activate a virtual environment in the repo root.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# If you only need sentence-transformers you can install it explicitly:
pip install sentence-transformers
```

Environment variables (in `src/config.py` defaults):
- `OLLAMA_BASE_URL` (default `http://localhost:11434`)
- `EMBEDDING_MODEL` (default `all-MiniLM-L6-v2`) — used by `get_default_embedder` and config
- `LLM_MODEL` (default `mistral`)
- `QDRANT_HOST` (default `localhost`)
- `QDRANT_PORT` (default `6333`)
- `CHUNK_SIZE`, `CHUNK_OVERLAP`, and other RAG params are read from env as well.

Set these in your shell or in an `.env` file as you prefer.

---

## Running services & commands

Start Qdrant (if you use the local Docker image):

```bash
# Docker (example):
docker run -p 6333:6333 qdrant/qdrant
```

Start the FastAPI app (serves the web UI and API endpoints):

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
# open http://localhost:8000/
```

Index all data (recreate collections with the configured `embedding_dim` and index):

```bash
python main.py --index --full
```

Ask a question from the CLI:

```bash
python main.py --question "How do I install Docker on my system?"
```

Helper scripts (diagnostic):

```bash
# Test embedding generation (sentence-transformers and default embedder)
python scripts/test_embeddings.py

# Inspect Qdrant client methods and a raw search response
python scripts/inspect_qdrant.py

# Index a small test point and perform search
python scripts/test_index_and_search.py
```

Other commands encountered during development:
- `ollama pull nomic-embed-text-v1.5` (attempted to pull Ollama model)
- `curl -i -X POST http://localhost:11434/api/embeddings -H "Content-Type: application/json" -d '{"model":"nomic-embed-text:latest","input":["How to install docker"]}'` — example Ollama request used while debugging.

---

## API Endpoints (via `api/main.py`)

- `GET /` — Serves a simple HTML web UI that posts to `/ask`.
- `POST /ask` — Accepts JSON `{ "question": "..." }` and returns the `RAGResponse` as JSON. The server initializes a `RAGPipeline` at startup (logged) and uses it to answer questions.
- `POST /upload-video` — Accepts video uploads.
- `GET /status/videos` and `GET /status/pdfs` — Return indexing/transcription status information.

Notes:
- If your frontend runs from a different origin (e.g. served on port `8080`), you must enable CORS in `api/main.py` or call the API from the same origin.

---

## Troubleshooting

Common issues and how to diagnose them:

- Embeddings are empty or `None`:
  - Run `python scripts/test_embeddings.py` to verify `sentence-transformers` loads and returns vectors.
  - Ensure the `sentence-transformers` package and a compatible `torch` wheel are installed. On macOS, installing `torch` correctly can require platform-specific wheels.

- Ollama returns empty embeddings (`{"embedding": []}`):
  - Confirm the exact model name using `ollama list` and use that identifier in requests.
  - Check Ollama server logs; try a curl request to `/api/embeddings` to inspect the raw response.

- Qdrant search returns no results:
  - Ensure you re-created collections with the correct `vector_size` (embedding dims). Use `python main.py --index --full` to clear and reindex.
  - Use `python scripts/inspect_qdrant.py` and `python scripts/test_index_and_search.py` to check raw `QueryResponse` shapes and confirm indexing/upsert succeeded.
  - The code normalizes `score` and `distance` returned by Qdrant — check logs for which field was present.

- FastAPI frontend returns 404 on `/ask`:
  - Make sure you restarted the server after recent edits to `api/main.py`. The endpoint `POST /ask` was added and requires the app to be reloaded.
  - If the frontend and API are on different origins, add CORS middleware to the FastAPI app.

Logs: check `logs/app.log` for detailed errors produced by pipeline components.

---

## Development notes

- The codebase uses Pydantic v2; models are serialized with `.model_dump()`.
- `src/indexing/vector_store.py` contains compatibility layers for multiple qdrant-client versions and normalizes returned results to make ranking stable across versions.
- The embedder factory `get_default_embedder()` prefers `SentenceTransformersEmbedder` (local) and falls back to `OllamaEmbedder` if the local package is unavailable.

If you plan to change embedding models, remember to:
1. Update the `EMBEDDING_MODEL` env var or `src/config.py` default.
2. Recreate Qdrant collections with matching vector size (run full reindex).

---

## Next steps and optional enhancements

- Add a toggle (`EMBEDDING_BACKEND=ollama|local`) to explicitly choose embedding backend at runtime.
- Add CI tests that validate embedding dimension and that a small sample query returns results after indexing.
- Improve score mapping (if you use a different `distance` metric) to avoid incorrect normalization.
- Add CORS middleware in `api/main.py` if you want to host a separate frontend on another port.

---

If you'd like, I can run a full reindex now and share the logs, or add CORS support and a small health endpoint that reports pipeline / Qdrant / Ollama readiness.

---

Maintainer: repository owner
