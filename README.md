# RAG Video Chatbot

A **Retrieval‑Augmented Generation (RAG)** system that lets you ask natural‑language questions about
video transcripts and PDF documents.  The pipeline ingests video/audio, transcribes it, crawls and
downloads PDFs, chunks text into semantically meaningful pieces, computes embeddings, stores them
in [Qdrant](https://qdrant.tech/) and serves answers via a FastAPI web UI or programmatic API.

It was built as an experiment in combining local LLMs and vector search with real‑world media
sources.  Although the code started as a simple demo, it now includes:

- YouTube and local video transcription (Whisper + yt‑dlp/ffmpeg).
- Pause‑aware chunking that tries to respect sentence boundaries.
- Paragraph extraction + PDF search/download based on “idea” keywords.
- Configurable embedding backends (Sentence‑Transformers default, Ollama fallback).
- Compatibility layer for multiple qdrant‑client versions.
- Pluggable retrieval and ranking strategies (cosine, RRF, ...).
- Simple FastAPI frontend + endpoints for uploads and status.
- CLI helpers for indexing, querying, transcription and PDF discovery.
- Comprehensive unit tests and diagnostic scripts.

---

## Features ✅

- **Ingestion**
  - Transcribe YouTube links or local videos (`VideoTranscriber`).
  - Load existing JSON transcripts (`VideoTranscriptLoader`).
  - Download/search PDFs from the web (`PDFFinder`).
  - Extract text with page/paragraph granularity using `PDFLoader`.
  - Token‑to‑timestamp mapping and validation utilities.
- **Chunking**
  - Pause‑based video chunker that merges/splits around silence.
  - Alternative fixed‑size and generic text chunkers.
  - PDF paragraph segments for indexing.
- **Indexing**
  - Uses Qdrant vector store with separate collections for videos and PDFs.
  - Embedding generation via Sentence‑Transformers (384‑dim by default) or
    Ollama models.
  - Indexing stats and persistent logging (`logs/videoindexing.json`).
- **Retrieval & Generation**
  - `VideoRetriever` and `PDFRetriever` with configurable thresholds/`top_k`.
  - Ranking strategies (cosine similarity, reciprocal rank fusion).
  - `AnswerGenerator` that calls local LLMs via Ollama to formulate responses.
- **API & UI**
  - FastAPI app with minimal HTML/JS frontend.
  - Endpoints: `/ask`, `/upload-video`, `/status/videos`, `/status/pdfs`.
  - CORS‑ready and static file support.
- **CLI**
  - `main.py` supports indexing, questions, transcription and PDF downloading.
  - Helpers for running diagnostics from scripts in `scripts/`.

---

## Repository structure

```
.
├── api/                       FastAPI app and web UI
├── src/
│   ├── config.py              environment‑driven configuration
│   ├── logger.py              logging setup
│   ├── models.py              Pydantic data models
│   ├── ingestion/             data loaders / chunkers / transcribers
│   ├── indexing/              embeddings & Qdrant vector store
│   ├── pipeline/              RAGPipeline orchestration
│   ├── retrieval/             retrievers & ranking strategies
│   └── static/                CSS/JS assets for frontend
├── scripts/                   diagnostic helpers
├── tests/                     pytest unit/integration tests
├── data/                      default data root (created automatically)
│   ├── pdfs/
│   ├── videos/
│   │   └── videos_input/      place local files for transcription
│   └── indices/               (unused/currently empty)
├── logs/                      runtime logs (e.g. videoindexing.json)
├── main.py                    CLI entry point
├── README.md
├── requirements.txt
└── ...other config/files...
```

---

## Quick start ☕️

```bash
# create & activate Python venv (macOS/Linux)
python -m venv venv
source venv/bin/activate

# install everything
pip install -r requirements.txt

# optional: install sentence-transformers separately if you only need it
# pip install sentence-transformers
```

Ensure Qdrant is running (local Docker example):

```bash
docker run -p 6334:6333 qdrant/qdrant
```

Environment variables are read by `src/config.py`.  Defaults are tuned for local development.

| Variable | Default | Purpose |
|----------|---------|---------|
| OLLAMA_BASE_URL | http://localhost:11434 | Ollama server URL |
| EMBEDDING_MODEL | all-MiniLM-L6-v2 | embedding model name |
| LLM_MODEL | mistral | LLM used by answer generator |
| QDRANT_HOST | localhost | Qdrant hostname |
| QDRANT_PORT | 6334 | Qdrant port |
| QDRANT_API_KEY | – | optional API key |
| CHUNK_SIZE | 256 | tokens per chunk (base) |
| CHUNK_OVERLAP | 64 | overlap between video chunks |
| VIDEO_RELEVANCE_THRESHOLD | 0.65 | min score to keep video result |
| PDF_RELEVANCE_THRESHOLD | 0.5 | min score to keep PDF result |
| TOP_K_VIDEO | 3 | number of video chunks to retrieve |
| TOP_K_PDF | 3 | number of PDF paragraphs to retrieve |
| MAX_CONTEXT_LENGTH | 2048 | max tokens for LLM prompt |
| PAUSE_THRESHOLD | 1.5 | seconds of silence indicating sentence break |
| MIN_VIDEO_CHUNK_SIZE | 8 | minimum tokens per video chunk |
| MAX_VIDEO_CHUNK_SIZE | 32 | maximum tokens per video chunk |
| DATA_DIR | data | root directory for videos/pdfs/indices/metadata |
| API_HOST / API_PORT | 0.0.0.0 / 8000 | FastAPI bind address/port |
| LOG_LEVEL | INFO | python logging level |
| DEBUG | False | enable debug mode |
| DATABASE_URL | – | if set, enables PostgresConfig parsing |

Directories under `DATA_DIR` are created automatically when the config is loaded.

---

## CLI usage

```bash
# index everything (video transcripts + PDFs)
python main.py --index

# clear and rebuild index
python main.py --index --full

# ask query from command line
python main.py --question "What is Docker?"

# transcribe YouTube links listed in video_links.txt
python main.py --transcribe-youtube

# transcribe local videos placed in data/videos/videos_input/
python main.py --transcribe-local

# search/download PDFs based on ideas file
python main.py --download-pdfs
```

The CLI prints human‑readable statistics and returns a JSON response when answering
questions.

---

## API endpoints

Start the server with:

```bash
uvicorn api.main:app --reload --host $API_HOST --port $API_PORT
```

Open `http://localhost:8000/` in a browser to use the built‑in UI.

| Path             | Method | Description |
|------------------|--------|-------------|
| `/`              | GET    | HTML web interface (simple JS) |
| `/ask`           | POST   | `{ "question": "..." }` → `RAGResponse` JSON |
| `/upload-video`  | POST   | Upload binary video files (saves to `data/videos/videos_input`) |
| `/status/videos` | GET    | Returns available transcript filenames |
| `/status/pdfs`   | GET    | Returns available PDF filenames |

> If calling from another origin, add CORS middleware in `api/main.py`.

---

## Ingestion details

### Video transcripts

- JSON files are expected under `data/videos/` and must conform to
  `src.models.VideoTranscriptFile` (list of tokens with `id`, `timestamp`,
  `word` fields).

- `VideoTranscriber` can fetch and transcribe videos:
  - YouTube: requires `yt-dlp` and `openai-whisper`.
  - Local files: uses `ffmpeg` + Whisper.

- `VideoChunker` splits tokens using pauses, minimum/maximum sizes, and
  overlap rules.  There are also fallback chunkers (`chunker-video_fixed_chunk_size.py`,
  `TextChunker`) useful for experiment or unit tests.

- A `logs/videoindexing.json` is created during indexing to record chunk
  boundaries and token counts.

### PDFs

- `PDFFinder` can crawl DuckDuckGo for `filetype:pdf` results, download
  them, perform rudimentary normalization, and store them under `data/pdfs/`.

- `PDFLoader` extracts page text with `PyPDF2` and segments into paragraphs.
  Paragraphs shorter than ~20 characters are discarded.

- `PDFChunker` (simple) returns paragraph‑level segments for indexing.

---

## Embeddings & vector store

- Default embedder is `SentenceTransformersEmbedder` (model configurable via
  `EMBEDDING_MODEL`).  If the library cannot be imported or you prefer
  Ollama, set `SENTENCE_TRANSFORMERS_FALLBACK_ONLY=1` and provide a valid
  `OllamaConfig` (see above).

- `QdrantVectorStore` manages two collections (`video_transcripts`, `pdf_documents`)
  with cosine distance.  It normalizes search scores/ distances to a 0‑1
  similarity scale and handles differences between qdrant‑client versions.

---

## Retrieval & ranking

- Pipelines attempt video retrieval first; if no satisfactory answer is
  obtained they fall back to PDFs.
- Returned results include metadata such as timestamps, filenames, and
  similarity `score`.
- Ranking strategies live in `src/retrieval/ranking.py`; you can inject
  custom strategies into `VideoRetriever`/`PDFRetriever`.

---

## Diagnostics & testing

- Helper scripts:
  - `scripts/test_embeddings.py`
  - `scripts/inspect_qdrant.py`
  - `scripts/test_index_and_search.py`

- Run the full test suite with `pytest -q`:

```bash
pytest -q
```

The `tests/` directory contains unit tests for chunking, embedding generation,
PDF extraction, token mapping and integration flows.

---

## Troubleshooting

- **Empty or missing embeddings**: run `scripts/test_embeddings.py`; ensure
  `sentence-transformers` and compatible `torch` wheel are installed.

- **Ollama issues**: verify the model name (`ollama list`) and test via curl
  to `/api/embeddings`.  See logs for raw responses.

- **Qdrant returns no results**: reindex with `python main.py --index --full`
  and examine `scripts/inspect_qdrant.py` output.  Ensure vector size matches
  `EMBEDDING_MODEL` dimension.

- **Frontend 404 on `/ask`**: restart Uvicorn after editing `api/main.py` or add
  CORS if using external frontend.

- **Video transcription failures**: confirm `yt-dlp`, `ffmpeg`, and
  `openai-whisper` are installed in your PATH and accessible.

- **PDF downloads fail**: DuckDuckGo may block scraping; modify `PDFFinder`
  or supply your own URLs.

Logs are written to `logs/app.log` by default (see `src/logger.py`).

---

## Development notes & further ideas

- Models, chunking rules and thresholds are all environment‑tunable.
- There’s optional Postgres support for metadata tracking if you supply a
  `DATABASE_URL`.
- Add a runtime toggle between embedding backends (`EMBEDDING_BACKEND`).
- Add CI to verify dimension consistency & basic query latency.
- Improve RRF/ensemble ranking or write a custom `AnswerGenerator` for
  multilingual output.
- Replace `Whisper` with any preferred transcription service (e.g. OpenAI
  or local wav2vec models).

