#!/usr/bin/env python3
"""Quick script to test embedding generation in this repo.

Run from the repo root (with your venv activated):

  python scripts/test_embeddings.py

It will:
- Load `sentence-transformers` model `all-MiniLM-L6-v2` via the project's embedder
- Print model name, embedding dimension, and a small sample of embedding values
- Also test `get_default_embedder` to ensure the factory path works

If this fails, the script prints a traceback to help debugging.
"""

import sys
import traceback
import json
from pathlib import Path

# Ensure repo root is on sys.path so `from src.*` imports work when running
# this script directly (e.g. `python scripts/test_embeddings.py`). The script
# lives in `scripts/` under the repo root, so the repo root is the parent of
# this file's parent directory.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.indexing.embeddings import SentenceTransformersEmbedder, get_default_embedder


def test_sentence_transformers():
    try:
        cfg = load_config()
        model_name = cfg.ollama.embedding_model or "all-MiniLM-L6-v2"
        print(f"[test] Config embedding_model: {model_name}")

        print("[test] Instantiating SentenceTransformersEmbedder...")
        emb = SentenceTransformersEmbedder(model_name)
        print(f"[test] Loaded model: {emb.model_name}")
        print(f"[test] Reported embedding_dim: {emb.embedding_dim}")

        sample = "How to install docker"
        v = emb.embed_text(sample)
        print(f"[test] Single embedding length: {None if v is None else len(v)}")
        if v:
            print("[test] First 8 values:", v[:8])

        texts = [sample, "Install Docker Desktop on macOS"]
        batch = emb.embed_batch(texts)
        print(f"[test] Batch lengths: {[None if x is None else len(x) for x in batch]}")

    except Exception:
        print("[error] Exception in test_sentence_transformers:")
        traceback.print_exc()
        sys.exit(2)


def test_default_embedder():
    try:
        cfg = load_config()
        print("[test] Instantiating default embedder via get_default_embedder()...")
        e = get_default_embedder(cfg.ollama)
        print("[test] Default embedder type:", type(e).__name__)
        s = "How to install docker"
        v = e.embed_text(s)
        print(f"[test] Default embedder produced embedding length: {None if v is None else len(v)}")
    except Exception:
        print("[error] Exception in test_default_embedder:")
        traceback.print_exc()
        sys.exit(3)


if __name__ == "__main__":
    print(json.dumps({"action": "start_embedding_tests"}))
    test_sentence_transformers()
    test_default_embedder()
    print(json.dumps({"action": "done"}))
    print("All tests completed")
