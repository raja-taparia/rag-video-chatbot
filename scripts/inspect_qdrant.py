#!/usr/bin/env python3
"""Inspect Qdrant client's available search methods and perform a raw search.

This script will:
- Load the repo config
- Instantiate `QdrantVectorStore`
- Print available methods on the underlying client objects
- Attempt to call the private `_perform_search` with a random vector and print raw results

Note: Qdrant must be running at the configured host/port for this to succeed.
"""

from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import traceback
import json
import random

from src.config import load_config
from src.indexing.vector_store import QdrantVectorStore


def main():
    try:
        cfg = load_config()
        print(f"Config Qdrant host={cfg.qdrant.host} port={cfg.qdrant.port}")

        vs = QdrantVectorStore(host=cfg.qdrant.host, port=cfg.qdrant.port, embedding_dim=cfg.ollama.embedding_dim)
        client = vs.client

        # Inspect methods on client and nested attributes
        def list_methods(obj):
            return sorted([m for m in dir(obj) if not m.startswith('_') and callable(getattr(obj, m))])

        print("\nTop-level QdrantClient methods:")
        print(list_methods(client)[:200])

        if hasattr(client, '_client'):
            print("\n_internal _client methods:")
            print(list_methods(getattr(client, '_client'))[:200])

        if hasattr(client, 'http'):
            print("\nhttp attribute methods:")
            print(list_methods(getattr(client, 'http'))[:200])

        # Build a dummy query vector (zeros) and call the private _perform_search
        print("\nPerforming a raw _perform_search with zero vector (may return empty or error)...")
        qvec = [0.0] * cfg.ollama.embedding_dim
        try:
            raw = vs._perform_search(vs.VIDEO_COLLECTION, query_vector=qvec, top_k=3, score_threshold=0.0)
            print("Raw results (repr):")
            print(repr(raw)[:2000])
            try:
                print(json.dumps([str(r) for r in raw], indent=2)[:4000])
            except Exception:
                print("Could not JSON-serialize raw results; printed repr above")
        except Exception as e:
            print("_perform_search raised an exception:")
            traceback.print_exc()

    except Exception:
        traceback.print_exc()
        sys.exit(2)


if __name__ == '__main__':
    main()
