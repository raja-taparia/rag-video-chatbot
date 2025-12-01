#!/usr/bin/env python3
"""Create a small test point in Qdrant and run search_video to verify indexing+search flow.

This script will:
- Load config
- Create a tiny video chunk with a known vector (e.g., [1, 0, 0, ...])
- Index it into the `video_transcripts` collection using `index_video_chunks`
- Run `search_video` with the same vector and print the results

Note: This will create/write to your Qdrant instance. Qdrant must be running.
"""

from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import traceback
import json

from src.config import load_config
from src.indexing.vector_store import QdrantVectorStore


def main():
    try:
        cfg = load_config()
        print(f"Using Qdrant at {cfg.qdrant.host}:{cfg.qdrant.port}, embedding_dim={cfg.ollama.embedding_dim}")

        vs = QdrantVectorStore(host=cfg.qdrant.host, port=cfg.qdrant.port, embedding_dim=cfg.ollama.embedding_dim)

        # Build test point with a simple vector
        dim = cfg.ollama.embedding_dim
        vec = [0.0] * dim
        # make a distinctive vector
        if dim > 0:
            vec[0] = 1.0

        chunk = {
            'chunk_id': 'test_chunk_1',
            'video_id': 'test_video_1',
            'start_token_id': 0,
            'end_token_id': 10,
            'start_timestamp': 0.0,
            'end_timestamp': 5.0,
            'text': 'This is a small test chunk about docker installation',
            'embedding': vec,
        }

        print('Indexing a single test chunk...')
        count = vs.index_video_chunks([chunk])
        print(f'Indexed points count (returned): {count}')

        print('Running raw _perform_search with the same vector (debug)...')
        try:
            raw = vs._perform_search(vs.VIDEO_COLLECTION, query_vector=vec, top_k=5, score_threshold=0.0)
            print('Raw _perform_search repr:')
            print(repr(raw)[:4000])
        except Exception as e:
            print('Raw _perform_search raised:')
            import traceback
            traceback.print_exc()

        print('\nRunning search_video (normalized) with the same vector...')
        results = vs.search_video(query_embedding=vec, top_k=5, threshold=0.0)
        print('Search results:')
        print(json.dumps(results, indent=2, default=str))

    except Exception:
        traceback.print_exc()
        sys.exit(2)


if __name__ == '__main__':
    main()
