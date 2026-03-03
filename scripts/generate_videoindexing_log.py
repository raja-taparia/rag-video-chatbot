"""
Run:
    python scripts/generate_videoindexing_log.py
"""

import json
import time
from pathlib import Path
import sys

# Ensure repo root is on path so 'src' package can be imported when running script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.ingestion.chunker import VideoChunker
from src.models import VideoTranscriptFile


def main():
    config = load_config()

    vc = VideoChunker(
        chunk_size=config.rag.chunk_size,
        chunk_overlap=config.rag.chunk_overlap,
        pause_threshold=config.rag.pause_threshold,
        min_chunk_size=config.rag.min_video_chunk_size,
        max_chunk_size=config.rag.max_video_chunk_size,
    )

    data_dir = Path("data/videos")
    if not data_dir.exists():
        print("No data/videos directory found. Exiting.")
        return

    transcripts_files = list(data_dir.glob("*.json"))
    log_start = time.time()
    log_entries = {
        "comment": f"Video indexing log generated at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(log_start))}",
        "videos": {}
    }

    for tf in transcripts_files:
        try:
            raw = json.loads(tf.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"Skipping {tf}: failed to read JSON: {e}")
            continue

        try:
            vt = VideoTranscriptFile(**raw)
        except Exception as e:
            print(f"Skipping {tf}: failed to parse VideoTranscriptFile: {e}")
            continue

        chunks = vc.create_chunks(vt)
        vid_entry = []
        for chunk in chunks:
            # Count tokens in this chunk
            token_count = len([t.id for t in vt.video_transcripts if t.id >= chunk.start_token_id and t.id <= chunk.end_token_id])
            vid_entry.append({
                "chunk_id": chunk.chunk_id,
                "start_time": chunk.start_timestamp,
                "end_time": chunk.end_timestamp,
                "start_token_id": chunk.start_token_id,
                "end_token_id": chunk.end_token_id,
                "token_count": token_count,
                "text": chunk.text,
            })

        log_entries["videos"][vt.video_id] = vid_entry

    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "videoindexing.json"
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(log_entries, indent=2, default=str))

    print(f"Wrote video indexing log to {log_path}")
    print(f"Videos processed: {len(log_entries['videos'])}")


if __name__ == '__main__':
    main()
