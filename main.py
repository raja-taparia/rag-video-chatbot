#!/usr/bin/env python3
"""
Main entry point for RAG chatbot system.

Usage:
    # Index data
    python main.py --index
    
    # Ask a question
    python main.py --question "How do I setup Kubernetes?"
    
    # Start API server
    python api/main.py
"""

import argparse
import logging
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.config import load_config
from src.logger import setup_logging
from src.pipeline.rag_pipeline import RAGPipeline

from src.ingestion.video_transcriber import VideoTranscriber
from src.ingestion.pdf_finder import PDFFinder

def main():
    """Main entry point."""


    # Parse arguments

    parser = argparse.ArgumentParser(description="RAG Chatbot for Video Transcripts & PDFs")
    parser.add_argument("--index", action="store_true", help="Index all data")
    parser.add_argument("--full", action="store_true", help="Full reindex (clear existing)")
    parser.add_argument("--question", type=str, help="Ask a question")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for logging during indexing")
    parser.add_argument("--transcribe-youtube", action="store_true", help="Transcribe YouTube videos")
    parser.add_argument("--transcribe-local", action="store_true", help="Transcribe local videos")
    parser.add_argument("--download-pdfs", action="store_true", help="Download PDFs based on ideas")

    args = parser.parse_args()
    
    # Load configuration
    config = load_config()
    setup_logging(log_level=config.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("RAG Chatbot System Starting")
    logger.info(f"Config: video_dir={config.data.video_dir}, pdf_dir={config.data.pdf_dir}")
    
    try:
        # Initialize pipeline
        pipeline = RAGPipeline(config)
        
        if args.index:
            # Index mode
            logger.info("Indexing mode activated")
            stats = pipeline.index_data(full_reindex=args.full)
            
            print("\n" + "="*60)
            print("INDEXING COMPLETE")
            print("="*60)
            print(json.dumps(stats, indent=2))
            print("="*60 + "\n")
            
        elif args.question:
            # Query mode
            logger.info(f"Query mode: {args.question}")
            response = pipeline.query(args.question)
            
            print("\n" + "="*60)
            print("QUERY RESPONSE")
            print("="*60)
            # Use Pydantic v2 `model_dump()` instead of deprecated `dict()`
            print(json.dumps(response.model_dump(), indent=2, default=str))
            print("="*60 + "\n")

        elif args.transcribe_youtube:
            transcriber = VideoTranscriber(
                videos_input_dir=Path("data/videos/videos_input"),
                output_dir=Path("data/videos")
            )
            transcriber.process_video_links_file(Path("video_links.txt"))

        elif args.transcribe_local:
            transcriber = VideoTranscriber(
                videos_input_dir=Path("data/videos/videos_input"),
                output_dir=Path("data/videos")
            )
            transcriber.process_local_videos_folder()

        elif args.download_pdfs:
            pdf_finder = PDFFinder(
                output_dir=Path("data/pdfs"),
                pdf_finder_file=Path("pdf_finder.txt")
            )
            pdf_finder.process_pdf_from_ideas_file()

        else:
            print("Usage:")
            print("  python main.py --index           # Index all data")
            print("  python main.py --question \"...\"  # Ask a question")
            print("\nRun 'python main.py --help' for full options")
    
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
