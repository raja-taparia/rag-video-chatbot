"""FastAPI server for RAG chatbot with REST endpoints."""

import logging
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.logger import setup_logging
from src.pipeline.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)


# Pydantic models for API
class QueryRequest(BaseModel):
    """Request model for /ask endpoint"""
    question: str
    top_k_video: Optional[int] = None
    top_k_pdf: Optional[int] = None
    video_threshold: Optional[float] = None
    pdf_threshold: Optional[float] = None


class IndexRequest(BaseModel):
    """Request model for /index endpoint"""
    full: bool = False


# Global pipeline instance
pipeline: Optional[RAGPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown context manager."""
    global pipeline
    
    # Startup
    logger.info("Starting FastAPI application")
    config = load_config()
    pipeline = RAGPipeline(config)
    
    yield
    
    # Shutdown
    logger.info("Shutting down FastAPI application")


app = FastAPI(
    title="RAG Chatbot API",
    description="Retrieval-Augmented Generation chatbot for video transcripts and PDFs",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/ask")
async def ask_question(request: QueryRequest) -> Dict[str, Any]:
    """
    Ask a question to the RAG chatbot.
    
    The system will first search video transcripts, then fall back to PDFs.
    """
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        response = pipeline.query(request.question)
        return response.dict(by_alias=True)
    
    except Exception as e:
        logger.error(f"Error processing question: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/index")
async def trigger_indexing(request: IndexRequest) -> Dict[str, Any]:
    """
    Trigger data indexing.
    
    Set full=true to clear and rebuild all indices from scratch.
    """
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        stats = pipeline.index_data(full_reindex=request.full)
        return {
            "status": "success",
            "stats": stats
        }
    
    except Exception as e:
        logger.error(f"Error during indexing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status")
async def get_status() -> Dict[str, Any]:
    """Get system status information."""
    if not pipeline:
        return {"status": "not_initialized"}
    
    return {
        "status": "ready",
        "embedding_model": pipeline.config.ollama.embedding_model,
        "llm_model": pipeline.config.ollama.llm_model,
        "video_dir": str(pipeline.config.data.video_dir),
        "pdf_dir": str(pipeline.config.data.pdf_dir),
    }


@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint with documentation link."""
    return {
        "message": "RAG Chatbot API",
        "docs": "/docs",
        "endpoints": {
            "POST /ask": "Ask a question",
            "POST /index": "Trigger data indexing",
            "GET /status": "Get system status",
            "GET /health": "Health check"
        }
    }


def main():
    """Run FastAPI server."""
    config = load_config()
    setup_logging(log_level=config.log_level)
    
    logger.info(f"Starting FastAPI server on {config.api_host}:{config.api_port}")
    
    uvicorn.run(
        app,
        host=config.api_host,
        port=config.api_port,
        log_level=config.log_level.lower(),
    )


if __name__ == "__main__":
    main()
