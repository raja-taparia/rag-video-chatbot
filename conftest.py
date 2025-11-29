# Pytest configuration and fixtures

import pytest
import json
from pathlib import Path
from typing import List
import tempfile
import shutil

from src.config import Config, OllamaConfig, QdrantConfig, RAGConfig, DataConfig
from src.models import VideoTranscriptFile, TokenData


@pytest.fixture
def temp_data_dir():
    """Create temporary data directory."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_video_data():
    """Create mock video transcript data."""
    return {
        "video_id": "TEST_VIDEO_001",
        "title": "Test Tutorial",
        "pdf_reference": "test_001.pdf",
        "duration_seconds": 120,
        "video_transcripts": [
            {"id": i, "timestamp": i * 0.5, "word": word}
            for i, word in enumerate([
                "Hello", "everyone", "today", "I", "will", "teach",
                "you", "Kubernetes", "setup", "First", "install",
                "Docker", "Then", "enable", "kubeadm", "Finally",
                "join", "the", "cluster", "nodes"
            ])
        ]
    }


@pytest.fixture
def mock_config(temp_data_dir):
    """Create mock configuration."""
    video_dir = temp_data_dir / "videos"
    pdf_dir = temp_data_dir / "pdfs"
    indices_dir = temp_data_dir / "indices"
    metadata_dir = temp_data_dir / "metadata"
    
    for d in [video_dir, pdf_dir, indices_dir, metadata_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    return Config(
        ollama=OllamaConfig(
            base_url="http://localhost:11434",
            embedding_model="nomic-embed-text",
            llm_model="mistral",
        ),
        qdrant=QdrantConfig(
            host="localhost",
            port=6333,
        ),
        postgres=None,
        rag=RAGConfig(
            chunk_size=512,
            chunk_overlap=128,
            video_relevance_threshold=0.7,
            pdf_relevance_threshold=0.6,
            top_k_video=3,
            top_k_pdf=3,
            max_context_length=2048,
        ),
        data=DataConfig(
            video_dir=video_dir,
            pdf_dir=pdf_dir,
            indices_dir=indices_dir,
            metadata_dir=metadata_dir,
        ),
        log_level="DEBUG",
        api_host="0.0.0.0",
        api_port=8000,
        debug=True,
    )


@pytest.fixture
def mock_transcript(mock_video_data):
    """Create mock VideoTranscriptFile."""
    return VideoTranscriptFile(**mock_video_data)
