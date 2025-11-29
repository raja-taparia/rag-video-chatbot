# Configuration management for RAG chatbot
# Loads environment variables and provides typed configuration

import os
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class OllamaConfig:
    """Configuration for Ollama connection and models"""
    base_url: str
    embedding_model: str
    llm_model: str
    embedding_dim: int = 768
    request_timeout: int = 120


@dataclass
class QdrantConfig:
    """Configuration for Qdrant vector database"""
    host: str
    port: int
    api_key: Optional[str] = None
    timeout: int = 30


@dataclass
class PostgresConfig:
    """Configuration for PostgreSQL metadata store"""
    host: str
    port: int
    database: str
    user: str
    password: str
    pool_size: int = 5


@dataclass
class RAGConfig:
    """Configuration for RAG pipeline parameters"""
    chunk_size: int  # tokens per chunk
    chunk_overlap: int  # overlap between chunks
    video_relevance_threshold: float  # 0.0-1.0
    pdf_relevance_threshold: float  # 0.0-1.0
    top_k_video: int  # number of video chunks to retrieve
    top_k_pdf: int  # number of PDF paragraphs to retrieve
    max_context_length: int  # max tokens in final answer


@dataclass
class DataConfig:
    """Configuration for data paths"""
    video_dir: Path
    pdf_dir: Path
    indices_dir: Path
    metadata_dir: Path


@dataclass
class Config:
    """Complete system configuration"""
    ollama: OllamaConfig
    qdrant: QdrantConfig
    postgres: Optional[PostgresConfig]
    rag: RAGConfig
    data: DataConfig
    log_level: str
    api_host: str
    api_port: int
    debug: bool


def load_config() -> Config:
    """
    Load configuration from environment variables.
    Falls back to sensible defaults for development.
    """
    
    # Ollama configuration
    ollama_config = OllamaConfig(
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
        llm_model=os.getenv("LLM_MODEL", "mistral"),
    )
    
    # Qdrant configuration
    qdrant_config = QdrantConfig(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", "6333")),
        api_key=os.getenv("QDRANT_API_KEY"),
    )
    
    # PostgreSQL configuration (optional)
    postgres_config = None
    if os.getenv("DATABASE_URL"):
        # Parse DATABASE_URL format: postgresql://user:pass@host:port/dbname
        db_url = os.getenv("DATABASE_URL")
        postgres_config = PostgresConfig(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            database=os.getenv("POSTGRES_DB", "rag_chatbot_db"),
            user=os.getenv("POSTGRES_USER", "rag_chatbot_user"),
            password=os.getenv("POSTGRES_PASSWORD", ""),
        )
    
    # RAG pipeline configuration
    rag_config = RAGConfig(
        chunk_size=int(os.getenv("CHUNK_SIZE", "512")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "128")),
        video_relevance_threshold=float(os.getenv("VIDEO_RELEVANCE_THRESHOLD", "0.7")),
        pdf_relevance_threshold=float(os.getenv("PDF_RELEVANCE_THRESHOLD", "0.6")),
        top_k_video=int(os.getenv("TOP_K_VIDEO", "3")),
        top_k_pdf=int(os.getenv("TOP_K_PDF", "3")),
        max_context_length=int(os.getenv("MAX_CONTEXT_LENGTH", "2048")),
    )
    
    # Data paths
    base_dir = Path(os.getenv("DATA_DIR", "data"))
    data_config = DataConfig(
        video_dir=base_dir / "videos",
        pdf_dir=base_dir / "pdfs",
        indices_dir=base_dir / "indices",
        metadata_dir=base_dir / "metadata",
    )
    
    # Create directories if they don't exist
    for directory in [data_config.video_dir, data_config.pdf_dir, 
                      data_config.indices_dir, data_config.metadata_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Main configuration
    config = Config(
        ollama=ollama_config,
        qdrant=qdrant_config,
        postgres=postgres_config,
        rag=rag_config,
        data=data_config,
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        api_host=os.getenv("API_HOST", "0.0.0.0"),
        api_port=int(os.getenv("API_PORT", "8000")),
        debug=os.getenv("DEBUG", "False").lower() == "true",
    )
    
    logger.info("Configuration loaded successfully")
    return config
