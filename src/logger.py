# Logging configuration for the RAG system

import logging
import logging.handlers
from pathlib import Path
from typing import Optional


def setup_logging(
    log_level: str = "INFO",
    log_dir: Optional[Path] = None
) -> None:
    """
    Configure logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory to store log files (default: logs/)
    """
    
    if log_dir is None:
        log_dir = Path("logs")
    
    log_dir.mkdir(exist_ok=True)
    
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level))
    
    # Formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level))
    console_handler.setFormatter(simple_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler - General
    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "app.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(getattr(logging, log_level))
    file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(file_handler)
    
    # File handler - Errors
    error_handler = logging.handlers.RotatingFileHandler(
        log_dir / "errors.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(error_handler)
    
    # Retrieval-specific logger
    retrieval_logger = logging.getLogger("retrieval")
    retrieval_file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "retrieval.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5
    )
    retrieval_file_handler.setFormatter(detailed_formatter)
    retrieval_logger.addHandler(retrieval_file_handler)
    
    # Indexing-specific logger
    indexing_logger = logging.getLogger("indexing")
    indexing_file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "indexing.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5
    )
    indexing_file_handler.setFormatter(detailed_formatter)
    indexing_logger.addHandler(indexing_file_handler)
