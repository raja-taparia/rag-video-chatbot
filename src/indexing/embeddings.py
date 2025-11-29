# Embedding generation using Ollama

import logging
import requests
import time
from typing import List, Optional
from src.config import OllamaConfig

logger = logging.getLogger(__name__)


class OllamaEmbedder:
    """
    Generate embeddings using Ollama's local embedding models.
    
    Features:
    - Local execution (no API costs)
    - Configurable model selection
    - Batch processing for efficiency
    - Retry logic for robustness
    """
    
    def __init__(self, config: OllamaConfig):
        """
        Initialize embedder with Ollama config.
        
        Args:
            config: OllamaConfig object
        """
        self.base_url = config.base_url
        self.model = config.embedding_model
        self.embedding_dim = config.embedding_dim
        self.request_timeout = config.request_timeout
        
        logger.info(f"OllamaEmbedder initialized: model={self.model}, dim={self.embedding_dim}")
        
        # Verify connection
        self._verify_connection()
    
    def _verify_connection(self) -> None:
        """Verify Ollama service is running and accessible."""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5
            )
            response.raise_for_status()
            logger.info("Successfully connected to Ollama service")
        except Exception as e:
            logger.error(f"Failed to connect to Ollama at {self.base_url}: {e}")
            raise
    
    def embed_text(self, text: str, retry_count: int = 3) -> Optional[List[float]]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            retry_count: Number of retries on failure
            
        Returns:
            List of floats representing the embedding, or None on failure
        """
        for attempt in range(retry_count):
            try:
                response = requests.post(
                    f"{self.base_url}/api/embeddings",
                    json={
                        "model": self.model,
                        "prompt": text
                    },
                    timeout=self.request_timeout
                )
                response.raise_for_status()
                
                data = response.json()
                embedding = data.get("embedding")
                
                if embedding:
                    return embedding
                else:
                    logger.warning(f"No embedding in response for text: {text[:50]}...")
                    return None
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout on attempt {attempt + 1}/{retry_count}")
                if attempt < retry_count - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
            except Exception as e:
                logger.error(f"Error generating embedding (attempt {attempt + 1}/{retry_count}): {e}")
                if attempt < retry_count - 1:
                    time.sleep(2 ** attempt)
        
        logger.error(f"Failed to generate embedding after {retry_count} attempts")
        return None
    
    def embed_batch(self, texts: List[str], batch_size: int = 10) -> List[Optional[List[float]]]:
        """
        Generate embeddings for multiple texts.
        
        Processes in batches for efficiency and better error handling.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process before logging progress
            
        Returns:
            List of embeddings (None for failed entries)
        """
        embeddings = []
        
        for i, text in enumerate(texts):
            embedding = self.embed_text(text)
            embeddings.append(embedding)
            
            if (i + 1) % batch_size == 0:
                logger.info(f"Embedded {i + 1}/{len(texts)} texts")
        
        logger.info(f"Embedding complete: {len(texts)} texts processed")
        return embeddings
