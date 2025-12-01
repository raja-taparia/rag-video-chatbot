"""Embedding generation utilities.

Provides two embedder implementations:
- `SentenceTransformersEmbedder` (primary): local, fast, CPU-friendly using
  the `sentence-transformers` library and the `all-MiniLM-L6-v2` model.
- `OllamaEmbedder` (optional): keeps existing Ollama-based embedding logic
  and is usable as a fallback.

The module exposes `get_default_embedder(config)` which returns a ready-to-use
embedder instance (sentence-transformers by default).
"""

import logging
import requests
import time
from typing import List, Optional

logger = logging.getLogger(__name__)


class SentenceTransformersEmbedder:
    """Embeddings using the `sentence-transformers` library.

    This is the preferred/default embedder for both PDF paragraphs and
    video transcript chunks. It exposes the same `embed_text` and
    `embed_batch` methods as the OllamaEmbedder to keep the rest of the
    codebase unchanged.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # Defer import so environments without sentence-transformers can still
        # import this module until the class is actually instantiated.
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:
            logger.error("sentence-transformers package is required for local embeddings: %s", e)
            raise

        self.model_name = model_name
        logger.info("Loading SentenceTransformer model: %s", model_name)
        self.model = SentenceTransformer(model_name)
        # Get dimensionality from the model
        self.embedding_dim = getattr(self.model, 'get_sentence_embedding_dimension', lambda: None)()
        logger.info("SentenceTransformersEmbedder ready: model=%s dim=%s", model_name, self.embedding_dim)

    def embed_text(self, text: str) -> Optional[List[float]]:
        try:
            emb = self.model.encode([text], show_progress_bar=False)
            return emb[0].tolist()
        except Exception as e:
            logger.error("Error embedding text with sentence-transformers: %s", e)
            return None

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[Optional[List[float]]]:
        try:
            embs = self.model.encode(texts, batch_size=batch_size, show_progress_bar=False)
            # `embs` is a numpy array; convert rows to lists
            return [row.tolist() for row in embs]
        except Exception as e:
            logger.error("Error embedding batch with sentence-transformers: %s", e)
            # Fallback: try embedding one-by-one to salvage partial results
            results: List[Optional[List[float]]] = []
            for text in texts:
                try:
                    single = self.model.encode([text], show_progress_bar=False)[0].tolist()
                    results.append(single)
                except Exception:
                    results.append(None)
            return results


from src.config import OllamaConfig


class OllamaEmbedder:
    """
    Generate embeddings using Ollama's local embedding models.
    """

    def __init__(self, config: OllamaConfig):
        self.base_url = config.base_url
        self.model = config.embedding_model
        self.embedding_dim = config.embedding_dim
        self.request_timeout = config.request_timeout

        logger.info(f"OllamaEmbedder initialized: model={self.model}, dim={self.embedding_dim}")

        # Verify connection
        self._verify_connection()

    def _verify_connection(self) -> None:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            logger.info("Successfully connected to Ollama service")
        except Exception as e:
            logger.error(f"Failed to connect to Ollama at {self.base_url}: {e}")
            raise

    def embed_text(self, text: str, retry_count: int = 3) -> Optional[List[float]]:
        for attempt in range(retry_count):
            try:
                response = requests.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.model, "input": [text]},
                    timeout=self.request_timeout,
                )
                response.raise_for_status()
                data = response.json()

                embedding = None
                if isinstance(data, dict):
                    if "embedding" in data and isinstance(data["embedding"], list):
                        embedding = data["embedding"]
                    elif "embeddings" in data and isinstance(data["embeddings"], list):
                        embedding = data["embeddings"][0]
                    elif "data" in data and isinstance(data["data"], list) and data["data"]:
                        first = data["data"][0]
                        if isinstance(first, dict) and "embedding" in first:
                            embedding = first["embedding"]

                if embedding is None and isinstance(data, list) and data:
                    first = data[0]
                    if isinstance(first, dict) and "embedding" in first:
                        embedding = first["embedding"]

                if embedding:
                    return embedding
                else:
                    logger.warning("No embedding in Ollama response; full_response=%s", data)
                    return None

            except requests.exceptions.Timeout:
                logger.warning(f"Timeout on attempt {attempt + 1}/{retry_count}")
                if attempt < retry_count - 1:
                    time.sleep(2 ** attempt)
            except Exception as e:
                logger.error(f"Error generating embedding (attempt {attempt + 1}/{retry_count}): {e}")
                if attempt < retry_count - 1:
                    time.sleep(2 ** attempt)

        logger.error(f"Failed to generate embedding after {retry_count} attempts")
        return None

    def embed_batch(self, texts: List[str], batch_size: int = 10) -> List[Optional[List[float]]]:
        embeddings = []
        for i, text in enumerate(texts):
            embedding = self.embed_text(text)
            embeddings.append(embedding)
            if (i + 1) % batch_size == 0:
                logger.info(f"Embedded {i + 1}/{len(texts)} texts via Ollama")
        logger.info(f"Ollama embedding complete: {len(texts)} texts processed")
        return embeddings


def get_default_embedder(config: OllamaConfig = None):
    """Return the default embedder.

    By default we prefer `sentence-transformers` (`all-MiniLM-L6-v2`) as it is
    fast and reliable for both PDF chunks and video transcripts. If
    `SENTENCE_TRANSFORMERS_FALLBACK_ONLY` is set in the environment or if
    sentence-transformers cannot be loaded, the function can fall back to
    Ollama if a valid `OllamaConfig` is provided.
    """
    # Prefer sentence-transformers
    try:
        return SentenceTransformersEmbedder(model_name=(config.embedding_model if config and config.embedding_model else "all-MiniLM-L6-v2"))
    except Exception:
        logger.warning("sentence-transformers unavailable; falling back to Ollama if configured")
        if config is not None:
            return OllamaEmbedder(config)
        raise
