# Unit tests for embedding generation

import pytest
import numpy as np
from typing import List
from src.indexing.embeddings import OllamaEmbedder


class TestEmbeddingGeneration:
    """Test embedding generation functionality."""
    
    def test_embed_single_text(self, mock_ollama_embedder):
        """Test embedding a single text."""
        embedder = mock_ollama_embedder
        
        text = "This is a sample text for embedding"
        embedding = embedder.embed_text(text)
        
        assert embedding is not None
        assert isinstance(embedding, (list, np.ndarray))
        assert len(embedding) == 384  # all-MiniLM-L6-v2 dimension
    
    def test_embed_batch_texts(self, mock_ollama_embedder):
        """Test embedding batch of texts."""
        embedder = mock_ollama_embedder
        
        texts = [
            "First text for embedding",
            "Second text for embedding",
            "Third text for embedding"
        ]
        
        embeddings = embedder.embed_batch(texts)
        
        assert len(embeddings) == 3
        for embedding in embeddings:
            assert len(embedding) == 384
    
    def test_embedding_dimension(self, mock_ollama_embedder):
        """Test that embeddings have correct dimension."""
        embedder = mock_ollama_embedder
        
        text = "Dimension test"
        embedding = embedder.embed_text(text)
        
        # all-MiniLM-L6-v2 produces 384-dimensional embeddings
        assert len(embedding) == 384
    
    def test_embedding_is_normalized(self, mock_ollama_embedder):
        """Test that embeddings are properly normalized."""
        embedder = mock_ollama_embedder
        
        text = "Test normalization"
        embedding = embedder.embed_text(text)
        
        # Calculate L2 norm
        norm = np.linalg.norm(embedding)
        
        # Should be close to 1.0 (normalized)
        assert 0.99 <= norm <= 1.01
    
    def test_similar_texts_similar_embeddings(self, mock_ollama_embedder):
        """Test that similar texts have similar embeddings."""
        embedder = mock_ollama_embedder
        
        text1 = "Docker is a containerization platform"
        text2 = "Docker is used for containers"
        text3 = "Kubernetes is an orchestration system"  # Different topic
        
        emb1 = embedder.embed_text(text1)
        emb2 = embedder.embed_text(text2)
        emb3 = embedder.embed_text(text3)
        
        # Cosine similarity between similar texts
        similarity_similar = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        # Cosine similarity between different topics
        similarity_different = np.dot(emb1, emb3) / (np.linalg.norm(emb1) * np.linalg.norm(emb3))
        
        # Similar texts should have higher similarity
        assert similarity_similar > similarity_different
    
    def test_embedding_consistency(self, mock_ollama_embedder):
        """Test that same text produces same embedding."""
        embedder = mock_ollama_embedder
        
        text = "Consistent embedding test"
        
        # Generate embedding twice
        emb1 = embedder.embed_text(text)
        emb2 = embedder.embed_text(text)
        
        # Should be identical
        np.testing.assert_array_almost_equal(emb1, emb2)
    
    def test_embed_empty_text(self, mock_ollama_embedder):
        """Test embedding empty text."""
        embedder = mock_ollama_embedder
        
        text = ""
        embedding = embedder.embed_text(text)
        
        # Should still return a valid embedding
        assert embedding is not None
        assert len(embedding) == 384
    
    def test_embed_very_long_text(self, mock_ollama_embedder):
        """Test embedding very long text."""
        embedder = mock_ollama_embedder
        
        # Create very long text
        text = "word " * 10000  # 50,000 characters
        embedding = embedder.embed_text(text)
        
        assert embedding is not None
        assert len(embedding) == 384


class TestEmbeddingQuality:
    """Test embedding quality metrics."""
    
    def test_cosine_similarity_calculation(self, mock_ollama_embedder):
        """Test cosine similarity between embeddings."""
        embedder = mock_ollama_embedder
        
        text1 = "Docker containers"
        text2 = "Docker containers"  # Identical
        
        emb1 = embedder.embed_text(text1)
        emb2 = embedder.embed_text(text2)
        
        # Calculate cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        
        # Identical texts should have similarity near 1.0
        assert 0.95 <= similarity <= 1.0
    
    def test_euclidean_distance(self, mock_ollama_embedder):
        """Test Euclidean distance between embeddings."""
        embedder = mock_ollama_embedder
        
        text1 = "Install Docker on macOS"
        text2 = "Install Docker on Linux"
        
        emb1 = embedder.embed_text(text1)
        emb2 = embedder.embed_text(text2)
        
        # Calculate Euclidean distance
        distance = np.linalg.norm(np.array(emb1) - np.array(emb2))
        
        # Distance should be positive but not too large
        assert 0 < distance < 100
    
    def test_embedding_isotropy(self, mock_ollama_embedder):
        """Test that embeddings are isotropic (well-distributed)."""
        embedder = mock_ollama_embedder
        
        # Generate embeddings for diverse texts
        texts = [
            "Docker is a container platform",
            "Kubernetes orchestrates containers",
            "Python is a programming language",
            "Coffee is a beverage",
            "Quantum computing is advanced"
        ]
        
        embeddings = embedder.embed_batch(texts)
        
        # Calculate pairwise similarities
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                similarities.append(sim)
        
        # Variance in similarities indicates good isotropy
        variance = np.var(similarities)
        assert variance > 0.01  # Some diversity in similarities


class TestEmbeddingPerformance:
    """Test embedding generation performance."""
    
    def test_embedding_speed(self, mock_ollama_embedder):
        """Test speed of embedding generation."""
        import time
        
        embedder = mock_ollama_embedder
        
        text = "Docker and Kubernetes are container technologies."
        
        start = time.time()
        embedding = embedder.embed_text(text)
        elapsed = time.time() - start
        
        # Should be fast (under 1 second for mock)
        assert elapsed < 1.0
        assert embedding is not None
    
    def test_batch_embedding_performance(self, mock_ollama_embedder):
        """Test performance of batch embedding."""
        import time
        
        embedder = mock_ollama_embedder
        
        # Create 100 texts
        texts = [f"Document {i}: Docker and Kubernetes content" for i in range(100)]
        
        start = time.time()
        embeddings = embedder.embed_batch(texts)
        elapsed = time.time() - start
        
        assert len(embeddings) == 100
        # Batch should be reasonably fast
        assert elapsed < 5.0
    
    def test_embedding_memory_efficiency(self, mock_ollama_embedder):
        """Test memory efficiency of embeddings."""
        embedder = mock_ollama_embedder
        
        # Create many embeddings
        embeddings = []
        for i in range(1000):
            text = f"Text {i}: Sample content for embedding"
            emb = embedder.embed_text(text)
            embeddings.append(emb)
        
        assert len(embeddings) == 1000
        
        # Each embedding should be 384 floats (~1.5KB each)
        total_size_mb = (len(embeddings) * 384 * 4) / (1024 * 1024)  # Convert to MB
        assert total_size_mb < 10  # Should be under 10MB for 1000 embeddings


class TestEmbeddingStorage:
    """Test embedding storage and retrieval."""
    
    def test_embedding_serialization(self, mock_ollama_embedder):
        """Test serializing embeddings to JSON."""
        import json
        
        embedder = mock_ollama_embedder
        
        text = "Test serialization"
        embedding = embedder.embed_text(text)
        
        # Convert to JSON-serializable format
        embedding_list = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
        json_str = json.dumps(embedding_list)

        # Deserialize
        loaded = json.loads(json_str)

        assert len(loaded) == 384
        np.testing.assert_array_almost_equal(embedding, loaded)
    
    def test_embedding_caching(self, mock_ollama_embedder):
        """Test caching of embeddings."""
        embedder = mock_ollama_embedder
        
        if hasattr(embedder, 'enable_cache'):
            embedder.enable_cache()
        
        text = "Cacheable text"
        
        # First embedding
        emb1 = embedder.embed_text(text)
        # Second embedding (should be from cache)
        emb2 = embedder.embed_text(text)
        
        np.testing.assert_array_equal(emb1, emb2)


class TestEmbeddingValidation:
    """Test embedding validation and error handling."""
    
    def test_validate_embedding_dimension(self, mock_ollama_embedder):
        """Test validation of embedding dimension."""
        embedder = mock_ollama_embedder
        
        text = "Validate dimension"
        embedding = embedder.embed_text(text)
        
        # Should be 384-dimensional
        assert len(embedding) == 384
    
    def test_validate_embedding_values(self, mock_ollama_embedder):
        """Test that embedding values are valid floats."""
        embedder = mock_ollama_embedder
        
        text = "Validate values"
        embedding = embedder.embed_text(text)
        
        # All values should be finite floats
        for value in embedding:
            assert isinstance(value, (float, np.floating))
            assert np.isfinite(value)
    
    def test_embedding_range(self, mock_ollama_embedder):
        """Test that embedding values are in reasonable range."""
        embedder = mock_ollama_embedder
        
        text = "Check range"
        embedding = embedder.embed_text(text)
        
        # Normalized embeddings should be between -1 and 1
        assert np.all(np.array(embedding) >= -1.5)
        assert np.all(np.array(embedding) <= 1.5)
    
    def test_handle_special_characters(self, mock_ollama_embedder):
        """Test embedding with special characters."""
        embedder = mock_ollama_embedder
        
        text = "Special chars: é à ñ 中文 🚀"
        embedding = embedder.embed_text(text)
        
        assert embedding is not None
        assert len(embedding) == 384
    
    def test_handle_languages(self, mock_ollama_embedder):
        """Test embedding multiple languages."""
        embedder = mock_ollama_embedder
        
        texts = [
            "English text",
            "Texte français",
            "Texto español",
            "Texto português"
        ]
        
        embeddings = embedder.embed_batch(texts)
        
        assert len(embeddings) == 4
        for emb in embeddings:
            assert len(emb) == 384


class TestEmbeddingComparison:
    """Test comparing embeddings."""
    
    def test_find_most_similar(self, mock_ollama_embedder):
        """Test finding most similar embedding."""
        embedder = mock_ollama_embedder
        
        texts = [
            "Docker containerization",
            "Container technology Docker",
            "Kubernetes orchestration",
            "Container management"
        ]
        
        embeddings = embedder.embed_batch(texts)
        
        # First text should be most similar to similar texts
        target_emb = embeddings[0]
        
        similarities = []
        for emb in embeddings[1:]:
            sim = np.dot(target_emb, emb) / (
                np.linalg.norm(target_emb) * np.linalg.norm(emb)
            )
            similarities.append(sim)
        
        # First and second are similar, should have high similarity
        assert similarities[0] > similarities[1]  # "Container technology Docker" is more similar than "Kubernetes"
    
    def test_embedding_clustering(self, mock_ollama_embedder):
        """Test that embeddings can be clustered by similarity."""
        embedder = mock_ollama_embedder
        
        docker_texts = [
            "Docker containerization",
            "Docker images and containers",
            "Docker tutorial"
        ]
        
        k8s_texts = [
            "Kubernetes orchestration",
            "Kubernetes pods and services",
            "Kubernetes tutorial"
        ]
        
        docker_embeddings = embedder.embed_batch(docker_texts)
        k8s_embeddings = embedder.embed_batch(k8s_texts)
        
        # Intra-group similarity should be higher than inter-group
        intra_similarity = np.dot(docker_embeddings[0], docker_embeddings[1]) / (
            np.linalg.norm(docker_embeddings[0]) * np.linalg.norm(docker_embeddings[1])
        )
        
        inter_similarity = np.dot(docker_embeddings[0], k8s_embeddings[0]) / (
            np.linalg.norm(docker_embeddings[0]) * np.linalg.norm(k8s_embeddings[0])
        )
        
        assert intra_similarity > inter_similarity
