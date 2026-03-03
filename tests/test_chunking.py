# Unit tests for chunking logic

import pytest
from pathlib import Path
from src.ingestion.chunker import TextChunker
from src.models import VideoChunk


class TestChunkingLogic:
    """Test video and text chunking functionality."""
    
    def test_chunk_video_by_tokens(self):
        """Test chunking video transcript by token count."""
        chunker = TextChunker(chunk_size=10, chunk_overlap=2)
        
        # Create mock tokens
        tokens = [
            {"id": i, "timestamp": float(i), "word": f"word{i}"}
            for i in range(1, 31)
        ]
        
        chunks = chunker.chunk_video_tokens(tokens, chunk_size=10, overlap=2)
        
        # Should create 3 chunks: [0:10], [8:18], [16:26], [24:30]
        assert len(chunks) >= 2
        assert chunks[0]["start_token_id"] == 1
        assert chunks[0]["end_token_id"] == 10
    
    def test_chunk_with_overlap(self):
        """Test that chunks properly overlap."""
        chunker = TextChunker(chunk_size=5, chunk_overlap=2)
        
        tokens = [
            {"id": i, "timestamp": float(i), "word": f"word{i}"}
            for i in range(1, 13)
        ]
        
        chunks = chunker.chunk_video_tokens(tokens, chunk_size=5, overlap=2)
        
        # Verify overlap between consecutive chunks
        for i in range(len(chunks) - 1):
            current_end = chunks[i]["end_token_id"]
            next_start = chunks[i + 1]["start_token_id"]
            
            # Should have 2 token overlap
            assert current_end - next_start == 2
    
    def test_chunk_preserves_timestamps(self):
        """Test that chunk timestamps are correctly mapped."""
        chunker = TextChunker(chunk_size=5, chunk_overlap=0)
        
        tokens = [
            {"id": 1, "timestamp": 0.0, "word": "start"},
            {"id": 2, "timestamp": 1.0, "word": "second"},
            {"id": 3, "timestamp": 2.0, "word": "third"},
            {"id": 4, "timestamp": 3.0, "word": "fourth"},
            {"id": 5, "timestamp": 4.0, "word": "fifth"},
            {"id": 6, "timestamp": 5.0, "word": "sixth"},
        ]
        
        chunks = chunker.chunk_video_tokens(tokens, chunk_size=5, overlap=0)
        
        # First chunk should start at 0.0s
        assert chunks[0]["start_timestamp"] == 0.0
        # First chunk should end at 4.0s (fifth token)
        assert chunks[0]["end_timestamp"] == 4.0
    
    def test_chunk_text_chunking(self):
        """Test plain text chunking."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        
        text = "This is a sample text. " * 20  # Create longer text
        
        chunks = chunker.chunk_text(text, chunk_size=100, overlap=20)
        
        assert len(chunks) > 0
        # All chunks should be under max size (allowing some tolerance)
        for chunk in chunks:
            assert len(chunk) <= 120  # 100 + tolerance for overlap
    
    def test_chunk_empty_input(self):
        """Test chunking with empty input."""
        chunker = TextChunker()
        
        empty_tokens = []
        chunks = chunker.chunk_video_tokens(empty_tokens)
        
        assert len(chunks) == 0
    
    def test_chunk_single_token(self):
        """Test chunking with single token."""
        chunker = TextChunker(chunk_size=10)
        
        tokens = [{"id": 1, "timestamp": 0.0, "word": "only"}]
        chunks = chunker.chunk_video_tokens(tokens)
        
        assert len(chunks) == 1
        assert chunks[0]["start_token_id"] == 1
        assert chunks[0]["end_token_id"] == 1
    
    def test_chunk_small_overlap(self):
        """Test chunks with very small overlap."""
        chunker = TextChunker(chunk_size=5, chunk_overlap=1)
        
        tokens = [
            {"id": i, "timestamp": float(i), "word": f"word{i}"}
            for i in range(1, 16)
        ]
        
        chunks = chunker.chunk_video_tokens(tokens, chunk_size=5, overlap=1)
        
        # Verify 1-token overlap
        for i in range(len(chunks) - 1):
            current_end = chunks[i]["end_token_id"]
            next_start = chunks[i + 1]["start_token_id"]
            assert current_end - next_start == 1
    
    def test_chunk_large_overlap(self):
        """Test chunks with large overlap relative to size."""
        chunker = TextChunker(chunk_size=5, chunk_overlap=4)
        
        tokens = [
            {"id": i, "timestamp": float(i), "word": f"word{i}"}
            for i in range(1, 16)
        ]
        
        chunks = chunker.chunk_video_tokens(tokens, chunk_size=5, overlap=4)
        
        # Verify 4-token overlap
        assert len(chunks) > 1  # Should have multiple chunks due to sliding window


class TestChunkingEdgeCases:
    """Test edge cases and error handling."""
    
    def test_chunk_size_larger_than_input(self):
        """Test when chunk size is larger than input."""
        chunker = TextChunker(chunk_size=100)
        
        tokens = [
            {"id": i, "timestamp": float(i), "word": f"word{i}"}
            for i in range(1, 6)
        ]
        
        chunks = chunker.chunk_video_tokens(tokens, chunk_size=100)
        
        # Should create single chunk with all tokens
        assert len(chunks) == 1
        assert chunks[0]["start_token_id"] == 1
        assert chunks[0]["end_token_id"] == 5
    
    def test_chunk_with_negative_overlap_error(self):
        """Test that negative overlap raises error."""
        chunker = TextChunker()
        
        tokens = [{"id": 1, "timestamp": 0.0, "word": "test"}]
        
        # Negative overlap should raise error
        with pytest.raises(ValueError):
            chunker.chunk_video_tokens(tokens, chunk_size=5, overlap=-1)
    
    def test_chunk_with_overlap_larger_than_size_error(self):
        """Test that overlap >= size raises error."""
        chunker = TextChunker()
        
        tokens = [{"id": i, "timestamp": float(i), "word": f"w{i}"} for i in range(1, 10)]
        
        # Overlap >= size should raise error
        with pytest.raises(ValueError):
            chunker.chunk_video_tokens(tokens, chunk_size=5, overlap=5)
    
    def test_chunk_timestamps_with_duplicates(self):
        """Test handling of duplicate timestamps."""
        chunker = TextChunker(chunk_size=5)
        
        tokens = [
            {"id": 1, "timestamp": 1.0, "word": "same"},
            {"id": 2, "timestamp": 1.0, "word": "time"},  # Duplicate timestamp
            {"id": 3, "timestamp": 2.0, "word": "different"},
            {"id": 4, "timestamp": 3.0, "word": "later"},
            {"id": 5, "timestamp": 3.0, "word": "also"},
        ]
        
        chunks = chunker.chunk_video_tokens(tokens)
        
        # Should still work, using first and last timestamps
        assert chunks[0]["start_timestamp"] == 1.0
    
    def test_chunk_non_sequential_token_ids(self):
        """Test tokens with non-sequential IDs."""
        chunker = TextChunker(chunk_size=3)
        
        tokens = [
            {"id": 10, "timestamp": 0.0, "word": "first"},
            {"id": 20, "timestamp": 1.0, "word": "second"},
            {"id": 30, "timestamp": 2.0, "word": "third"},
            {"id": 40, "timestamp": 3.0, "word": "fourth"},
        ]
        
        chunks = chunker.chunk_video_tokens(tokens)
        
        # Should use actual token IDs, not indices
        assert chunks[0]["start_token_id"] == 10
        assert chunks[0]["end_token_id"] == 30  # Last token in first chunk


class TestChunkingPerformance:
    """Test chunking with realistic data sizes."""
    
    def test_chunk_large_transcript(self):
        """Test chunking with large transcript (1000+ tokens)."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=10)
        
        # Create large token list
        tokens = [
            {"id": i, "timestamp": float(i * 0.1), "word": f"word{i}"}
            for i in range(1, 1001)
        ]
        
        chunks = chunker.chunk_video_tokens(tokens)
        
        assert len(chunks) > 5
        # Verify no gaps between chunks
        for i in range(len(chunks) - 1):
            current_end = chunks[i]["end_token_id"]
            next_start = chunks[i + 1]["start_token_id"]
            # Next chunk should start at or before current end
            assert next_start <= current_end
    
    def test_chunk_performance_with_many_tokens(self):
        """Test that chunking is performant with many tokens."""
        import time
        
        chunker = TextChunker(chunk_size=500)
        
        # Create 10000 tokens
        tokens = [
            {"id": i, "timestamp": float(i * 0.01), "word": f"word{i}"}
            for i in range(1, 10001)
        ]
        
        start_time = time.time()
        chunks = chunker.chunk_video_tokens(tokens)
        elapsed = time.time() - start_time
        
        # Should complete in under 1 second
        assert elapsed < 1.0
        assert len(chunks) > 0


class TestChunkGeneration:
    """Test chunk object generation and validation."""
    
    def test_chunk_object_creation(self):
        """Test creating proper chunk objects."""
        chunker = TextChunker()
        
        tokens = [
            {"id": 1, "timestamp": 0.0, "word": "hello"},
            {"id": 2, "timestamp": 0.5, "word": "world"},
            {"id": 3, "timestamp": 1.0, "word": "test"},
        ]
        
        chunks = chunker.chunk_video_tokens(tokens)
        chunk = chunks[0]
        
        # Verify required fields
        assert "chunk_id" in chunk
        assert "video_id" in chunk or chunk["chunk_id"].startswith("chunk_")
        assert "start_token_id" in chunk
        assert "end_token_id" in chunk
        assert "start_timestamp" in chunk
        assert "end_timestamp" in chunk
        assert "text" in chunk
    
    def test_chunk_text_content(self):
        """Test that chunk text contains reconstructed transcript."""
        chunker = TextChunker(chunk_size=3)
        
        tokens = [
            {"id": 1, "timestamp": 0.0, "word": "The"},
            {"id": 2, "timestamp": 0.5, "word": "quick"},
            {"id": 3, "timestamp": 1.0, "word": "brown"},
            {"id": 4, "timestamp": 1.5, "word": "fox"},
        ]
        
        chunks = chunker.chunk_video_tokens(tokens)
        
        # First chunk should contain reconstructed text
        chunk_text = chunks[0]["text"]
        assert "quick" in chunk_text.lower() or "brown" in chunk_text.lower()
