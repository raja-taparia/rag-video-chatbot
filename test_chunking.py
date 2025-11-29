# Tests for chunking logic

import pytest
from src.ingestion.chunker import VideoChunker
from src.models import VideoTranscriptFile


def test_video_chunker_initialization():
    """Test VideoChunker initialization with valid parameters."""
    chunker = VideoChunker(chunk_size=512, chunk_overlap=128)
    assert chunker.chunk_size == 512
    assert chunker.chunk_overlap == 128


def test_video_chunker_invalid_parameters():
    """Test VideoChunker raises error for invalid parameters."""
    with pytest.raises(ValueError):
        VideoChunker(chunk_size=512, chunk_overlap=600)


def test_video_chunker_create_chunks(mock_transcript):
    """Test chunk creation from transcript."""
    chunker = VideoChunker(chunk_size=10, chunk_overlap=2)
    chunks = chunker.create_chunks(mock_transcript)
    
    assert len(chunks) > 0
    assert all(chunk.video_id == mock_transcript.video_id for chunk in chunks)
    
    # Verify token continuity
    for chunk in chunks:
        assert chunk.start_token_id <= chunk.end_token_id
        assert chunk.start_timestamp <= chunk.end_timestamp


def test_video_chunker_overlap():
    """Test that chunks have proper overlap."""
    # Create transcript with many tokens
    tokens = [
        {"id": i, "timestamp": i * 0.1, "word": f"word{i}"}
        for i in range(100)
    ]
    transcript = VideoTranscriptFile(
        video_id="TEST",
        video_transcripts=tokens
    )
    
    chunker = VideoChunker(chunk_size=20, chunk_overlap=5)
    chunks = chunker.create_chunks(transcript)
    
    # Check first two chunks for overlap
    if len(chunks) >= 2:
        chunk1_end = chunks[0].end_token_id
        chunk2_start = chunks[1].start_token_id
        overlap = chunk1_end - chunk2_start + 1
        assert overlap == 5  # Should have 5 token overlap


def test_video_chunker_timestamp_mapping(mock_transcript):
    """Test that timestamps are correctly mapped."""
    chunker = VideoChunker(chunk_size=5, chunk_overlap=1)
    chunks = chunker.create_chunks(mock_transcript)
    
    for chunk in chunks:
        # Find corresponding tokens
        tokens = mock_transcript.video_transcripts
        start_token = next(t for t in tokens if t.id == chunk.start_token_id)
        end_token = next(t for t in tokens if t.id == chunk.end_token_id)
        
        assert chunk.start_timestamp == start_token.timestamp
        assert chunk.end_timestamp == end_token.timestamp


def test_video_chunker_empty_transcript():
    """Test chunking with empty transcript."""
    chunker = VideoChunker()
    empty_transcript = VideoTranscriptFile(
        video_id="EMPTY",
        video_transcripts=[]
    )
    
    chunks = chunker.create_chunks(empty_transcript)
    assert len(chunks) == 0


def test_video_chunker_batch_processing(mock_config):
    """Test batch chunking of multiple transcripts."""
    from src.ingestion.video_loader import VideoTranscriptLoader
    
    # Create multiple mock transcripts
    loader = VideoTranscriptLoader(mock_config.data.video_dir)
    
    transcripts = [
        VideoTranscriptFile(
            video_id=f"VIDEO_{i}",
            video_transcripts=[
                {"id": j, "timestamp": j * 0.5, "word": f"word{j}"}
                for j in range(20)
            ]
        )
        for i in range(3)
    ]
    
    chunker = VideoChunker(chunk_size=8, chunk_overlap=2)
    all_chunks = chunker.chunk_all(transcripts)
    
    # Should have chunks from all videos
    assert len(all_chunks) > 0
    
    # Verify distribution across videos
    video_ids = set(chunk.video_id for chunk in all_chunks)
    assert len(video_ids) == 3
