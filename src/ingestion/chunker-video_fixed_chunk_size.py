# Chunking strategy for video transcripts and PDFs

import logging
from typing import List, Tuple, Optional
from src.models import VideoTranscriptFile, TokenData, VideoChunk

logger = logging.getLogger(__name__)


class VideoChunker:
    """
    Semantic chunking for video transcripts using sliding window approach.
    
    Strategy:
    - Groups tokens into chunks of fixed size (default 512 tokens)
    - Maintains overlap between chunks (default 128 tokens) for context
    - Preserves timestamp information for exact playback positioning
    - Attempts to preserve sentence boundaries when possible
    """
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 128):
        """
        Initialize chunker with parameters.
        
        Args:
            chunk_size: Target tokens per chunk
            chunk_overlap: Overlap tokens between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        
        logger.info(f"VideoChunker initialized: size={chunk_size}, overlap={chunk_overlap}")
    
    def create_chunks(self, transcript: VideoTranscriptFile) -> List[VideoChunk]:
        """
        Create semantic chunks from a video transcript.
        
        Algorithm:
        1. Start at token 0
        2. Create chunk of chunk_size tokens
        3. Map to timestamps (first token and last token)
        4. Join tokens into text
        5. Advance by (chunk_size - chunk_overlap) tokens
        6. Repeat until all tokens processed
        
        Args:
            transcript: VideoTranscriptFile object
            
        Returns:
            List of VideoChunk objects
        """
        if not transcript.video_transcripts:
            logger.warning(f"No transcripts to chunk for {transcript.video_id}")
            return []
        
        tokens = transcript.video_transcripts
        chunks = []
        step_size = self.chunk_size - self.chunk_overlap
        
        chunk_idx = 0
        for start_idx in range(0, len(tokens), step_size):
            end_idx = min(start_idx + self.chunk_size, len(tokens))
            
            # Extract chunk tokens
            chunk_tokens = tokens[start_idx:end_idx]
            
            if not chunk_tokens:
                continue
            
            # Extract token IDs and timestamps
            start_token_id = chunk_tokens[0].id
            end_token_id = chunk_tokens[-1].id
            start_timestamp = chunk_tokens[0].timestamp
            end_timestamp = chunk_tokens[-1].timestamp
            
            # Join tokens into text (with spaces)
            text = ' '.join([t.word for t in chunk_tokens])
            
            # Create chunk
            chunk = VideoChunk(
                chunk_id=f"{transcript.video_id}_chunk_{chunk_idx}",
                video_id=transcript.video_id,
                start_token_id=start_token_id,
                end_token_id=end_token_id,
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
                text=text,
            )
            
            chunks.append(chunk)
            chunk_idx += 1
            
            # Stop if we've processed all tokens
            if end_idx >= len(tokens):
                break
        
        logger.info(f"Created {len(chunks)} chunks from {transcript.video_id}")
        return chunks
    
    def chunk_all(self, transcripts: List[VideoTranscriptFile]) -> List[VideoChunk]:
        """
        Create chunks from multiple transcripts.
        
        Args:
            transcripts: List of VideoTranscriptFile objects
            
        Returns:
            List of all VideoChunk objects
        """
        all_chunks = []
        
        for transcript in transcripts:
            chunks = self.create_chunks(transcript)
            all_chunks.extend(chunks)
        
        logger.info(f"Total chunks created: {len(all_chunks)}")
        return all_chunks


class PDFChunker:
    """
    Chunking strategy for PDF paragraphs.
    
    For PDFs, we maintain paragraph-level granularity since PDFs
    don't have timestamp information and we need to preserve citations.
    """
    
    @staticmethod
    def create_paragraph_segments(
        pdf_filename: str,
        paragraphs: List[Tuple[int, int, str]]  # (page, para_idx, text)
    ) -> List[dict]:
        """
        Create segments from PDF paragraphs.
        
        Args:
            pdf_filename: Name of PDF file
            paragraphs: List of (page_number, paragraph_index, text) tuples
            
        Returns:
            List of segment dictionaries with metadata
        """
        segments = []
        
        for page_num, para_idx, text in paragraphs:
            segment = {
                'pdf_filename': pdf_filename,
                'page_number': page_num,
                'paragraph_index': para_idx,
                'text': text,
            }
            segments.append(segment)
        
        logger.info(f"Created {len(segments)} segments from {pdf_filename}")
        return segments


class TextChunker:
    """
    Flexible chunking for token sequences and plain text.
    
    Supports both video transcripts (with timestamps) and plain text chunking
    with configurable chunk size and overlap.
    """
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 0):
        """
        Initialize chunker with parameters.
        
        Args:
            chunk_size: Target size per chunk
            chunk_overlap: Overlap between consecutive chunks (default 0, no overlap)
            
        Raises:
            ValueError: If chunk_overlap >= chunk_size
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        
        logger.info(f"TextChunker initialized: size={chunk_size}, overlap={chunk_overlap}")
    
    def chunk_video_tokens(
        self,
        tokens: List[dict],
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None
    ) -> List[dict]:
        """
        Chunk a list of token dicts with sliding window approach.
        
        Each token dict should have: id, timestamp, word
        
        Args:
            tokens: List of token dictionaries with id, timestamp, word
            chunk_size: Override default chunk_size (optional)
            overlap: Override default chunk_overlap (optional)
            
        Returns:
            List of chunk dictionaries with start/end ids, timestamps, and text
            
        Raises:
            ValueError: If overlap < 0 or overlap >= chunk_size
        """
        chunk_size = chunk_size if chunk_size is not None else self.chunk_size
        overlap = overlap if overlap is not None else self.chunk_overlap
        
        # Validate parameters
        if overlap < 0:
            raise ValueError("overlap cannot be negative")
        if overlap >= chunk_size:
            raise ValueError("overlap must be less than chunk_size")
        
        if not tokens:
            return []
        
        chunks = []
        # Step size creates overlaps where end_token_id - next_start_token_id == overlap
        step_size = max(1, chunk_size - 1 - overlap)
        chunk_idx = 0
        
        for start_idx in range(0, len(tokens), step_size):
            end_idx = min(start_idx + chunk_size, len(tokens))
            
            chunk_tokens = tokens[start_idx:end_idx]
            
            if not chunk_tokens:
                continue
            
            # Extract metadata from first and last tokens
            start_token_id = chunk_tokens[0].get("id", chunk_tokens[0].get("id", start_idx))
            end_token_id = chunk_tokens[-1].get("id", chunk_tokens[-1].get("id", end_idx - 1))
            start_timestamp = chunk_tokens[0].get("timestamp", 0.0)
            end_timestamp = chunk_tokens[-1].get("timestamp", 0.0)
            
            # Reconstruct text from tokens
            text = " ".join([t.get("word", "") for t in chunk_tokens])
            
            # Create chunk dict
            chunk = {
                "chunk_id": f"chunk_{chunk_idx}",
                "video_id": None,  # Not available in token-only mode
                "start_token_id": start_token_id,
                "end_token_id": end_token_id,
                "start_timestamp": start_timestamp,
                "end_timestamp": end_timestamp,
                "text": text,
            }
            
            chunks.append(chunk)
            chunk_idx += 1
            
            # Stop if we've processed all tokens
            if end_idx >= len(tokens):
                break
        
        logger.info(f"Created {len(chunks)} chunks from {len(tokens)} tokens")
        return chunks
    
    def chunk_text(
        self,
        text: str,
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None
    ) -> List[str]:
        """
        Chunk plain text by character count.
        
        Args:
            text: Text to chunk
            chunk_size: Override default chunk_size (optional, in characters)
            overlap: Override default chunk_overlap (optional, in characters)
            
        Returns:
            List of text chunks
            
        Raises:
            ValueError: If overlap >= chunk_size
        """
        chunk_size = chunk_size if chunk_size is not None else self.chunk_size
        overlap = overlap if overlap is not None else self.chunk_overlap
        
        if overlap >= chunk_size:
            raise ValueError("overlap must be less than chunk_size")
        
        if not text:
            return []
        
        chunks = []
        step_size = chunk_size - overlap
        
        for start_idx in range(0, len(text), step_size):
            end_idx = min(start_idx + chunk_size, len(text))
            chunk = text[start_idx:end_idx]
            
            if chunk:
                chunks.append(chunk)
            
            if end_idx >= len(text):
                break
        
        logger.info(f"Created {len(chunks)} text chunks from {len(text)} characters")
        return chunks

# Re-export TokenMapper for backward compatibility
from src.ingestion.token_mapper import TokenMapper

__all__ = ['VideoChunker', 'PDFChunker', 'TextChunker', 'TokenMapper']
