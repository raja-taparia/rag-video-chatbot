# Chunking strategy for video transcripts and PDFs

import logging
from typing import List, Tuple, Optional
from src.models import VideoTranscriptFile, TokenData, VideoChunk

logger = logging.getLogger(__name__)


class VideoChunker:
    """
    Pause-based chunking for video transcripts.

    Strategy:
    - Detect sentence boundaries using pauses (gaps between token timestamps)
    - Split tokens into chunks at pauses >= `pause_threshold`
    - Enforce `min_video_chunk_size` by merging very short chunks
    - Enforce `max_video_chunk_size` by splitting long segments at largest internal gaps
    - Fall back to regular splits when speech is uniform
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        pause_threshold: float = 1.5,
        min_chunk_size: int = 32,
        max_chunk_size: int = 512,
    ):
        # Keep backwards-compatible parameters but prefer pause-based settings
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.pause_threshold = float(pause_threshold)
        self.min_chunk_size = int(min_chunk_size)
        self.max_chunk_size = int(max_chunk_size)

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

        logger.info(
            f"VideoChunker initialized: pause_threshold={self.pause_threshold}, "
            f"min_chunk_size={self.min_chunk_size}, max_chunk_size={self.max_chunk_size}"
        )

    def _find_boundaries_by_pause(self, tokens: List[TokenData]) -> List[int]:
        """Return list of boundary indices (start indices of segments).

        Boundaries are token indices where a new segment starts. 0 is always a boundary.
        """
        if not tokens:
            return []

        boundaries = [0]
        for i in range(1, len(tokens)):
            gap = tokens[i].timestamp - tokens[i - 1].timestamp
            if gap >= self.pause_threshold:
                boundaries.append(i)

        return boundaries

    def _split_segment_by_size(self, tokens: List[TokenData]) -> List[List[TokenData]]:
        """Ensure segment lengths do not exceed `max_chunk_size`.

        Try to split at the largest internal gaps. If there are no gaps,
        fall back to regular size-based splits.
        """
        if len(tokens) <= self.max_chunk_size:
            return [tokens]

        # Compute internal gaps and their indices
        gaps = []  # (gap, index) where index is position after which gap occurs
        for i in range(1, len(tokens)):
            g = tokens[i].timestamp - tokens[i - 1].timestamp
            gaps.append((g, i))

        # Sort gaps descending
        gaps_sorted = sorted(gaps, key=lambda x: x[0], reverse=True)

        # We'll attempt to place splits at the largest gaps until sizes are satisfied
        split_points = set()
        segments = [tokens]

        # Greedy: choose largest gaps as split points until segments small enough
        for gap_value, gap_idx in gaps_sorted:
            # Add split and re-evaluate
            split_points.add(gap_idx)
            # Build segments from split points
            sorted_splits = sorted(split_points)
            new_segments = []
            prev = 0
            for sp in sorted_splits:
                new_segments.append(tokens[prev:sp])
                prev = sp
            new_segments.append(tokens[prev:])

            if all(len(s) <= self.max_chunk_size for s in new_segments):
                segments = new_segments
                break

        # If still too large, fall back to regular splits
        if any(len(s) > self.max_chunk_size for s in segments):
            segments = []
            for start in range(0, len(tokens), self.max_chunk_size):
                segments.append(tokens[start : min(start + self.max_chunk_size, len(tokens))])

        return segments

    def _merge_short_segments(self, segments: List[List[TokenData]]) -> List[List[TokenData]]:
        """Merge segments smaller than `min_chunk_size` with neighbors."""
        if not segments:
            return []

        merged = []
        i = 0
        while i < len(segments):
            seg = segments[i]
            if len(seg) < self.min_chunk_size:
                # Try merge with previous
                if merged:
                    merged[-1].extend(seg)
                else:
                    # Merge with next if possible
                    if i + 1 < len(segments):
                        segments[i + 1] = seg + segments[i + 1]
                    else:
                        # single small segment remaining, keep as is
                        merged.append(seg)
                i += 1
            else:
                merged.append(seg)
                i += 1

        # After merging, it's possible some segments exceed max size; re-split if needed
        final_segments = []
        for seg in merged:
            if len(seg) > self.max_chunk_size:
                final_segments.extend(self._split_segment_by_size(seg))
            else:
                final_segments.append(seg)

        return final_segments

    def create_chunks(self, transcript: VideoTranscriptFile) -> List[VideoChunk]:
        """Create pause-based chunks from a video transcript."""
        if not transcript.video_transcripts:
            logger.warning(f"No transcripts to chunk for {transcript.video_id}")
            return []

        tokens = transcript.video_transcripts

        # Find initial boundaries using pauses
        boundaries = self._find_boundaries_by_pause(tokens)

        # Build segments between boundaries
        raw_segments: List[List[TokenData]] = []
        for idx in range(len(boundaries)):
            start = boundaries[idx]
            end = boundaries[idx + 1] if idx + 1 < len(boundaries) else len(tokens)
            raw_segments.append(tokens[start:end])

        # Handle very long segments by splitting
        split_segments: List[List[TokenData]] = []
        for seg in raw_segments:
            if not seg:
                continue
            if len(seg) > self.max_chunk_size:
                split_segments.extend(self._split_segment_by_size(seg))
            else:
                split_segments.append(seg)

        # Merge very short segments
        final_segments = self._merge_short_segments(split_segments)

        # If no boundaries detected (uniform speech), fall back to regular splits
        if not final_segments:
            final_segments = []
            for start in range(0, len(tokens), self.max_chunk_size):
                final_segments.append(tokens[start : min(start + self.max_chunk_size, len(tokens))])

        # Create VideoChunk objects preserving timestamps and token ids
        chunks: List[VideoChunk] = []
        for idx, seg in enumerate(final_segments):
            if not seg:
                continue
            start_token_id = seg[0].id
            end_token_id = seg[-1].id
            start_timestamp = seg[0].timestamp
            end_timestamp = seg[-1].timestamp
            text = " ".join([t.word for t in seg])

            chunk = VideoChunk(
                chunk_id=f"{transcript.video_id}_chunk_{idx}",
                video_id=transcript.video_id,
                start_token_id=start_token_id,
                end_token_id=end_token_id,
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
                text=text,
            )
            chunks.append(chunk)

        logger.info(f"Created {len(chunks)} pause-based chunks from {transcript.video_id}")
        return chunks

    def chunk_all(self, transcripts: List[VideoTranscriptFile]) -> List[VideoChunk]:
        all_chunks: List[VideoChunk] = []
        for transcript in transcripts:
            all_chunks.extend(self.create_chunks(transcript))
        logger.info(f"Total pause-based chunks created: {len(all_chunks)}")
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
