# Video transcript loading and parsing

import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from src.models import VideoTranscriptFile, TokenData

logger = logging.getLogger(__name__)


class VideoTranscriptLoader:
    """
    Loads and validates video transcript JSON files.
    
    Expected JSON structure:
    {
        "video_id": "VIDEO_123",
        "title": "Setup Tutorial",
        "pdf_reference": "doc_001.pdf",
        "duration_seconds": 1200,
        "video_transcripts": [
            {"id": 1, "timestamp": 0.5, "word": "Hello"},
            ...
        ]
    }
    """
    
    def __init__(self, video_dir: Path):
        """
        Initialize loader with video directory.
        
        Args:
            video_dir: Path to directory containing JSON transcript files
        """
        self.video_dir = video_dir
        self.video_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"VideoTranscriptLoader initialized with directory: {video_dir}")
    
    def discover_files(self) -> List[Path]:
        """
        Discover all JSON files in the video directory.
        
        Returns:
            List of Path objects for JSON files
        """
        json_files = sorted(self.video_dir.glob("*.json"))
        logger.info(f"Discovered {len(json_files)} JSON transcript files")
        return json_files
    
    def load_file(self, file_path: Path) -> Optional[VideoTranscriptFile]:
        """
        Load and validate a single transcript file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            VideoTranscriptFile object if valid, None otherwise
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Validate structure
            transcript = VideoTranscriptFile(**data)
            logger.info(f"Loaded transcript: {transcript.video_id} with {len(transcript.video_transcripts)} tokens")
            return transcript
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {file_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            return None
    
    def load_all(self) -> List[VideoTranscriptFile]:
        """
        Load all transcript files from directory.
        
        Returns:
            List of VideoTranscriptFile objects
        """
        files = self.discover_files()
        transcripts = []
        
        for file_path in files:
            transcript = self.load_file(file_path)
            if transcript:
                transcripts.append(transcript)
        
        logger.info(f"Successfully loaded {len(transcripts)}/{len(files)} transcript files")
        return transcripts
    
    def validate_token_sequence(self, tokens: List[TokenData]) -> bool:
        """
        Validate token sequence for consistency.
        
        Checks:
        - Token IDs are strictly increasing
        - Timestamps are non-decreasing
        - All tokens have non-empty words
        
        Args:
            tokens: List of TokenData objects
            
        Returns:
            True if valid, False otherwise
        """
        if not tokens:
            return False
        
        prev_id = -1
        prev_timestamp = -1.0
        
        for i, token in enumerate(tokens):
            # Check ID sequence
            if token.id <= prev_id:
                logger.warning(f"Token ID not strictly increasing at index {i}: {token.id} <= {prev_id}")
                return False
            
            # Check timestamp sequence
            if token.timestamp < prev_timestamp:
                logger.warning(f"Timestamp decreased at index {i}: {token.timestamp} < {prev_timestamp}")
                return False
            
            # Check non-empty word
            if not token.word or not token.word.strip():
                logger.warning(f"Empty word at index {i}")
                return False
            
            prev_id = token.id
            prev_timestamp = token.timestamp
        
        return True
