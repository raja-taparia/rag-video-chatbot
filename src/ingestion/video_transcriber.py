# YouTube and Video File Ingestion with Transcription

import logging
import json
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
import subprocess

logger = logging.getLogger("ingestion")


class VideoTranscriber:
    """
    Transcribes videos from YouTube links and local video files.
    
    Supports:
    - YouTube video links (via yt-dlp and Whisper)
    - Local video files (MP4, WebM, etc.)
    - Outputs JSON format compatible with VideoTranscriptFile
    """
    
    def __init__(self, videos_input_dir: Path, output_dir: Path):
        """
        Initialize transcriber.
        
        Args:
            videos_input_dir: Directory containing local video files
            output_dir: Directory where JSON transcripts will be saved
        """
        self.videos_input_dir = Path(videos_input_dir)
        self.output_dir = Path(output_dir)
        
        self.videos_input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"VideoTranscriber initialized: {videos_input_dir} → {output_dir}")
    
    def transcribe_from_youtube(self, youtube_url: str) -> Optional[Dict[str, Any]]:
        """
        Transcribe YouTube video using yt-dlp and Whisper.
        
        Requirements:
        - yt-dlp: pip install yt-dlp
        - openai-whisper: pip install openai-whisper
        
        Args:
            youtube_url: Full YouTube URL or video ID
            
        Returns:
            Dictionary with video_id, transcript data, or None if failed
        """
        try:
            logger.info(f"Transcribing YouTube video: {youtube_url}")
            
            # Download video info using yt-dlp
            logger.debug("Downloading video metadata with yt-dlp...")
            cmd_info = [
                "yt-dlp",
                "-j",  # JSON output
                "--no-warnings",
                youtube_url
            ]
            
            result = subprocess.run(cmd_info, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                logger.error(f"yt-dlp failed: {result.stderr}")
                return None
            
            video_info = json.loads(result.stdout)
            video_id = f"VIDEO_YOUTUBE_{video_info['id'][:11]}"
            title = video_info.get('title', 'Unknown Title')
            duration = video_info.get('duration', 0)
            
            logger.info(f"Video info: {video_id}, Title: {title}, Duration: {duration}s")
            
            # Download audio
            logger.debug("Downloading audio...")
            audio_file = f"/tmp/{video_id}_audio.mp3"
            cmd_download = [
                "yt-dlp",
                "-f", "bestaudio[ext=m4a]/bestaudio",
                "-o", audio_file,
                "--extract-audio",
                "--audio-format", "mp3",
                youtube_url
            ]
            
            result = subprocess.run(cmd_download, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                logger.error(f"Audio download failed: {result.stderr}")
                return None
            
            # Transcribe with Whisper
            logger.debug("Transcribing with Whisper...")
            cmd_transcribe = [
                "whisper",
                audio_file,
                "--language", "en",
                "--output_format", "json",
                "--output_dir", "/tmp"
            ]
            
            result = subprocess.run(cmd_transcribe, capture_output=True, text=True, timeout=600)
            
            if result.returncode != 0:
                logger.error(f"Whisper transcription failed: {result.stderr}")
                return None
            
            # Parse transcript
            transcript_file = audio_file.replace(".mp3", ".json")
            with open(transcript_file, 'r') as f:
                whisper_output = json.load(f)
            
            # Convert Whisper output to our format
            logger.debug("Converting transcript format...")
            tokens = []
            token_id = 1
            
            for segment in whisper_output.get('segments', []):
                words = segment['text'].split()
                start_time = segment['start']
                
                # Distribute words across segment time
                for i, word in enumerate(words):
                    # Simple: assume uniform time distribution
                    timestamp = start_time + (i / len(words)) * (segment['end'] - segment['start'])
                    tokens.append({
                        'id': token_id,
                        'timestamp': round(timestamp, 2),
                        'word': word
                    })
                    token_id += 1
            
            # Create transcript object
            transcript_data = {
                'video_id': video_id,
                'title': title,
                'source_url': youtube_url,
                'duration_seconds': duration,
                'video_transcripts': tokens
            }
            
            logger.info(f"Transcription complete: {len(tokens)} tokens extracted")
            
            # Cleanup
            os.remove(audio_file)
            os.remove(transcript_file)
            
            return transcript_data
            
        except subprocess.TimeoutExpired:
            logger.error("Transcription timeout - video too long or network issue")
            return None
        except Exception as e:
            logger.error(f"Error transcribing YouTube video: {e}", exc_info=True)
            return None
    
    def transcribe_local_video(self, video_file: Path) -> Optional[Dict[str, Any]]:
        """
        Transcribe local video file using Whisper.
        
        Supported formats: MP4, WebM, AVI, MOV, MKV, etc.
        
        Args:
            video_file: Path to local video file
            
        Returns:
            Dictionary with video_id, transcript data, or None if failed
        """
        try:
            logger.info(f"Transcribing local video: {video_file}")
            
            if not video_file.exists():
                logger.error(f"Video file not found: {video_file}")
                return None
            
            # Get video info using ffprobe
            logger.debug("Getting video metadata with ffprobe...")
            cmd_probe = [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1:noprint_wrappers=1",
                str(video_file)
            ]
            
            result = subprocess.run(cmd_probe, capture_output=True, text=True, timeout=10)
            duration = float(result.stdout.strip()) if result.stdout.strip() else 0
            
            video_id = f"VIDEO_LOCAL_{video_file.stem.upper()[:10]}"
            title = video_file.stem
            
            logger.info(f"Video info: {video_id}, Title: {title}, Duration: {duration}s")
            
            # Extract audio using ffmpeg
            logger.debug("Extracting audio with ffmpeg...")
            audio_file = f"/tmp/{video_id}_audio.mp3"
            cmd_extract = [
                "ffmpeg",
                "-i", str(video_file),
                "-q:a", "9",
                "-n",
                audio_file
            ]
            
            result = subprocess.run(cmd_extract, capture_output=True, timeout=300)
            
            if result.returncode != 0:
                logger.error(f"Audio extraction failed")
                return None
            
            # Transcribe with Whisper
            logger.debug("Transcribing with Whisper...")
            cmd_transcribe = [
                "whisper",
                audio_file,
                "--language", "en",
                "--output_format", "json",
                "--output_dir", "/tmp"
            ]
            
            result = subprocess.run(cmd_transcribe, capture_output=True, text=True, timeout=600)
            
            if result.returncode != 0:
                logger.error(f"Whisper transcription failed")
                return None
            
            # Parse transcript (same as YouTube)
            transcript_file = audio_file.replace(".mp3", ".json")
            with open(transcript_file, 'r') as f:
                whisper_output = json.load(f)
            
            tokens = []
            token_id = 1
            
            for segment in whisper_output.get('segments', []):
                words = segment['text'].split()
                start_time = segment['start']
                
                for i, word in enumerate(words):
                    timestamp = start_time + (i / len(words)) * (segment['end'] - segment['start'])
                    tokens.append({
                        'id': token_id,
                        'timestamp': round(timestamp, 2),
                        'word': word
                    })
                    token_id += 1
            
            transcript_data = {
                'video_id': video_id,
                'title': title,
                'source_file': str(video_file),
                'duration_seconds': duration,
                'video_transcripts': tokens
            }
            
            logger.info(f"Transcription complete: {len(tokens)} tokens extracted")
            
            # Cleanup
            os.remove(audio_file)
            os.remove(transcript_file)
            
            return transcript_data
            
        except subprocess.TimeoutExpired:
            logger.error("Transcription timeout")
            return None
        except Exception as e:
            logger.error(f"Error transcribing local video: {e}", exc_info=True)
            return None
    
    def process_video_links_file(self, links_file: Path) -> List[Dict[str, Any]]:
        """
        Process video_links.txt file containing YouTube URLs.
        
        File format (one URL per line):
        https://www.youtube.com/watch?v=...
        https://www.youtube.com/watch?v=...
        
        Args:
            links_file: Path to video_links.txt
            
        Returns:
            List of transcript data dictionaries
        """
        try:
            logger.info(f"Processing video links file: {links_file}")
            
            if not links_file.exists():
                logger.warning(f"Links file not found: {links_file}")
                return []
            
            transcripts = []
            
            with open(links_file, 'r') as f:
                lines = f.readlines()
            
            logger.info(f"Found {len(lines)} video links")
            
            for i, line in enumerate(lines, 1):
                url = line.strip()
                
                if not url or url.startswith('#'):  # Skip empty lines and comments
                    continue
                
                logger.info(f"[{i}] Processing: {url[:50]}...")
                transcript = self.transcribe_from_youtube(url)
                
                if transcript:
                    # Save to JSON
                    output_file = self.output_dir / f"{transcript['video_id'].lower()}.json"
                    with open(output_file, 'w') as f:
                        json.dump(transcript, f, indent=2)
                    
                    logger.info(f"Saved transcript to: {output_file}")
                    transcripts.append(transcript)
                else:
                    logger.warning(f"Failed to transcribe: {url}")
            
            logger.info(f"Successfully processed {len(transcripts)} videos")
            return transcripts
            
        except Exception as e:
            logger.error(f"Error processing links file: {e}", exc_info=True)
            return []
    
    def process_local_videos_folder(self) -> List[Dict[str, Any]]:
        """
        Process all video files in videos_input folder.
        
        Supported formats: MP4, WebM, AVI, MOV, MKV
        
        Returns:
            List of transcript data dictionaries
        """
        try:
            logger.info(f"Processing local videos folder: {self.videos_input_dir}")
            
            video_extensions = {'.mp4', '.webm', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
            video_files = [f for f in self.videos_input_dir.iterdir() 
                          if f.suffix.lower() in video_extensions]
            
            logger.info(f"Found {len(video_files)} video files")
            
            transcripts = []
            
            for video_file in video_files:
                logger.info(f"Processing: {video_file.name}")
                transcript = self.transcribe_local_video(video_file)
                
                if transcript:
                    # Save to JSON
                    output_file = self.output_dir / f"{transcript['video_id'].lower()}.json"
                    with open(output_file, 'w') as f:
                        json.dump(transcript, f, indent=2)
                    
                    logger.info(f"Saved transcript to: {output_file}")
                    transcripts.append(transcript)
                else:
                    logger.warning(f"Failed to transcribe: {video_file}")
            
            logger.info(f"Successfully processed {len(transcripts)} videos")
            return transcripts
            
        except Exception as e:
            logger.error(f"Error processing videos folder: {e}", exc_info=True)
            return []
