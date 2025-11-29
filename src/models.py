# Data models using Pydantic for type safety and validation

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class TokenData(BaseModel):
    """Single token from video transcript"""
    id: int = Field(..., description="Token index")
    timestamp: float = Field(..., description="Time in seconds from video start")
    word: str = Field(..., description="The word token")


class VideoTranscriptFile(BaseModel):
    """Complete video transcript JSON structure"""
    video_id: str = Field(..., description="Unique video identifier")
    title: Optional[str] = Field(None, description="Video title")
    pdf_reference: Optional[str] = Field(None, description="Linked PDF filename")
    duration_seconds: Optional[float] = Field(None, description="Video duration")
    video_transcripts: List[TokenData] = Field(..., description="List of tokens")


class VideoChunk(BaseModel):
    """Semantic chunk from video transcript"""
    chunk_id: str = Field(..., description="Unique chunk identifier")
    video_id: str = Field(..., description="Source video ID")
    start_token_id: int = Field(..., description="First token index in chunk")
    end_token_id: int = Field(..., description="Last token index in chunk")
    start_timestamp: float = Field(..., description="Start time in seconds")
    end_timestamp: float = Field(..., description="End time in seconds")
    text: str = Field(..., description="Joined text of tokens")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding")
    created_at: datetime = Field(default_factory=datetime.utcnow)


class PDFParagraph(BaseModel):
    """Paragraph extracted from PDF"""
    para_id: str = Field(..., description="Unique paragraph identifier")
    pdf_filename: str = Field(..., description="Source PDF filename")
    page_number: int = Field(..., description="Page number (1-indexed)")
    paragraph_index: int = Field(..., description="Paragraph index within page")
    text: str = Field(..., description="Paragraph text content")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding")
    created_at: datetime = Field(default_factory=datetime.utcnow)


class VideoAnswer(BaseModel):
    """Answer sourced from video transcript"""
    source_type: str = Field(default="video")
    video_id: str
    title: Optional[str] = None
    answer_text: Optional[str] = None
    segments: List[dict] = Field(default_factory=list)
    pdf_reference: Optional[str] = None
    confidence: float


class PDFAnswer(BaseModel):
    """Answer sourced from PDF"""
    source_type: str = Field(default="pdf")
    pdf_filename: str
    page_number: int
    paragraph_index: int
    answer_text: str
    source_snippet: str
    confidence: float


class NoAnswer(BaseModel):
    """No relevant answer found"""
    source_type: str = Field(default="none")
    message: str = Field(default="No relevant answer found in available sources")
    suggestions: Optional[str] = None


class RAGResponse(BaseModel):
    """Response from RAG pipeline"""
    query: str
    response: dict  # VideoAnswer or PDFAnswer or NoAnswer
    processing_time_ms: float
    model_used: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
