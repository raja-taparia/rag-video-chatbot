# Main RAG pipeline orchestration

import logging
import time
from typing import Dict, Any, Optional, List
import json

from src.config import Config
from src.ingestion.video_loader import VideoTranscriptLoader
from src.ingestion.pdf_loader import PDFLoader
from src.ingestion.chunker import VideoChunker, PDFChunker
from src.indexing.embeddings import get_default_embedder
from src.indexing.vector_store import QdrantVectorStore
from src.retrieval.video_retriever import VideoRetriever
from src.retrieval.pdf_retriever import PDFRetriever
from src.generation.answer_generator import AnswerGenerator
from src.models import VideoAnswer, PDFAnswer, NoAnswer, RAGResponse

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    End-to-end RAG pipeline orchestrating ingestion, indexing, retrieval, and generation.
    """
    
    def __init__(self, config: Config):
        """
        Initialize RAG pipeline with configuration.
        
        Args:
            config: Config object with all settings
        """
        self.config = config
        
        # Initialize components
        logger.info("Initializing RAG pipeline components...")
        
        # Use the default embedder (sentence-transformers by default);
        # falls back to Ollama if sentence-transformers is not available.
        self.embedder = get_default_embedder(config.ollama)
        # Ensure the vector store expects the embedding dimensionality we will
        # produce (sentence-transformers `all-MiniLM-L6-v2` -> 384 dims by default).
        self.vector_store = QdrantVectorStore(
            host=config.qdrant.host,
            port=config.qdrant.port,
            embedding_dim=config.ollama.embedding_dim
        )
        
        self.video_loader = VideoTranscriptLoader(config.data.video_dir)
        self.pdf_loader = PDFLoader(config.data.pdf_dir)
        
        self.video_chunker = VideoChunker(
            chunk_size=config.rag.chunk_size,
            chunk_overlap=config.rag.chunk_overlap
        )
        
        self.video_retriever = VideoRetriever(self.vector_store, self.embedder)
        self.pdf_retriever = PDFRetriever(self.vector_store, self.embedder)
        self.answer_generator = AnswerGenerator(config.ollama)
        
        logger.info("RAG pipeline initialized successfully")
    
    def index_data(self, full_reindex: bool = False) -> Dict[str, int]:
        """
        Index all available data (videos and PDFs).
        
        Args:
            full_reindex: If True, clear and rebuild all indices
            
        Returns:
            Dictionary with indexing statistics
        """
        logger.info("Starting data indexing...")
        start_time = time.time()
        
        stats = {
            'videos_loaded': 0,
            'video_chunks_created': 0,
            'video_chunks_indexed': 0,
            'pdfs_loaded': 0,
            'pdf_paragraphs_extracted': 0,
            'pdf_paragraphs_indexed': 0,
            'total_embeddings_generated': 0,
            'indexing_time_seconds': 0
        }
        
        # Clear if full reindex
        if full_reindex:
            logger.info("Clearing existing indices for full reindex...")
            self.vector_store.clear_all()
        
        # Index video transcripts
        logger.info("Indexing video transcripts...")
        video_stats = self._index_videos()
        stats.update(video_stats)
        
        # Index PDF documents
        logger.info("Indexing PDF documents...")
        pdf_stats = self._index_pdfs()
        stats.update(pdf_stats)
        
        stats['indexing_time_seconds'] = time.time() - start_time
        
        logger.info(f"Indexing complete: {stats}")
        return stats
    
    def _index_videos(self) -> Dict[str, int]:
        """Index all video transcripts."""
        stats = {
            'videos_loaded': 0,
            'video_chunks_created': 0,
            'video_chunks_indexed': 0,
            'total_embeddings_generated': 0,
        }
        
        # Load transcripts
        transcripts = self.video_loader.load_all()
        stats['videos_loaded'] = len(transcripts)
        
        if not transcripts:
            logger.warning("No video transcripts found")
            return stats
        
        # Create chunks
        chunks = self.video_chunker.chunk_all(transcripts)
        stats['video_chunks_created'] = len(chunks)
        
        if not chunks:
            logger.warning("No chunks created from transcripts")
            return stats
        
        # Generate embeddings
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedder.embed_batch(texts)
        stats['total_embeddings_generated'] += len([e for e in embeddings if e])
        
        # Convert to dictionaries with embeddings
        chunk_dicts = []
        for chunk, embedding in zip(chunks, embeddings):
            if embedding:
                # `VideoChunk` is a Pydantic `BaseModel`.
                # Use `model_dump()` (Pydantic v2) to convert to a dictionary representation.
                chunk_dict = chunk.model_dump()
                chunk_dict['embedding'] = embedding
                chunk_dicts.append(chunk_dict)
        
        # Index in vector store
        indexed_count = self.vector_store.index_video_chunks(chunk_dicts)
        stats['video_chunks_indexed'] = indexed_count
        
        return stats
    
    def _index_pdfs(self) -> Dict[str, int]:
        """Index all PDF documents."""
        stats = {
            'pdfs_loaded': 0,
            'pdf_paragraphs_extracted': 0,
            'pdf_paragraphs_indexed': 0,
            'total_embeddings_generated': 0,
        }
        
        # Load PDFs
        all_paragraphs = self.pdf_loader.load_all()
        
        if not all_paragraphs:
            logger.warning("No PDFs found")
            return stats
        
        stats['pdf_paragraphs_extracted'] = len(all_paragraphs)
        
        # Generate embeddings for paragraphs
        texts = [para[3] for para in all_paragraphs]  # text is 4th element
        embeddings = self.embedder.embed_batch(texts)
        stats['total_embeddings_generated'] = len([e for e in embeddings if e])
        
        # Convert to dictionaries with embeddings
        para_dicts = []
        for (pdf_name, page_num, para_idx, text), embedding in zip(all_paragraphs, embeddings):
            if embedding:
                para_dict = {
                    'para_id': f"{pdf_name}_p{page_num}_para{para_idx}",
                    'pdf_filename': pdf_name,
                    'page_number': page_num,
                    'paragraph_index': para_idx,
                    'text': text,
                    'embedding': embedding,
                }
                para_dicts.append(para_dict)
        
        # Index in vector store
        indexed_count = self.vector_store.index_pdf_paragraphs(para_dicts)
        stats['pdf_paragraphs_indexed'] = indexed_count
        
        return stats
    
    def query(self, question: str) -> RAGResponse:
        """
        Execute complete RAG pipeline for a user question.
        
        Process:
        1. Retrieve from video transcripts (primary)
        2. If insufficient, retrieve from PDFs (fallback)
        3. Generate answer using LLM
        
        Args:
            question: User question
            
        Returns:
            RAGResponse with answer and metadata
        """
        start_time = time.time()
        logger.info(f"Processing query: {question}")
        
        # Try video retrieval first
        logger.info("Retrieving from video transcripts...")
        video_results = self.video_retriever.retrieve(
            query=question,
            top_k=self.config.rag.top_k_video,
            threshold=self.config.rag.video_relevance_threshold
        )
        
        if video_results:
            logger.info(f"Found {len(video_results)} relevant video segments")
            answer = self.answer_generator.generate_from_video(question, video_results)
            
            if answer:
                processing_time = (time.time() - start_time) * 1000
                return RAGResponse(
                    query=question,
                    response=answer.model_dump(),
                    processing_time_ms=processing_time,
                    llm_model=self.config.ollama.llm_model
                )
        
        # Fallback to PDF retrieval
        logger.info("Retrieving from PDF documents...")
        pdf_results = self.pdf_retriever.retrieve(
            query=question,
            top_k=self.config.rag.top_k_pdf,
            threshold=self.config.rag.pdf_relevance_threshold
        )
        
        if pdf_results:
            logger.info(f"Found {len(pdf_results)} relevant PDF segments")
            answer = self.answer_generator.generate_from_pdf(question, pdf_results)
            
            if answer:
                processing_time = (time.time() - start_time) * 1000
                return RAGResponse(
                    query=question,
                    response=answer.model_dump(),
                    processing_time_ms=processing_time,
                    llm_model=self.config.ollama.llm_model
                )
        
        # No answer found
        logger.info("No relevant answer found in available sources")
        processing_time = (time.time() - start_time) * 1000
        
        no_answer = NoAnswer(
            message="No relevant answer found in available sources",
            suggestions="Try rephrasing your question or check if the topic is covered in the documentation"
        )
        
        return RAGResponse(
            query=question,
            response=no_answer.model_dump(),
            processing_time_ms=processing_time
        )
