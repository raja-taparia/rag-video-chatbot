# Vector store operations using Qdrant

import logging
from typing import List, Dict, Optional, Any
import json
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
import uuid

logger = logging.getLogger(__name__)


class QdrantVectorStore:
    """
    Manages vector storage and retrieval using Qdrant.
    
    Features:
    - Separate collections for video chunks and PDF paragraphs
    - Metadata filtering for source tracking
    - Semantic search via vector similarity
    - Scalable to millions of vectors
    """
    
    VIDEO_COLLECTION = "video_transcripts"
    PDF_COLLECTION = "pdf_documents"
    
    def __init__(self, host: str, port: int, embedding_dim: int = 768):
        """
        Initialize Qdrant client.
        
        Args:
            host: Qdrant server hostname
            port: Qdrant server port
            embedding_dim: Dimension of embedding vectors
        """
        try:
            self.client = QdrantClient(host=host, port=port)
            self.embedding_dim = embedding_dim
            logger.info(f"Connected to Qdrant at {host}:{port}")
            
            # Ensure collections exist
            self._ensure_collections()
            
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {e}")
            raise
    
    def _ensure_collections(self) -> None:
        """Create collections if they don't exist."""
        collections = self.client.get_collections()
        existing_names = [c.name for c in collections.collections]
        
        for collection_name in [self.VIDEO_COLLECTION, self.PDF_COLLECTION]:
            if collection_name not in existing_names:
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created collection: {collection_name}")
    
    def index_video_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Index video chunks into Qdrant.
        
        Expected chunk structure:
        {
            'chunk_id': str,
            'video_id': str,
            'start_token_id': int,
            'end_token_id': int,
            'start_timestamp': float,
            'end_timestamp': float,
            'text': str,
            'embedding': List[float]
        }
        
        Args:
            chunks: List of video chunk dictionaries
            
        Returns:
            Number of chunks indexed
        """
        points = []
        
        for chunk in chunks:
            if not chunk.get('embedding'):
                logger.warning(f"Skipping chunk {chunk['chunk_id']} - no embedding")
                continue
            
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=chunk['embedding'],
                payload={
                    'chunk_id': chunk['chunk_id'],
                    'video_id': chunk['video_id'],
                    'start_token_id': chunk['start_token_id'],
                    'end_token_id': chunk['end_token_id'],
                    'start_timestamp': chunk['start_timestamp'],
                    'end_timestamp': chunk['end_timestamp'],
                    'text': chunk['text'],
                    'source_type': 'video',
                }
            )
            points.append(point)
        
        if points:
            self.client.upsert(
                collection_name=self.VIDEO_COLLECTION,
                points=points
            )
            logger.info(f"Indexed {len(points)} video chunks")
        
        return len(points)
    
    def index_pdf_paragraphs(self, paragraphs: List[Dict[str, Any]]) -> int:
        """
        Index PDF paragraphs into Qdrant.
        
        Expected paragraph structure:
        {
            'para_id': str,
            'pdf_filename': str,
            'page_number': int,
            'paragraph_index': int,
            'text': str,
            'embedding': List[float]
        }
        
        Args:
            paragraphs: List of paragraph dictionaries
            
        Returns:
            Number of paragraphs indexed
        """
        points = []
        
        for para in paragraphs:
            if not para.get('embedding'):
                logger.warning(f"Skipping paragraph {para['para_id']} - no embedding")
                continue
            
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=para['embedding'],
                payload={
                    'para_id': para['para_id'],
                    'pdf_filename': para['pdf_filename'],
                    'page_number': para['page_number'],
                    'paragraph_index': para['paragraph_index'],
                    'text': para['text'],
                    'source_type': 'pdf',
                }
            )
            points.append(point)
        
        if points:
            self.client.upsert(
                collection_name=self.PDF_COLLECTION,
                points=points
            )
            logger.info(f"Indexed {len(points)} PDF paragraphs")
        
        return len(points)
    
    def search_video(
        self,
        query_embedding: List[float],
        top_k: int = 3,
        threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Search for similar video chunks.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            threshold: Minimum similarity score (0.0-1.0)
            
        Returns:
            List of relevant chunks with scores
        """
        try:
            results = self.client.search(
                collection_name=self.VIDEO_COLLECTION,
                query_vector=query_embedding,
                limit=top_k,
                score_threshold=threshold,
            )
            
            output = []
            for result in results:
                output.append({
                    'chunk_id': result.payload['chunk_id'],
                    'video_id': result.payload['video_id'],
                    'start_token_id': result.payload['start_token_id'],
                    'end_token_id': result.payload['end_token_id'],
                    'start_timestamp': result.payload['start_timestamp'],
                    'end_timestamp': result.payload['end_timestamp'],
                    'text': result.payload['text'],
                    'score': result.score,
                })
            
            logger.info(f"Found {len(output)} relevant video chunks (threshold={threshold})")
            return output
            
        except Exception as e:
            logger.error(f"Error searching video collection: {e}")
            return []
    
    def search_pdf(
        self,
        query_embedding: List[float],
        top_k: int = 3,
        threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Search for similar PDF paragraphs.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            threshold: Minimum similarity score
            
        Returns:
            List of relevant paragraphs with scores
        """
        try:
            results = self.client.search(
                collection_name=self.PDF_COLLECTION,
                query_vector=query_embedding,
                limit=top_k,
                score_threshold=threshold,
            )
            
            output = []
            for result in results:
                output.append({
                    'para_id': result.payload['para_id'],
                    'pdf_filename': result.payload['pdf_filename'],
                    'page_number': result.payload['page_number'],
                    'paragraph_index': result.payload['paragraph_index'],
                    'text': result.payload['text'],
                    'score': result.score,
                })
            
            logger.info(f"Found {len(output)} relevant PDF paragraphs (threshold={threshold})")
            return output
            
        except Exception as e:
            logger.error(f"Error searching PDF collection: {e}")
            return []
    
    def clear_all(self) -> None:
        """Clear all collections (useful for reindexing)."""
        try:
            self.client.delete_collection(self.VIDEO_COLLECTION)
            self.client.delete_collection(self.PDF_COLLECTION)
            self._ensure_collections()
            logger.info("Cleared all collections")
        except Exception as e:
            logger.error(f"Error clearing collections: {e}")
