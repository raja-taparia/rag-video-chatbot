# Video chunk retrieval with ranking

import logging
from typing import List, Dict, Any, Optional
from src.retrieval.ranking import RankingStrategy, CosineSimilarityRanking

logger = logging.getLogger("retrieval")


class VideoRetriever:
    """
    Retrieves relevant video chunks from vector store.
    
    Performs semantic search over video transcripts and returns
    top-k results ranked by relevance score.
    """
    
    def __init__(self, vector_store, embedder, ranking_strategy: Optional[RankingStrategy] = None):
        """
        Initialize retriever.
        
        Args:
            vector_store: QdrantVectorStore instance for vector operations
            embedder: OllamaEmbedder instance for generating embeddings
            ranking_strategy: Optional custom ranking strategy (default: cosine similarity)
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.ranking_strategy = ranking_strategy or CosineSimilarityRanking()
        logger.info("VideoRetriever initialized")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant video chunks for query.
        
        Process:
        1. Generate embedding for query
        2. Search vector store for similar video chunks
        3. Filter by relevance threshold
        4. Rank results
        
        Args:
            query: User question or search query
            top_k: Number of results to return
            threshold: Minimum relevance score (0.0-1.0)
            
        Returns:
            List of relevant chunks with metadata, sorted by score
            
        Example return:
        [
            {
                'chunk_id': 'VIDEO_K8S_chunk_0',
                'video_id': 'VIDEO_K8S',
                'start_token_id': 10,
                'end_token_id': 120,
                'start_timestamp': 5.2,
                'end_timestamp': 45.8,
                'text': 'To setup Kubernetes first install Docker...',
                'score': 0.82
            }
        ]
        """
        try:
            logger.info(f"Retrieving video chunks for query: {query[:50]}...")
            
            # Step 1: Generate query embedding
            query_embedding = self.embedder.embed_text(query)
            if not query_embedding:
                logger.warning(f"Failed to embed query: {query}")
                return []
            
            logger.debug(f"Generated query embedding (dim={len(query_embedding)})")
            
            # Step 2: Search vector store
            logger.debug(f"Searching with threshold={threshold}, top_k={top_k}")
            results = self.vector_store.search_video(
                query_embedding=query_embedding,
                top_k=top_k,
                threshold=threshold
            )
            
            # Step 3: Rank results
            ranked = self.ranking_strategy.rank(results)
            
            logger.info(f"Retrieved {len(ranked)} relevant video chunks (threshold={threshold})")
            
            # Log results for debugging
            for i, result in enumerate(ranked):
                logger.debug(f"  [{i}] score={result.get('score', 0):.3f}, "
                           f"video={result.get('video_id')}, "
                           f"timestamp={result.get('start_timestamp')}-{result.get('end_timestamp')}")
            
            return ranked
            
        except Exception as e:
            logger.error(f"Error retrieving video chunks: {e}", exc_info=True)
            return []
    
    def retrieve_by_video_id(
        self,
        video_id: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Retrieve all chunks from a specific video.
        
        Args:
            video_id: Specific video to retrieve from
            top_k: Maximum chunks to return
            
        Returns:
            List of all chunks from this video, sorted by timestamp
        """
        try:
            logger.info(f"Retrieving all chunks from video: {video_id}")
            
            # Create query embedding from video_id (dummy query)
            dummy_query = f"content from video {video_id}"
            query_embedding = self.embedder.embed_text(dummy_query)
            
            if not query_embedding:
                logger.warning(f"Failed to create query for video {video_id}")
                return []
            
            # Search with low threshold to get all
            results = self.vector_store.search_video(
                query_embedding=query_embedding,
                top_k=top_k,
                threshold=0.0  # Get everything
            )
            
            # Filter to only this video
            filtered = [r for r in results if r.get('video_id') == video_id]
            
            # Sort by timestamp
            sorted_results = sorted(filtered, key=lambda x: x.get('start_timestamp', 0))
            
            logger.info(f"Retrieved {len(sorted_results)} chunks from {video_id}")
            return sorted_results
            
        except Exception as e:
            logger.error(f"Error retrieving chunks by video_id: {e}", exc_info=True)
            return []
