# PDF paragraph retrieval with ranking

import logging
from typing import List, Dict, Any, Optional
from src.retrieval.ranking import RankingStrategy, CosineSimilarityRanking

logger = logging.getLogger("retrieval")


class PDFRetriever:
    """
    Retrieves relevant PDF paragraphs from vector store.
    
    Performs semantic search over PDF content and returns
    top-k paragraphs ranked by relevance with page citations.
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
        logger.info("PDFRetriever initialized")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        threshold: float = 0.6
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant PDF paragraphs for query.
        
        Process:
        1. Generate embedding for query
        2. Search vector store for similar PDF paragraphs
        3. Filter by relevance threshold
        4. Rank results by score
        
        Args:
            query: User question or search query
            top_k: Number of results to return
            threshold: Minimum relevance score (0.0-1.0, typically lower than video)
            
        Returns:
            List of relevant paragraphs with citations, sorted by score
            
        Example return:
        [
            {
                'para_id': 'doc_001_p12_para3',
                'pdf_filename': 'kubernetes_guide.pdf',
                'page_number': 12,
                'paragraph_index': 3,
                'text': 'Kubernetes is a container orchestration platform...',
                'score': 0.75
            }
        ]
        """
        try:
            logger.info(f"Retrieving PDF paragraphs for query: {query[:50]}...")
            
            # Step 1: Generate query embedding
            query_embedding = self.embedder.embed_text(query)
            if not query_embedding:
                logger.warning(f"Failed to embed query: {query}")
                return []
            
            logger.debug(f"Generated query embedding (dim={len(query_embedding)})")
            
            # Step 2: Search vector store
            logger.debug(f"Searching PDFs with threshold={threshold}, top_k={top_k}")
            results = self.vector_store.search_pdf(
                query_embedding=query_embedding,
                top_k=top_k,
                threshold=threshold
            )
            
            # Step 3: Rank results
            ranked = self.ranking_strategy.rank(results)
            
            logger.info(f"Retrieved {len(ranked)} relevant PDF paragraphs (threshold={threshold})")
            
            # Log results for debugging
            for i, result in enumerate(ranked):
                logger.debug(f"  [{i}] score={result.get('score', 0):.3f}, "
                           f"pdf={result.get('pdf_filename')}, "
                           f"page={result.get('page_number')}, "
                           f"para={result.get('paragraph_index')}")
            
            return ranked
            
        except Exception as e:
            logger.error(f"Error retrieving PDF paragraphs: {e}", exc_info=True)
            return []
    
    def retrieve_by_pdf(
        self,
        pdf_filename: str,
        top_k: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Retrieve all paragraphs from a specific PDF.
        
        Args:
            pdf_filename: Name of PDF file to retrieve from
            top_k: Maximum paragraphs to return
            
        Returns:
            List of all paragraphs from this PDF, sorted by page and paragraph index
        """
        try:
            logger.info(f"Retrieving all paragraphs from PDF: {pdf_filename}")
            
            # Create dummy query for this PDF
            dummy_query = f"content from {pdf_filename}"
            query_embedding = self.embedder.embed_text(dummy_query)
            
            if not query_embedding:
                logger.warning(f"Failed to create query for PDF {pdf_filename}")
                return []
            
            # Search with low threshold to get all
            results = self.vector_store.search_pdf(
                query_embedding=query_embedding,
                top_k=top_k,
                threshold=0.0
            )
            
            # Filter to only this PDF
            filtered = [r for r in results if r.get('pdf_filename') == pdf_filename]
            
            # Sort by page and paragraph index
            sorted_results = sorted(
                filtered,
                key=lambda x: (x.get('page_number', 0), x.get('paragraph_index', 0))
            )
            
            logger.info(f"Retrieved {len(sorted_results)} paragraphs from {pdf_filename}")
            return sorted_results
            
        except Exception as e:
            logger.error(f"Error retrieving paragraphs by PDF: {e}", exc_info=True)
            return []
    
    def retrieve_by_page(
        self,
        pdf_filename: str,
        page_number: int
    ) -> List[Dict[str, Any]]:
        """
        Retrieve all paragraphs from a specific page of a PDF.
        
        Args:
            pdf_filename: Name of PDF file
            page_number: Page number (1-indexed)
            
        Returns:
            All paragraphs from that page, sorted by index
        """
        try:
            logger.info(f"Retrieving page {page_number} from {pdf_filename}")
            
            # Get all paragraphs from this PDF
            all_paragraphs = self.retrieve_by_pdf(pdf_filename)
            
            # Filter to specific page
            page_paragraphs = [
                p for p in all_paragraphs
                if p.get('page_number') == page_number
            ]
            
            logger.info(f"Retrieved {len(page_paragraphs)} paragraphs from page {page_number}")
            return page_paragraphs
            
        except Exception as e:
            logger.error(f"Error retrieving page {page_number}: {e}", exc_info=True)
            return []
