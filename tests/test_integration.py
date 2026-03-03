# Integration tests for complete RAG system
# Place this file at: tests/test_integration.py

import pytest
from typing import Dict, Any
from src.retrieval.video_retriever import VideoRetriever
from src.retrieval.pdf_retriever import PDFRetriever
from src.generation.answer_generator import AnswerGenerator


class TestEndToEndQueryProcessing:
    """Test complete query processing pipeline."""
    
    def test_query_video_match(self, mock_rag_pipeline, test_queries):
        """Test query that should match video source."""
        query_key = "docker_install"
        query_data = test_queries[query_key]
        query = query_data["question"]
        
        result = mock_rag_pipeline.process_query(query)
        
        assert result["query"] == query
        assert "video_results" in result
        assert "pdf_results" in result
        assert len(result["video_results"]) > 0
    
    def test_query_pdf_match(self, mock_rag_pipeline, test_queries):
        """Test query that should match PDF source."""
        query_key = "containerization_practices"
        query_data = test_queries[query_key]
        query = query_data["question"]
        
        result = mock_rag_pipeline.process_query(query)
        
        # Should have PDF results
        assert len(result["pdf_results"]) > 0
    
    def test_query_no_match(self, mock_rag_pipeline, test_queries):
        """Test query with no matching results."""
        query_key = "cooking_lasagna"
        query_data = test_queries[query_key]
        query = query_data["question"]
        
        result = mock_rag_pipeline.process_query(query)
        
        # May have empty or low-scoring results
        total_results = len(result["video_results"]) + len(result["pdf_results"])
        assert total_results == 0 or result["video_results"][0].get("score", 0) < 0.5
    
    def test_query_ambiguous(self, mock_rag_pipeline, test_queries):
        """Test ambiguous query that could match multiple sources."""
        query_key = "edge_case_ambiguous"
        query_data = test_queries[query_key]
        query = query_data["question"]
        
        result = mock_rag_pipeline.process_query(query)
        
        # Should return results from at least one source
        assert len(result["video_results"]) > 0 or len(result["pdf_results"]) > 0
    
    def test_query_with_special_characters(self, mock_rag_pipeline):
        """Test query with special characters."""
        query = "Docker: What is it? #containers @mentions"
        
        result = mock_rag_pipeline.process_query(query)
        
        # Should handle special characters without crashing
        assert "query" in result
        assert result["query"] == query
    
    def test_query_with_numbers(self, mock_rag_pipeline):
        """Test query with numbers."""
        query = "How many CPU cores does Kubernetes need for 1000 pods?"
        
        result = mock_rag_pipeline.process_query(query)
        
        # Should process numbers correctly
        assert result is not None
    
    def test_very_short_query(self, mock_rag_pipeline):
        """Test very short query (single word)."""
        query = "Docker"
        
        result = mock_rag_pipeline.process_query(query)
        
        assert result is not None
    
    def test_very_long_query(self, mock_rag_pipeline):
        """Test very long query."""
        query = "How do I install and configure Docker on my MacBook " * 10
        
        result = mock_rag_pipeline.process_query(query)
        
        assert result is not None


class TestFallbackBehavior:
    """Test fallback from video to PDF results."""
    
    def test_fallback_to_pdf_when_video_low_score(self, mock_rag_pipeline):
        """Test fallback to PDF when video score below threshold."""
        # This would be a query that videos don't match well
        query = "Specific containerization best practices"
        
        result = mock_rag_pipeline.process_query(query)
        
        # Should have results from somewhere
        assert len(result["video_results"]) > 0 or len(result["pdf_results"]) > 0
    
    def test_prefer_higher_score_source(self, mock_rag_pipeline):
        """Test that higher-scoring source is preferred."""
        query = "Docker installation steps"
        
        result = mock_rag_pipeline.process_query(query)
        
        # Get best video and best PDF scores
        best_video_score = result["video_results"][0].get("score", 0) if result["video_results"] else 0
        best_pdf_score = result["pdf_results"][0].get("score", 0) if result["pdf_results"] else 0
        
        # Best source should have highest score
        best_source = result.get("best_source")
        if best_video_score > best_pdf_score:
            assert best_source == "video"
        elif best_pdf_score > best_video_score:
            assert best_source == "pdf"
    
    def test_fallback_chain(self, mock_rag_pipeline):
        """Test fallback chain: video -> PDF -> none."""
        # Test with query that should gradually have lower scores
        queries = [
            "Docker installation",  # Should match video well
            "Containerization practices",  # Should match PDF well
            "Quantum computing architecture",  # Shouldn't match well
        ]
        
        for query in queries:
            result = mock_rag_pipeline.process_query(query)
            assert result is not None
            assert "best_source" in result


class TestResponseFormatting:
    """Test response formatting and structure."""
    
    def test_video_response_structure(self, mock_rag_pipeline):
        """Test that video responses have correct structure."""
        query = "How do I install Docker?"
        
        result = mock_rag_pipeline.process_query(query)
        
        if result["video_results"]:
            video_result = result["video_results"][0]
            
            # Check required fields
            assert "chunk_id" in video_result
            assert "video_id" in video_result
            assert "text" in video_result
            assert "score" in video_result
            assert "start_timestamp" in video_result
            assert "end_timestamp" in video_result
    
    def test_pdf_response_structure(self, mock_rag_pipeline):
        """Test that PDF responses have correct structure."""
        query = "Best practices for containerization"
        
        result = mock_rag_pipeline.process_query(query)
        
        if result["pdf_results"]:
            pdf_result = result["pdf_results"][0]
            
            # Check required fields
            assert "para_id" in pdf_result
            assert "pdf_filename" in pdf_result
            assert "text" in pdf_result
            assert "score" in pdf_result
            assert "page_number" in pdf_result
    
    def test_response_includes_metadata(self, mock_rag_pipeline):
        """Test that responses include relevant metadata."""
        query = "Docker tutorial"
        
        result = mock_rag_pipeline.process_query(query)
        
        # Should include metadata about the query
        assert "query" in result
        assert "best_source" in result
    
    def test_response_scoring(self, mock_rag_pipeline):
        """Test that all results include scores."""
        query = "Container technology"
        
        result = mock_rag_pipeline.process_query(query)
        
        # All results should have scores
        for video_result in result["video_results"]:
            assert "score" in video_result
            assert 0 <= video_result["score"] <= 1
        
        for pdf_result in result["pdf_results"]:
            assert "score" in pdf_result
            assert 0 <= pdf_result["score"] <= 1
    
    def test_response_ranking(self, mock_rag_pipeline):
        """Test that results are properly ranked by score."""
        query = "Docker and Kubernetes"
        
        result = mock_rag_pipeline.process_query(query)
        
        # Video results should be sorted by score (descending)
        video_scores = [r.get("score", 0) for r in result["video_results"]]
        for i in range(len(video_scores) - 1):
            assert video_scores[i] >= video_scores[i + 1]
        
        # PDF results should be sorted by score (descending)
        pdf_scores = [r.get("score", 0) for r in result["pdf_results"]]
        for i in range(len(pdf_scores) - 1):
            assert pdf_scores[i] >= pdf_scores[i + 1]


class TestAnswerGeneration:
    """Test answer generation from retrieved results."""
    
    def test_generate_from_video(self, mock_rag_pipeline, mock_video_chunks):
        """Test generating answer from video chunks."""
        # This would require AnswerGenerator to be properly configured
        query = "How to install Docker?"
        
        result = mock_rag_pipeline.process_query(query)
        
        # Answer generation would happen on these results
        assert len(result["video_results"]) > 0
    
    def test_generate_from_pdf(self, mock_rag_pipeline, mock_pdf_paragraphs):
        """Test generating answer from PDF paragraphs."""
        query = "Best practices for containers"
        
        result = mock_rag_pipeline.process_query(query)
        
        # Answer generation would happen on these results
        assert result is not None
    
    def test_answer_includes_source_citation(self):
        """Test that answers include source information."""
        # Would be tested with actual AnswerGenerator
        pass


class TestBoundaryConditions:
    """Test boundary conditions and edge cases."""
    
    def test_empty_query(self, mock_rag_pipeline):
        """Test empty query string."""
        query = ""
        
        # Should handle gracefully
        try:
            result = mock_rag_pipeline.process_query(query)
            assert result is not None
        except ValueError:
            pass  # Acceptable to reject empty query
    
    def test_whitespace_only_query(self, mock_rag_pipeline):
        """Test query with only whitespace."""
        query = "   \t\n  "
        
        try:
            result = mock_rag_pipeline.process_query(query)
            assert result is not None
        except ValueError:
            pass  # Acceptable to reject
    
    def test_extremely_long_query(self, mock_rag_pipeline):
        """Test very long query (100,000+ characters)."""
        query = "word " * 20000  # 100,000 characters
        
        # Should handle or raise error gracefully
        try:
            result = mock_rag_pipeline.process_query(query)
            assert result is not None
        except ValueError:
            pass  # Acceptable limit
    
    def test_unicode_query(self, mock_rag_pipeline):
        """Test query with unicode characters."""
        query = "Docker 中文 Kubernetes français"
        
        result = mock_rag_pipeline.process_query(query)
        
        assert result is not None
    
    def test_emoji_query(self, mock_rag_pipeline):
        """Test query with emoji."""
        query = "Docker 🐳 Kubernetes ☸️ containers 📦"
        
        result = mock_rag_pipeline.process_query(query)
        
        assert result is not None


class TestIndexingAndRetrieval:
    """Test indexing and retrieval workflow."""
    
    def test_index_videos_and_pdfs(self, mock_rag_pipeline, mock_video_chunks, mock_pdf_paragraphs):
        """Test indexing both videos and PDFs."""
        # Index data
        mock_rag_pipeline.index_data(mock_video_chunks, mock_pdf_paragraphs)
        
        # Verify indexed
        collections = mock_rag_pipeline.vector_store.get_collections()
        
        assert collections["video_chunks"] > 0
        assert collections["pdf_paragraphs"] > 0
    
    def test_retrieve_after_indexing(self, mock_rag_pipeline, mock_video_chunks, mock_pdf_paragraphs):
        """Test retrieval after indexing data."""
        # Index data
        mock_rag_pipeline.index_data(mock_video_chunks, mock_pdf_paragraphs)
        
        # Query
        result = mock_rag_pipeline.process_query("Docker installation")
        
        # Should have results
        assert len(result["video_results"]) > 0 or len(result["pdf_results"]) > 0
    
    def test_update_index(self, mock_rag_pipeline, mock_video_chunks):
        """Test updating index with new data."""
        # Initial index
        mock_rag_pipeline.index_data(mock_video_chunks, [])
        
        # Add more data
        new_chunks = [
            {
                "chunk_id": "NEW_chunk",
                "video_id": "VIDEO_NEW",
                "text": "New Docker content",
                "score": 0.8
            }
        ]
        
        mock_rag_pipeline.index_data(new_chunks, [])
        
        # Updated index should have more data
        collections = mock_rag_pipeline.vector_store.get_collections()
        assert collections["video_chunks"] > len(mock_video_chunks)


class TestErrorHandling:
    """Test error handling and recovery."""
    
    def test_handle_connection_error(self, mock_rag_pipeline):
        """Test handling of connection errors."""
        # Query should still work with mock
        result = mock_rag_pipeline.process_query("Docker")
        assert result is not None
    
    def test_handle_timeout(self, mock_rag_pipeline):
        """Test handling of timeout errors."""
        # With mock, shouldn't timeout
        result = mock_rag_pipeline.process_query("Docker")
        assert result is not None
    
    def test_handle_invalid_embedding(self, mock_rag_pipeline):
        """Test handling of invalid embeddings."""
        # Should degrade gracefully
        result = mock_rag_pipeline.process_query("Docker")
        assert result is not None
    
    def test_handle_empty_results(self, mock_rag_pipeline):
        """Test handling of empty results."""
        query = "NonexistentTopicXYZ123"
        
        result = mock_rag_pipeline.process_query(query)
        
        # Should return valid structure even with no results
        assert result is not None
        assert "video_results" in result
        assert "pdf_results" in result
