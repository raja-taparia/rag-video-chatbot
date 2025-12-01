# Generate natural language answers from retrieved chunks

import logging
from typing import Dict, Any, Optional, List
import requests
from src.models import VideoAnswer, PDFAnswer, NoAnswer

logger = logging.getLogger(__name__)


class AnswerGenerator:
    """
    Generate natural language answers using LLM.
    
    Takes retrieved context (video chunks or PDF paragraphs) and uses
    Ollama LLM to generate coherent, conversational answers to user queries.
    """
    
    def __init__(self, ollama_config):
        """
        Initialize answer generator with Ollama configuration.
        
        Args:
            ollama_config: OllamaConfig object with model settings
        """
        self.base_url = ollama_config.base_url
        self.model = ollama_config.llm_model
        logger.info(f"AnswerGenerator initialized with model: {self.model}")
    
    def generate_from_video(
        self,
        query: str,
        video_chunks: List[Dict[str, Any]]
    ) -> Optional[VideoAnswer]:
        """
        Generate answer from video chunks.
        
        Process:
        1. Select best video chunk (highest score)
        2. Create prompt with query + context
        3. Call LLM to generate natural language answer
        4. Format response with video metadata
        
        Args:
            query: User question
            video_chunks: List of retrieved video chunks with scores
            
        Returns:
            VideoAnswer object with answer and source metadata, or None if failed
        """
        if not video_chunks:
            logger.warning("No video chunks provided for answer generation")
            return None
        
        try:
            # Select best chunk (highest score)
            best_chunk = video_chunks[0]
            
            logger.info(f"Generating video answer using chunk: {best_chunk['chunk_id']}")
            logger.debug(f"Chunk score: {best_chunk.get('score', 0):.3f}")
            
            # Create prompt for LLM
            prompt = self._create_video_prompt(query, best_chunk)
            
            # Call LLM
            answer_text = self._call_llm(prompt)
            
            # Fallback to raw text if LLM fails
            if not answer_text:
                logger.warning("LLM generation failed, using raw transcript snippet")
                answer_text = best_chunk['text'][:500]
            
            # Create response
            video_answer = VideoAnswer(
                video_id=best_chunk['video_id'],
                title=best_chunk.get('video_title', f"Video: {best_chunk['video_id']}"),
                answer_text=answer_text,
                segments=[{
                    'segment_id': 0,
                    'start_timestamp': best_chunk['start_timestamp'],
                    'start_token_id': best_chunk['start_token_id'],
                    'end_timestamp': best_chunk['end_timestamp'],
                    'end_token_id': best_chunk['end_token_id'],
                    'transcript_snippet': best_chunk['text'][:300],
                    'confidence': best_chunk.get('score', 0.0),
                }],
                confidence=best_chunk.get('score', 0.0)
            )
            
            logger.info(f"Video answer generated successfully (confidence: {video_answer.confidence:.3f})")
            return video_answer
            
        except Exception as e:
            logger.error(f"Error generating video answer: {e}", exc_info=True)
            return None
    
    def generate_from_pdf(
        self,
        query: str,
        pdf_paragraphs: List[Dict[str, Any]]
    ) -> Optional[PDFAnswer]:
        """
        Generate answer from PDF paragraphs.
        
        Process:
        1. Select best paragraph (highest score)
        2. Create prompt with query + context
        3. Call LLM to generate natural language answer
        4. Format response with PDF citation metadata
        
        Args:
            query: User question
            pdf_paragraphs: List of retrieved PDF paragraphs with scores
            
        Returns:
            PDFAnswer object with answer and citation metadata, or None if failed
        """
        if not pdf_paragraphs:
            logger.warning("No PDF paragraphs provided for answer generation")
            return None
        
        try:
            # Select best paragraph (highest score)
            best_para = pdf_paragraphs[0]
            
            logger.info(f"Generating PDF answer using paragraph: {best_para['para_id']}")
            logger.debug(f"Paragraph score: {best_para.get('score', 0):.3f}")
            logger.debug(f"Citation: {best_para['pdf_filename']}, Page {best_para['page_number']}")
            
            # Create prompt for LLM
            prompt = self._create_pdf_prompt(query, best_para)
            
            # Call LLM
            answer_text = self._call_llm(prompt)
            
            # Fallback to raw text if LLM fails
            if not answer_text:
                logger.warning("LLM generation failed, using raw PDF text")
                answer_text = best_para['text'][:500]
            
            # Create response
            pdf_answer = PDFAnswer(
                pdf_filename=best_para['pdf_filename'],
                page_number=best_para['page_number'],
                paragraph_index=best_para['paragraph_index'],
                answer_text=answer_text,
                source_snippet=best_para['text'][:300],
                confidence=best_para.get('score', 0.0)
            )
            
            logger.info(f"PDF answer generated successfully (confidence: {pdf_answer.confidence:.3f})")
            return pdf_answer
            
        except Exception as e:
            logger.error(f"Error generating PDF answer: {e}", exc_info=True)
            return None
    
    def _create_video_prompt(self, query: str, chunk: Dict[str, Any]) -> str:
        """
        Create prompt for video transcript context.
        
        Args:
            query: User question
            chunk: Video chunk with text
            
        Returns:
            Formatted prompt for LLM
        """
        return f"""You are a helpful assistant. Answer the following question based on the provided video transcript excerpt.

Question: {query}

Video Transcript:
{chunk['text'][:500]}

Guidelines:
- Answer directly and concisely
- If the transcript doesn't contain the answer, say "This information is not in the transcript"
- Keep your answer to 2-3 sentences

Answer:"""
    
    def _create_pdf_prompt(self, query: str, para: Dict[str, Any]) -> str:
        """
        Create prompt for PDF document context.
        
        Args:
            query: User question
            para: PDF paragraph with text
            
        Returns:
            Formatted prompt for LLM
        """
        return f"""You are a helpful assistant. Answer the following question based on the provided document excerpt.

Question: {query}

Document Excerpt:
{para['text'][:500]}

Guidelines:
- Answer directly and concisely
- If the document doesn't contain the answer, say "This information is not in the document"
- Keep your answer to 2-3 sentences

Answer:"""
    
    def _call_llm(self, prompt: str, max_tokens: int = 200, temperature: float = 0.3) -> Optional[str]:
        """
        Call Ollama LLM to generate answer.
        
        Args:
            prompt: Prompt with context and question
            max_tokens: Maximum tokens in response
            temperature: Creativity level (0.0 = deterministic, 1.0 = creative)
            
        Returns:
            Generated text response, or None if failed
        """
        try:
            logger.debug(f"Calling LLM: {self.model}")
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
                timeout=60
            )
            
            response.raise_for_status()
            result = response.json()
            generated_text = result.get('response', '').strip()
            
            if generated_text:
                logger.debug(f"LLM generated: {generated_text[:100]}...")
                return generated_text
            else:
                logger.warning("LLM returned empty response")
                return None
                
        except requests.exceptions.Timeout:
            logger.error(f"LLM request timeout after 60 seconds")
            return None
        except requests.exceptions.ConnectionError:
            logger.error(f"Failed to connect to LLM at {self.base_url}")
            return None
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            return None
    
    def batch_generate(
        self,
        query: str,
        video_chunks: List[Dict[str, Any]],
        pdf_paragraphs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate answers from both video and PDF for comparison.
        
        Args:
            query: User question
            video_chunks: Retrieved video chunks
            pdf_paragraphs: Retrieved PDF paragraphs
            
        Returns:
            Dictionary with both answers for A/B comparison
        """
        results = {
            'query': query,
            'video_answer': None,
            'pdf_answer': None,
        }
        
        if video_chunks:
            results['video_answer'] = self.generate_from_video(query, video_chunks)
        
        if pdf_paragraphs:
            results['pdf_answer'] = self.generate_from_pdf(query, pdf_paragraphs)
        
        return results
