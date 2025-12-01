# PDF Search, Download, and Processing

import logging
import requests
from pathlib import Path
from typing import List, Optional, Dict, Any
from urllib.parse import urljoin
import subprocess
import json

logger = logging.getLogger("ingestion")


class PDFFinder:
    """
    Searches for, downloads, and processes PDFs based on ideas.
    
    Features:
    - Search Google for PDF documents
    - Download 2 PDFs per idea/keyword
    - Extract text and organize by page/paragraph
    - Convert to standard format for RAG ingestion
    """
    
    def __init__(self, output_dir: Path, pdf_finder_file: Path):
        """
        Initialize PDF finder.
        
        Args:
            output_dir: Directory to save downloaded PDFs
            pdf_finder_file: File containing PDF search ideas
        """
        self.output_dir = Path(output_dir)
        self.pdf_finder_file = Path(pdf_finder_file)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"PDFFinderInitialized: {output_dir}")
    
    def search_google_for_pdfs(self, query: str, num_results: int = 5) -> List[str]:
        """
        Search Google for PDF files matching query.
        
        Note: This uses DuckDuckGo API (free) since Google requires authentication.
        
        Args:
            query: Search query (e.g., "Kubernetes tutorial PDF")
            num_results: Number of PDF links to find
            
        Returns:
            List of PDF URLs
        """
        try:
            logger.info(f"Searching for PDFs: {query}")
            
            # Add filetype:pdf to search
            search_query = f"{query} filetype:pdf"
            
            # Use DuckDuckGo API (free, no auth required)
            url = "https://duckduckgo.com/html"
            params = {
                'q': search_query,
                'kl': 'en-us'
            }
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Parse HTML for PDF links (simplified regex search)
            import re
            pdf_pattern = r'href=["\']([^"\']*\.pdf)'
            matches = re.findall(pdf_pattern, response.text, re.IGNORECASE)
            
            pdf_urls = list(set(matches))[:num_results]  # Remove duplicates, limit results
            
            logger.info(f"Found {len(pdf_urls)} PDF URLs for: {query}")
            
            return pdf_urls
            
        except Exception as e:
            logger.error(f"Error searching for PDFs: {e}")
            return []
    
    def download_pdf(self, url: str, output_file: Path, timeout: int = 30) -> bool:
        """
        Download PDF from URL with error handling.
        
        Args:
            url: PDF URL
            output_file: Where to save the PDF
            timeout: Download timeout in seconds
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Downloading PDF: {url[:80]}...")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=timeout, stream=True)
            response.raise_for_status()
            
            # Check if it's actually a PDF
            content_type = response.headers.get('content-type', '')
            if 'application/pdf' not in content_type and 'application/octet-stream' not in content_type:
                logger.warning(f"Not a PDF (content-type: {content_type})")
                return False
            
            # Check file size (max 50MB)
            content_length = response.headers.get('content-length', '')
            if content_length and int(content_length) > 50 * 1024 * 1024:
                logger.warning(f"PDF too large ({content_length} bytes)")
                return False
            
            # Download
            with open(output_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            logger.info(f"Downloaded to: {output_file}")
            return True
            
        except requests.exceptions.Timeout:
            logger.error(f"Download timeout")
            return False
        except Exception as e:
            logger.error(f"Error downloading PDF: {e}")
            return False
    
    def extract_pdf_text(self, pdf_file: Path) -> Optional[Dict[int, str]]:
        """
        Extract text from PDF by page using PyPDF2.
        
        Args:
            pdf_file: Path to PDF file
            
        Returns:
            Dictionary mapping page_number (1-indexed) to text, or None if failed
        """
        try:
            logger.info(f"Extracting text from PDF: {pdf_file.name}")
            
            import PyPDF2
            
            pages_text = {}
            
            with open(pdf_file, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                
                for page_num in range(len(reader.pages)):
                    try:
                        page = reader.pages[page_num]
                        text = page.extract_text()
                        
                        if text:
                            pages_text[page_num + 1] = text  # 1-indexed
                    
                    except Exception as e:
                        logger.warning(f"Error extracting page {page_num}: {e}")
                        continue
            
            logger.info(f"Extracted {len(pages_text)} pages")
            return pages_text
            
        except Exception as e:
            logger.error(f"Error extracting PDF text: {e}")
            return None
    
    def normalize_pdf(self, pdf_file: Path) -> Optional[Path]:
        """
        Normalize PDF if needed (convert images to text, repair, etc.).
        
        Uses PyPDF2 to repair or OCR if necessary.
        
        Args:
            pdf_file: Original PDF file
            
        Returns:
            Path to normalized PDF, or None if failed
        """
        try:
            logger.info(f"Normalizing PDF: {pdf_file.name}")
            
            # For now, just check if it's readable
            pages = self.extract_pdf_text(pdf_file)
            
            if pages and len(pages) > 0:
                logger.info(f"PDF is readable with {len(pages)} pages")
                return pdf_file
            else:
                # PDF might be scanned (images)
                logger.warning(f"PDF appears to be scanned or corrupted")
                
                # Try OCR if available
                try:
                    import pytesseract
                    from PIL import Image
                    
                    logger.info("Attempting OCR...")
                    # This is complex - for now just log
                    logger.info("OCR not implemented in this version")
                    return None
                
                except ImportError:
                    logger.warning("pytesseract not installed for OCR")
                    return None
            
        except Exception as e:
            logger.error(f"Error normalizing PDF: {e}")
            return None
    
    def process_pdf_from_ideas_file(self) -> Dict[str, List[Path]]:
        """
        Process all ideas in pdf_finder file.
        
        File format (one idea per line, max 100 words):
        Kubernetes tutorial for beginners
        Docker containerization best practices
        
        Returns:
            Dictionary mapping idea to list of downloaded PDF paths
        """
        try:
            logger.info(f"Processing PDF finder file: {self.pdf_finder_file}")
            
            if not self.pdf_finder_file.exists():
                logger.warning(f"PDF finder file not found: {self.pdf_finder_file}")
                return {}
            
            results = {}
            
            with open(self.pdf_finder_file, 'r') as f:
                lines = f.readlines()
            
            logger.info(f"Found {len(lines)} ideas")
            
            for line_num, line in enumerate(lines, 1):
                idea = line.strip()
                
                if not idea or idea.startswith('#'):  # Skip empty and comments
                    continue
                
                logger.info(f"[{line_num}] Processing idea: {idea[:60]}...")
                
                # Search for 2 PDFs per idea
                pdf_urls = self.search_google_for_pdfs(idea, num_results=2)
                
                downloaded_files = []
                
                for url_idx, pdf_url in enumerate(pdf_urls, 1):
                    # Generate filename
                    filename = f"{idea[:30].replace(' ', '_')}_{url_idx}.pdf"
                    output_file = self.output_dir / filename
                    
                    # Download
                    if self.download_pdf(pdf_url, output_file):
                        # Normalize
                        normalized_file = self.normalize_pdf(output_file)
                        
                        if normalized_file:
                            downloaded_files.append(normalized_file)
                            logger.info(f"Successfully processed: {filename}")
                        else:
                            logger.warning(f"Failed to normalize: {filename}")
                    else:
                        logger.warning(f"Failed to download: {pdf_url[:50]}...")
                
                results[idea] = downloaded_files
            
            logger.info(f"Downloaded {sum(len(v) for v in results.values())} PDFs from {len(results)} ideas")
            return results
            
        except Exception as e:
            logger.error(f"Error processing PDF finder file: {e}", exc_info=True)
            return {}
