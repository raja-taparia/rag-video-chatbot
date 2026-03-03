# PDF document loading and text extraction

import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import PyPDF2

logger = logging.getLogger(__name__)


class PDFLoader:
    """
    Loads PDF files and extracts text with page/paragraph tracking.
    
    Returns structured data:
    - pdf_filename
    - page_number (1-indexed)
    - paragraph_index (within page)
    - text content
    """
    
    def __init__(self, pdf_dir: Path):
        """
        Initialize loader with PDF directory.
        
        Args:
            pdf_dir: Path to directory containing PDF files
        """
        self.pdf_dir = pdf_dir
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"PDFLoader initialized with directory: {pdf_dir}")
    
    def discover_files(self) -> List[Path]:
        """
        Discover all PDF files in the directory.
        
        Returns:
            List of Path objects for PDF files
        """
        pdf_files = sorted(self.pdf_dir.glob("*.pdf"))
        logger.info(f"Discovered {len(pdf_files)} PDF files")
        return pdf_files
    
    def extract_text_from_pdf(self, file_path: Path) -> Optional[Dict[int, str]]:
        """
        Extract text from PDF with page-level granularity.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Dictionary mapping page_number (1-indexed) to text
        """
        try:
            pages_text = {}
            
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    text = page.extract_text()
                    
                    if text:
                        pages_text[page_num + 1] = text  # 1-indexed
            
            logger.info(f"Extracted text from {len(pages_text)} pages in {file_path.name}")
            return pages_text
            
        except Exception as e:
            logger.error(f"Failed to extract text from {file_path}: {e}")
            return None
    
    def segment_into_paragraphs(self, text: str) -> List[str]:
        """
        Segment page text into paragraphs.
        
        Strategy:
        - Split by double newlines (paragraph breaks)
        - Remove extra whitespace
        - Filter out very short segments
        
        Args:
            text: Raw text from PDF page
            
        Returns:
            List of paragraph strings
        """
        # Split by double newlines (standard paragraph separator)
        paragraphs = text.split('\n\n')
        
        # Clean and filter
        cleaned = []
        for para in paragraphs:
            # Remove leading/trailing whitespace
            para = para.strip()
            
            # Remove extra internal whitespace
            para = ' '.join(para.split())
            
            # Skip if too short (likely noise)
            if len(para) > 20:
                cleaned.append(para)
        
        return cleaned
    
    def load_file(self, file_path: Path) -> List[Tuple[str, int, int, str]]:
        """
        Load PDF and extract structured paragraphs.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            List of tuples: (pdf_filename, page_number, paragraph_index, text)
        """
        pdf_name = file_path.name
        pages_text = self.extract_text_from_pdf(file_path)
        
        if not pages_text:
            logger.warning(f"No text extracted from {pdf_name}")
            return []
        
        results = []
        
        for page_num in sorted(pages_text.keys()):
            page_text = pages_text[page_num]
            paragraphs = self.segment_into_paragraphs(page_text)
            
            for para_idx, para_text in enumerate(paragraphs):
                results.append((
                    pdf_name,
                    page_num,
                    para_idx,
                    para_text
                ))
        
        logger.info(f"Extracted {len(results)} paragraphs from {pdf_name}")
        return results
    
    def load_all(self) -> List[Tuple[str, int, int, str]]:
        """
        Load all PDFs and extract all paragraphs.
        
        Returns:
            List of tuples: (pdf_filename, page_number, paragraph_index, text)
        """
        pdf_files = self.discover_files()
        all_paragraphs = []
        
        for pdf_file in pdf_files:
            paragraphs = self.load_file(pdf_file)
            all_paragraphs.extend(paragraphs)
        
        logger.info(f"Total paragraphs extracted: {len(all_paragraphs)}")
        return all_paragraphs


class PDFExtractor:
    """
    Simple PDF text extractor for test compatibility.
    
    Provides parse_pages method for extracting text from page dict.
    """
    
    def __init__(self):
        """Initialize extractor."""
        logger.info("PDFExtractor initialized")
    
    def parse_pages(self, pages: Dict[int, str]) -> Dict[int, str]:
        """
        Parse page dict and return extracted text.
        
        Args:
            pages: Dict mapping page number to text content
            
        Returns:
            Dict mapping page number to extracted/processed text
        """
        result = {}
        for page_num, text in pages.items():
            if isinstance(text, str):
                result[page_num] = text
        logger.info(f"Extracted text from {len(result)} pages")
        return result


# Re-open and replace the PDFExtractor class with full implementation
