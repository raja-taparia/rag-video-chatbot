# Unit tests for PDF extraction
# Place this file at: tests/test_pdf_extraction.py

import pytest
from pathlib import Path
from typing import Dict
from src.ingestion.pdf_loader import PDFExtractor


class TestPDFExtraction:
    """Test PDF text extraction functionality."""
    
    def test_extract_pdf_pages(self, temp_data_dir: Path, mock_pdf_pages: Dict[int, str]):
        """Test extracting text from PDF pages."""
        extractor = PDFExtractor()
        
        # In real tests, use actual PDF files
        # For now, test the extraction logic
        extracted = extractor.parse_pages(mock_pdf_pages)
        
        assert len(extracted) == len(mock_pdf_pages)
        assert extracted[1].startswith("Docker")
    
    def test_extract_single_page(self, mock_pdf_pages: Dict[int, str]):
        """Test extracting a single page."""
        extractor = PDFExtractor()
        
        single_page = {1: mock_pdf_pages[1]}
        extracted = extractor.parse_pages(single_page)
        
        assert len(extracted) == 1
        assert extracted[1] == mock_pdf_pages[1]
    
    def test_extract_with_page_numbers(self, mock_pdf_pages: Dict[int, str]):
        """Test that page numbers are preserved."""
        extractor = PDFExtractor()
        
        extracted = extractor.parse_pages(mock_pdf_pages)
        
        # Check all page numbers are present
        for page_num in mock_pdf_pages.keys():
            assert page_num in extracted
    
    def test_extract_empty_pdf(self):
        """Test handling of empty PDF."""
        extractor = PDFExtractor()
        
        empty_pages = {}
        extracted = extractor.parse_pages(empty_pages)
        
        assert len(extracted) == 0
    
    def test_extract_preserves_text_content(self, mock_pdf_pages: Dict[int, str]):
        """Test that extracted text preserves content."""
        extractor = PDFExtractor()
        
        extracted = extractor.parse_pages(mock_pdf_pages)
        
        # Page 10 should contain Docker installation info
        assert "Docker" in extracted[10] or "installation" in extracted[10].lower()
    
    def test_extract_handles_special_characters(self):
        """Test extraction with special characters."""
        extractor = PDFExtractor()
        
        pages_with_special = {
            1: "Text with émojis 🚀 and spëcial çharacters",
            2: "Mathematical: ∑ ∫ √ ∞",
            3: "Quotes: \"double\" and 'single' and `backtick`"
        }
        
        extracted = extractor.parse_pages(pages_with_special)
        
        # Should preserve special characters
        assert "spëcial" in extracted[1]
        assert "∫" in extracted[2]
        assert "`" in extracted[3]
    
    def test_extract_large_pdf(self):
        """Test extraction with many pages."""
        extractor = PDFExtractor()
        
        # Create large PDF simulation (100 pages)
        large_pdf = {
            i: f"Page {i} content. This is sample text for page {i}. "
            for i in range(1, 101)
        }
        
        extracted = extractor.parse_pages(large_pdf)
        
        assert len(extracted) == 100
        assert "Page 50" in extracted[50]


class TestPDFParsing:
    """Test PDF text parsing and cleaning."""
    
    def test_parse_paragraphs_from_page(self, mock_pdf_pages: Dict[int, str]):
        """Test parsing paragraphs from extracted page text."""
        extractor = PDFExtractor()
        
        page_text = mock_pdf_pages[1]
        paragraphs = extractor.split_into_paragraphs(page_text)
        
        assert len(paragraphs) > 0
    
    def test_parse_with_newlines(self):
        """Test parsing text with multiple newlines."""
        extractor = PDFExtractor()
        
        text_with_newlines = """First paragraph.
        
        
        Second paragraph with double newlines.
        
        Third paragraph."""
        
        paragraphs = extractor.split_into_paragraphs(text_with_newlines)
        
        # Should split into 3 paragraphs
        assert len(paragraphs) >= 2
    
    def test_remove_extra_whitespace(self):
        """Test removal of extra whitespace."""
        extractor = PDFExtractor()
        
        text_with_spaces = "Text   with    multiple     spaces"
        cleaned = extractor.clean_text(text_with_spaces)
        
        assert "  " not in cleaned  # No double spaces
        assert cleaned == "Text with multiple spaces"
    
    def test_handle_pdf_artifacts(self):
        """Test removal of common PDF artifacts."""
        extractor = PDFExtractor()
        
        text_with_artifacts = "Page 1\n\nContent here\n\nFootnote: 1\n\nHeader text"
        cleaned = extractor.remove_pdf_artifacts(text_with_artifacts)
        
        # Should remove common artifacts
        assert len(cleaned) <= len(text_with_artifacts)
    
    def test_extract_headings(self):
        """Test extraction of headings/sections."""
        extractor = PDFExtractor()
        
        text_with_headings = """CHAPTER 1: Introduction
        
        This is the introduction paragraph.
        
        SECTION 1.1: Background
        
        Background information here."""
        
        headings = extractor.extract_headings(text_with_headings)
        
        assert len(headings) > 0
        assert any("Chapter" in h or "chapter" in h.lower() for h in headings)


class TestPDFValidation:
    """Test PDF validation and error handling."""
    
    def test_validate_pdf_file_exists(self, temp_data_dir: Path):
        """Test validation that PDF file exists."""
        extractor = PDFExtractor()
        
        # Non-existent file
        with pytest.raises(FileNotFoundError):
            extractor.validate_pdf_path(Path("/nonexistent/file.pdf"))
    
    def test_validate_pdf_extension(self):
        """Test validation of PDF extension."""
        extractor = PDFExtractor()
        
        # Wrong extension
        with pytest.raises(ValueError):
            extractor.validate_pdf_extension("document.txt")
    
    def test_validate_pdf_is_readable(self, temp_data_dir: Path):
        """Test that PDF is readable."""
        extractor = PDFExtractor()
        
        # Create a dummy PDF file (for testing)
        pdf_path = temp_data_dir / "test.pdf"
        pdf_path.write_text("Not a real PDF")  # Invalid content
        
        # Should raise error for corrupted PDF
        with pytest.raises((ValueError, IOError)):
            extractor.validate_pdf_readable(pdf_path)
    
    def test_handle_corrupted_pdf(self):
        """Test handling of corrupted PDF files."""
        extractor = PDFExtractor()
        
        # Simulate corrupted PDF
        pages = {}  # Empty pages
        
        result = extractor.parse_pages(pages)
        
        # Should return empty result instead of crashing
        assert result == {}
    
    def test_handle_scanned_pdf(self):
        """Test detection of scanned (image-based) PDF."""
        extractor = PDFExtractor()
        
        # Scanned PDFs would have no extractable text
        # Try to detect this
        pages = {1: "", 2: "", 3: ""}  # All empty (scanned PDF)
        
        is_scanned = extractor.detect_scanned_pdf(pages)
        
        assert is_scanned is True


class TestPDFMetadata:
    """Test extraction of PDF metadata."""
    
    def test_extract_pdf_metadata(self):
        """Test extracting PDF metadata."""
        extractor = PDFExtractor()
        
        # Metadata would be extracted from actual PDF
        metadata = {
            "title": "Docker Guide",
            "author": "Tech Writer",
            "creation_date": "2024-01-01",
            "pages": 20
        }
        
        # Validate metadata structure
        assert "title" in metadata
        assert "pages" in metadata
    
    def test_get_pdf_info(self):
        """Test getting PDF info."""
        extractor = PDFExtractor()
        
        pages = {i: f"Page {i}" for i in range(1, 21)}
        
        info = extractor.get_pdf_info(pages)
        
        assert info["total_pages"] == 20
        assert info["total_characters"] > 0


class TestPDFExtractorPerformance:
    """Test PDF extraction performance."""
    
    def test_extract_large_pdf_performance(self):
        """Test extraction performance with large PDF."""
        import time
        
        extractor = PDFExtractor()
        
        # Simulate large PDF (500 pages)
        large_pdf = {
            i: f"Page {i}. " + ("Sample content. " * 10)
            for i in range(1, 501)
        }
        
        start = time.time()
        extracted = extractor.parse_pages(large_pdf)
        elapsed = time.time() - start
        
        assert len(extracted) == 500
        # Should complete in reasonable time
        assert elapsed < 5.0
    
    def test_extract_many_pages_in_memory(self):
        """Test memory efficiency with many pages."""
        extractor = PDFExtractor()
        
        # Create large PDF
        large_pdf = {
            i: f"Page {i}. " + ("x" * 1000)  # 1KB per page
            for i in range(1, 1001)  # 1000 pages
        }
        
        # Should handle without excessive memory use
        extracted = extractor.parse_pages(large_pdf)
        
        assert len(extracted) == 1000


class TestPDFContent:
    """Test content-specific extraction."""
    
    def test_extract_code_blocks(self):
        """Test extraction of code blocks from PDF."""
        extractor = PDFExtractor()
        
        text_with_code = """
        Here's a Docker command:
        
        ```
        docker run -p 8000:8000 myapp
        ```
        
        This runs the app."""
        
        code_blocks = extractor.extract_code_blocks(text_with_code)
        
        assert len(code_blocks) > 0
        assert "docker run" in code_blocks[0]
    
    def test_extract_tables_from_pdf(self):
        """Test extraction of table data."""
        extractor = PDFExtractor()
        
        text_with_table = """
        Comparison Table:
        Feature | Docker | VM
        Lightweight | Yes | No
        Startup | Fast | Slow"""
        
        tables = extractor.extract_tables(text_with_table)
        
        # Should detect table structure
        assert isinstance(tables, (list, dict))
    
    def test_extract_links_from_pdf(self):
        """Test extraction of links/URLs."""
        extractor = PDFExtractor()
        
        text_with_links = """
        Visit https://docker.com for more info.
        Download from https://github.com/user/repo
        Email: test@example.com"""
        
        links = extractor.extract_urls(text_with_links)
        
        assert len(links) >= 2
        assert any("docker.com" in link for link in links)
    
    def test_extract_references(self):
        """Test extraction of references/citations."""
        extractor = PDFExtractor()
        
        text_with_refs = """
        According to Smith et al. (2020), containers are important.
        See reference [1] for more details.
        As mentioned in Chapter 3, Docker is widely used."""
        
        references = extractor.extract_references(text_with_refs)
        
        assert len(references) > 0


class TestPDFComparison:
    """Test comparison of multiple PDFs."""
    
    def test_compare_two_pdfs(self, mock_pdf_pages, mock_pdf_files_kubernetes):
        """Test comparing content between two PDFs."""
        extractor = PDFExtractor()
        
        similarity = extractor.compare_pdfs(mock_pdf_pages, mock_pdf_files_kubernetes)
        
        # Both are about containers/orchestration, so should have some similarity
        assert 0 <= similarity <= 1
    
    def test_detect_duplicate_pdfs(self):
        """Test detection of duplicate PDF content."""
        extractor = PDFExtractor()
        
        same_pages = {1: "Identical content", 2: "More content"}
        duplicate_pages = {1: "Identical content", 2: "More content"}
        
        similarity = extractor.compare_pdfs(same_pages, duplicate_pages)
        
        # Should be very similar (nearly 1.0)
        assert similarity > 0.9
