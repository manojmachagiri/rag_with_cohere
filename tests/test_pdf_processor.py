"""
Unit tests for the PDF processor module.
"""

import os
import unittest
import tempfile
from io import BytesIO
from unittest.mock import MagicMock, patch

from utils.pdf_processor import save_uploaded_file, extract_text_from_pdf, split_text

class TestPdfProcessor(unittest.TestCase):
    """Test cases for PDF processor functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock PDF file for testing
        self.test_pdf_content = b"%PDF-1.7\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>\nendobj\n4 0 obj\n<< /Length 44 >>\nstream\nBT\n/F1 12 Tf\n100 700 Td\n(Test PDF Content) Tj\nET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f\n0000000010 00000 n\n0000000060 00000 n\n0000000120 00000 n\n0000000210 00000 n\ntrailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n300\n%%EOF"
        
        # Create a mock uploaded file
        self.mock_uploaded_file = MagicMock()
        self.mock_uploaded_file.name = "test.pdf"
        self.mock_uploaded_file.getvalue.return_value = self.test_pdf_content
        
    def test_save_uploaded_file(self):
        """Test saving an uploaded file."""
        with patch('tempfile.NamedTemporaryFile') as mock_temp:
            # Set up the mock temporary file
            mock_temp_instance = MagicMock()
            mock_temp_instance.name = "/tmp/test_temp.pdf"
            mock_temp.return_value.__enter__.return_value = mock_temp_instance
            
            # Call the function
            result = save_uploaded_file(self.mock_uploaded_file)
            
            # Assertions
            self.assertEqual(result, "/tmp/test_temp.pdf")
            mock_temp_instance.write.assert_called_once_with(self.test_pdf_content)
    
    @patch('PyPDF2.PdfReader')
    def test_extract_text_from_pdf(self, mock_pdf_reader):
        """Test extracting text from a PDF file."""
        # Set up the mock PDF reader
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Test PDF Content"
        mock_pdf_reader.return_value.pages = [mock_page]
        
        # Create a temporary file for testing
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(self.test_pdf_content)
            tmp_path = tmp_file.name
        
        try:
            # Call the function
            result = extract_text_from_pdf(tmp_path)
            
            # Assertions
            self.assertEqual(result, "Test PDF Content\n")
            mock_pdf_reader.assert_called_once_with(tmp_path)
            mock_page.extract_text.assert_called_once()
        finally:
            # Clean up if the file still exists (in case the function didn't delete it)
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def test_split_text(self):
        """Test splitting text into chunks."""
        # Test text
        test_text = "This is a test paragraph.\n\nThis is another paragraph with more content.\n\nAnd here's a third paragraph to ensure we have enough text to split."
        
        # Call the function with different chunk sizes
        chunks_large = split_text(test_text, chunk_size=100, chunk_overlap=0)
        chunks_small = split_text(test_text, chunk_size=30, chunk_overlap=5)
        
        # Assertions for large chunks (should be one chunk)
        self.assertEqual(len(chunks_large), 1)
        self.assertEqual(chunks_large[0], test_text)
        
        # Assertions for small chunks (should be multiple chunks)
        self.assertGreater(len(chunks_small), 1)
        
        # Test with empty text
        empty_chunks = split_text("", chunk_size=100, chunk_overlap=0)
        self.assertEqual(empty_chunks, [])
        
        # Test with None
        none_chunks = split_text(None, chunk_size=100, chunk_overlap=0)
        self.assertEqual(none_chunks, [])

if __name__ == '__main__':
    unittest.main()
