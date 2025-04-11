"""PDF processing utilities for the RAG application.

This module provides functions for handling PDF files, including:
- Saving uploaded files to temporary locations
- Extracting text from PDF files
- Splitting text into manageable chunks for processing
"""

import os
import tempfile
from typing import List, Optional
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from logger import setup_logger
from config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

# Set up logger
logger = setup_logger(__name__)

def save_uploaded_file(uploaded_file) -> str:
    """Save the uploaded file to a temporary location and return the path.

    Args:
        uploaded_file: The uploaded file object from Streamlit

    Returns:
        str: Path to the saved temporary file

    Raises:
        Exception: If there's an error saving the file
    """
    logger.info(f"Saving uploaded file: {uploaded_file.name}")
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            logger.debug(f"File saved to temporary location: {tmp_file.name}")
            return tmp_file.name
    except Exception as e:
        logger.error(f"Error saving uploaded file: {e}")
        raise Exception(f"Error saving uploaded file: {e}")

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF file.

    Args:
        file_path: Path to the PDF file

    Returns:
        str: Extracted text from the PDF

    Raises:
        Exception: If there's an error extracting text
    """
    logger.info(f"Extracting text from PDF: {file_path}")
    try:
        reader = PdfReader(file_path)
        text = ""
        total_pages = len(reader.pages)
        logger.info(f"PDF has {total_pages} pages")

        for i, page in enumerate(reader.pages):
            logger.debug(f"Processing page {i+1}/{total_pages}")
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
            else:
                logger.warning(f"No text extracted from page {i+1}")

        logger.info(f"Extracted {len(text)} characters from PDF")
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        raise Exception(f"Error extracting text from PDF: {e}")
    finally:
        # Clean up the temporary file
        if os.path.exists(file_path):
            logger.debug(f"Removing temporary file: {file_path}")
            os.remove(file_path)

def split_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE,
             chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[str]:
    """Split text into chunks for processing.

    Args:
        text: The text to split
        chunk_size: Maximum size of each chunk (default from config)
        chunk_overlap: Overlap between chunks (default from config)

    Returns:
        List[str]: List of text chunks

    Raises:
        Exception: If there's an error splitting the text
    """
    logger.info(f"Splitting text into chunks (size={chunk_size}, overlap={chunk_overlap})")

    if not text or len(text.strip()) == 0:
        logger.warning("Empty text provided for splitting")
        return []

    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_text(text)
        logger.info(f"Text split into {len(chunks)} chunks")

        # Log some stats about chunks
        if chunks:
            avg_chunk_size = sum(len(chunk) for chunk in chunks) / len(chunks)
            logger.debug(f"Average chunk size: {avg_chunk_size:.2f} characters")
            logger.debug(f"Smallest chunk: {min(len(chunk) for chunk in chunks)} characters")
            logger.debug(f"Largest chunk: {max(len(chunk) for chunk in chunks)} characters")

        return chunks
    except Exception as e:
        logger.error(f"Error splitting text: {e}")
        raise Exception(f"Error splitting text: {e}")
