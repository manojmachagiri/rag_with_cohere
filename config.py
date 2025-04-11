"""
Configuration settings for the PDF RAG application.
"""

import os
from typing import Dict, Any

# Cohere API settings
COHERE_API_KEY = "1N8SXLVTH2IJkc8Z2HhER0nZaU1V8pdcMclS39Ke"
COHERE_EMBEDDING_MODEL = "embed-english-v3.0"
COHERE_GENERATION_MODEL = "command"
COHERE_TIMEOUT = 60.0  # seconds

# Default model setting
DEFAULT_MODEL = "command"  # Default model for compatibility with existing code

# PDF processing settings
DEFAULT_CHUNK_SIZE = 2000
DEFAULT_CHUNK_OVERLAP = 1000
DEFAULT_NUM_RESULTS = 10

# Vector store settings
VECTOR_DB_DIR = os.path.join(os.getcwd(), "chroma_db")
COLLECTION_NAME = "pdf_collection"

# System prompts
SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.
If the answer is not in the context, say so. Be concise and accurate."""

RAG_PROMPT_TEMPLATE = """
I'll provide you with some context information and then ask a question.
Please use the context to answer the question accurately. If the answer is not in the context, say so.

Context:
{context}

Question: {query}

Answer:
"""

# Logging settings
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = "app.log"

# Application settings
APP_TITLE = "📚 PDF RAG with Cohere"
APP_DESCRIPTION = "Upload a PDF, ask questions, and get answers using Cohere with RAG"
APP_ICON = "📚"
APP_LAYOUT = "wide"
SIDEBAR_STATE = "expanded"

# Function to get all config as a dictionary
def get_config() -> Dict[str, Any]:
    """Return all configuration settings as a dictionary."""
    return {k: v for k, v in globals().items()
            if not k.startswith('_') and k.isupper()}
