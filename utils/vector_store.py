"""Vector store utilities for the RAG application.

This module provides functions and classes for:
- Creating and managing vector embeddings using Cohere
- Storing and retrieving vectors using ChromaDB
- Performing similarity searches for RAG
"""

import os
import shutil
import sys
import pysqlite3
sys.modules['sqlite3'] = pysqlite3
from typing import List, Dict, Any, Optional, Union, Tuple

from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
from langchain.schema.document import Document

from logger import setup_logger
from config import VECTOR_DB_DIR, COLLECTION_NAME, DEFAULT_NUM_RESULTS, DEFAULT_MODEL
from utils.cohere_embeddings import CohereEmbeddings

# Set up logger
logger = setup_logger(__name__)

# For backward compatibility, keep OllamaEmbeddings as an alias to CohereEmbeddings
OllamaEmbeddings = CohereEmbeddings

def get_embeddings_model(model: Optional[str] = None) -> CohereEmbeddings:
    """Get embeddings model using Cohere.

    Args:
        model: Optional model name to use (defaults to config value)

    Returns:
        CohereEmbeddings instance
    """
    if model:
        logger.info(f"Creating embeddings model with specified model: {model}")
        return CohereEmbeddings(model=model)
    else:
        logger.info(f"Creating embeddings model with default model")
        return CohereEmbeddings()

def clear_vector_store(collection_name: str = COLLECTION_NAME) -> bool:
    """Clear an existing vector store collection.

    Args:
        collection_name: Name of the collection to clear

    Returns:
        bool: True if successful, False otherwise
    """
    logger.info(f"Clearing vector store collection: {collection_name}")
    try:
        persist_directory = VECTOR_DB_DIR
        if os.path.exists(persist_directory):
            # Remove the entire directory to ensure clean state
            shutil.rmtree(persist_directory)
            logger.info(f"Removed vector store directory: {persist_directory}")
            return True
        else:
            logger.info(f"Vector store directory does not exist: {persist_directory}")
            return True
    except Exception as e:
        logger.error(f"Error clearing vector store: {e}")
        return False

def create_vector_store(chunks: List[str], embeddings: Optional[Embeddings] = None,
                       collection_name: str = COLLECTION_NAME) -> Chroma:
    """Create a vector store from text chunks.

    Args:
        chunks: List of text chunks to store
        embeddings: Optional embeddings model (defaults to Ollama)
        collection_name: Name for the vector collection

    Returns:
        Chroma: The vector store instance

    Raises:
        Exception: If there's an error creating the vector store
    """
    logger.info(f"Creating vector store with {len(chunks)} chunks")
    try:
        # Create a directory for the database if it doesn't exist
        persist_directory = VECTOR_DB_DIR
        os.makedirs(persist_directory, exist_ok=True)

        # Clear existing collection to avoid dimension mismatch
        clear_vector_store(collection_name)

        # Get embeddings model if not provided
        if embeddings is None:
            embeddings = get_embeddings_model()

        # Create and persist the vector store
        logger.info(f"Creating Chroma vector store in {persist_directory}")
        vectordb = Chroma.from_texts(
            texts=chunks,
            embedding=embeddings,
            persist_directory=persist_directory,
            collection_name=collection_name
        )
        vectordb.persist()
        logger.info(f"Vector store created and persisted successfully")
        return vectordb
    except Exception as e:
        logger.error(f"Error creating vector store: {e}")
        raise Exception(f"Error creating vector store: {e}")

def get_vector_store(embeddings: Optional[Embeddings] = None,
                    collection_name: str = COLLECTION_NAME) -> Chroma:
    """Get an existing vector store.

    Args:
        embeddings: Optional embeddings model (defaults to Ollama)
        collection_name: Name of the collection to load

    Returns:
        Chroma: The vector store instance

    Raises:
        Exception: If there's an error loading the vector store
    """
    logger.info(f"Loading vector store collection: {collection_name}")
    try:
        persist_directory = VECTOR_DB_DIR

        if not os.path.exists(persist_directory):
            logger.warning(f"Vector store directory does not exist: {persist_directory}")
            raise Exception(f"Vector store directory does not exist: {persist_directory}")

        # Get embeddings model if not provided
        if embeddings is None:
            embeddings = get_embeddings_model()

        # Load the persisted vector store
        logger.info(f"Loading Chroma vector store from {persist_directory}")
        vectordb = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
            collection_name=collection_name
        )
        logger.info(f"Vector store loaded successfully")
        return vectordb
    except Exception as e:
        logger.error(f"Error loading vector store: {e}")
        raise Exception(f"Error loading vector store: {e}")

def similarity_search(vectordb: Chroma, query: str, k: int = DEFAULT_NUM_RESULTS) -> List[Document]:
    """Search for similar documents in the vector store.

    Args:
        vectordb: The vector store to search
        query: The query text
        k: Number of results to return

    Returns:
        List[Document]: List of similar documents

    Raises:
        Exception: If there's an error searching the vector store
    """
    logger.info(f"Performing similarity search for query: {query[:50]}...")
    try:
        logger.debug(f"Searching for {k} similar documents")
        docs = vectordb.similarity_search(query, k=k)
        logger.info(f"Found {len(docs)} similar documents")
        return docs
    except Exception as e:
        logger.error(f"Error searching vector store: {e}")
        raise Exception(f"Error searching vector store: {e}")
