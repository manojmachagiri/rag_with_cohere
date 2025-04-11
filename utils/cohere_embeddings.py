"""Cohere embeddings implementation for LangChain.

This module provides a LangChain-compatible embeddings class for Cohere.
"""

import cohere
from typing import List, Dict, Any, Optional
from langchain.embeddings.base import Embeddings

from logger import setup_logger
from config import COHERE_API_KEY, COHERE_EMBEDDING_MODEL

# Set up logger
logger = setup_logger(__name__)

class CohereEmbeddings(Embeddings):
    """Cohere embeddings wrapper for LangChain.

    This class implements the Embeddings interface for LangChain,
    allowing Cohere models to be used for generating embeddings.
    """

    def __init__(self, api_key: str = COHERE_API_KEY, model: str = COHERE_EMBEDDING_MODEL):
        """Initialize with Cohere API.

        Args:
            api_key: The Cohere API key
            model: The name of the Cohere embedding model to use
        """
        self.api_key = api_key
        self.model = model
        self.client = cohere.Client(api_key)
        self.embedding_dim = 1024  # Default dimension for Cohere embeddings
        logger.info(f"Initialized CohereEmbeddings with model: {model}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents using Cohere API.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        logger.info(f"Embedding {len(texts)} documents")
        
        try:
            # Process in batches of 96 (Cohere API limit)
            batch_size = 96
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                if i % batch_size == 0 and i > 0:
                    logger.debug(f"Embedded {i}/{len(texts)} documents")
                
                response = self.client.embed(
                    texts=batch,
                    model=self.model,
                    input_type="search_document"
                )
                
                all_embeddings.extend(response.embeddings)
            
            # Update embedding dimension based on actual result
            if all_embeddings and len(all_embeddings[0]) > 0:
                self.embedding_dim = len(all_embeddings[0])
                
            logger.info(f"Successfully embedded {len(all_embeddings)} documents")
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Error embedding documents: {e}")
            # Return zero vectors as fallback
            return [[0.0] * self.embedding_dim for _ in range(len(texts))]

    def embed_query(self, text: str) -> List[float]:
        """Embed query text using Cohere API.

        Args:
            text: The query text to embed

        Returns:
            Embedding vector for the query
        """
        logger.info(f"Embedding query: {text[:50]}...")
        try:
            response = self.client.embed(
                texts=[text],
                model=self.model,
                input_type="search_query"  # Use search_query for queries
            )
            
            embedding = response.embeddings[0]
            
            # Update embedding dimension based on actual result
            if embedding and len(embedding) > 0:
                self.embedding_dim = len(embedding)
                
            return embedding
            
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            # Return a zero vector as fallback
            return [0.0] * self.embedding_dim
