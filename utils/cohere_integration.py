"""Cohere integration utilities for the RAG application.

This module provides functions and classes for:
- Connecting to the Cohere API
- Generating text responses
- Working with embeddings
"""

import cohere
from typing import List, Dict, Any, Optional, Tuple, Union

from logger import setup_logger
from config import (
    COHERE_API_KEY, COHERE_EMBEDDING_MODEL, COHERE_GENERATION_MODEL, 
    COHERE_TIMEOUT, SYSTEM_PROMPT, RAG_PROMPT_TEMPLATE
)

# Set up logger
logger = setup_logger(__name__)

class CohereClient:
    def __init__(self, api_key: str = COHERE_API_KEY, 
                 generation_model: str = COHERE_GENERATION_MODEL,
                 embedding_model: str = COHERE_EMBEDDING_MODEL):
        """Initialize the Cohere client with API key and models.

        Args:
            api_key: The Cohere API key
            generation_model: The name of the Cohere generation model to use
            embedding_model: The name of the Cohere embedding model to use
        """
        self.api_key = api_key
        self.generation_model = generation_model
        self.embedding_model = embedding_model
        self.client = cohere.Client(api_key)
        logger.info(f"Initialized CohereClient with generation model: {generation_model} and embedding model: {embedding_model}")

    def list_models(self) -> List[Dict[str, Any]]:
        """List available models.

        Returns:
            List of model information dictionaries

        Note:
            This is a compatibility method that returns a fixed list of Cohere models
        """
        logger.info("Listing available Cohere models")
        # Return a fixed list of Cohere models for compatibility with the UI
        models = [
            {"name": "command"},
            {"name": "command-light"},
            {"name": "command-nightly"},
            {"name": "command-r"},
            {"name": "command-r-plus"}
        ]
        logger.info(f"Returning {len(models)} available Cohere models")
        return models

    def generate(self, prompt: str, system_prompt: Optional[str] = None,
                context: Optional[List[int]] = None) -> Tuple[str, Optional[List[int]]]:
        """Generate a response from the model.

        Args:
            prompt: The prompt text to send to the model
            system_prompt: Optional system prompt for instruction
            context: Optional context (ignored, for compatibility)

        Returns:
            Tuple containing (response_text, None)

        Raises:
            Exception: If there's an error generating a response
        """
        logger.info(f"Generating response for prompt: {prompt[:50]}...")
        try:
            # Prepare the message for Cohere
            messages = []
            
            # Add system message if provided
            if system_prompt:
                messages.append({"role": "SYSTEM", "message": system_prompt})
                logger.debug(f"Using system prompt: {system_prompt[:50]}...")
            
            # Add user message
            messages.append({"role": "USER", "message": prompt})
            
            # Generate response
            response = self.client.chat(
                message=prompt,
                model=self.generation_model,
                preamble=system_prompt,
                temperature=0.7,
            )
            
            response_text = response.text
            logger.info(f"Generated response of length {len(response_text)}")
            
            # Return response and None for context (for compatibility)
            return response_text, None
            
        except Exception as e:
            error_msg = f"Error generating response: {e}"
            logger.error(error_msg)
            raise Exception(error_msg)

    def get_embedding(self, text: str) -> List[float]:
        """Get embeddings from Cohere API.

        Args:
            text: The text to embed

        Returns:
            List of embedding values

        Raises:
            Exception: If there's an error getting embeddings
        """
        logger.info(f"Getting embedding for text: {text[:50]}...")
        try:
            response = self.client.embed(
                texts=[text],
                model=self.embedding_model,
                input_type="search_document"
            )
            
            embedding = response.embeddings[0]
            logger.info(f"Got embedding of dimension {len(embedding)}")
            return embedding
            
        except Exception as e:
            error_msg = f"Error getting embedding: {e}"
            logger.error(error_msg)
            raise Exception(error_msg)

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors

        Raises:
            Exception: If there's an error getting embeddings
        """
        logger.info(f"Getting embeddings for {len(texts)} texts")
        try:
            # Process in batches of 96 (Cohere API limit)
            batch_size = 96
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                logger.debug(f"Processing batch {i//batch_size + 1} with {len(batch)} texts")
                
                response = self.client.embed(
                    texts=batch,
                    model=self.embedding_model,
                    input_type="search_document"
                )
                
                all_embeddings.extend(response.embeddings)
                
            logger.info(f"Got embeddings for {len(all_embeddings)} texts")
            return all_embeddings
            
        except Exception as e:
            error_msg = f"Error getting embeddings: {e}"
            logger.error(error_msg)
            raise Exception(error_msg)

    def generate_with_context(self, query: str, context_docs: List[Any],
                            system_prompt: Optional[str] = SYSTEM_PROMPT) -> str:
        """Generate a response using RAG with context documents.

        Args:
            query: The user's question
            context_docs: List of document objects with relevant context
            system_prompt: Optional system prompt (defaults to config)

        Returns:
            Generated response text

        Raises:
            Exception: If there's an error generating a response
        """
        logger.info(f"Generating response for query with {len(context_docs)} context documents")
        try:
            # Format context from retrieved documents
            context_text = "\n\n".join([doc.page_content for doc in context_docs])
            logger.debug(f"Combined context length: {len(context_text)} characters")

            # Create a prompt with the context and query using template from config
            rag_prompt = RAG_PROMPT_TEMPLATE.format(context=context_text, query=query)

            # Generate response
            logger.info("Sending RAG prompt to model")
            response, _ = self.generate(rag_prompt, system_prompt)
            return response
        except Exception as e:
            error_msg = f"Error generating response with context: {e}"
            logger.error(error_msg)
            raise Exception(error_msg)
