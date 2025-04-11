"""
Unit tests for the vector store module.
"""

import os
import unittest
import shutil
from unittest.mock import MagicMock, patch

from utils.vector_store import (
    OllamaEmbeddings, 
    get_embeddings_model, 
    clear_vector_store, 
    create_vector_store, 
    get_vector_store, 
    similarity_search
)

class TestVectorStore(unittest.TestCase):
    """Test cases for vector store functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock for the vector DB directory
        self.test_dir = os.path.join(os.getcwd(), "test_chroma_db")
        
        # Create test chunks
        self.test_chunks = [
            "This is the first test chunk.",
            "This is the second test chunk with different content.",
            "And here is a third chunk with some more information."
        ]
        
        # Create a mock embedding
        self.mock_embedding = [0.1] * 1536
        
    def tearDown(self):
        """Clean up after tests."""
        # Remove test directory if it exists
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    @patch('httpx.Client')
    def test_ollama_embeddings(self, mock_client):
        """Test the OllamaEmbeddings class."""
        # Set up the mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"embedding": self.mock_embedding}
        
        mock_client_instance = MagicMock()
        mock_client_instance.post.return_value = mock_response
        mock_client.return_value.__enter__.return_value = mock_client_instance
        
        # Create an instance of OllamaEmbeddings
        embeddings = OllamaEmbeddings(base_url="http://test-url", model="test-model")
        
        # Test embed_documents
        result = embeddings.embed_documents(["Test text"])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], self.mock_embedding)
        
        # Test embed_query
        query_result = embeddings.embed_query("Test query")
        self.assertEqual(query_result, self.mock_embedding)
        
        # Test error handling
        mock_response.status_code = 500
        mock_response.text = "Error"
        
        # Should return a zero vector on error
        error_result = embeddings.embed_query("Error query")
        self.assertEqual(len(error_result), 1536)
        self.assertEqual(error_result, [0.0] * 1536)
    
    @patch('utils.vector_store.OllamaEmbeddings')
    def test_get_embeddings_model(self, mock_embeddings_class):
        """Test getting an embeddings model."""
        # Set up the mock
        mock_embeddings_instance = MagicMock()
        mock_embeddings_class.return_value = mock_embeddings_instance
        
        # Call the function
        result = get_embeddings_model()
        
        # Assertions
        self.assertEqual(result, mock_embeddings_instance)
        mock_embeddings_class.assert_called_once()
        
        # Test with specific model
        result_with_model = get_embeddings_model(model="specific-model")
        mock_embeddings_class.assert_called_with(model="specific-model")
    
    @patch('os.path.exists')
    @patch('shutil.rmtree')
    def test_clear_vector_store(self, mock_rmtree, mock_exists):
        """Test clearing a vector store."""
        # Set up the mocks
        mock_exists.return_value = True
        
        # Call the function
        result = clear_vector_store(collection_name="test_collection")
        
        # Assertions
        self.assertTrue(result)
        mock_rmtree.assert_called_once()
        
        # Test when directory doesn't exist
        mock_exists.return_value = False
        result_not_exists = clear_vector_store(collection_name="test_collection")
        self.assertTrue(result_not_exists)
    
    @patch('utils.vector_store.clear_vector_store')
    @patch('utils.vector_store.Chroma')
    @patch('utils.vector_store.get_embeddings_model')
    @patch('os.makedirs')
    def test_create_vector_store(self, mock_makedirs, mock_get_embeddings, mock_chroma, mock_clear):
        """Test creating a vector store."""
        # Set up the mocks
        mock_embeddings = MagicMock()
        mock_get_embeddings.return_value = mock_embeddings
        
        mock_vectordb = MagicMock()
        mock_chroma.from_texts.return_value = mock_vectordb
        
        mock_clear.return_value = True
        
        # Call the function
        result = create_vector_store(self.test_chunks, collection_name="test_collection")
        
        # Assertions
        self.assertEqual(result, mock_vectordb)
        mock_makedirs.assert_called_once()
        mock_clear.assert_called_once_with("test_collection")
        mock_chroma.from_texts.assert_called_once()
        mock_vectordb.persist.assert_called_once()
    
    @patch('utils.vector_store.Chroma')
    @patch('utils.vector_store.get_embeddings_model')
    @patch('os.path.exists')
    def test_get_vector_store(self, mock_exists, mock_get_embeddings, mock_chroma):
        """Test getting a vector store."""
        # Set up the mocks
        mock_exists.return_value = True
        
        mock_embeddings = MagicMock()
        mock_get_embeddings.return_value = mock_embeddings
        
        mock_vectordb = MagicMock()
        mock_chroma.return_value = mock_vectordb
        
        # Call the function
        result = get_vector_store(collection_name="test_collection")
        
        # Assertions
        self.assertEqual(result, mock_vectordb)
        mock_chroma.assert_called_once()
        
        # Test when directory doesn't exist
        mock_exists.return_value = False
        with self.assertRaises(Exception):
            get_vector_store(collection_name="test_collection")
    
    def test_similarity_search(self):
        """Test similarity search."""
        # Create a mock vector store
        mock_vectordb = MagicMock()
        mock_docs = [MagicMock(), MagicMock()]
        mock_vectordb.similarity_search.return_value = mock_docs
        
        # Call the function
        result = similarity_search(mock_vectordb, "test query", k=2)
        
        # Assertions
        self.assertEqual(result, mock_docs)
        mock_vectordb.similarity_search.assert_called_once_with("test query", k=2)
        
        # Test error handling
        mock_vectordb.similarity_search.side_effect = Exception("Test error")
        with self.assertRaises(Exception):
            similarity_search(mock_vectordb, "error query")

if __name__ == '__main__':
    unittest.main()
