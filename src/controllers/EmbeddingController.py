"""
Embedding Controller for handling document embeddings and vector storage.
Supports FAISS and ChromaDB for vector similarity search.
"""

from .BaseController import BaseController
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class EmbeddingController(BaseController):
    """
    Controller for managing document embeddings and vector storage.
    Handles embedding generation, storage, and similarity search.
    """
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding controller.
        
        Args:
            embedding_model (str): Sentence transformer model name
        """
        super().__init__()
        self.embedding_model_name = embedding_model
        self.embedding_model = None
        self.vector_index = None
        self.document_metadata = []
        self.embedding_dimension = 384  # Default for all-MiniLM-L6-v2
        
    def load_embedding_model(self) -> bool:
        """
        Load the sentence transformer model for embeddings.
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            self.embedding_dimension = self.embedding_model.get_sentence_embedding_dimension()
            logger.info(f"Embedding model loaded. Dimension: {self.embedding_dimension}")
            return True
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            return False
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts (List[str]): List of text documents to embed
            
        Returns:
            np.ndarray: Array of embeddings
        """
        if not self.embedding_model:
            if not self.load_embedding_model():
                raise Exception("Could not load embedding model")
        
        try:
            embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
            logger.info(f"Generated embeddings for {len(texts)} documents")
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def create_vector_index(self, embeddings: np.ndarray) -> bool:
        """
        Create a FAISS vector index from embeddings.
        
        Args:
            embeddings (np.ndarray): Array of embeddings
            
        Returns:
            bool: True if index created successfully, False otherwise
        """
        try:
            # Create FAISS index
            self.vector_index = faiss.IndexFlatIP(self.embedding_dimension)  # Inner product for cosine similarity
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Add embeddings to index
            self.vector_index.add(embeddings.astype('float32'))
            
            logger.info(f"Created FAISS index with {self.vector_index.ntotal} vectors")
            return True
            
        except Exception as e:
            logger.error(f"Error creating vector index: {e}")
            return False
    
    def add_documents(self, documents: List[Dict[str, str]]) -> bool:
        """
        Add documents to the vector index.
        
        Args:
            documents (List[Dict[str, str]]): List of documents with 'content' and 'metadata'
            
        Returns:
            bool: True if documents added successfully, False otherwise
        """
        try:
            # Extract text content
            texts = [doc['content'] for doc in documents]
            
            # Generate embeddings
            embeddings = self.generate_embeddings(texts)
            
            # Create or update index
            if self.vector_index is None:
                if not self.create_vector_index(embeddings):
                    return False
            else:
                # Add to existing index
                faiss.normalize_L2(embeddings)
                self.vector_index.add(embeddings.astype('float32'))
            
            # Store metadata
            self.document_metadata.extend(documents)
            
            logger.info(f"Added {len(documents)} documents to index")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return False
    
    def search_similar(self, query: str, top_k: int = 5) -> List[Dict[str, any]]:
        """
        Search for similar documents using vector similarity.
        
        Args:
            query (str): Search query
            top_k (int): Number of top results to return
            
        Returns:
            List[Dict[str, any]]: List of similar documents with scores
        """
        if not self.vector_index or not self.embedding_model:
            logger.error("Vector index or embedding model not loaded")
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.generate_embeddings([query])
            faiss.normalize_L2(query_embedding)
            
            # Search for similar vectors
            scores, indices = self.vector_index.search(query_embedding.astype('float32'), top_k)
            
            # Prepare results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.document_metadata):
                    result = {
                        'document': self.document_metadata[idx],
                        'score': float(score),
                        'index': int(idx)
                    }
                    results.append(result)
            
            logger.info(f"Found {len(results)} similar documents for query")
            return results
            
        except Exception as e:
            logger.error(f"Error searching similar documents: {e}")
            return []
    
    def save_index(self, filepath: str) -> bool:
        """
        Save the vector index and metadata to disk.
        
        Args:
            filepath (str): Path to save the index
            
        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.vector_index, f"{filepath}.index")
            
            # Save metadata
            with open(f"{filepath}.metadata", 'wb') as f:
                pickle.dump(self.document_metadata, f)
            
            logger.info(f"Saved index and metadata to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving index: {e}")
            return False
    
    def load_index(self, filepath: str) -> bool:
        """
        Load the vector index and metadata from disk.
        
        Args:
            filepath (str): Path to load the index from
            
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        try:
            # Load FAISS index
            self.vector_index = faiss.read_index(f"{filepath}.index")
            
            # Load metadata
            with open(f"{filepath}.metadata", 'rb') as f:
                self.document_metadata = pickle.load(f)
            
            logger.info(f"Loaded index with {self.vector_index.ntotal} vectors")
            return True
            
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            return False
    
    def get_index_info(self) -> Dict[str, any]:
        """
        Get information about the current index.
        
        Returns:
            Dict[str, any]: Index information
        """
        return {
            "embedding_model": self.embedding_model_name,
            "embedding_dimension": self.embedding_dimension,
            "total_documents": len(self.document_metadata),
            "index_size": self.vector_index.ntotal if self.vector_index else 0,
            "is_loaded": self.vector_index is not None
        }
