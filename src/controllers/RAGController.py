"""
RAG Controller for combining retrieval and generation.
Integrates document retrieval with LLM response generation.
"""

from .BaseController import BaseController
from .LLMController import LLMController
from .EmbeddingController import EmbeddingController
from .ProcessController import ProcessController
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class RAGController(BaseController):
    """
    Controller for Retrieval-Augmented Generation (RAG).
    Combines document retrieval with LLM response generation.
    """
    
    def __init__(self, project_id: str, llm_model: str = "microsoft/DialoGPT-medium", embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize the RAG controller.
        
        Args:
            project_id (str): Project identifier
            llm_model (str): LLM model name for generation
            embedding_model (str): Embedding model name for retrieval
        """
        super().__init__()
        self.project_id = project_id
        self.llm_controller = LLMController(model_name=llm_model)
        self.embedding_controller = EmbeddingController(embedding_model=embedding_model)
        self.process_controller = ProcessController(project_id=project_id)
        
    def initialize_models(self) -> bool:
        """
        Initialize both LLM and embedding models.
        
        Returns:
            bool: True if both models initialized successfully, False otherwise
        """
        try:
            logger.info("Initializing RAG models...")
            
            # Load LLM model
            if not self.llm_controller.load_model():
                logger.error("Failed to load LLM model")
                return False
            
            # Load embedding model
            if not self.embedding_controller.load_embedding_model():
                logger.error("Failed to load embedding model")
                return False
            
            logger.info("RAG models initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing RAG models: {e}")
            return False
    
    def index_documents(self, file_ids: List[str], chunk_size: int = 100, chunk_overlap: int = 20) -> bool:
        """
        Index documents for retrieval.
        
        Args:
            file_ids (List[str]): List of file IDs to index
            chunk_size (int): Size of text chunks
            chunk_overlap (int): Overlap between chunks
            
        Returns:
            bool: True if indexing successful, False otherwise
        """
        try:
            logger.info(f"Indexing {len(file_ids)} documents...")
            
            all_documents = []
            
            for file_id in file_ids:
                # Get file content
                file_content = self.process_controller.get_file_content(filename=file_id)
                if not file_content:
                    logger.warning(f"Could not load content for file: {file_id}")
                    continue
                
                # Process file content into chunks
                chunks = self.process_controller.process_file_content(
                    file_content=file_content,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                
                if not chunks:
                    logger.warning(f"No chunks created for file: {file_id}")
                    continue
                
                # Convert chunks to document format
                for i, chunk in enumerate(chunks):
                    document = {
                        'content': chunk.page_content,
                        'metadata': {
                            'file_id': file_id,
                            'chunk_index': i,
                            'source': chunk.metadata.get('source', 'unknown'),
                            **chunk.metadata
                        }
                    }
                    all_documents.append(document)
            
            if not all_documents:
                logger.error("No documents to index")
                return False
            
            # Add documents to embedding index
            success = self.embedding_controller.add_documents(all_documents)
            
            if success:
                logger.info(f"Successfully indexed {len(all_documents)} document chunks")
            
            return success
            
        except Exception as e:
            logger.error(f"Error indexing documents: {e}")
            return False
    
    def ask_question(self, question: str, top_k: int = 3, max_response_length: int = 200) -> Dict[str, any]:
        """
        Ask a question using RAG (retrieve relevant documents and generate answer).
        
        Args:
            question (str): User's question
            top_k (int): Number of relevant documents to retrieve
            max_response_length (int): Maximum length of generated response
            
        Returns:
            Dict[str, any]: Response with answer, retrieved documents, and metadata
        """
        try:
            logger.info(f"Processing question: {question}")
            
            # Step 1: Retrieve relevant documents
            similar_docs = self.embedding_controller.search_similar(question, top_k=top_k)
            
            if not similar_docs:
                return {
                    'answer': "I couldn't find any relevant information to answer your question.",
                    'retrieved_documents': [],
                    'metadata': {
                        'question': question,
                        'retrieved_count': 0,
                        'error': 'No relevant documents found'
                    }
                }
            
            # Step 2: Prepare context from retrieved documents
            context_parts = []
            for doc_result in similar_docs:
                doc = doc_result['document']
                context_parts.append(f"Document: {doc['content']}")
            
            context = "\n\n".join(context_parts)
            
            # Step 3: Generate answer using LLM with context
            answer = self.llm_controller.generate_rag_response(
                question=question,
                context=context,
                max_length=max_response_length
            )
            
            # Step 4: Prepare response
            response = {
                'answer': answer,
                'retrieved_documents': [
                    {
                        'content': doc_result['document']['content'],
                        'metadata': doc_result['document']['metadata'],
                        'similarity_score': doc_result['score']
                    }
                    for doc_result in similar_docs
                ],
                'metadata': {
                    'question': question,
                    'retrieved_count': len(similar_docs),
                    'context_length': len(context),
                    'llm_model': self.llm_controller.model_name,
                    'embedding_model': self.embedding_controller.embedding_model_name
                }
            }
            
            logger.info(f"Generated answer with {len(similar_docs)} retrieved documents")
            return response
            
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            return {
                'answer': f"Error processing your question: {str(e)}",
                'retrieved_documents': [],
                'metadata': {
                    'question': question,
                    'retrieved_count': 0,
                    'error': str(e)
                }
            }
    
    def get_system_status(self) -> Dict[str, any]:
        """
        Get the current status of the RAG system.
        
        Returns:
            Dict[str, any]: System status information
        """
        return {
            'project_id': self.project_id,
            'llm_status': self.llm_controller.get_model_info(),
            'embedding_status': self.embedding_controller.get_index_info(),
            'models_initialized': (
                self.llm_controller.is_model_loaded() and 
                self.embedding_controller.embedding_model is not None
            )
        }
    
    def save_index(self, filepath: str) -> bool:
        """
        Save the current embedding index.
        
        Args:
            filepath (str): Path to save the index
            
        Returns:
            bool: True if saved successfully, False otherwise
        """
        return self.embedding_controller.save_index(filepath)
    
    def load_index(self, filepath: str) -> bool:
        """
        Load an existing embedding index.
        
        Args:
            filepath (str): Path to load the index from
            
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        return self.embedding_controller.load_index(filepath)
