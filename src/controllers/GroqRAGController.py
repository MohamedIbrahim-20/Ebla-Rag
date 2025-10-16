"""
Groq RAG Controller - Based on existing RAG implementation
Integrates LangChain and LlamaIndex with Groq API for FastAPI endpoints
"""

import os
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import logging

logger = logging.getLogger(__name__)

# LangChain imports
try:
    from langchain_community.document_loaders import CSVLoader
    from langchain_community.vectorstores import FAISS
    from langchain_groq import ChatGroq
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.chains import RetrievalQA
    from langchain_core.documents import Document
    from langchain_core.prompts import PromptTemplate
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    logger.warning(f"LangChain imports failed: {e}")
    LANGCHAIN_AVAILABLE = False

# LlamaIndex imports
try:
    from llama_index.core import (
        VectorStoreIndex,
        Settings,
        StorageContext,
        load_index_from_storage,
        Document as LlamaDocument,
    )
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.llms.groq import Groq
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    LLAMAINDEX_AVAILABLE = True
except ImportError as e:
    logger.warning(f"LlamaIndex imports failed: {e}")
    LLAMAINDEX_AVAILABLE = False

from .BaseController import BaseController
from helpers.config import get_settings, Settings

class GroqRAGController(BaseController):
    """
    RAG Controller using Groq API with LangChain and LlamaIndex
    Based on the existing RAG implementation
    """
    
    def __init__(self, project_id: str = "2", settings: Settings = None):
        super().__init__()
        self.project_id = project_id
        self.project_path = os.path.join(self.files_dir, project_id)
        
        # Use provided settings or get default settings
        self.settings = settings or get_settings()
        
        # Configuration from settings
        self.GROQ_API_KEY = self.settings.GROQ_API_KEY
        self.LLM_MODEL = self.settings.DEFAULT_LLM_MODEL
        self.EMBEDDING_MODEL = self.settings.DEFAULT_EMBEDDING_MODEL
        self.CHUNK_SIZE = self.settings.DEFAULT_CHUNK_SIZE
        self.CHUNK_OVERLAP = self.settings.DEFAULT_CHUNK_OVERLAP
        self.TOP_K_RESULTS = self.settings.DEFAULT_TOP_K
        self.TEMPERATURE = 0.1
        
        # Storage paths
        self.LANGCHAIN_DB_PATH = os.path.join(self.project_path, "langchain_faiss")
        self.LLAMAINDEX_DB_PATH = os.path.join(self.project_path, "llamaindex_storage")
        
        # Initialize components
        self.llm = None
        self.embeddings = None
        self.vectorstore = None
        self.llamaindex_index = None
        
    def _configure_llamaindex(self):
        """Configure global LlamaIndex settings (like in your working version)"""
        try:
            from llama_index.core import Settings
            from llama_index.llms.groq import Groq
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            
            Settings.llm = Groq(model=self.LLM_MODEL, api_key=self.settings.GROQ_API_KEY, temperature=self.TEMPERATURE)
            Settings.embed_model = HuggingFaceEmbedding(model_name=self.EMBEDDING_MODEL)
            Settings.chunk_size = self.CHUNK_SIZE
            Settings.chunk_overlap = self.CHUNK_OVERLAP
            
            logger.info("LlamaIndex settings configured successfully")
        except Exception as e:
            logger.error(f"Error configuring LlamaIndex: {e}")
            raise e

    def initialize_models(self) -> bool:
        """Initialize Groq LLM and embedding models"""
        try:
            if not self.settings.GROQ_API_KEY:
                logger.error("GROQ_API_KEY not found in environment variables")
                return False
            
            if not LANGCHAIN_AVAILABLE:
                logger.error("LangChain not available. Please install required packages.")
                return False
            
            # Initialize Groq LLM
            if not self.settings.GROQ_API_KEY or self.settings.GROQ_API_KEY == "":
                logger.warning("GROQ_API_KEY not set - RAG functionality will be limited")
                self.llm = None
            else:
                self.llm = ChatGroq(
                    model=self.LLM_MODEL, 
                    temperature=self.TEMPERATURE, 
                    api_key=self.settings.GROQ_API_KEY
                )
            
            # Initialize embeddings
            self.embeddings = HuggingFaceEmbeddings(model_name=self.EMBEDDING_MODEL)
            
            logger.info("Groq models initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            return False
    
    def load_csv_to_documents(self, csv_path: str) -> Tuple[List[Document], List[LlamaDocument]]:
        """
        Load CSV and create documents for LangChain and LlamaIndex
        Based on the existing implementation
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        # Load CSV, fill NaNs with empty string
        df = pd.read_csv(csv_path).fillna("")
        
        # Combine title and content columns
        content_columns = ("title", "content")
        content_series = df[list(content_columns)].agg(" | ".join, axis=1)
        
        # Metadata = all other columns
        metadata_cols = [col for col in df.columns if col not in content_columns]
        metadata_list = df[metadata_cols].to_dict(orient="records")
        
        # Cast numeric metadata back to int where possible
        for meta in metadata_list:
            for key, val in meta.items():
                if isinstance(val, float) and val.is_integer():
                    meta[key] = int(val)
        
        # Create LangChain Documents
        langchain_docs = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(content_series, metadata_list)
        ]
        
        # Create LlamaIndex Documents
        llamaindex_docs = [
            LlamaDocument(text=text, metadata=meta)
            for text, meta in zip(content_series, metadata_list)
        ]
        
        logger.info(f"Loaded {len(langchain_docs)} documents from {csv_path}")
        return langchain_docs, llamaindex_docs
    
    def create_langchain_index(self, documents: List[Document]) -> bool:
        """Create or load LangChain FAISS index"""
        try:
            # Split documents into chunks with improved separators
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.CHUNK_SIZE,
                chunk_overlap=self.CHUNK_OVERLAP,
                separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ": ", " ", ""],
                length_function=len,
                is_separator_regex=False,
                keep_separator=True
            )
            chunks = text_splitter.split_documents(documents)
            logger.info(f"Split into {len(chunks)} chunks")
            
            # Create or load vector store
            if os.path.exists(self.LANGCHAIN_DB_PATH):
                logger.info(f"Loading existing vector store from {self.LANGCHAIN_DB_PATH}")
                self.vectorstore = FAISS.load_local(
                    self.LANGCHAIN_DB_PATH, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                # Add new documents to existing index instead of overwriting
                if chunks:
                    logger.info(f"Adding {len(chunks)} new chunks to existing index")
                    self.vectorstore.add_documents(chunks)
                    self.vectorstore.save_local(self.LANGCHAIN_DB_PATH)
            else:
                logger.info(f"Creating new vector store at {self.LANGCHAIN_DB_PATH}")
                os.makedirs(os.path.dirname(self.LANGCHAIN_DB_PATH), exist_ok=True)
                self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
                self.vectorstore.save_local(self.LANGCHAIN_DB_PATH)
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating LangChain index: {e}")
            return False
    
    def create_llamaindex_index(self, documents: List) -> bool:
        """Create or load LlamaIndex index"""
        try:
            if not LLAMAINDEX_AVAILABLE:
                logger.warning("LlamaIndex not available, skipping LlamaIndex creation")
                return True  # Return True to not block the process
            
            if not self.settings.ENABLE_LLAMAINDEX:
                logger.info("LlamaIndex disabled in configuration, skipping")
                return True
            
            # Configure LlamaIndex settings first (like in your working version)
            self._configure_llamaindex()
            
            # Load or create index
            if os.path.exists(self.LLAMAINDEX_DB_PATH):
                logger.info(f"Loading existing LlamaIndex from {self.LLAMAINDEX_DB_PATH}")
                storage_context = StorageContext.from_defaults(persist_dir=self.LLAMAINDEX_DB_PATH)
                self.llamaindex_index = load_index_from_storage(storage_context)
            else:
                logger.info(f"Creating new LlamaIndex at {self.LLAMAINDEX_DB_PATH}")
                # Create index with explicit embedding model
                self.llamaindex_index = VectorStoreIndex.from_documents(
                    documents, 
                    embed_model=embed_model
                )
                os.makedirs(self.LLAMAINDEX_DB_PATH, exist_ok=True)
                self.llamaindex_index.storage_context.persist(persist_dir=self.LLAMAINDEX_DB_PATH)
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating LlamaIndex: {e}")
            # Don't fail the entire process if LlamaIndex fails
            logger.warning("Continuing without LlamaIndex support")
            return True
    
    def index_documents(self, csv_file: str = "tech_100_long_real.csv") -> bool:
        """Index documents from CSV file"""
        try:
            csv_path = os.path.join(self.project_path, csv_file)
            
            # Load documents
            langchain_docs, llamaindex_docs = self.load_csv_to_documents(csv_path)
            
            # Create indexes - LangChain is required, LlamaIndex is optional
            langchain_success = self.create_langchain_index(langchain_docs)
            llamaindex_success = self.create_llamaindex_index(llamaindex_docs)
            
            # Return True if LangChain succeeds (LlamaIndex is optional)
            if langchain_success:
                if llamaindex_success:
                    logger.info("Both LangChain and LlamaIndex indexes created successfully")
                else:
                    logger.warning("LangChain index created successfully, LlamaIndex failed")
                return True
            else:
                logger.error("LangChain index creation failed")
                return False
            
        except Exception as e:
            logger.error(f"Error indexing documents: {e}")
            return False
    
    def search_documents(self, query: str, method: str = "langchain") -> List[Dict]:
        """Search for relevant documents"""
        try:
            if method == "langchain":
                # Load vectorstore if not already loaded
                if not self.vectorstore and os.path.exists(self.LANGCHAIN_DB_PATH):
                    self.vectorstore = FAISS.load_local(
                        self.LANGCHAIN_DB_PATH, 
                        self.embeddings,
                        allow_dangerous_deserialization=True
                    )
                
                if self.vectorstore:
                    # Use similarity search with score for better ranking
                    relevant_docs_with_scores = self.vectorstore.similarity_search_with_score(
                        query, k=self.TOP_K_RESULTS
                    )
                    
                    results = []
                    for doc, score in relevant_docs_with_scores:
                        # Lower score means higher similarity in FAISS
                        similarity_score = 1.0 / (1.0 + score)  # Convert to 0-1 range
                        results.append({
                            "content": doc.page_content,
                            "metadata": doc.metadata,
                            "score": similarity_score
                        })
                    
                    # Sort by score descending (highest similarity first)
                    results.sort(key=lambda x: x["score"], reverse=True)
                    return results
                else:
                    logger.error("LangChain vectorstore not available")
                    return []
                
            elif method == "llamaindex":
                # Load LlamaIndex if not already loaded
                if not self.llamaindex_index and os.path.exists(self.LLAMAINDEX_DB_PATH):
                    try:
                        # Configure LlamaIndex settings first (like in your working version)
                        self._configure_llamaindex()
                        
                        from llama_index.core import StorageContext, load_index_from_storage
                        storage_context = StorageContext.from_defaults(persist_dir=self.LLAMAINDEX_DB_PATH)
                        self.llamaindex_index = load_index_from_storage(storage_context)
                        logger.info("Loaded existing LlamaIndex from disk for search")
                    except Exception as e:
                        logger.error(f"Failed to load LlamaIndex for search: {e}")
                        return []
                
                if self.llamaindex_index:
                    query_engine = self.llamaindex_index.as_query_engine(
                        similarity_top_k=self.TOP_K_RESULTS,
                        response_mode="compact"
                    )
                    response = query_engine.query(query)
                    
                    # Extract source nodes
                    results = []
                    for node in response.source_nodes:
                        results.append({
                            "content": node.text,
                            "metadata": node.metadata,
                            "score": node.score
                        })
                    return results
                else:
                    logger.error("LlamaIndex not available")
                    return []
            
            return []
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    def ask_question_langchain(self, question: str) -> Dict:
        """Ask question using LangChain RAG"""
        try:
            if not self.llm:
                return {"error": "LLM not initialized. Please set GROQ_API_KEY in config.env"}
            
            # Load vectorstore if not already loaded
            if not self.vectorstore and os.path.exists(self.LANGCHAIN_DB_PATH):
                self.vectorstore = FAISS.load_local(
                    self.LANGCHAIN_DB_PATH, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            
            if not self.vectorstore:
                return {"error": "Vectorstore not available"}
            
            # Retrieve relevant documents
            retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.TOP_K_RESULTS}
            )
            relevant_docs = retriever.invoke(question)
            
            # Build context
            context = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs])
            
            # Create prompt
            prompt = f"""You are a helpful AI assistant. Answer the question based ONLY on the context provided below.

If the answer is not in the context, respond with: "I don't have enough information to answer that question."

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
            
            # Generate answer
            response = self.llm.invoke(question)
            answer = response.content
            
            return {
                "answer": answer,
                "context": context,
                "retrieved_documents": [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    }
                    for doc in relevant_docs
                ],
                "method": "langchain"
            }
            
        except Exception as e:
            logger.error(f"Error in LangChain RAG: {e}")
            return {"error": str(e)}
    
    def ask_question_llamaindex(self, question: str) -> Dict:
        """Ask question using LlamaIndex RAG"""
        try:
            if not LLAMAINDEX_AVAILABLE:
                return {"error": "LlamaIndex not available. Please install required packages."}
            
            if not self.settings.ENABLE_LLAMAINDEX:
                return {"error": "LlamaIndex is disabled in configuration."}
            
            if not self.llamaindex_index:
                # Configure LlamaIndex settings first (like in your working version)
                self._configure_llamaindex()
                
                # Try to load existing index
                if os.path.exists(self.LLAMAINDEX_DB_PATH):
                    try:
                        from llama_index.core import StorageContext, load_index_from_storage
                        storage_context = StorageContext.from_defaults(persist_dir=self.LLAMAINDEX_DB_PATH)
                        self.llamaindex_index = load_index_from_storage(storage_context)
                        logger.info("Loaded existing LlamaIndex from disk")
                    except Exception as e:
                        logger.error(f"Failed to load LlamaIndex: {e}")
                        return {"error": f"Failed to load LlamaIndex: {e}"}
                else:
                    return {"error": "LlamaIndex not found. Please run /index endpoint first."}
            
            # Create query engine
            query_engine = self.llamaindex_index.as_query_engine(
                similarity_top_k=self.TOP_K_RESULTS,
                response_mode="compact"
            )
            
            # Query
            response = query_engine.query(question)
            answer = str(response)
            
            # Extract source nodes
            retrieved_docs = []
            for node in response.source_nodes:
                retrieved_docs.append({
                    "content": node.text,
                    "metadata": node.metadata,
                    "score": node.score
                })
            
            return {
                "answer": answer,
                "retrieved_documents": retrieved_docs,
                "method": "llamaindex"
            }
            
        except Exception as e:
            logger.error(f"Error in LlamaIndex RAG: {e}")
            return {"error": str(e)}
    
    def get_system_status(self) -> Dict:
        """Get system status"""
        return {
            "project_id": self.project_id,
            "groq_api_configured": bool(self.GROQ_API_KEY),
            "models_initialized": bool(self.llm and self.embeddings),
            "langchain_index_exists": os.path.exists(self.LANGCHAIN_DB_PATH),
            "llamaindex_exists": os.path.exists(self.LLAMAINDEX_DB_PATH),
            "llm_model": self.LLM_MODEL,
            "embedding_model": self.EMBEDDING_MODEL
        }
