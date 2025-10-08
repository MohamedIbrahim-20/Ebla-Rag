"""
LLM Controller for handling local language model interactions.
Supports multiple local LLM backends including HuggingFace transformers.
"""

from .BaseController import BaseController
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import List, Dict, Optional
import torch
import logging

logger = logging.getLogger(__name__)

class LLMController(BaseController):
    """
    Controller for managing local LLM interactions.
    Supports text generation, question answering, and conversation.
    """
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        """
        Initialize the LLM controller with a specified model.
        
        Args:
            model_name (str): HuggingFace model name to use
        """
        super().__init__()
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
    def load_model(self) -> bool:
        """
        Load the language model and tokenizer.
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            logger.info(f"Loading model: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Add padding token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            # Create pipeline for easier text generation
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )
            
            logger.info("Model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def generate_response(self, prompt: str, max_length: int = 150, temperature: float = 0.7) -> str:
        """
        Generate a response to a given prompt.
        
        Args:
            prompt (str): Input prompt for generation
            max_length (int): Maximum length of generated text
            temperature (float): Sampling temperature (0.0 to 1.0)
            
        Returns:
            str: Generated response
        """
        if not self.pipeline:
            if not self.load_model():
                return "Error: Could not load the language model."
        
        try:
            # Generate response
            response = self.pipeline(
                prompt,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1
            )
            
            # Extract the generated text
            generated_text = response[0]['generated_text']
            
            # Remove the original prompt from the response
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"
    
    def generate_rag_response(self, question: str, context: str, max_length: int = 200) -> str:
        """
        Generate a response using RAG (Retrieval-Augmented Generation).
        
        Args:
            question (str): User's question
            context (str): Retrieved context/documents
            max_length (int): Maximum length of generated response
            
        Returns:
            str: Generated response based on context
        """
        # Create a prompt that includes the context
        prompt = f"""Context: {context}

Question: {question}

Answer:"""
        
        return self.generate_response(prompt, max_length=max_length, temperature=0.3)
    
    def is_model_loaded(self) -> bool:
        """
        Check if the model is loaded and ready to use.
        
        Returns:
            bool: True if model is loaded, False otherwise
        """
        return self.pipeline is not None and self.model is not None
    
    def get_model_info(self) -> Dict[str, str]:
        """
        Get information about the loaded model.
        
        Returns:
            Dict[str, str]: Model information
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "is_loaded": self.is_model_loaded(),
            "model_type": "causal_lm"
        }
