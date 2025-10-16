from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
    
    APP_NAME: str = "Ebla-RAG"
    APP_VERSION: str = "1.0.0"
    
    FILE_ALLOWED_EXTENSIONS: list = ["application/pdf", "text/plain"]
    MAX_FILE_SIZE: int = 10  # in MB
    FILE_DEFAULT_CHUNK_SIZE: int = 8192  # in bytes
    
    # RAG Configuration
    DEFAULT_LLM_MODEL: str = "llama-3.3-70b-versatile"
    DEFAULT_EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2"  # Better quality embeddings
    DEFAULT_CHUNK_SIZE: int = 800  # Optimized chunk size for better context
    DEFAULT_CHUNK_OVERLAP: int = 150  # Increased overlap for better continuity
    DEFAULT_TOP_K: int = 7  # More results for better context
    
    # Groq API Configuration
    GROQ_API_KEY: str = ""
    
    # RAG Method Configuration
    ENABLE_LLAMAINDEX: bool = True
    
    class Config:
        env_file = ".env"  # Use a different name that's not blocked
        env_file_encoding = "utf-8"


def get_settings():
    return Settings()