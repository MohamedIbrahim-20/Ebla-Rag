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
    DEFAULT_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    DEFAULT_CHUNK_SIZE: int = 500  # Smaller chunks for better granularity
    DEFAULT_CHUNK_OVERLAP: int = 150  # More overlap to avoid missing context
    DEFAULT_TOP_K: int = 10  # Retrieve more documents for better coverage
    
    # Groq API Configuration
    GROQ_API_KEY: str = ""
    
    # RAG Method Configuration
    ENABLE_LLAMAINDEX: bool = True
    
    class Config:
        env_file = ".env"  # Use a different name that's not blocked
        env_file_encoding = "utf-8"


def get_settings():
    return Settings()