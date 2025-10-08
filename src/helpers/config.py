from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
    
    APP_NAME: str = "Ebla-RAG"
    APP_VERSION: str = "1.0.0"
    
    FILE_ALLOWED_EXTENSIONS: list = ["application/pdf", "text/plain"]
    MAX_FILE_SIZE: int = 10  # in MB
    FILE_DEFAULT_CHUNK_SIZE: int = 8192  # in bytes
    
    # RAG Configuration
    DEFAULT_LLM_MODEL: str = "microsoft/DialoGPT-medium"
    DEFAULT_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    DEFAULT_CHUNK_SIZE: int = 100
    DEFAULT_CHUNK_OVERLAP: int = 20
    DEFAULT_TOP_K: int = 3
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


def get_settings():
    return Settings()