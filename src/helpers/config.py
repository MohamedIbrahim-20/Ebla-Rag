from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
    
    APP_NAME: str
    APP_VERSION: str
    
    FILE_ALLOWED_EXTENSIONS: list
    MAX_FILE_SIZE: int  # in MB
    FILE_DEFAULT_CHUNK_SIZE: int  # in bytes
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


def get_settings():
    return Settings()