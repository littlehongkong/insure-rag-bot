import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    OPENAI_API_KEY: str = ""
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL: str = "gpt-3.5-turbo"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    class Config:
        env_file = ".env"

settings = Settings()
