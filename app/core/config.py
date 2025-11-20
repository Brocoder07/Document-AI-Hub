from pydantic_settings import BaseSettings
from pydantic import Field
import os

class Settings(BaseSettings):
    # Project
    PROJECT_NAME: str = "Document AI Hub"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    # Database - PostgreSQL (now from .env)
    DATABASE_URL: str = Field(..., env="DATABASE_URL")

    # Security (now from .env)
    SECRET_KEY: str = Field(..., env="SECRET_KEY")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # AI Models
    EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2"
    
    # Groq Configuration (now from .env)
    GROQ_API_KEY: str = Field(..., env="GROQ_API_KEY")
    GROQ_MODEL: str = Field(default="llama-3.1-8b-instant", env="GROQ_MODEL")

    # ChromaDB
    CHROMA_PERSIST_DIR: str = "data/embeddings/chroma_db"

    # File Storage
    UPLOAD_DIR: str = "data/documents"
    OCR_TEMP_DIR: str = "data/ocr_temp"

    # OCR - FIX: Use forward slashes to avoid "\t" being read as a Tab character
    TESSERACT_PATH: str = "C:/Program Files/Tesseract-OCR/tesseract.exe"
    POPPLER_PATH: str = r"C:\Program Files\poppler-24.02.0\Library\bin"

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"

settings = Settings()

# Ensure directories exist
os.makedirs(settings.OCR_TEMP_DIR, exist_ok=True)
os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)