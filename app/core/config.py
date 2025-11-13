from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    # Project
    PROJECT_NAME: str = "Document AI Hub"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    # Database - PostgreSQL
    DATABASE_URL: str = "postgresql://ai_user:password@localhost/document_ai_hub"

    # Security
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # AI Models
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    TEXT_GEN_MODEL: str = "distilgpt2"           # lightweight CPU-safe generator
    MAX_NEW_TOKENS: int = 150

    # ChromaDB
    CHROMA_PERSIST_DIR: str = "data/embeddings/chroma_db"

    # File Storage
    UPLOAD_DIR: str = "data/documents"
    OCR_TEMP_DIR: str = "data/ocr_temp"

    # OCR
    TESSERACT_PATH: str = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()

# Ensure directories exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.OCR_TEMP_DIR, exist_ok=True)
os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)
