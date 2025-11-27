from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
import os

class Settings(BaseSettings):
    # Project
    PROJECT_NAME: str = "Document AI Hub"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    # Database
    DATABASE_URL: str = Field(..., validation_alias="DATABASE_URL")

    # Security
    SECRET_KEY: str = Field(..., validation_alias="SECRET_KEY")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    MESSAGE_ENCRYPTION_KEY: str = Field(..., validation_alias="MESSAGE_ENCRYPTION_KEY")

    # AI Models
    EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2"
    
    # Groq Configuration
    GROQ_API_KEY: str = Field(..., validation_alias="GROQ_API_KEY")
    GROQ_MODEL: str = Field(default="llama-3.1-8b-instant", validation_alias="GROQ_MODEL")

    # Paths
    CHROMA_PERSIST_DIR: str = "data/embeddings/chroma_db"
    UPLOAD_DIR: str = "data/documents"
    OCR_TEMP_DIR: str = "data/ocr_temp"

    # OCR Paths
    TESSERACT_PATH: str = "C:/Program Files/Tesseract-OCR/tesseract.exe"
    POPPLER_PATH: str = r"C:\Program Files\poppler-24.02.0\Library\bin"

    # Pydantic V2 Config
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore"
    )

settings = Settings()

# Ensure directories exist
os.makedirs(settings.OCR_TEMP_DIR, exist_ok=True)
os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)