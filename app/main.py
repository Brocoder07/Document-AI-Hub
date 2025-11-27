from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.core.config import settings
import warnings
# --- FIX: Direct Imports to avoid AttributeError ---
from app.api.auth import router as auth_router
from app.api.users import router as users_router
from app.api.upload import router as upload_router
from app.api.ocr import router as ocr_router
from fastapi.middleware.cors import CORSMiddleware
from app.api.transcription import router as transcription_router
from app.api.embeddings import router as embeddings_router
from app.api.search import router as search_router
from app.api.vector_db import db_client
from app.api.rag import router as rag_router
from app.api.summarize import router as summarize_router
from app.api.format import router as format_router
# ---------------------------------------------------

from app.services.transcription_service import get_whisper_model
from app.services.embedding_service import get_model as get_embedding_model

@asynccontextmanager
async def lifespan(app: FastAPI):
    # (Keep your existing lifespan logic)
    print("🚀 SYSTEM STARTUP")
    # get_embedding_model() # Uncomment if needed
    yield
    print("🛑 SYSTEM SHUTDOWN")
    db_client.close()

app = FastAPI(
    title=settings.PROJECT_NAME,
    version="3.0.0",
    lifespan=lifespan
)

warnings.filterwarnings("ignore", message="Accessing argon2.__version__ is deprecated")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For dev, allow all. For prod, use ["http://localhost:8501"]
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (POST, GET, etc.)
    allow_headers=["*"],  # Allow all headers
)

# --- FIX: Use the specific router variables ---
app.include_router(auth_router, prefix="/auth", tags=["Authentication"])
app.include_router(users_router, prefix="/users", tags=["Users"]) 
app.include_router(upload_router, prefix="/upload", tags=["Upload"])
app.include_router(ocr_router, prefix="/ocr", tags=["Processing"]) # <--- Fixed
app.include_router(transcription_router, prefix="/transcription", tags=["Processing"])
app.include_router(embeddings_router, prefix="/embeddings", tags=["AI Utilities"])
app.include_router(search_router, prefix="/search", tags=["AI Utilities"])
app.include_router(rag_router, prefix="/rag", tags=["AI Utilities"])
app.include_router(summarize_router, prefix="/summarize", tags=["AI Utilities"])
app.include_router(format_router, prefix="/format", tags=["AI Utilities"])

@app.get("/")
async def read_root():
    return {"message": "Welcome to AI Hub"}