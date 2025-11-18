from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.core.config import settings
from app.api import (
    auth, users, upload, ocr, 
    embeddings, search, rag, 
    summarize, format, transcription
)
from app.services.transcription_service import get_whisper_model
from app.services.embedding_service import get_model as get_embedding_model

# --- 1. DEFINE LIFESPAN (Pre-load models) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("-----------------------------------------------------")
    print("🚀 SYSTEM STARTUP: Pre-loading AI Models...")
    
    print("1. Loading Embedding Model...")
    get_embedding_model()
    print("✅ Embedding Model Ready.")
    
    print("2. Loading Whisper Model (This may take a moment)...")
    get_whisper_model()
    print("✅ Whisper Model Ready.")
    
    print("-----------------------------------------------------")
    yield
    print("🛑 SYSTEM SHUTDOWN")

# --- 2. APPLY LIFESPAN TO APP ---
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url="/docs", # Reverted to standard URL
    lifespan=lifespan,
    swagger_ui_parameters={"syntaxHighlight.theme": "obsidian"}
)

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    from fastapi.openapi.docs import get_swagger_ui_html
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
    )

# Mount all the API routers
app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
app.include_router(users.router, prefix="/users", tags=["Users"]) 
app.include_router(upload.router, prefix="/upload", tags=["Upload"])
app.include_router(ocr.router, prefix="/ocr", tags=["Processing"])
app.include_router(transcription.router, prefix="/transcription", tags=["Processing"])
app.include_router(embeddings.router, prefix="/embeddings", tags=["AI Utilities"])
app.include_router(search.router, prefix="/search", tags=["AI Utilities"])
app.include_router(rag.router, prefix="/rag", tags=["AI Utilities"])
app.include_router(summarize.router, prefix="/summarize", tags=["AI Utilities"])
app.include_router(format.router, prefix="/format", tags=["AI Utilities"])


@app.get("/", tags=["Root"])
async def read_root():
    return {"message": f"Welcome to the {settings.PROJECT_NAME}!"}