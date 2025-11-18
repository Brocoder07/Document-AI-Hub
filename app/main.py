from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from fastapi.openapi.docs import get_swagger_ui_html
from contextlib import asynccontextmanager
from app.core.config import settings
from app.api import (
    auth, users, upload, ocr, 
    embeddings, search, rag, 
    summarize, format, transcription
)
from app.services.transcription_service import get_whisper_model
from app.services.embedding_service import get_model as get_embedding_model

# --- 1. DEFINE LIFESPAN (Pre-load models here) ---
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
    docs_url=None, # <--- DISABLE DEFAULT DOCS
    redoc_url=None,
    lifespan=lifespan,
    swagger_ui_parameters={"syntaxHighlight.theme": "obsidian"}
)

# --- 3. CUSTOM DOCS ENDPOINTS ---

def custom_openapi_user():
    """
    Generates an OpenAPI schema that EXCLUDES any route tagged with 'Admin'.
    """
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=app.title + " (User)",
        version=app.version,
        routes=app.routes,
    )
    
    # Filter out paths that have the "Admin" tag
    paths = openapi_schema.get("paths", {})
    filtered_paths = {}
    
    for path, methods in paths.items():
        new_methods = {}
        for method, details in methods.items():
            tags = details.get("tags", [])
            if "Admin" not in tags:
                new_methods[method] = details
        if new_methods:
            filtered_paths[path] = new_methods
            
    openapi_schema["paths"] = filtered_paths
    return openapi_schema

def custom_openapi_admin():
    """
    Generates the full OpenAPI schema (includes Admin routes).
    """
    return get_openapi(
        title=app.title + " (Admin)",
        version=app.version,
        routes=app.routes,
    )

# Serve the filtered JSON for Users
@app.get("/user/openapi.json", include_in_schema=False)
async def get_user_openapi_json():
    return custom_openapi_user()

# Serve the full JSON for Admins
@app.get("/admin/openapi.json", include_in_schema=False)
async def get_admin_openapi_json():
    return custom_openapi_admin()

# --- 4. THE TWO UI PAGES ---

@app.get("/user/docs", include_in_schema=False)
async def user_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url="/user/openapi.json", # Point to filtered schema
        title=app.title + " - User Docs",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
    )

@app.get("/admin/docs", include_in_schema=False)
async def admin_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url="/admin/openapi.json", # Point to full schema
        title=app.title + " - Admin Docs",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
    )


# Mount all the API routers
app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
app.include_router(users.router, prefix="/users", tags=["Users"]) # Admin routes inside will carry "Users" AND "Admin" tags
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