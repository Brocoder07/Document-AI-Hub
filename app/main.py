from fastapi import FastAPI
from app.core.config import settings
from app.api import (
    auth, users, upload, ocr, 
    embeddings, search, rag, 
    summarize, format, transcription
)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url="/docs", # <-- Explicitly set docs URL
    redoc_url=None, # <-- Disable Redoc if you don't use it
    swagger_ui_parameters={"syntaxHighlight.theme": "obsidian"} # Optional: theme
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
# Note: The 'prefix' assumes you move all router files to 'app/api/'
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