from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from app.api.dependencies import get_current_active_user, UserInDB
from app.services.embedding_service import embed_texts

router = APIRouter()

# --- Pydantic Schemas ---

class EmbeddingRequest(BaseModel):
    text: str | list[str]

class EmbeddingResponse(BaseModel):
    embedding: list[float] | list[list[float]]
    model: str

# --- Endpoints ---

@router.post("/generate", response_model=EmbeddingResponse)
async def generate_embeddings(
    request: EmbeddingRequest,
    current_user: UserInDB = Depends(get_current_active_user) # Secure endpoint
):
    """
    Generates sentence embeddings for a given text or list of texts.
    """
    try:
        if isinstance(request.text, str):
            # Single text input
            embeddings = embed_texts([request.text])[0]
        else:
            # Batch text input
            embeddings = embed_texts(request.text)
            
        return EmbeddingResponse(
            embedding=embeddings,
            model="sentence-transformers/all-MiniLM-L6-v2" # From config
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate embeddings: {str(e)}"
        )