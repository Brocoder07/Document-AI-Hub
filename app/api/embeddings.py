from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from typing import Union, List
from app.api.dependencies import get_current_active_user, UserInDB
from app.services.embedding_service import embed_texts

router = APIRouter()

# --- Pydantic Schemas ---

class EmbeddingRequest(BaseModel):
    text: Union[str, List[str]]

class EmbeddingResponse(BaseModel):
    embedding: Union[List[float], List[List[float]]]
    model: str

# --- Endpoints ---

@router.post("/generate", response_model=EmbeddingResponse)
async def generate_embeddings_endpoint(
    request: EmbeddingRequest,
    current_user: UserInDB = Depends(get_current_active_user)
):
    """
    Generates sentence embeddings for a given text or list of texts.
    """
    try:
        # Determine if single or batch
        if isinstance(request.text, str):
            embeddings = embed_texts([request.text])[0]
        else:
            embeddings = embed_texts(request.text)
            
        return EmbeddingResponse(
            embedding=embeddings,
            model="sentence-transformers/all-mpnet-base-v2"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate embeddings: {str(e)}"
        )