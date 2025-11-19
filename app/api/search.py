from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from app.api.dependencies import get_current_active_user, UserInDB
from app.services.rag_service import retrieve_with_scores 
from typing import Dict, Any

router = APIRouter()

# --- Pydantic Schemas ---

class SearchQueryRequest(BaseModel):
    query: str
    top_k: int = Field(default=3, gt=0, le=10)
    mode: str = "general" 

class RetrievedDoc(BaseModel):
    document_id: str 
    score: float
    metadata: Dict[str, Any] # <--- NEW FIELD

class SearchResponse(BaseModel):
    results: list[RetrievedDoc]

# --- Endpoints ---

@router.post("/similarity", response_model=SearchResponse)
def search_similar_documents(
    request: SearchQueryRequest,
    current_user: UserInDB = Depends(get_current_active_user)
):
    """
    Performs a vector similarity search on the user's documents.
    Returns document IDs, scores, and METADATA (filename, etc.).
    """
    try:
        user_id = current_user.email
        
        # retrieve_with_scores returns: (results, retrieval_time, scores)
        results_list, _, _ = retrieve_with_scores(
            query=request.query,
            user_id=user_id,
            file_id=None, 
            mode=request.mode, 
            top_k=request.top_k
        )
        
        # Format for the pydantic response
        formatted_results = [
            RetrievedDoc(
                document_id=doc["id"], 
                score=doc["score"],
                metadata=doc["meta"] # <--- Pass metadata to response
            ) 
            for doc in results_list
        ]
        
        return SearchResponse(results=formatted_results)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during search: {str(e)}"
        )