from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from app.api.dependencies import get_current_active_user, UserInDB
# FIX: Import the new function
from app.services.rag_service import retrieve_with_scores 

router = APIRouter()

# --- Pydantic Schemas ---

class SearchQueryRequest(BaseModel):
    query: str
    file_id: str | None = None
    top_k: int = Field(default=3, gt=0, le=10)
    # Optional: Allow mode selection in search too, defaulting to general
    mode: str = "general" 

class RetrievedDoc(BaseModel):
    text: str
    score: float # <--- Added score field

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
    Returns text segments with their similarity scores (0 to 1).
    """
    try:
        user_id = current_user.email
        
        # FIX: Call the new function
        # retrieve_with_scores returns: (results, retrieval_time, scores)
        results_list, _, _ = retrieve_with_scores(
            query=request.query,
            user_id=user_id,
            file_id=request.file_id,
            mode=request.mode,
            top_k=request.top_k
        )
        
        # Format for the pydantic response
        # The results_list already contains dicts with 'text' and 'score' keys
        # from our update in rag_service.py
        formatted_results = [
            RetrievedDoc(text=doc["text"], score=doc["score"]) 
            for doc in results_list
        ]
        
        return SearchResponse(results=formatted_results)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during search: {str(e)}"
        )