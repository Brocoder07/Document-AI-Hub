from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from app.api.dependencies import get_current_active_user, UserInDB
# --- THIS IS THE FIX ---
from app.services.rag_service import retrieve_docs  # (Was 'retrieve')

router = APIRouter()

# --- Pydantic Schemas ---

class SearchQueryRequest(BaseModel):
    query: str
    file_id: str | None = None
    top_k: int = Field(default=3, gt=0, le=10)

class RetrievedDoc(BaseModel):
    # This is simplified based on the new rag_service output
    text: str

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
    """
    try:
        user_id = current_user.email
        
        # Get the list of rich dictionaries
        results_list = retrieve_docs(
            query=request.query,
            user_id=user_id,
            file_id=request.file_id,
            top_k=request.top_k
        )
        
        # FIX: Extract just the 'text' field because SearchResponse 
        # currently only expects text.
        results = [{"text": doc["text"]} for doc in results_list]
        
        return SearchResponse(results=results)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during search: {str(e)}"
        )