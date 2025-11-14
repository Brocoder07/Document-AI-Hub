from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from app.api.dependencies import get_current_active_user, UserInDB
from app.services.rag_service import answer_query

router = APIRouter()

# --- Pydantic Schemas ---

class RagQueryRequest(BaseModel):
    query: str
    file_id: str | None = None # User can optionally scope query to one file

class RetrievedDoc(BaseModel):
    id: str
    text: str
    meta: dict

class RagResponse(BaseModel):
    answer: str
    retrieved: list[RetrievedDoc]

# --- Endpoints ---

@router.post("/answer", response_model=RagResponse)
async def get_rag_answer(
    request: RagQueryRequest,
    current_user: UserInDB = Depends(get_current_active_user)
):
    """
    Generates an Al-powered answer from documents (RAG).
    Filters by the authenticated user and optionally by file_id.
    """
    try:
        user_id = current_user.email # Get user ID for data scoping
        
        result = await answer_query(
            query=request.query,
            user_id=user_id,
            file_id=request.file_id
        )
        return RagResponse(**result)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating RAG answer: {str(e)}"
        )