from fastapi import APIRouter, Depends, HTTPException, status, Request
from pydantic import BaseModel
from enum import Enum
from typing import Optional, List
from app.api.dependencies import get_current_active_user, UserInDB
from app.services.rag_service import answer_query
from app.core.limiter import limiter 

router = APIRouter()

# --- 1. Define Modes & Roles ---
class QueryMode(str, Enum):
    general = "general"
    legal = "legal"
    finance = "finance"
    academic = "academic"
    healthcare = "healthcare"
    business = "business"

# --- 2. Pydantic Schemas ---
class RagQueryRequest(BaseModel):
    query: str
    file_id: str | None = None
    mode: QueryMode = QueryMode.general

class RetrievedDoc(BaseModel):
    id: str
    text: str
    meta: dict
    score: float # Added score field

class TokenMetrics(BaseModel):
    input: int
    output: int
    total: int

class RagMetrics(BaseModel):
    processing_time_total: float
    retrieval_time: float
    generation_time: float
    token_usage: TokenMetrics
    similarity_score: float
    confidence_category: str
    confidence_score: float
    hallucination_risk: str

class RagResponse(BaseModel):
    answer: str
    retrieved: list[RetrievedDoc]
    metrics: RagMetrics # Added metrics field

# --- 3. Helper: RBAC Logic ---
def check_mode_permission(user: UserInDB, mode: QueryMode):
    """
    Verifies if the current user has the required role for the selected mode.
    """
    permissions = {
        QueryMode.legal: ["lawyer"],
        QueryMode.healthcare: ["doctor"],
        QueryMode.finance: ["banker", "financial_analyst"],
        QueryMode.academic: ["researcher", "student"],
        QueryMode.business: ["employee", "executive"],
        QueryMode.general: ["*"] 
    }
    
    allowed_roles = permissions.get(mode, ["*"])
    
    if "*" not in allowed_roles and user.role not in allowed_roles:
         raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Access Denied: User role '{user.role}' is not authorized to use '{mode}' mode."
        )

# --- 4. Endpoint ---
@router.post("/answer", response_model=RagResponse)
@limiter.limit("5/minute") 
async def get_rag_answer(
    request: Request, 
    rag_request: RagQueryRequest,
    current_user: UserInDB = Depends(get_current_active_user)
):
    # 1. Enforce RBAC
    check_mode_permission(current_user, rag_request.mode)

    try:
        user_id = current_user.email
        
        # 2. Call the Service
        result = await answer_query(
            query=rag_request.query,
            user_id=user_id,
            file_id=rag_request.file_id,
            mode=rag_request.mode.value
        )
        return RagResponse(**result)
        
    except HTTPException:
        raise 
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating RAG answer: {str(e)}"
        )