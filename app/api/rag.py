from fastapi import APIRouter, Depends, HTTPException, status, Request
from pydantic import BaseModel
from enum import Enum
from typing import Optional, List
from sqlalchemy.orm import Session

from app.api.dependencies import get_current_active_user, UserInDB
from app.db.session import get_db
from app.services.rag_service import answer_query
# Ensure app/services/chat_service.py exists as discussed
from app.services import chat_service  
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
    session_id: str | None = None # <--- New Field for Chat History

class RetrievedDoc(BaseModel):
    id: str
    text: str
    meta: dict
    score: float

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
    citation_validation: dict
    evaluation: dict = {}

class RagResponse(BaseModel):
    answer: str
    retrieved: list[RetrievedDoc]
    metrics: RagMetrics
    session_id: str # <--- Return ID to frontend

# --- 3. Helper: RBAC Logic ---
def check_mode_permission(user: UserInDB, mode: QueryMode):
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
@limiter.limit("10/minute") 
async def get_rag_answer(
    request: Request, 
    rag_request: RagQueryRequest,
    current_user: UserInDB = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    # 1. Enforce RBAC
    check_mode_permission(current_user, rag_request.mode)

    try:
        user_id = current_user.email
        
        # 2. Handle Session / Chat History
        session_id = rag_request.session_id
        chat_history_str = ""

        if session_id:
            # Verify session exists and belongs to user
            session = chat_service.get_session(db, session_id, current_user.id)
            if not session:
                raise HTTPException(status_code=404, detail="Chat session not found or access denied")
            
            # Retrieve previous context (e.g., last 6 messages)
            chat_history_str = chat_service.get_chat_history_string(db, session_id)
        else:
            # Create New Session
            # Title is just the first 30 chars of the first query
            title = rag_request.query[:30] + "..."
            new_session = chat_service.create_session(db, current_user.id, title=title)
            session_id = new_session.id

        # 3. Call RAG Service (Injecting History)
        result = await answer_query(
            query=rag_request.query,
            user_id=user_id,
            file_id=rag_request.file_id,
            mode=rag_request.mode.value,
            chat_history=chat_history_str # <--- Pass history context
        )
        
        # 4. Save Interaction to DB
        chat_service.add_message(db, session_id, "user", rag_request.query)
        chat_service.add_message(db, session_id, "assistant", result["answer"])

        # 5. Return response with session_id
        return RagResponse(
            answer=result["answer"],
            retrieved=result["retrieved"],
            metrics=result["metrics"],
            session_id=session_id
        )
        
    except HTTPException:
        raise 
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating RAG answer: {str(e)}"
        )