from fastapi import APIRouter, Depends, HTTPException, status, Request, BackgroundTasks
from pydantic import BaseModel
from enum import Enum
from typing import Optional, List
from sqlalchemy.orm import Session
from langchain.prompts import PromptTemplate
from app.core import security
from app.api.dependencies import get_current_active_user, UserInDB
from app.db.session import get_db
from app.services.rag_service import answer_query
from app.services import chat_service
from app.core.limiter import limiter
from app.core.llm import get_llm

router = APIRouter()

# --- 1. Schemas ---
# SENIOR ENG FIX: Removed 'general' mode.
class QueryMode(str, Enum):
    legal = "legal"
    finance = "finance"
    academic = "academic"
    healthcare = "healthcare"
    business = "business"

class RagQueryRequest(BaseModel):
    query: str
    file_id: str | None = None
    # mode field is inferred from user role, so it is not in the request body
    session_id: str | None = None 

class RetrievedDoc(BaseModel):
    id: str
    text: str
    metadata: dict = {}
    score: float

class RagMetrics(BaseModel):
    processing_time_total: float = 0.0
    retrieval_time: float = 0.0
    generation_time: float = 0.0
    token_usage: dict = {}
    similarity_score: float = 0.0
    confidence_category: str = "N/A"
    confidence_score: float = 0.0
    hallucination_risk: str = "Unknown"
    citation_validation: dict = {}
    evaluation: dict = {}

class RagResponse(BaseModel):
    answer: str
    retrieved: list[RetrievedDoc]
    metrics: RagMetrics
    session_id: str

class ChatSessionSchema(BaseModel):
    id: str
    title: str
    created_at: str

class ChatMessageSchema(BaseModel):
    role: str
    content: str
    created_at: str
    retrieved_docs: Optional[List[dict]] = None

# --- 2. Helpers ---
async def generate_smart_title(query: str, answer: str) -> str:
    """Generates a short title based on the first interaction."""
    llm = get_llm()
    prompt = PromptTemplate.from_template(
        "Summarize the following interaction into a short, 3-6 word title. "
        "Do not use quotes.\n\nUser: {query}\nAI: {answer}\n\nTitle:"
    )
    try:
        res = await prompt.ainvoke({"query": query, "answer": answer})
        return res.content.strip()
    except:
        return query[:30] + "..."

def map_role_to_mode(role: str) -> str:
    """
    Maps a user's business role to a technical RAG system mode.
    Fallbacks to 'business' instead of 'general'.
    """
    role = role.lower().strip() if role else "employee"
    
    mapping = {
        "banker": "finance",
        "lawyer": "legal",
        "doctor": "healthcare",
        "student": "academic",
        "researcher": "academic",
        "business man": "business",
        "business_man": "business",
        "employee": "business"  # SENIOR ENG FIX: Mapped Employee to Business
    }
    # Default fallback is now business
    return mapping.get(role, "business")

# --- 3. Endpoints ---

@router.get("/history", response_model=List[ChatSessionSchema])
def get_user_chat_history(
    current_user: UserInDB = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    sessions = chat_service.get_user_sessions(db, current_user.id)
    clean_history = []
    for s in sessions:
        display_title = s.title
        try:
            if s.title:
                display_title = security.decrypt_message(s.title)
        except Exception:
            pass
            
        clean_history.append(
            ChatSessionSchema(
                id=s.id, 
                title=display_title, 
                created_at=s.created_at.strftime("%Y-%m-%d %H:%M")
            )
        )
    return clean_history

@router.get("/history/{session_id}", response_model=List[ChatMessageSchema])
def get_session_messages(
    session_id: str,
    current_user: UserInDB = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    session = chat_service.get_session(db, session_id, current_user.id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    msgs = chat_service.get_session_messages(db, session_id)
    return [
        ChatMessageSchema(
            role=m.role, 
            content=m.content, 
            created_at=m.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            retrieved_docs=m.retrieved_docs
        ) for m in msgs
    ]

@router.post("/answer", response_model=RagResponse)
@limiter.limit("10/minute") 
async def get_rag_answer(
    request: Request, 
    rag_request: RagQueryRequest,
    background_tasks: BackgroundTasks,
    current_user: UserInDB = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    try:
        user_id = current_user.email
        session_id = rag_request.session_id
        
        # --- LOGIC: Automatic Role-Based Mode ---
        user_role = current_user.role 
        system_mode = map_role_to_mode(user_role)
        
        # --- Session Handling ---
        is_new_session = False
        if session_id:
            if not chat_service.get_session(db, session_id, current_user.id):
                session_id = None 
        
        if not session_id:
            is_new_session = True
            new_session = chat_service.create_session(db, current_user.id, title="New Chat")
            session_id = new_session.id
        
        chat_history_str = chat_service.get_chat_history_string(db, session_id)

        # Pass 'system_mode' (which is now guaranteed not to be 'general')
        result = await answer_query(
            query=rag_request.query,
            user_id=user_id,
            file_id=rag_request.file_id,
            mode=system_mode, 
            chat_history=chat_history_str
        )
        
        chat_service.add_message(db, session_id, "user", rag_request.query)
        chat_service.add_message(db, session_id, "assistant", result["answer"], retrieved_docs=result["retrieved"])

        if is_new_session:
            async def update_title_task(sid, q, a):
                new_title = await generate_smart_title(q, a)
                from app.db.session import SessionLocal
                with SessionLocal() as local_db:
                    chat_service.update_session_title(local_db, sid, new_title)

            background_tasks.add_task(update_title_task, session_id, rag_request.query, result["answer"])

        return RagResponse(
            answer=result["answer"],
            retrieved=result["retrieved"],
            metrics=result["metrics"],
            session_id=session_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG Error: {str(e)}")
    
@router.patch("/history/{session_id}/title")
def update_chat_session_title(
    session_id: str,
    title_data: dict, 
    current_user: UserInDB = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    session = chat_service.get_session(db, session_id, current_user.id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    new_title = title_data.get("title", "").strip()
    if not new_title:
        raise HTTPException(status_code=400, detail="Title cannot be empty")
    
    encrypted_title = security.encrypt_message(new_title)
    chat_service.update_session_title(db, session_id, encrypted_title)
    
    return {"message": "Title updated successfully", "title": new_title}

@router.delete("/history/{session_id}")
def delete_chat_session(
    session_id: str,
    current_user: UserInDB = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    session = chat_service.get_session(db, session_id, current_user.id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    chat_service.delete_session(db, session_id)
    return {"message": "Chat session deleted successfully"}