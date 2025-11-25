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
class QueryMode(str, Enum):
    general = "general"
    legal = "legal"
    finance = "finance"
    academic = "academic"
    healthcare = "healthcare"
    business = "business"

class RagQueryRequest(BaseModel):
    query: str
    file_id: str | None = None
    mode: QueryMode = QueryMode.general
    session_id: str | None = None 

# (Keep RetrievedDoc, RagMetrics, RagResponse schemas as they were...)
# ... [Assuming standard schemas from previous context] ...
# For brevity, ensuring the response model handles session_id
class RetrievedDoc(BaseModel):
    id: str
    text: str
    metadata: dict = {}
    score: float

class RagMetrics(BaseModel):
    # (Simplified for brevity, ensure matches your existing schema)
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

# --- 2. Title Generator ---
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

# --- 3. Endpoints ---

@router.get("/history", response_model=List[ChatSessionSchema])
def get_user_chat_history(
    current_user: UserInDB = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Fetch all chat sessions for the current user.
    FIX: Decrypts titles if they are encrypted, handles plain text gracefully.
    """
    sessions = chat_service.get_user_sessions(db, current_user.id)
    
    clean_history = []
    for s in sessions:
        # 1. Default to the raw title
        display_title = s.title
        
        # 2. Try to decrypt it
        try:
            if s.title:
                # security.decrypt_message usually raises an error if the string 
                # isn't a valid encrypted token (e.g., if it's plain text)
                display_title = security.decrypt_message(s.title)
        except Exception:
            # If decryption fails, it was likely already plain text 
            # (e.g., created by the auto-generator which doesn't seem to encrypt)
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
    """Fetch messages for a specific session."""
    # Security check: ensure session belongs to user
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
    background_tasks: BackgroundTasks, # <--- Added for non-blocking title generation
    current_user: UserInDB = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    # (RBAC check...)
    # check_mode_permission(current_user, rag_request.mode) # Ensure this is imported/defined

    try:
        user_id = current_user.email
        session_id = rag_request.session_id
        is_new_session = False

        if session_id:
            # Check existence
            if not chat_service.get_session(db, session_id, current_user.id):
                session_id = None # Fallback to new session if invalid
        
        if not session_id:
            is_new_session = True
            # Temporary title
            new_session = chat_service.create_session(db, current_user.id, title="New Chat")
            session_id = new_session.id
        
        # Get history context
        chat_history_str = chat_service.get_chat_history_string(db, session_id)

        # Get Answer
        result = await answer_query(
            query=rag_request.query,
            user_id=user_id,
            file_id=rag_request.file_id,
            mode=rag_request.mode.value,
            chat_history=chat_history_str
        )
        
        # Save Messages
        chat_service.add_message(db, session_id, "user", rag_request.query)
        chat_service.add_message(db, session_id, "assistant", result["answer"], retrieved_docs=result["retrieved"])

        # Smart Title Update (Background Task)
        if is_new_session:
            async def update_title_task(sid, q, a):
                new_title = await generate_smart_title(q, a)
                # Need a new DB session for background task usually, 
                # but simplified here: logic must handle DB scope carefully.
                # Ideally, call a service that creates its own session or pass the ID.
                # For safety in FastAPI background tasks with SQLAlchemy:
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
    title_data: dict,  # Expects {"title": "new title"}
    current_user: UserInDB = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Update the title of a chat session."""
    session = chat_service.get_session(db, session_id, current_user.id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    new_title = title_data.get("title", "").strip()
    if not new_title:
        raise HTTPException(status_code=400, detail="Title cannot be empty")
    
    # Encrypt the new title
    encrypted_title = security.encrypt_message(new_title)
    chat_service.update_session_title(db, session_id, encrypted_title)
    
    return {"message": "Title updated successfully", "title": new_title}

@router.delete("/history/{session_id}")
def delete_chat_session(
    session_id: str,
    current_user: UserInDB = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Delete a chat session and all its messages."""
    session = chat_service.get_session(db, session_id, current_user.id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Delete the session (this should cascade to messages due to relationship)
    chat_service.delete_session(db, session_id)
    
    return {"message": "Chat session deleted successfully"}