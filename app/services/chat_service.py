from sqlalchemy.orm import Session
from app.models.chat import ChatSession, ChatMessage
import uuid

def create_session(db: Session, user_id: int, title: str = "New Chat") -> ChatSession:
    session_id = str(uuid.uuid4())
    session = ChatSession(id=session_id, user_id=user_id, title=title)
    db.add(session)
    db.commit()
    db.refresh(session)
    return session

def get_session(db: Session, session_id: str, user_id: int) -> ChatSession | None:
    return db.query(ChatSession).filter(
        ChatSession.id == session_id, 
        ChatSession.user_id == user_id
    ).first()

def add_message(db: Session, session_id: str, role: str, content: str):
    msg = ChatMessage(session_id=session_id, role=role, content=content)
    db.add(msg)
    db.commit()

def get_chat_history_string(db: Session, session_id: str, limit: int = 6) -> str:
    """
    Fetches last 'limit' messages and formats them as a string for the LLM prompt.
    """
    # Fetch recent messages
    messages = db.query(ChatMessage)\
        .filter(ChatMessage.session_id == session_id)\
        .order_by(ChatMessage.created_at.desc())\
        .limit(limit)\
        .all()
    
    # Reverse to chronological order
    messages = messages[::-1]
    
    history_str = ""
    for msg in messages:
        role_prefix = "User" if msg.role == "user" else "Assistant"
        history_str += f"{role_prefix}: {msg.content}\n"
        
    return history_str