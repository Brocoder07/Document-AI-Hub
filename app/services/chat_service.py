from sqlalchemy.orm import Session
import json
from app.models.chat import ChatSession, ChatMessage
import uuid
from app.core.security import encrypt_message, decrypt_message # Ensure these are imported

# --- 1. Session Management ---

def create_session(db: Session, user_id: int, title: str = "New Chat") -> ChatSession:
    """
    Creates a new chat session with an encrypted title.
    """
    session_id = str(uuid.uuid4())
    
    # Encrypt the title before saving
    encrypted_title = encrypt_message(title)
    
    session = ChatSession(id=session_id, user_id=user_id, title=encrypted_title)
    db.add(session)
    db.commit()
    db.refresh(session)
    
    # Decrypt title for immediate return to API
    session.title = decrypt_message(session.title)
    return session

def get_session(db: Session, session_id: str, user_id: int) -> ChatSession | None:
    """
    Retrieves a specific session. Does NOT auto-decrypt title here 
    because we usually access title via get_user_sessions.
    """
    return db.query(ChatSession).filter(
        ChatSession.id == session_id, 
        ChatSession.user_id == user_id
    ).first()

def get_user_sessions(db: Session, user_id: int):
    """
    Returns all chat sessions for a user, sorted by newest first.
    Decrypts titles before returning.
    """
    sessions = db.query(ChatSession)\
        .filter(ChatSession.user_id == user_id)\
        .order_by(ChatSession.created_at.desc())\
        .all()
    
    # Decrypt titles for display
    for s in sessions:
        s.title = decrypt_message(s.title)
        
    return sessions

def update_session_title(db: Session, session_id: str, title: str):
    """
    Updates the title of a chat session (e.g. smart renaming).
    Encrypts the new title before saving.
    """
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if session:
        session.title = encrypt_message(title) # Encrypt!
        db.commit()
        db.refresh(session)

def delete_session(db: Session, session_id: str):
    """
    Delete a chat session and all its messages (cascade delete).
    """
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if session:
        db.delete(session)
        db.commit()
        return True
    return False

# --- 2. Message Management ---

def add_message(db: Session, session_id: str, role: str, content: str, retrieved_docs: list = None):
    """
    Adds a new message to the session. Encrypts content before saving.
    Now also stores retrieved documents for assistant messages.
    """
    encrypted_content = encrypt_message(content)
    
    # NEW: Encrypt retrieved docs if they exist
    encrypted_retrieved = None
    if retrieved_docs and role == "assistant":
        encrypted_retrieved = encrypt_message(json.dumps(retrieved_docs))
    
    msg = ChatMessage(
        session_id=session_id, 
        role=role, 
        content=encrypted_content,
        retrieved_docs=encrypted_retrieved  # <-- STORE RETRIEVED DOCS
    )
    db.add(msg)
    db.commit()

def get_chat_history_string(db: Session, session_id: str, limit: int = 6) -> str:
    """
    Fetches last 'limit' messages for the RAG context window.
    Decrypts content so the LLM can read it.
    """
    messages = db.query(ChatMessage)\
        .filter(ChatMessage.session_id == session_id)\
        .order_by(ChatMessage.created_at.desc())\
        .limit(limit)\
        .all()
    
    # Reverse to chronological order (Oldest -> Newest)
    messages = messages[::-1]
    
    history_str = ""
    for msg in messages:
        # Decrypt content
        decrypted_content = decrypt_message(msg.content)
        role_prefix = "User" if msg.role == "user" else "Assistant"
        history_str += f"{role_prefix}: {decrypted_content}\n"
        
    return history_str

def get_session_messages(db: Session, session_id: str):
    """
    Returns full message history for a session (for Frontend UI).
    Decrypts content AND retrieved_docs before returning.
    """
    messages = db.query(ChatMessage)\
        .filter(ChatMessage.session_id == session_id)\
        .order_by(ChatMessage.created_at.asc())\
        .all()
        
    # Decrypt content and retrieved_docs in-memory
    for msg in messages:
        msg.content = decrypt_message(msg.content)
        
        # NEW: Decrypt and parse retrieved_docs
        if msg.retrieved_docs:
            try:
                retrieved_json = decrypt_message(msg.retrieved_docs)
                msg.retrieved_docs = json.loads(retrieved_json)
            except (json.JSONDecodeError, Exception):
                msg.retrieved_docs = None  # Handle corrupted data gracefully
        else:
            msg.retrieved_docs = None
            
    return messages
