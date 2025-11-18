from sqlalchemy.orm import Session
from app.models.documents import Document
import os

def create_document_record(db: Session, doc_data: dict) -> Document:
    db_doc = Document(
        file_id=doc_data["file_id"],
        filename=doc_data["filename"],
        file_type=doc_data["file_type"],
        file_path=doc_data["file_path"],
        file_size=doc_data["file_size"],
        content_hash=doc_data["content_hash"], # FIX: Save hash
        user_id=doc_data["user_id"]
    )
    db.add(db_doc)
    db.commit()
    db.refresh(db_doc)
    return db_doc

def get_user_documents(db: Session, user_id: int) -> list[Document]:
    return db.query(Document).filter(Document.user_id == user_id).all()

def get_document_by_file_id(db: Session, file_id: str) -> Document | None:
    return db.query(Document).filter(Document.file_id == file_id).first()

def get_existing_document(db: Session, user_id: int, content_hash: str) -> Document | None:
    """Finds a document for this user with the exact same content hash."""
    return db.query(Document).filter(
        Document.user_id == user_id,
        Document.content_hash == content_hash
    ).first()

# --- NEW: Hard Delete Function ---

def hard_delete_document(db: Session, file_id: str, user_id: int) -> bool:
    """
    Hard deletes a document:
    1. Removes from database
    2. Deletes file from disk
    3. Returns True if successful, False if document not found or unauthorized
    """
    # Find document
    doc = db.query(Document).filter(Document.file_id == file_id).first()
    
    if not doc:
        return False
    
    # Security check: ensure user owns this document
    if doc.user_id != user_id:
        return False
    
    try:
        # Delete from disk first
        if doc.file_path and os.path.exists(doc.file_path):
            os.remove(doc.file_path)
        
        # Delete from database
        db.delete(doc)
        db.commit()
        
        return True
        
    except Exception as e:
        db.rollback()
        raise Exception(f"Failed to delete document: {str(e)}")