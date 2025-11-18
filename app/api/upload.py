from fastapi import APIRouter, Depends, UploadFile, File, BackgroundTasks, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session
import os
import uuid
import shutil
import hashlib

from app.api.dependencies import get_current_active_user, UserInDB
from app.db.session import get_db
from app.services.file_processing_service import process_document
from app.services.document_service import (
    create_document_record, 
    get_user_documents, 
    get_existing_document,
    hard_delete_document
)

router = APIRouter()

# --- Pydantic Schemas ---
class UploadResponse(BaseModel):
    status: str
    file_id: str
    filename: str
    message: str = "File uploaded successfully" # Added message field

class DocumentSchema(BaseModel):
    file_id: str
    filename: str
    file_type: str
    file_size: int
    upload_date: str 

    class Config:
        from_attributes = True

class UserFilesResponse(BaseModel):
    files: list[DocumentSchema]

# --- NEW: Delete Response Schema ---
class DeleteResponse(BaseModel):
    status: str
    message: str
    file_id: str

# --- Helper Functions ---
def get_user_files_dir(user_email: str) -> str:
    user_dir = os.path.join("data", "users", user_email, "documents")
    os.makedirs(user_dir, exist_ok=True)
    return user_dir

def calculate_file_hash(file_obj) -> str:
    """Calculate MD5 hash of the file stream."""
    hash_md5 = hashlib.md5()
    # Read chunks to avoid loading large files entirely into RAM
    for chunk in iter(lambda: file_obj.read(4096), b""):
        hash_md5.update(chunk)
    # Reset file cursor to start so it can be read again later
    file_obj.seek(0)
    return hash_md5.hexdigest()

# --- Endpoints ---

@router.post("/file", response_model=UploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_user: UserInDB = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
        
    try:
        # 1. Calculate Hash First (Deduplication Logic)
        content_hash = calculate_file_hash(file.file)
        
        # 2. Check if file already exists for this user
        existing_doc = get_existing_document(db, user_id=current_user.id, content_hash=content_hash)
        
        if existing_doc:
            # STOP! Do not process. Do not save. Return existing info.
            return UploadResponse(
                status="exists",
                file_id=existing_doc.file_id,
                filename=existing_doc.filename,
                message="File already exists. Returned existing record."
            )

        # --- IF NEW FILE, PROCEED AS BEFORE ---
        
        file_uuid = str(uuid.uuid4())
        original_filename = os.path.basename(file.filename)
        _, ext = os.path.splitext(original_filename)
        storage_filename = f"{file_uuid}{ext}"
        
        user_dir = get_user_files_dir(current_user.email)
        final_path = os.path.join(user_dir, storage_filename)
        
        # Save to Disk
        with open(final_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        file_size = os.path.getsize(final_path)
        file_type = ext.lstrip(".").lower() or "unknown"
        
        # Save to Database with Hash
        try:
            create_document_record(db, {
                "file_id": file_uuid,
                "filename": original_filename,
                "file_type": file_type,
                "file_path": str(final_path),
                "file_size": file_size,
                "content_hash": content_hash, # Store the hash
                "user_id": current_user.id 
            })
        except Exception as db_err:
            if os.path.exists(final_path):
                os.remove(final_path)
            raise HTTPException(status_code=500, detail=f"Database error: {str(db_err)}")

        # Trigger Processing
        background_tasks.add_task(
            process_document, 
            saved_path=final_path, 
            file_id=file_uuid,          
            filename=original_filename, 
            user_id=current_user.email,
            user_role=current_user.role 
        )
        
        return UploadResponse(
            status="uploaded",
            file_id=file_uuid,
            filename=original_filename,
            message="File uploaded successfully"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.get("/files", response_model=UserFilesResponse)
async def list_user_files(
    current_user: UserInDB = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """List all files uploaded by the current user."""
    docs = get_user_documents(db, current_user.id)
    
    results = []
    for d in docs:
        results.append({
            "file_id": d.file_id,
            "filename": d.filename,
            "file_type": d.file_type,
            "file_size": d.file_size,
            "upload_date": d.upload_date.strftime("%Y-%m-%d %H:%M:%S")
        })
        
    return UserFilesResponse(files=results)

# --- NEW: Delete Endpoint ---

@router.delete("/file/{file_id}", response_model=DeleteResponse)
async def delete_user_document(
    file_id: str,
    current_user: UserInDB = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Hard deletes a document uploaded by the current user.
    Removes file from disk and database record.
    """
    try:
        success = hard_delete_document(db, file_id=file_id, user_id=current_user.id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found or you do not have permission to delete it."
            )
        
        return DeleteResponse(
            status="deleted",
            message="Document successfully deleted.",
            file_id=file_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document: {str(e)}"
        )