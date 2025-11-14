from fastapi import APIRouter, Depends, UploadFile, File, BackgroundTasks, HTTPException, status
from pydantic import BaseModel
from app.api.dependencies import get_current_active_user, UserInDB
from app.common_utils.file_handler import save_upload
from app.services.file_processing_service import process_document
import os

router = APIRouter()

# --- Pydantic Schemas ---

class UploadResponse(BaseModel):
    status: str
    file_id: str
    filename: str

# --- Endpoints ---

@router.post("/file", response_model=UploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_user: UserInDB = Depends(get_current_active_user)
):
    """
    Handles file uploads, saves the file, and triggers a background
    task for processing (OCR, chunking, embedding).
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
        
    # Sanitize filename (basic)
    filename = os.path.basename(file.filename)
    
    try:
        # Save the file to disk
        # We pass file.file (the SpooledTemporaryFile) to the saver
        saved_path = save_upload(file.file, filename)
        
        # Get user ID for metadata
        user_id = current_user.email # Or a UUID if you have one
        
        # Schedule the heavy processing to run in the background
        background_tasks.add_task(process_document, saved_path, filename, user_id)
        
        # TODO: Log this upload to your PostgreSQL 'documents' table
        # e.g., db.create_document(file_id=file_id, user_id=user_id, status="processing")

        # Return immediately
        return UploadResponse(
            status="File uploaded, processing in background.",
            file_id=filename, # TODO: Replace with a unique ID from your DB
            filename=filename
        )
        
    except Exception as e:
        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload file: {str(e)}"
        )