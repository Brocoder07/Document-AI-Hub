from fastapi import APIRouter, Depends, UploadFile, File, BackgroundTasks, HTTPException, status
from pydantic import BaseModel
import hashlib
import os
from app.api.dependencies import get_current_active_user, UserInDB
from app.common_utils.file_handler import save_upload
from app.services.file_processing_service import process_document

router = APIRouter()

# --- Pydantic Schemas ---

class UploadResponse(BaseModel):
    status: str
    file_id: str  # Simple filename-based ID
    filename: str

class UserFilesResponse(BaseModel):
    files: list[dict]

# --- Helper Functions ---

def get_file_hash(file_path: str) -> str:
    """Generate MD5 hash of file content for duplicate detection"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def get_user_files_dir(user_id: str) -> str:
    """Get user-specific files directory"""
    user_dir = os.path.join("data", "users", user_id, "documents")
    os.makedirs(user_dir, exist_ok=True)
    return user_dir

# --- Endpoints ---

@router.post("/file", response_model=UploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_user: UserInDB = Depends(get_current_active_user)
):
    """
    Handles file uploads with duplicate detection and user-specific storage.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
        
    # Sanitize filename
    filename = os.path.basename(file.filename)
    
    try:
        # Create user-specific directory
        user_dir = get_user_files_dir(current_user.email)
        
        # Temporary save to check for duplicates
        temp_path = os.path.join(user_dir, f"temp_{filename}")
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Check for duplicate by content hash
        file_hash = get_file_hash(temp_path)
        
        # Check if this file already exists for this user
        existing_files = []
        for existing_file in os.listdir(user_dir):
            if existing_file.startswith("temp_"):
                continue
            existing_path = os.path.join(user_dir, existing_file)
            if get_file_hash(existing_path) == file_hash:
                existing_files.append(existing_file)
        
        if existing_files:
            os.remove(temp_path)  # Remove temp file
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"File already exists: {existing_files[0]}"
            )
        
        # Move temp file to final location
        final_path = os.path.join(user_dir, filename)
        os.rename(temp_path, final_path)
        
        # Schedule processing
        background_tasks.add_task(process_document, final_path, filename, current_user.email)
        
        return UploadResponse(
            status="File uploaded successfully",
            file_id=filename,  # Use filename as simple ID
            filename=filename
        )
        
    except HTTPException:
        raise
    except Exception as e:
        # Clean up temp file if it exists
        temp_path = os.path.join(get_user_files_dir(current_user.email), f"temp_{filename}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload file: {str(e)}"
        )

@router.get("/files", response_model=UserFilesResponse)
async def list_user_files(
    current_user: UserInDB = Depends(get_current_active_user)
):
    """
    List all files uploaded by the current user.
    """
    user_dir = get_user_files_dir(current_user.email)
    
    files = []
    if os.path.exists(user_dir):
        for filename in os.listdir(user_dir):
            if not filename.startswith("temp_"):  # Skip temp files
                file_path = os.path.join(user_dir, filename)
                files.append({
                    "filename": filename,
                    "file_id": filename,  # Use filename as ID
                    "size": os.path.getsize(file_path),
                    "uploaded_at": os.path.getctime(file_path)
                })
    
    return UserFilesResponse(files=files)