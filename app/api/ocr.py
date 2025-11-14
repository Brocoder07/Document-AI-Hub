from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from pydantic import BaseModel
from app.api.dependencies import get_current_active_user, UserInDB
from app.services.ocr_service import extract_text_from_pdf
from app.core.config import settings # To get UPLOAD_DIR
import os

router = APIRouter()

# --- Pydantic Schemas ---

class OcrRequest(BaseModel):
    file_id: str # This should be the unique ID of the file

class OcrResponse(BaseModel):
    file_id: str
    extracted_text: str

# --- Endpoints ---

@router.post("/extract", response_model=OcrResponse)
async def extract_text_ocr(
    request: OcrRequest,
    current_user: UserInDB = Depends(get_current_active_user)
):
    """
    Runs OCR on an *already uploaded* PDF file.
    NOTE: This is a slow, blocking operation.
    """
    
    # TODO: Add logic to check if current_user owns file_id
    # 1. Get user_id from current_user.email
    # 2. Query PostgreSQL: get_document(file_id=request.file_id, user_id=user_id)
    # 3. If doc is None, raise HTTPException 404 or 403
    # 4. Get the document's 'saved_path' from the DB
    
    # --- FAKE PATH (replace with DB logic) ---
    fake_path = os.path.join(settings.UPLOAD_DIR, request.file_id)
    if not os.path.exists(fake_path):
         raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found. (Note: Use the *unique* file ID, not the filename)"
        )
    # --- END FAKE PATH ---

    try:
        # This is very slow!
        text = extract_text_from_pdf(fake_path)
        
        return OcrResponse(
            file_id=request.file_id,
            extracted_text=text
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to perform OCR: {str(e)}"
        )