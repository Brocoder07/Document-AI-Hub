from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from app.api.dependencies import get_current_active_user, UserInDB
from app.services.ocr_service import extract_text_from_pdf, extract_text_from_image
import os
from PIL import Image

router = APIRouter()

# --- Pydantic Schemas ---

class OcrRequest(BaseModel):
    file_id: str 

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
    Runs OCR on an *already uploaded* file (PDF or Image).
    """
    
    # FIX: Construct the correct user-specific path
    user_dir = os.path.join("data", "users", current_user.email, "documents")
    file_path = os.path.join(user_dir, request.file_id)

    if not os.path.exists(file_path):
         raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File not found at {file_path}. Ensure you used the correct file_id."
        )

    try:
        # Check file extension to decide between PDF or Image OCR
        ext = request.file_id.split('.')[-1].lower()
        
        if ext == "pdf":
            text = extract_text_from_pdf(file_path)
        elif ext in ["png", "jpg", "jpeg", "tiff", "bmp"]:
            # For images, we need to open the file object
            image = Image.open(file_path)
            text = extract_text_from_image(image)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file type for OCR: {ext}"
            )
        
        return OcrResponse(
            file_id=request.file_id,
            extracted_text=text
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to perform OCR: {str(e)}"
        )