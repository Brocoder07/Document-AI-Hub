from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session
from app.api.dependencies import get_current_active_user, UserInDB
from app.services.ocr_service import extract_text_from_pdf, extract_text_from_image
from app.services.document_service import get_document_by_file_id
from app.db.session import get_db
import os
from PIL import Image

router = APIRouter()

class OcrRequest(BaseModel):
    file_id: str # Expects UUID

class OcrResponse(BaseModel):
    file_id: str
    extracted_text: str

@router.post("/extract", response_model=OcrResponse)
async def extract_text_ocr(
    request: OcrRequest,
    current_user: UserInDB = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    # 1. Lookup Document in DB using UUID
    doc = get_document_by_file_id(db, request.file_id)
    
    # 2. Validation: Exists? Belongs to User?
    if not doc:
        raise HTTPException(status_code=404, detail="File ID not found.")
    if doc.user_id != current_user.id: # Security Check
        raise HTTPException(status_code=403, detail="Not authorized to access this file.")
        
    file_path = doc.file_path
    if not os.path.exists(file_path):
         raise HTTPException(status_code=404, detail="File missing from disk.")

    try:
        # 3. Process based on DB file_type or filename
        if doc.file_type == "pdf" or doc.filename.endswith(".pdf"):
            text = extract_text_from_pdf(file_path)
        elif doc.file_type in ["png", "jpg", "jpeg", "tiff", "bmp"]:
            image = Image.open(file_path)
            text = extract_text_from_image(image)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {doc.file_type}")
        
        return OcrResponse(file_id=request.file_id, extracted_text=text)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR Failed: {str(e)}")