from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session
from app.api.dependencies import get_current_active_user, UserInDB
from app.services.ocr_service import extract_text_from_pdf, extract_text_from_image
from app.services.document_reader import read_text_file, read_docx_file
from app.services.document_service import get_document_by_file_id
from app.db.session import get_db
import os
from PIL import Image
import logging

# Setup Logger
logger = logging.getLogger(__name__)

router = APIRouter()

class OcrRequest(BaseModel):
    file_id: str

class OcrResponse(BaseModel):
    file_id: str
    extracted_text: str

@router.post("/extract", response_model=OcrResponse)
async def extract_text_unified(
    request: OcrRequest,
    current_user: UserInDB = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    logger.info(f"OCR Request received for File ID: {request.file_id}")

    try:
        # 1. Database Lookup
        doc = get_document_by_file_id(db, request.file_id)
        
        if not doc:
            logger.error(f"Document not found: {request.file_id}")
            raise HTTPException(status_code=404, detail="File ID not found in database.")
            
        # 2. Access Control
        if str(doc.user_id) != str(current_user.id):
            logger.warning(f"Unauthorized access attempt by user {current_user.id}")
            raise HTTPException(status_code=403, detail="Not authorized.")
            
        file_path = doc.file_path
        if not os.path.exists(file_path):
             logger.error(f"File missing on disk: {file_path}")
             raise HTTPException(status_code=404, detail="Physical file missing from server storage.")

        # 3. Processing
        text = ""
        ftype = str(doc.file_type).lower() # Safely cast to string
        filename = str(doc.filename).lower()

        logger.info(f"Processing file type: {ftype} ({filename})")

        try:
            if ftype == "pdf" or filename.endswith(".pdf"):
                text = extract_text_from_pdf(file_path)
            elif ftype in ["png", "jpg", "jpeg", "tiff", "bmp"]:
                image = Image.open(file_path)
                text = extract_text_from_image(image)
            elif ftype in ["txt", "md"]:
                text = read_text_file(file_path, user_id=current_user.email)
            elif ftype in ["docx"]:
                text = read_docx_file(file_path)
            elif ftype == "doc":
                raise HTTPException(status_code=400, detail="Legacy .doc format not supported.")
            else:
                raise HTTPException(status_code=400, detail= f"Unsupported format: {ftype}")
        except Exception as e:
            logger.error(f"Extraction internal error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Extraction Engine Failed: {str(e)}")
        
        return OcrResponse(file_id=request.file_id, extracted_text=text)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Database/System error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"System Error: {str(e)}")