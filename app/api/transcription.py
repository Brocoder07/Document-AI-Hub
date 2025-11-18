from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session
from app.api.dependencies import get_current_active_user, UserInDB
from app.services.transcription_service import transcribe_audio
from app.services.document_service import get_document_by_file_id
from app.db.session import get_db
import os

router = APIRouter()

class TranscriptionRequest(BaseModel):
    file_id: str # Expects UUID

class TranscriptionResponse(BaseModel):
    file_id: str
    transcription: str

@router.post("/audio", response_model=TranscriptionResponse)
async def transcribe_audio_file(
    request: TranscriptionRequest,
    current_user: UserInDB = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    # 1. Lookup Document
    doc = get_document_by_file_id(db, request.file_id)
    
    if not doc:
        raise HTTPException(status_code=404, detail="File ID not found.")
    if doc.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized.")
        
    if not os.path.exists(doc.file_path):
         raise HTTPException(status_code=404, detail="File missing from disk.")

    try:
        # 2. Process
        transcription_text = transcribe_audio(doc.file_path)
        return TranscriptionResponse(file_id=request.file_id, transcription=transcription_text)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription Failed: {str(e)}")