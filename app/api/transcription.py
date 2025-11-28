from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from app.api.dependencies import get_current_active_user, UserInDB
from app.services.transcription_service import transcribe_audio
from app.services.summarization_service import summarize_text_async
from app.services.document_service import get_document_by_file_id
from app.db.session import get_db
import os

router = APIRouter()

# --- Pydantic Schemas ---

class TranscriptionRequest(BaseModel):
    file_id: str # Expects UUID
    summarize: bool = Field(default=False, description="If True, generates a summary after transcription.")
    # REMOVED: context field

class TranscriptionResponse(BaseModel):
    file_id: str
    transcription: str
    summary: str | None = None 

# --- Endpoint ---

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
        # 2. Process Transcription
        # Run blocking CPU operation in threadpool
        transcription_text = await run_in_threadpool(transcribe_audio, doc.file_path)
        
        summary_text = None

        # 3. Optional Summarization
        if request.summarize and transcription_text:
            try:
                # Use a default "abstractive" style since context input was removed
                summary_text = await summarize_text_async(
                    text=transcription_text,
                    method="abstractive", 
                    length="concise" 
                )
            except Exception as e:
                print(f"Summarization failed: {e}")
                summary_text = "Error generating summary."

        return TranscriptionResponse(
            file_id=request.file_id, 
            transcription=transcription_text,
            summary=summary_text
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing Failed: {str(e)}")