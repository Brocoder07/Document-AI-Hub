from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from pydantic import BaseModel
from app.api.dependencies import get_current_active_user, UserInDB
from app.core.config import settings
import os
import app.services.transcription_service as transcription_service

router = APIRouter()

# --- Pydantic Schemas ---

class TranscriptionRequest(BaseModel):
    file_id: str # Unique ID of the audio file

class TranscriptionResponse(BaseModel):
    file_id: str
    transcription: str

# --- Endpoints ---

@router.post("/audio", response_model=TranscriptionResponse)
async def transcribe_audio_file(
    request: TranscriptionRequest,
    current_user: UserInDB = Depends(get_current_active_user)
):
    """
    Transcribes an *already uploaded* audio file.
    This uses openai-whisper and is VERY SLOW.
    """
    
    # TODO: Add logic to check if current_user owns file_id
    # 1. Get user_id from current_user.email
    # 2. Query PostgreSQL: get_document(file_id=request.file_id, user_id=user_id)
    # 3. If doc is None, raise 404/403
    # 4. Get the 'saved_path'
    
    # --- FAKE PATH (replace with DB logic) ---
    fake_path = os.path.join(settings.UPLOAD_DIR, request.file_id)
    if not os.path.exists(fake_path):
         raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found. (Note: Use the *unique* file ID, not the filename)"
        )
    # --- END FAKE PATH ---

    try:
        # Call the transcription service
        transcription_text = transcription_service.transcribe_audio(fake_path)
        
        return TranscriptionResponse(
            file_id=request.file_id,
            transcription=transcription_text
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to transcribe audio: {str(e)}"
        )