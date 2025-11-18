from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from pydantic import BaseModel
from app.api.dependencies import get_current_active_user, UserInDB
# REMOVE: from app.core.config import settings
import os
import app.services.transcription_service as transcription_service

router = APIRouter()

# --- Pydantic Schemas ---
class TranscriptionRequest(BaseModel):
    file_id: str 

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
        # Call the transcription service
        transcription_text = transcription_service.transcribe_audio(file_path)
        
        return TranscriptionResponse(
            file_id=request.file_id,
            transcription=transcription_text
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to transcribe audio: {str(e)}"
        )