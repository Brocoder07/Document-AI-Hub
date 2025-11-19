from fastapi import APIRouter, Depends, HTTPException, status, Form
from pydantic import BaseModel
from app.api.dependencies import get_current_active_user, UserInDB
from app.services.summarization_service import summarize_text_async 

router = APIRouter()

# --- Pydantic Schemas ---
# We keep the Response model as Pydantic/JSON
class SummarizeResponse(BaseModel):
    summary: str

# --- Endpoints ---

@router.post("/text", response_model=SummarizeResponse)
async def summarize_large_text(
    # CHANGED: Use Form(...) instead of a JSON Body.
    # This allows users to send raw text with newlines without JSON errors.
    text: str = Form(..., description="The large text to summarize. Newlines are allowed."),
    method: str = Form("extractive", description="Hint for summarization style (e.g., extractive, abstractive)"),
    current_user: UserInDB = Depends(get_current_active_user)
):
    """
    Summarizes a large block of text. 
    Now accepts 'application/x-www-form-urlencoded' or 'multipart/form-data'.
    This solves the 'JSON decode error' for text containing actual newlines.
    """
    
    if not text or text.isspace():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text cannot be empty."
        )

    try:
        # Call service with the requested text and method.
        summary = await summarize_text_async(
            text=text,
            method=method,
            length="a few sentences" 
        )
        
        return SummarizeResponse(summary=summary)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate summary: {str(e)}"
        )