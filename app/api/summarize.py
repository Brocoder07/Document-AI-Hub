from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from app.api.dependencies import get_current_active_user, UserInDB
# --- IMPORT THE NEW LANGCHAIN SERVICE ---
from app.services.summarization_service import summarize_text_async 
# --- (The old 'generator_service' import is gone) ---

router = APIRouter()

# --- Pydantic Schemas (Unchanged) ---

class SummarizeRequest(BaseModel):
    text: str
    method: str = Field(default="extractive", description="Hint for summarization style (e.g., extractive, abstractive)")
    length: str = Field(default="a few sentences", description="e.g., 'one paragraph', 'three bullet points'")

class SummarizeResponse(BaseModel):
    summary: str

# --- Endpoints (Updated) ---

@router.post("/text", response_model=SummarizeResponse)
async def summarize_large_text(
    request: SummarizeRequest,
    current_user: UserInDB = Depends(get_current_active_user)
):
    """
    Summarizes a large block of text.
    This endpoint now calls the summarization service
    which handles complex logic like map-reduce.
    """
    
    if not request.text or request.text.isspace():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text cannot be empty."
        )

    try:
        # --- LOGIC MOVED TO SERVICE ---
        # Call the new async service function
        summary = await summarize_text_async(
            text=request.text,
            method=request.method,
            length=request.length
        )
        # --- END OF MOVED LOGIC ---
        
        return SummarizeResponse(summary=summary)
        
    except Exception as e:
        # This will catch errors from the Groq API or LangChain
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate summary: {str(e)}"
        )