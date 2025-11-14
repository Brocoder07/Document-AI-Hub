from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from enum import Enum
from app.api.dependencies import get_current_active_user, UserInDB
from app.core.llm import get_llm # <-- IMPORT THE NEW FUNCTION

from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

router = APIRouter()

# --- REMOVED LLM AND PARSER FROM HERE ---

# --- Pydantic Schemas (Unchanged) ---
class FormatType(str, Enum):
    markdown = "markdown"
    json = "json"
    bullet_points = "bullet_points"

class FormatRequest(BaseModel):
    text: str
    format: FormatType

class FormatResponse(BaseModel):
    formatted_text: str

# --- Endpoint (Updated) ---
@router.post("/response", response_model=FormatResponse)
async def format_text_response(
    request: FormatRequest,
    current_user: UserInDB = Depends(get_current_active_user)
):
    """
    Formats a block of text into a specified format using LangChain.
    """
    
    # --- GET LLM AND PARSER INSIDE THE FUNCTION ---
    llm = get_llm()
    output_parser = StrOutputParser()
    
    if request.format == FormatType.json:
        prompt_instruction = "Format the following text as a structured JSON object. Infer the schema."
    elif request.format == FormatType.markdown:
        prompt_instruction = "Format the following text as clean, well-structured Markdown."
    else:
        prompt_instruction = "Format the following text as a concise list of bullet points."

    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_instruction),
        ("user", "{text_input}")
    ])

    chain = prompt | llm | output_parser

    try:
        formatted_text = await chain.ainvoke({"text_input": request.text})
        return FormatResponse(formatted_text=formatted_text)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to format text: {str(e)}"
        )