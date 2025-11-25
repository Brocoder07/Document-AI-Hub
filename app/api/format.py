from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from app.core.llm import get_llm
from langchain.prompts import PromptTemplate
from app.api.dependencies import get_current_active_user

router = APIRouter()

class FormatRequest(BaseModel):
    text: str
    format: str  # markdown, bullet points, table, json

class FormatResponse(BaseModel):
    formatted_text: str

@router.post("/response", response_model=FormatResponse)
async def format_response_endpoint(
    req: FormatRequest,
    current_user = Depends(get_current_active_user)
):
    llm = get_llm()
    
    if req.format == "json":
        instruction = "Format the following text as a valid JSON object. Do not include Markdown code blocks."
    elif req.format == "table":
        instruction = "Format the following text as a Markdown table."
    elif req.format == "bullet points":
        instruction = "Summarize the following text as a bulleted list."
    else:
        instruction = "Format the following text using clean Markdown headers and paragraphs."

    prompt = PromptTemplate.from_template(
        "{instruction}\n\nTEXT:\n{text}\n\nFORMATTED:"
    )
    
    try:
        # Create and invoke the chain
        chain = prompt | llm
        result = await chain.ainvoke({
            "instruction": instruction, 
            "text": req.text
        })
        
        # Handle different response types safely
        if hasattr(result, 'content'):
            # This is an AIMessage object
            formatted_text = result.content.strip()
        elif hasattr(result, 'text'):
            # This is a StringPromptValue object
            formatted_text = result.text.strip()
        elif isinstance(result, str):
            # This is a plain string
            formatted_text = result.strip()
        else:
            # Fallback: convert to string
            formatted_text = str(result).strip()
        
        return FormatResponse(formatted_text=formatted_text)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Formatting failed: {str(e)}")