from app.services.chunking import chunk_text
from app.core.llm import get_llm # <-- IMPORT THE NEW FUNCTION

from langchain.prompts import ChatPromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
import asyncio

# --- REMOVED LLM FROM HERE ---

async def summarize_text_async(text: str, method: str, length: str) -> str:
    """
    Summarizes text using LangChain's map_reduce chain.
    """
    
    # --- GET LLM INSIDE THE FUNCTION ---
    llm = get_llm()
    
    chunks = chunk_text(text, chunk_size=4000, overlap=200)
    docs = [Document(page_content=chunk) for chunk in chunks]
    
    map_template = f"""
    Concisely summarize the following text chunk:
    "{'{text}'}"
    CONCISE SUMMARY:
    """
    map_prompt = ChatPromptTemplate.from_template(map_template)

    combine_template = f"""
    Write a cohesive summary of the following text, formatted to be {length}.
    The user prefers an '{method}' style.
    "{'{text}'}"
    SUMMARY:
    """
    combine_prompt = ChatPromptTemplate.from_template(combine_template)

    chain = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce",
        map_prompt=map_prompt,
        combine_prompt=combine_prompt,
    )

    result = await chain.ainvoke(docs)
    
    return result.get("output_text", "Summarization failed.")