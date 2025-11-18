from app.api.chroma_client import get_collection
from app.services.embedding_service import embed_texts
from app.core.llm import get_llm

from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableParallel

# --- 1. Define Specialized Prompts with Safety Clauses ---

PROMPT_TEMPLATES = {
    "general": """
    [INST] You are a helpful assistant. Use the following context to answer the question.
    If the answer is not present in the context, say "I could not find the answer in the provided documents."
    
    CONTEXT:
    {context}
    
    QUESTION:
    {query} [/INST]
    """,

    "legal": """
    [INST] You are a legal AI assistant. Your task is to analyze contracts and legal documents.
    - Extract specific legal clauses, terms, and obligations.
    - Highlight potential risks or binding terms.
    - Cite the specific section or wording from the context verbatim where possible.
    - If the answer is not found in the context, state "I could not find the answer in the provided documents."
    
    CONTEXT:
    {context}
    
    LEGAL QUERY:
    {query} [/INST]
    """,

    "finance": """
    [INST] You are a financial analyst AI. Your task is to interpret financial documents, bank policies, and credit terms.
    - Focus on numerical accuracy, interest rates, repayment terms, and fees.
    - Explain complex financial jargon in simple terms.
    - If the answer is not found in the context, state "I could not find the answer in the provided documents."
    
    CONTEXT:
    {context}
    
    FINANCE QUERY:
    {query} [/INST]
    """,

    "academic": """
    [INST] You are an academic research assistant. Your task is to summarize research papers and generate insights.
    - Focus on the methodology, key findings, and conclusions.
    - Generate citations or references to specific authors/sections mentioned in the text.
    - If the answer is not found in the context, state "I could not find the answer in the provided documents."
    
    CONTEXT:
    {context}
    
    ACADEMIC QUERY:
    {query} [/INST]
    """,

    "healthcare": """
    [INST] You are a medical information assistant. Your task is to extract patient history and treatment details from reports.
    - IMPORTANT: Do not provide medical advice or diagnosis. Only summarize what is explicitly stated in the text.
    - If the answer is not found in the context, state "I could not find the answer in the provided documents."
    
    CONTEXT:
    {context}
    
    HEALTHCARE QUERY:
    {query} [/INST]
    """,

    "business": """
    [INST] You are a business executive assistant. Your task is to analyze meeting transcripts and business documents.
    - Extract key action items, decisions made, and deadlines.
    - Identify who is responsible for which task (owners).
    - If the answer is not found in the context, state "I could not find the answer in the provided documents."
    
    CONTEXT:
    {context}
    
    BUSINESS QUERY:
    {query} [/INST]
    """
}

# --- 2. Retrieval Logic ---

def retrieve_docs(query: str, user_id: str, file_id: str | None = None, mode: str = "general", top_k: int = 4):
    """
    Retrieves documents from the specific collection associated with the selected Mode.
    """
    
    # --- MULTI-TENANCY: Map Mode to Collection ---
    # This ensures 'Legal Mode' only searches 'legal_docs'
    collection_map = {
        "legal": "legal_docs",
        "healthcare": "medical_docs",
        "academic": "academic_docs",
        "finance": "finance_docs",
        "business": "general_docs",
        "general": "general_docs" 
    }
    
    target_collection = collection_map.get(mode, "general_docs")
    
    # Connect to that specific DB
    col = get_collection(target_collection)
    
    q_emb = embed_texts([query])[0]

    # Filter Logic
    base_filter = {"user_id": user_id}
    
    if file_id:
        where_filter = {
            "$and": [
                base_filter,
                {"file_id": file_id}
            ]
        }
    else:
        where_filter = base_filter

    res = col.query(
        query_embeddings=[q_emb], 
        n_results=top_k,
        where=where_filter,
        # We remove "ids" from include because Chroma returns them by default
        include=["documents", "metadatas"] 
    )
    
    results = []
    if res["documents"]:
        for i in range(len(res["documents"][0])):
            results.append({
                "id": res["ids"][0][i],
                "text": res["documents"][0][i],
                "meta": res["metadatas"][0][i]
            })
            
    return results

def format_docs(docs: list[dict]) -> str:
    """Helper function to format retrieved docs into a string for the LLM."""
    return "\n\n---\n\n".join([d["text"] for d in docs])

# --- 3. Main RAG Pipeline ---

async def answer_query(query: str, user_id: str, file_id: str | None = None, mode: str = "general"):
    """
    The main LangChain RAG pipeline.
    """
    
    llm = get_llm()
    
    # Select the appropriate template based on the mode
    template = PROMPT_TEMPLATES.get(mode, PROMPT_TEMPLATES["general"])
    rag_prompt = ChatPromptTemplate.from_template(template)
    
    def retriever(query_str: str):
        # Pass 'mode' to retriever to select the correct collection
        docs = retrieve_docs(query_str, user_id, file_id, mode)
        return format_docs(docs)

    chain = (
        RunnableParallel(
            context=(RunnablePassthrough() | retriever), 
            query=RunnablePassthrough()
        )
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    answer = await chain.ainvoke(query)
    
    # Fetch docs again to return them in the API response
    retrieved_for_response = retrieve_docs(query, user_id, file_id, mode)
    
    return {
        "answer": answer, 
        "retrieved": retrieved_for_response
    }