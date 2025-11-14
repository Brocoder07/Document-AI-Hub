# --- THIS IS THE FIX ---
from app.api.chroma_client import get_collection
# --- (Was app.api.utility.chroma_client) ---
from app.services.embedding_service import embed_texts
from app.core.llm import get_llm

from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableParallel


def retrieve_docs(query: str, user_id: str, file_id: str | None = None, top_k: int = 4):
    """
    Retrieves document chunks from ChromaDB, with security filters.
    """
    col = get_collection("documents")
    q_emb = embed_texts([query])[0]

    where_filter = {"user_id": user_id}
    if file_id:
        where_filter["file_id"] = file_id

    res = col.query(
        query_embeddings=[q_emb], 
        n_results=top_k,
        where=where_filter,
        include=["documents"]
    )
    
    return [doc for doc in res["documents"][0]] if res["documents"] else []

def format_docs(docs: list[str]) -> str:
    """Helper function to format retrieved docs into a string."""
    return "\n\n---\n\n".join(docs)


# --- LangChain RAG Pipeline ---

async def answer_query(query: str, user_id: str, file_id: str | None = None):
    """
    The main LangChain RAG pipeline.
    """
    
    llm = get_llm()
    
    RAG_PROMPT_TEMPLATE = """
    [INST] You are a helpful assistant. Use the following context to answer the question.
    If the answer is not present in the context, say "I could not find the answer in the provided documents."

    CONTEXT:
    {context}

    QUESTION:
    {query} [/INST]
    """
    rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    
    def retriever(query_str: str):
        docs = retrieve_docs(query_str, user_id, file_id)
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
    
    retrieved_for_response = retrieve_docs(query, user_id, file_id)
    
    return {
        "answer": answer, 
        "retrieved": [{"text": doc} for doc in retrieved_for_response]
    }