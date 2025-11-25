from app.api.chroma_client import get_collection
from app.services.embedding_service import embed_texts
from app.core.llm import get_llm
from app.generation.citation_enforcer import validate_citations
from app.metrics.confidence_calculator import calculate_confidence
from app.services.evaluator import evaluator
from langchain.prompts import ChatPromptTemplate, PromptTemplate
import time
import logging
import re
from typing import Dict, Any

logger = logging.getLogger(__name__)

# --- HELPERS ---

async def route_query(query: str) -> str:
    llm = get_llm()
    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Classify as 'general' or 'specific'."), ("user", "{query}")
        ])
        return (await (prompt | llm).ainvoke({"query": query})).content.strip().lower()
    except: return "specific"

async def generate_hyde_query(query: str) -> str:
    llm = get_llm()
    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Generate a hypothetical answer."), ("user", "{query}")
        ])
        return (await (prompt | llm).ainvoke({"query": query})).content.strip()
    except: return query

def normalize_citations(text: str) -> str:
    """
    Robustly standardizes citations.
    Converts: (Source 1), [Source 1], (1), [1], [Doc 1] -> [Source 1]
    """
    # Regex explanation:
    # [\(\[]       -> Match either '(' or '['
    # (?:...)?     -> Optional non-capturing group for prefix "Doc" or "Source"
    # (\d+)        -> Capture the number (Group 1)
    # [\)\]]       -> Match either ')' or ']'
    return re.sub(
        r"[\(\[](?:Doc\s?|Source\s?)?(\d+)[\)\]]", 
        lambda m: f"[Source {m.group(1)}]", 
        text, 
        flags=re.IGNORECASE
    )

# --- RETRIEVAL ---

async def retrieve_with_scores(query: str, user_id: str, file_id: str | None, mode: str, top_k: int = 6):
    start = time.time()
    search_query = query
    
    # Smart Retrieval (HyDE)
    if not file_id and mode in ["general", "academic"]:
        if "general" in await route_query(query):
            search_query = await generate_hyde_query(query)

    # Collection Selection
    col_name = {
        "legal": "legal_docs", "healthcare": "medical_docs",
        "academic": "academic_docs", "finance": "finance_docs", 
        "business": "business_docs"
    }.get(mode, "general_docs")
    
    col = get_collection(col_name)
    q_emb = embed_texts([search_query])[0]
    
    # Filtering
    where = {"user_id": user_id}
    if file_id: where = {"$and": [{"user_id": user_id}, {"file_id": file_id}]}

    res = col.query(query_embeddings=[q_emb], n_results=top_k, where=where, include=["documents", "metadatas", "distances"])
    
    results = []
    scores = []
    if res["documents"]:
        for i in range(len(res["documents"][0])):
            score = max(0, 1.0 - res["distances"][0][i])
            scores.append(score)
            results.append({
                "id": res["ids"][0][i],
                "text": res["documents"][0][i],
                "metadata": res["metadatas"][0][i],
                "score": round(score, 4)
            })
            
    return results, time.time() - start, scores

# --- MAIN PIPELINE ---

async def answer_query(query: str, user_id: str, file_id: str | None = None, mode: str = "general", chat_history: str = ""):
    total_start = time.time()
    llm = get_llm()
    
    # 1. Retrieve
    retrieved, ret_time, scores = await retrieve_with_scores(query, user_id, file_id, mode)
    
    if not retrieved:
        return _empty_response(query, ret_time, total_start)

    # 2. Context Building
    formatted_docs = []
    for i, d in enumerate(retrieved):
        clean_text = d["text"].replace("\n", " ")
        formatted_docs.append(f"[Source {i+1}] {clean_text}")
    context_str = "\n\n".join(formatted_docs)
    
    # 3. Prompting
    source_list = ", ".join([f"[Source {i+1}]" for i in range(len(retrieved))])
    role = "You are a helpful assistant."
    if mode == "legal": role = "You are a legal expert."
    elif mode == "academic": role = "You are a researcher."

    prompt_str = f"""{role}
INSTRUCTIONS:
1. Answer using ONLY the context.
2. Cite every claim as [Source X].
3. If unsure, say "I cannot find the answer."

AVAILABLE SOURCES:
{source_list}

HISTORY:
{chat_history}

CONTEXT:
{context_str}

QUESTION: {query}
ANSWER:"""

    # 4. Generation
    gen_start = time.time()
    ai_msg = await (PromptTemplate.from_template(prompt_str) | llm).ainvoke({})
    gen_time = time.time() - gen_start
    
    # 5. Token Mapping
    raw_usage = ai_msg.response_metadata.get("token_usage", {})
    token_metrics = {
        "input": raw_usage.get("prompt_tokens", 0),
        "output": raw_usage.get("completion_tokens", 0),
        "total": raw_usage.get("total_tokens", 0)
    }

    # 6. Processing Answer (Normalization)
    # This now handles (Source 1) -> [Source 1] conversion
    final_answer = normalize_citations(ai_msg.content)
    
    # 7. Validation
    val_docs = [{"id": f"Source {i+1}", "text": d["text"]} for i, d in enumerate(retrieved)]
    
    # Helper to calculate validity since validate_citations only returns dict
    citation_metrics = validate_citations(final_answer, val_docs)
    
    # 8. Metrics
    conf = calculate_confidence(query, retrieved, final_answer, citation_metrics)
    
    metrics = {
        "processing_time_total": round(time.time() - total_start, 3),
        "retrieval_time": round(ret_time, 3),
        "generation_time": round(gen_time, 3),
        "token_usage": token_metrics,
        "similarity_score": round(sum(scores)/len(scores), 3) if scores else 0,
        "confidence_category": conf["confidence_category"],
        "confidence_score": conf["confidence_score"],
        "hallucination_risk": "Low" if citation_metrics.get("coverage", 0) > 0.6 else "Potential",
        "citation_validation": citation_metrics
    }
    
    return {
        "answer": final_answer,
        "retrieved": retrieved,
        "metrics": metrics
    }

def _empty_response(query, ret_time, start_time):
    return {
        "answer": "I could not find relevant documents.",
        "retrieved": [],
        "metrics": {
            "processing_time_total": round(time.time() - start_time, 3),
            "retrieval_time": round(ret_time, 3),
            "generation_time": 0.0,
            "token_usage": {"input": 0, "output": 0, "total": 0},
            "similarity_score": 0.0,
            "confidence_category": "Low",
            "confidence_score": 0.0,
            "hallucination_risk": "High",
            "citation_validation": {},
            "evaluation": {}
        }
    }

# Compat wrapper
async def retrieve_docs(query, user_id, file_id=None, mode="general", top_k=6):
    d, _, _ = await retrieve_with_scores(query, user_id, file_id, mode, top_k)
    return d