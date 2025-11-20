from app.api.chroma_client import get_collection
from app.services.embedding_service import embed_texts
from app.core.llm import get_llm
from app.generation.citation_enforcer import validate_citations
from app.metrics.confidence_calculator import calculate_confidence

from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableParallel
import time
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# --- 1. Retrieval Logic with Scoring ---

def retrieve_with_scores(query: str, user_id: str, file_id: str | None = None, mode: str = "general", top_k: int = 6):
    """
    Retrieves documents AND their similarity scores.
    """
    start_time = time.time()
    
    collection_map = {
        "legal": "legal_docs",
        "healthcare": "medical_docs",
        "academic": "academic_docs",
        "finance": "finance_docs",
        # FIX: Now points to the dedicated 'business_docs' collection
        "business": "business_docs",
        "general": "general_docs" 
    }
    target_collection = collection_map.get(mode, "general_docs")
    col = get_collection(target_collection)
    
    q_emb = embed_texts([query])[0]

    base_filter = {"user_id": user_id}
    if file_id:
        where_filter = {"$and": [base_filter, {"file_id": file_id}]}
    else:
        where_filter = base_filter

    res = col.query(
        query_embeddings=[q_emb], 
        n_results=top_k,
        where=where_filter,
        include=["documents", "metadatas", "distances"] 
    )
    
    retrieval_time = time.time() - start_time
    results = []
    
    scores = []

    if res["documents"]:
        for i in range(len(res["documents"][0])):
            dist = res["distances"][0][i]
            sim_score = max(0, 1.0 - dist) 
            scores.append(sim_score)
            
            results.append({
                "id": res["ids"][0][i],
                "text": res["documents"][0][i],
                "meta": res["metadatas"][0][i],
                "score": round(sim_score, 4)
            })
            
    return results, retrieval_time, scores

def format_docs_with_aliases(docs: list[dict]) -> str:
    """
    Formats documents with simple aliases [DOC 0], [DOC 1] instead of full UUIDs.
    This helps the LLM distinguish between content numbers and citation IDs.
    """
    formatted = []
    for i, d in enumerate(docs):
        # Clean text to remove newlines for cleaner prompt context
        clean_text = d["text"].replace("\n", " ")
        formatted.append(f"[DOC {i}] {clean_text}")
    return "\n\n".join(formatted)

# --- 2. STRICT Citation Validation ---

def validate_answer_citations(answer: str, retrieved_docs: list[dict], mode: str = "general") -> tuple[bool, dict]:
    """
    Validates that answers are properly cited.
    """
    citation_val = validate_citations(answer, retrieved_docs)
    
    logger.debug(f"Citation validation - Mode: {mode}")
    
    refusal_phrases = [
        "i could not find the answer",
        "i cannot find",
        "information is not available",
        "not found in the provided documents"
    ]
    
    ans_lower = answer.lower().strip()
    is_refusal_text = any(ans_lower.startswith(phrase) for phrase in refusal_phrases)
    
    valid_cites = citation_val.get("valid_citations", 0)
    total_cites = citation_val.get("total_citations", 0)

    # Logic: If it looks like a refusal BUT has 0 citations, it's valid. 
    # If it has citations, we ignore the refusal text and validate the citations.
    if is_refusal_text and valid_cites == 0:
        logger.info("Answer is a refusal (and has no citations) - marking as valid")
        return True, citation_val
    
    if mode == "academic":
        if total_cites == 0:
            return False, citation_val
        coverage = citation_val.get("coverage", 0)
        if coverage < 0.75:
            return False, citation_val
        return True, citation_val
    
    else:
        if total_cites == 0:
            return False, citation_val
        if valid_cites >= 1:
            return True, citation_val
        if total_cites > 0 and valid_cites == 0:
            return False, citation_val
        return True, citation_val

# --- 3. Main RAG Pipeline ---

async def answer_query(
    query: str, 
    user_id: str, 
    file_id: str | None = None, 
    mode: str = "general",
    chat_history: str = ""
):
    """
    STRICT RAG pipeline with Chat History support.
    Returns Answer + Rich Metrics.
    """
    total_start_time = time.time()
    llm = get_llm()
    
    # 1. Retrieval Step
    retrieved_docs, retrieval_time, scores = retrieve_with_scores(query, user_id, file_id, mode, top_k=6)
    
    # Generate simple aliases: 0, 1, 2...
    # And a map to get back to real UUIDs: {'0': 'uuid-123...', '1': 'uuid-456...'}
    doc_id_map = {str(i): doc["id"] for i, doc in enumerate(retrieved_docs)}
    
    # Format context using these aliases
    context_str = format_docs_with_aliases(retrieved_docs)
    
    avg_similarity = sum(scores) / len(scores) if scores else 0.0
    logger.info(f"RAG Query - Mode: {mode}, Retrieved: {len(retrieved_docs)} docs")
    
    # 2. Mode Policy Check
    if not retrieved_docs:
        return {
            "answer": "I could not find the answer in the provided documents.",
            "retrieved": [],
            "metrics": {
                "processing_time_total": round(time.time() - total_start_time, 3),
                "retrieval_time": round(retrieval_time, 3),
                "generation_time": 0,
                "token_usage": {"input": 0, "output": 0, "total": 0},
                "similarity_score": 0,
                "confidence_category": "Low",
                "confidence_score": 0,
                "hallucination_risk": "Potential",
                "citation_validation": {}
            }
        }

    # 3. Generation Step - Build Prompt with Aliased IDs (0, 1, 2)
    aliased_indices = list(doc_id_map.keys())
    template = _build_dynamic_prompt(mode, aliased_indices, chat_history)
    rag_prompt = PromptTemplate.from_template(template)
    
    chain = rag_prompt | llm
    
    gen_start_time = time.time()
    
    ai_message = await chain.ainvoke({
        "context": context_str, 
        "query": query,
        "history": chat_history 
    })
    gen_time = time.time() - gen_start_time
    
    # 4. Extract answer
    answer_text_aliased = ai_message.content
    meta = ai_message.response_metadata
    raw_usage = meta.get("token_usage", {})
    
    # 5. ID REPLACEMENT (Robust Swap: [DOC 0, 2] -> [DOC uuid1] [DOC uuid2])
    def replace_alias_in_match(match):
        full_tag = match.group(0)
        # Extract all integers from the tag (handles "1, 3", "1, DOC 3", etc.)
        numbers = re.findall(r'\d+', full_tag)
        
        real_ids = []
        for num in numbers:
            if num in doc_id_map:
                real_ids.append(doc_id_map[num])
        
        if not real_ids:
            return full_tag # Return original if no valid map found
            
        # Return as separate standard citations
        return " ".join([f"[DOC {rid}]" for rid in real_ids])

    # Match any bracket that starts with DOC and contains text/numbers until closing bracket
    answer_text_real = re.sub(r"\[DOC\s+[^\]]+\]", replace_alias_in_match, answer_text_aliased, flags=re.IGNORECASE)
    
    # 6. VALIDATION (Check against real UUIDs)
    is_valid, citation_val = validate_answer_citations(answer_text_real, retrieved_docs, mode)
    
    if not is_valid:
        logger.warning(f"Validation Failed. Mode: {mode}")
        answer_text_real = "I could not find the answer in the provided documents."
        citation_val["coverage"] = 0.0
    
    # 7. Metrics
    conf = calculate_confidence(query, retrieved_docs, answer_text_real, citation_val)
    total_time = time.time() - total_start_time

    metrics = {
        "processing_time_total": round(total_time, 3),
        "retrieval_time": round(retrieval_time, 3),
        "generation_time": round(gen_time, 3),
        "token_usage": {
            "input": raw_usage.get("prompt_tokens", 0),
            "output": raw_usage.get("completion_tokens", 0),
            "total": raw_usage.get("total_tokens", 0)
        },
        "similarity_score": round(avg_similarity, 3),
        "confidence_category": conf["confidence_category"],
        "confidence_score": conf["confidence_score"],
        "hallucination_risk": "Low" if citation_val.get("coverage", 0) > 0.7 else "Potential",
        "citation_validation": citation_val
    }
    
    return {
        "answer": answer_text_real, 
        "retrieved": retrieved_docs,
        "metrics": metrics
    }

# --- Dynamic Prompt Builder (Updated for Simple Aliases) ---

def _build_dynamic_prompt(mode: str, aliased_ids: list[str], chat_history: str) -> str:
    """
    Builds prompt using simple ID aliases (0, 1, 2) instead of complex UUIDs.
    """
    
    ids_str = ", ".join([f"[DOC {i}]" for i in aliased_ids])
    
    history_block = ""
    if chat_history:
        history_block = f"""
PREVIOUS CONVERSATION HISTORY:
{{history}}
(Use this history to understand context, but your facts MUST come from the documents below)
"""

    # Base instruction
    if mode == "academic":
        role_desc = "You are a rigorous academic researcher. Explain concepts STRICTLY from the provided scholarly materials."
        mode_rules = """
5. WARNING: Do NOT use page numbers or slide numbers (e.g., "Slide 28") as citations. ONLY use the exact strings listed in "ALLOWED SOURCES".
6. If the answer cannot be inferred from the documents, say: "I could not find the answer in the provided documents."
"""
    elif mode == "legal":
        role_desc = "You are a legal analysis expert. Analyze the provided legal documents STRICTLY."
        mode_rules = """
4. WARNING: Do NOT use Section numbers, Clause numbers, or Article numbers (e.g., "Section 5") as citation IDs.
5. ONLY use the [DOC X] format aliases provided in "ALLOWED SOURCES".
6. If the answer cannot be inferred from the documents, say: "I could not find the answer in the provided documents."
"""
    elif mode == "finance":
        role_desc = "You are a financial analyst. Interpret financial documents STRICTLY."
        mode_rules = """
4. WARNING: Do NOT use Line numbers or Row numbers as citation IDs.
5. ONLY use the [DOC X] format aliases provided in "ALLOWED SOURCES".
6. If the answer cannot be inferred from the documents, say: "I could not find the answer in the provided documents."
"""
    elif mode == "healthcare":
        role_desc = "You are a medical information specialist. Extract patient information STRICTLY."
        mode_rules = """
4. WARNING: Do NOT use Patient IDs or Record numbers as citation IDs.
5. ONLY use the [DOC X] format aliases provided in "ALLOWED SOURCES".
6. If the answer cannot be inferred from the documents, say: "I could not find the answer in the provided documents."
"""
    elif mode == "business":
        role_desc = "You are a business operations analyst. Analyze business documents STRICTLY."
        mode_rules = """
4. WARNING: Do NOT use Agenda item numbers or List numbers as citation IDs.
5. ONLY use the [DOC X] format aliases provided in "ALLOWED SOURCES".
6. If the answer cannot be inferred from the documents, say: "I could not find the answer in the provided documents."
"""
    else:
        role_desc = "You are a helpful assistant. Answer using ONLY the provided documents."
        mode_rules = '4. If the answer cannot be inferred from the documents, say: "I could not find the answer in the provided documents."'

    # Rules modified to focus on the simple [DOC X] format
    common_rules = f"""
⚠️ CRITICAL SECURITY INSTRUCTION ⚠️
You are provided with document chunks labeled [DOC 0], [DOC 1], etc.

ALLOWED SOURCES (ONLY):
{ids_str}

RULES:
1. Use ONLY information from the provided documents.
2. For EVERY factual claim, cite the source using the [DOC X] format.
3. Example: "The clause specifies a 30-day notice [DOC 0]."
{mode_rules}
"""

    return f"""{role_desc}
{common_rules}

{history_block}

CONTEXT:
{{context}}

QUESTION: {{query}}

ANSWER:"""

# --- Backward Compatibility Wrapper ---
def retrieve_docs(query: str, user_id: str, file_id: str | None = None, mode: str = "general", top_k: int = 6):
    docs, _, _ = retrieve_with_scores(query, user_id, file_id, mode, top_k)
    return docs