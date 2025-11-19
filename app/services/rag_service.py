from app.api.chroma_client import get_collection
from app.services.embedding_service import embed_texts
from app.core.llm import get_llm
from app.generation.citation_enforcer import validate_citations
from app.metrics.confidence_calculator import calculate_confidence

from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableParallel
import time
import logging

logger = logging.getLogger(__name__)

# --- 1. STRICT PROMPTS with CITATION EXAMPLES ---
# (PROMPT_TEMPLATES dict remains, but logic is overridden by dynamic builder below)

# --- 2. Retrieval Logic with Scoring ---

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
        "business": "general_docs",
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

def format_docs(docs: list[dict]) -> str:
    return "\n\n---\n\n".join([d["text"] for d in docs])

# --- 3. STRICT Citation Validation (FIXED) ---

def validate_answer_citations(answer: str, retrieved_docs: list[dict], mode: str = "general") -> tuple[bool, dict]:
    """
    Validates that answers are properly cited.
    Different modes have different strictness levels.
    Returns: (is_valid, validation_details)
    """
    citation_val = validate_citations(answer, retrieved_docs)
    
    logger.debug(f"Citation validation - Mode: {mode}")
    logger.debug(f"  Total citations found: {citation_val.get('total_citations', 0)}")
    logger.debug(f"  Valid citations: {citation_val.get('valid_citations', 0)}")
    logger.debug(f"  Coverage: {citation_val.get('coverage', 0):.2%}")
    
    # --- FIX: Smarter Refusal Detection ---
    refusal_phrases = [
        "i could not find the answer",
        "i cannot find",
        "information is not available",
        "not found in the provided documents"
    ]
    
    # Check if answer STARTS with a refusal phrase (ignoring case/whitespace)
    ans_lower = answer.lower().strip()
    is_refusal_text = any(ans_lower.startswith(phrase) for phrase in refusal_phrases)
    
    valid_cites = citation_val.get("valid_citations", 0)

    # Only consider it a "Refusal" if there are NO valid citations.
    # This allows "Mixed Refusals" (e.g., "I couldn't find X, but here is Y [DOC 1]") to pass.
    if is_refusal_text and valid_cites == 0:
        logger.info("Answer is a refusal (and has no citations) - marking as valid")
        return True, citation_val
    
    # If it's a substantive answer (even if it hedged), check citations
    total_cites = citation_val.get("total_citations", 0)
    
    # Academic mode: STRICT - require at least 1 citation, then check coverage
    if mode == "academic":
        if total_cites == 0:
            logger.warning("Academic mode: NO citations found in answer - INVALID")
            return False, citation_val
        
        coverage = citation_val.get("coverage", 0)
        if coverage < 0.75:
            logger.warning(f"Academic mode: Citation coverage {coverage:.2%} < 0.75 - INVALID")
            return False, citation_val
        
        logger.info(f"Academic mode: Valid answer with {total_cites} citations, coverage {coverage:.2%}")
        return True, citation_val
    
    # Other modes: LENIENT - accept if LLM attempted citations
    else:
        if total_cites == 0:
            logger.warning(f"{mode} mode: NO citations found in answer - INVALID")
            return False, citation_val
        
        # LENIENT: If LLM generated citations AND at least 1 is valid, accept
        if valid_cites >= 1:
            logger.info(f"{mode} mode: Valid answer with {valid_cites}/{total_cites} valid citations (coverage {citation_val.get('coverage', 0):.2%})")
            return True, citation_val
        
        # Stricter: If ALL citations are invalid, reject
        if total_cites > 0 and valid_cites == 0:
            logger.warning(f"{mode} mode: LLM generated {total_cites} citations but ALL are invalid - INVALID")
            return False, citation_val
        
        return True, citation_val

# --- 4. Main RAG Pipeline ---

async def answer_query(query: str, user_id: str, file_id: str | None = None, mode: str = "general"):
    """
    STRICT RAG pipeline: evidence-only, citation-enforced, validation-checked.
    Prompts are dynamically built with actual doc IDs for citation examples.
    Returns Answer + Rich Metrics.
    """
    total_start_time = time.time()
    llm = get_llm()
    
    # 1. Retrieval Step
    retrieved_docs, retrieval_time, scores = retrieve_with_scores(query, user_id, file_id, mode, top_k=6)
    context_str = format_docs(retrieved_docs)
    
    avg_similarity = sum(scores) / len(scores) if scores else 0.0
    
    logger.info(f"RAG Query - Mode: {mode}, Retrieved: {len(retrieved_docs)} docs, Avg similarity: {avg_similarity:.3f}")
    
    # 2. Mode Policy Check (Require documents)
    if not retrieved_docs:
        logger.info("No documents retrieved - returning refusal")
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
                "hallucination_risk": "Potential"
            }
        }

    # 3. Generation Step - Build Dynamic Prompt with Actual Doc IDs
    template = _build_dynamic_prompt(mode, retrieved_docs)
    rag_prompt = PromptTemplate.from_template(template)
    
    chain = rag_prompt | llm
    
    gen_start_time = time.time()
    ai_message = await chain.ainvoke({"context": context_str, "query": query})
    gen_time = time.time() - gen_start_time
    
    # 4. Extract answer and metrics
    answer_text = ai_message.content
    meta = ai_message.response_metadata
    token_usage = meta.get("token_usage", {})
    
    input_tokens = token_usage.get("prompt_tokens", 0)
    output_tokens = token_usage.get("completion_tokens", 0)
    total_tokens = token_usage.get("total_tokens", 0)
    
    logger.debug(f"LLM generated {output_tokens} tokens. Answer preview: {answer_text[:100]}...")
    
    # 5. STRICT VALIDATION: Check citations (with mode-aware logic)
    is_valid, citation_val = validate_answer_citations(answer_text, retrieved_docs, mode)
    
    if not is_valid:
        logger.warning(f"Answer validation FAILED for {mode} mode. Total citations: {citation_val.get('total_citations', 0)}, Coverage: {citation_val.get('coverage', 0)}")
        logger.warning(f"Rejected answer preview: {answer_text[:150]}...")
        
        answer_text = "I could not find the answer in the provided documents."
        citation_val["coverage"] = 0.0
    
    # 6. Calculate improved confidence using multi-factor heuristic
    conf = calculate_confidence(query, retrieved_docs, answer_text, citation_val)
    
    total_time = time.time() - total_start_time

    # 7. Metrics Object
    metrics = {
        "processing_time_total": round(total_time, 3),
        "retrieval_time": round(retrieval_time, 3),
        "generation_time": round(gen_time, 3),
        "token_usage": {
            "input": input_tokens,
            "output": output_tokens,
            "total": total_tokens
        },
        "similarity_score": round(avg_similarity, 3),
        "confidence_category": conf["confidence_category"],
        "confidence_score": conf["confidence_score"],
        "hallucination_risk": "Low" if citation_val.get("coverage", 0) > 0.7 else "Potential",
        "citation_validation": {
            "total_citations": citation_val.get("total_citations", 0),
            "valid_citations": citation_val.get("valid_citations", 0),
            "coverage": round(citation_val.get("coverage", 0), 2)
        }
    }
    
    return {
        "answer": answer_text, 
        "retrieved": retrieved_docs,
        "metrics": metrics
    }

# --- Dynamic Prompt Builder ---

def _build_dynamic_prompt(mode: str, retrieved_docs: list[dict]) -> str:
    """
    Builds mode-specific prompt with ACTUAL doc IDs from retrieved documents.
    """
    
    # Extract ALL actual doc IDs
    doc_ids = [doc["id"] for doc in retrieved_docs]
    doc_ids_str = ", ".join(doc_ids)
    
    if mode == "academic":
        return f"""You are a rigorous academic researcher. Explain concepts STRICTLY from the provided scholarly materials.

⚠️ CRITICAL SECURITY INSTRUCTION ⚠️
YOU MUST FOLLOW THIS OR YOUR RESPONSE WILL BE REJECTED:

ALLOWED DOC IDS (ONLY these IDs exist in the provided documents):
{doc_ids_str}

RULES:
1. Use ONLY information from the provided documents.
2. For EVERY factual claim, cite EXACTLY ONE of the allowed doc IDs above.
3. Citation format: [DOC 7a0e43c6-4e1d-44c2-9fed-447bf1981204_XX] where XX is from the ALLOWED list.
4. Example CORRECT citation: [DOC {doc_ids[0]}]
5. If you cite an ID not in the ALLOWED list, your answer will be REJECTED.
6. Use exact terminology from documents (e.g., "vector" not "array").
7. If the answer cannot be inferred from the documents, say: "I could not find the answer in the provided documents."
8. Structure: Definition, Key Components, Architecture, Examples, Applications.

REMEMBER: Only cite IDs from this list:
{doc_ids_str}

CONTEXT:
{{context}}

ACADEMIC QUESTION: {{query}}

EXPLANATION:"""
    
    elif mode == "legal":
        return f"""You are a legal analysis expert. Analyze the provided legal documents STRICTLY.

⚠️ CRITICAL SECURITY INSTRUCTION ⚠️

ALLOWED DOC IDS (ONLY):
{doc_ids_str}

RULES:
1. Extract legal clauses EXACTLY as stated.
2. Cite [DOC allowed_id_from_above] for EVERY legal point.
3. ONLY cite IDs from the allowed list above.
4. If you cite an ID not in the list, your answer WILL BE REJECTED.
5. Highlight risks ONLY if explicitly in documents.
6. If not in documents, say: "I could not find the answer in the provided documents."

CONTEXT:
{{context}}

LEGAL QUESTION: {{query}}

ANALYSIS:"""
    
    elif mode == "finance":
        return f"""You are a financial analyst. Interpret financial documents STRICTLY.

⚠️ CRITICAL SECURITY INSTRUCTION ⚠️

ALLOWED DOC IDS (ONLY):
{doc_ids_str}

RULES:
1. Report financial figures EXACTLY as stated.
2. Cite [DOC allowed_id] for EVERY number or metric.
3. ONLY cite IDs from the list above.
4. Do NOT invent doc IDs.
5. Provide context for each figure.
6. If missing, say: "I could not find the answer in the provided documents."

CONTEXT:
{{context}}

FINANCIAL QUESTION: {{query}}

ANALYSIS:"""
    
    elif mode == "healthcare":
        return f"""You are a medical information specialist. Extract patient information STRICTLY.

⚠️ CRITICAL SECURITY INSTRUCTION ⚠️

ALLOWED DOC IDS (ONLY):
{doc_ids_str}

RULES:
1. Summarize ONLY what is explicitly stated.
2. Cite [DOC allowed_id] for EVERY fact.
3. ONLY cite IDs from the list above.
4. Do NOT cite IDs outside this list.
5. No medical advice or diagnosis.
6. If missing, say: "I could not find the answer in the provided documents."

CONTEXT:
{{context}}

HEALTHCARE QUESTION: {{query}}

SUMMARY:"""
    
    elif mode == "business":
        return f"""You are a business operations analyst. Analyze business documents STRICTLY.

⚠️ CRITICAL SECURITY INSTRUCTION ⚠️

ALLOWED DOC IDS (ONLY):
{doc_ids_str}

RULES:
1. Extract action items, decisions, deadlines as documented.
2. Cite [DOC allowed_id] for EACH fact.
3. ONLY cite IDs from the list above.
4. Identify responsible owners.
5. If incomplete, say: "I could not find the answer in the provided documents."

CONTEXT:
{{context}}

BUSINESS QUESTION: {{query}}

REPORT:"""
    
    else:  # general mode - NOW WITH STRICT FORMAT
        return f"""You are a helpful assistant. Answer using ONLY the provided documents.

⚠️ CRITICAL SECURITY INSTRUCTION ⚠️
YOU MUST FOLLOW THIS OR YOUR RESPONSE WILL BE REJECTED:

ALLOWED DOC IDS (ONLY these IDs exist in the provided documents):
{doc_ids_str}

RULES:
1. Base answer ONLY on provided context.
2. Cite [DOC allowed_id] for EVERY factual claim.
3. Citation format: [DOC {doc_ids[0]}] (exact format shown here).
4. ONLY cite IDs from the list above.
5. Example CORRECT citation: "The laptop is a Dell XPS13 [DOC {doc_ids[0]}]"
6. Example WRONG citation: "[{doc_ids[0]}]" or "({doc_ids[0]})" - DO NOT DO THIS
7. If you cite any ID not in the ALLOWED list, your answer will be REJECTED.
8. If information is missing, say: "I could not find the answer in the provided documents."

REMEMBER: Only cite IDs from this list:
{doc_ids_str}

CONTEXT:
{{context}}

QUESTION: {{query}}

ANSWER:"""

# --- Backward Compatibility Wrapper ---
def retrieve_docs(query: str, user_id: str, file_id: str | None = None, mode: str = "general", top_k: int = 6):
    docs, _, _ = retrieve_with_scores(query, user_id, file_id, mode, top_k)
    return docs