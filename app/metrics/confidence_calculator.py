"""
Confidence calculation: multi-factor heuristic based on retrieval, citation, and coherence.
"""
from typing import List, Dict, Any

def calculate_confidence(
    query: str,
    retrieved_docs: List[Dict[str, Any]],
    answer: str,
    citation_validation: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Confidence score combining:
     1. Retrieval quality (avg similarity score of top-k)
     2. Citation coverage (% of claims that cite valid docs)
     3. Query coverage (does answer address all major query terms?)
     4. Answer length (longer, detailed answers → higher confidence for academic)
    
    Returns: {'confidence_score': 0-100, 'confidence_category': 'Low'|'Medium'|'High', 'factors': {...}}
    """
    
    # --- FIX: Detect Instructional/Summarization Intent ---
    # Vector similarity is often low for "Summarize this" because the query 
    # (instruction) is semantically different from the document (content).
    # We define keywords that indicate the user wants to process the *whole* document.
    summary_keywords = ["summarize", "summary", "overview", "brief", "detail"]
    is_summarization = any(w in query.lower() for w in summary_keywords)

    # Factor 1: Retrieval quality (weight 0.3)
    if retrieved_docs:
        if is_summarization:
            # If the user asked to summarize and we found a document (likely via file_id),
            # we treat this as "Perfect Retrieval" regardless of vector distance.
            retrieval_score = 100.0
        else:
            avg_sim = sum(d.get("score", 0) for d in retrieved_docs) / len(retrieved_docs)
            retrieval_score = min(avg_sim * 100, 100)  # 0.62 → 62
    else:
        retrieval_score = 0
    
    # Factor 2: Citation coverage (weight 0.3)
    citation_coverage = citation_validation.get("coverage", 0)
    citation_score = citation_coverage * 100
    
    # Factor 3: Query coverage (weight 0.2)
    # Simple heuristic: check if major query terms appear in answer
    query_terms = set(query.lower().split())
    answer_terms = set(answer.lower().split())
    
    # For summarization, we don't expect the word "summarize" to appear in the answer text.
    # So we penalize less or skip common instruction words.
    if is_summarization:
        # Heuristic: Assume good coverage for summaries if we have citations
        query_score = citation_score 
    else:
        query_coverage = len(query_terms & answer_terms) / len(query_terms) if query_terms else 0
        query_score = query_coverage * 100
    
    # Factor 4: Answer depth (weight 0.2)
    # Longer, detailed answers (100+ tokens) get higher confidence for academic
    answer_words = len(answer.split())
    
    # Adjust depth expectation for summaries (which are meant to be concise)
    if is_summarization:
        # Lower the bar for "full depth" score on summaries (e.g., 100 words is plenty)
        depth_score = min((answer_words / 100) * 100, 100)
    else:
        depth_score = min((answer_words / 500) * 100, 100)  # 500 words → 100
    
    # Weighted sum
    confidence_score = (
        0.3 * retrieval_score +
        0.3 * citation_score +
        0.2 * query_score +
        0.2 * depth_score
    )
    
    # Categorize
    if confidence_score >= 75:
        confidence_category = "High"
    elif confidence_score >= 50:
        confidence_category = "Medium"
    else:
        confidence_category = "Low"
    
    return {
        "confidence_score": round(confidence_score, 1),
        "confidence_category": confidence_category,
        "factors": {
            "retrieval_quality": round(retrieval_score, 1),
            "citation_coverage": round(citation_score, 1),
            "query_coverage": round(query_score, 1),
            "answer_depth": round(depth_score, 1),
        }
    }