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
    Confidence score combining retrieval, citation coverage, and depth.
    """
    
    summary_keywords = ["summarize", "summary", "overview", "brief", "detail", "explain", "review"]
    is_summarization = any(w in query.lower() for w in summary_keywords)

    # 1. Retrieval Score
    if retrieved_docs:
        if is_summarization:
            # Boost score for instructional queries, but do not assume perfection.
            # Even if we found the file, relevance might vary. Cap at 95.
            retrieval_score = 95.0 
        else:
            avg_sim = sum(d.get("score", 0) for d in retrieved_docs) / len(retrieved_docs)
            retrieval_score = min(avg_sim * 100, 100)
    else:
        retrieval_score = 0
    
    # 2. Citation Score (The strongest signal of hallucination resistance)
    citation_coverage = citation_validation.get("coverage", 0)
    citation_score = citation_coverage * 100
    
    # 3. Query Coverage
    if is_summarization:
        query_score = citation_score 
    else:
        query_terms = set(query.lower().split())
        answer_terms = set(answer.lower().split())
        # Avoid division by zero
        if not query_terms:
             query_score = 0
        else:
             query_coverage = len(query_terms & answer_terms) / len(query_terms)
             query_score = query_coverage * 100
    
    # 4. Depth Score
    answer_words = len(answer.split())
    if is_summarization:
        depth_score = min((answer_words / 100) * 100, 100)
    else:
        depth_score = min((answer_words / 500) * 100, 100)
    
    # Weighted sum
    raw_score = (
        0.3 * retrieval_score +
        0.3 * citation_score +
        0.2 * query_score +
        0.2 * depth_score
    )
    
    # --- CALIBRATION FIX ---
    # Cap the maximum confidence at 95% to account for inherent LLM uncertainty.
    # No model is 100% accurate.
    confidence_score = min(raw_score, 95.0)
    
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