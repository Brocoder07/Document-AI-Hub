"""
Confidence Calculation — Production Grade

Multi-factor confidence scoring that prioritizes:
1. Faithfulness Score (40%) — Are claims grounded in sources?
2. Retrieval Quality (25%) — How relevant were the retrieved chunks?
3. Citation Coverage (20%) — Are claims properly cited with accurate references?
4. Answer Coherence (15%) — Is the answer substantive and well-structured?

The faithfulness score from the hallucination guard is now the PRIMARY signal.
"""

from typing import List, Dict, Any


def calculate_confidence(
    query: str,
    retrieved_docs: List[Dict[str, Any]],
    answer: str,
    citation_validation: Dict[str, Any],
    faithfulness: Dict[str, Any] = None,
    rerank_scores: List[float] = None,
) -> Dict[str, Any]:
    """
    Production-grade confidence score combining faithfulness, retrieval, 
    citation coverage, and answer quality signals.
    """
    
    # 1. Faithfulness Score (The strongest signal — from hallucination guard)
    if faithfulness and faithfulness.get("faithfulness_score") is not None:
        faithfulness_score = faithfulness["faithfulness_score"]
    else:
        # Fallback: estimate from citation coverage
        faithfulness_score = citation_validation.get("coverage", 0) * 100
    
    # 2. Retrieval Quality Score
    if rerank_scores:
        # Use re-ranker scores (much more accurate than raw vector similarity)
        avg_rerank = sum(rerank_scores) / len(rerank_scores)
        # Cross-encoder scores are typically in [-10, 10] range, normalize to 0-100
        retrieval_score = min(max((avg_rerank + 2) / 6 * 100, 0), 100)
    elif retrieved_docs:
        avg_sim = sum(d.get("score", 0) for d in retrieved_docs) / len(retrieved_docs)
        retrieval_score = min(avg_sim * 100, 100)
    else:
        retrieval_score = 0
    
    # 3. Citation Coverage Score
    citation_coverage = citation_validation.get("coverage", 0)
    citation_score = citation_coverage * 100
    
    # 4. Answer Coherence Score
    answer_words = len(answer.split())
    if answer_words > 300:
        coherence_score = 90  # Long, detailed answer
    elif answer_words > 100:
        coherence_score = min((answer_words / 300) * 100, 100)
    elif answer_words > 30:
        coherence_score = min((answer_words / 100) * 80, 80)
    else:
        coherence_score = 30  # Very short answer
    
    # Check for refusal (which is GOOD — shows the system knows its limits)
    refusal_phrases = ["i cannot find", "i could not find", "not available", "no relevant"]
    if any(phrase in answer.lower() for phrase in refusal_phrases):
        coherence_score = 70  # Appropriate refusal is a sign of reliability
    
    # Weighted sum — Faithfulness dominates
    raw_score = (
        0.40 * faithfulness_score +
        0.25 * retrieval_score +
        0.20 * citation_score +
        0.15 * coherence_score
    )
    
    # Cap at 95% — no AI system is 100% accurate
    confidence_score = min(raw_score, 95.0)
    
    # Categorize
    if confidence_score >= 80:
        confidence_category = "High"
    elif confidence_score >= 60:
        confidence_category = "Medium"
    elif confidence_score >= 40:
        confidence_category = "Low"
    else:
        confidence_category = "Very Low"
    
    # Hallucination risk assessment
    if faithfulness and faithfulness.get("verdict"):
        verdict = faithfulness["verdict"]
        if verdict == "FAITHFUL":
            hallucination_risk = "Very Low"
        elif verdict == "MOSTLY_FAITHFUL":
            hallucination_risk = "Low"
        elif verdict == "PARTIALLY_FAITHFUL":
            hallucination_risk = "Medium"
        else:
            hallucination_risk = "High"
    else:
        hallucination_risk = "Low" if citation_coverage > 0.6 else "Potential"
    
    return {
        "confidence_score": round(confidence_score, 1),
        "confidence_category": confidence_category,
        "hallucination_risk": hallucination_risk,
        "factors": {
            "faithfulness": round(faithfulness_score, 1),
            "retrieval_quality": round(retrieval_score, 1),
            "citation_coverage": round(citation_score, 1),
            "answer_coherence": round(coherence_score, 1),
        }
    }