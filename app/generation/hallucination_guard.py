"""
Hallucination Guard — Post-Generation Verification

The core differentiator for achieving near-zero hallucination.

Pipeline:
1. Extract individual factual claims from the LLM's answer
2. Check each claim against retrieved source texts using semantic similarity
3. Score each claim as SUPPORTED / PARTIALLY_SUPPORTED / UNSUPPORTED
4. Compute an overall faithfulness score
5. Flag or strip unsupported claims

This is fundamentally different from citation checking — it verifies the 
CONTENT of claims, not just whether a [Source X] tag exists.
"""

from sentence_transformers import SentenceTransformer, util
from app.core.config import settings
from typing import List, Dict, Any, Tuple
import re
import logging

logger = logging.getLogger(__name__)

_grounding_model = None


def get_grounding_model():
    """Reuse the embedding model for grounding checks."""
    global _grounding_model
    if _grounding_model is None:
        _grounding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
    return _grounding_model


def extract_claims(answer: str) -> List[str]:
    """
    Split an LLM answer into individual factual claims.
    
    Strategy:
    - Split by sentences
    - Filter out non-factual content (questions, disclaimers, meta-statements)
    - Keep only substantive claims that can be verified
    """
    # Remove citation markers for clean claim extraction
    clean = re.sub(r'\[Source\s*\d+\]', '', answer)
    clean = re.sub(r'\[Doc\s*\d+\]', '', clean)
    
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', clean)
    
    claims = []
    skip_patterns = [
        r'^(I cannot|I could not|I don\'t|Based on|According to)',  # Meta-statements
        r'^(Note:|Disclaimer:|Warning:)',                            # Disclaimers
        r'\?$',                                                      # Questions
        r'^(Yes|No|Sure|Of course)[,.]',                            # Filler
    ]
    
    for sentence in sentences:
        sentence = sentence.strip()
        
        # Skip empty or very short sentences
        if len(sentence) < 15:
            continue
        
        # Skip non-factual patterns
        should_skip = False
        for pattern in skip_patterns:
            if re.match(pattern, sentence, re.IGNORECASE):
                should_skip = True
                break
        
        if not should_skip:
            claims.append(sentence)
    
    return claims


def verify_claims(
    claims: List[str], 
    source_texts: List[str],
    support_threshold: float = 0.55,
    partial_threshold: float = 0.40
) -> List[Dict[str, Any]]:
    """
    Verify each claim against source texts using semantic similarity.
    
    Returns:
        List of claim verification results:
        {
            "claim": str,
            "status": "SUPPORTED" | "PARTIALLY_SUPPORTED" | "UNSUPPORTED",
            "max_similarity": float,
            "best_source_idx": int
        }
    """
    if not claims or not source_texts:
        return []
    
    model = get_grounding_model()
    
    # Encode all claims and sources
    claim_embeddings = model.encode(claims, convert_to_tensor=True)
    source_embeddings = model.encode(source_texts, convert_to_tensor=True)
    
    # Compute similarity matrix: claims x sources
    similarity_matrix = util.cos_sim(claim_embeddings, source_embeddings)
    
    results = []
    for i, claim in enumerate(claims):
        # Get max similarity across all sources
        max_sim = float(similarity_matrix[i].max())
        best_source = int(similarity_matrix[i].argmax())
        
        if max_sim >= support_threshold:
            status = "SUPPORTED"
        elif max_sim >= partial_threshold:
            status = "PARTIALLY_SUPPORTED"
        else:
            status = "UNSUPPORTED"
        
        results.append({
            "claim": claim,
            "status": status,
            "max_similarity": round(max_sim, 3),
            "best_source_idx": best_source
        })
    
    return results


def compute_faithfulness_score(claim_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute an overall faithfulness score from individual claim verifications.
    
    Returns:
        {
            "faithfulness_score": float (0-100),
            "supported_count": int,
            "partially_supported_count": int,
            "unsupported_count": int,
            "total_claims": int,
            "claim_details": List[Dict],
            "verdict": str
        }
    """
    if not claim_results:
        return {
            "faithfulness_score": 0.0,
            "supported_count": 0,
            "partially_supported_count": 0,
            "unsupported_count": 0,
            "total_claims": 0,
            "claim_details": [],
            "verdict": "NO_CLAIMS"
        }
    
    supported = sum(1 for r in claim_results if r["status"] == "SUPPORTED")
    partial = sum(1 for r in claim_results if r["status"] == "PARTIALLY_SUPPORTED")
    unsupported = sum(1 for r in claim_results if r["status"] == "UNSUPPORTED")
    total = len(claim_results)
    
    # Weighted score: SUPPORTED=1.0, PARTIAL=0.5, UNSUPPORTED=0.0
    raw_score = ((supported * 1.0) + (partial * 0.5)) / total * 100
    
    # Determine verdict
    if raw_score >= 85:
        verdict = "FAITHFUL"
    elif raw_score >= 60:
        verdict = "MOSTLY_FAITHFUL"
    elif raw_score >= 40:
        verdict = "PARTIALLY_FAITHFUL"
    else:
        verdict = "UNRELIABLE"
    
    return {
        "faithfulness_score": round(raw_score, 1),
        "supported_count": supported,
        "partially_supported_count": partial,
        "unsupported_count": unsupported,
        "total_claims": total,
        "claim_details": claim_results,
        "verdict": verdict
    }


def run_hallucination_check(answer: str, source_texts: List[str]) -> Dict[str, Any]:
    """
    Full hallucination detection pipeline.
    
    Args:
        answer: The LLM's generated answer
        source_texts: List of retrieved source document texts
    
    Returns:
        Complete faithfulness analysis with claim-level details
    """
    # 1. Extract claims
    claims = extract_claims(answer)
    
    if not claims:
        logger.info(" [Hallucination Guard] No verifiable claims found in answer.")
        return compute_faithfulness_score([])
    
    logger.info(f" [Hallucination Guard] Extracted {len(claims)} claims to verify.")
    
    # 2. Verify each claim
    claim_results = verify_claims(claims, source_texts)
    
    # 3. Compute overall score
    result = compute_faithfulness_score(claim_results)
    
    # Log summary
    logger.info(
        f" [Hallucination Guard] Faithfulness: {result['faithfulness_score']}% "
        f"({result['verdict']}) | "
        f" {result['supported_count']} supported, "
        f" {result['partially_supported_count']} partial, "
        f" {result['unsupported_count']} unsupported"
    )
    
    return result
