"""
Cross-Encoder Re-ranking Service

Uses a cross-encoder model to re-rank retrieved documents by actual relevance
to the query. This is far more accurate than bi-encoder similarity scores alone.

Cross-encoders process (query, document) pairs jointly, enabling deep 
token-level interaction — critical for technical 3GPP terminology.
"""

from sentence_transformers import CrossEncoder
from app.core.config import settings
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

_reranker = None


def get_reranker():
    """Lazy-load the cross-encoder re-ranking model."""
    global _reranker
    if _reranker is None:
        model_name = settings.RERANKER_MODEL
        logger.info(f" Loading re-ranker model: {model_name}")
        _reranker = CrossEncoder(model_name, max_length=512)
        logger.info(" Re-ranker model loaded.")
    return _reranker


def rerank_documents(
    query: str, 
    documents: List[Dict[str, Any]], 
    top_k: int = 8,
    relevance_threshold: float = 0.15
) -> List[Dict[str, Any]]:
    """
    Re-ranks retrieved documents using a cross-encoder.
    
    Args:
        query: The user's search query
        documents: List of retrieved document dicts (must have 'text' key)
        top_k: Maximum number of documents to return after re-ranking
        relevance_threshold: Minimum relevance score to keep a document.
                            Documents below this are discarded as irrelevant.
    
    Returns:
        Re-ranked and filtered list of document dicts, sorted by relevance.
        Each dict gets an updated 'rerank_score' field.
    """
    if not documents:
        return []
    
    reranker = get_reranker()
    
    # Build query-document pairs for the cross-encoder
    pairs = [(query, doc["text"]) for doc in documents]
    
    # Score all pairs
    scores = reranker.predict(pairs, show_progress_bar=False)
    
    # Attach scores to documents
    scored_docs = []
    for i, doc in enumerate(documents):
        doc_copy = doc.copy()
        doc_copy["rerank_score"] = float(scores[i])
        scored_docs.append(doc_copy)
    
    # Sort by re-rank score (highest first)
    scored_docs.sort(key=lambda x: x["rerank_score"], reverse=True)
    
    # Filter by relevance threshold
    filtered = [d for d in scored_docs if d["rerank_score"] >= relevance_threshold]
    
    if len(filtered) < len(scored_docs):
        discarded = len(scored_docs) - len(filtered)
        logger.info(
            f" [Re-rank] Discarded {discarded}/{len(scored_docs)} chunks "
            f"below threshold ({relevance_threshold})"
        )
    
    # Take top_k
    result = filtered[:top_k]
    
    if result:
        logger.info(
            f" [Re-rank] Kept {len(result)} docs. "
            f"Score range: {result[0]['rerank_score']:.3f} → {result[-1]['rerank_score']:.3f}"
        )
    else:
        logger.warning(" [Re-rank] No documents passed relevance threshold!")
    
    return result
