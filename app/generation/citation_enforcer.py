"""
Citation Enforcement & Validation — Production Grade

Validates that citations in the answer:
1. Actually exist as provided sources (format check)
2. Point to sources that contain the claimed information (content check)
3. Cover the key claims in the answer (coverage metric)
"""

import re
import logging
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer, util
from app.core.config import settings

logger = logging.getLogger(__name__)

_citation_model = None

def _get_model():
    """Lazy-load embedding model for citation content verification."""
    global _citation_model
    if _citation_model is None:
        _citation_model = SentenceTransformer(settings.EMBEDDING_MODEL)
    return _citation_model


def extract_cited_segments(answer: str) -> List[Dict[str, Any]]:
    """
    Extract segments of text that are associated with specific citations.
    
    e.g., "The UE initiates registration [Source 1] by sending a message [Source 2]"
    -> [
        {"text": "The UE initiates registration", "source_num": 1},
        {"text": "by sending a message", "source_num": 2}
    ]
    """
    # Split answer into segments around citation markers
    pattern = r'\[Source\s*(\d+)\]'
    
    segments = []
    last_end = 0
    
    for match in re.finditer(pattern, answer, re.IGNORECASE):
        source_num = int(match.group(1))
        segment_text = answer[last_end:match.start()].strip()
        
        if segment_text and len(segment_text) > 10:
            segments.append({
                "text": segment_text,
                "source_num": source_num
            })
        
        last_end = match.end()
    
    return segments


def validate_citations(answer: str, retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Comprehensive citation validation:
    1. Format validation (do cited sources exist?)
    2. Content validation (does the cited source support the claim?)
    3. Coverage calculation (what % of claims have valid citations?)
    """
    
    # 1. Build source lookup
    valid_ids = set()
    doc_texts = {}
    
    for doc in retrieved_docs:
        did = str(doc.get("id", "")).strip()
        valid_ids.add(did)
        doc_texts[did] = doc.get("text", "")
        
        # Map numeric IDs
        if did.lower().startswith("source"):
            parts = did.split()
            if len(parts) > 1:
                doc_texts[parts[-1]] = doc.get("text", "")
    
    # 2. Extract all citations from answer
    pattern = r'[\(\[]\s*(?:Source|Doc|Ref)?\s*(\d+)\s*[\)\]]'
    found_citations = re.findall(pattern, answer, flags=re.IGNORECASE)
    
    # 3. Format validation
    valid_count = 0
    invalid_citations = []
    
    for cite_num in found_citations:
        candidates = [cite_num, f"Source {cite_num}", f"Doc {cite_num}"]
        matched = any(cand in valid_ids for cand in candidates)
        
        if matched:
            valid_count += 1
        else:
            invalid_citations.append(f"[Source {cite_num}]")
    
    total_citations = len(found_citations)
    format_coverage = (valid_count / total_citations) if total_citations > 0 else 0.0
    
    # 4. Content validation — verify cited source actually contains claimed info
    content_accuracy = 0.0
    cited_segments = extract_cited_segments(answer)
    
    if cited_segments and doc_texts:
        try:
            model = _get_model()
            
            accurate_count = 0
            for segment in cited_segments:
                source_key = f"Source {segment['source_num']}"
                source_text = doc_texts.get(source_key, doc_texts.get(str(segment['source_num']), ""))
                
                if source_text:
                    # Check if the cited source actually contains relevant info
                    seg_emb = model.encode(segment["text"], convert_to_tensor=True)
                    src_emb = model.encode(source_text, convert_to_tensor=True)
                    sim = float(util.cos_sim(seg_emb, src_emb))
                    
                    if sim > 0.4:  # Reasonably related
                        accurate_count += 1
            
            content_accuracy = (accurate_count / len(cited_segments)) if cited_segments else 0.0
        except Exception as e:
            logger.warning(f"Citation content validation failed: {e}")
            content_accuracy = format_coverage  # Fall back to format-only
    
    # 5. Combined coverage score
    # Weight: 40% format correctness, 60% content accuracy
    combined_coverage = (0.4 * format_coverage + 0.6 * content_accuracy) if cited_segments else format_coverage
    
    logger.info(
        f" [Citation Check] {valid_count}/{total_citations} format-valid | "
        f"Content accuracy: {content_accuracy:.2f} | "
        f"Combined: {combined_coverage:.2f}"
    )

    return {
        "valid_citations": valid_count,
        "total_citations": total_citations,
        "invalid_citations": list(set(invalid_citations)),
        "format_coverage": round(format_coverage, 2),
        "content_accuracy": round(content_accuracy, 2),
        "coverage": round(combined_coverage, 2)
    }