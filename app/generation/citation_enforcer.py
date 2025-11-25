import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def validate_citations(answer: str, retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validates that citations in the answer correspond to the provided documents.
    Supports formats: [Source 1], [Doc 1], [1], (Source 1), etc.
    """
    
    # 1. Normalize IDs in Retrieved Docs
    # We create a lookup set. If docs passed in have IDs like "Source 1", we store "Source 1".
    # If they use UUIDs, we store UUIDs.
    valid_ids = set()
    doc_id_map = {} # Map "1" -> "Source 1" if needed
    
    for doc in retrieved_docs:
        did = str(doc.get("id", "")).strip()
        valid_ids.add(did)
        
        # Heuristic: If ID is "Source 1", map "1" to it for easy lookup
        if did.lower().startswith("source"):
            parts = did.split()
            if len(parts) > 1:
                doc_id_map[parts[-1]] = did
        elif did.isdigit():
             doc_id_map[did] = did

    # 2. Extract Citations from Answer using Robust Regex
    # Matches: [Source 1], [Doc 1], [1], (Source 1), (1)
    # Group 1: Prefix (Source/Doc)
    # Group 2: The ID/Number
    pattern = r"[\(\[]\s*(?:Source|Doc|Ref)?\s*(\d+)\s*[\)\]]"
    
    # We also want to support full string IDs if they are used (like [DOC-123])
    # But for this specific "Source X" system, the digit extractor is best.
    
    found_citations = re.findall(pattern, answer, flags=re.IGNORECASE)
    
    valid_count = 0
    invalid_citations = []
    
    # 3. Validate Logic
    for cite_num in found_citations:
        # Construct possible keys to look up
        # 1. Direct number ("1")
        # 2. "Source {num}" ("Source 1")
        
        candidates = [cite_num, f"Source {cite_num}", f"Doc {cite_num}"]
        
        matched = False
        for cand in candidates:
            if cand in valid_ids:
                matched = True
                break
        
        if matched:
            valid_count += 1
        else:
            invalid_citations.append(f"[Source {cite_num}]")

    total_citations = len(found_citations)
    
    # 4. Coverage Calculation
    # Simple metric: % of citations that are valid
    coverage = (valid_count / total_citations) if total_citations > 0 else 0.0

    logger.info(f"Citation Check: {valid_count}/{total_citations} valid. IDs found: {found_citations}")

    return {
        "valid_citations": valid_count,
        "total_citations": total_citations,
        "invalid_citations": list(set(invalid_citations)),
        "coverage": round(coverage, 2)
    }