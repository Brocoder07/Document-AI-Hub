"""
Citation enforcement: prompt generator to cite chunk IDs, then validate citations.
"""
from typing import List, Dict, Any
import re
import logging

logger = logging.getLogger(__name__)

def build_citation_prompt(query: str, retrieved_docs: List[Dict[str, Any]]) -> str:
	"""
	Builds a prompt that forces the generator to cite chunk IDs for every factual claim.
	"""
	doc_context = "\n".join([
		f"[DOC {doc['id']}] {doc['text'][:300]}...\n"
		for doc in retrieved_docs
	])
	
	prompt = f"""You are an academic assistant. Answer the query ONLY using the provided documents.

Query: {query}

Documents:
{doc_context}

Instructions:
1. Answer the query thoroughly and in detail.
2. For EVERY factual claim, cite the document ID in brackets, e.g., "[DOC 7a0e43c6-4e1d-44c2-9fed-447bf1981204_23]"
3. Do NOT use external knowledge. Only cite the provided documents.
4. Use clear section headings (e.g., ### Definition, ### Key Components, ### Process).
5. Ensure citations match actual content in the documents.

Answer:"""
	return prompt

def validate_citations(answer: str, retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
	"""
	Post-generation validation: extract citations and check they exist in retrieved_docs.
	Returns: {'valid_citations': int, 'total_citations': int, 'coverage': float}
	"""
	# Extract cited doc IDs: [DOC <id>]
	# Updated regex to be more flexible with spacing
	cited_ids = set(re.findall(r'\[DOC\s+([^\]]+)\]', answer))
	
	logger.debug(f"Cited IDs extracted from answer: {cited_ids}")
	
	available_ids = {doc["id"] for doc in retrieved_docs}
	logger.debug(f"Available doc IDs from retrieval: {available_ids}")
	
	valid_cites = cited_ids & available_ids
	invalid_cites = cited_ids - available_ids
	
	valid_count = len(valid_cites)
	total_count = len(cited_ids)
	coverage = valid_count / total_count if total_count > 0 else 0.0
	
	logger.info(f"Citation validation - Total: {total_count}, Valid: {valid_count}, Coverage: {coverage:.2%}")
	if invalid_cites:
		logger.warning(f"Invalid citations (doc IDs not found): {invalid_cites}")
	
	return {
		"valid_citations": valid_count,
		"total_citations": total_count,
		"invalid_citations": list(invalid_cites),
		"coverage": coverage,
	}
