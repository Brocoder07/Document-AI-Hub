"""
Consolidated query handler: retrieval -> enforcement -> generation -> confidence.
No separate files for mode enforcement or retrieval debug (inlined for simplicity).
"""
import logging
from typing import Dict, Any, List
import re

logger = logging.getLogger("query_handler")

# ===== Query Expansion (inline helper) =====
def _expand_query_for_academic(query: str) -> List[str]:
	"""For academic mode: expand vague queries to sub-queries."""
	triggers = {
		"explain in detail": ["definition", "key concepts", "examples", "process", "components"],
		"what is": ["definition", "characteristics"],
		"how does": ["mechanism", "process", "steps"],
	}
	expanded = [query]
	for trigger, aspects in triggers.items():
		if trigger.lower() in query.lower():
			topic = query.replace(trigger, "").strip("?. ")
			for aspect in aspects:
				expanded.append(f"{aspect} of {topic}")
			break
	return expanded

# ===== Retrieval with Expansion =====
def retrieve_documents(query: str, file_id: str, mode: str, retrieval_fn, embed_fn, top_k: int = 5) -> List[dict]:
	"""Retrieve documents; expand query for academic mode."""
	if mode == "academic":
		queries = _expand_query_for_academic(query)
		all_results = {}
		for q in queries:
			q_emb = embed_fn(q)
			results = retrieval_fn(q_emb, top_k=top_k, filters={"file_id": file_id})
			for doc in results:
				doc_id = doc["id"]
				if doc_id not in all_results or doc["score"] > all_results[doc_id]["score"]:
					all_results[doc_id] = doc
		return sorted(all_results.values(), key=lambda d: d["score"], reverse=True)[:top_k]
	else:
		# general mode: single retrieval
		q_emb = embed_fn(query)
		return retrieval_fn(q_emb, top_k=top_k, filters={"file_id": file_id})

# ===== Mode Enforcement (inline) =====
def enforce_mode_policy(mode: str, retrieved: List[Dict[str, Any]]) -> Dict[str, Any]:
	"""Enforce per-mode policy: both academic and general require documents."""
	if mode not in ("academic", "general"):
		return {"allowed": False, "reason": "Invalid mode."}
	if not retrieved:
		return {"allowed": False, "reason": "No supporting documents found."}
	return {"allowed": True, "reason": "Evidence available."}

# ===== Citation Enforcement =====
def build_citation_prompt(query: str, retrieved_docs: List[Dict[str, Any]]) -> str:
	"""Build prompt that forces citations."""
	doc_context = "\n".join([
		f"[DOC {doc['id']}] {doc['text'][:300]}...\n"
		for doc in retrieved_docs
	])
	return f"""Answer ONLY using provided documents. For every factual claim, cite [DOC <id>].

Query: {query}

Documents:
{doc_context}

Answer:"""

def validate_citations(answer: str, retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
	"""Validate citations in answer."""
	cited_ids = set(re.findall(r'\[DOC\s+([^\]]+)\]', answer))
	available_ids = {doc["id"] for doc in retrieved_docs}
	valid_count = len(cited_ids & available_ids)
	total_count = len(cited_ids)
	coverage = valid_count / total_count if total_count > 0 else 0.0
	return {
		"valid_citations": valid_count,
		"total_citations": total_count,
		"coverage": coverage,
	}

# ===== Confidence Calculation =====
def calculate_confidence(query: str, retrieved_docs: List[Dict[str, Any]], answer: str, citation_val: Dict[str, Any]) -> Dict[str, Any]:
	"""Multi-factor confidence score."""
	avg_sim = sum(d.get("score", 0) for d in retrieved_docs) / len(retrieved_docs) if retrieved_docs else 0
	retrieval_score = min(avg_sim * 100, 100)
	citation_score = citation_val.get("coverage", 0) * 100
	query_coverage = len(set(query.lower().split()) & set(answer.lower().split())) / len(set(query.lower().split())) if query else 0
	query_score = query_coverage * 100
	depth_score = min((len(answer.split()) / 500) * 100, 100)
	
	confidence_score = 0.3 * retrieval_score + 0.3 * citation_score + 0.2 * query_score + 0.2 * depth_score
	confidence_category = "High" if confidence_score >= 75 else ("Medium" if confidence_score >= 50 else "Low")
	
	return {
		"confidence_score": round(confidence_score, 1),
		"confidence_category": confidence_category,
		"factors": {
			"retrieval": round(retrieval_score, 1),
			"citations": round(citation_score, 1),
			"query_coverage": round(query_score, 1),
			"depth": round(depth_score, 1),
		}
	}

# ===== Main Handler =====
def handle_query(query: str, file_id: str, mode: str, retrieval_fn, embed_fn, generator_fn, top_k: int = 5) -> Dict[str, Any]:
	"""Main orchestrator."""
	# 1) Retrieve
	retrieved = retrieve_documents(query, file_id, mode, retrieval_fn, embed_fn, top_k)
	
	# 2) Enforce policy
	policy = enforce_mode_policy(mode, retrieved)
	if not policy["allowed"]:
		return {
			"answer": "I could not find the answer in the provided documents.",
			"retrieved": [],
			"similarity_score": 0,
			"confidence_score": 0,
			"confidence_category": "Low",
		}
	
	# 3) Generate with citations
	prompt = build_citation_prompt(query, retrieved)
	answer = generator_fn(prompt)
	
	# 4) Validate and score
	citation_val = validate_citations(answer, retrieved)
	conf = calculate_confidence(query, retrieved, answer, citation_val)
	
	return {
		"answer": answer,
		"retrieved": retrieved,
		"similarity_score": max((r.get("score", 0) for r in retrieved), default=0),
		"confidence_score": conf["confidence_score"],
		"confidence_category": conf["confidence_category"],
		"hallucination_risk": "Low" if citation_val["coverage"] > 0.8 else "Medium",
		"metrics": conf["factors"],
	}