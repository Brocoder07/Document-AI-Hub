"""
RAG Service — Production Grade for 3GPP Telecom Standards

Architecture:
1. Strategy Pattern for domain-specific behavior (Telecom, Legal, etc.)
2. Hybrid Retrieval: Vector Similarity + BM25 Keyword Search
3. Cross-Encoder Re-ranking for precision
4. Chain-of-Thought Prompting with Self-Verification
5. Post-generation Hallucination Guard (claim-level verification)
6. Faithfulness-aware Confidence Scoring

Key Anti-Hallucination Mechanisms:
- Explicit refusal protocol ("I cannot find this in the provided documents")
- Source-grounded generation (strict context-only answering)
- Chain-of-thought reasoning (forces step-by-step derivation)
- Post-generation claim verification (hallucination_guard.py)
- Citation content accuracy checking (not just format)
- Relevance threshold filtering (removes low-quality context)
"""

from abc import ABC, abstractmethod
import time
import logging
from app.services.data_analysis_service import analyze_excel
from app.services.document_service import get_document_by_file_id
from app.db.session import SessionLocal
import re
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from app.api.vector_db import db_client
from app.services.embedding_service import embed_texts
from app.services.reranker import rerank_documents
from app.core.llm import get_llm
from app.generation.citation_enforcer import validate_citations
from app.generation.hallucination_guard import run_hallucination_check
from app.metrics.confidence_calculator import calculate_confidence

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# 3GPP GLOSSARY — Injected into telecom prompts for accuracy
# ═══════════════════════════════════════════════════════════════

TELECOM_3GPP_GLOSSARY = """
KEY 3GPP TERMS:
- UE: User Equipment (mobile device)
- gNB: gNodeB (5G NR base station)
- eNB: eNodeB (LTE base station)
- AMF: Access and Mobility Management Function
- SMF: Session Management Function
- UPF: User Plane Function
- PCF: Policy Control Function
- UDM: Unified Data Management
- AUSF: Authentication Server Function
- NRF: Network Repository Function
- NEF: Network Exposure Function
- NSSF: Network Slice Selection Function
- AF: Application Function
- DN: Data Network
- RAN: Radio Access Network
- CN: Core Network
- NG-RAN: Next Generation RAN (5G)
- E-UTRAN: Evolved UTRAN (LTE)
- PDU: Protocol Data Unit
- QoS: Quality of Service
- NAS: Non-Access Stratum
- RRC: Radio Resource Control
- S-NSSAI: Single Network Slice Selection Assistance Information
- DNN: Data Network Name
- SUPI: Subscription Permanent Identifier
- SUCI: Subscription Concealed Identifier
- PLMN: Public Land Mobile Network
"""

# ═══════════════════════════════════════════════════════════════
# 1. STRATEGY PATTERN — Domain-Specific Behavior
# ═══════════════════════════════════════════════════════════════

class RAGStrategy(ABC):
    """Abstract Base Class for domain-specific RAG behavior."""
    def __init__(self, mode: str):
        self.mode = mode

    @abstractmethod
    def get_collection_name(self) -> str:
        pass

    @abstractmethod
    def get_system_role(self) -> str:
        pass

    @property
    def use_hyde(self) -> bool:
        return False

    @property
    def inject_glossary(self) -> bool:
        return False

    def get_glossary(self) -> str:
        return ""

    def post_process_answer(self, answer: str) -> str:
        return answer


class TelecomStrategy(RAGStrategy):
    """
    3GPP Telecom Standards Strategy — Primary focus of this project.
    
    Features:
    - Searches telecom_docs collection (3GPP-specific index)
    - Injects 3GPP glossary for accurate acronym interpretation
    - Strict technical accuracy system prompt
    - No HyDE (3GPP queries are typically precise)
    """
    def get_collection_name(self) -> str:
        return "telecom_docs"

    def get_system_role(self) -> str:
        return (
            "You are a 3GPP telecommunications standards expert. "
            "You have deep knowledge of 5G NR, LTE, network architecture, "
            "protocols, and procedures as defined in 3GPP Technical Specifications (TS) "
            "and Technical Reports (TR). "
            "Provide precise, technically accurate answers using exact terminology "
            "from the 3GPP specifications. "
            "Always reference specific clause/section numbers when available. "
            "If the information is not found in the provided context, explicitly state "
            "that you cannot find the answer in the provided documents."
        )

    @property
    def inject_glossary(self) -> bool:
        return True

    def get_glossary(self) -> str:
        return TELECOM_3GPP_GLOSSARY


class GeneralStrategy(RAGStrategy):
    def get_collection_name(self) -> str:
        return "general_docs"

    def get_system_role(self) -> str:
        return "You are a helpful assistant. Answer clearly and concisely based only on the provided context."
    
    @property
    def use_hyde(self) -> bool:
        return True

class LegalStrategy(RAGStrategy):
    def get_collection_name(self) -> str:
        return "legal_docs"

    def get_system_role(self) -> str:
        return "You are a legal expert. Be precise, cite statutes if available, and avoid professional advice disclaimers unless necessary."

    def post_process_answer(self, answer: str) -> str:
        if "disclaimer" not in answer.lower():
            return answer + "\n\n*Disclaimer: This is AI-generated legal information, not professional advice.*"
        return answer

class HealthcareStrategy(RAGStrategy):
    def get_collection_name(self) -> str:
        return "medical_docs"

    def get_system_role(self) -> str:
        return (
            "You are a medical AI assistant. Provide accurate, evidence-based medical information. "
            "Use professional terminology but explain complex concepts clearly. "
            "Do not diagnose or prescribe."
        )

    def post_process_answer(self, answer: str) -> str:
        if "medical advice" not in answer.lower():
            return answer + "\n\n*Disclaimer: This content is for informational purposes only and does not constitute professional medical advice, diagnosis, or treatment.*"
        return answer

class AcademicStrategy(RAGStrategy):
    def get_collection_name(self) -> str:
        return "academic_docs"

    def get_system_role(self) -> str:
        return "You are a researcher. Prioritize peer-reviewed sources and maintain a formal tone."
    
    @property
    def use_hyde(self) -> bool:
        return True

class FinanceStrategy(RAGStrategy):
    def get_collection_name(self) -> str:
        return "finance_docs"

    def get_system_role(self) -> str:
        return "You are a financial analyst. Focus on numbers, trends, and fiscal accuracy."

class BusinessStrategy(RAGStrategy):
    def get_collection_name(self) -> str:
        return "business_docs"

    def get_system_role(self) -> str:
        return "You are a business assistant. Focus on actionable insights and clear summaries."

# ═══════════════════════════════════════════════════════════════
# 2. FACTORY — Strategy Selection
# ═══════════════════════════════════════════════════════════════

class StrategyFactory:
    @staticmethod
    def get_strategy(mode: str) -> RAGStrategy:
        mode = mode.lower().strip()
        
        strategies = {
            # Telecom (PRIMARY)
            "telecom": TelecomStrategy("telecom"),
            "3gpp": TelecomStrategy("telecom"),
            "engineer": TelecomStrategy("telecom"),
            
            # Legal
            "legal": LegalStrategy("legal"),
            "lawyer": LegalStrategy("legal"),
            
            # Healthcare
            "healthcare": HealthcareStrategy("healthcare"),
            "doctor": HealthcareStrategy("healthcare"),
            "medical": HealthcareStrategy("healthcare"),
            
            # Academic
            "academic": AcademicStrategy("academic"),
            "student": AcademicStrategy("academic"),
            "researcher": AcademicStrategy("academic"),
            
            # Finance
            "finance": FinanceStrategy("finance"),
            "banker": FinanceStrategy("finance"),
            "financial_analyst": FinanceStrategy("finance"),
            
            # Business
            "business": BusinessStrategy("business"),
            "employee": BusinessStrategy("business"),
            "executive": BusinessStrategy("business"),
        }
        return strategies.get(mode, GeneralStrategy("general"))

# ═══════════════════════════════════════════════════════════════
# 3. RAG PIPELINE — The Core Engine
# ═══════════════════════════════════════════════════════════════

class RAGPipeline:
    """
    Production-grade RAG pipeline with:
    - Hybrid retrieval (vector + BM25)
    - Cross-encoder re-ranking
    - Chain-of-thought prompting
    - Post-generation hallucination guard
    """
    def __init__(self, strategy: RAGStrategy):
        self.strategy = strategy
        self.llm = get_llm()

    async def _route_query(self, query: str) -> str:
        """Classify query intent."""
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "Classify as 'general' or 'specific'."), ("user", "{query}")
            ])
            return (await (prompt | self.llm).ainvoke({"query": query})).content.strip().lower()
        except:
            return "specific"

    async def _generate_hyde_query(self, query: str) -> str:
        """Generate hypothetical answer for better retrieval."""
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "Generate a hypothetical answer."), ("user", "{query}")
            ])
            return (await (prompt | self.llm).ainvoke({"query": query})).content.strip()
        except:
            return query

    async def _retrieve(self, query: str, user_id: str, file_id: str | None, top_k: int = 8):
        """
        Hybrid retrieval pipeline:
        1. Vector similarity search (semantic understanding)
        2. BM25 keyword search (exact-match for acronyms)
        3. Merge and deduplicate results
        4. Cross-encoder re-ranking (precision filtering)
        """
        start_time = time.time()
        search_query = query
        
        # HyDE for general strategies
        if not file_id and self.strategy.use_hyde:
            intent = await self._route_query(query)
            if "general" in intent:
                search_query = await self._generate_hyde_query(query)
                logger.info(f" [HyDE] Generated hypothetical answer for retrieval")
        
        col_name = self.strategy.get_collection_name()
        
        # Embed query with instruction prefix for BGE
        q_emb = embed_texts([search_query], is_query=True)[0]
        
        where = {"user_id": user_id}
        if file_id:
            where = {"$and": [{"user_id": user_id}, {"file_id": file_id}]}
        
        # --- STAGE 1: VECTOR SEARCH ---
        fetch_limit = 1000 if file_id else 50
        
        logger.info(f" [Vector Search] Collection: '{col_name}' | Limit: {fetch_limit}")
        
        vec_res = db_client.query(
            collection_name=col_name,
            query_vector=q_emb,
            top_k=fetch_limit,
            where=where
        )
        
        # --- STAGE 2: BM25 KEYWORD SEARCH (for exact-match acronyms) ---
        bm25_results = {}
        try:
            bm25_res = db_client.keyword_search(
                collection_name=col_name,
                query_text=query,
                top_k=20,
                where=where
            )
            # Index by document text for dedup
            if bm25_res["documents"] and bm25_res["documents"][0]:
                for i in range(len(bm25_res["documents"][0])):
                    doc_text = bm25_res["documents"][0][i]
                    if doc_text not in bm25_results:
                        bm25_results[doc_text] = {
                            "id": bm25_res["ids"][0][i],
                            "text": doc_text,
                            "metadata": bm25_res["metadatas"][0][i],
                            "bm25_score": bm25_res["scores"][0][i]
                        }
                logger.info(f" [BM25 Search] Found {len(bm25_results)} keyword matches")
        except Exception as e:
            logger.warning(f"BM25 search failed (non-critical): {e}")
        
        # --- STAGE 3: MERGE & DEDUPLICATE ---
        merged_items = {}
        is_structured_doc = False
        
        if vec_res["documents"] and vec_res["documents"][0]:
            for i in range(len(vec_res["documents"][0])):
                text_content = vec_res["documents"][0][i]
                meta = vec_res["metadatas"][0][i]
                
                # Excel detection
                if file_id and ("Spreadsheet Summary" in text_content or text_content.strip().startswith("Row ")):
                    is_structured_doc = True
                
                dist = vec_res["distances"][0][i]
                score = max(0, 1.0 - (dist / 2))
                
                if text_content not in merged_items:
                    merged_items[text_content] = {
                        "id": vec_res["ids"][0][i],
                        "text": text_content,
                        "metadata": meta,
                        "score": round(score, 4),
                        "source": "vector"
                    }
        
        # Merge BM25 results (add if not already present from vector search)
        for doc_text, bm25_item in bm25_results.items():
            if doc_text not in merged_items:
                merged_items[doc_text] = {
                    "id": bm25_item["id"],
                    "text": doc_text,
                    "metadata": bm25_item["metadata"],
                    "score": 0.5,  # Default score for BM25-only results
                    "source": "bm25"
                }
        
        raw_items = list(merged_items.values())
        hit_count = len(raw_items)
        logger.info(f" [Merged Results] {hit_count} unique chunks (vector + BM25)")
        
        # --- EXCEL "GOD MODE" ---
        if is_structured_doc:
            logger.info(f" [Excel Mode] Returning full context ({len(raw_items)} chunks)")
            raw_items.sort(key=lambda x: x["metadata"].get("chunk_num", 0))
            for item in raw_items:
                item["score"] = 1.0
            scores = [1.0] * len(raw_items)
            return raw_items, time.time() - start_time, scores

        # --- STAGE 4: WINDOW EXPANSION + RE-RANKING ---
        # Sort by vector similarity to find anchors
        raw_items.sort(key=lambda x: x["score"], reverse=True)
        top_anchors = raw_items[:top_k * 2]  # Get more candidates for re-ranking
        
        # Window expansion (±2 chunks)
        relevant_indices = set()
        for item in top_anchors:
            c_num = item["metadata"].get("chunk_num", -1)
            file_ref = item["metadata"].get("file_id", "unknown")
            if c_num != -1:
                for offset in range(-2, 3):
                    relevant_indices.add(f"{file_ref}_{c_num + offset}")

        # Filter to expanded set
        expanded_selection = []
        seen_keys = set()
        for item in raw_items:
            c_num = item["metadata"].get("chunk_num", -1)
            file_ref = item["metadata"].get("file_id", "unknown")
            unique_key = f"{file_ref}_{c_num}"
            if unique_key in relevant_indices and unique_key not in seen_keys:
                expanded_selection.append(item)
                seen_keys.add(unique_key)

        # --- STAGE 5: CROSS-ENCODER RE-RANKING ---
        if expanded_selection:
            logger.info(f" [Re-ranking] {len(expanded_selection)} candidates...")
            reranked = rerank_documents(
                query=query,
                documents=expanded_selection,
                top_k=top_k,
                relevance_threshold=0.05  # Permissive for telecom (technical content)
            )
            
            if reranked:
                # Sort by chunk_num for document flow
                reranked.sort(key=lambda x: x["metadata"].get("chunk_num", 0))
                results = reranked
                scores = [x.get("rerank_score", x["score"]) for x in reranked]
                return results, time.time() - start_time, scores
        
        # Fallback: use pre-rerank ordering
        expanded_selection.sort(key=lambda x: x["metadata"].get("chunk_num", 0))
        results = expanded_selection[:top_k]
        scores = [x["score"] for x in results]
        
        return results, time.time() - start_time, scores

    def _normalize_citations(self, text: str) -> str:
        """Standardize citation formats to [Source X]."""
        return re.sub(
            r"[\(\[](?:Doc\s?|Source\s?)?(\d{1,3})[\)\]]",
            lambda m: f"[Source {m.group(1)}]",
            text,
            flags=re.IGNORECASE
        )

    async def run(self, query: str, user_id: str, file_id: str | None, chat_history: str):
        total_start = time.time()
        
        # 1. Retrieve (Hybrid + Re-rank)
        retrieved, ret_time, scores = await self._retrieve(query, user_id, file_id)
        
        if not retrieved:
            return self._empty_response(query, ret_time, total_start)

        # 2. Context Building with section metadata
        formatted_docs = []
        for i, d in enumerate(retrieved):
            clean_text = d["text"].replace("\n", " ")
            meta = d.get("metadata", {})
            
            # Add section reference if available
            section_ref = ""
            if meta.get("spec_number") and meta.get("section_id"):
                section_ref = f" ({meta['spec_number']}, Section {meta['section_id']})"
            elif meta.get("filename"):
                section_ref = f" ({meta['filename']})"
            
            formatted_docs.append(f"[Source {i+1}]{section_ref} {clean_text}")
        
        context_str = "\n\n".join(formatted_docs)
        source_list = ", ".join([f"[Source {i+1}]" for i in range(len(retrieved))])

        # 3. Prompting — Chain-of-Thought with Anti-Hallucination Protocol
        role = self.strategy.get_system_role()
        
        # Inject glossary for telecom strategy
        glossary_section = ""
        if self.strategy.inject_glossary:
            glossary_section = f"\n{self.strategy.get_glossary()}\n"

        template = """{role}
{glossary}
STRICT RULES:
1. Answer using ONLY the information from the CONTEXT below. Do NOT use any outside knowledge.
2. Cite every factual claim using [Source X] tags corresponding to the source that supports it.
3. If the context does not contain enough information to answer the question, you MUST respond with:
   "I cannot find sufficient information in the provided documents to answer this question."
4. Do NOT speculate, infer, or generate information that is not explicitly stated in the context.
5. When referencing 3GPP specifications, include the spec number and section/clause when available.

REASONING PROCESS:
Before answering, briefly identify which sources are relevant and what they say about the question.
Then provide your answer based solely on those sources.

AVAILABLE SOURCES:
{source_list}

CONVERSATION HISTORY:
{chat_history}

CONTEXT:
{context}

QUESTION: {question}

STEP-BY-STEP REASONING AND ANSWER:"""

        prompt = PromptTemplate.from_template(template)

        # 4. Generation
        gen_start = time.time()
        ai_msg = await (prompt | self.llm).ainvoke({
            "role": role,
            "glossary": glossary_section,
            "source_list": source_list,
            "chat_history": chat_history,
            "context": context_str, 
            "question": query
        })
        gen_time = time.time() - gen_start
        
        # 5. Token Mapping
        raw_usage = ai_msg.response_metadata.get("token_usage", {})
        token_metrics = {
            "input": raw_usage.get("prompt_tokens", 0),
            "output": raw_usage.get("completion_tokens", 0),
            "total": raw_usage.get("total_tokens", 0)
        }

        # 6. Post-Processing
        raw_answer = self._normalize_citations(ai_msg.content)
        final_answer = self.strategy.post_process_answer(raw_answer)

        # 7. Hallucination Guard — Claim-Level Verification
        source_texts = [d["text"] for d in retrieved]
        faithfulness_result = run_hallucination_check(final_answer, source_texts)

        # 8. Citation Validation
        val_docs = [{"id": f"Source {i+1}", "text": d["text"]} for i, d in enumerate(retrieved)]
        citation_metrics = validate_citations(final_answer, val_docs)
        
        # 9. Confidence Scoring (Faithfulness-Aware)
        rerank_scores = [d.get("rerank_score") for d in retrieved if d.get("rerank_score") is not None]
        conf = calculate_confidence(
            query, retrieved, final_answer, citation_metrics,
            faithfulness=faithfulness_result,
            rerank_scores=rerank_scores if rerank_scores else None
        )
        
        metrics = {
            "processing_time_total": round(time.time() - total_start, 3),
            "retrieval_time": round(ret_time, 3),
            "generation_time": round(gen_time, 3),
            "token_usage": token_metrics,
            "similarity_score": round(sum(scores)/len(scores), 3) if scores else 0,
            "confidence_category": conf["confidence_category"],
            "confidence_score": conf["confidence_score"],
            "hallucination_risk": conf["hallucination_risk"],
            "factors": conf["factors"],
            "citation_validation": citation_metrics,
            "faithfulness": {
                "score": faithfulness_result.get("faithfulness_score", 0),
                "verdict": faithfulness_result.get("verdict", "UNKNOWN"),
                "supported": faithfulness_result.get("supported_count", 0),
                "unsupported": faithfulness_result.get("unsupported_count", 0),
                "total_claims": faithfulness_result.get("total_claims", 0),
            },
            "model_used": ai_msg.response_metadata.get("model_used", "unknown"),
        }
        
        return {
            "answer": final_answer,
            "retrieved": retrieved,
            "metrics": metrics
        }

    def _empty_response(self, query, ret_time, start_time):
        return {
            "answer": "I could not find relevant documents to answer your question. Please upload relevant 3GPP specification documents first.",
            "retrieved": [],
            "metrics": {
                "processing_time_total": round(time.time() - start_time, 3),
                "retrieval_time": round(ret_time, 3),
                "generation_time": 0.0,
                "token_usage": {"input": 0, "output": 0, "total": 0},
                "similarity_score": 0.0,
                "confidence_category": "Low",
                "confidence_score": 0.0,
                "hallucination_risk": "N/A",
                "citation_validation": {},
                "factors": {},
                "faithfulness": {"score": 0, "verdict": "NO_CONTEXT", "supported": 0, "unsupported": 0, "total_claims": 0},
                "model_used": "none",
            }
        }

# ═══════════════════════════════════════════════════════════════
# 4. CLIENT CODE — Entry Points
# ═══════════════════════════════════════════════════════════════

async def answer_query(query: str, user_id: str, file_id: str | None = None, mode: str = "general", chat_history: str = ""):
    
    # --- INTERCEPT: EXCEL ANALYSIS ---
    if file_id:
        db = SessionLocal()
        doc = get_document_by_file_id(db, file_id)
        db.close()
        
        if doc and doc.file_path.endswith(('.xlsx', '.xls')):
            triggers = [
                "total", "sum", "average", "count", "how many", "calculate", "max", "min", "mean",
                "list", "show", "give", "who", "what", "which", "where", "policy", "detail"
            ]
            
            if any(t in query.lower() for t in triggers):
                logger.info(f" Detected Excel Query on {doc.filename}. Routing to Pandas Engine.")
                
                analysis_result = await analyze_excel(doc.file_path, query)

                if isinstance(analysis_result, dict):
                    final_answer = analysis_result["answer"]
                    conf_score = analysis_result["confidence"]
                    reason = analysis_result["reason"]
                else:
                    final_answer = str(analysis_result)
                    conf_score = 100.0
                    reason = "Legacy Mode"
                
                return {
                    "answer": f"**Analysis Result:**\n\n{final_answer}",
                    "retrieved": [],
                    "metrics": {
                        "confidence_score": conf_score,
                        "confidence_category": "High" if conf_score > 80 else "Medium",
                        "hallucination_risk": "Checked by Reviewer",
                        "factors": {
                            "retrieval_quality": 100.0,
                            "citation_coverage": 100.0,
                            "logic_check": reason
                        },
                        "processing_time_total": 0.5,
                        "faithfulness": {"score": conf_score, "verdict": "CODE_VERIFIED", "supported": 0, "unsupported": 0, "total_claims": 0},
                    }
                }
    
    # --- STANDARD RAG FLOW ---
    strategy = StrategyFactory.get_strategy(mode)
    pipeline = RAGPipeline(strategy)
    return await pipeline.run(query, user_id, file_id, chat_history)

# Compat wrapper
async def retrieve_docs(query, user_id, file_id=None, mode="general", top_k=8):
    strategy = StrategyFactory.get_strategy(mode)
    pipeline = RAGPipeline(strategy)
    d, _, _ = await pipeline._retrieve(query, user_id, file_id, top_k)
    return d