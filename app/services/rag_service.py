from abc import ABC, abstractmethod
import time
import logging
import re
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from app.api.vector_db import db_client
from app.services.embedding_service import embed_texts
from app.core.llm import get_llm
from app.generation.citation_enforcer import validate_citations
from app.metrics.confidence_calculator import calculate_confidence

logger = logging.getLogger(__name__)

# --- 1. ABSTRACTION: The Strategy Interface ---

class RAGStrategy(ABC):
    """
    Abstract Base Class that defines the behavior for different RAG modes.
    Encapsulates domain-specific logic like prompts, collections, and rules.
    """
    def __init__(self, mode: str):
        self.mode = mode

    @abstractmethod
    def get_collection_name(self) -> str:
        """Returns the specific Vector DB collection/class to search."""
        pass

    @abstractmethod
    def get_system_role(self) -> str:
        """Returns the persona/system prompt for the LLM."""
        pass

    @property
    def use_hyde(self) -> bool:
        """Determines if HyDE (Hypothetical Document Embeddings) should be used."""
        return False

    def post_process_answer(self, answer: str) -> str:
        """Hook for any domain-specific answer cleanup."""
        return answer

# --- 2. INHERITANCE: Concrete Strategies ---

class GeneralStrategy(RAGStrategy):
    def get_collection_name(self) -> str:
        return "general_docs"

    def get_system_role(self) -> str:
        return "You are a helpful assistant. Answer clearly and concisely."
    
    @property
    def use_hyde(self) -> bool:
        return True  # General queries often benefit from HyDE

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
    """
    NEW: Strategy for Doctors and Medical Professionals.
    """
    def get_collection_name(self) -> str:
        return "medical_docs"

    def get_system_role(self) -> str:
        return (
            "You are a medical AI assistant. Provide accurate, evidence-based medical information. "
            "Use professional terminology but explain complex concepts clearly. "
            "Do not diagnose or prescribe."
        )

    def post_process_answer(self, answer: str) -> str:
        # Mandatory Medical Disclaimer
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

# --- 3. FACTORY PATTERN: Object Creation ---

class StrategyFactory:
    """
    Factory class to instantiate the correct strategy based on user role or mode.
    """
    @staticmethod
    def get_strategy(mode: str) -> RAGStrategy:
        mode = mode.lower().strip()
        
        strategies = {
            # Legal
            "legal": LegalStrategy("legal"),
            "lawyer": LegalStrategy("legal"),
            
            # Healthcare (NEW)
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
        # Default fallback
        return strategies.get(mode, GeneralStrategy("general"))

# --- 4. ENCAPSULATION: The Context / Pipeline ---

class RAGPipeline:
    """
    The Context class. It orchestrates the RAG flow but delegates 
    specific behaviors to the injected Strategy object.
    """
    def __init__(self, strategy: RAGStrategy):
        self.strategy = strategy
        self.llm = get_llm()

    async def _route_query(self, query: str) -> str:
        """Helper to classify query intent."""
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "Classify as 'general' or 'specific'."), ("user", "{query}")
            ])
            return (await (prompt | self.llm).ainvoke({"query": query})).content.strip().lower()
        except:
            return "specific"

    async def _generate_hyde_query(self, query: str) -> str:
        """Helper to generate hypothetical answer for better retrieval."""
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "Generate a hypothetical answer."), ("user", "{query}")
            ])
            return (await (prompt | self.llm).ainvoke({"query": query})).content.strip()
        except:
            return query

    async def _retrieve(self, query: str, user_id: str, file_id: str | None, top_k: int = 6):
        start_time = time.time()
        search_query = query
        
        # Strategy Hook: Check if this mode uses HyDE
        if not file_id and self.strategy.use_hyde:
            if "general" in await self._route_query(query):
                search_query = await self._generate_hyde_query(query)
        
        # Polymorphism: Get collection name from strategy
        col_name = self.strategy.get_collection_name()
        
        # Embed Query
        q_emb = embed_texts([search_query])[0]
        
        # Construct Filter
        where = {"user_id": user_id}
        if file_id:
            where = {"$and": [{"user_id": user_id}, {"file_id": file_id}]}
        
        logger.info(f"🕵️ [RAG Search] Query: '{query}'")
        logger.info(f"🎯 [Target] Collection: '{col_name}' | User Filter: {where}")

        # Execute Query via Weaviate Adapter
        res = db_client.query(
            collection_name=col_name,
            query_vector=q_emb,
            top_k=top_k,
            where=where
        )
        
        # --- RESULT LOGGING ---
        hit_count = len(res["documents"][0]) if res["documents"] else 0
        logger.info(f"🔢 [Results] Found {hit_count} raw matches in Weaviate.")
        
        if hit_count > 0:
            # Inspect the first match metadata
            first_meta = res["metadatas"][0][0]
            logger.info(f"🥇 [Top Hit] ID: {first_meta.get('file_id')} | User: {first_meta.get('user_id')}")
        else:
            logger.warning("⚠️ [No Hits] Weaviate returned 0 results. Check User ID match.")

        results = []
        scores = []
        # Standardize results from Adapter format
        if res["documents"] and res["documents"][0]:
            for i in range(len(res["documents"][0])):
                # Calculate Score (Distance to Similarity)
                # Weaviate usually returns distance (lower is better). 
                # Basic inversion: score = 1 / (1 + distance) or similar.
                # Assuming Adapter returns standardized cosine distance [0, 2]:
                dist = res["distances"][0][i]
                score = max(0, 1.0 - (dist / 2)) # Approximate normalization
                
                scores.append(score)
                results.append({
                    "id": res["ids"][0][i],
                    "text": res["documents"][0][i],
                    "metadata": res["metadatas"][0][i],
                    "score": round(score, 4)
                })
        
        return results, time.time() - start_time, scores

    def _normalize_citations(self, text: str) -> str:
        """
        Robustly standardizes citations.
        Converts: (Source 1), [Source 1], (1), [1] -> [Source 1]
        Ignores years like (2025) by restricting digits to 1-3.
        """
        return re.sub(
            # The Match:
            # 1. [\(\[]       -> Starts with '(' or '['
            # 2. (?:...)?     -> Optional "Doc" or "Source" prefix
            # 3. (\d{1,3})    -> Capture 1 to 3 digits ONLY. (Stops matching 2025)
            # 4. [\)\]]       -> Ends with ')' or ']'
            r"[\(\[](?:Doc\s?|Source\s?)?(\d{1,3})[\)\]]",
            lambda m: f"[Source {m.group(1)}]",
            text,
            flags=re.IGNORECASE
        )

    async def run(self, query: str, user_id: str, file_id: str | None, chat_history: str):
        total_start = time.time()
        
        # 1. Retrieve
        retrieved, ret_time, scores = await self._retrieve(query, user_id, file_id)
        
        if not retrieved:
            return self._empty_response(query, ret_time, total_start)

        # 2. Context Building
        formatted_docs = []
        for i, d in enumerate(retrieved):
            clean_text = d["text"].replace("\n", " ")
            formatted_docs.append(f"[Source {i+1}] {clean_text}")
        context_str = "\n\n".join(formatted_docs)
        source_list = ", ".join([f"[Source {i+1}]" for i in range(len(retrieved))])

        # 3. Prompting (Strategy Hook)
        role = self.strategy.get_system_role()
        
        prompt_str = f"""{role}
INSTRUCTIONS:
1. Answer using ONLY the context.
2. Cite every claim as [Source X].
3. If unsure, say "I cannot find the answer."

AVAILABLE SOURCES:
{source_list}

HISTORY:
{chat_history}

CONTEXT:
{context_str}

QUESTION: {query}
ANSWER:"""

        # 4. Generation
        gen_start = time.time()
        ai_msg = await (PromptTemplate.from_template(prompt_str) | self.llm).ainvoke({})
        gen_time = time.time() - gen_start
        
        # 5. Token Mapping
        raw_usage = ai_msg.response_metadata.get("token_usage", {})
        token_metrics = {
            "input": raw_usage.get("prompt_tokens", 0),
            "output": raw_usage.get("completion_tokens", 0),
            "total": raw_usage.get("total_tokens", 0)
        }

        # 6. Post-Processing (Normalization + Strategy Hook)
        raw_answer = self._normalize_citations(ai_msg.content)
        final_answer = self.strategy.post_process_answer(raw_answer)

        # 7. Validation & Metrics
        val_docs = [{"id": f"Source {i+1}", "text": d["text"]} for i, d in enumerate(retrieved)]
        citation_metrics = validate_citations(final_answer, val_docs)
        conf = calculate_confidence(query, retrieved, final_answer, citation_metrics)
        
        metrics = {
            "processing_time_total": round(time.time() - total_start, 3),
            "retrieval_time": round(ret_time, 3),
            "generation_time": round(gen_time, 3),
            "token_usage": token_metrics,
            "similarity_score": round(sum(scores)/len(scores), 3) if scores else 0,
            "confidence_category": conf["confidence_category"],
            "confidence_score": conf["confidence_score"],
            "hallucination_risk": "Low" if citation_metrics.get("coverage", 0) > 0.6 else "Potential",
            "citation_validation": citation_metrics
        }
        
        return {
            "answer": final_answer,
            "retrieved": retrieved,
            "metrics": metrics
        }

    def _empty_response(self, query, ret_time, start_time):
        return {
            "answer": "I could not find relevant documents.",
            "retrieved": [],
            "metrics": {
                "processing_time_total": round(time.time() - start_time, 3),
                "retrieval_time": round(ret_time, 3),
                "generation_time": 0.0,
                "token_usage": {"input": 0, "output": 0, "total": 0},
                "similarity_score": 0.0,
                "confidence_category": "Low",
                "confidence_score": 0.0,
                "hallucination_risk": "High",
                "citation_validation": {},
                "evaluation": {}
            }
        }

# --- 5. CLIENT CODE: Main Entry Point ---

async def answer_query(query: str, user_id: str, file_id: str | None = None, mode: str = "general", chat_history: str = ""):
    """
    Main entry point used by the API.
    It uses the Factory to get the correct Strategy and runs the Pipeline.
    """
    # 1. Get the correct Strategy based on the mode (role)
    strategy = StrategyFactory.get_strategy(mode)
    
    # 2. Initialize the Pipeline with that strategy
    pipeline = RAGPipeline(strategy)
    
    # 3. Execute
    return await pipeline.run(query, user_id, file_id, chat_history)

# Compat wrapper for other services
async def retrieve_docs(query, user_id, file_id=None, mode="general", top_k=6):
    strategy = StrategyFactory.get_strategy(mode)
    pipeline = RAGPipeline(strategy)
    d, _, _ = await pipeline._retrieve(query, user_id, file_id, top_k)
    return d