import logging
import numpy as np
from typing import List, Dict, Any
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
from app.core.config import settings

logger = logging.getLogger(__name__)

class RAGEvaluator:
    """
    Evaluates RAG performance by calculating:
    1. Grounding Confidence: Semantic similarity between Answer and Context.
    2. Retrieval Quality: Average similarity score of retrieved chunks.
    """
    
    def __init__(self):
        self.performance_thresholds = {
            "min_similarity": 0.35,
            "min_grounding": 0.4,
            "max_retrieval_time": 4.0,
        }
        
        try:
            # Reuse the model defined in your settings
            self.grounding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
            logger.info(f"✅ Evaluator loaded model: {settings.EMBEDDING_MODEL}")
        except Exception as e:
            logger.error(f"❌ Failed to load grounding model: {e}")
            self.grounding_model = None

    def _calculate_grounding_confidence(self, answer: str, context: str) -> float:
        """
        Measures how well the answer is supported by the context.
        Low scores indicate potential hallucinations.
        """
        if not self.grounding_model or not answer or not context:
            return 0.0
        
        try:
            # Compute embeddings
            answer_emb = self.grounding_model.encode(answer, convert_to_tensor=True)
            context_emb = self.grounding_model.encode(context, convert_to_tensor=True)
            
            # Calculate cosine similarity
            similarity = util.cos_sim(answer_emb, context_emb).item()
            return max(0.0, min(1.0, similarity))
        except Exception as e:
            logger.warning(f"Failed to calculate grounding confidence: {e}")
            return 0.0

    def evaluate_query(self, 
                      question: str, 
                      answer: str,
                      context: str,
                      retrieved_docs: List[Dict], 
                      response_time: float
                      ) -> Dict[str, Any]:
        
        # 1. Calculate Retrieval Quality (Avg chunk score)
        avg_retrieval_score = 0.0
        if retrieved_docs:
            scores = [d.get("score", 0) for d in retrieved_docs]
            avg_retrieval_score = np.mean(scores)

        # 2. Calculate Grounding Confidence
        grounding_conf = self._calculate_grounding_confidence(answer, context)
        
        # 3. Detect Hallucination Risk
        hallucination_detected = grounding_conf < self.performance_thresholds["min_grounding"]

        metrics = {
            "avg_retrieval_score": round(avg_retrieval_score, 3),
            "grounding_confidence": round(grounding_conf, 3),
            "hallucination_warning": hallucination_detected,
            "response_time": round(response_time, 3),
            "timestamp": datetime.now().isoformat()
        }
        
        return metrics

# Global instance
evaluator = RAGEvaluator()