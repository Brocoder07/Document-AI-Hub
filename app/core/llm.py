from langchain.chat_models.base import BaseChatModel
from langchain.schema import (
    AIMessage, 
    ChatGeneration, 
    ChatResult, 
    HumanMessage, 
    SystemMessage, 
    BaseMessage
)
from typing import Any, List, Optional, Dict
from app.core.config import settings
from groq import Groq as GroqClient
import time
import logging

logger = logging.getLogger(__name__)

class CustomGroqLLM(BaseChatModel):
    """
    Production-grade ChatModel wrapper for Groq API.
    Features:
    - Low temperature (0.1) for deterministic, faithful outputs
    - Exponential backoff retry for rate limit handling (free tier)
    - Automatic fallback to smaller model on persistent failures
    - Nucleus sampling (top_p=0.9) to prevent wild generation
    """
    model: str = settings.GROQ_MODEL
    fallback_model: str = settings.GROQ_MODEL_FALLBACK
    groq_api_key: str = settings.GROQ_API_KEY
    temperature: float = 0.1  # Lower = more deterministic, less hallucination
    top_p: float = 0.9        # Nucleus sampling for controlled generation
    max_retries: int = 3

    @property
    def _llm_type(self) -> str:
        return "custom_groq_chat"

    def _call_groq(self, groq_messages: list, model: str, stop: Optional[List[str]] = None):
        """Makes the actual Groq API call with retry logic."""
        client = GroqClient(api_key=self.groq_api_key)
        
        for attempt in range(self.max_retries):
            try:
                response = client.chat.completions.create(
                    messages=groq_messages,
                    model=model,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    stop=stop
                )
                return response
            except Exception as e:
                error_str = str(e).lower()
                
                # Rate limit hit — wait and retry with exponential backoff
                if "rate_limit" in error_str or "429" in error_str or "too many" in error_str:
                    wait_time = (2 ** attempt) * 5  # 5s, 10s, 20s
                    logger.warning(
                        f" [Rate Limit] Attempt {attempt+1}/{self.max_retries}. "
                        f"Waiting {wait_time}s before retry..."
                    )
                    time.sleep(wait_time)
                    continue
                
                # Model overloaded — try fallback
                if "overloaded" in error_str or "503" in error_str:
                    if model != self.fallback_model:
                        logger.warning(
                            f" [Fallback] Primary model '{model}' overloaded. "
                            f"Switching to '{self.fallback_model}'..."
                        )
                        return self._call_groq(groq_messages, self.fallback_model, stop)
                    raise
                
                # Unknown error — re-raise
                raise
        
        # All retries exhausted — try fallback model as last resort
        if model != self.fallback_model:
            logger.warning(
                f" [Fallback] All retries exhausted for '{model}'. "
                f"Trying fallback '{self.fallback_model}'..."
            )
            return self._call_groq(groq_messages, self.fallback_model, stop)
        
        raise Exception(f"Groq API failed after {self.max_retries} retries on both models")

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Main entry point for the LLM. Generates a response with retry and fallback logic.
        """
        # 1. Convert LangChain messages to Groq format
        groq_messages = []
        for msg in messages:
            role = "user"
            if isinstance(msg, SystemMessage):
                role = "system"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            elif isinstance(msg, HumanMessage):
                role = "user"
            
            groq_messages.append({"role": role, "content": msg.content})

        # 2. Call Groq API with retry + fallback
        response = self._call_groq(groq_messages, self.model, stop)

        # 3. Extract Content and Metrics
        content = response.choices[0].message.content
        
        # Safely access usage stats
        usage = response.usage
        token_usage = {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0),
            "completion_tokens": getattr(usage, "completion_tokens", 0),
            "total_tokens": getattr(usage, "total_tokens", 0)
        }

        # Log model used (useful for tracking fallback usage)
        model_used = response.model if hasattr(response, 'model') else self.model
        logger.info(f" [LLM] Generated with '{model_used}' | Tokens: {token_usage['total_tokens']}")

        # 4. Return Rich AIMessage
        msg = AIMessage(
            content=content, 
            response_metadata={
                "token_usage": token_usage,
                "model_used": model_used
            }
        )

        generation = ChatGeneration(message=msg)
        return ChatResult(generations=[generation])

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model": self.model, 
            "fallback_model": self.fallback_model,
            "temperature": self.temperature,
            "top_p": self.top_p
        }

_llm_client = None

def get_llm():
    """
    Returns a single, shared instance of the Groq LLM client.
    """
    global _llm_client
    if _llm_client is None:
        logger.info(f" Initializing Groq Chat Model: {settings.GROQ_MODEL}")
        logger.info(f"   Fallback: {settings.GROQ_MODEL_FALLBACK}")
        _llm_client = CustomGroqLLM()
        logger.info(" Groq Chat Model Initialized.")
        
    return _llm_client