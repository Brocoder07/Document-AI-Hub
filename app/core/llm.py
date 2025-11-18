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

class CustomGroqLLM(BaseChatModel):
    """
    Custom ChatModel wrapper for Groq API.
    Returns AIMessage objects with metadata (token usage), enabling rich RAG metrics.
    """
    model: str = settings.GROQ_MODEL
    groq_api_key: str = settings.GROQ_API_KEY
    temperature: float = 0.2

    @property
    def _llm_type(self) -> str:
        return "custom_groq_chat"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Main entry point for the LLM. Generates a response and captures metadata.
        """
        client = GroqClient(api_key=self.groq_api_key)
        
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

        # 2. Call Groq API
        response = client.chat.completions.create(
            messages=groq_messages,
            model=self.model,
            temperature=self.temperature,
            stop=stop
        )

        # 3. Extract Content and Metrics
        content = response.choices[0].message.content
        
        # Safely access usage stats (defaults to 0 if missing)
        usage = response.usage
        token_usage = {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0),
            "completion_tokens": getattr(usage, "completion_tokens", 0),
            "total_tokens": getattr(usage, "total_tokens", 0)
        }

        # 4. Return Rich AIMessage
        # This structure satisfies the rag_service.py requirements
        msg = AIMessage(
            content=content, 
            response_metadata={"token_usage": token_usage}
        )

        generation = ChatGeneration(message=msg)
        return ChatResult(generations=[generation])

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model": self.model, "temperature": self.temperature}

_llm_client = None

def get_llm():
    """
    Returns a single, shared instance of the Groq LLM client.
    """
    global _llm_client
    if _llm_client is None:
        print("Initializing Groq Chat Model...")
        _llm_client = CustomGroqLLM()
        print("Groq Chat Model Initialized.")
        
    return _llm_client