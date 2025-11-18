from app.core.config import settings
from groq import Groq as GroqClient
from langchain.llms.base import LLM
from typing import Optional, List, Any, Mapping

class CustomGroqLLM(LLM):
    model: str = settings.GROQ_MODEL
    groq_api_key: str = settings.GROQ_API_KEY
    temperature: float = 0.2
    
    @property
    def _llm_type(self) -> str:
        return "custom_groq"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None, run_manager: Optional[Any] = None, **kwargs) -> str:
        """Fix: Added **kwargs to handle extra parameters"""
        client = GroqClient(api_key=self.groq_api_key)
        
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
            temperature=self.temperature,
            stop=stop
        )
        
        return response.choices[0].message.content
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model": self.model, "temperature": self.temperature}

_llm_client = None

def get_llm():
    """
    Returns a single, shared instance of the Groq LLM client.
    """
    global _llm_client
    
    if _llm_client is None:
        print("Initializing Groq LLM Client...")
        _llm_client = CustomGroqLLM()
        print("Groq LLM Client Initialized.")
        
    return _llm_client