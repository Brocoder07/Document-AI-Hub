from app.core.config import settings
from langchain_community.chat_models import ChatGroq

_llm_client = None

def get_llm():
    """
    Returns a single, shared instance of the ChatGroq LLM client.
    """
    global _llm_client
    
    if _llm_client is None:
        print("Initializing Groq LLM Client...")
        _llm_client = ChatGroq(
            model_name=settings.GROQ_MODEL_NAME,
            groq_api_key=settings.GROQ_API_KEY,
            temperature=0.2
        )
        print("Groq LLM Client Initialized.")
        
    return _llm_client