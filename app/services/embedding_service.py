from sentence_transformers import SentenceTransformer
from app.core.config import settings

_model = None

def get_model():
    global _model
    if _model is None:
        # Use getattr to prevent errors if config is missing the key
        model_name = getattr(settings, "EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        _model = SentenceTransformer(model_name)
    return _model

def embed_texts(texts):
    """
    Generate embeddings for a list of texts.
    Returns a list of list of floats.
    """
    model = get_model()
    # Ensure input is a list
    if isinstance(texts, str):
        texts = [texts]
        
    embs = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return [e.tolist() for e in embs]

def generate_embedding(text: str):
    """
    Generate embedding for a single string.
    Wrapper for compatibility with search service.
    """
    embeddings = embed_texts([text])
    return embeddings[0]