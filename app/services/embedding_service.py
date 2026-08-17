from sentence_transformers import SentenceTransformer
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

_model = None

# BGE models require instruction prefixes for optimal performance.
# Query prefix improves retrieval accuracy by 3-5% on benchmarks.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

def get_model():
    global _model
    if _model is None:
        model_name = getattr(settings, "EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
        logger.info(f" Loading embedding model: {model_name}")
        _model = SentenceTransformer(model_name)
        logger.info(f" Embedding model loaded. Dimension: {_model.get_sentence_embedding_dimension()}")
    return _model

def embed_texts(texts, is_query: bool = False):
    """
    Generate embeddings for a list of texts.
    
    Args:
        texts: List of strings or single string to embed
        is_query: If True, adds BGE query instruction prefix for better retrieval.
                  Set to True for search queries, False for documents being indexed.
    
    Returns:
        List of embedding vectors (list of floats)
    """
    model = get_model()
    
    # Ensure input is a list
    if isinstance(texts, str):
        texts = [texts]
    
    # BGE models benefit from instruction prefixes on queries
    if is_query:
        texts = [QUERY_INSTRUCTION + t for t in texts]
    
    embs = model.encode(
        texts, 
        show_progress_bar=False, 
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2-normalize for cosine similarity
        batch_size=32
    )
    return [e.tolist() for e in embs]

def generate_embedding(text: str):
    """
    Generate embedding for a single string.
    Wrapper for compatibility with search service.
    """
    embeddings = embed_texts([text], is_query=True)
    return embeddings[0]