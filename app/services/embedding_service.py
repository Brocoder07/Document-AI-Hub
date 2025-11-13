from sentence_transformers import SentenceTransformer
from app.core.config import settings

_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.EMBEDDING_MODEL)
    return _model

def embed_texts(texts):
    model = get_model()
    embs = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return [e.tolist() for e in embs]