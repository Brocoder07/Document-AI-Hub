import chromadb
from app.core.config import settings

_client = None

def get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIR
        )
    return _client

def get_collection(name: str):
    client = get_client()
    try:
        # FIX: Explicitly set the distance metric to "cosine"
        return client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"} 
        )
    except Exception as e:
        print(f"Error getting collection {name}: {e}")
        raise e