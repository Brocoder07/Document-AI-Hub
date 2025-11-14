from chromadb.config import Settings
import chromadb
from app.core.config import settings

_client = None

def get_client():
    global _client
    if _client is None:
        # Use the new ChromaDB client configuration
        _client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIR
        )
    return _client

def get_collection(name="documents"):
    client = get_client()
    try:
        return client.get_collection(name)
    except:
        return client.create_collection(name)