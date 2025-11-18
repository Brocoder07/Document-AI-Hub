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

# FIX: Allow passing a dynamic name
def get_collection(name: str):
    client = get_client()
    try:
        # try to get it, create if not exists
        return client.get_or_create_collection(name)
    except Exception as e:
        print(f"Error getting collection {name}: {e}")
        raise e