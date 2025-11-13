from chromadb.config import Settings
import chromadb
from app.core.config import settings

_client = None

def get_client():
    global _client
    if _client is None:
        _client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=settings.CHROMA_PERSIST_DIR
        ))
    return _client

def get_collection(name="documents"):
    client = get_client()
    names = [c.name for c in client.list_collections()]
    if name in names:
        return client.get_collection(name)
    return client.create_collection(name)
