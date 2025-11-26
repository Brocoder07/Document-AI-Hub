from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List
from app.api.dependencies import get_current_active_user
from app.services import embedding_service
from app.api.chroma_client import get_collection 

router = APIRouter()

# SENIOR ENG FIX: Changed collection name from "documents" to "general_docs"
# to match what is defined in file_processing_service.py
collection = get_collection("general_docs")

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

class SearchResult(BaseModel):
    id: str
    text: str
    metadata: dict = {}
    score: float

@router.post("/similarity", response_model=List[SearchResult])
def similarity_search(
    request: SearchRequest,
    current_user = Depends(get_current_active_user)
):
    """
    Perform a vector similarity search on the 'general_docs' collection.
    """
    try:
        # 1. Generate Embedding
        query_embedding = embedding_service.generate_embedding(request.query)
        
        # 2. Query Vector DB (Chroma)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=request.top_k,
            include=["metadatas", "documents", "distances"]
        )
        
        # 3. Format Results
        formatted_results = []
        if results and results['ids']:
            ids = results['ids'][0]
            docs = results['documents'][0]
            metas = results['metadatas'][0]
            dists = results['distances'][0] if 'distances' in results else [0.0]*len(ids)
            
            for i in range(len(ids)):
                formatted_results.append(SearchResult(
                    id=ids[i],
                    text=docs[i],
                    metadata=metas[i] if metas[i] else {},
                    score=dists[i]
                ))
                
        return formatted_results

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))