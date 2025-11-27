from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional

# --- Imports ---
from app.api.dependencies import get_current_active_user, UserInDB
from app.services.embedding_service import embed_texts
# FIX: Use the new Weaviate Adapter
from app.api.vector_db import db_client

router = APIRouter()

class SearchResult(BaseModel):
    id: str
    text: str
    metadata: dict
    score: float

@router.get("/", response_model=List[SearchResult])
async def search_documents(
    q: str,
    top_k: int = 5,
    collection_name: str = "general_docs", # Weaviate class name (lowercase ok, adapter fixes it)
    file_id: Optional[str] = None,
    current_user: UserInDB = Depends(get_current_active_user)
):
    """
    Search endpoint that now uses Weaviate via the Adapter.
    """
    try:
        # 1. Embed the Query
        query_vector = embed_texts([q])[0]

        # 2. Build Security Filters
        # We ensure users can ONLY see their own documents
        where = {"user_id": str(current_user.id)}
        
        if file_id:
            # If filtering by a specific file, combine logic
            where = {"$and": [{"user_id": str(current_user.id)}, {"file_id": file_id}]}

        # 3. Execute Search (Using new Adapter)
        results = db_client.query(
            collection_name=collection_name,
            query_vector=query_vector,
            top_k=top_k,
            where=where
        )

        # 4. Format Results
        formatted_results = []
        
        # Check if we got hits (Adapter returns lists inside a dict)
        if results["documents"] and results["documents"][0]:
            count = len(results["documents"][0])
            for i in range(count):
                # Calculate a rough similarity score from distance
                # Weaviate returns distance (lower is better), we want similarity (higher is better)
                dist = results["distances"][0][i]
                score = max(0, 1.0 - (dist / 2)) 

                formatted_results.append(
                    SearchResult(
                        id=results["ids"][0][i],
                        text=results["documents"][0][i],
                        metadata=results["metadatas"][0][i],
                        score=round(score, 4)
                    )
                )
        
        return formatted_results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")