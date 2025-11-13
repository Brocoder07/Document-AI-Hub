from app.api.utility.chroma_client import get_collection
from app.services.embedding_service import embed_texts
from app.services.generator_service import generate_answer

def retrieve(query: str, top_k: int = 4):
    col = get_collection("documents")
    q_emb = embed_texts([query])[0]

    res = col.query(query_embeddings=[q_emb], n_results=top_k,
                    include=["documents", "metadatas", "ids"])
    
    docs = res["documents"][0] if res["documents"] else []
    metas = res["metadatas"][0] if res["metadatas"] else []
    ids = res["ids"][0] if res["ids"] else []

    return [{"id": i, "text": d, "meta": m} for i, d, m in zip(ids, docs, metas)]

def answer_query(query: str):
    retrieved = retrieve(query)
    
    context = "\n\n".join(
        f"[{r['meta'].get('file_id', '?')}] {r['text']}" for r in retrieved
    )

    prompt = f"""
Use the following context to answer the question. 
If the answer is not present, say you don't know.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""

    answer = generate_answer(prompt)
    return {"answer": answer, "retrieved": retrieved}
