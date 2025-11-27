import sys
import os
import logging

# Add root to path
sys.path.append(os.getcwd())

# Import the actual tools used by the app
from app.api.vector_db import db_client
from app.services.embedding_service import embed_texts

# Setup logging to see what the adapter is doing
logging.basicConfig(level=logging.INFO)

def test_vector_search():
    print("🧪 Testing ADAPTER Vector Search...")
    
    # 1. Setup exact variables from your failure logs
    query = "Explain the car racing environment in detail"
    user_id = "5"
    file_id = "3d686cbd-59ff-4545-a2f7-bd12d78138f7"
    collection = "academic_docs" # App uses lowercase, adapter handles capitalization

    print(f"   Query: '{query}'")
    print(f"   Filters: User={user_id}, File={file_id}")

    # 2. Embed the query (Just like the App does)
    print("   ... Generating Embedding ...")
    query_vector = embed_texts([query])[0]
    print(f"   ... Vector generated ({len(query_vector)} dimensions).")

    # 3. Build the Filter dictionary (Just like the App does)
    where = {
        "$and": [
            {"user_id": user_id},
            {"file_id": file_id}
        ]
    }

    # 4. Call the Adapter's query method (The point of failure)
    print("   ... Calling db_client.query() ...")
    try:
        results = db_client.query(
            collection_name=collection,
            query_vector=query_vector,
            top_k=5,
            where=where
        )

        # 5. Analyze Results
        hits = len(results["documents"][0]) if results["documents"] else 0
        print(f"\n🔍 RESULT: Found {hits} matches.")
        
        if hits > 0:
            print("✅ The Adapter works! The issue is likely elsewhere.")
        else:
            print("❌ The Adapter FAILED. The bug is inside 'app/api/vector_db.py'.")

    except Exception as e:
        print(f"💥 CRASH: {e}")

if __name__ == "__main__":
    test_vector_search()