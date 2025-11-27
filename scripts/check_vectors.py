import sys
import os
sys.path.append(os.getcwd())
from app.api.vector_db import db_client

def check_vectors():
    print("🕵️ Checking if objects have Vectors...")
    collection = db_client.client.collections.get("AcademicDocs")
    
    # Fetch 5 objects AND request their vectors
    response = collection.query.fetch_objects(
        limit=5,
        include_vector=True # <--- This is the key check
    )
    
    if not response.objects:
        print("❌ No objects found in AcademicDocs.")
        return

    for i, obj in enumerate(response.objects):
        has_vector = obj.vector is not None and len(obj.vector) > 0
        vec_len = len(obj.vector) if has_vector else 0
        
        status = "✅" if has_vector else "❌"
        print(f"[{i}] UUID: {obj.uuid} | Vector Length: {vec_len} {status}")
        
        if not has_vector:
            print("   ⚠️  CRITICAL: Object exists but has NO VECTOR. 'near_vector' will never find it.")

if __name__ == "__main__":
    check_vectors()