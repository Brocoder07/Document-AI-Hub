import sys
import os
sys.path.append(os.getcwd())
from app.api.vector_db import db_client

def inspect():
    print("🕵️ Deep Inspecting Vector Content...")
    coll = db_client.client.collections.get("AcademicDocs")
    
    # Fetch 1 object with vectors
    res = coll.query.fetch_objects(limit=1, include_vector=True)
    
    if res.objects:
        obj = res.objects[0]
        vec = obj.vector
        print(f"\n🔍 Raw Vector Object Type: {type(vec)}")
        print(f"🔍 Raw Vector Content: {vec}")
        
        if isinstance(vec, dict):
            print("\n✅ CONFIRMED: Weaviate is using Named Vectors.")
            print(f"   Keys: {vec.keys()}")
            first_key = list(vec.keys())[0]
            print(f"   Vector Length inside '{first_key}': {len(vec[first_key])}")
    else:
        print("❌ No objects found.")

if __name__ == "__main__":
    inspect()