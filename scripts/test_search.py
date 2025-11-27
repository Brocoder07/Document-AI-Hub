import sys
import os
import weaviate
from weaviate.classes.query import Filter

# Add root to path
sys.path.append(os.getcwd())
from app.api.vector_db import db_client

def test_search():
    # 1. SETUP: Use the EXACT values from your debug logs
    # From your logs: User ID: 5, File ID: 3d686cbd...
    TARGET_USER_ID = "5"
    TARGET_FILE_ID = "3d686cbd-59ff-4545-a2f7-bd12d78138f7"
    
    # Check BOTH collections (just in case)
    for COLLECTION in ["AcademicDocs", "GeneralDocs"]:
        print(f"\n🧪 Testing Collection: '{COLLECTION}'")
        
        if not db_client.client.collections.exists(COLLECTION):
            print(f"   ❌ Collection '{COLLECTION}' not found.")
            continue

        coll = db_client.client.collections.get(COLLECTION)

        # --- TEST A: User ID Only ---
        # This checks if the "5" matches "5"
        res_user = coll.query.fetch_objects(
            filters=Filter.by_property("user_id").equal(TARGET_USER_ID),
            limit=5
        )
        print(f"   A. Filter by User ID ('{TARGET_USER_ID}'): Found {len(res_user.objects)} matches.")

        # --- TEST B: File ID Only ---
        # This checks if the long UUID string matches
        res_file = coll.query.fetch_objects(
            filters=Filter.by_property("file_id").equal(TARGET_FILE_ID),
            limit=5
        )
        print(f"   B. Filter by File ID ('{TARGET_FILE_ID[:8]}...'): Found {len(res_file.objects)} matches.")

        # --- TEST C: Combined (The RAG Query) ---
        # This checks the AND logic
        res_both = coll.query.fetch_objects(
            filters=Filter.by_property("user_id").equal(TARGET_USER_ID) & 
                    Filter.by_property("file_id").equal(TARGET_FILE_ID),
            limit=5
        )
        print(f"   C. Combined Filter (User + File): Found {len(res_both.objects)} matches.")

if __name__ == "__main__":
    test_search()