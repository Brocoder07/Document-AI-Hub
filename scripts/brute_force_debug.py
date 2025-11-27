import sys
import os
import weaviate
from weaviate.classes.query import Filter, MetadataQuery

# Add root path
sys.path.append(os.getcwd())
from app.services.embedding_service import embed_texts

def brute_force_debug():
    print("🔨 STARTING BRUTE FORCE DEBUG...")
    
    # 1. SETUP - Connect directly (No Adapter)
    client = weaviate.connect_to_local(
        headers={"X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY", "")}
    )
    
    try:
        collection_name = "AcademicDocs"
        if not client.collections.exists(collection_name):
            print(f"❌ Collection {collection_name} not found!")
            return

        col = client.collections.get(collection_name)
        
        # 2. DATA - Use the IDs you confirmed exist
        target_user = "5"
        target_file = "3d686cbd-59ff-4545-a2f7-bd12d78138f7"
        query_text = "Explain the car racing environment in detail"
        
        print("   Generating Embedding...")
        vec = embed_texts([query_text])[0]
        
        # 3. DEFINE FILTER (We know this works from previous test)
        f = Filter.by_property("user_id").equal(target_user) & \
            Filter.by_property("file_id").equal(target_file)

        print("\n🧪 TEST 1: Standard near_vector (No target_vector param)")
        try:
            res1 = col.query.near_vector(
                near_vector=vec,
                limit=5,
                filters=f,
                return_metadata=MetadataQuery(distance=True)
            )
            print(f"   👉 Results: {len(res1.objects)}")
        except Exception as e:
            print(f"   💥 Error: {e}")

        print("\n🧪 TEST 2: near_vector + target_vector='default'")
        try:
            res2 = col.query.near_vector(
                near_vector=vec,
                limit=5,
                filters=f,
                return_metadata=MetadataQuery(distance=True),
                target_vector="default" # <--- Explicitly targeting the name
            )
            print(f"   👉 Results: {len(res2.objects)}")
        except Exception as e:
            print(f"   💥 Error: {e}")

        print("\n🧪 TEST 3: near_vector WITHOUT Filters (Is the vector index working?)")
        try:
            res3 = col.query.near_vector(
                near_vector=vec,
                limit=5,
                target_vector="default"
            )
            print(f"   👉 Results: {len(res3.objects)}")
        except Exception as e:
            print(f"   💥 Error: {e}")

    finally:
        client.close()

if __name__ == "__main__":
    brute_force_debug()