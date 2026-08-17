"""
Weaviate Migration Script — For Embedding Model Upgrade

When switching from all-mpnet-base-v2 (384-dim) to BAAI/bge-large-en-v1.5 (1024-dim),
all existing vectors become incompatible. This script:

1. Connects to Weaviate
2. Lists all collections  
3. Deletes all collections (vectors are incompatible with new model)
4. Prints instructions for re-uploading documents

Usage:
    python scripts/migrate_embeddings.py

IMPORTANT: Run this AFTER updating the embedding model in config.py
           and BEFORE re-uploading documents.
"""

import weaviate
import sys

def main():
    print("=" * 60)
    print("🔄 WEAVIATE MIGRATION: Embedding Model Upgrade")
    print("   Old: all-mpnet-base-v2 (384-dim)")
    print("   New: BAAI/bge-large-en-v1.5 (1024-dim)")
    print("=" * 60)
    
    try:
        client = weaviate.connect_to_local()
        print("✅ Connected to Weaviate")
    except Exception as e:
        print(f"❌ Cannot connect to Weaviate: {e}")
        print("   Make sure Weaviate is running: docker compose up -d")
        sys.exit(1)
    
    try:
        # List existing collections
        collections = client.collections.list_all()
        collection_names = list(collections.keys()) if isinstance(collections, dict) else [c.name for c in collections]
        
        if not collection_names:
            print("ℹ️  No existing collections found. Nothing to migrate.")
            print("   You can start uploading 3GPP documents right away!")
            return
        
        print(f"\n📋 Found {len(collection_names)} collections:")
        for name in collection_names:
            print(f"   - {name}")
        
        # Confirm deletion
        print(f"\n⚠️  All {len(collection_names)} collections will be DELETED.")
        print("   This is necessary because the old vectors (384-dim) are incompatible")
        print("   with the new embedding model (1024-dim).")
        
        confirm = input("\n   Type 'YES' to proceed: ").strip()
        
        if confirm != "YES":
            print("❌ Migration cancelled.")
            return
        
        # Delete all collections
        for name in collection_names:
            try:
                client.collections.delete(name)
                print(f"   🗑️  Deleted: {name}")
            except Exception as e:
                print(f"   ⚠️  Failed to delete {name}: {e}")
        
        print(f"\n✅ Migration complete! All {len(collection_names)} collections deleted.")
        print("\n📌 NEXT STEPS:")
        print("   1. Start the FastAPI server: uvicorn app.main:app --reload")
        print("   2. Log in to the frontend")
        print("   3. Upload your 3GPP specification PDFs")
        print("   4. The new BGE embedding model will be used automatically")
        print("   5. New 'telecom_docs' collection will be created for 3GPP docs")
        
    except Exception as e:
        print(f"❌ Error during migration: {e}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
