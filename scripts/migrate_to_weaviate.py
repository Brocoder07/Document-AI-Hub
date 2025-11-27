import sys
import os
import logging

# Add the project root to python path so we can import 'app'
sys.path.append(os.getcwd())

from app.db.session import SessionLocal
# --- FIX: Import User so SQLAlchemy knows about it ---
from app.models.users import User  
from app.models.documents import Document
from app.services.file_processing_service import process_document
from app.api.vector_db import db_client

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("migration")

def run_migration():
    print("🚀 Starting Migration: ChromaDB -> Weaviate (Re-Ingestion)")
    
    db = SessionLocal()
    
    try:
        # 1. Get all documents
        docs = db.query(Document).all()
        
        print(f"Found {len(docs)} documents in the database.")
        
        for doc in docs:
            print(f"Processing: {doc.filename} (ID: {doc.file_id})...")
            
            # Check if file exists on disk
            if not os.path.exists(doc.file_path):
                print(f"❌ File missing from disk: {doc.file_path}. Skipping.")
                continue
                
            # 2. Get User Role
            user_role = "employee" 
            if doc.owner:
                user_role = doc.owner.role
            
            try:
                # 3. Trigger Processing (Chunking + Weaviate Ingestion)
                process_document(
                    saved_path=doc.file_path,
                    file_id=doc.file_id,
                    filename=doc.filename,
                    user_id=str(doc.user_id),
                    user_role=user_role
                )
                print(f"✅ Successfully migrated {doc.filename}")
                
            except Exception as e:
                print(f"❌ Failed to migrate {doc.filename}: {e}")

    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
    finally:
        db.close()
        print("🏁 Migration Job Finished.")

if __name__ == "__main__":
    confirm = input("This will re-process ALL documents. Are you sure? (y/n): ")
    if confirm.lower() == 'y':
        run_migration()
    else:
        print("Migration cancelled.")