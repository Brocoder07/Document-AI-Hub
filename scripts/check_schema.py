import sys
import os
import weaviate

# Add root
sys.path.append(os.getcwd())
from app.api.vector_db import db_client

def check_schema():
    print("🕵️ Checking Weaviate Schema...")
    
    for cls in ["AcademicDocs", "GeneralDocs"]:
        if not db_client.client.collections.exists(cls):
            print(f"❌ Class {cls} not found.")
            continue
            
        coll = db_client.client.collections.get(cls)
        config = coll.config.get()
        
        print(f"\n📘 Class: {cls}")
        for prop in config.properties:
            if prop.name in ["file_id", "user_id"]:
                # We want tokenization to be 'field' (Exact Match)
                # If it is 'word' or 'whitespace', THAT IS THE BUG.
                print(f"   - Property '{prop.name}': Tokenization = {prop.tokenization}")
                
                if str(prop.tokenization) != "Tokenization.FIELD":
                    print(f"     ⚠️  CRITICAL: {prop.name} is configured WRONG. It must be 'field'.")
                else:
                    print(f"     ✅  {prop.name} is OK.")

if __name__ == "__main__":
    check_schema()