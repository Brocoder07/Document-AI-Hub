import sys
import os
import json

# Add root to path
sys.path.append(os.getcwd())

from app.api.vector_db import db_client

def inspect_class(class_name):
    print(f"\n--- INSPECTING CLASS: {class_name} ---")
    try:
        # Check if class exists
        if not db_client.client.collections.exists(class_name):
            print(f"❌ Class '{class_name}' does not exist.")
            return

        collection = db_client.client.collections.get(class_name)
        
        # Fetch 20 objects to inspect metadata
        response = collection.query.fetch_objects(limit=20, include_vector=False)
        
        if not response.objects:
            print("⚠️  Class exists but is EMPTY.")
            return
            
        print(f"✅ Found {len(response.objects)} objects. Showing first 5:\n")
        
        for i, obj in enumerate(response.objects[:5]):
            props = obj.properties
            print(f"[{i}] UUID: {obj.uuid}")
            print(f"    User ID: {props.get('user_id')} (Type: {type(props.get('user_id'))})")
            print(f"    File ID: {props.get('file_id')}")
            print(f"    Filename: {props.get('filename')}")
            print(f"    Text Snippet: {props.get('text', '')[:50]}...")
            print("-" * 40)
            
    except Exception as e:
        print(f"Error inspecting {class_name}: {e}")

if __name__ == "__main__":
    # Inspect both places a file might be
    inspect_class("GeneralDocs")
    inspect_class("AcademicDocs")