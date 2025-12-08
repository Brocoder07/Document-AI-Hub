import sys
import os
sys.path.append(os.getcwd())
from app.api.vector_db import db_client

def fix():
    print("🗑️  Deleting broken 'BusinessDocs' class...")
    try:
        db_client.client.collections.delete("BusinessDocs")
        print("✅ Deleted. Next upload/migration will recreate it with the correct schema.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    fix()