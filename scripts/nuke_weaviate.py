import sys
import os
sys.path.append(os.getcwd())
from app.api.vector_db import db_client

# Delete the classes with the bad schema
for cls in ["GeneralDocs", "AcademicDocs", "LegalDocs", "BusinessDocs", "FinanceDocs", "MedicalDocs"]:
    try:
        db_client.client.collections.delete(cls)
        print(f"🗑️ Deleted {cls}")
    except:
        pass
print("✅ Weaviate Cleared.")