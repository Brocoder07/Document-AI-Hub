from app.db.session import SessionLocal
from sqlalchemy import text

def clear_documents():
    db = SessionLocal()
    try:
        print("Cleaning 'documents' table...")
        # We use raw SQL because the Python model 'Document' 
        # might be out of sync with the actual database table state.
        db.execute(text("DELETE FROM documents;"))
        db.commit()
        print("✅ Successfully deleted all old document records.")
    except Exception as e:
        print(f"❌ Error: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    clear_documents()