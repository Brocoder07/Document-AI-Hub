from app.db.session import SessionLocal
from app.models.users import User

def promote_tester():
    db = SessionLocal()
    try:
        # 1. Find the user
        user_email = "tester@gmail.com"
        user = db.query(User).filter(User.email == user_email).first()
        
        if not user:
            print(f"❌ User {user_email} not found!")
            return

        print(f"Current Role: {user.role}")

        # 2. Update the role
        # Options: "student", "researcher", "admin"
        new_role = "admin" 
        user.role = new_role
        
        db.commit()
        print(f"✅ Successfully promoted {user_email} to '{new_role}'")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    promote_tester()