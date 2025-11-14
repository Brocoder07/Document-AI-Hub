from sqlalchemy.orm import Session
from app.models.users import User
from app.core.security import get_password_hash, verify_password
from pydantic import EmailStr

def get_user_by_email(db: Session, email: EmailStr) -> User | None:
    """
    Retrieves a user from the database by their email.
    """
    return db.query(User).filter(User.email == email).first()

def get_user_by_username(db: Session, username: str) -> User | None:
    """
    Retrieves a user from the database by their username.
    """
    return db.query(User).filter(User.username == username).first()

def create_user(db: Session, user_data: dict) -> User:
    """
    Creates a new user in the database.
    'user_data' should be a dict with 'username', 'email', 'full_name', 'password'.
    """
    hashed_password = get_password_hash(user_data["password"])
    
    db_user = User(
        username=user_data["username"],
        email=user_data["email"],
        full_name=user_data.get("full_name"),
        hashed_password=hashed_password,
        is_active=True,
        role="user" # Default role
    )
    
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def authenticate_user(db: Session, email: EmailStr, password: str) -> User | None:
    """
    Authenticates a user.
    Returns the user object if successful, None otherwise.
    """
    user = get_user_by_email(db, email=email)
    
    if not user:
        return None # User not found
    
    if not verify_password(password, user.hashed_password):
        return None # Incorrect password
        
    return user