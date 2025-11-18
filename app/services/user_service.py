from sqlalchemy.orm import Session
from app.models.users import User
from app.core.security import get_password_hash, verify_password
from pydantic import EmailStr

def get_user_by_email(db: Session, email: EmailStr) -> User | None:
    return db.query(User).filter(User.email == email).first()

def get_user_by_username(db: Session, username: str) -> User | None:
    return db.query(User).filter(User.username == username).first()

def get_user_by_id(db: Session, user_id: int) -> User | None:
    return db.query(User).filter(User.id == user_id).first()

def create_user(db: Session, user_data: dict) -> User:
    """
    Creates a new user.
    Strictly enforces role validation.
    """
    if len(user_data["password"].encode('utf-8')) > 72:
        raise ValueError("Password exceeds maximum length of 72 bytes.")
    
    hashed_password = get_password_hash(user_data["password"])
    
    # 1. Get requested role
    requested_role = user_data.get("role")
    if not requested_role:
        raise ValueError("Role is required. You must specify if you are a student, lawyer, doctor, etc.")
    
    # 2. Validate Role Whitelist (REMOVED "admin" from this list)
    valid_roles = ["student", "researcher", "lawyer", "doctor", "banker", "employee"]
    
    if requested_role not in valid_roles:
        raise ValueError(f"Invalid role '{requested_role}'. Allowed roles: {', '.join(valid_roles)}")

    db_user = User(
        username=user_data["username"],
        email=user_data["email"],
        full_name=user_data.get("full_name"),
        hashed_password=hashed_password,
        is_active=True,
        role=requested_role
    )
    
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def authenticate_user(db: Session, email: EmailStr, password: str) -> User | None:
    user = get_user_by_email(db, email=email)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user

def update_user(db: Session, db_user: User, update_data: dict) -> User:
    for key, value in update_data.items():
        if key == "password" and value:
            if len(value.encode('utf-8')) > 72:
                raise ValueError("Password exceeds maximum length of 72 bytes.")
            hashed_pw = get_password_hash(value)
            setattr(db_user, "hashed_password", hashed_pw)
        else:
            setattr(db_user, key, value)

    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def soft_delete_user(db: Session, db_user: User) -> User:
    db_user.is_active = False
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user