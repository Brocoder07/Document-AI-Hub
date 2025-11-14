from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session

import app.core.security as security
import app.services.user_service as user_service # <-- IMPORT NEW SERVICE
from app.db.session import get_db # <-- IMPORT DB SESSION
from app.models.users import User # <-- IMPORT DB MODEL

# Pydantic model for user *from the database*
# This replaces the old UserInDB
class UserInDB(BaseModel):
    username: str
    email: EmailStr
    full_name: str
    hashed_password: str
    is_active: bool
    role: str

    class Config:
        from_attributes = True # Replaces orm_mode = True

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

def get_current_user(
    db: Session = Depends(get_db), 
    token: str = Depends(oauth2_scheme)
) -> User:
    """
    Dependency to get the current user from a JWT token.
    This is the core of your API's security.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    payload = security.decode_token(token)
    if payload is None or payload.sub is None:
        raise credentials_exception
        
    email: EmailStr = payload.sub
    user = user_service.get_user_by_email(db, email=email)
    
    if user is None:
        raise credentials_exception
        
    return user

def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Dependency to get the current *active* user.
    Checks if the user (from the token) is marked as inactive.
    """
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user