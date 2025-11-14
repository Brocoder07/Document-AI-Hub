from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session
from datetime import timedelta

import app.core.security as security
import app.services.user_service as user_service
from app.api.dependencies import get_current_active_user
from app.db.session import get_db
from app.core.config import settings

router = APIRouter()

# --- Pydantic Schemas ---

class Token(BaseModel):
    access_token: str
    token_type: str

class UserCreate(BaseModel):
    username: str
    email: EmailStr
    full_name: str
    password: str

class UserPublic(BaseModel):
    username: str
    email: EmailStr
    full_name: str

    class Config:
        from_attributes = True # Pydantic v2 way to read from ORM


# --- Endpoints ---

@router.post("/token", response_model=Token)
async def login_for_access_token(
    db: Session = Depends(get_db),
    form_data: OAuth2PasswordRequestForm = Depends()
):
    """
    OAuth2 compatible token endpoint.
    Takes username (email) and password from a form body.
    """
    user = user_service.authenticate_user(
        db, 
        email=form_data.username, 
        password=form_data.password
    )
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = security.create_access_token(
        subject=user.email, expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}


@router.post("/register", response_model=UserPublic, status_code=status.HTTP_201_CREATED)
async def register_user(
    user_in: UserCreate,
    db: Session = Depends(get_db)
):
    """
    User registration endpoint.
    """
    if user_service.get_user_by_email(db, email=user_in.email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )
    if user_service.get_user_by_username(db, username=user_in.username):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered",
        )
        
    user = user_service.create_user(db, user_data=user_in.model_dump())
    
    # --- THIS IS THE FIX ---
    # Change from .model_from_object() to .model_validate()
    return UserPublic.model_validate(user)