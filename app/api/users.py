from fastapi import APIRouter, Depends, status
from pydantic import BaseModel, EmailStr
from typing import Optional
from sqlalchemy.orm import Session
from app.api.dependencies import get_current_active_user, UserInDB
from app.services.user_service import update_user, soft_delete_user
from app.db.session import get_db

router = APIRouter()

# --- Pydantic Schemas ---

class UserProfile(BaseModel):
    id: int
    username: str
    email: str
    full_name: Optional[str] = None
    role: str
    is_active: bool

class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    password: Optional[str] = None

# --- Endpoints: Self-Management ---

@router.get("/me", response_model=UserProfile)
async def read_users_me(
    current_user: UserInDB = Depends(get_current_active_user)
):
    """
    Get profile for the currently authenticated user.
    """
    return current_user

@router.patch("/me", response_model=UserProfile)
async def update_user_me(
    user_in: UserUpdate,
    current_user: UserInDB = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Update your own profile.
    """
    update_data = user_in.model_dump(exclude_unset=True)
    updated_user = update_user(db, current_user, update_data)
    return updated_user

@router.delete("/me", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user_me(
    current_user: UserInDB = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Soft delete your own account.
    """
    soft_delete_user(db, current_user)
    return None