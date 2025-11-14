from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from app.api.dependencies import get_current_active_user, UserInDB

router = APIRouter()

# --- Pydantic Schemas ---

class UserProfile(BaseModel):
    username: str
    email: str
    full_name: str
    role: str

# --- Endpoints ---

@router.get("/me", response_model=UserProfile)
async def read_users_me(
    current_user: UserInDB = Depends(get_current_active_user)
):
    """
    Get profile for the currently authenticated user.
    """
    return UserProfile(
        username=current_user.username,
        email=current_user.email,
        full_name=current_user.full_name,
        role=current_user.role
    )