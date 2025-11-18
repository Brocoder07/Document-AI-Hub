from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from sqlalchemy.orm import Session

from app.api.dependencies import get_current_active_user, UserInDB, RoleChecker
from app.services.user_service import update_user, soft_delete_user, get_user_by_id, create_user, get_user_by_email, get_all_users
from app.db.session import get_db

router = APIRouter()

# --- Pydantic Schemas ---

class UserCreateAdmin(BaseModel):
    username: str
    email: EmailStr
    password: str
    full_name: str
    role: str 

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

class UserUpdateAdmin(UserUpdate):
    role: Optional[str] = None 

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

# --- Endpoints: Admin Management (RBAC) ---

@router.get("/", response_model=list[UserProfile], dependencies=[Depends(RoleChecker(["admin"]))], tags=["Admin"])
async def list_all_users(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """
    [Admin Only] List all users (active and inactive).
    """
    users = get_all_users(db, skip=skip, limit=limit)
    return users

@router.post("/", response_model=UserProfile, status_code=status.HTTP_201_CREATED, dependencies=[Depends(RoleChecker(["admin"]))], tags=["Admin"])
async def create_user_by_admin(
    user_in: UserCreateAdmin,
    db: Session = Depends(get_db)
):
    """
    [Admin Only] Create a new user with ANY role.
    """
    if get_user_by_email(db, email=user_in.email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )
    
    try:
        user = create_user(
            db, 
            user_data=user_in.model_dump(), 
            allow_admin_role=True 
        )
        return user
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

@router.patch("/{user_id}", response_model=UserProfile, dependencies=[Depends(RoleChecker(["admin"]))], tags=["Admin"])
async def update_user_by_id(
    user_id: int,
    user_in: UserUpdateAdmin,
    db: Session = Depends(get_db)
):
    """
    [Admin Only] Update a specific user.
    """
    user = get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
        
    update_data = user_in.model_dump(exclude_unset=True)
    updated_user = update_user(db, user, update_data)
    return updated_user

@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT, dependencies=[Depends(RoleChecker(["admin"]))], tags=["Admin"])
async def delete_user_by_id(
    user_id: int,
    db: Session = Depends(get_db)
):
    """
    [Admin Only] Soft delete a specific user.
    """
    user = get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
        
    soft_delete_user(db, user)
    return None