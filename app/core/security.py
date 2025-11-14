import jwt
from jwt import PyJWTError
from passlib.context import CryptContext
from datetime import datetime, timedelta, timezone
from pydantic import BaseModel, ValidationError
from typing import Any # <-- 1. IMPORT 'Any'

from app.core.config import settings

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Pydantic schema for token payload
class TokenPayload(BaseModel):
    sub: str | None = None
    exp: int | None = None

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plain password against a hashed one."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Hashes a plain password."""
    return pwd_context.hash(password)

def create_access_token(
    subject: str | Any, expires_delta: timedelta | None = None # <-- 2. CHANGE 'any' to 'Any'
) -> str:
    """
    Creates a new JWT access token.
    """
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode = {"exp": expire, "sub": str(subject)}
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

def decode_token(token: str) -> TokenPayload | None:
    """
    Decodes a JWT token and returns its payload.
    Returns None if the token is invalid or expired.
    """
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        # Use Pydantic model to validate payload structure
        token_data = TokenPayload(**payload)
        
        # Check if token is expired
        if token_data.exp is None or datetime.now(timezone.utc) > datetime.fromtimestamp(token_data.exp, tz=timezone.utc):
            return None # Token expired
            
    except (PyJWTError, ValidationError):
        return None # Invalid token
    
    return token_data