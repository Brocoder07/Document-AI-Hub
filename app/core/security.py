import jwt
from jwt import PyJWTError
from passlib.context import CryptContext
from datetime import datetime, timedelta, timezone
from pydantic import BaseModel, ValidationError
from cryptography.fernet import Fernet
from typing import Any# <-- 1. IMPORT 'Any'

from app.core.config import settings

# Password hashing context
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")

# Pydantic schema for token payload
class TokenPayload(BaseModel):
    sub: str | None = None
    exp: int | None = None

cipher_suite = Fernet(settings.MESSAGE_ENCRYPTION_KEY)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plain password against a hashed one."""
    if len(plain_password.encode('utf-8')) > 72:
        plain_password = plain_password[:72]
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Hashes a plain password."""
    if len(password.encode('utf-8')) > 72:
        password = password[:72]
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

def encrypt_message(plain_text: str) -> str:
    """Encrypts raw text into a Fernet token string."""
    if not plain_text:
        return ""
    # Fernet encrypt expects bytes, returns bytes. We decode back to str for DB storage.
    return cipher_suite.encrypt(plain_text.encode('utf-8')).decode('utf-8')

def decrypt_message(cipher_text: str) -> str:
    """
    Decrypts a Fernet token string back to raw text.
    Handles legacy (unencrypted) data gracefully.
    """
    if not cipher_text:
        return ""
    try:
        # Try to decrypt
        return cipher_suite.decrypt(cipher_text.encode('utf-8')).decode('utf-8')
    except Exception:
        # Fallback: If decryption fails, assume it's old unencrypted text
        return cipher_text