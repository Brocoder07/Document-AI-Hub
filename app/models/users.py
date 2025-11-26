from sqlalchemy import Column, Integer, String, Boolean, DateTime
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.db.base_class import Base # You will need to create this base_class

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    
    # User details
    username = Column(String(100), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    full_name = Column(String(100), nullable=True)
    
    # Security
    hashed_password = Column(String(255), nullable=False)
    
    # Status and Roles
    is_active = Column(Boolean(), default=True)
    # This 'role' column matches your 'UserInDB' Pydantic model
    role = Column(String(50), default="user", nullable=False) 
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    documents = relationship("Document", back_populates="owner", cascade="all, delete-orphan")