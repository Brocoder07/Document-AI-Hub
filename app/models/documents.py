from sqlalchemy import Column, Integer, String, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.db.base_class import Base

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    # The unique ID on disk (e.g., "uuid_filename.pdf")
    file_id = Column(String, unique=True, index=True, nullable=False) 
    # The original name (e.g., "My Resume.pdf")
    filename = Column(String, nullable=False) 
    file_type = Column(String, nullable=False) # pdf, mp3, png
    file_path = Column(String, nullable=False) # Full path on disk
    file_size = Column(Integer, default=0)     # Size in bytes
    
    content_hash = Column(String, index=True, nullable=False)
    upload_date = Column(DateTime(timezone=True), server_default=func.now())
    
    user_id = Column(Integer, ForeignKey("users.id"))
    owner = relationship("User", back_populates="documents")