from PyPDF2 import PdfReader
from app.services.chunking import chunk_text
from app.services.embedding_service import embed_texts
from app.api.chroma_client import get_collection 
from app.services.ocr_service import extract_text_from_pdf, extract_text_from_image
from app.services.transcription_service import transcribe_audio
import logging
from PIL import Image
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Helper function to determine file type ---
def get_file_type(filename: str) -> str:
    """Guesses file type based on extension."""
    extension = filename.split('.')[-1].lower()
    if extension == 'pdf':
        return 'pdf'
    if extension in ['png', 'jpg', 'jpeg', 'tiff', 'bmp']:
        return 'image'
    if extension in ['mp3', 'wav', 'm4a', 'mp4']:
        return 'audio'
    if extension in ['txt', 'md']:
        return 'text'
    return 'other'

# --- Main Service Logic ---
def process_document(saved_path: str, file_id: str, filename: str, user_id: str, user_role: str):
    """
    The core background task for processing an uploaded document.
    saved_path: The physical path on disk (likely named with UUID).
    file_id: The UUID to use for Database/Vector tagging.
    filename: The original filename (used for detecting type).
    """
    logger.info(f"[USER:{user_id}] Starting processing for {filename} (ID: {file_id})")
    
    try:
        text = ""
        file_type = get_file_type(filename) # Use original name for type detection

        # --- Route based on file type ---
        
        if file_type == 'pdf':
            logger.info(f"[USER:{user_id}] Processing as PDF...")
            try:
                text = extract_text_from_pdf(saved_path)
            except Exception as e:
                logger.warning(f"[USER:{user_id}] PDF OCR failed ({e}), falling back to text extraction.")
                reader = PdfReader(saved_path)
                text = "\n".join([page.extract_text() or "" for page in reader.pages])
        
        elif file_type == 'image':
            logger.info(f"[USER:{user_id}] Processing as Image for OCR...")
            try:
                text = extract_text_from_image(Image.open(saved_path))
            except Exception as e:
                logger.error(f"[USER:{user_id}] Image OCR failed: {e}")
                return
        
        elif file_type == 'audio':
            logger.info(f"[USER:{user_id}] Processing as Audio for Transcription...")
            try:
                text = transcribe_audio(saved_path)
            except Exception as e:
                logger.error(f"[USER:{user_id}] Audio transcription failed: {e}")
                return
        
        elif file_type == 'text':
            logger.info(f"[USER:{user_id}] Processing as Text...")
            try:
                with open(saved_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            except Exception as e:
                logger.error(f"[USER:{user_id}] Could not read {saved_path} as text. Error: {e}")
                return

        else:
            logger.warning(f"[USER:{user_id}] Skipping file: Unsupported file type '{filename}'")
            return

        if not text or text.isspace():
            logger.warning(f"[USER:{user_id}] No text extracted from {filename}.")
            return

        chunks = chunk_text(text)
        if not chunks:
            logger.warning(f"[USER:{user_id}] No chunks generated for {filename}.")
            return

        embs = embed_texts(chunks)

        # --- MULTI-TENANCY: Select Collection(s) based on Role ---
        # ALL users can access general_docs
        # Specialized roles get ADDITIONAL specialized collections
        collections_to_index = ["general_docs"]  # Default for ALL users
        
        if user_role == "lawyer": 
            collections_to_index.append("legal_docs")
        elif user_role == "doctor": 
            collections_to_index.append("medical_docs")
        elif user_role == "researcher": 
            collections_to_index.append("academic_docs")
        elif user_role == "student":
            collections_to_index.append("academic_docs")  # <-- ADD THIS BACK
        elif user_role == "banker": 
            collections_to_index.append("finance_docs")
        
        # Index to all applicable collections
        ids = [f"{file_id}_{i}" for i in range(len(chunks))]
        metas = [
            {"file_id": file_id, "user_id": user_id, "chunk_num": i, "filename": filename} 
            for i in range(len(chunks))
        ]
        
        for collection_name in collections_to_index:
            col = get_collection(collection_name)
            col.upsert(ids=ids, documents=chunks, embeddings=embs, metadatas=metas)
            logger.info(f"[USER:{user_id}] Successfully indexed {len(chunks)} chunks into '{collection_name}'.")

    except Exception as e:
        logger.error(f"[USER:{user_id}] CRITICAL: Failed processing {filename}. Error: {e}", exc_info=True)