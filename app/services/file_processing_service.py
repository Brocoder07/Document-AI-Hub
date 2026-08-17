"""
File Processing Service — Production Grade

Handles document ingestion pipeline:
1. Text Extraction (PDF, DOCX, Image OCR, Audio, Excel)
2. 3GPP-Aware Chunking with section metadata
3. BGE Embedding (document-mode, no query prefix)
4. Multi-collection Weaviate Indexing with rich metadata
"""

import time
import logging
from PyPDF2 import PdfReader
from PIL import Image

# --- Internal Imports ---
from app.services.chunking import chunk_text, chunk_text_3gpp, is_3gpp_document
from app.services.embedding_service import embed_texts
from app.services.ocr_service import extract_text_from_pdf, extract_text_from_image
from app.services.transcription_service import transcribe_audio
from app.services.document_reader import read_text_file, read_docx_file, read_excel_file
from app.services.document_service import update_document_status
from app.db.session import SessionLocal
from app.api.vector_db import db_client 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_file_type(filename: str) -> str:
    extension = filename.split('.')[-1].lower()
    if extension == 'pdf': return 'pdf'
    if extension in ['png', 'jpg', 'jpeg', 'tiff', 'bmp']: return 'image'
    if extension in ['mp3', 'wav', 'm4a', 'mp4']: return 'audio'
    if extension in ['txt', 'md']: return 'text'
    if extension in ['docx', 'doc']: return 'docx'
    if extension in ['xlsx', 'xls']: return 'excel'
    return 'other'

def process_document(saved_path: str, file_id: str, filename: str, user_id: str, user_role: str):
    start_time = time.time()
    logger.info(f" [Start] Processing {filename} | ID: {file_id} | User: {user_id}")
    
    db = SessionLocal()
    
    try:
        update_document_status(db, file_id, "processing")

        text = ""
        file_type = get_file_type(filename)
        
        # --- PHASE 1: EXTRACTION ---
        ext_start = time.time()
        
        if file_type == 'pdf':
            logger.info(f" [PDF] Attempting direct text extraction...")
            try:
                reader = PdfReader(saved_path)
                text = "\n".join([page.extract_text() or "" for page in reader.pages])
            except Exception as e:
                logger.warning(f" Direct extraction failed: {e}")
            
            # Fallback to OCR if text is sparse
            if not text or len(text.strip()) < 50:
                logger.info(f" [OCR] Text sparse/empty. Switching to OCR...")
                text = extract_text_from_pdf(saved_path)
            else:
                logger.info(f" [PDF] Direct extraction success ({len(text)} chars). Skipping OCR.")

        elif file_type == 'image':
            text = extract_text_from_image(Image.open(saved_path))
        elif file_type == 'audio':
            text = transcribe_audio(saved_path)
        elif file_type == 'text':
            text = read_text_file(saved_path, user_id)
        elif file_type == 'docx':
            text = read_docx_file(saved_path)
        elif file_type == 'excel':
            logger.info(f"[USER:{user_id}] Processing as Excel Spreadsheet...")
            try:
                text = read_excel_file(saved_path)
            except Exception as e:
                logger.error(f"[USER:{user_id}] Excel processing failed: {e}")
                update_document_status(db, file_id, "failed")
                return
        else:
            logger.warning(f" Unsupported type: {filename}")
            update_document_status(db, file_id, "failed")
            return

        logger.info(f" [Timing] Extraction took {round(time.time() - ext_start, 2)}s")

        if not text or text.isspace():
            logger.warning(" No text extracted.")
            update_document_status(db, file_id, "failed")
            return

        # --- PHASE 2: CHUNKING (3GPP-Aware) ---
        chunk_start = time.time()
        
        is_telecom = is_3gpp_document(text, filename)
        
        if is_telecom:
            logger.info(f" [3GPP Detected] Using telecom-aware chunking for {filename}")
            chunks, chunk_metas = chunk_text_3gpp(text, filename)
        else:
            logger.info(f" [General Doc] Using standard chunking for {filename}")
            chunks = chunk_text(text, filename=filename)
            chunk_metas = [
                {"section_id": "", "section_title": "", "parent_context": "", "spec_number": ""}
                for _ in chunks
            ]
        
        if not chunks:
            logger.warning(" No chunks generated.")
            update_document_status(db, file_id, "failed")
            return
        
        logger.info(f" Generated {len(chunks)} chunks in {round(time.time() - chunk_start, 2)}s")

        # --- PHASE 3: EMBEDDING (Document mode — no query prefix) ---
        emb_start = time.time()
        embs = embed_texts(chunks, is_query=False)
        logger.info(f" [Timing] Embedding took {round(time.time() - emb_start, 2)}s")

        # --- PHASE 4: INDEXING WITH RICH METADATA ---
        idx_start = time.time()
        
        # Collection Strategy — 3GPP docs go to telecom + general
        collections_to_index = ["general_docs"]
        
        if is_telecom:
            collections_to_index.append("telecom_docs")
            logger.info(f" [Indexing] 3GPP doc will be indexed in: telecom_docs + general_docs")
        
        # Role-based collections (keep existing functionality)
        if user_role in ["lawyer"]: collections_to_index.append("legal_docs")
        elif user_role in ["doctor", "medical"]: collections_to_index.append("medical_docs")
        elif user_role in ["researcher", "student", "academic"]: collections_to_index.append("academic_docs")
        elif user_role in ["banker", "financial_analyst"]: collections_to_index.append("finance_docs")
        elif user_role in ["employee", "executive", "business"]: collections_to_index.append("business_docs") 
        
        ids = [f"{file_id}_{i}" for i in range(len(chunks))]
        metas = []
        for i in range(len(chunks)):
            meta = {
                "file_id": file_id, 
                "user_id": user_id, 
                "chunk_num": i, 
                "filename": filename,
                # 3GPP-specific metadata
                "section_id": chunk_metas[i].get("section_id", ""),
                "section_title": chunk_metas[i].get("section_title", ""),
                "spec_number": chunk_metas[i].get("spec_number", ""),
            }
            metas.append(meta)
        
        for collection_name in collections_to_index:
            db_client.upsert(
                collection_name=collection_name,
                ids=ids, 
                documents=chunks, 
                embeddings=embs, 
                metadatas=metas
            )
            logger.info(f" Indexed in '{collection_name}'")

        logger.info(f" [Timing] Indexing took {round(time.time() - idx_start, 2)}s")
        update_document_status(db, file_id, "completed")
        
        total_time = round(time.time() - start_time, 2)
        logger.info(f" [COMPLETED] {filename} | {len(chunks)} chunks | {total_time}s total")

    except Exception as e:
        logger.error(f" [CRITICAL] Failed: {e}", exc_info=True)
        update_document_status(db, file_id, "failed")
    finally:
        db.close()