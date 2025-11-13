from fastapi import APIRouter, UploadFile, File, Form
from PyPDF2 import PdfReader
from app.common_utils.file_handler import save_upload
from app.services.chunking import chunk_text
from app.services.embedding_service import embed_texts
from app.api.utility.chroma_client import get_collection
from app.services.rag_service import answer_query
from app.services.ocr_service import extract_text_from_pdf

router = APIRouter()

@router.post("/upload/")
def upload(file: UploadFile = File(...)):
    saved_path = save_upload(file.file, file.filename)

    if file.filename.lower().endswith(".pdf"):
        # OCR for scanned PDF
        try:
            text = extract_text_from_pdf(saved_path)
        except:
            # fallback to text-based PDF extraction
            reader = PdfReader(saved_path)
            text = "\n".join([page.extract_text() or "" for page in reader.pages])
    else:
        text = saved_path

    chunks = chunk_text(text)
    embs = embed_texts(chunks)

    col = get_collection("documents")

    ids = [f"{file.filename}_{i}" for i in range(len(chunks))]
    metas = [{"file_id": file.filename, "chunk": i} for i in range(len(chunks))]

    col.add(ids=ids, documents=chunks, embeddings=embs, metadatas=metas)

    return {"status": "indexed", "chunks": len(chunks)}

@router.post("/query/")
def query(q: str = Form(...)):
    return answer_query(q)
