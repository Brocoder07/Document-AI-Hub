"""
Direct 3GPP Document Ingestion Script

Bypasses the frontend upload entirely. Reads the huge .docx files directly,
extracts key sections, chunks them, embeds them, and indexes into Weaviate.

Features:
- Progress bar for each step
- Batch embedding (50 chunks at a time) to avoid memory issues
- Selective section extraction (skips annexes, references, change history)
- Estimated time remaining

Usage:
    python scripts/ingest_3gpp.py
    python scripts/ingest_3gpp.py --file data/documents/TS_23.501.docx
    python scripts/ingest_3gpp.py --max-sections 50
"""

import os
import sys
import re
import time
import argparse
import logging
import uuid
import docx

# Fix Windows terminal encoding — prevents crash on emoji in log messages
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.chunking import chunk_text_3gpp, is_3gpp_document
from app.services.embedding_service import embed_texts
from app.api.vector_db import db_client

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────
# Sections to SKIP — these add noise, not knowledge
# ────────────────────────────────────────────────────
SKIP_PATTERNS = [
    r'^Annex\s+[A-Z]',           # Annex A, B, C...
    r'^Change\s+[Hh]istory',     # Change history tables
    r'^Foreword',                 # Boilerplate
    r'^Introduction$',           # Usually boilerplate  
    r'^References$',             # Bibliography
    r'^Definitions',             # Definitions (keep abbreviations though)
    r'^Abbreviations$',          # Just a list
    r'^History$',                # Version history
    r'^Contents$',               # Table of contents
    r'^List of',                 # List of figures/tables
]

SKIP_REGEX = re.compile('|'.join(SKIP_PATTERNS), re.IGNORECASE)


def read_docx_streaming(file_path: str, skip_boilerplate: bool = True) -> str:
    """
    Read a .docx file paragraph-by-paragraph (memory efficient).
    Optionally skip boilerplate sections.
    """
    logger.info(f"📖 Reading {os.path.basename(file_path)}...")
    
    doc = docx.Document(file_path)
    paragraphs = []
    skip_mode = False
    skipped_sections = []
    
    section_header_re = re.compile(r'^(\d+(?:\.\d+)*\.?|[A-Z]\.\d+(?:\.\d+)*\.?)\s+(.+)$')
    
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            paragraphs.append("")  # preserve spacing
            continue
        
        # Check if this is a section header
        header_match = section_header_re.match(text)
        if header_match and skip_boilerplate:
            title = header_match.group(2).strip()
            if SKIP_REGEX.match(title):
                skip_mode = True
                skipped_sections.append(text)
                continue
            else:
                skip_mode = False
        
        if not skip_mode:
            paragraphs.append(text)
    
    if skipped_sections:
        logger.info(f"⏩ Skipped {len(skipped_sections)} boilerplate sections:")
        for s in skipped_sections[:5]:
            logger.info(f"   - {s}")
        if len(skipped_sections) > 5:
            logger.info(f"   ... and {len(skipped_sections) - 5} more")
    
    full_text = "\n".join(paragraphs)
    logger.info(f"📝 Extracted {len(full_text):,} characters ({len(paragraphs):,} paragraphs)")
    return full_text


def batch_embed(chunks: list[str], batch_size: int = 50) -> list:
    """
    Embed chunks in batches to avoid memory issues with large documents.
    Shows progress for each batch.
    """
    all_embeddings = []
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        
        logger.info(f"   🔢 Embedding batch {batch_num}/{total_batches} ({len(batch)} chunks)...")
        start = time.time()
        embeddings = embed_texts(batch, is_query=False)
        elapsed = time.time() - start
        logger.info(f"   ✅ Batch {batch_num} done in {elapsed:.1f}s")
        
        all_embeddings.extend(embeddings)
    
    return all_embeddings


def batch_upsert(collection_name: str, ids: list, chunks: list, embeddings: list, metas: list, batch_size: int = 100):
    """
    Upsert into Weaviate in batches.
    """
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    
    for i in range(0, len(chunks), batch_size):
        end = min(i + batch_size, len(chunks))
        batch_num = (i // batch_size) + 1
        
        logger.info(f"   💾 Indexing batch {batch_num}/{total_batches} ({end - i} objects)...")
        db_client.upsert(
            collection_name=collection_name,
            ids=ids[i:end],
            documents=chunks[i:end],
            embeddings=embeddings[i:end],
            metadatas=metas[i:end],
        )
    
    logger.info(f"   ✅ All {len(chunks)} objects indexed in '{collection_name}'")


def ingest_file(file_path: str, user_id: str = "system", max_sections: int = None):
    """
    Full ingestion pipeline for a single 3GPP document.
    """
    filename = os.path.basename(file_path)
    file_id = f"3gpp_{filename.replace('.', '_')}_{int(time.time())}"
    
    logger.info(f"\n{'='*60}")
    logger.info(f"📡 INGESTING: {filename}")
    logger.info(f"{'='*60}")
    total_start = time.time()
    
    # ── Step 1: Extract Text ──
    step_start = time.time()
    
    if file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        logger.info(f"Read {len(text):,} chars from .txt file")
    else:
        text = read_docx_streaming(file_path, skip_boilerplate=True)
    
    if not text or text.isspace():
        logger.error(f"No text extracted from {filename}")
        return False
    
    logger.info(f"Extraction: {time.time() - step_start:.1f}s")
    
    # ── Step 2: Chunk ──
    step_start = time.time()
    chunks, chunk_metas = chunk_text_3gpp(text, filename)
    
    if max_sections and len(chunks) > max_sections:
        logger.info(f"✂️  Limiting from {len(chunks)} to {max_sections} chunks (--max-sections)")
        chunks = chunks[:max_sections]
        chunk_metas = chunk_metas[:max_sections]
    
    logger.info(f"🧩 Generated {len(chunks)} chunks in {time.time() - step_start:.1f}s")
    
    # ── Step 3: Embed (in batches) ──
    step_start = time.time()
    logger.info(f"🔢 Embedding {len(chunks)} chunks...")
    embeddings = batch_embed(chunks, batch_size=50)
    logger.info(f"⏱️  Embedding: {time.time() - step_start:.1f}s")
    
    # ── Step 4: Index into Weaviate ──
    step_start = time.time()
    ids = [f"{file_id}_{i}" for i in range(len(chunks))]
    metas = []
    for i in range(len(chunks)):
        metas.append({
            "file_id": file_id,
            "user_id": user_id,
            "chunk_num": i,
            "filename": filename,
            "section_id": chunk_metas[i].get("section_id", ""),
            "section_title": chunk_metas[i].get("section_title", ""),
            "spec_number": chunk_metas[i].get("spec_number", ""),
        })
    
    # Index into both telecom_docs and general_docs
    logger.info(f"💾 Indexing into Weaviate...")
    batch_upsert("telecom_docs", ids, chunks, embeddings, metas)
    batch_upsert("general_docs", ids, chunks, embeddings, metas)
    
    logger.info(f"⏱️  Indexing: {time.time() - step_start:.1f}s")
    
    total_time = time.time() - total_start
    logger.info(f"\n✅ COMPLETED: {filename}")
    logger.info(f"   📊 {len(chunks)} chunks | {total_time:.1f}s total")
    logger.info(f"   📍 Indexed in: telecom_docs, general_docs")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Ingest 3GPP documents directly into Weaviate")
    parser.add_argument("--file", type=str, help="Path to a specific .docx file to ingest")
    parser.add_argument("--dir", type=str, default="data/documents", help="Directory containing .docx files")
    parser.add_argument("--user-id", type=str, default="system", help="User ID for metadata")
    parser.add_argument("--max-sections", type=int, default=None, help="Max chunks per document (for testing)")
    parser.add_argument("--batch-size", type=int, default=50, help="Embedding batch size")
    args = parser.parse_args()
    
    print("=" * 60)
    print("📡 3GPP DIRECT INGESTION PIPELINE")
    print("   Bypasses frontend — processes large docs locally")
    print("=" * 60)
    
    if args.file:
        # Process single file
        if not os.path.exists(args.file):
            logger.error(f"File not found: {args.file}")
            sys.exit(1)
        ingest_file(args.file, args.user_id, args.max_sections)
    else:
        # Process all .txt and .docx files in directory
        doc_dir = args.dir
        if not os.path.isdir(doc_dir):
            logger.error(f"Directory not found: {doc_dir}")
            sys.exit(1)
        
        docx_files = [f for f in os.listdir(doc_dir) if f.endswith('.docx') or f.endswith('.txt')]
        
        if not docx_files:
            logger.error(f"No .docx or .txt files found in {doc_dir}")
            sys.exit(1)
        
        logger.info(f"\nFound {len(docx_files)} documents to ingest:")
        for f in docx_files:
            size_mb = os.path.getsize(os.path.join(doc_dir, f)) / (1024 * 1024)
            logger.info(f"   - {f} ({size_mb:.1f} MB)")
        
        success = 0
        failed = 0
        
        for f in docx_files:
            file_path = os.path.join(doc_dir, f)
            try:
                if ingest_file(file_path, args.user_id, args.max_sections):
                    success += 1
                else:
                    failed += 1
            except Exception as e:
                logger.error(f"💥 Failed to ingest {f}: {e}", exc_info=True)
                failed += 1
        
        print("\n" + "=" * 60)
        print(f"📊 INGESTION SUMMARY")
        print(f"   ✅ Success: {success}")
        print(f"   ❌ Failed:  {failed}")
        print("=" * 60)
    
    # Clean up Weaviate connection
    db_client.close()


if __name__ == "__main__":
    main()
