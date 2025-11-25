import re
import logging

logger = logging.getLogger(__name__)

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """
    Robust Chunking:
    1. Cleans text (fixes '45,000perannum' issue).
    2. Respects word boundaries (prevents splitting words in half).
    """
    if not text:
        return []

    # --- STEP 1: SMART CLEANING ---
    # Replace all newlines, tabs, and multiple spaces with a single space.
    # This prevents the "joined words" issue from PDF/OCR line breaks.
    clean_text = re.sub(r'\s+', ' ', text).strip()

    if len(clean_text) <= chunk_size:
        return [clean_text]

    # --- STEP 2: WORD-AWARE SPLITTING ---
    chunks = []
    start = 0
    text_len = len(clean_text)

    while start < text_len:
        # Define the hard cutoff point
        end = start + chunk_size
        
        # If we reached the end, add the remainder
        if end >= text_len:
            chunks.append(clean_text[start:])
            break
            
        # Find the last space within the limit to avoid splitting a word
        # We search backwards from 'end'
        segment = clean_text[start:end]
        last_space_relative = segment.rfind(' ')
        
        if last_space_relative == -1:
            # Case: A single word is longer than chunk_size (unlikely, but safe to handle)
            # Force split at limit
            chunks.append(clean_text[start:end])
            start += (chunk_size - overlap)
        else:
            # Clean split at the space
            cut_point = start + last_space_relative
            chunks.append(clean_text[start:cut_point])
            
            # Move start pointer, ensuring we overlap correctly
            # We want the next chunk to start 'overlap' characters *before* the current cut
            start = max(start + 1, cut_point - overlap)
            
            # Sanity check: ensure we don't get stuck in a loop if overlap >= chunk_size
            if start <= start: 
                 start = cut_point + 1 # Force forward progress if overlap logic fails

    return chunks