import logging
import docx
import os

logger = logging.getLogger(__name__)

def read_text_file(file_path: str, user_id: str = "system") -> str:
    """
    Reads a text file with robust encoding fallback (UTF-8 -> CP1252 -> Latin-1).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        # Try UTF-8 first (Standard)
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        # Fallback to CP1252 (Common on Windows for smart quotes, etc.)
        logger.warning(f"[USER:{user_id}] UTF-8 decode failed. Retrying with cp1252...")
        try:
            with open(file_path, 'r', encoding='cp1252') as f:
                return f.read()
        except Exception:
            # Last resort: Latin-1 (Reads bytes directly, never fails but might garble chars)
            logger.warning(f"[USER:{user_id}] CP1252 failed. Retrying with latin-1...")
            with open(file_path, 'r', encoding='latin-1') as f:
                return f.read()
    except Exception as e:
        raise Exception(f"Failed to read text file: {str(e)}")

def read_docx_file(file_path: str) -> str:
    """
    Reads a .docx file and extracts text from paragraphs.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        raise Exception(f"Failed to read DOCX file: {str(e)}")