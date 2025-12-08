import logging
import docx
import os
import pandas as pd # Required for Excel processing

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

def read_excel_file(file_path: str) -> str:
    """
    Reads an Excel file and converts it to 'Contextual Row' text format.
    
    Why this format?
    Standard CSV/Markdown loses header context when chunked. 
    By converting each row to "ColName: Value", we ensure the LLM
    knows exactly what every number represents in every chunk.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        # Load the excel file
        # 'openpyxl' is required for .xlsx (ensure it's in requirements.txt)
        df = pd.read_excel(file_path, engine='openpyxl')
        
        # 1. Clean Data: Fill NaNs to prevent "nan" text confusion
        df = df.fillna("N/A")
        
        # 2. Convert to String for consistent processing
        df = df.astype(str)
        
        text_output = []
        columns = df.columns.tolist()
        
        # 3. Add High-Level Metadata
        text_output.append(f"Spreadsheet Summary: {len(df)} rows, {len(columns)} columns.")
        text_output.append(f"Column Headers: {', '.join(columns)}\n")
        text_output.append("-" * 30)
        
        # 4. Contextual Row Serialization
        # Transforms:  | John | 30 | 
        # To:          "Row 1: { Name: John, Age: 30 }"
        for index, row in df.iterrows():
            row_items = []
            for col in columns:
                val = row[col]
                # Skip columns with "N/A" to save tokens if they are empty, 
                # but you can keep them if "N/A" is meaningful.
                row_items.append(f"{col}: {val}")
            
            # Format as a structured object string
            row_str = f"Row {index + 1}: {{ " + ", ".join(row_items) + " }}"
            text_output.append(row_str)
            
        return "\n".join(text_output)

    except Exception as e:
        logger.error(f"Error reading Excel file: {e}")
        raise Exception(f"Failed to read Excel file: {str(e)}")