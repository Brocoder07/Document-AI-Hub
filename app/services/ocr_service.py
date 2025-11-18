import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import os
from app.core.config import settings

# Set specific Tesseract path (Windows)
pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_PATH

def extract_text_from_image(image: Image.Image) -> str:
    return pytesseract.image_to_string(image)

def extract_text_from_pdf(pdf_path: str) -> str:
    # FIX: Pass poppler_path to the function
    pages = convert_from_path(
        pdf_path, 
        dpi=200, 
        output_folder=settings.OCR_TEMP_DIR,
        poppler_path=settings.POPPLER_PATH # <--- ADD THIS ARGUMENT
    )
    text = ""
    for page in pages:
        text += pytesseract.image_to_string(page) + "\n"
    return text