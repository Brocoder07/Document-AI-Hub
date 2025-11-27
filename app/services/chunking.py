from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging

logger = logging.getLogger(__name__)

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """
    Structure-Aware Chunking:
    Uses Regex to respect numbered lists (1., 2., etc.) as primary split points.
    """
    if not text:
        return []

    # Regex Explanation:
    # 1. \n\n           -> Standard paragraph break
    # 2. \n(?=\d+\.)    -> Newline followed by a Number and a Dot (e.g., "1.", "2.")
    #                      The (?=...) is a "lookahead", so it splits at the newline
    #                      but keeps the number "1." at the start of the new chunk.
    # 3. \n             -> Standard line break
    # 4. " "            -> Word break
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        is_separator_regex=True,  # <--- ENABLE REGEX
        separators=[
            r"\n\n",           # Priority 1: Paragraphs
            r"\n(?=\d+\.)",    # Priority 2: Numbered Lists (1., 2., etc.)
            r"\n",             # Priority 3: Line breaks
            r" ",              # Priority 4: Words
            ""                 # Priority 5: Characters
        ]
    )

    chunks = text_splitter.split_text(text)
    
    logger.info(f"Split text into {len(chunks)} chunks using Regex-Recursive Splitter")
    
    return chunks