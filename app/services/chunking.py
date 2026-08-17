"""
3GPP-Aware Intelligent Chunking Service

Designed specifically for 3GPP specification documents (TS/TR series).
Key features:
1. Respects 3GPP section numbering (4.2.3.1, A.1.2, etc.) as split boundaries
2. Preserves tables, figures, and cross-references within chunks
3. Injects parent section context into each chunk for LLM comprehension
4. Larger chunk sizes (1500 chars) for dense telecom content
5. Falls back to general-purpose chunking for non-3GPP documents
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
import logging
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)

# Regex patterns for 3GPP document structure
# Matches section headers like "4.2.3.1 Registration Procedure" or "A.1 Annex"
SECTION_HEADER_PATTERN = re.compile(
    r'^(\d+(?:\.\d+)*\.?|[A-Z]\.\d+(?:\.\d+)*\.?)\s+(.+)$',
    re.MULTILINE
)

# Matches 3GPP spec references like "TS 23.501", "TR 38.913"
SPEC_REFERENCE_PATTERN = re.compile(
    r'(?:TS|TR)\s*\d{2}\.\d{3}',
    re.IGNORECASE
)

# Matches clause cross-references like "clause 4.2.3" or "subclause 6.1.2.3"
CLAUSE_REF_PATTERN = re.compile(
    r'(?:clause|subclause|section)\s+(\d+(?:\.\d+)*)',
    re.IGNORECASE
)


def extract_section_hierarchy(text: str) -> List[Dict]:
    """
    Extract the section hierarchy from a 3GPP document.
    Returns a list of {section_id, title, start_pos, end_pos} dicts.
    """
    sections = []
    for match in SECTION_HEADER_PATTERN.finditer(text):
        section_id = match.group(1).rstrip('.')
        title = match.group(2).strip()
        sections.append({
            "section_id": section_id,
            "title": title,
            "start_pos": match.start(),
        })
    
    # Calculate end positions
    for i in range(len(sections)):
        if i + 1 < len(sections):
            sections[i]["end_pos"] = sections[i + 1]["start_pos"]
        else:
            sections[i]["end_pos"] = len(text)
    
    return sections


def get_parent_context(section_id: str, sections: List[Dict]) -> str:
    """
    Build a breadcrumb trail for a given section.
    e.g., "4.2.3.1" -> "4 System Architecture > 4.2 Registration > 4.2.3 Procedures > 4.2.3.1 Initial Registration"
    """
    parts = section_id.split('.')
    breadcrumbs = []
    
    for i in range(1, len(parts) + 1):
        parent_id = '.'.join(parts[:i])
        for sec in sections:
            if sec["section_id"] == parent_id:
                breadcrumbs.append(f"{sec['section_id']} {sec['title']}")
                break
    
    return " > ".join(breadcrumbs) if breadcrumbs else ""


def detect_spec_number(text: str, filename: str = "") -> str:
    """
    Try to detect the 3GPP spec number from the document content or filename.
    e.g., "TS 23.501" from "3GPP TS 23.501 V17.6.0"
    """
    # Check filename first
    spec_match = SPEC_REFERENCE_PATTERN.search(filename)
    if spec_match:
        return spec_match.group(0).upper().replace(" ", " ")
    
    # Check first 2000 chars of content
    spec_match = SPEC_REFERENCE_PATTERN.search(text[:2000])
    if spec_match:
        return spec_match.group(0).upper().replace(" ", " ")
    
    return "Unknown"


def is_3gpp_document(text: str, filename: str = "") -> bool:
    """
    Heuristic to detect if a document is a 3GPP specification.
    """
    indicators = [
        "3GPP", "3rd Generation Partnership Project",
        "TS 2", "TS 3", "TR 2", "TR 3",
        "ETSI", "Technical Specification",
        "eNB", "gNB", "UE ", "AMF", "SMF", "UPF",
        "E-UTRAN", "NG-RAN", "NR ", "LTE ",
    ]
    
    # Check filename
    fn_upper = filename.upper()
    if any(ind.upper() in fn_upper for ind in ["TS", "TR", "3GPP"]):
        return True
    
    # Check content (first 3000 chars)
    header = text[:3000].upper()
    matches = sum(1 for ind in indicators if ind.upper() in header)
    return matches >= 2  # At least 2 indicators


def chunk_text_3gpp(text: str, filename: str = "", chunk_size: int = 1500, overlap: int = 300) -> Tuple[List[str], List[Dict]]:
    """
    3GPP-aware chunking that preserves document structure.
    
    Returns:
        Tuple of (chunks, chunk_metadata_list)
        Each metadata dict contains: section_id, section_title, parent_context, spec_number
    """
    if not text:
        return [], []
    
    spec_number = detect_spec_number(text, filename)
    sections = extract_section_hierarchy(text)
    
    chunks = []
    chunk_metas = []
    
    if sections:
        logger.info(f" [3GPP Chunking] Found {len(sections)} sections in {spec_number}")
        
        for section in sections:
            section_text = text[section["start_pos"]:section["end_pos"]].strip()
            
            if not section_text:
                continue
            
            # Build context prefix for this section
            parent_ctx = get_parent_context(section["section_id"], sections)
            context_prefix = f"[{spec_number}] {parent_ctx}\n\n" if parent_ctx else f"[{spec_number}]\n\n"
            
            # If section fits in one chunk, keep it whole
            if len(section_text) <= chunk_size:
                full_chunk = context_prefix + section_text
                chunks.append(full_chunk)
                chunk_metas.append({
                    "section_id": section["section_id"],
                    "section_title": section["title"],
                    "parent_context": parent_ctx,
                    "spec_number": spec_number,
                })
            else:
                # Split large sections using recursive splitter but keep context prefix
                sub_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size - len(context_prefix),
                    chunk_overlap=overlap,
                    length_function=len,
                    is_separator_regex=True,
                    separators=[
                        r"\n\n",                    # Paragraphs
                        r"\n(?=\d+(?:\.\d+)*\.?\s)",  # 3GPP sub-section numbers
                        r"\n(?=-\s)",               # Bullet points
                        r"\n(?=NOTE\s)",             # 3GPP NOTE markers
                        r"\n",                      # Line breaks
                        r" ",                       # Words
                        "",                         # Characters
                    ]
                )
                
                sub_chunks = sub_splitter.split_text(section_text)
                for sc in sub_chunks:
                    full_chunk = context_prefix + sc
                    chunks.append(full_chunk)
                    chunk_metas.append({
                        "section_id": section["section_id"],
                        "section_title": section["title"],
                        "parent_context": parent_ctx,
                        "spec_number": spec_number,
                    })
    else:
        # No sections found — fallback to paragraph-based splitting with spec context
        logger.info(f" [3GPP Chunking] No sections detected. Using paragraph-based splitting.")
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=len,
            is_separator_regex=True,
            separators=[
                r"\n\n",
                r"\n(?=\d+(?:\.\d+)*\.?\s)",
                r"\n(?=-\s)",
                r"\n",
                r" ",
                "",
            ]
        )
        
        raw_chunks = splitter.split_text(text)
        for rc in raw_chunks:
            prefix = f"[{spec_number}]\n\n" if spec_number != "Unknown" else ""
            chunks.append(prefix + rc)
            chunk_metas.append({
                "section_id": "",
                "section_title": "",
                "parent_context": "",
                "spec_number": spec_number,
            })
    
    logger.info(f" [3GPP Chunking] Produced {len(chunks)} chunks from {spec_number}")
    return chunks, chunk_metas


def chunk_text(text: str, chunk_size: int = 1500, overlap: int = 300, filename: str = "") -> list[str]:
    """
    Main entry point for chunking. Auto-detects 3GPP documents and applies
    appropriate strategy.
    
    For backwards compatibility, returns just the chunks list.
    Use chunk_text_3gpp() directly if you need metadata.
    """
    if not text:
        return []
    
    # Auto-detect and use 3GPP-aware chunking
    if is_3gpp_document(text, filename):
        logger.info(f" [Chunking] 3GPP document detected! Using telecom-aware chunking.")
        chunks, _ = chunk_text_3gpp(text, filename, chunk_size, overlap)
        return chunks
    
    # General-purpose chunking for non-3GPP documents
    logger.info(f" [Chunking] General document. Using standard recursive splitting.")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        is_separator_regex=True,
        separators=[
            r"\n\n",           # Priority 1: Paragraphs
            r"\n(?=\d+\.)",    # Priority 2: Numbered Lists (1., 2., etc.)
            r"\n",             # Priority 3: Line breaks
            r" ",              # Priority 4: Words
            ""                 # Priority 5: Characters
        ]
    )

    chunks = text_splitter.split_text(text)
    logger.info(f" [Chunking] Split text into {len(chunks)} chunks")
    
    return chunks