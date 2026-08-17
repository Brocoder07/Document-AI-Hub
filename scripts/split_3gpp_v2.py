"""
3GPP Document Splitter v2 — Extracts ALL core content, splits by size.

Instead of trying to match specific sections, this:
1. Reads the entire .docx
2. Skips boilerplate (Annex, Change History, Foreword, ToC)
3. Splits remaining content into ~100KB .txt files
4. Names each file based on the spec + part number

Usage:
    python scripts/split_3gpp_v2.py
"""

import os
import sys
import re
import docx

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Sections to SKIP entirely (boilerplate that adds noise)
SKIP_SECTIONS = {
    "foreword", "introduction", "scope", "references",
    "change history", "history", "contents",
    "list of figures", "list of tables",
    "intellectual property", "copyright",
}

# Annex pattern — skip Annex A, B, C, etc.
ANNEX_RE = re.compile(r'^Annex\s+[A-Z]', re.IGNORECASE)

# Top-level section header (e.g. "4 System Architecture" or "4\tSystem Architecture")
TOP_SECTION_RE = re.compile(r'^(\d+)\s+(.+)$')


def should_skip_section(title: str) -> bool:
    """Check if a section title is boilerplate to skip."""
    title_lower = title.lower().strip()
    if any(skip in title_lower for skip in SKIP_SECTIONS):
        return True
    if ANNEX_RE.match(title):
        return True
    return False


def extract_core_content(file_path: str) -> str:
    """
    Read docx, skip boilerplate sections, return core technical content.
    """
    print(f"  Reading {os.path.basename(file_path)}...")
    doc = docx.Document(file_path)
    
    paragraphs = []
    skip_mode = False
    skipped = []
    
    for para in doc.paragraphs:
        text = para.text.strip()
        
        if not text:
            if not skip_mode:
                paragraphs.append("")
            continue
        
        # Check if this looks like a top-level section header
        top_match = TOP_SECTION_RE.match(text)
        annex_match = ANNEX_RE.match(text)
        
        if top_match:
            title = top_match.group(2).strip()
            # Remove any trailing numbers/tabs from title (page numbers etc)
            title_clean = re.sub(r'\t.*$', '', title).strip()
            
            if should_skip_section(title_clean):
                skip_mode = True
                skipped.append(text[:60])
                continue
            else:
                skip_mode = False
        elif annex_match:
            skip_mode = True
            skipped.append(text[:60])
            continue
        
        if not skip_mode:
            paragraphs.append(text)
    
    if skipped:
        print(f"  Skipped {len(skipped)} boilerplate sections")
    
    content = "\n".join(paragraphs)
    
    # Remove excessive blank lines
    content = re.sub(r'\n{4,}', '\n\n\n', content)
    
    print(f"  Extracted {len(content):,} chars from {len(paragraphs):,} paragraphs")
    return content


def split_into_files(content: str, spec_id: str, spec_name: str, out_dir: str, 
                     target_size_kb: int = 100) -> int:
    """
    Split content into files of approximately target_size_kb each.
    Tries to split at section boundaries.
    """
    target_size = target_size_kb * 1024  # Convert to bytes
    
    # Split at major section boundaries (lines starting with a number)
    section_re = re.compile(r'\n(?=\d+(?:\.\d+)*\s+[A-Z])')
    parts = section_re.split(content)
    
    files_created = 0
    current_chunk = ""
    current_part = 1
    
    for part in parts:
        # If adding this part would exceed target, save current and start new
        if len(current_chunk.encode('utf-8')) + len(part.encode('utf-8')) > target_size and current_chunk:
            files_created += save_chunk(current_chunk, spec_id, spec_name, current_part, out_dir)
            current_part += 1
            current_chunk = part
        else:
            current_chunk += ("\n" if current_chunk else "") + part
    
    # Save the last chunk
    if current_chunk.strip():
        files_created += save_chunk(current_chunk, spec_id, spec_name, current_part, out_dir)
    
    return files_created


def save_chunk(content: str, spec_id: str, spec_name: str, part_num: int, out_dir: str) -> int:
    """Save a chunk to a .txt file with a descriptive header."""
    if len(content.strip()) < 50:
        return 0
    
    # Try to find the first section title for a meaningful name
    first_section = ""
    for line in content.split('\n'):
        match = re.match(r'^(\d+(?:\.\d+)*)\s+(.+)$', line.strip())
        if match:
            first_section = match.group(2).strip()[:40]
            break
    
    # Add header
    header = f"3GPP {spec_id.replace('_', ' ')} - {spec_name}\n"
    header += f"Part {part_num}"
    if first_section:
        header += f" (starting from: {first_section})"
    header += "\n" + "=" * 60 + "\n\n"
    
    full_content = header + content
    
    # Filename
    safe_section = re.sub(r'[^\w -]', '', first_section).strip().replace(' ', '_')[:30] if first_section else ""
    suffix = f"_{safe_section}" if safe_section else ""
    filename = f"{spec_id}_Part{part_num}{suffix}.txt"
    filepath = os.path.join(out_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(full_content)
    
    size_kb = os.path.getsize(filepath) / 1024
    print(f"    Part {part_num}: {size_kb:.0f} KB -> {filename}")
    return 1


def main():
    doc_dir = os.path.join("data", "documents")
    out_dir = os.path.join("data", "documents", "split")
    os.makedirs(out_dir, exist_ok=True)
    
    # Clean old splits
    for f in os.listdir(out_dir):
        if f.endswith('.txt'):
            os.remove(os.path.join(out_dir, f))
    
    print("=" * 60)
    print("  3GPP DOCUMENT SPLITTER v2")
    print("  Extracts all core content, splits into ~100KB files")
    print("=" * 60)
    
    specs = [
        ("TS_23.501", "5G System Architecture"),
        ("TS_38.300", "NR and NG-RAN Overall Description"),
        ("TS_23.502", "Procedures for 5G System"),
        ("TS_29.500", "5GC APIs Technical Realization"),
    ]
    
    total_files = 0
    
    for spec_id, spec_name in specs:
        docx_path = os.path.join(doc_dir, f"{spec_id}.docx")
        
        if not os.path.exists(docx_path):
            print(f"\n  [SKIP] {spec_id} - not found")
            continue
        
        size_mb = os.path.getsize(docx_path) / (1024 * 1024)
        print(f"\n--- {spec_id}: {spec_name} ({size_mb:.1f} MB) ---")
        
        content = extract_core_content(docx_path)
        
        if len(content.strip()) < 100:
            print(f"  [SKIP] No usable content extracted")
            continue
        
        files = split_into_files(content, spec_id, spec_name, out_dir, target_size_kb=100)
        total_files += files
    
    # Summary
    print(f"\n{'=' * 60}")
    print(f"  DONE! Created {total_files} files in: {out_dir}")
    print(f"{'=' * 60}")
    
    total_size = 0
    for f in sorted(os.listdir(out_dir)):
        if f.endswith('.txt'):
            size = os.path.getsize(os.path.join(out_dir, f))
            total_size += size
            print(f"  {f} ({size/1024:.0f} KB)")
    
    print(f"\n  Total: {total_size/1024:.0f} KB ({total_files} files)")
    print(f"  Upload these via the frontend Document Library page.")


if __name__ == "__main__":
    main()
