"""
3GPP Document Splitter — Creates smaller, focused files from massive specs.

Takes the huge 3GPP .docx files and extracts the most important core sections
into small .txt files (~50-100KB each) that can be uploaded through the frontend.

Usage:
    python scripts/split_3gpp.py
"""

import os
import sys
import re
import docx

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ─── Key sections to extract from each spec ───
# These are the most important sections for a RAG chatbot to know about.
# We skip annexes, change history, definitions lists, etc.

SPEC_SECTIONS = {
    "TS_23.501": {
        "name": "5G System Architecture",
        "key_sections": [
            # Core architecture
            (4, "System Architecture"),
            (5, "Network Functions"),
            (6, "Network Slicing"),
            (7, "QoS"),
            (8, "UE Policy"),
            (9, "Registration and Connection"),
            (10, "Session Management"),
        ]
    },
    "TS_38.300": {
        "name": "NR and NG-RAN Overall Description",
        "key_sections": [
            (4, "NG-RAN Architecture"),
            (5, "Physical Layer"),
            (6, "Layer 2"),
            (7, "RRC"),
            (8, "NG Interface"),
            (9, "Xn Interface"),
            (10, "Security"),
            (11, "Mobility"),
            (12, "Scheduling"),
        ]
    },
    "TS_23.502": {
        "name": "Procedures for 5G System",
        "key_sections": [
            (4, "General Procedures"),
            (5, "Mobility Management"),
            (6, "Session Management"),
            (7, "Policy and Charging"),
        ]
    },
    "TS_29.500": {
        "name": "5GC APIs Technical Realization",
        "key_sections": [
            (4, "Overview"),
            (5, "API Design"),
            (6, "HTTP Messages"),
            (7, "Serialization"),
        ]
    },
}


def extract_sections_from_docx(file_path: str, target_sections: list) -> dict:
    """
    Read a .docx and extract only the paragraphs belonging to target top-level sections.
    
    target_sections: list of (section_num, label) tuples, e.g. [(4, "Architecture"), (5, "NFs")]
    
    Returns: dict of {section_num: text_content}
    """
    print(f"  Reading {os.path.basename(file_path)}...")
    doc = docx.Document(file_path)
    
    # Pattern to match top-level section headers like "4\tSystem Architecture" or "4 System Architecture"
    header_re = re.compile(r'^(\d+)(?:\.\d+)*[\t\s]+(.+)$')
    
    target_nums = {s[0] for s in target_sections}
    
    sections = {}
    current_section = None
    current_text = []
    
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            if current_section is not None:
                current_text.append("")
            continue
        
        match = header_re.match(text)
        if match:
            top_level_num = int(match.group(1))
            
            # Save previous section if it was a target
            if current_section is not None and current_section in target_nums:
                sections[current_section] = "\n".join(current_text)
            
            # Check if this new section is one we want
            if top_level_num in target_nums:
                current_section = top_level_num
                current_text = [text]
            else:
                # We've moved past our target sections
                if current_section is not None and current_section in target_nums:
                    sections[current_section] = "\n".join(current_text)
                current_section = top_level_num
                current_text = [text]
        else:
            if current_section in target_nums:
                current_text.append(text)
    
    # Don't forget the last section
    if current_section is not None and current_section in target_nums:
        sections[current_section] = "\n".join(current_text)
    
    return sections


def main():
    doc_dir = os.path.join("data", "documents")
    out_dir = os.path.join("data", "documents", "split")
    os.makedirs(out_dir, exist_ok=True)
    
    print("=" * 60)
    print("  3GPP DOCUMENT SPLITTER")
    print("  Creates small, focused .txt files from massive specs")
    print("=" * 60)
    
    total_files = 0
    
    for spec_id, spec_info in SPEC_SECTIONS.items():
        docx_path = os.path.join(doc_dir, f"{spec_id}.docx")
        
        # Also check for the raw downloaded filename format
        alt_path = None
        if not os.path.exists(docx_path):
            # Try finding by pattern
            for f in os.listdir(doc_dir):
                if spec_id.replace("_", "").replace(".", "").lower() in f.replace("-", "").replace(".", "").lower():
                    alt_path = os.path.join(doc_dir, f)
                    break
        
        actual_path = docx_path if os.path.exists(docx_path) else alt_path
        
        if not actual_path or not os.path.exists(actual_path):
            print(f"\n  [SKIP] {spec_id} - file not found at {docx_path}")
            continue
        
        print(f"\n--- {spec_id}: {spec_info['name']} ---")
        
        sections = extract_sections_from_docx(actual_path, spec_info["key_sections"])
        
        for sec_num, label in spec_info["key_sections"]:
            if sec_num not in sections:
                print(f"  [SKIP] Section {sec_num} ({label}) - not found in document")
                continue
            
            content = sections[sec_num]
            
            if len(content.strip()) < 100:
                print(f"  [SKIP] Section {sec_num} ({label}) - too short ({len(content)} chars)")
                continue
            
            # Add a clear header to help the RAG system
            header = f"3GPP {spec_id.replace('_', ' ')} - {spec_info['name']}\n"
            header += f"Section {sec_num}: {label}\n"
            header += "=" * 50 + "\n\n"
            
            full_content = header + content
            
            # Create filename
            safe_label = re.sub(r'[^\w\s-]', '', label).strip().replace(' ', '_')
            out_filename = f"{spec_id}_Sec{sec_num}_{safe_label}.txt"
            out_path = os.path.join(out_dir, out_filename)
            
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write(full_content)
            
            size_kb = len(full_content.encode('utf-8')) / 1024
            print(f"  [OK] Section {sec_num} ({label}): {size_kb:.0f} KB -> {out_filename}")
            total_files += 1
    
    print(f"\n{'=' * 60}")
    print(f"  DONE! Created {total_files} files in: {out_dir}")
    print(f"  You can now upload these through the frontend.")
    print(f"{'=' * 60}")
    
    # List all created files with sizes
    print(f"\n  Files created:")
    total_size = 0
    for f in sorted(os.listdir(out_dir)):
        if f.endswith('.txt'):
            size = os.path.getsize(os.path.join(out_dir, f))
            total_size += size
            print(f"    {f} ({size/1024:.0f} KB)")
    print(f"\n  Total size: {total_size/1024:.0f} KB (vs ~30 MB original)")


if __name__ == "__main__":
    main()
