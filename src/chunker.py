import fitz
import json
import re
from pathlib import Path

RAW_DIR = Path("data/raw_acts")
OUTPUT = Path("data/chunks.jsonl")

# Map filename to act details
ACTS = {
    "consumer_protection_2019.pdf": ("Consumer Protection Act, 2019", "consumer", "India", 2019),
    "payment_of_wages_1936.pdf": ("Payment of Wages Act, 1936", "employment", "India", 1936),
    "it_act_2000.pdf": ("Information Technology Act, 2000", "cyber", "India", 2000),
    "rti_act_2005.pdf": ("Right to Information Act, 2005", "RTI", "India", 2005),
    "sexual_harassment_2013.pdf": ("Sexual Harassment of Women at Workplace Act, 2013", "workplace_harassment", "India", 2013),
    "rent_control_maharashtra.pdf": ("Maharashtra Rent Control Act, 1999", "tenancy", "Maharashtra", 1999),
    "rent_control_delhi.pdf": ("Delhi Rent Control Act, 1958", "tenancy", "Delhi", 1958),
    "rent_control_karnataka.pdf": ("Karnataka Rent Control Act, 1961", "tenancy", "Karnataka", 1961),
}

def extract_clean_text(pdf_path):
    """Return plain text without page numbers or short headers."""
    doc = fitz.open(pdf_path)
    all_text = ""
    for page in doc:
        # Get text blocks as plain text (ignore images)
        page_text = page.get_text("text")
        all_text += page_text + "\n"
    doc.close()
    # Remove lines that are just numbers (page numbers)
    lines = all_text.split("\n")
    cleaned = [l.strip() for l in lines if not l.strip().isdigit()]
    return "\n".join(cleaned)

def chunk_by_sections(text):
    """
    Split text at section headers like '1. Short title...' or 'Section 1 - ...'.
    Returns list of (section_number, section_text).
    """
    # Pattern: optional "Section" + digits + . or - or space + rest
    pattern = r'(?:Section\s+)?(\d+)[\.\s\-]+'
    # Split using regex but keep the delimiter
    parts = re.split(pattern, text)
    chunks = []
    # parts will be: [text_before, num1, text_after_num1, num2, text_after_num2, ...]
    # Skip first empty part if text starts with a section
    i = 0
    while i < len(parts):
        try:
            # parts[i] is the section number (digits), parts[i+1] is the content
            num = parts[i]
            if num.isdigit():
                content = parts[i+1] if i+1 < len(parts) else ""
                chunks.append((num, content.strip()))
                i += 2
            else:
                # not a section number, advance
                i += 1
        except IndexError:
            break
    if not chunks:
        chunks.append(("0", text))
    return chunks

def main():
    all_chunks = []
    for pdf_file in RAW_DIR.glob("*.pdf"):
        fname = pdf_file.name
        if fname not in ACTS:
            print(f"Skipping {fname} — no metadata.")
            continue
        act_name, category, jurisdiction, year = ACTS[fname]
        print(f"Processing {fname}...")
        text = extract_clean_text(pdf_file)
        sections = chunk_by_sections(text)
        for sec_num, sec_text in sections:
            chunk = {
                "act_name": act_name,
                "section_number": sec_num,
                "jurisdiction": jurisdiction,
                "category": category,
                "year": year,
                "text": sec_text
            }
            all_chunks.append(chunk)
        print(f"  {len(sections)} sections found.")

    # Save
    with open(OUTPUT, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"Saved {len(all_chunks)} chunks to {OUTPUT}")

if __name__ == "__main__":
    main()