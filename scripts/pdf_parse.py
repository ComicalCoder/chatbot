import os
import glob
import json
import re
import pdfplumber
from pathlib import Path

def clean_text(text):
    """Clean text by removing extra whitespace and normalizing."""
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII characters
    return text

def extract_title(page):
    """Extract title from the first page using keywords and font properties."""
    text = page.extract_text() or ""
    lines = text.split('\n')[:6]  # Check first 6 lines for title
    title_parts = []
    for line in lines:
        line = clean_text(line)
        # Look for title keywords
        if any(keyword in line.upper() for keyword in ['PEKELILING', 'BIL.', 'PERKHIDMATAN']):
            # Remove codes like [JPAN: ...] and stop at section markers
            title_part = re.sub(r'\[.*?\]', '', line).strip()
            title_part = re.split(r'(TUJUAN|LATAR BELAKANG|SABAH MAJU JAYA)', title_part, flags=re.IGNORECASE)[0].strip()
            if title_part:
                title_parts.append(title_part)
        # Stop if we hit a section marker or empty line
        if re.search(r'^(TUJUAN|LATAR BELAKANG|\d+\.\s|$)', line, re.IGNORECASE):
            break
    
    title = ' '.join(title_parts).strip()
    
    # Fallback: Check for bold or large font
    if not title or len(title) < 10 or title.upper() in ['NEGERI SABAH', 'SABAH']:  # Avoid vague titles
        for char in page.chars:
            if char.get('size', 0) > 12 or 'Bold' in char.get('fontname', ''):  # Adjust threshold as needed
                title_text = clean_text(char['text'])
                if any(keyword in title_text.upper() for keyword in ['PEKELILING', 'BIL.', 'PERKHIDMATAN']):
                    title = title_text
                    break
        else:
            title = clean_text(lines[0]) if lines else "Untitled"
    
    return title if title else "Untitled"

def extract_tables(page):
    """Extract tables from a page and convert to structured format."""
    tables = []
    for table in page.extract_tables():
        if not table or len(table) == 0:
            continue
        # Clean headers, assign placeholders for empty ones
        headers = [clean_text(h) if h else f"Column_{i}" for i, h in enumerate(table[0])]
        # Filter headers: keep if non-placeholder or column has non-empty data
        valid_indices = [
            i for i, h in enumerate(headers)
            if h and not h.startswith("Column_") or any(row[i] and row[i].strip() for row in table[1:] if i < len(row))
        ]
        if not valid_indices:
            continue
        filtered_headers = [headers[i] for i in valid_indices]
        table_data = []
        for row in table[1:]:
            row_dict = {
                filtered_headers[i]: clean_text(row[j]) if j < len(row) and row[j] else ""
                for i, j in enumerate(valid_indices)
            }
            table_data.append(row_dict)
        tables.append(table_data)
    return tables

def extract_forms(page_text, page_num):
    """Extract form fields as key-value pairs with strict regex."""
    forms = {}
    if page_num == 1:  # Skip forms on page 1 to avoid title misdetection
        return forms
    lines = page_text.split('\n')
    # Exclude section headers and common non-form keywords
    exclude_keywords = {
        'NEGERI', 'LATAR', 'LATAR BELAKANG', 'TUJUAN', 'TANGGUNGJAWAB', 'BERTUGAS',
        'BERTUGAS RASMI', 'SABAH', 'PEKELILING', 'PERKHIDMATAN', 'TAFSIRAN', 'KADAR',
        'SYARAT', 'JADUAL'
    }
    for line in lines:
        # Match patterns like 'NAMA: value', 'JAWATAN value', or 'NAMA [value]'
        # Ensure value doesn't start with a number or section-like text
        match = re.match(r'^\s*(\d+\.\s+)?([A-Z\s]{1,15})\s*[:\s]+([^0-9\s][^\n[]+)$', line)
        if match:
            key = clean_text(match.group(2)).replace(' ', '')  # Normalize key for exclusion check
            if key in exclude_keywords:
                continue
            value = clean_text(match.group(3))
            if len(value.split()) > 50:  # Skip if value is too long (likely a paragraph)
                continue
            forms[clean_text(match.group(2))] = value
        else:
            # Try fields without colons (e.g., 'NAMA [value]')
            match = re.match(r'^\s*(\d+\.\s+)?([A-Z\s]{1,15})\s+\[([^\]]*)\]$', line)
            if match:
                key = clean_text(match.group(2)).replace(' ', '')  # Normalize key
                if key in exclude_keywords:
                    continue
                value = clean_text(match.group(3))
                if len(value.split()) > 50:
                    continue
                forms[clean_text(match.group(2))] = value
    return forms

def chunk_text(text):
    """Split text into paragraphs for LLM-friendliness."""
    paragraphs = [clean_text(p) for p in text.split('\n\n') if p.strip()]
    return paragraphs

def parse_pdf(pdf_path):
    """Parse a single PDF and return structured data."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            title = None
            pages_data = []
            for page_num, page in enumerate(pdf.pages, 1):
                # Extract text
                text = page.extract_text() or ""
                text = clean_text(text)
                
                # Extract title from first page
                if page_num == 1:
                    title = extract_title(page)
                
                # Extract tables
                tables = extract_tables(page)
                
                # Extract forms
                forms = extract_forms(text, page_num)
                
                # Chunk text into paragraphs
                paragraphs = chunk_text(text)
                
                # Structure page data
                page_data = {
                    "page_num": page_num,
                    "paragraphs": paragraphs,
                    "tables": tables,
                    "forms": forms
                }
                pages_data.append(page_data)
            
            # Add metadata
            metadata = {
                "page_count": len(pdf.pages),
                "file_size_bytes": os.path.getsize(pdf_path)
            }
            
            print(f"Parsed {pdf_path}: Title='{title}', Pages={len(pages_data)}, Tables={sum(len(p['tables']) for p in pages_data)}, Forms={sum(len(p['forms']) for p in pages_data)}")
            
            return {
                "filename": os.path.basename(pdf_path),
                "title": title or "Untitled",
                "metadata": metadata,
                "pages": pages_data
            }
    except Exception as e:
        print(f"Error parsing {pdf_path}: {str(e)}")
        return None

def main():
    """Parse all PDFs in input_files/ and save JSONs to data/."""
    input_dir = Path("~/chatbot/input_files").expanduser()
    output_dir = Path("~/chatbot/data").expanduser()
    
    # Ensure output directory exists
    output_dir.mkdir(exist_ok=True)
    
    # Find all PDFs
    pdf_files = glob.glob(str(input_dir / "*.pdf"))
    print(f"Found {len(pdf_files)} PDFs in {input_dir}")
    
    for pdf_path in pdf_files:
        print(f"Processing {pdf_path}...")
        parsed_data = parse_pdf(pdf_path)
        if parsed_data:
            # Generate JSON filename with 'pdf_' prefix
            json_filename = f"pdf_{os.path.splitext(os.path.basename(pdf_path))[0]}.json"
            json_path = output_dir / json_filename
            
            # Save JSON
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(parsed_data, f, ensure_ascii=False, indent=2)
            print(f"Saved parsed data to {json_path}")
        else:
            print(f"Skipped {pdf_path} due to parsing error")

if __name__ == "__main__":
    main()