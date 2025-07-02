import os
import json
import re
import unicodedata
import pdfplumber
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import io
import logging

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "input_files")
DATA_DIR = os.path.join(BASE_DIR, "data")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
os.makedirs(DATA_DIR, exist_ok=True)

# Setup logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def chunk_text(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Split large text into RAG-friendly chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    logger.debug(f"Created {len(chunks)} chunks for text: {text[:100]}...")
    return chunks

def clean_text(text):
    """Clean PDF text to remove noise."""
    text = re.sub(r"(?i)(SABAH\.?GOV\.?MY|JPKN|Page \d+|©.*|Dasar Privasi|Notis Penafian|\+6088.*|WAKTU PEJABAT|Hari Bekerja|Loading\.\.\.|SABAH MAJU JAYA|JABATAN PERKHIDMATAN KOMPUTER NEGERI|jpkn@sabah\.gov\.my)", "", text)
    text = re.sub(r"[©\(\)\*\!\|\[\]\{\}]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    logger.debug(f"Cleaned text: {text[:200]}...")
    return text

def normalize_text(text):
    """Normalize text: fix whitespace and Unicode issues."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_text_with_ocr(page):
    """Extract text from a page using OCR if standard extraction fails."""
    try:
        text = page.extract_text() or ""
        if text.strip():
            logger.debug(f"Extracted text from page {page.page_number}: {text[:100]}...")
            return text
        # Fall back to OCR
        logger.debug(f"Performing OCR on page {page.page_number}")
        image = page.to_image(resolution=300).original
        image = image.convert("L")  # Grayscale
        image = ImageEnhance.Contrast(image).enhance(3.0)  # Increase contrast
        image = image.filter(ImageFilter.MedianFilter(size=3))  # Reduce noise
        image = image.point(lambda x: 0 if x < 140 else 255)  # Binarize
        config = "--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789:.- "
        text = pytesseract.image_to_string(image, lang="eng+msa", config=config)
        logger.debug(f"OCR text from page {page.page_number}: {text[:100]}...")
        return text.strip()
    except Exception as e:
        logger.error(f"OCR failed on page {page.page_number}: {e}")
        return ""

def extract_tables(page):
    """Extract tables from a page."""
    try:
        tables = page.extract_tables()
        table_data = []
        for table in tables:
            for row in table:
                if len(row) >= 2:  # Adjust based on PDF structure
                    cleaned_row = [normalize_text(cell) if cell else "" for cell in row]
                    table_data.append(cleaned_row)
        if table_data:
            logger.debug(f"Extracted {len(table_data)} table rows on page {page.page_number}")
        return table_data
    except Exception as e:
        logger.error(f"Table extraction failed on page {page.page_number}: {e}")
        return []

def detect_headings_and_content(page):
    """Detect headings and group content, tailored for structured PDFs."""
    content_by_heading = []
    current_heading = {"text": "No Heading", "content": []}
    try:
        words = page.extract_words() or []
        text = extract_text_with_ocr(page).split("\n")
        word_idx = 0
        for line in text:
            line = line.strip()
            if not line:
                continue
            # Heuristic for headings: font size, bold, uppercase, or numbered
            is_heading = False
            while word_idx < len(words) and words[word_idx]["text"].strip() in line:
                font_size = words[word_idx].get("size", 0)
                font_name = words[word_idx].get("font", "").lower()
                if font_size > 12 or "bold" in font_name:
                    is_heading = True
                    break
                word_idx += 1
            # Additional heuristic: uppercase or numbered sections
            if (line.isupper() and len(line) > 2) or re.match(r"^\d+\.\d+\.?$|^\d+\.\s", line):
                is_heading = True
            if is_heading:
                if current_heading["content"]:
                    content_by_heading.append(current_heading)
                current_heading = {"text": line, "content": []}
            else:
                current_heading["content"].append(line)
        if current_heading["content"]:
            content_by_heading.append(current_heading)
    except Exception as e:
        logger.error(f"Heading detection failed on page {page.page_number}: {e}")
        text = extract_text_with_ocr(page)
        if text.strip():
            content_by_heading.append({"text": "No Heading", "content": text.split("\n")})
    return content_by_heading

def parse_pdf(filepath):
    """Extract text, tables, headings, and metadata from a PDF."""
    try:
        logger.info(f"Processing PDF: {filepath}")
        with pdfplumber.open(filepath) as pdf:
            text_sections = []
            table_data = []
            all_content_by_heading = []
            metadata = pdf.metadata or {}

            for page in pdf.pages:
                # Extract text
                text = extract_text_with_ocr(page)
                if text:
                    text_sections.append(normalize_text(text))
                
                # Extract tables
                page_tables = extract_tables(page)
                table_data.extend(page_tables)
                
                # Detect headings and content
                page_content = detect_headings_and_content(page)
                if page_content:
                    all_content_by_heading.extend(page_content)

            full_text = "\n".join(text_sections).strip()
            if not full_text:
                logger.warning(f"No extractable content in {filepath}")
                return None

            cleaned_text = clean_text(full_text)
            chunks = chunk_text(cleaned_text)

            # Merge content by heading
            merged_content = {}
            for item in all_content_by_heading:
                heading = item["text"]
                content = "\n".join([c for c in item["content"] if c.strip()])
                if heading in merged_content:
                    merged_content[heading] += "\n" + content
                else:
                    merged_content[heading] = content
            structured_content = [
                {"heading": k, "content": v} for k, v in merged_content.items() if v.strip()
            ]

            return {
                "source": os.path.basename(filepath),
                "title": os.path.splitext(os.path.basename(filepath))[0],
                "content": cleaned_text,
                "chunks": chunks,
                "table_data": table_data if table_data else None,
                "structured_content": structured_content if structured_content else None,
                "metadata": metadata if metadata else None
            }
    except Exception as e:
        logger.error(f"Failed to parse {filepath}: {e}")
        return None

def save_json(data, output_path):
    """Save parsed data to a JSON file."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    logger.info(f"Saved JSON to {output_path}")

def main():
    """Process all PDFs in INPUT_DIR and save to DATA_DIR."""
    pdf_files = list(Path(INPUT_DIR).glob("*.pdf"))
    if not pdf_files:
        logger.error("No PDFs found in input_files/ directory")
        return

    for pdf_path in pdf_files:
        data = parse_pdf(pdf_path)
        if data:
            safe_name = pdf_path.stem.replace(" ", "_").replace("?", "").replace("&", "")
            output_path = os.path.join(DATA_DIR, f"{safe_name}.json")
            save_json(data, output_path)

if __name__ == "__main__":
    main()