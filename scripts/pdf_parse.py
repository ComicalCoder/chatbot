import os
import json
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Define input and output directories
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "input_files")
DATA_DIR = os.path.join(BASE_DIR, "data")

def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    """Split large text into RAG-friendly chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    return text_splitter.split_text(text)

def parse_pdf(filepath):
    """Extract text and chunks from a PDF."""
    try:
        with pdfplumber.open(filepath) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages).strip()
            if not text:
                print(f"[SKIP] No extractable text in {filepath}")
                return None
            chunks = chunk_text(text)
            return {
                "source": os.path.basename(filepath),
                "title": os.path.splitext(os.path.basename(filepath))[0],
                "content": text,
                "chunks": chunks
            }
    except Exception as e:
        print(f"[ERROR] Failed to parse {filepath}: {e}")
        return None

def save_json(data, output_path):
    """Save parsed data to a JSON file."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"[✅ SAVED] {output_path}")

def process_files():
    """Parse all supported files in INPUT_DIR and save to DATA_DIR."""
    os.makedirs(DATA_DIR, exist_ok=True)
    for filename in os.listdir(INPUT_DIR):
        filepath = os.path.join(INPUT_DIR, filename)
        ext = os.path.splitext(filename)[1].lower()
        if not os.path.isfile(filepath):
            continue
        if ext == ".pdf":
            print(f"[📄 PDF] Parsing {filename}...")
            data = parse_pdf(filepath)
        else:
            print(f"[SKIP] Unsupported file type: {filename}")
            continue

        if data:
            safe_name = os.path.splitext(filename)[0].replace(" ", "_").replace("?", "").replace("&", "")
            output_path = os.path.join(DATA_DIR, f"{safe_name}.json")
            save_json(data, output_path)

if __name__ == "__main__":
    process_files()
