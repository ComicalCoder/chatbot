import os
import json
import hashlib
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
import unicodedata

# --- CONFIGURATION ---
# Directory where your PDF files are located
INPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "input_files")
# Directory where the parsed JSON output will be saved
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

# Text splitting configuration (consistent with website_parse.py)
TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)

# --- HELPER FUNCTIONS ---

def chunk_text(text, splitter=TEXT_SPLITTER):
    """Split text into RAG-friendly chunks using the configured splitter."""
    if not text or not text.strip():
        return []
    chunks = splitter.split_text(text)
    return [c.strip() for c in chunks if c.strip()]

def normalize_text(text):
    """Normalize text: fix whitespace and Unicode issues."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def save_documents_as_json(original_filepath, list_of_documents):
    """Save a list of structured documents to a JSON file.
    Filename is based on the original PDF name and a hash for uniqueness."""
    if not list_of_documents:
        print(f"  [Info] No documents to save for {original_filepath}.")
        return

    # Create a consistent filename based on the original PDF filename and its content hash
    base_filename = os.path.basename(original_filepath)
    # Remove extension and sanitize for filename
    name_without_ext = os.path.splitext(base_filename)[0]
    sanitized_name = re.sub(r'[^\w\s-]', '', name_without_ext).strip()
    sanitized_name = re.sub(r'[-\s]+', '-', sanitized_name) # Replace spaces/multiple dashes with single dash

    # Use a hash of the file path and content to ensure uniqueness
    file_hash = hashlib.md5(original_filepath.encode('utf-8')).hexdigest()

    output_filename = f"{sanitized_name}_{file_hash}.json"
    filepath = os.path.join(OUTPUT_DIR, output_filename)

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(list_of_documents, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved {len(list_of_documents)} documents from '{base_filename}' to '{filepath}'")
    except Exception as e:
        print(f"[ERROR] Failed to save JSON for '{base_filename}': {e}")

# --- MAIN PDF PARSING LOGIC ---

def parse_pdf(filepath):
    """
    Parses a PDF file, extracts text page by page, chunks it,
    and returns a list of structured documents.
    """
    documents = []
    pdf_name = os.path.basename(filepath)
    print(f"📄 Processing PDF: '{pdf_name}'")

    try:
        reader = PdfReader(filepath)
        num_pages = len(reader.pages)
        full_text = []

        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                full_text.append(normalize_text(text))
            print(f"  Extracted text from page {i+1}/{num_pages}")

        combined_text = "\n\n".join(full_text)

        if not combined_text.strip():
            print(f"  [Warning] No readable text extracted from '{pdf_name}'. Skipping.")
            return []

        # Chunk the combined text
        text_chunks = chunk_text(combined_text)

        for idx, chunk_content in enumerate(text_chunks):
            documents.append({
                "page_content": chunk_content,
                "metadata": {
                    "source_file": pdf_name,
                    "original_path": filepath,
                    "type": "pdf_content",
                    "chunk_index": idx,
                    "total_chunks": len(text_chunks),
                    "num_pages": num_pages,
                    "title": os.path.splitext(pdf_name)[0] # Use filename as title
                }
            })
        print(f"  [Info] Generated {len(documents)} document chunks from '{pdf_name}'.")
    except Exception as e:
        print(f"[ERROR] Failed to parse PDF '{pdf_name}': {e}")
        return []

    return documents

# --- MAIN EXECUTION ---

def main():
    # Ensure input and output directories exist
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Starting PDF parsing from '{INPUT_DIR}'...")

    pdf_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith('.pdf')]

    if not pdf_files:
        print(f"No PDF files found in '{INPUT_DIR}'. Please place your PDFs there.")
        return

    for pdf_file in pdf_files:
        pdf_filepath = os.path.join(INPUT_DIR, pdf_file)
        documents = parse_pdf(pdf_filepath)
        save_documents_as_json(pdf_filepath, documents)

    print("PDF parsing finished.")

if __name__ == "__main__":
    main()
