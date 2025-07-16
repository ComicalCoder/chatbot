import os
import json
import hashlib
import re
import unicodedata
import shutil # Ensure shutil is imported for rmtree

import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document # Import Document for consistency

# --- CONFIGURATION ---
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
CHROMA_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")
COLLECTION_NAME = "jpkn_website_content" # Must match context_retrieval.py
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" # Must match context_retrieval.py

# Text splitting configuration (consistent with parsing scripts)
TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)

# --- HELPER FUNCTIONS ---

def normalize_text(text: str) -> str:
    """Normalize text: fix whitespace and Unicode issues."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def sanitize_metadata(metadata: dict) -> dict:
    """
    Sanitizes metadata dictionary to ensure all values are simple types
    (str, int, float, bool) for ChromaDB compatibility.
    Converts lists to comma-separated strings.
    --- KEY CHANGE: Convert None to empty string ---
    """
    sanitized = {}
    for key, value in metadata.items():
        if value is None:
            sanitized[key] = "" # Convert None to empty string
        elif isinstance(value, list):
            sanitized[key] = ", ".join(map(str, value))
        elif isinstance(value, (str, int, float, bool)):
            sanitized[key] = value
        else:
            sanitized[key] = str(value) # Convert any other complex types to string
    return sanitized

def load_documents_from_json(data_directory: str) -> list[Document]:
    """
    Loads all JSON documents from the specified directory.
    Each JSON file is expected to contain a list of dictionaries,
    where each dictionary represents a document chunk with 'page_content' and 'metadata'.
    """
    all_documents = []
    print(f"Loading documents from '{data_directory}'...")
    
    for filename in os.listdir(data_directory):
        if filename.lower().endswith('.json'):
            filepath = os.path.join(data_directory, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for doc_dict in data:
                            if 'page_content' in doc_dict and isinstance(doc_dict['page_content'], str):
                                if 'metadata' not in doc_dict or not isinstance(doc_dict['metadata'], dict):
                                    doc_dict['metadata'] = {}
                                
                                # Ensure metadata is sanitized before creating Document object
                                sanitized_meta = sanitize_metadata(doc_dict['metadata'])
                                
                                # Create LangChain Document object
                                all_documents.append(Document(
                                    page_content=normalize_text(doc_dict['page_content']),
                                    metadata=sanitized_meta
                                ))
                                # --- DEBUGGING: Print loaded document content ---
                                if "ERNYWATI DEWI ABAS" in doc_dict['page_content']:
                                    print(f"  [DEBUG] Found 'ERNYWATI DEWI ABAS' in document from {filename}:")
                                    print(f"    Content: {doc_dict['page_content']}")
                                    print(f"    Metadata: {doc_dict['metadata']}")
                                # --- END DEBUGGING ---
                            else:
                                print(f"[Warning] Skipping malformed document in {filename}: Missing 'page_content' or not a string.")
                    else:
                        print(f"[Warning] Skipping {filename}: Expected a list of documents, found {type(data)}.")
            except json.JSONDecodeError as e:
                print(f"[ERROR] Failed to decode JSON from {filename}: {e}")
            except Exception as e:
                print(f"[ERROR] Error reading {filename}: {e}")
    
    print(f"Loaded {len(all_documents)} document chunks in total.")
    return all_documents

# --- MAIN EXECUTION ---

def main():
    # Ensure Chroma DB directory exists
    os.makedirs(CHROMA_DB_PATH, exist_ok=True)

    # --- KEY CHANGE: Delete existing ChromaDB before re-creating using direct client ---
    if os.path.exists(CHROMA_DB_PATH):
        print(f"Deleting existing Chroma DB at '{CHROMA_DB_PATH}'...")
        try:
            shutil.rmtree(CHROMA_DB_PATH)
            print("Existing Chroma DB deleted.")
        except OSError as e:
            print(f"[ERROR] Error deleting directory {CHROMA_DB_PATH}: {e}. Please ensure no other process is using it.")
            return

    # Initialize ChromaDB client and collection
    print("Initializing ChromaDB client and collection...")
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"} # Consistent with context_retrieval.py
    )
    print("ChromaDB client and collection initialized.")

    # Load documents from the data directory
    documents_to_embed = load_documents_from_json(DATA_DIR)

    if not documents_to_embed:
        print("No documents found to create the vector store. Please run parsing scripts first.")
        return

    print(f"Adding {len(documents_to_embed)} documents to Chroma collection '{COLLECTION_NAME}'...")
    
    # Prepare data for ChromaDB's add method
    texts = [doc.page_content for doc in documents_to_embed]
    metadatas = [doc.metadata for doc in documents_to_embed]
    
    # Generate unique IDs for each document
    ids = []
    for i, doc in enumerate(documents_to_embed):
        # Use a combination of URL hash and chunk index for a robust unique ID
        url_hash = hashlib.md5(doc.metadata.get('url', '').encode()).hexdigest()
        chunk_identifier = doc.metadata.get('chunk_index', i) # Use existing chunk_index or simple index
        unique_id = f"{url_hash}-{chunk_identifier}-{i}" # Add 'i' for guaranteed uniqueness if chunk_index is not unique
        ids.append(hashlib.md5(unique_id.encode()).hexdigest())

    try:
        collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Successfully added {len(documents_to_embed)} documents to Chroma collection.")
        print(f"Vector store should now be saved to: {CHROMA_DB_PATH}")
    except Exception as e:
        print(f"[CRITICAL ERROR] Failed to add documents to Chroma collection: {e}")

if __name__ == "__main__":
    main()
