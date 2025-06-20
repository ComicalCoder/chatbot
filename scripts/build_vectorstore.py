import os
import json
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

# Set up directories
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
FAISS_DIR = os.path.join(DATA_DIR, "faiss_index")

def load_json_files():
    """Load all parsed chunks from JSON files in data/"""
    documents = []
    print(f"📂 Processing JSON files in {DATA_DIR}")
    for filename in os.listdir(DATA_DIR):
        print(f"📄 Found file: {filename}")
        if filename.endswith(".json"):
            filepath = os.path.join(DATA_DIR, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    print(f"✅ Loaded {filename}, chunks: {len(data.get('chunks', []))}")
                    for chunk in data.get("chunks", []):
                        metadata = {"source": data.get("source", data.get("url", filename))}
                        documents.append(Document(page_content=chunk, metadata=metadata))
            except Exception as e:
                print(f"[❌ ERROR] Failed to load {filename}: {e}")
    print(f"📚 Total documents: {len(documents)}")
    return documents

def build_vector_store():
    """Embed documents and save FAISS vector index"""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    documents = load_json_files()
    if not documents:
        raise ValueError("No documents found in data directory")
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local(FAISS_DIR)
    return vector_store

def load_vector_store():
    """Load FAISS vector index from disk"""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)

def main():
    print("🔧 Building vector store...")
    vector_store = build_vector_store()
    print("✅ Vector store built and saved at:", FAISS_DIR)

if __name__ == "__main__":
    main()