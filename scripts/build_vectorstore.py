import os
import json
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
FAISS_DIR = os.path.join(DATA_DIR, "faiss_index")

def load_json_files():
    """Load all parsed chunks, tables, and structured content from JSON files in data/"""
    documents = []
    print(f"📂 Processing JSON files in {DATA_DIR}")
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".json"):
            filepath = os.path.join(DATA_DIR, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                print(f"✅ Loaded {filename}, chunks: {len(data.get('chunks', []))}")
                
                # Determine source
                source = data.get("source", data.get("url", filename))
                
                # Add chunks
                for chunk in data.get("chunks", []):
                    documents.append(Document(page_content=chunk, metadata={"source": source, "type": "chunk"}))
                
                # Add table_data
                if data.get("table_data"):
                    table_text = " ".join([str(row) for row in data["table_data"]])
                    documents.append(Document(page_content=table_text, metadata={"source": source, "type": "table"}))
                    print(f"📋 Added table_data from {filename}")
                
                # Add structured_content (PDF) or structured_data (web)
                if data.get("structured_content"):
                    for item in data["structured_content"]:
                        content = f"{item['heading']}: {item['content']}"
                        documents.append(Document(page_content=content, metadata={"source": source, "type": "structured"}))
                    print(f"🗂️ Added structured_content from {filename}")
                elif data.get("structured_data"):
                    for item in data["structured_data"]:
                        content = f"{item['header']}: {item['content']}"
                        documents.append(Document(page_content=content, metadata={"source": source, "type": "structured"}))
                    print(f"🗂️ Added structured_data from {filename}")
                    
            except Exception as e:
                print(f"[❌ ERROR] Failed to load {filename}: {e}")
    print(f"📚 Total documents: {len(documents)}")
    return documents

def build_vector_store():
    """Embed documents and save FAISS vector index"""
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        documents = load_json_files()
        if not documents:
            raise ValueError("No documents found in data directory")
        vector_store = FAISS.from_documents(documents, embeddings)
        os.makedirs(FAISS_DIR, exist_ok=True)
        vector_store.save_local(FAISS_DIR)
        print(f"✅ Vector store built and saved at: {FAISS_DIR}")
        return vector_store
    except Exception as e:
        print(f"[❌ ERROR] Failed to build vectorstore: {e}")
        raise

def main():
    print("🔧 Building vector store...")
    build_vector_store()

if __name__ == "__main__":
    main()