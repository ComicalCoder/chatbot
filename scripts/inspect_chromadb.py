import os
import chromadb
from chromadb.utils import embedding_functions
from langchain_core.documents import Document

# --- CONFIGURATION (Must match your other scripts) ---
CHROMA_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")
COLLECTION_NAME = "jpkn_website_content"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def get_embedding_function():
    """Returns the configured SentenceTransformer embedding function."""
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

def main():
    print(f"Attempting to connect to ChromaDB at: {CHROMA_DB_PATH}")
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        embedding_function = get_embedding_function()
        
        # Get the collection. If it doesn't exist, this will raise an error or create an empty one.
        # Use get_or_create_collection for robustness, but it should exist if create_vector_store.py ran.
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_function,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"Successfully connected to collection: {collection.name}")
        print(f"Number of documents in collection: {collection.count()}")

        if collection.count() == 0:
            print("Collection is empty. Please ensure create_vector_store.py ran successfully and added documents.")
            return

        print("\n--- Fetching a few sample documents and their metadata ---")
        # Fetch a few documents to inspect their metadata directly
        # We'll try to fetch documents that should contain "Direktori | JPKN" page_title
        # and also some general documents.

        # First, try a query that should hit the directory page
        print("\n--- Querying for 'Direktori | JPKN' page_title (expecting results) ---")
        try:
            # We'll use a simple query text and a filter to see if it works here
            # This is a test of the filter itself
            results_with_filter = collection.query(
                query_texts=["direktori jpkn"], # A simple query text
                n_results=5,
                where={"page_title": {"$eq": "Direktori | JPKN"}},
                include=['documents', 'metadatas']
            )
            if results_with_filter and results_with_filter['documents'] and results_with_filter['documents'][0]:
                print(f"Found {len(results_with_filter['documents'][0])} documents with page_title 'Direktori | JPKN':")
                for i in range(len(results_with_filter['documents'][0])):
                    doc_content = results_with_filter['documents'][0][i]
                    doc_metadata = results_with_filter['metadatas'][0][i]
                    print(f"  --- Document {i+1} ---")
                    print(f"    Content (first 100 chars): {doc_content[:100]}...")
                    print(f"    Metadata: {doc_metadata}")
            else:
                print("No documents found with page_title 'Direktori | JPKN' using $eq filter.")
        except Exception as e:
            print(f"[ERROR] Error querying with specific page_title filter: {e}")
            print("This might indicate an issue with the filter syntax or metadata values.")


        print("\n--- Querying for general documents (expecting results) ---")
        try:
            # Query without any filters to see if any documents are returned at all
            results_no_filter = collection.query(
                query_texts=["JPKN"], # A very general query
                n_results=5,
                include=['documents', 'metadatas']
            )
            if results_no_filter and results_no_filter['documents'] and results_no_filter['documents'][0]:
                print(f"Found {len(results_no_filter['documents'][0])} general documents:")
                for i in range(len(results_no_filter['documents'][0])):
                    doc_content = results_no_filter['documents'][0][i]
                    doc_metadata = results_no_filter['metadatas'][0][i]
                    print(f"  --- Document {i+1} ---")
                    print(f"    Content (first 100 chars): {doc_content[:100]}...")
                    print(f"    Metadata: {doc_metadata}")
            else:
                print("No general documents found even without filters. This would be a critical issue.")
        except Exception as e:
            print(f"[ERROR] Error querying without filters: {e}")


    except Exception as e:
        print(f"[CRITICAL ERROR] Failed to connect to ChromaDB or get collection: {e}")
        print("Please ensure the CHROMA_DB_PATH is correct and the database was created successfully.")


if __name__ == "__main__":
    main()
