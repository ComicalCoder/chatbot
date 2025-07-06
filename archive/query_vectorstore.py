import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    return FAISS.load_local(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "faiss_index"), embeddings, allow_dangerous_deserialization=True)

def query_vectorstore(query, vector_store, top_k=7):
    docs = vector_store.similarity_search(query, k=top_k)
    return [(doc.page_content, doc.metadata["source"]) for doc in docs]

def main():
    try:
        vector_store = load_vector_store()
        queries = [
            "What services does JPKN Sabah provide?",
            "What is in JPAN_700-28(44).pdf?"
        ]
        for query in queries:
            print(f"\nQuery: {query}")
            results = query_vectorstore(query, vector_store)
            for i, (content, source) in enumerate(results, 1):
                print(f"\nResult {i}:")
                print(f"Source: {source}")
                print(f"Content: {content[:200]}...")
    except Exception as e:
        print(f"[ERROR] Failed to run query: {e}")

if __name__ == "__main__":
    main()