import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "faiss_index"), embeddings, allow_dangerous_deserialization=True)

def query_vectorstore(query, vector_store, top_k=3):
    docs = vector_store.similarity_search(query, k=top_k)
    return [(doc.page_content, doc.metadata["source"]) for doc in docs]

def main():
    vector_store = load_vector_store()
    query = "What services does JPKN Sabah provide?"
    results = query_vectorstore(query, vector_store)
    print(f"\nQuery: {query}")
    for i, (content, source) in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"Source: {source}")
        print(f"Content: {content[:200]}...")

if __name__ == "__main__":
    main()