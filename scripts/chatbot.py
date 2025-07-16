import os
import re # For language detection regex
# --- KEY FIX: Re-import ChatPromptTemplate ---
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
# Import necessary components from context_retrieval
from context_retrieval import get_relevant_documents, ChromaDBClient, CHROMA_DB_PATH
from langchain_community.chat_models import ChatOllama # Re-import ChatOllama

# --- LANGUAGE DETECTION ---
MALAY_KEYWORDS = [
    "siapa", "apa", "bila", "mana", "kenapa", "bagaimana", "berapa", "adakah", 
    "waktu", "pejabat", "jam", "masa", "kerja", "buka", "tutup", "operasi", 
    "jawatan", "pengarah", "timbalan", "ketua", "pegawai", 
    "jabatan", "negeri", "malaysia", "sabah", "hubungi", "telefon", "nombor", "alamat"
]
MALAY_REGEX = re.compile(r'(?i)\b(?:' + '|'.join(re.escape(k) for k in MALAY_KEYWORDS) + r')\b')

def detect_language(text: str) -> str:
    """
    Detects the primary language of the text (English or Malay) based on keywords.
    Defaults to English if mixed or no strong Malay indicators.
    """
    text_lower = text.lower()
    if MALAY_REGEX.search(text_lower):
        print(f"[DEBUG] Detected Malay keywords in query: '{text}'")
        return "Malay"
    print(f"[DEBUG] Detected English or mixed/no strong Malay keywords in query: '{text}'")
    return "English"

# --- MAIN INTERACTION LOOP ---

def main():
    # Initialize RAG components once
    # context_retrieval.setup_rag_chain now returns db_client and llm
    db_client = ChromaDBClient()
    
    print("\n--- Initializing RAG components ---")
    print(f"Setting up Ollama LLM (mixtral)...")
    llm = ChatOllama(model="mixtral", temperature=0.2) 
    print("Ollama LLM setup complete.")
    
    print("\n--- Chatbot Ready ---")
    print("Type your questions below. Type 'exit' to quit.")

    while True:
        question = input("\nYour Question: ")
        if question.lower() == 'exit':
            print("Goodbye!")
            break
        
        try:
            detected_lang = detect_language(question)
            print(f"[DEBUG] Chatbot will respond in: {detected_lang}")

            # --- DEBUGGING STEP: Direct retrieval using get_relevant_documents ---
            print(f"\n--- Performing dynamic retrieval for: '{question}' ---")
            retrieved_docs_for_debug = get_relevant_documents(question, db_client) 
            
            if not retrieved_docs_for_debug:
                print("Dynamic retrieval found no results.")
            else:
                print(f"Dynamic retrieval found {len(retrieved_docs_for_debug)} results:")
                for i, doc in enumerate(retrieved_docs_for_debug):
                    print(f"  Result {i+1}:")
                    print(f"    Content: {doc.page_content[:200]}...")
                    print(f"    Metadata: {doc.metadata}")
            print("--- End Dynamic Retrieval Debug ---\n")
            # --- END DEBUGGING STEP ---

            # --- KEY CHANGE: Dynamic prompt creation per query ---
            language_instruction = ""
            if detected_lang == "Malay":
                language_instruction = "Jawab dalam Bahasa Melayu. Jika soalan adalah campuran, jawab dalam Bahasa Inggeris."
            else: 
                language_instruction = "Respond in English. If the query is mixed, respond in English."

            template = f"""You are an AI assistant for the Malaysian government, specifically focusing on the Sabah State Computer Services Department (Jabatan Perkhidmatan Komputer Negeri - JPKN). Your goal is to provide accurate and helpful information based ONLY on the provided context.

            {language_instruction}

            If the question cannot be answered from the given context, politely state that you don't have enough information from the provided sources. Do NOT make up answers.

            Provide short and concise answers, directly addressing the question without unnecessary elaboration.

            Context:
            {{context}}

            Question: {{question}}

            Answer:"""
            
            prompt = ChatPromptTemplate.from_template(template)

            # Re-create the RAG chain for each query with the correct prompt
            rag_chain = (
                {"context": RunnableLambda(lambda q: get_relevant_documents(q, db_client)) | RunnableLambda(lambda docs: "\n\n".join([doc.page_content for doc in docs])), 
                 "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

            response = rag_chain.invoke(question)
            print(f"Chatbot: {response}")
        except Exception as e:
            print(f"[ERROR] An error occurred during response generation: {e}")
            print("Please ensure Ollama is running and the 'mixtral' model is available (ollama serve & ollama pull mixtral).")

if __name__ == "__main__":
    main()
