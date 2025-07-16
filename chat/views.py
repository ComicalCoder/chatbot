import os
import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.conf import settings

# Core LangChain components
from langchain_huggingface.embeddings import HuggingFaceEmbeddings # Updated import
from langchain_chroma import Chroma # Updated import
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# NEW: Import ChatOllama for local LLM integration
from langchain_community.chat_models import ChatOllama

# --- GLOBAL RAG COMPONENTS (Loaded once on app startup) ---
_embeddings = None
_vector_store = None
_llm = None
_rag_chain = None

# --- CONFIGURATION (consistent with other scripts) ---
# settings.BASE_DIR points to the directory containing manage.py (your outer 'chatbot' folder)
CHROMA_DB_DIR = os.path.join(settings.BASE_DIR, "chroma_db")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL_NAME = "mistral" # Ollama model name

# --- RAG COMPONENT LOADING FUNCTIONS ---

def load_embedding_model():
    """Loads the HuggingFace embedding model."""
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    print("Embedding model loaded.")
    return embeddings

def load_vector_store(embeddings_instance):
    """Loads the Chroma vector store."""
    print(f"Loading Chroma vector store from '{CHROMA_DB_DIR}'...")
    if not os.path.exists(CHROMA_DB_DIR) or not os.listdir(CHROMA_DB_DIR):
        raise FileNotFoundError(f"Chroma DB not found at {CHROMA_DB_DIR}. Please run create_vector_store.py first.")
    
    db = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings_instance)
    print("Chroma vector store loaded.")
    return db

def setup_llm():
    """Sets up the Ollama LLM (Mistral)."""
    print(f"Setting up Ollama LLM ({OLLAMA_MODEL_NAME})...")
    # Ensure Ollama server is running and the specified model is pulled
    llm = ChatOllama(model=OLLAMA_MODEL_NAME, temperature=0.2)
    print("Ollama LLM setup complete.")
    return llm

def get_retriever(vector_store_instance):
    """Creates a retriever from the vector store."""
    return vector_store_instance.as_retriever(search_kwargs={"k": 5})

def create_rag_chain_instance(llm_instance, retriever_instance):
    """Constructs the RAG chain."""
    template = """You are an AI assistant for the Malaysian government, specifically focusing on the Sabah State Computer Services Department (Jabatan Perkhidmatan Komputer Negeri - JPKN). Your goal is to provide accurate and helpful information based ONLY on the provided context.

    If the question cannot be answered from the given context, politely state that you don't have enough information from the provided sources. Do NOT make up answers.

    Context:
    {context}

    Question: {question}

    Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    rag_chain = (
        {"context": retriever_instance | RunnableLambda(lambda docs: "\n\n".join([doc.page_content for doc in docs])), "question": RunnablePassthrough()}
        | prompt
        | llm_instance
        | StrOutputParser()
    )
    print("RAG chain created.")
    return rag_chain

# --- APP CONFIGURATION FOR INITIALIZATION ---
class ChatAppConfig(object):
    name = 'chat' # This must match your app name 'chat'
    verbose_name = "RAG Chatbot"

    def ready(self):
        """
        This method is called when Django starts.
        It's the ideal place to load heavy, long-lived resources.
        """
        global _embeddings, _vector_store, _llm, _rag_chain
        if _rag_chain is None: # Only load if not already loaded
            print("\n--- Initializing RAG Chatbot Components (Django App Ready) ---")
            try:
                _embeddings = load_embedding_model()
                _vector_store = load_vector_store(_embeddings)
                _llm = setup_llm()
                _retriever = get_retriever(_vector_store)
                _rag_chain = create_rag_chain_instance(_llm, _retriever)
                print("--- RAG Chatbot Components Initialized Successfully ---")
            except Exception as e:
                print(f"[CRITICAL ERROR] Failed to initialize RAG Chatbot components: {e}")
                # In a production environment, you might want to log this error
                # and potentially prevent the server from starting if critical.

# --- API ENDPOINT ---

@method_decorator(csrf_exempt, name='dispatch')
def chat_api(request):
    """
    Django API endpoint to receive user questions and return chatbot responses.
    Expects a POST request with JSON body: {"question": "Your question here"}
    """
    if request.method == 'POST':
        if not _rag_chain:
            return JsonResponse({"error": "Chatbot is not fully initialized. Please wait or check server logs."}, status=503)

        try:
            data = json.loads(request.body)
            question = data.get('question')

            if not question:
                return JsonResponse({"error": "Missing 'question' in request body"}, status=400)

            print(f"Received question: '{question}'")
            response = _rag_chain.invoke(question)
            print(f"Generated response: '{response}'")
            return JsonResponse({"response": response})
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON in request body"}, status=400)
        except Exception as e:
            print(f"[ERROR] An error occurred during response generation: {e}")
            return JsonResponse({"error": "Internal server error during response generation."}, status=500)
    else:
        return JsonResponse({"error": "Only POST requests are allowed"}, status=405)

