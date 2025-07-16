import os
import re
import json
import hashlib
import unicodedata
from typing import List, Dict, Any, Tuple
import chromadb
from chromadb.utils import embedding_functions
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, Runnable
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser

# --- CONFIGURATION ---
CHROMA_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
COLLECTION_NAME = "jpkn_website_content" 
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" 

OLLAMA_MODEL_NAME = "mixtral" 

# --- EMBEDDING FUNCTION ---
def get_embedding_function():
    """Returns the configured SentenceTransformer embedding function."""
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

# --- CHROMA DB CLIENT (Encapsulates ChromaDB interaction) ---
class ChromaDBClient:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self.embedding_function = get_embedding_function()
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"ChromaDB client initialized. Collection: {COLLECTION_NAME}")

    def query_documents(self, query_texts: List[str], n_results: int = 5, where_filter: Dict = None) -> List[Document]:
        """Queries the ChromaDB for relevant documents."""
        try:
            if where_filter:
                results = self.collection.query(
                    query_texts=query_texts,
                    n_results=n_results,
                    where=where_filter,
                    include=['documents', 'metadatas']
                )
            else:
                results = self.collection.query(
                    query_texts=query_texts,
                    n_results=n_results,
                    include=['documents', 'metadatas']
                )

            langchain_documents = []
            if results and results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    doc_content = results['documents'][0][i]
                    doc_metadata = results['metadatas'][0][i]
                    langchain_documents.append(Document(page_content=doc_content, metadata=doc_metadata))
            return langchain_documents
        except Exception as e:
            print(f"[ERROR] Error querying ChromaDB: {e}")
            return []

# --- DYNAMIC RETRIEVER LOGIC ---

# Keywords that suggest a query about directory, people, or roles
ROLE_KEYWORDS = [
    "direktori", "director", "pengarah", "timbalan", "ketua", "pegawai", 
    "staff", "siapa", "who is", "manager", "engineer", "jurutera", 
    "juruter", # for juruteknik
    "pembantu", "penolong", # for pembantu pengarah, penolong pengarah etc.
    "rank", "jawatan", "position", "pangkat",
    "timbalan pengarah", "ketua penolong pengarah", "ketua pasukan inovasi",
    "pengarahbahagian" 
]

ROLE_KEYWORDS_PATTERN_PARTS = []
for keyword in ROLE_KEYWORDS:
    if " " in keyword:
        ROLE_KEYWORDS_PATTERN_PARTS.append(re.escape(keyword).replace(r'\ ', r'.*?'))
    else:
        ROLE_KEYWORDS_PATTERN_PARTS.append(r'\b' + re.escape(keyword) + r'\b')

ROLE_KEYWORDS_PATTERN_PARTS.append(r'jawatan\s*:\s*') 
ROLE_KEYWORDS_PATTERN_PARTS.append(r'pengarah\s*') 
ROLE_KEYWORDS_PATTERN_PARTS.append(r'director\s*') 

ROLE_KEYWORDS_REGEX = re.compile(r'(?i)' + '|'.join(ROLE_KEYWORDS_PATTERN_PARTS))

# Keywords for contact/hours queries
CONTACT_HOURS_KEYWORDS = [
    "waktu", "pejabat", "jam", "masa", "kerja", "buka", "tutup", "operasi", 
    "hours", "office", "contact", "telefon", "email", "alamat", "address", 
    "hubungi", "telefon", "nombor"
]
CONTACT_HOURS_REGEX = re.compile(r'(?i)\b(?:' + '|'.join(re.escape(k) for k in CONTACT_HOURS_KEYWORDS) + r')\b')

# Specific director keywords for aggressive search
DIRECTOR_SPECIFIC_KEYWORDS = ["pengarah", "director"]
DIRECTOR_SPECIFIC_REGEX = re.compile(r'(?i)\b(?:' + '|'.join(re.escape(k) for k in DIRECTOR_SPECIFIC_KEYWORDS) + r')\b')


def _is_directory_query(query: str) -> bool:
    """Checks if the query is likely about the organizational directory or people using regex."""
    return bool(ROLE_KEYWORDS_REGEX.search(query))

def _is_contact_hours_query(query: str) -> bool:
    """Checks if the query is likely about contact information or office hours."""
    return bool(CONTACT_HOURS_REGEX.search(query))

def _is_director_query_specific(query: str) -> bool:
    """Checks if the query is specifically asking about the director."""
    return bool(DIRECTOR_SPECIFIC_REGEX.search(query))


def _get_person_name_from_query(query: str) -> str | None:
    """
    Extracts a potential person's name from the query.
    This is a basic heuristic and might need improvement for complex names.
    """
    query_words = query.split()
    potential_names_parts = []
    # Keywords that indicate a role or question word, to avoid picking them up as names
    role_or_question_keywords = set([re.escape(k).lower() for k in ROLE_KEYWORDS + ["siapa", "who", "is", "apakah", "what", "punya"]])

    for word in query_words:
        # Check if word starts with a capital letter, is longer than 1 character (to avoid 'A', 'B' as names),
        # and is not a known role/question keyword.
        # Allow words that are entirely uppercase if they are not in role_or_question_keywords (e.g., acronyms that are names)
        if word and word[0].isupper() and len(word) > 1 and word.lower() not in role_or_question_keywords:
            potential_names_parts.append(word)
    
    # --- DEBUG: Print extracted parts ---
    print(f"[DEBUG] _get_person_name_from_query: Query='{query}', Potential parts={potential_names_parts}")

    # Combine parts to form a potential full name.
    # Simple heuristic: if we found at least one capitalized word not a role/question keyword
    if potential_names_parts:
        # If the query is "siapakah Ernywati Dewi Abas", we want "Ernywati Dewi Abas"
        # If the query is "siapakah Ernywati", we want "Ernywati"
        # This simple join should work for most cases.
        return " ".join(potential_names_parts).strip()
    return None

# --- NEW FUNCTION: Perform keyword search on raw JSON files ---
def _keyword_search_from_json(query: str, data_dir: str, person_name: str | None) -> List[Document]:
    """
    Performs a keyword search directly on raw JSON files in the data directory
    for directory-related content. Prioritizes exact matches.
    """
    keyword_matched_docs = []
    query_lower = query.lower()
    
    # Find the directory JSON file
    directory_file_hash = hashlib.md5("https://jpkn.sabah.gov.my/direktori/".encode()).hexdigest()
    directory_filename_pattern = re.compile(rf"Direktori_JPKN_.*{directory_file_hash[:12]}\.json")

    for filename in os.listdir(data_dir):
        if directory_filename_pattern.match(filename):
            filepath = os.path.join(data_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for doc_dict in data:
                        content = doc_dict.get('page_content', '')
                        metadata = doc_dict.get('metadata', {})
                        
                        # Normalize content for robust matching
                        normalized_content = unicodedata.normalize("NFKC", content).lower()
                        
                        # Aggressive match for director by known name
                        if "ernywati dewi abas" in normalized_content and "pengarah" in normalized_content:
                            print(f"[DEBUG] Keyword search found exact director match ('Ernywati Dewi Abas') in {filename}.")
                            # Add to the front to prioritize
                            keyword_matched_docs.insert(0, Document(page_content=content, metadata=metadata))
                        
                        # General keyword match for other roles/names if person_name is present
                        elif person_name: # Only proceed if a person_name was detected in the query
                            # Try to match the full detected name or parts of it
                            if person_name.lower() in normalized_content:
                                print(f"[DEBUG] Keyword search found general person name match ('{person_name}') in {filename}.")
                                keyword_matched_docs.append(Document(page_content=content, metadata=metadata))
                            else:
                                # Try matching individual parts of the name if full name not found
                                person_name_parts_lower = person_name.lower().split()
                                if any(part in normalized_content for part in person_name_parts_lower):
                                    print(f"[DEBUG] Keyword search found partial person name match ('{person_name}') in {filename}.")
                                    keyword_matched_docs.append(Document(page_content=content, metadata=metadata))
                        
                        # Fallback for general role keywords if no specific person name match
                        elif any(re.search(re.escape(kw).replace(r'\ ', r'.*?'), normalized_content) for kw in ROLE_KEYWORDS):
                             if Document(page_content=content, metadata=metadata) not in keyword_matched_docs: # Avoid duplicates if already added by person_name
                                print(f"[DEBUG] Keyword search found general role keyword match in {filename}.")
                                keyword_matched_docs.append(Document(page_content=content, metadata=metadata))

            except Exception as e:
                print(f"[ERROR] Error reading or processing {filename} for keyword search: {e}")
    
    # Remove duplicates
    unique_docs = []
    seen_contents = set()
    for doc in keyword_matched_docs:
        if doc.page_content not in seen_contents:
            unique_docs.append(doc)
            seen_contents.add(doc.page_content)
    
    return unique_docs


def _post_filter_directory_docs(
    documents: List[Document], 
    query: str, 
    person_name: str | None = None
) -> List[Document]:
    """
    Post-filters directory documents to prioritize those most relevant to the query,
    especially for specific person or rank queries, using regex for content matching.
    """
    filtered_docs = []
    query_lower = query.lower()

    content_role_regex = re.compile(r'(?i)' + '|'.join(re.escape(k).replace(r'\ ', r'.*?') for k in ROLE_KEYWORDS))

    # Aggressive prioritization for director queries by name (already handled by keyword search, but good safeguard)
    if _is_director_query_specific(query) or ("ernywati dewi abas" in query_lower and "pengarah" in query_lower):
        director_candidates = []
        for doc in documents:
            content_lower = doc.page_content.lower()
            if "ernywati dewi abas" in content_lower and "pengarah" in content_lower:
                director_candidates.append(doc)
        
        if director_candidates:
            director_candidates.sort(key=lambda x: x.metadata.get("type") == "table_data", reverse=True)
            print(f"[DEBUG] Post-filter found specific director candidates ('Ernywati Dewi Abas'): {len(director_candidates)}")
            return director_candidates[:1] # Return only the top director candidate

    if person_name:
        person_name_lower = person_name.lower()
        
        name_matched_docs = []
        for doc in documents:
            content_lower = doc.page_content.lower()
            if person_name_lower in content_lower: 
                name_matched_docs.append(doc)
        
        if name_matched_docs:
            # Sort to prioritize table_data and role-containing docs
            sorted_docs = sorted(name_matched_docs, key=lambda x: (
                x.metadata.get("type") == "table_data" and bool(content_role_regex.search(x.page_content)),
                bool(content_role_regex.search(x.page_content)) 
            ), reverse=True)
            print(f"[DEBUG] Post-filter found {len(sorted_docs)} documents matching person name '{person_name}'.")
            return sorted_docs
    
    # If no specific person, or no direct name hits, or if it's a general rank query
    # Filter for documents from "Direktori | JPKN" page_title and containing relevant role keywords
    general_filtered_docs = []
    for doc in documents:
        if doc.metadata.get("page_title") == "Direktori | JPKN":
            content_lower = doc.page_content.lower()
            if content_role_regex.search(content_lower): 
                general_filtered_docs.append(doc)
    
    general_filtered_docs.sort(key=lambda x: x.metadata.get("type") == "table_data", reverse=True)

    if not general_filtered_docs and documents:
        print("[DEBUG] Post-filter yielded no specific directory matches. Returning top 3 from initial retrieval.")
        return documents[:3] 
        
    return general_filtered_docs

def _post_filter_contact_hours_docs(documents: List[Document], query: str) -> List[Document]:
    """
    Post-filters documents to prioritize those containing contact information or office hours.
    Prioritizes header/footer types.
    """
    filtered_docs = []
    query_lower = query.lower()
    
    # Prioritize documents explicitly marked as header or footer
    header_footer_docs = [
        doc for doc in documents 
        if doc.metadata.get("type") in ["header", "footer"]
    ]
    
    # If header/footer docs are found, check if they contain relevant keywords
    if header_footer_docs:
        for doc in header_footer_docs:
            content_lower = doc.page_content.lower()
            if CONTACT_HOURS_REGEX.search(content_lower):
                filtered_docs.append(doc)
        
        if filtered_docs:
            print(f"[DEBUG] Post-filter found {len(filtered_docs)} header/footer docs with contact/hours keywords.")
            return filtered_docs
    
    # Fallback: if no specific header/footer matches, look for general content with keywords
    for doc in documents:
        content_lower = doc.page_content.lower()
        if CONTACT_HOURS_REGEX.search(content_lower):
            filtered_docs.append(doc)

    if not filtered_docs and documents:
        print("[DEBUG] Post-filter yielded no specific contact/hours matches. Returning top 3 from initial retrieval.")
        return documents[:3] 
    
    return filtered_docs[:5] 


def get_relevant_documents(query: str, db_client: ChromaDBClient) -> List[Document]:
    """
    Retrieves relevant documents from ChromaDB, applying dynamic filters based on query intent.
    """
    retrieved_docs = []
    
    # --- KEY CHANGE: Extract person name once and use for intent classification ---
    person_name = _get_person_name_from_query(query)
    # Trigger directory intent if a role keyword is found OR a person's name is detected
    is_directory_intent = _is_directory_query(query) or (person_name is not None)

    if is_directory_intent:
        print(f"[DEBUG] Query '{query}' identified as a directory-related query (or person name '{person_name}' detected).")
        
        # Perform keyword search first for directory queries
        keyword_docs = _keyword_search_from_json(query, DATA_DIR, person_name) # Pass person_name
        print(f"[DEBUG] Keyword search found {len(keyword_docs)} documents.")

        # If director is specifically asked by name or role, and we found it via keyword, prioritize that
        if (_is_director_query_specific(query) or ("ernywati dewi abas" in query.lower())) and \
           any("ernywati dewi abas" in doc.page_content.lower() for doc in keyword_docs):
            print("[DEBUG] Director specific query and found via keyword search. Prioritizing.")
            director_specific_docs = [
                doc for doc in keyword_docs 
                if "ernywati dewi abas" in doc.page_content.lower() and "pengarah" in doc.page_content.lower()
            ]
            if director_specific_docs:
                director_specific_docs.sort(key=lambda x: x.metadata.get("type") == "table_data", reverse=True)
                return director_specific_docs[:1] # Return only the top director doc

        # Prepare query texts for semantic search. Add the person's name to boost relevance.
        semantic_query_texts = [query]
        if person_name:
            semantic_query_texts.append(person_name)
            print(f"[DEBUG] Boosting semantic query with person name: {semantic_query_texts}")

        pre_filter = {"page_title": {"$eq": "Direktori | JPKN"}}
        print(f"[DEBUG] Applying pre-filter for directory: {pre_filter}")
        initial_retrieved_docs = db_client.query_documents(query_texts=semantic_query_texts, n_results=50, where_filter=pre_filter) 
        print(f"[DEBUG] Initial semantic retrieval (before post-filter) found {len(initial_retrieved_docs)} documents.")
        
        # Combine keyword-matched documents with semantically retrieved ones
        combined_docs = []
        seen_contents = set()
        
        # Add keyword docs first (already potentially ordered by _keyword_search_from_json)
        for doc in keyword_docs:
            if doc.page_content not in seen_contents:
                combined_docs.append(doc)
                seen_contents.add(doc.page_content)
        
        # Add semantic docs, avoiding duplicates
        for doc in initial_retrieved_docs:
            if doc.page_content not in seen_contents:
                combined_docs.append(doc)
                seen_contents.add(doc.page_content)

        print(f"[DEBUG] Combined initial and keyword search documents: {len(combined_docs)}")

        retrieved_docs = _post_filter_directory_docs(combined_docs, query, person_name) # Pass person_name
        print(f"[DEBUG] Post-filter refined to {len(retrieved_docs)} results for directory query.")
    
    elif _is_contact_hours_query(query):
        print(f"[DEBUG] Query '{query}' identified as a contact/hours query.")
        contact_hours_pre_filter = {"type": {"$in": ["header", "footer", "main_content"]}}
        print(f"[DEBUG] Applying pre-filter for contact/hours: {contact_hours_pre_filter}")
        initial_retrieved_docs = db_client.query_documents(query_texts=[query], n_results=20, where_filter=contact_hours_pre_filter) 
        print(f"[DEBUG] Initial retrieval (before post-filter) found {len(initial_retrieved_docs)} documents.")
        retrieved_docs = _post_filter_contact_hours_docs(initial_retrieved_docs, query)
        print(f"[DEBUG] Post-filter refined to {len(retrieved_docs)} results for contact/hours query.")

    if not retrieved_docs:
        print("[DEBUG] No specific dynamic filter applied or dynamic filter returned no results. Querying all documents without specific filter.")
        retrieved_docs = db_client.query_documents(query_texts=[query], n_results=5) 
    
    return retrieved_docs

# --- RAG CHAIN SETUP ---

def setup_rag_chain():
    """Initializes and returns the complete RAG chain."""
    print("\n--- Initializing RAG components ---")
    
    db_client = ChromaDBClient()

    print(f"Setting up Ollama LLM ({OLLAMA_MODEL_NAME})...")
    llm = ChatOllama(model=OLLAMA_MODEL_NAME, temperature=0.2)
    print("Ollama LLM setup complete.")

    print("RAG chain setup complete (components returned).")
    return db_client, llm 

if __name__ == "__main__":
    print("This script provides RAG functions. Run chatbot.py to interact with the chatbot.")
