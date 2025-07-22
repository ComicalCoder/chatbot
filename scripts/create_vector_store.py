import os
import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import List, Dict, Any
import requests 

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Import agency-specific configurations
from agency_config import (
    AGENCY_EMBEDDING_MODEL,
    AGENCY_CHUNK_SIZE, 
    AGENCY_CHUNK_OVERLAP, 
    AGENCY_PAGE_TITLE_MAPPING,
    AGENCY_EXCLUDE_HEADERS_FOOTERS, 
    AGENCY_EXCLUDE_COLLAPSIBLE_SECTIONS, 
    agency_format_content_case
)

# --- CONFIGURATION ---
VECTOR_DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "vector_db")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")

# --- SETUP LOGGING ---
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'create_vector_store.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Text Normalization Helper ---
def normalize_text(text: str) -> str:
    """Normalize text: fix whitespace and Unicode issues."""
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = re.sub(r"\s+", " ", text).strip()
    return text

# --- Document Loading and Processing ---
def load_and_process_documents(data_dir: str) -> List[Document]:
    """
    Loads documents from JSON files, extracts content and metadata,
    and applies initial processing.
    """
    documents = []
    logger.info(f"Loading documents from: {data_dir}")

    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)
        if not filename.endswith('.json'):
            logger.debug(f"Skipping non-JSON file: {filename}")
            continue

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if not isinstance(data, list):
                    logger.warning(f"Skipping {filename}: Expected a list of dictionaries, got {type(data)}")
                    continue

                for item in data:
                    if not isinstance(item, dict):
                        logger.warning(f"Skipping item in {filename}: Expected a dictionary, got {type(item)}")
                        continue

                    page_content = item.get('page_content')
                    metadata = item.get('metadata', {})

                    # --- CRITICAL FIX FOR ATTENDANCE RECORDS ---
                    if filename == "attendance_raw.json":
                        # For attendance records, store the entire JSON object as a string
                        # in page_content, and also add its fields to metadata.
                        # This allows LLM to parse structured data directly.
                        if isinstance(item, dict):
                            page_content = json.dumps(item, ensure_ascii=False) # Store the whole record as JSON string
                            # Add relevant attendance fields to metadata for filtering/retrieval
                            metadata.update({
                                "source_type": "attendance_record",
                                "original_filename": filename,
                                "id": item.get('id'),
                                "emp_id": item.get('emp_id'),
                                "icno": item.get('icno'),
                                "att_date": item.get('att_date'),
                                "clock_in": item.get('clock_in'),
                                "clock_out": item.get('clock_out'),
                                "total_hours": item.get('total_hours'),
                                "cuti": item.get('cuti'),
                                "public_holiday": item.get('public_holiday'),
                                # Add other fields as needed for metadata
                            })
                        else:
                            logger.warning(f"Skipping malformed attendance record in {filename}: {item}")
                            continue
                    # --- END CRITICAL FIX ---
                    
                    # Ensure page_content is a string for all other documents
                    if not isinstance(page_content, str):
                        logger.warning(f"Skipping document with non-string page_content in {filename}: {type(page_content)} - {str(page_content)[:50]}...")
                        continue

                    # Apply agency-specific content formatting
                    formatted_content = agency_format_content_case(page_content)

                    # --- NEW FIX: Extract Nama and Jawatan for Directory entries and add to metadata ---
                    if metadata.get('page_title') == AGENCY_PAGE_TITLE_MAPPING.get("directory") and metadata.get('type') == 'table_data':
                        # Try to parse "NAMA: X, JAWATAN: Y" format first
                        name_match = re.search(r'NAMA:\s*(.*?)(?:,\s*JAWATAN:|$)', formatted_content, re.IGNORECASE)
                        jawatan_match = re.search(r'JAWATAN:\s*(.*?)(?:,|\Z)', formatted_content, re.IGNORECASE)
                        
                        if name_match:
                            metadata['Nama'] = normalize_text(name_match.group(1))
                            logger.debug(f"Extracted Nama from page_content: {metadata['Nama']}")
                        
                        if jawatan_match:
                            metadata['Jawatan'] = normalize_text(jawatan_match.group(1))
                            logger.debug(f"Extracted Jawatan from page_content: {metadata['Jawatan']}")
                        
                        # Fallback for "Column X: VALUE" format if direct key-value not found
                        if 'Nama' not in metadata and 'Jawatan' not in metadata:
                            column_matches = re.findall(r'Column (\d+):\s*(.*?)(?=(?:,\s*Column \d+:\s*|$))', formatted_content, re.IGNORECASE)
                            column_map = {
                                '2': 'Nama',
                                '3': 'Jawatan'
                            }
                            for col_num, value in column_matches:
                                field_name = column_map.get(col_num)
                                if field_name and field_name not in metadata:
                                    metadata[field_name] = normalize_text(value)
                                    logger.debug(f"Extracted {field_name} from column: {metadata[field_name]}")
                    # --- END NEW FIX ---

                    # Add original filename to metadata
                    metadata['original_filename'] = filename
                    
                    # Add URL to metadata if available
                    if 'url' in item:
                        metadata['url'] = item['url']

                    # Add language to metadata if available or infer
                    if 'language' in item:
                        metadata['language'] = item['language']
                    elif 'lang' in item: # Handle 'lang' as an alternative key
                        metadata['language'] = item['lang']
                    else:
                        # Simple heuristic: if content contains common Malay words, assume Malay
                        if re.search(r'\b(dan|yang|untuk|ini|tidak|adalah)\b', formatted_content, re.IGNORECASE):
                            metadata['language'] = 'ms'
                        else:
                            metadata['language'] = 'en' # Default to English

                    doc = Document(page_content=formatted_content, metadata=metadata)
                    documents.append(doc)
                    logger.debug(f"Loaded document from {filename}, page_title: {metadata.get('page_title')}, type: {metadata.get('type')}")

        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON from {filename}: {e}")
        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
    
    logger.info(f"Finished loading. Total documents prepared: {len(documents)}")
    return documents

def filter_documents(documents: List[Document]) -> List[Document]:
    """
    Filters documents based on metadata, excluding headers, footers,
    and collapsible sections if configured.
    """
    filtered_docs = []
    for doc in documents:
        doc_type = doc.metadata.get('type')
        page_title = doc.metadata.get('page_title')
        
        # Exclude headers and footers if configured
        if AGENCY_EXCLUDE_HEADERS_FOOTERS and doc_type in ['header', 'footer']:
            logger.debug(f"Excluding header/footer: {doc.metadata.get('page_title')} - {doc_type}")
            continue
        
        # Exclude collapsible sections if configured
        if AGENCY_EXCLUDE_COLLAPSIBLE_SECTIONS and doc_type == 'collapsible_section':
            logger.debug(f"Excluding collapsible section: {doc.metadata.get('page_title')} - {doc_type}")
            continue

        # Keep specific types of documents always, regardless of other filters
        # Attendance records are always kept as their content is crucial for calculations
        if doc.metadata.get('source_type') == 'attendance_record':
            filtered_docs.append(doc)
            continue
        
        # Ensure 'table_data' documents are always included for directory information
        if doc_type == 'table_data' and page_title == AGENCY_PAGE_TITLE_MAPPING.get("directory"):
            filtered_docs.append(doc)
            continue

        # General content types to include
        if doc_type in ['main_content', 'contact_info', 'office_hours', 'list_item', 'paragraph', 'title', 'heading']:
            filtered_docs.append(doc)
            continue

        # Include documents from specific page titles if they are not explicitly excluded
        if page_title in [
            AGENCY_PAGE_TITLE_MAPPING.get("corporate_info"),
            AGENCY_PAGE_TITLE_MAPPING.get("office_locations"),
            AGENCY_PAGE_TITLE_MAPPING.get("contact_us")
        ]:
            filtered_docs.append(doc)
            continue
            
        logger.debug(f"Excluding document by default filter: {doc.metadata.get('page_title')} - {doc.type}")

    logger.info(f"Finished filtering. Documents after filtering: {len(filtered_docs)}")
    return filtered_docs

def split_documents(documents: List[Document]) -> List[Document]:
    """Splits documents into smaller chunks for embedding."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=AGENCY_CHUNK_SIZE,
        chunk_overlap=AGENCY_CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )
    
    # Do not split attendance records or table data, pass them as is
    # This is crucial for the LLM to process the structured JSON directly
    chunks = []
    for doc in documents:
        if doc.metadata.get('source_type') == 'attendance_record' or \
           (doc.metadata.get('page_title') == AGENCY_PAGE_TITLE_MAPPING.get("directory") and doc.metadata.get('type') == 'table_data'):
            chunks.append(doc)
            logger.debug(f"Keeping document as single chunk (attendance/table_data): {doc.metadata.get('original_filename')}")
        else:
            split_docs = text_splitter.split_documents([doc])
            chunks.extend(split_docs)
            logger.debug(f"Split document from {doc.metadata.get('original_filename')} into {len(split_docs)} chunks.")

    logger.info(f"Finished splitting. Total chunks created: {len(chunks)}")
    return chunks

def create_vector_store():
    """
    Main function to create and persist the ChromaDB vector store.
    """
    logger.info("Starting vector store creation process...")

    # 1. Load and process documents
    raw_documents = load_and_process_documents(DATA_DIR)
    
    # 2. Filter documents
    filtered_documents = filter_documents(raw_documents)

    # 3. Split documents into chunks
    chunks = split_documents(filtered_documents)

    if not chunks:
        logger.warning("No valid document chunks to add to the vector store. Exiting.")
        return

    # 4. Initialize embeddings and vector store
    try:
        embeddings = OllamaEmbeddings(model=AGENCY_EMBEDDING_MODEL)
        # Test connection to embeddings model
        embeddings.embed_query("test query")
        logger.info(f"Ollama embeddings model '{AGENCY_EMBEDDING_MODEL}' initialized.")
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Failed to connect to Ollama server for embeddings: {e}. Please ensure Ollama server is running and '{AGENCY_EMBEDDING_MODEL}' model is pulled.")
        return
    except Exception as e:
        logger.error(f"Failed to initialize Ollama embeddings: {e}")
        return

    logger.info(f"Attempting to create ChromaDB at: {VECTOR_DB_DIR}")
    Path(VECTOR_DB_DIR).mkdir(parents=True, exist_ok=True) # Ensure directory exists

    try:
        # Create or load the vector store and add documents
        # If persist_directory exists and contains data, it will load. Otherwise, it creates.
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=VECTOR_DB_DIR
        )
        vectorstore.persist()
        logger.info(f"ChromaDB vector store created/updated and persisted successfully at {VECTOR_DB_DIR}.")
    except Exception as e:
        logger.error(f"An error occurred during ChromaDB creation/persistence: {e}")

if __name__ == "__main__":
    create_vector_store()
