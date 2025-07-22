import os
import logging
import re
import json
import unicodedata
from typing import List, Dict, Any, Optional
from pathlib import Path 
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document
from datetime import datetime
import requests # Ensure requests is imported for ConnectionError handling

# Import agency-specific configurations
from agency_config import (
    AGENCY_EMBEDDING_MODEL,
    AGENCY_ROLE_KEYWORDS,
    AGENCY_CONTACT_HOURS_KEYWORDS,
    AGENCY_LOCATION_KEYWORDS,
    AGENCY_ATTENDANCE_KEYWORDS, # Now includes more terms for counting attendance
    AGENCY_PAGE_TITLE_MAPPING,
    AGENCY_SPECIFIC_RANKS,
    AGENCY_QUERY_EXPANSIONS,
    AGENCY_COMMON_NON_NAMES_LOWER,
    AGENCY_MALAY_KEYWORDS,
    AGENCY_ENGLISH_KEYWORDS, 
    agency_format_content_case,
    AGENCY_OFFICE_HOURS_DATA,
    AGENCY_CLEANING_REGEX # Imported for consistent cleaning of extracted Jawatan
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
        logging.FileHandler(os.path.join(LOG_DIR, 'context_retrieval.log')), 
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- GLOBAL COMPONENTS (initialized once) ---
_vectorstore = None
_embeddings = None
_ollama_embeddings_available = False # Flag to track Ollama embeddings availability

def _get_vector_store():
    """Initializes or returns the existing ChromaDB vector store."""
    global _vectorstore
    global _embeddings
    global _ollama_embeddings_available

    if _vectorstore is None:
        try:
            _embeddings = OllamaEmbeddings(model=AGENCY_EMBEDDING_MODEL)
            # Test connection to embeddings model
            _embeddings.embed_query("test query") 
            _ollama_embeddings_available = True
            _vectorstore = Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=_embeddings)
            logger.info("ChromaDB vector store loaded successfully.")
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Failed to connect to Ollama server for embeddings: {e}. Please ensure Ollama server is running and '{AGENCY_EMBEDDING_MODEL}' model is pulled.")
            _vectorstore = None
            _embeddings = None
            _ollama_embeddings_available = False
        except Exception as e:
            logger.error(f"Failed to load ChromaDB vector store or initialize embeddings: {e}. Please ensure create_vector_store.py has been run and the VECTOR_DB_DIR exists and contains data.")
            _vectorstore = None 
            _embeddings = None
            _ollama_embeddings_available = False
    return _vectorstore

# --- Text Normalization Helper (consistent with other scripts) ---
def normalize_text(text: str) -> str:
    """Normalize text: fix whitespace and Unicode issues."""
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = re.sub(r"\s+", " ", text).strip()
    return text

# --- Intent Detection and Document Retrieval Helper Functions ---

ROLE_KEYWORDS_PATTERN_PARTS = []
for keyword in AGENCY_ROLE_KEYWORDS:
    if " " in keyword:
        ROLE_KEYWORDS_PATTERN_PARTS.append(r'\b' + r'\s*'.join(re.escape(word) for word in keyword.split()) + r'\b')
    else:
        ROLE_KEYWORDS_PATTERN_PARTS.append(r'\b' + re.escape(keyword) + r'\b')

ROLE_KEYWORDS_PATTERN_PARTS.append(r'\b(?:jawatan|pengarah|director|ketua)\s*[:\s]') 
ROLE_KEYWORDS_REGEX = re.compile(r'(?i)(?:' + '|'.join(ROLE_KEYWORDS_PATTERN_PARTS) + r')')

CONTACT_HOURS_REGEX = re.compile(r'(?i)\b(?:' + '|'.join(re.escape(k) for k in AGENCY_CONTACT_HOURS_KEYWORDS) + r')\b')
LOCATION_REGEX = re.compile(r'(?i)\b(?:' + '|'.join(re.escape(k) for k in AGENCY_LOCATION_KEYWORDS) + r')\b')

# Updated ATTENDANCE_REGEX to include more counting-related terms
ATTENDANCE_REGEX = re.compile(r'(?i)\b(?:' + '|'.join(re.escape(k) for k in AGENCY_ATTENDANCE_KEYWORDS) + r')\b')


def is_malay(query: str) -> bool:
    """Detects if the query is likely in Malay based on keywords, prioritizing strong English indicators."""
    query_lower = query.lower()
    
    # Count English keywords
    english_keyword_count = sum(1 for eng_keyword in AGENCY_ENGLISH_KEYWORDS if re.search(r'\b' + re.escape(eng_keyword) + r'\b', query_lower))
    
    # Count Malay keywords
    malay_keyword_count = sum(1 for malay_keyword in AGENCY_MALAY_KEYWORDS if re.search(r'\b' + re.escape(malay_keyword) + r'\b', query_lower))
    
    words = re.findall(r'\b[a-zA-Z]+\b', query_lower)
    total_words = len(words)

    logger.debug(f"is_malay: Query='{query}', English keywords found: {english_keyword_count}, Malay keywords found: {malay_keyword_count}, Total words: {total_words}")

    # If there's a significant presence of English keywords, especially for shorter queries, classify as English
    if (english_keyword_count > 0 and total_words <= 5) or \
       (english_keyword_count / max(1, total_words) >= 0.3 and english_keyword_count > malay_keyword_count):
        logger.debug(f"is_malay: Classifying as English due to strong English keyword presence.")
        return False # It's English

    # If Malay keywords dominate or are the only significant words, classify as Malay
    if (malay_keyword_count > 0 and total_words <= 5) or \
       (malay_keyword_count / max(1, total_words) >= 0.4 and malay_keyword_count >= english_keyword_count):
        logger.debug(f"is_malay: Classifying as Malay due to strong Malay keyword presence.")
        return True
    
    logger.debug(f"is_malay: Defaulting to English (no clear language dominance).")
    return False # Default to English if no strong indicators for either


def extract_person_name_and_rank(query: str) -> tuple[str | None, str | None]:
    """
    Extracts a potential person's name and rank from the query using agency-specific configurations.
    Prioritizes specific ranks and filters common non-names.
    """
    query_lower = query.lower()
    
    extracted_rank = None
    # Try to extract specific ranks first, longest first to catch multi-word ranks
    for rank in sorted(AGENCY_SPECIFIC_RANKS, key=len, reverse=True):
        if rank.lower() in query_lower: # Ensure case-insensitive check
            extracted_rank = rank
            # Remove found rank for name extraction, but keep a copy of original query_lower
            query_lower_without_rank = query_lower.replace(rank.lower(), "").strip() 
            logger.debug(f"extract_person_name_and_rank: Found rank '{extracted_rank}'. Remaining query for name: '{query_lower_without_rank}'")
            break 
    else:
        query_lower_without_rank = query_lower # If no rank found, use original query for name extraction

    # Filter out common non-names and role keywords from the query for name extraction
    # Use the original query words to preserve capitalization initially
    original_query_words = re.findall(r'\b\w+\b', query) # Get words with original casing
    logger.debug(f"extract_person_name_and_rank: Original query words: {original_query_words}")
    
    potential_name_parts = []
    role_keywords_lower = set(k.lower() for k in AGENCY_ROLE_KEYWORDS)
    common_non_names_and_roles = AGENCY_COMMON_NON_NAMES_LOWER.union(role_keywords_lower)

    for word in original_query_words:
        cleaned_word = re.sub(r'[^\w]', '', word) 
        if not cleaned_word:
            logger.debug(f"extract_person_name_and_rank: Skipping empty word after cleaning.")
            continue

        cleaned_word_lower = cleaned_word.lower()
        logger.debug(f"extract_person_name_and_rank: Processing word: '{cleaned_word}' (lower: '{cleaned_word_lower}')")

        if cleaned_word_lower in common_non_names_and_roles:
            logger.debug(f"extract_person_name_and_rank: Skipping common non-name/role: '{cleaned_word}'")
            continue
        
        # --- Enhanced Name Extraction Logic ---
        # Heuristic 1: If a word is capitalized (or all caps) and not a common non-name/role
        if cleaned_word[0].isupper() or cleaned_word.isupper():
            potential_name_parts.append(cleaned_word)
            logger.debug(f"extract_person_name_and_rank: Adding capitalized potential name part: '{cleaned_word}'. Current: {potential_name_parts}")
        # Heuristic 2: If it's a short, common name (like "Ian", "Ghani") and not a role keyword,
        # and it's not already captured by capitalization, consider it.
        # This is a more direct attempt to catch names like 'ian' if they appear lowercase.
        elif len(cleaned_word) > 1 and cleaned_word.isalpha() and cleaned_word_lower not in role_keywords_lower:
            # For now, let's just add it if it's not a role keyword and not already captured.
            # The LLM will ultimately decide if it's a name from context.
            potential_name_parts.append(cleaned_word)
            logger.debug(f"extract_person_name_and_rank: Adding lowercase potential name part (heuristic 2): '{cleaned_word}'. Current: {potential_name_parts}")


    # Filter out any remaining role keywords from potential_name_parts
    final_names_parts = []
    for part in potential_name_parts:
        if part.lower() not in role_keywords_lower:
            final_names_parts.append(part)
        else:
            logger.debug(f"extract_person_name_and_rank: Filtering out role keyword from name part: '{part}'")
    logger.debug(f"extract_person_name_and_rank: Final names parts before join: {final_names_parts}")

    extracted_name = " ".join(final_names_parts).strip()

    # If no name parts found but there are still some words left after filtering roles/common words,
    # and they are not just single letters, try to form a name. This is a last resort.
    if not extracted_name and not final_names_parts and len(original_query_words) > 0:
        remaining_words = [word for word in original_query_words if word.lower() not in common_non_names_and_roles and len(word) > 1]
        if remaining_words:
            # Capitalize the first letter of each remaining word for better name formation
            extracted_name = " ".join([word.capitalize() for word in remaining_words]).strip()
            logger.debug(f"extract_person_name_and_rank: Fallback name extraction: '{extracted_name}' from remaining words: {remaining_words}")


    logger.debug(f"extract_person_name_and_rank: Query='{query}', Extracted Name='{extracted_name}', Extracted Rank='{extracted_rank}'")
    return extracted_name if extracted_name else None, extracted_rank

def extract_branch_name_from_query(query: str) -> str | None:
    """
    Extracts a potential branch or region name from the query.
    Looks for phrases like "cawangan [name]", "wilayah [name]", or specific known branch names.
    """
    query_lower = query.lower()
    
    # Specific patterns for "cawangan [name]" or "wilayah [name]"
    branch_pattern = re.search(r'(?i)(cawangan|wilayah)\s+([\w\s]+)', query_lower)
    if branch_pattern:
        # Return the full matched phrase including "cawangan" or "wilayah" and the name
        return branch_pattern.group(0).strip()
    
    # Look for specific branch names that might not follow the "cawangan X" pattern
    # This list should ideally come from a config or scraped data for robustness
    # These should be lowercased to match query_lower
    known_branch_names = [
        "pedalaman bawah", "beaufort", "pedalaman atas", "kuala penyu", "kota kinabalu",
        "sandakan", "tawau", "kudat", "lahad datu", "keningau", "sibu", "semporna", "ranau",
        "cawangan pedalaman bawah", "wilayah pedalaman bawah" # Added common phrases
    ]
    for branch in known_branch_names:
        if branch in query_lower:
            return branch
            
    return None

def expand_query(query: str) -> str:
    """Expands the query with synonyms/related terms from agency_config."""
    expanded_terms = []
    query_lower = query.lower().strip() # Ensure query is stripped for consistent matching
    logger.debug(f"expand_query: Normalized query_lower for expansion: '{query_lower}'")

    for term, expansions in AGENCY_QUERY_EXPANSIONS.items():
        # Use regex with word boundaries to ensure whole word matching
        # This prevents "tel" from matching "hotel"
        if re.search(r'\b' + re.escape(term) + r'\b', query_lower):
            # Split expansions by space and add unique parts
            for exp_part in expansions.split():
                if exp_part.lower() not in query_lower.split(): # Avoid adding original query words (case-insensitive check)
                    expanded_terms.append(exp_part)
            logger.debug(f"expand_query: Matched term '{term}', adding expansions '{expansions}'. Current expanded_terms: {expanded_terms}")
    
    if expanded_terms:
        # Only add unique expanded terms to avoid redundancy
        unique_expanded_terms = list(set(expanded_terms)) # Use set on the parts
        expanded_query_str = query + " " + " ".join(unique_expanded_terms)
        logger.info(f"expand_query: Final expanded query: '{expanded_query_str}'")
        return expanded_query_str
    return query

def _is_directory_query(query: str) -> bool:
    """Checks if the query is likely about the organizational directory or people using regex."""
    return bool(ROLE_KEYWORDS_REGEX.search(query))

def _is_contact_hours_query(query: str) -> bool:
    """Checks if the query is likely about contact information or office hours."""
    return bool(CONTACT_HOURS_REGEX.search(query))

def _is_location_query(query: str) -> bool:
    """Checks if the query is likely about office locations."""
    return bool(LOCATION_REGEX.search(query))

def _is_attendance_query(query: str) -> bool:
    """Checks if the query is likely about attendance records."""
    return bool(ATTENDANCE_REGEX.search(query))

def _get_specific_rank_from_query(query: str) -> str | None:
    """
    Attempts to extract a specific, multi-word rank title from the query.
    Uses AGENCY_SPECIFIC_RANKS from config.
    """
    query_lower = query.lower()
    # Sort by length descending to match longer phrases first (e.g., "ketua penolong pengarah" before "ketua")
    for rank in sorted(AGENCY_SPECIFIC_RANKS, key=len, reverse=True):
        if rank.lower() in query_lower: # Ensure case-insensitive check
            logger.debug(f"_get_specific_rank_from_query: Found rank '{rank}' in query.")
            return rank
    return None

def _get_person_name_from_query(query: str) -> str | None:
    """
    Extracts a potential person's name from the query.
    Uses AGENCY_COMMON_NON_NAMES_LOWER and AGENCY_ROLE_KEYWORDS from config.
    """
    query_words = query.split()
    potential_name_parts = []
    
    role_keywords_lower = set(k.lower() for k in AGENCY_ROLE_KEYWORDS)
    common_non_names_and_roles = AGENCY_COMMON_NON_NAMES_LOWER.union(role_keywords_lower)

    for word in query_words:
        cleaned_word = re.sub(r'[^\w]', '', word) 
        if not cleaned_word:
            continue

        cleaned_word_lower = cleaned_word.lower()

        if cleaned_word_lower in common_non_names_and_roles:
            continue
        
        # Heuristic: if a word is capitalized (or all caps) and not a common non-name/role, consider it part of a name
        if (cleaned_word[0].isupper() or cleaned_word.isupper()):
            potential_name_parts.append(cleaned_word)
        elif len(cleaned_word) > 1 and cleaned_word.isalpha() and cleaned_word_lower not in common_non_names_and_roles:
            # If it's a multi-letter alphabetic word and not a common non-name/role,
            # and we haven't found any capitalized parts yet, consider it.
            # This helps with names like "ian" if they appear lowercase in query.
            if not potential_name_parts: # Only add if no capitalized parts have been found yet
                potential_name_parts.append(cleaned_word.capitalize())


    final_names_parts = []
    if len(potential_name_parts) == 1:
        single_part_lower = potential_name_parts[0].lower()
        if single_part_lower in role_keywords_lower: 
            logger.debug(f"_get_person_name_from_query: Filtered out single word '{potential_name_parts[0]}' as it's a role keyword.")
        else: 
            final_names_parts.append(potential_name_parts[0])
    else: 
        final_names_parts = potential_name_parts

    logger.debug(f"_get_person_name_from_query: Query='{query}', Final potential parts={final_names_parts}")

    if final_names_parts:
        return " ".join(final_names_parts).strip()
    return None

# --- Function to extract structured location data based on metadata ---
def _get_location_data_from_json(data_dir: str) -> List[Dict[str, str]]:
    """
    Extracts structured location data by consolidating content from documents
    with the configured 'office_locations' page_title.
    """
    full_alamat_cawangan_content = ""
    
    location_page_title = AGENCY_PAGE_TITLE_MAPPING.get("office_locations")
    if not location_page_title:
        logger.error("[ERROR] 'office_locations' page title not configured in AGENCY_PAGE_TITLE_MAPPING.")
        return []

    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)
        if filename.endswith('.json'): 
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for doc_dict in data:
                            if doc_dict.get('metadata', {}).get('page_title') == location_page_title:
                                # Ensure page_content is a string before appending
                                page_content = doc_dict.get('page_content', '')
                                if isinstance(page_content, str):
                                    full_alamat_cawangan_content += page_content + "\n"
                                else:
                                    logger.warning(f"Skipping non-string page_content for location data consolidation in {filename}: {type(page_content)} - {str(page_content)[:50]}...")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode JSON from {filename}: {e}")
            except Exception as e:
                logger.error(f"Error processing {filename} for location data consolidation: {e}")

    if not full_alamat_cawangan_content:
        logger.debug(f"No '{location_page_title}' content found across JSON files.")
        return []

    locations = []
    
    hq_match = re.search(
        r"IBU PEJABAT JPKN\s*Jabatan Perkhidmatan Komputer Negeri\s*(.*?)(?:No Tel/faks:|Pautan Webmail)",
        full_alamat_cawangan_content, re.DOTALL | re.IGNORECASE
    )
    if hq_match:
        hq_block_raw = hq_match.group(1).strip()
        hq_address_match = re.search(r"(Tingkat.*?SABAH)", hq_block_raw, re.DOTALL | re.IGNORECASE)
        
        hq_tel_fax_matches = re.findall(
            r"(?i)No Tel/faks:\s*([\d\s\+\-]+(?:[\r\n\s]*[\d\s\+\-]+)*)",
            full_alamat_cawangan_content
        )
        hq_tel_fax = ", ".join([t.replace('\n', ', ').strip() for t in hq_tel_fax_matches]) if hq_tel_fax_matches else "N/A"
        
        hq_email_match = re.search(r"(?i)email:\s*([\w\.-]+@[\w\.-]+)", full_alamat_cawangan_content)

        hq_info = {
            "Nama": "IBU PEJABAT JPKN",
            "Alamat": normalize_text(hq_address_match.group(1)) if hq_address_match else "N/A",
            "No Telefon/Faks": hq_tel_fax,
            "Emel": normalize_text(hq_email_match.group(1)) if hq_email_match else "N/A"
        }
        locations.append(hq_info)
        logger.debug(f"Extracted HQ: {hq_info['Nama']}")

    branch_pattern = re.compile(
        r"(Cawangan|Wilayah)\s*(.*?)\s*Alamat:\s*(.*?)(?=(?:Cawangan|Wilayah|No Tel/faks:|Sijil Dan Pengikhtirafan|WAKTU PEJABAT|\Z))",
        re.DOTALL | re.IGNORECASE
    )
    
    for match in branch_pattern.finditer(full_alamat_cawangan_content):
        branch_type = match.group(1).strip()
        branch_name_raw = match.group(2).strip()
        branch_block = match.group(3).strip()

        branch_name = re.sub(r'Alamat:', '', branch_name_raw).strip()
        
        address_match = re.search(r"(.*?)(?:Pejabat Pentadbiran:|Tel:|Emel:|Ketua Cawangan:|Penolong Pegawai Teknologi Maklumat:|Juruteknik Komputer:|Faks:|\Z)", branch_block, re.DOTALL | re.IGNORECASE)
        address = normalize_text(address_match.group(1).replace('\n', ' ').strip()) if address_match else "N/A"
        address = re.sub(r'^\s*JPKN\s*(?:Cawangan|Wilayah)\s*.*?,?\s*', '', address, flags=re.IGNORECASE).strip()
        
        pejabat_pentadbiran_tel = re.search(r"Pejabat Pentadbiran:\s*([\d\s\-\+]+)", branch_block, re.IGNORECASE)
        ketua_cawangan_tel = re.search(r"Ketua Cawangan:\s*([\d\s\-\+]+)", branch_block, re.IGNORECASE)
        penolong_pegawai_tel = re.search(r"Penolong Pegawai Teknologi Maklumat:\s*([\d\s\-\+]+)", branch_block, re.IGNORECASE)
        juruteknik_komputer_tel = re.search(r"Juruteknik Komputer(?:\s*\(kaunter\))?:\s*([\d\s\-\+]+)", branch_block, re.IGNORECASE)
        general_tel = re.search(r"Tel:\s*([\d\s\-\+]+)", branch_block, re.IGNORECASE)
        fax_num = re.search(r"Faks:\s*([\d\s\-\+]+)", branch_block, re.IGNORECASE)
        branch_email = re.search(r"Emel:\s*([\w\.-]+@[\w\.-]+)", branch_block, re.IGNORECASE)
        
        office_info = {
            "Nama Cawangan": f"{branch_type} {branch_name}",
            "Alamat": address,
            "No. Pejabat Pentadbiran": normalize_text(pejabat_pentadbiran_tel.group(1)) if pejabat_pentadbiran_tel else "N/A",
            "No. Ketua Cawangan": normalize_text(ketua_cawangan_tel.group(1)) if ketua_cawangan_tel else "N/A",
            "No. Penolong Pegawai Teknologi Maklumat": normalize_text(penolong_pegawai_tel.group(1)) if penolong_pegawai_tel else "N/A",
            "No. Juruteknik Komputer": normalize_text(juruteknik_komputer_tel.group(1)) if juruteknik_komputer_tel else "N/A",
            "No. Telefon Umum": normalize_text(general_tel.group(1)) if general_tel else "N/A",
            "No. Faks": normalize_text(fax_num.group(1)) if fax_num else "N/A",
            "Emel Cawangan": normalize_text(branch_email.group(1)) if branch_email else "N/A"
        }
        locations.append(office_info)
        logger.debug(f"Extracted Branch: {office_info['Nama Cawangan']}")

    return locations

# --- Keyword Search on Raw JSON Files ---
def _keyword_search_from_json(query: str, data_dir: str, person_name: str | None, specific_rank_in_query: str | None, branch_name_in_query: str | None) -> List[Document]:
    """
    Performs a keyword search directly on raw JSON files in the data directory
    for directory-related content. Prioritizes exact matches.
    """
    results = [] 
    query_lower = query.lower()
    
    directory_page_title = AGENCY_PAGE_TITLE_MAPPING.get("directory")
    office_locations_page_title = AGENCY_PAGE_TITLE_MAPPING.get("office_locations")
    corporate_info_page_title = AGENCY_PAGE_TITLE_MAPPING.get("corporate_info") 

    debug_messages = {
        "director_match": 0, "person_rank_match": 0, "person_match": 0, "rank_match": 0,
        "office_hours_exact": 0, "office_hours_keywords": 0, "location_keywords": 0,
        "location_terms": 0, "attendance_match": 0, "branch_match": 0
    }

    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)
        if not filename.endswith('.json'): 
            continue

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if not isinstance(data, list): 
                    continue
                for doc_dict in data:
                    # Skip attendance records, they are handled by _get_all_attendance_records_from_json
                    if doc_dict.get('metadata', {}).get('source_type') == 'attendance_record':
                        continue 

                    content = doc_dict.get('page_content', '')
                    metadata = doc_dict.get('metadata', {}) 
                    
                    # Ensure content is a string before processing
                    if not isinstance(content, str):
                        logger.warning(f"Skipping document with non-string page_content in {filename}: {type(content)} - {str(content)[:50]}...")
                        continue

                    formatted_content = agency_format_content_case(content) 
                    doc = Document(page_content=formatted_content, metadata=metadata) 
                    doc_content_lower = normalize_text(formatted_content).lower() 
                    
                    score = 0 
                    
                    if metadata.get('page_title') == directory_page_title:
                        doc_jawatan = metadata.get('Jawatan', '').lower()
                        if not doc_jawatan: 
                            jawatan_match_key_value = re.search(r'(?:JAWATAN|Jawatan):\s*(.*?)(?:,|\Z)', doc_content_lower, re.IGNORECASE)
                            jawatan_match_column_3 = re.search(r'Column 3:\s*(.*?)(?:,|\Z)', doc_content_lower, re.IGNORECASE)
                            jawatan_match_column_2 = re.search(r'Column 2:\s*(.*?)(?:,|\Z)', doc_content_lower, re.IGNORECASE)
                            
                            if jawatan_match_key_value:
                                doc_jawatan = jawatan_match_key_value.group(1).strip().lower()
                            elif jawatan_match_column_3:
                                doc_jawatan = jawatan_match_column_3.group(1).strip().lower()
                            elif jawatan_match_column_2:
                                doc_jawatan = jawatan_match_column_2.group(1).strip().lower()

                        for pattern_str, replacement_str, *flags in AGENCY_CLEANING_REGEX:
                            if isinstance(pattern_str, str) and (
                               pattern_str.startswith(r'(DIGITAL)(KEMENTERIAN)') or
                               pattern_str.startswith(r'(TEKNOLOGI)(DAN)') or
                               pattern_str.startswith(r'(SAINS)(TEKNOLOGI)')
                            ):
                                flags = flags[0] if flags else 0
                                doc_jawatan = re.sub(pattern_str, replacement_str.lower(), doc_jawatan, flags=flags)


                        if person_name and person_name.lower() in doc_content_lower: 
                            score += 50 
                            debug_messages["person_match"] += 1
                            if specific_rank_in_query and specific_rank_in_query.lower() in doc_content_lower: 
                                score += 20
                                debug_messages["person_rank_match"] += 1
                            if branch_name_in_query and branch_name_in_query.lower() in doc_content_lower: 
                                score += 20
                                debug_messages["branch_match"] += 1
                        
                        elif specific_rank_in_query and specific_rank_in_query.lower() in doc_jawatan:
                            score += 30 
                            debug_messages["rank_match"] += 1
                            logger.debug(f"Keyword search: Matched specific_rank_in_query '{specific_rank_in_query}' in doc_jawatan '{doc_jawatan}'. Score: {score}")
                            if branch_name_in_query and branch_name_in_query.lower() in doc_content_lower: 
                                score += 15
                                debug_messages["branch_match"] += 1
                        
                        is_query_for_director = ("pengarah" in query_lower or "director" in query_lower)
                        is_doc_exact_director = re.search(r'\b(pengarah|director)\b', doc_jawatan) 
                        is_doc_sub_director = ("timbalan pengarah" in doc_jawatan or "ketua penolong pengarah" in doc_jawatan)

                        if is_query_for_director and is_doc_exact_director and not is_doc_sub_director:
                            score += 2000 
                            debug_messages["director_match"] += 1
                            logger.debug(f"Keyword search: OVERWHELMING BOOST for exact 'pengarah'/'director' in doc_jawatan '{doc_jawatan}'. Score: {score}")
                        elif is_query_for_director and is_doc_sub_director:
                            score -= 500 
                            logger.debug(f"Keyword search: NEGATIVE BOOST for sub-director role '{doc_jawatan}' when querying for main director. Score: {score}")


                    if metadata.get('page_title') == corporate_info_page_title or metadata.get('type') == "office_hours":
                        if any(k in doc_content_lower for k in ["waktu pejabat", "masa bekerja", "7:30 am", "17:00 pm", "13:00", "14:00", "hari bekerja", "cuti umum", "waktu perkhidmatan", "waktu urusan"]): 
                            score += 25 
                            debug_messages["office_hours_exact"] += 1
                        elif any(k in doc_content_lower for k in AGENCY_CONTACT_HOURS_KEYWORDS): 
                            score += 15 
                            debug_messages["office_hours_keywords"] += 1
                    
                    if metadata.get('page_title') == office_locations_page_title or metadata.get('type') == "contact_info":
                        if any(k in doc_content_lower for k in AGENCY_LOCATION_KEYWORDS): 
                            score += 20 
                            debug_messages["location_keywords"] += 1
                        if "ibu pejabat" in doc_content_lower or "cawangan" in doc_content_lower or "wilayah" in doc_content_lower: 
                            score += 10 
                            debug_messages["location_terms"] += 1

                    if score > 0:
                        results.append((doc, score)) 
        except Exception as e:
            logger.error(f"Error reading or processing {filename} for keyword search: {e}")
    
    for msg_type, count in debug_messages.items():
        if count > 0:
            logger.debug(f"Keyword search found {count} {msg_type.replace('_', ' ')}.")

    unique_docs = []
    seen_contents = set()
    results.sort(key=lambda x: x[1], reverse=True)
    for doc, score in results: 
        try:
            if isinstance(doc, Document) and isinstance(doc.page_content, str):
                if doc.page_content not in seen_contents:
                    unique_docs.append(doc)
                    seen_contents.add(doc.page_content) 
            else:
                logger.warning(f"Skipping non-Document or invalid page_content during keyword search deduplication. Type of doc: {type(doc)}, Type of page_content: {type(doc.page_content) if isinstance(doc, Document) else 'N/A'}. Content snippet: {doc.page_content[:50] if isinstance(doc, Document) and isinstance(doc.page_content, str) else 'N/A'}")
        except TypeError as e:
            logger.error(f"TypeError during keyword search deduplication: {e}. Doc type: {type(doc)}, Page content type: {type(doc.page_content) if isinstance(doc, Document) else 'N/A'}. Content snippet: {doc.page_content[:50] if isinstance(doc, Document) and isinstance(doc.page_content, str) else 'N/A'}")
            continue 
    
    return unique_docs

# --- Function to extract all attendance records from raw JSON ---
def _get_all_attendance_records_from_json(data_dir: str, employee_id: str | None = None, target_year: str | None = None) -> List[Document]:
    """
    Extracts all attendance records from the raw JSON file, optionally filtering by employee ID and year.
    Dynamically finds the attendance file by looking for 'attendance' in the filename.
    """
    attendance_docs = []
    filepath = None

    # Dynamically find attendance file
    for filename in os.listdir(data_dir):
        if "attendance" in filename.lower() and filename.endswith('.json'):
            filepath = os.path.join(data_dir, filename)
            logger.info(f"Found attendance data file: {filepath}")
            break
    
    if not filepath:
        logger.warning(f"No attendance data file containing 'attendance' found in {data_dir}.")
        return []

    try: 
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if not isinstance(data, list):
                logger.error(f"Attendance data in {filepath} is not a list of records.")
                return []

            for record in data: # Each 'record' here is a dict like {'page_content': '...', 'metadata': {...}}
                record_metadata = record.get('metadata', {})
                if not all(k in record_metadata for k in ['emp_id', 'icno', 'att_date']):
                    logger.warning(f"Skipping malformed attendance record (missing key in metadata): {record}")
                    continue

                record_emp_id = str(record_metadata.get('emp_id', '')).strip()
                record_icno = str(record_metadata.get('icno', '')).strip()
                record_date = record_metadata.get('att_date', '')
                record_year = record_date[:4] if record_date else None

                # Filter by employee ID if provided
                if employee_id and not (record_emp_id == employee_id or record_icno == employee_id):
                    continue

                # Filter by year if provided
                if target_year and record_year != target_year:
                    continue

                # Create a Document object for each relevant record
                # We'll use the raw record as page_content and its fields as metadata
                doc_content = json.dumps(record, ensure_ascii=False) # Store raw record as content for LLM to parse
                metadata = {
                    "source_type": "attendance_record",
                    "original_filename": os.path.basename(filepath), # Use the actual filename found
                    "id": record_metadata.get('id'), # Use original ID if available
                    "emp_id": record_emp_id,
                    "icno": record_icno,
                    "att_date": record_date,
                    "clock_in": record_metadata.get('clock_in'),
                    "clock_out": record_metadata.get('clock_out'),
                    "total_hours": record_metadata.get('total_hours'),
                    "incomplete_clock_in_out": record_metadata.get('incomplete_clock_in_out'),
                    "late_clock_in": record_metadata.get('late_clock_in'),
                    "early_clock_out": record_metadata.get('early_clock_out'),
                    "cuti": record_metadata.get('cuti'),
                    "cuti_type": record_metadata.get('cuti_type'),
                    "public_holiday": record_metadata.get('public_holiday'),
                    "catatan": record_metadata.get('catatan'),
                    # Include other relevant metadata fields as needed
                }
                attendance_docs.append(Document(page_content=doc_content, metadata=metadata))
            
            logger.info(f"Direct JSON search for attendance found {len(attendance_docs)} records for employee ID '{employee_id}' and year '{target_year}'.")
            return attendance_docs

    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode attendance JSON from {filepath}: {e}")
        return []
    except Exception as e:
        logger.error(f"Error processing attendance data from {filepath}: {e}")
        return []


def _post_filter_directory_docs(
    documents: List[Document], 
    query: str, 
    person_name: str | None = None,
    specific_rank_in_query: str | None = None,
    branch_name_in_query: str | None = None 
) -> List[Document]:
    """
    Post-filters directory documents to prioritize those most relevant to the query,
    especially for specific person or rank queries, and returns all relevant for counting.
    """
    filtered_docs = []
    query_lower = query.lower()

    content_role_regex = re.compile(r'(?i)' + '|'.join(re.escape(k).replace(r'\ ', r'.*?') for k in AGENCY_ROLE_KEYWORDS))

    is_counting_query = any(word in query_lower for word in ["how many", "berapa", "bilangan", "jumlah"])

    scored_docs = []
    debug_messages = { # Initialize debug_messages for this function
        "director_match": 0, "person_rank_match": 0, "person_match": 0, "rank_match": 0,
        "branch_match": 0, "general_role_match": 0, "counting_rank_match": 0
    }

    for doc in documents:
        if not isinstance(doc.page_content, str):
            logger.warning(f"Skipping document with non-string page_content in post-filter: {type(doc.page_content)} - {str(doc.page_content)[:50]}...")
            continue

        doc_content_lower = normalize_text(doc.page_content).lower() 
        metadata = doc.metadata
        match_score = 0
        
        doc_jawatan = metadata.get('Jawatan', '').lower()
        if not doc_jawatan: 
            jawatan_match_key_value = re.search(r'(?:JAWATAN|Jawatan):\s*(.*?)(?:,|\Z)', doc_content_lower, re.IGNORECASE)
            jawatan_match_column_3 = re.search(r'Column 3:\s*(.*?)(?:,|\Z)', doc_content_lower, re.IGNORECASE)
            jawatan_match_column_2 = re.search(r'Column 2:\s*(.*?)(?:,|\Z)', doc_content_lower, re.IGNORECASE)
            
            if jawatan_match_key_value:
                doc_jawatan = jawatan_match_key_value.group(1).strip().lower()
            elif jawatan_match_column_3:
                doc_jawatan = jawatan_match_column_3.group(1).strip().lower()
            elif jawatan_match_column_2:
                doc_jawatan = jawatan_match_column_2.group(1).strip().lower()

        for pattern_str, replacement_str, *flags in AGENCY_CLEANING_REGEX:
            if isinstance(pattern_str, str) and (
               pattern_str.startswith(r'(DIGITAL)(KEMENTERIAN)') or
               pattern_str.startswith(r'(TEKNOLOGI)(DAN)') or
               pattern_str.startswith(r'(SAINS)(TEKNOLOGI)')
            ):
                flags = flags[0] if flags else 0
                doc_jawatan = re.sub(pattern_str, replacement_str.lower(), doc_jawatan, flags=flags)


        if person_name and person_name.lower() in doc_content_lower: 
            match_score += 50
            debug_messages["person_match"] += 1
            if specific_rank_in_query and specific_rank_in_query.lower() in doc_content_lower: 
                match_score += 20
                debug_messages["person_rank_match"] += 1
            if branch_name_in_query and branch_name_in_query.lower() in doc_content_lower: 
                match_score += 20
                debug_messages["branch_match"] += 1
        
        elif specific_rank_in_query and specific_rank_in_query.lower() in doc_jawatan:
            match_score += 30 
            debug_messages["rank_match"] += 1
            logger.debug(f"Post-filter: Boost for rank '{specific_rank_in_query}' in doc Jawatan: '{doc_jawatan}'. Score: {match_score}")
            if branch_name_in_query and branch_name_in_query.lower() in doc_content_lower: 
                match_score += 15
                debug_messages["branch_match"] += 1
        
        if branch_name_in_query and branch_name_in_query.lower() in doc_content_lower: 
            match_score += 10 
            debug_messages["branch_match"] += 1

        is_query_for_director = ("pengarah" in query_lower or "director" in query_lower)
        is_doc_exact_director = re.search(r'\b(pengarah|director)\b', doc_jawatan) 
        is_doc_sub_director = ("timbalan pengarah" in doc_jawatan or "ketua penolong pengarah" in doc_jawatan)

        if is_query_for_director and is_doc_exact_director and not is_doc_sub_director:
            match_score += 5000 
            debug_messages["director_match"] += 1
            logger.debug(f"Post-filter: OVERWHELMING BOOST for exact 'pengarah'/'director' in doc_jawatan '{doc_jawatan}'. Score: {match_score}")
        elif is_query_for_director and is_doc_sub_director:
            match_score -= 1000 
            logger.debug(f"Post-filter: SIGNIFICANT NEGATIVE BOOST for sub-director role '{doc_jawatan}' when querying for main director. Score: {match_score}")

        if metadata.get("page_title") == AGENCY_PAGE_TITLE_MAPPING.get("directory") and content_role_regex.search(doc_content_lower): 
            match_score += 1 
            debug_messages["general_role_match"] += 1
        
        # --- NEW FIX: For counting queries, collect all matching documents ---
        if is_counting_query and specific_rank_in_query:
            # Check if the document's Jawatan (case-insensitive) contains the specific_rank_in_query
            # Use re.search with word boundaries for more robust matching of "digital apprentice"
            if re.search(r'\b' + re.escape(specific_rank_in_query.lower()) + r'\b', doc_jawatan):
                match_score += 10000 # Give an extremely high score to ensure it's always included
                debug_messages["counting_rank_match"] += 1
                logger.debug(f"Post-filter: COUNTING QUERY - EXTREME BOOST for rank '{specific_rank_in_query}' in doc Jawatan: '{doc_jawatan}'")

        if match_score > 0:
            scored_docs.append((doc, match_score))
    
    for msg_type, count in debug_messages.items(): 
        if count > 0:
            logger.debug(f"Post-filter (directory) found {count} {msg_type.replace('_', ' ')}.")

    if scored_docs:
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        if is_counting_query and specific_rank_in_query:
            logger.debug(f"Post-filter detected counting query for specific rank: '{specific_rank_in_query}'. Returning all documents with counting_rank_match score.")
            # Return all documents that received the extreme boost for counting
            return [doc for doc, score in scored_docs if score >= 10000] 
        else: 
            logger.debug(f"Post-filter found {len(scored_docs)} documents matching specific rank/name/branch criteria. Returning top 5.")
            return [doc for doc, _ in scored_docs[:5]] 
        
    general_filtered_docs = []
    for doc in documents:
        if doc.metadata.get("page_title") == AGENCY_PAGE_TITLE_MAPPING.get("directory"):
            if not isinstance(doc.page_content, str):
                logger.warning(f"Skipping document with non-string page_content in general filter: {type(doc.page_content)} - {str(doc.page_content)[:50]}...")
                continue
            doc_content_lower = normalize_text(doc.page_content).lower() 
            if content_role_regex.search(doc_content_lower): 
                general_filtered_docs.append(doc)
    
    general_filtered_docs.sort(key=lambda x: x.metadata.get("type") == "table_data", reverse=True)

    if not general_filtered_docs and documents:
        logger.debug("Post-filter yielded no specific directory matches. Returning top 3 from initial retrieval.")
        return documents[:3]

    if is_counting_query:
        logger.debug(f"Post-filter detected counting query. Returning all {len(general_filtered_docs)} general directory documents.")
        return general_filtered_docs
    return general_filtered_docs[:5] 

def _post_filter_contact_hours_docs(documents: List[Document], query: str) -> List[Document]:
    """
    Post-filters documents to prioritize those containing contact information or office hours.
    Prioritizes static office hours data.
    """
    filtered_docs = []
    query_lower = query.lower()
    
    scored_docs = []
    debug_messages = { 
        "static_office_hours": 0, "office_hours_exact": 0, "office_hours_keywords": 0,
        "header_footer_type": 0, "main_content_type": 0
    }

    for doc in documents:
        if not isinstance(doc.page_content, str):
            logger.warning(f"Skipping document with non-string page_content in contact hours filter: {type(doc.page_content)} - {str(doc.page_content)[:50]}...")
            continue

        doc_content_lower = normalize_text(doc.page_content).lower() 
        score = 0
        
        if doc.metadata.get('source_type') == 'static_config' and doc.metadata.get('type') == 'office_hours':
            score += 100 
            debug_messages["static_office_hours"] += 1
            logger.debug("Prioritizing static office hours document.")
        
        if any(k in doc_content_lower for k in ["waktu pejabat", "masa bekerja", "7:30 am", "17:00 pm", "13:00", "14:00", "hari bekerja", "cuti umum", "waktu perkhidmatan", "waktu urusan"]): 
            score += 25
            debug_messages["office_hours_exact"] += 1
        elif any(k in doc_content_lower for k in AGENCY_CONTACT_HOURS_KEYWORDS): 
            score += 15 
            debug_messages["office_hours_keywords"] += 1
        
        if doc.metadata.get("type") in ["header", "footer", "office_hours"]: 
            score += 10
            debug_messages["header_footer_type"] += 1
        elif doc.metadata.get("type") == "main_content":
            score += 5
            debug_messages["main_content_type"] += 1
        
        if score > 0:
            scored_docs.append((doc, score))
            
    for msg_type, count in debug_messages.items(): 
        if count > 0:
            logger.debug(f"Post-filter (contact hours) found {count} {msg_type.replace('_', ' ')}.")

    if scored_docs:
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        logger.debug(f"Post-filter found {len(scored_docs)} scored docs for contact/hours.")
        return [doc for doc, _ in scored_docs[:5]] 
    
    if not documents: 
        logger.debug("Post-filter received no documents for contact/hours.")
        return []

    logger.debug("Post-filter yielded no specific contact/hours matches. Returning top 3 from initial retrieval.")
    return documents[:3] 

def _post_filter_location_docs(documents: List[Document], query: str) -> List[List[Document]]:
    """
    Post-filters location documents to prioritize the most relevant ones.
    Returns a list of lists, where each inner list contains documents for a specific location.
    """
    filtered_docs_by_location = []
    query_lower = query.lower()

    specific_location_keywords = AGENCY_LOCATION_KEYWORDS 
    
    # Group documents by a unique identifier for each location (e.g., Nama Cawangan or Nama for HQ)
    grouped_docs: Dict[str, List[Document]] = {}
    for doc in documents:
        location_id = doc.metadata.get('Nama Cawangan') or doc.metadata.get('Nama')
        if location_id:
            if location_id not in grouped_docs:
                grouped_docs[location_id] = []
            grouped_docs[location_id].append(doc)

    scored_locations = [] 
    debug_messages = { 
        "specific_location_match": 0, "general_location_keywords": 0, "office_locations_page": 0
    }

    for location_id, docs_for_location in grouped_docs.items():
        location_score = 0
        
        location_content_lower_parts = []
        for doc in docs_for_location:
            if isinstance(doc.page_content, str):
                location_content_lower_parts.append(normalize_text(doc.page_content).lower())
            else:
                logger.warning(f"Skipping non-string page_content for location doc: {type(doc.page_content)} - {str(doc.page_content)[:50]}...")
        location_content_lower = " ".join(location_content_lower_parts)


        if any(doc.metadata.get("page_title") == AGENCY_PAGE_TITLE_MAPPING.get("office_locations") or doc.metadata.get("type") == "contact_info" for doc in docs_for_location): 
            location_score += 10
            debug_messages["office_locations_page"] += 1

        for kw in specific_location_keywords:
            if kw in query_lower and kw in location_content_lower:
                location_score += 15 
                debug_messages["specific_location_match"] += 1

        if any(k in location_content_lower for k in AGENCY_LOCATION_KEYWORDS):
            location_score += 5
            debug_messages["general_location_keywords"] += 1
        
        if location_score > 0:
            scored_locations.append((location_id, location_score))
            
    for msg_type, count in debug_messages.items(): 
        if count > 0:
            logger.debug(f"Post-filter (location) found {count} {msg_type.replace('_', ' ')}.")

    if scored_locations:
        scored_locations.sort(key=lambda x: x[1], reverse=True)
        logger.debug(f"Post-filter found {len(scored_locations)} scored locations.")
        
        for location_id, _ in scored_locations[:3]:
            filtered_docs_by_location.append(grouped_docs[location_id])
        return filtered_docs_by_location
    
    if not documents: 
        logger.debug("Post-filter received no documents for locations.")
        return []

    logger.debug("Post-filter yielded no specific location matches. Returning top 3 from initial retrieval.")
    for location_id, docs in grouped_docs.items():
        filtered_docs_by_location.append(docs)
    return filtered_docs_by_location[:3] 

def _post_filter_attendance_docs(documents: List[Document], query: str) -> List[Document]:
    """
    Post-filters attendance documents to prioritize those most relevant to the query.
    Ensures all relevant records are returned for average calculation.
    Filters by year if specified in the query.
    Note: Initial filtering by employee ID and year is now done by _get_all_attendance_records_from_json.
    This function primarily sorts and prepares for LLM.
    """
    filtered_docs = []
    query_lower = query.lower()
    
    id_match = re.search(r'\b(?:ic\s*number|nombor\s*ic|ic|employee\s*id|emp_id)\s*(\d+)\b', query_lower)
    target_id = id_match.group(1) if id_match else None

    date_match = re.search(r'\b(\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4})\b', query_lower) 
    target_date = date_match.group(1) if date_match else None
    if target_date and '/' in target_date: 
        day, month, year = target_date.split('/')
        target_date = f"{year}-{month}-{day}"

    year_match = re.search(r'\b(20\d{2})\b', query_lower) 
    target_year = year_match.group(1) if year_match else None

    is_average_query = any(word in query_lower for word in ["average", "purata", "mean", "rata-rata", "average work hour", "purata jam bekerja"])
    is_count_times_query = any(word in query_lower for word in ["how many times", "berapa kali", "count times", "bilangan kali", "work days", "hari bekerja", "didn't come to work", "tidak datang kerja"])

    logger.debug(f"[_post_filter_attendance_docs] Query: '{query}'")
    logger.debug(f"[_post_filter_attendance_docs] is_average_query: {is_average_query}")
    logger.debug(f"[_post_filter_attendance_docs] is_count_times_query: {is_count_times_query}")
    logger.debug(f"[_post_filter_attendance_docs] target_id: {target_id}")
    logger.debug(f"[_post_filter_attendance_docs] target_year: {target_year}")
    logger.debug(f"[_post_filter_attendance_docs] Number of initial documents to score: {len(documents)}")

    scored_docs = []
    debug_messages = { 
        "base_attendance_score": 0, "date_match": 0, "keyword_match": 0,
        "late_clock_in_match": 0, "leave_match": 0
    }

    for doc in documents:
        if doc.metadata.get('source_type') != 'attendance_record':
            logger.warning(f"Skipping non-attendance document in _post_filter_attendance_docs: {doc.metadata.get('source_type')}")
            continue

        if not isinstance(doc.page_content, str):
            logger.warning(f"Skipping document with non-string page_content in attendance filter: {type(doc.page_content)} - {str(doc.page_content)[:50]}...")
            continue

        try:
            record_data = json.loads(doc.page_content)
            metadata = doc.metadata 
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse attendance record JSON from page_content for scoring: {doc.page_content[:100]}...")
            continue 

        score = 0
        score += 20 
        debug_messages["base_attendance_score"] += 1

        if target_date:
            doc_date = metadata.get('att_date')
            if doc_date and doc_date == target_date:
                score += 20 
                debug_messages["date_match"] += 1
        
        for kw in AGENCY_ATTENDANCE_KEYWORDS: 
            if kw in query_lower and kw in doc.page_content.lower(): 
                score += 5
                debug_messages["keyword_match"] += 1
        
        if "late" in query_lower or "lewat" in query_lower or "lambat" in query_lower:
            if record_data.get('late_clock_in') is True:
                score += 15
                debug_messages["late_clock_in_match"] += 1
        if "cuti" in query_lower or "leave" in query_lower or "tidak datang kerja" in query_lower or "didn't come to work" in query_lower:
            if record_data.get('cuti') is True or record_data.get('public_holiday') is True or record_data.get('total_hours') == 0:
                score += 15
                debug_messages["leave_match"] += 1

        if score > 0:
            scored_docs.append((doc, score))
    
    for msg_type, count in debug_messages.items(): 
        if count > 0:
            logger.debug(f"Post-filter (attendance) found {count} {msg_type.replace('_', ' ')}.")

    logger.debug(f"[_post_filter_attendance_docs] Number of scored documents: {len(scored_docs)}")

    if scored_docs:
        scored_docs.sort(key=lambda x: (x[1], x[0].metadata.get('att_date', '0000-00-00')), reverse=True)
        
        if (is_average_query or is_count_times_query) and target_id:
            logger.debug(f"[_post_filter_attendance_docs] Average/Count query for ID '{target_id}' detected. Returning all {len(documents)} initial documents for calculation.")
            return documents 
        
        else: 
            logger.debug(f"[_post_filter_attendance_docs] Not an average/count query. Returning top 10 scored docs.")
            return [doc for doc, _ in scored_docs[:10]] 
    
    logger.debug(f"[_post_filter_attendance_docs] No scored attendance docs found. Returning empty list.")
    return []


# --- Main Retrieval Function ---
def retrieve_context(query: str, k: int = 10) -> List[Document]: # Default k back to 10 for general queries
    """
    Retrieves relevant context from the vector store based on the query.
    Applies query expansion and dynamic filtering based on detected intent.
    This function now orchestrates hybrid search and intent-based filtering.
    """
    vectorstore = _get_vector_store()
    # Check if vectorstore is available and embeddings are working
    if vectorstore is None or not _ollama_embeddings_available:
        logger.warning("Vector store or Ollama embeddings not available. Falling back to keyword search only.")
        person_name = _get_person_name_from_query(query)
        specific_rank_in_query = _get_specific_rank_from_query(query) 
        branch_name_in_query = extract_branch_name_from_query(query) 
        keyword_fallback_docs = _keyword_search_from_json(query, DATA_DIR, person_name, specific_rank_in_query, branch_name_in_query)
        
        # Log retrieved filenames for fallback
        if keyword_fallback_docs:
            logger.info(f"Keyword fallback retrieved {len(keyword_fallback_docs)} documents:")
            for doc_idx, doc in enumerate(keyword_fallback_docs):
                logger.info(f"   Doc {doc_idx + 1} (File: {doc.metadata.get('original_filename', 'N/A')}, Page Title: {doc.metadata.get('page_title', 'N/A')}, Source Type: {doc.metadata.get('source_type', 'N/A')})")
        else:
            logger.info("Keyword fallback retrieved 0 documents.")
        return keyword_fallback_docs[:k]


    expanded_query = expand_query(query)
    logger.info(f"Original Query: '{query}'")
    logger.info(f"Expanded Query: '{expanded_query}'")

    person_name = _get_person_name_from_query(query)
    specific_rank_in_query = _get_specific_rank_from_query(query) 
    branch_name_in_query = extract_branch_name_from_query(query) 
    
    is_counting_query = any(word in query.lower() for word in ["how many", "berapa", "bilangan", "jumlah"]) 
    is_directory_role_query = _is_directory_query(query) 
    is_contact_hours_query = _is_contact_hours_query(query)
    is_location_query = _is_location_query(query)
    is_attendance_query_detected = _is_attendance_query(query)
    
    logger.debug(f"get_relevant_documents: Query='{query}'")
    logger.debug(f"   _is_directory_query(query) result: {is_directory_role_query}")
    logger.debug(f"   _is_contact_hours_query(query) result: {is_contact_hours_query}")
    logger.debug(f"   _is_location_query(query) result: {is_location_query}")
    logger.debug(f"   _is_attendance_query(query) result: {is_attendance_query_detected}")
    logger.debug(f"   Detected person_name: '{person_name}'")
    logger.debug(f"   Detected specific_rank_in_query: '{specific_rank_in_query}'") 
    logger.debug(f"   Detected branch_name_in_query: '{branch_name_in_query}'") 

    filters = {}
    k_for_similarity_search = k 
    initial_retrieved_docs = []
    keyword_docs_for_initial_combine = [] 

    employee_id_for_attendance = None
    target_year_for_attendance = None
    if is_attendance_query_detected:
        id_match = re.search(r'\b(?:employee|emp_id|ic|ic\s*number|nombor\s*ic)\s*(\d+)\b', query.lower())
        employee_id_for_attendance = id_match.group(1) if id_match else None
        
        if not employee_id_for_attendance:
            id_only_match = re.search(r'\bemployee\s*(\d+)\b', query.lower())
            employee_id_for_attendance = id_only_match.group(1) if id_only_match else employee_id_for_attendance
        
        if employee_id_for_attendance:
            logger.info(f"Attendance query detected. Extracted employee ID: '{employee_id_for_attendance}'.")
        else:
            logger.warning(f"Attendance query detected but no employee ID found in query: '{query}'. Cannot filter precisely for direct JSON search.")

        year_match = re.search(r'\b(20\d{2})\b', query.lower()) 
        target_year_for_attendance = year_match.group(1) if year_match else None

        logger.info(f"Primary intent: Attendance. Performing direct JSON search for all relevant records.")
        initial_retrieved_docs = _get_all_attendance_records_from_json(DATA_DIR, employee_id_for_attendance, target_year_for_attendance)
        logger.info(f"Direct JSON search for attendance retrieved {len(initial_retrieved_docs)} documents.")
        
        retrieved_docs = _post_filter_attendance_docs(initial_retrieved_docs, query)
        
        logger.info(f"Final retrieved documents after attendance-specific filtering: {len(retrieved_docs)} documents.")
        for doc_idx, doc in enumerate(retrieved_docs):
            logger.info(f"   Doc {doc_idx + 1} (File: {doc.metadata.get('original_filename', 'N/A')}, Page Title: {doc.metadata.get('page_title', 'N/A')}, Source Type: {doc.metadata.get('source_type', 'N/A')})")
        return retrieved_docs
        
    elif is_directory_role_query: 
        logger.info("Primary intent: Directory/Role. Filtering for type: 'table_data'.")
        filters = {"type": "table_data"} 
        # --- NEW FIX: Set k_for_similarity_search to retrieve all documents if counting query ---
        if is_counting_query and specific_rank_in_query:
            try:
                # Attempt to get the total count of documents in the collection
                k_for_similarity_search = vectorstore._collection.count()
                logger.info(f"Counting query for directory role detected. Setting k_for_similarity_search to total documents in collection: {k_for_similarity_search}.")
            except Exception as e:
                logger.warning(f"Could not get total document count from ChromaDB: {e}. Falling back to large fixed k.")
                k_for_similarity_search = 2000 # Fallback to a very large number
        else:
            k_for_similarity_search = max(k, 100) 
        # --- END NEW FIX ---
        
        try:
            initial_retrieved_docs = vectorstore.similarity_search(expanded_query, k=k_for_similarity_search, filter=filters)
            logger.info(f"Retrieved {len(initial_retrieved_docs)} documents from vector store via similarity search for directory.")
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error during directory retrieval (similarity search): {e}.")
            initial_retrieved_docs = [] 
        except Exception as e:
            logger.error(f"General error during directory retrieval (similarity search): {e}.")
            initial_retrieved_docs = [] 

    elif is_location_query:
        logger.info(f"Primary intent: Location.")
        structured_locations = _get_location_data_from_json(DATA_DIR)
        
        if structured_locations:
            logger.debug(f"Structured location data found: {len(structured_locations)} entries.")
            location_docs = []
            query_lower = query.lower()
            
            specific_location_match_found = False
            for loc_data in structured_locations:
                branch_name_lower = loc_data.get('Nama Cawangan', '').lower()
                hq_name_lower = loc_data.get('Nama', '').lower()

                match_for_this_loc = False
                for kw in AGENCY_LOCATION_KEYWORDS:
                    if kw in query_lower and (kw in branch_name_lower or kw in hq_name_lower):
                        match_for_this_loc = True
                        break
                
                if branch_name_in_query and (branch_name_in_query.lower() in branch_name_lower or branch_name_in_query.lower() in hq_name_lower):
                    match_for_this_loc = True

                if match_for_this_loc:
                    content_parts = []
                    for key, value in loc_data.items():
                        if value and value != "N/A": 
                            content_parts.append(f"{key}: {value}")
                    general_description = f"Location information for {loc_data.get('Nama', loc_data.get('Nama Cawangan', 'JPKN office'))}."
                    location_docs.append(Document(
                        page_content=f"{general_description}\n" + "\n".join(content_parts),
                        metadata={"source_type": "structured_location", "page_title": AGENCY_PAGE_TITLE_MAPPING.get("office_locations"), **loc_data} 
                    ))
                    specific_location_match_found = True
            
            if location_docs:
                retrieved_grouped_docs = _post_filter_location_docs(location_docs, query)
                retrieved_docs = [doc for sublist in retrieved_grouped_docs for doc in sublist]
                logger.debug(f"Post-filter refined to {len(retrieved_docs)} results for location query from structured data.")
                return retrieved_docs
            elif not specific_location_match_found and structured_locations:
                logger.debug("No specific branch matched in query, returning all structured locations.")
                all_location_docs = []
                for loc_data in structured_locations:
                    content_parts = []
                    for key, value in loc_data.items():
                        if value and value != "N/A":
                            content_parts.append(f"{key}: {value}")
                    general_description = f"Location information for {loc_data.get('Nama', loc_data.get('Nama Cawangan', 'JPKN office'))}."
                    all_location_docs.append(Document(
                        page_content=f"{general_description}\n" + "\n".join(content_parts),
                        metadata={"source_type": "structured_location", "page_title": AGENCY_PAGE_TITLE_MAPPING.get("office_locations"), **loc_data} 
                    ))
                retrieved_grouped_docs = _post_filter_location_docs(all_location_docs, query)
                retrieved_docs = [doc for sublist in retrieved_grouped_docs for doc in sublist]
                return retrieved_docs
            else:
                logger.debug("No structured location data found. Falling back to general keyword search for locations.")
                filters = {"type": {"$in": ["contact_info", "main_content", "footer", "office_locations"]}}
                try:
                    initial_retrieved_docs = vectorstore.similarity_search(expanded_query, k=k_for_similarity_search, filter=filters)
                    logger.info(f"Retrieved {len(initial_retrieved_docs)} documents from vector store via similarity search for location fallback.")
                except requests.exceptions.ConnectionError as e:
                    logger.error(f"Connection error during location fallback retrieval (similarity search): {e}.")
                    initial_retrieved_docs = [] 
                except Exception as e:
                    logger.error(f"General error during location fallback retrieval (similarity search): {e}.")
                    initial_retrieved_docs = [] 
        
    elif is_contact_hours_query:
        filters = {"source_type": "static_config", "type": "office_hours"}
        logger.info("Detected 'contact/hours' query. Prioritizing static office hours document.")
        try:
            initial_retrieved_docs = vectorstore.similarity_search(expanded_query, k=k_for_similarity_search, filter=filters)
            logger.info(f"Retrieved {len(initial_retrieved_docs)} documents from vector store via similarity search for contact/hours.")
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error during contact/hours retrieval (similarity search): {e}.")
            initial_retrieved_docs = [] 
        except Exception as e:
            logger.error(f"General error during contact/hours retrieval (similarity search): {e}.")
            initial_retrieved_docs = [] 
        
    else: 
        filters = {"type": {"$in": ["main_content", "header", "footer", "table_data", "contact_info", "office_hours", "collapsible_section"]}}
        logger.info("No specific intent detected. Searching across all relevant document types.")
        try:
            initial_retrieved_docs = vectorstore.similarity_search(expanded_query, k=k_for_similarity_search, filter=filters)
            logger.info(f"Retrieved {len(initial_retrieved_docs)} documents from vector store via similarity search for general query.")
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error during general retrieval (similarity search): {e}.")
            initial_retrieved_docs = [] 
        except Exception as e:
            logger.error(f"General error during general retrieval (similarity search): {e}.")
            initial_retrieved_docs = [] 

    combined_docs = []
    seen_ids = set() 
    seen_content_hashes = set() 

    for doc in initial_retrieved_docs:
        try:
            if not isinstance(doc.page_content, str):
                logger.warning(f"Skipping non-string page_content from initial_retrieved_docs: {type(doc.page_content)} - {str(doc.page_content)[:50]}...")
                continue

            if doc.metadata.get('source_type') == 'attendance_record' and doc.metadata.get('id'):
                if doc.metadata['id'] not in seen_ids:
                    combined_docs.append(doc)
                    seen_ids.add(doc.metadata['id'])
            else:
                doc_key = f"{doc.page_content}_{doc.metadata.get('url', '')}"
                content_hash = hash(doc_key)
                if content_hash not in seen_content_hashes:
                    combined_docs.append(doc)
                    seen_content_hashes.add(content_hash)
        except TypeError as e:
            logger.error(f"TypeError when adding to seen_contents from initial_retrieved_docs: {e}. Doc type: {type(doc)}, Content type: {type(doc.page_content) if isinstance(doc, Document) else 'N/A'}. Content snippet: {doc.page_content[:50] if isinstance(doc, Document) and isinstance(doc.page_content, str) else 'N/A'}")
            continue 
    
    for doc in keyword_docs_for_initial_combine: 
        try:
            if not isinstance(doc.page_content, str):
                logger.warning(f"Skipping non-string page_content from keyword_docs_for_initial_combine: {type(doc.page_content)} - {str(doc.page_content)[:50]}...")
                continue

            if doc.metadata.get('source_type') == 'attendance_record' and doc.metadata.get('id'):
                if doc.metadata['id'] not in seen_ids:
                    combined_docs.append(doc)
                    seen_ids.add(doc.metadata['id'])
                else: 
                    continue
            else:
                doc_key = f"{doc.page_content}_{doc.metadata.get('url', '')}"
                content_hash = hash(doc_key)
                if content_hash not in seen_content_hashes:
                    combined_docs.append(doc)
                    seen_content_hashes.add(content_hash)
        except TypeError as e:
            logger.error(f"TypeError when adding to seen_contents from keyword_docs_for_initial_combine: {e}. Doc type: {type(doc)}, Content type: {type(doc.page_content) if isinstance(doc, Document) else 'N/A'}. Content snippet: {doc.page_content[:50] if isinstance(doc, Document) and isinstance(doc.page_content, str) else 'N/A'}")
            continue 

    logger.info("--- Retrieved Documents (Before Post-Filtering) ---")
    if not combined_docs:
        logger.info("No documents retrieved by similarity or keyword search.")
    else:
        logger.info(f"Initial combined retrieval found {len(combined_docs)} documents.")
        for doc_idx, doc in enumerate(combined_docs):
            logger.info(f"   Doc {doc_idx + 1} (File: {doc.metadata.get('original_filename', 'N/A')}, Page Title: {doc.metadata.get('page_title', 'N/A')}, Source Type: {doc.metadata.get('source_type', 'N/A')})")
    logger.info("--- End Initial Retrieved Documents ---")

    retrieved_docs = [] 

    if is_directory_role_query: 
        retrieved_docs = _post_filter_directory_docs(combined_docs, query, person_name, specific_rank_in_query, branch_name_in_query)
    elif is_contact_hours_query:
        retrieved_docs = _post_filter_contact_hours_docs(combined_docs, query)
    elif is_location_query: 
        retrieved_grouped_docs = _post_filter_location_docs(combined_docs, query)
        retrieved_docs = [doc for sublist in retrieved_grouped_docs for doc in sublist] 
    else:
        retrieved_docs = combined_docs 

    if not retrieved_docs or len(retrieved_docs) < 2: 
        logger.info("Post-filtering yielded too few results. Performing a broader keyword search as final fallback.")
        final_fallback_keyword_docs = _keyword_search_from_json(query, DATA_DIR, None, None, None) 
        
        for doc in final_fallback_keyword_docs:
            try:
                if not isinstance(doc.page_content, str):
                    logger.warning(f"Skipping non-string page_content from final_fallback_keyword_docs: {type(doc.page_content)} - {str(doc.page_content)[:50]}...")
                    continue

                if doc.metadata.get('source_type') == 'attendance_record' and doc.metadata.get('id'):
                    if doc.metadata['id'] not in seen_ids:
                        retrieved_docs.append(doc)
                        seen_ids.add(doc.metadata['id'])
                    else: 
                        continue
                else:
                    doc_key = f"{doc.page_content}_{doc.metadata.get('url', '')}"
                    content_hash = hash(doc_key)
                    if content_hash not in seen_content_hashes:
                        retrieved_docs.append(doc)
                        seen_content_hashes.add(content_hash)
            except TypeError as e:
                logger.error(f"TypeError when adding to seen_contents from final_fallback_keyword_docs: {e}. Doc type: {type(doc)}, Content type: {type(doc.page_content) if isinstance(doc, Document) else 'N/A'}. Content snippet: {doc.page_content[:50] if isinstance(doc, Document) and isinstance(doc.page_content, str) else 'N/A'}")
                continue 
        
        if is_directory_role_query:
            retrieved_docs = _post_filter_directory_docs(retrieved_docs, query, person_name, specific_rank_in_query, branch_name_in_query)
        elif is_contact_hours_query:
            retrieved_docs = _post_filter_contact_hours_docs(retrieved_docs, query)
        elif is_location_query:
            retrieved_grouped_docs = _post_filter_location_docs(retrieved_docs, query)
            retrieved_docs = [doc for sublist in retrieved_grouped_docs for doc in sublist] 
        
        logger.info(f"Final fallback retrieved {len(retrieved_docs)} documents.")
        for doc_idx, doc in enumerate(retrieved_docs):
            logger.info(f"   Doc {doc_idx + 1} (File: {doc.metadata.get('original_filename', 'N/A')}, Page Title: {doc.metadata.get('page_title', 'N/A')}, Source Type: {doc.metadata.get('source_type', 'N/A')})")

        return retrieved_docs 

    logger.info(f"Final retrieved documents after all filtering: {len(retrieved_docs)} documents.")
    for doc_idx, doc in enumerate(retrieved_docs):
        logger.info(f"   Doc {doc_idx + 1} (File: {doc.metadata.get('original_filename', 'N/A')}, Page Title: {doc.metadata.get('page_title', 'N/A')}, Source Type: {doc.metadata.get('source_type', 'N/A')})")

    return retrieved_docs
