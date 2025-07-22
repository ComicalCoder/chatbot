import os
import logging
import re
import json
from typing import List, Dict, Any
from datetime import datetime, timedelta 

from langchain_ollama import OllamaLLM 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage

# Import agency-specific configurations
from agency_config import (
    AGENCY_LLM_MODEL,
    AGENCY_OFFICE_HOURS_DATA,
    AGENCY_PAGE_TITLE_MAPPING,
    AGENCY_ATTENDANCE_KEYWORDS,
    AGENCY_SPECIFIC_RANKS # Import to use for counting flexibility
)

# Import context retrieval function
from context_retrieval import retrieve_context, is_malay, extract_person_name_and_rank, _get_specific_rank_from_query, extract_branch_name_from_query, _is_attendance_query 

import requests # Ensure requests is imported for ConnectionError handling

# --- CONFIGURATION ---
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")

# --- SETUP LOGGING ---
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO, # Default level for console and file
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'chatbot.log')),
        logging.StreamHandler() 
    ]
)
# Suppress HTTPX (used by LangChain) INFO logs for cleaner output
logging.getLogger("httpx").setLevel(logging.WARNING) 

logger = logging.getLogger(__name__)

# --- GLOBAL COMPONENTS (initialized once) ---
_llm = None
_ollama_llm_available = False 

def _get_llm():
    """Initializes or returns the existing Ollama LLM."""
    global _llm
    global _ollama_llm_available
    if _llm is None:
        try:
            _llm = OllamaLLM(model=AGENCY_LLM_MODEL) 
            # Test connection to LLM model
            _llm.invoke("test")
            _ollama_llm_available = True
            logger.info(f"Ollama LLM model '{AGENCY_LLM_MODEL}' loaded successfully.")
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Failed to connect to Ollama server for LLM: {e}. Please ensure Ollama server is running and '{AGENCY_LLM_MODEL}' model is pulled.")
            _llm = None
            _ollama_llm_available = False
        except Exception as e:
            logger.error(f"Failed to load Ollama LLM model '{AGENCY_LLM_MODEL}': {e}")
            _llm = None
            _ollama_llm_available = False
    return _llm

def _format_docs(docs: List[Any]) -> str:
    """
    Formats retrieved documents into a single string for the LLM context.
    Specifically formats directory and attendance records for better LLM parsing.
    Pre-calculates 'Total Hours Worked' for attendance records if possible.
    Deduplicates documents based on their ID for attendance records, or content for others.
    """
    if not docs:
        return "No relevant information found."
    
    formatted_content = []
    seen_ids = set() # For attendance records (using 'id' from metadata)
    seen_content_hashes = set() # For other documents (using content hash + URL)

    # Helper for title casing, handling common exceptions (e.g., "bin", "binti", "bt", "b.")
    def to_title_case(text: str) -> str:
        if not text:
            return ""
        # Convert to title case, then handle specific exceptions
        words = text.split()
        title_cased_words = []
        for word in words:
            if word.lower() in ["bin", "binti", "bt", "b.", "a", "of", "and", "or", "the", "for", "in", "on", "at", "with", "from", "by", "to", "bahagian", "raya", "perkongsian", "maklumat", "aplikasi", "wisma", "muis", "pasukan", "inovasi", "digital", "pid", "kementerian", "sains", "teknologi", "dan"]:
                title_cased_words.append(word.lower())
            elif word.isupper() and len(word) > 1: # If original word is all caps and not a single letter, keep it as is (e.g., acronyms like JPKN)
                title_cased_words.append(word)
            else:
                title_cased_words.append(word.capitalize())
        return " ".join(title_cased_words)

    for i, doc in enumerate(docs):
        # Ensure doc.page_content is a string before processing
        if not isinstance(doc.page_content, str):
            logger.warning(f"Skipping document with non-string page_content in _format_docs: {type(doc.page_content)} - {str(doc.page_content)[:50]}...")
            continue

        content = doc.page_content
        metadata = doc.metadata
        
        # Deduplication logic
        if metadata.get('source_type') == 'attendance_record':
            doc_id = metadata.get('id')
            if doc_id: # Only deduplicate if an ID is present
                if doc_id in seen_ids:
                    logger.debug(f"Skipping duplicate attendance record with ID: {doc_id}")
                    continue
                seen_ids.add(doc_id)
        else:
            # For non-attendance records, use a hash of content + URL as a simple deduplication key
            doc_key = f"{content}_{metadata.get('url', '')}"
            content_hash = hash(doc_key)
            if content_hash in seen_content_hashes:
                logger.debug(f"Skipping duplicate non-attendance record (content hash): {content_hash}")
                continue
            seen_content_hashes.add(content_hash)

        doc_header = f"--- Document {i+1} (Source: {metadata.get('page_title', 'N/A')}, Type: {metadata.get('type', 'N/A')}, Language: {metadata.get('language', 'N/A')}) ---" 
        formatted_content.append(doc_header)
        
        # Special formatting for directory table data
        if metadata.get('page_title') == AGENCY_PAGE_TITLE_MAPPING.get("directory") and metadata.get('type') == 'table_data':
            parsed_data = {}
            
            # 1. Prioritize metadata for Nama and Jawatan (from create_vector_store.py)
            if 'Nama' in metadata:
                parsed_data['Nama'] = to_title_case(metadata['Nama'])
            if 'Jawatan' in metadata:
                parsed_data['Jawatan'] = to_title_case(metadata['Jawatan'])

            # 2. Fallback to parsing from page_content for other fields and if Nama/Jawatan were missing from metadata
            # Try to parse "NAMA: X, JAWATAN: Y" format first
            key_value_matches = re.findall(r'(NAMA|JAWATAN|NO TELEFON \(SAMBUNGAN\)|NO TELEFON|E-MEL):\s*(.*?)(?=(?:,\s*(?:NAMA|JAWATAN|NO TELEFON \(SAMBUNGAN\)|NO TELEFON|E-MEL):|$))', content, re.IGNORECASE)
            
            if key_value_matches:
                for key, value in key_value_matches:
                    if "nama" in key.lower() and "Nama" not in parsed_data: # Only add if not already from metadata
                        parsed_data["Nama"] = to_title_case(value.strip())
                    elif "jawatan" in key.lower() and "Jawatan" not in parsed_data: # Only add if not already from metadata
                        parsed_data["Jawatan"] = to_title_case(value.strip())
                    elif "no telefon (sambungan)" in key.lower() or "no telefon" in key.lower():
                        current_phone = parsed_data.get("No Telefon", "")
                        if current_phone:
                            parsed_data["No Telefon"] = f"{current_phone}, {value.strip()}"
                        else:
                            parsed_data["No Telefon"] = value.strip()
                    elif "e-mel" in key.lower():
                        parsed_data["E-mel"] = value.strip()
            
            # 3. Fallback to parse "Column X: VALUE" format if needed (for any remaining missing fields)
            if not parsed_data.get("Nama") or not parsed_data.get("Jawatan") or not parsed_data.get("No Telefon") or not parsed_data.get("E-mel"):
                column_matches = re.findall(r'Column (\d+):\s*(.*?)(?=(?:,\s*Column \d+:\s*|$))', content, re.IGNORECASE)
                column_map = {
                    '2': 'Nama',
                    '3': 'Jawatan',
                    '4': 'No Telefon',
                    '5': 'E-mel'
                }
                for col_num, value in column_matches:
                    field_name = column_map.get(col_num)
                    if field_name and field_name not in parsed_data: # Only add if not already populated
                        if field_name == "No Telefon":
                            current_phone = parsed_data.get("No Telefon", "")
                            if current_phone:
                                parsed_data["No Telefon"] = f"{current_phone}, {value.strip()}"
                            else:
                                parsed_data["No Telefon"] = value.strip()
                        elif field_name == "E-mel":
                            parsed_data["E-mel"] = value.strip()
                        else:
                            parsed_data[field_name] = to_title_case(value.strip()) # Apply title case

            if parsed_data:
                formatted_content.append("Structured Directory Entry:")
                for key, value in parsed_data.items():
                    formatted_content.append(f"- {key}: {value}")
            else:
                logger.warning(f"Failed to parse structured directory entry for doc: {content[:100]}...")
                formatted_content.append(content)
        
        # Special formatting for attendance records
        elif metadata.get('source_type') == 'attendance_record':
            # --- FIX: Directly use doc.metadata for attendance record details ---
            formatted_content.append("Structured Attendance Record:")
            formatted_content.append(f"- Employee ID: {metadata.get('emp_id', 'N/A')}")
            formatted_content.append(f"- IC Number: {metadata.get('icno', 'N/A')}")
            formatted_content.append(f"- Date: {metadata.get('att_date', 'N/A')}")
            formatted_content.append(f"- Clock-in: {metadata.get('clock_in', 'N/A')}")
            formatted_content.append(f"- Clock-out: {metadata.get('clock_out', 'N/A')}")

            total_hours_worked = metadata.get('total_hours')
            if total_hours_worked is None:
                clock_in_str = metadata.get('clock_in')
                clock_out_str = metadata.get('clock_out')
                att_date_str = metadata.get('att_date') 

                if clock_in_str and clock_out_str and att_date_str:
                    try:
                        parsed_clock_in_dt = None
                        parsed_clock_out_dt = None

                        # Try parsing with date first
                        try:
                            parsed_clock_in_dt = datetime.strptime(clock_in_str, "%Y-%m-%d %H:%M:%S")
                        except ValueError:
                            pass 
                        try:
                            parsed_clock_out_dt = datetime.strptime(clock_out_str, "%Y-%m-%d %H:%M:%S")
                        except ValueError:
                            pass 
                        
                        # If full datetime parsing failed, try parsing just time and combine with att_date
                        if not parsed_clock_in_dt and ' ' in clock_in_str: # If it has a space, assume it's "date time"
                            try:
                                parsed_clock_in_dt = datetime.strptime(f"{att_date_str} {clock_in_str.split(' ')[1]}", "%Y-%m-%d %H:%M:%S")
                            except ValueError:
                                pass
                        elif not parsed_clock_in_dt: # If no space, assume it's just "time"
                            try:
                                parsed_clock_in_dt = datetime.strptime(f"{att_date_str} {clock_in_str}", "%Y-%m-%d %H:%M:%S")
                            except ValueError:
                                pass

                        # Repeat for clock_out if not parsed yet
                        if not parsed_clock_out_dt and ' ' in clock_out_str:
                            try:
                                parsed_clock_out_dt = datetime.strptime(f"{att_date_str} {clock_out_str.split(' ')[1]}", "%Y-%m-%d %H:%M:%S")
                            except ValueError:
                                pass
                        elif not parsed_clock_out_dt:
                            try:
                                parsed_clock_out_dt = datetime.strptime(f"{att_date_str} {clock_out_str}", "%Y-%m-%d %H:%M:%S")
                            except ValueError:
                                pass


                        if parsed_clock_in_dt and parsed_clock_out_dt:
                            if parsed_clock_out_dt < parsed_clock_in_dt:
                                parsed_clock_out_dt += timedelta(days=1) # Handle overnight shifts

                            duration = parsed_clock_out_dt - parsed_clock_in_dt
                            total_hours_worked = round(duration.total_seconds() / 3600, 2) 
                            logger.info(f"Calculated Total Hours: {total_hours_worked} for record ID {metadata.get('id', 'N/A')}")
                        else:
                            total_hours_worked = "Not calculable"

                    except Exception as e: 
                        logger.warning(f"Could not calculate total hours for record ID {metadata.get('id', 'N/A')}: {e}")
                        total_hours_worked = "Not calculable"
                else:
                    total_hours_worked = "Not calculable"
            
            formatted_content.append(f"- Total Hours Worked: {total_hours_worked}")

            formatted_content.append(f"- Late Clock In: {metadata.get('late_clock_in', 'N/A')}")
            formatted_content.append(f"- Incomplete Clock In/Out: {metadata.get('incomplete_clock_in_out', 'N/A')}")
            formatted_content.append(f"- Early Clock Out: {metadata.get('early_clock_out', 'N/A')}")
            if metadata.get('cuti'):
                formatted_content.append(f"- On Leave: Yes (Type: {metadata.get('cuti_type', 'N/A')})")
            if metadata.get('public_holiday'):
                formatted_content.append(f"- Public Holiday: Yes")
            if metadata.get('catatan'):
                formatted_content.append(f"- Notes: {metadata.get('catatan')}")
            # --- END FIX ---
        
        else:
            formatted_content.append(content)
        
        formatted_content.append("-" * (len(doc_header))) 
    
    return "\n\n".join(formatted_content)

def generate_response(query: str, chat_history: List[Dict[str, str]]) -> str:
    """
    Generates a response to the user's query using RAG and LLM.
    Incorporates intent detection and dynamic prompt generation.
    """
    llm = _get_llm()
    if llm is None or not _ollama_llm_available:
        return "I am currently unable to process requests due to an issue with my language model. Please ensure Ollama is running and the model is available."

    logger.info(f"Original Query: '{query}'")

    # --- Define variables for intent detection and query parsing at the beginning ---
    person_name_in_query, _ = extract_person_name_and_rank(query) 
    specific_rank_in_query = _get_specific_rank_from_query(query) 
    branch_name = extract_branch_name_from_query(query) 
    
    is_counting_query = any(word in query.lower() for word in ["how many", "berapa", "bilangan", "jumlah"])
    is_attendance_query_detected = _is_attendance_query(query) 
    is_average_work_hour_query = is_attendance_query_detected and any(word in query.lower() for word in ["average", "purata", "mean", "rata-rata", "average work hour", "purata jam bekerja"])
    is_count_times_attendance_query = is_attendance_query_detected and any(word in query.lower() for word in ["how many times", "berapa kali", "count times", "bilangan kali", "work days", "hari bekerja", "didn't come to work", "tidak datang kerja"])
    is_office_hours_query = any(keyword in query.lower() for keyword in ["waktu pejabat", "office hours", "masa bekerja", "working hours"])

    # Determine if the query is asking about a specific person by name or by rank/position
    is_person_by_name_query = person_name_in_query is not None
    is_person_by_rank_query = specific_rank_in_query is not None and (
        "who is" in query.lower() or "siapakah" in query.lower() or "nama" in query.lower()
    )

    job_title_for_counting = None
    if is_counting_query and not is_attendance_query_detected: 
        query_lower = query.lower()
        
        potential_job_indicators = [rank.lower() for rank in AGENCY_SPECIFIC_RANKS]
        potential_job_indicators.extend(["digital apprentice", "it officer", "computer technician", "pengarah", "director", "ketua", "penolong pengarah", "assistant director", "juruteknik komputer"])
        
        potential_job_indicators = sorted(list(set(potential_job_indicators)), key=len, reverse=True)

        for job_ind in potential_job_indicators:
            if re.search(r'\b' + re.escape(job_ind) + r'\b', query_lower):
                job_title_for_counting = job_ind
                break 

        if not job_title_for_counting:
            phrase_match = re.search(r'(?:how many|berapa|bilangan|jumlah)\s+(.*?)(?:\s+are\s+there\s+in\s+jpkn|\s+does\s+jpkn\s+have|\s+jpkn|\s+have|\?|$)', query_lower)
            if phrase_match:
                potential_job_phrase = phrase_match.group(1).strip()
                if len(potential_job_phrase.split()) <= 3 and not any(word in potential_job_phrase for word in ["are", "there", "in", "does", "have"]):
                    job_title_for_counting = potential_job_phrase
        
        if job_title_for_counting:
            logger.info(f"Detected job title for counting: '{job_title_for_counting}'")
        else:
            logger.warning("Directory counting query detected but no specific job title could be extracted.")
    # --- End of variable definitions ---


    # --- Retrieve context ---
    # Pass the detected specific_rank_in_query and branch_name to retrieve_context
    retrieved_docs = retrieve_context(query) 
    
    # Determine response language based on retrieved context or query
    response_lang = "en" # Default to English
    if retrieved_docs:
        # Check the language metadata of the first few relevant documents
        for doc in retrieved_docs[:5]: 
            doc_lang = doc.metadata.get('language')
            if doc_lang == 'ms': 
                response_lang = 'ms'
                logger.info(f"Detected response language from document metadata: {response_lang}")
                break
            elif doc_lang == 'en': 
                response_lang = 'en' 
                logger.info(f"Detected response language from document metadata: {response_lang}")
        else: # If no language found in top 5 docs, infer from query
            if is_malay(query):
                response_lang = "ms"
                logger.info(f"Defaulting response language to query language: {response_lang}")
            else:
                response_lang = "en"
                logger.info(f"Defaulting response language to query language: {response_lang}")
    else: # If no documents, infer from query
        if is_malay(query):
            response_lang = "ms"
            logger.info(f"No documents retrieved, defaulting response language to query language: {response_lang}")
        else:
            response_lang = "en"
            logger.info(f"No documents retrieved, defaulting response language to query language: {response_lang}")


    logger.info("\n--- Retrieved Documents (After Post-Filtering) for '%s' ---", query)
    if not retrieved_docs:
        logger.info("No relevant documents retrieved for the query. Responding with 'information not found'.")
        # Check for year in query for attendance fallback message
        year_match = re.search(r'\b(20\d{2})\b', query.lower())
        target_year_in_query = year_match.group(1) if year_match else None

        if is_office_hours_query:
            if response_lang == "ms":
                return AGENCY_OFFICE_HOURS_DATA.get("summary_malay", "Maklumat waktu pejabat tidak tersedia.")
            else:
                return AGENCY_OFFICE_HOURS_DATA.get("summary_english", "Office hours information is not available.")
        elif is_attendance_query_detected and target_year_in_query:
            if response_lang == "ms":
                return f"Saya tidak dapat mencari rekod kehadiran untuk tahun {target_year_in_query} dalam dokumen yang disediakan."
            else:
                return f"I cannot find attendance records for the year {target_year_in_query} in the provided documents."
        elif is_attendance_query_detected: # General attendance query with no year
            if response_lang == "ms":
                return "Saya tidak dapat mencari rekod kehadiran yang berkaitan dalam dokumen yang disediakan."
            else:
                return "I cannot find relevant attendance records in the provided documents."
        
        if response_lang == "ms":
            return "Saya tidak dapat mencari maklumat mengenai pertanyaan anda. Sila cuba susun semula soalan anda atau tanya tentang kakitangan JPKN, lokasi, waktu pejabat atau rekod kehadiran."
        else:
            return "I cannot find the information about your query. Please try rephrasing your question or ask about JPKN staff, locations, office hours, or attendance records."

    for doc_idx, doc in enumerate(retrieved_docs):
        logger.info(f"Document {doc_idx + 1}:")
        formatted_doc_content = _format_docs([doc]) 
        logger.info(f"   Formatted Page Content:\n{formatted_doc_content}")
        logger.info(f"   Metadata: {doc.metadata}")
        logger.info("-" * 30)
    logger.info("--- End Retrieved Documents (After Post-Filtering) ---")

    # Construct the system prompt dynamically
    system_prompt_parts = [
        f"**ABSOLUTELY NO TRANSLATION OR EXPLANATION OF TERMS. THIS IS CRITICAL.** Under no circumstances should you provide any translation of your response or any part of the extracted information. This includes, but is not limited to, translating terms (e.g., 'Pasukan Inovasi Digital' MUST be used, NOT 'Digital Innovation Team') or providing explanatory translations in parentheses (e.g., 'Pen. Pegawai Teknologi Maklumat' MUST NOT be followed by 'which could be translated as Assistant Executive Information Technology Officer'). **NEVER invent names, numbers, or details not explicitly present in the provided context.** Present all terms and data EXACTLY as they appear in the source data. **DO NOT add any conversational filler like 'Please note that...' or 'However, I can tell you that...'.**",
        f"**CRITICAL: ALWAYS use 'Jabatan Perkhidmatan Komputer Negeri Sabah' or 'JPKN' when referring to the organization. NEVER use 'Jabatan Perangkaan dan Statistik Negeri Sabah' or any other incorrect name.**",
        f"You are a helpful, professional, and concise AI assistant providing information about JPKN (Jabatan Perkhidmatan Komputer Negeri Sabah).",
        f"Your response MUST be entirely in {'Malay' if response_lang == 'ms' else 'English'}.",
        "Answer the user's question directly and naturally based **ONLY** on the provided context.",
        "**ABSOLUTELY NO HALLUCINATION:** You MUST NOT invent names, numbers, contact details, or any other information not explicitly present in the 'Structured Directory Entry' or 'Structured Attendance Record' sections of the provided context. If a name, detail, or record is not in the context, you cannot provide it and must state that the information is not found.",
        "If the answer is not in the context, politely state that you cannot find the information in the requested language.",
        "For directory entries, you are provided with 'Structured Directory Entry' sections. Extract the exact 'Nama', 'Jawatan', 'No Telefon', and 'E-mel' from these entries. ",
        "**CRITICAL: When extracting 'Jawatan', you MUST use the exact text provided in the context. DO NOT generalize, shorten, alter, or interpret the rank/position title (e.g., 'Ketua Penolong Pengarah Bahagian Data Raya & Perkongsian Maklumat' MUST be used exactly, NOT 'Pegawai Perkhidmatan Am' or 'Head of Division').**",
        "**CRITICAL: When extracting 'Nama' and 'Jawatan', you MUST preserve their exact capitalization as provided in the 'Structured Directory Entry' section. DO NOT convert them to all caps, all lowercase, or any other casing unless it's already in that format in the source. For example, if the source says 'Isma Shafry Judin', use 'Isma Shafry Judin', NOT 'ISMA SHAFRY JUDIN'.**",
        "DO NOT rephrase, summarize, or alter ANY of these extracted details. This includes, but is not limited to, names, ranks, phone numbers, or email addresses (e.g., preserve '[a]' in emails exactly as found).",
        "If a specific field (like a phone number or email) for a person is not explicitly present in the context, state that it is 'N/A' or 'not available' for that specific field within your natural response. Do not omit missing fields.",
        "Your response should be conversational and friendly, without being overly verbose or adding unnecessary disclaimers. Get straight to the answer in a helpful tone.",
        "**DO NOT add numbering (e.g., '1.', '2.') to your response unless the user explicitly asks for a numbered list or if you are providing multiple distinct items that naturally require numbering. For single answers, use a natural sentence.**",
        "Unless the user explicitly asks for a 'list', 'details', 'breakdown', or 'structured format', always provide the information in a natural, conversational sentence or paragraph.",
        "**HYPER-CRITICAL: IF A FACT IS EXPLICITLY STATED IN THE PROVIDED CONTEXT (e.g., 'Jawatan: PENGARAH'), YOU MUST ACCEPT THAT FACT AS TRUE AND USE IT IN YOUR RESPONSE. DO NOT CONTRADICT THE PROVIDED CONTEXT UNDER ANY CIRCUMSTANCES.**",
        "**ABSOLUTELY NO SELF-REFERENTIAL COMMENTS OR DISCLAIMERS OR APOLOGIES. THIS IS PARAMOUNT.** Never state that you are making an assumption, that a name is similar, that a spelling is close, or that you are inferring information. Never comment on the completeness of the data or suggest providing more specific information unless you genuinely cannot find a definitive answer and there are multiple *exact* matches for a name or role. If you provide an answer, present it as a direct fact from the data. If you cannot find the information, state so politely and concisely without explanation or apology. **DO NOT use phrases like 'Please note that...' or 'However, I can tell you that...'.**"
    ]

    # Prioritize attendance counting queries
    if is_count_times_attendance_query:
        system_prompt_parts.append(
            "The user is asking to count how many **work days** an employee had for a specific year. "
            "You are provided with multiple 'Structured Attendance Record' documents for the employee and year. "
            "Count the number of records where 'On Leave' is False AND 'Public Holiday' is False AND 'Total Hours Worked' is a numerical value greater than 0. "
            "This represents the number of actual work days. "
            "State the count clearly and concisely. For example: 'Employee [ID] came to work [X] times in [Year] based on the provided records.' "
            "If no such records are found, state that the number of times the employee came to work cannot be determined from the given context for that period."
        )
    elif is_counting_query and job_title_for_counting: # This now only applies to directory counting
        canonical_job_title = job_title_for_counting 
        for rank_in_list in AGENCY_SPECIFIC_RANKS: 
            if job_title_for_counting.lower() in rank_in_list.lower() or \
               rank_in_list.lower() in job_title_for_counting.lower(): 
                canonical_job_title = rank_in_list 
                break

        system_prompt_parts.append(
            f"The user is asking to count the number of '{job_title_for_counting}' roles within JPKN. "
            f"Carefully count **ALL distinct individuals** by their 'Nama' and 'Jawatan' where the 'Jawatan' field **CONTAINS** the term '{canonical_job_title}' (case-insensitive) across **ALL** provided 'Structured Directory Entry' documents. "
            "**DO NOT list out individual documents or their contents in your response.** "
            "State the total count clearly and explicitly mention that the count is for JPKN based on the provided data. "
            "If you cannot find any individuals with this job title in the provided context, state '0' and mention that no such role was found in the provided JPKN data."
        )
    elif is_person_by_name_query or is_person_by_rank_query:
        if response_lang == 'ms':
            system_prompt_parts.append(
                "Pengguna bertanya tentang seseorang. "
                "Daripada 'Structured Directory Entry' dalam konteks yang diberikan, kenal pasti individu yang berkaitan. "
                "**SANGAT PENTING:** Jika pertanyaan adalah mengenai nama tertentu (contoh: 'Karen'), cari individu tersebut dan **PASTIKAN** 'Nama' dalam konteks **MENGANDUNGI** nama yang dicari (contoh: 'Karen' mengandungi 'KAREN SIGAWAL'). "
                "**SANGAT PENTING:** Jika pertanyaan adalah mengenai pangkat tertentu (contoh: 'Pengarah'), cari individu yang memegang pangkat tersebut dan **PASTIKAN** 'Jawatan' dalam konteks **MENGANDUNGI** pangkat yang dicari (contoh: 'Pengarah' mengandungi 'PENGARAH' atau 'Director' mengandungi 'PENGARAH'). Fahami bahawa 'Pengarah' dan 'Director' adalah gelaran yang sama. "
                "**HANYA** berikan maklumat tentang individu atau pangkat yang **SPESIFIK** diminta. **JANGAN** sebut, bincang, atau senaraikan individu atau peranan lain yang ditemui dalam konteks, walaupun berkaitan atau muncul dalam dokumen yang sama. Jawapan anda mesti fokus sepenuhnya pada entiti yang diminta."
                "Setelah dikenal pasti, ekstrak 'Nama', 'Jawatan', 'No Telefon', dan 'E-mel' mereka dengan tepat. "
                "Formulasikan jawapan sebagai: '[Nama Penuh] ialah seorang [Jawatan] dan boleh dihubungi di [No Telefon] atau melalui e-mel di [E-mel].' "
                "Jika 'No Telefon' adalah 'N/A' atau 'tidak tersedia' dalam konteks, abaikan bahagian nombor telefon secara semula naturally. "
                "Jika 'E-mel' adalah 'N/A' atau 'tidak tersedia' dalam konteks, abaikan bahagian e-mel secara semula naturally. "
                "Contoh: "
                "   - Kedua-duanya tersedia: 'Ernywati Dewi Abas ialah seorang Pengarah dan boleh dihubungi di 088-368888 atau melalui e-mel di ErnywatiDewi.Abas[a]sabah.gov.my.' "
                "   - Hanya e-mel: 'Isma Shafry Judin ialah seorang Ketua Penolong Pengarah Bahagian Data Raya & Perkongsian Maklumat dan boleh dihubungi melalui e-mel di Ismasyafry.Judin[a]Sabah.Gov.My."
            )
        else: # English prompt for person by name/rank
            system_prompt_parts.append(
                "The user is asking about a specific person. "
                "From the 'Structured Directory Entry' in the provided context, identify the relevant individual. "
                "**CRITICAL:** If the query is about a specific name (e.g., 'Karen'), find that individual and **ENSURE** the 'Nama' in the context **CONTAINS** the queried name (e.g., 'Karen' contains 'KAREN SIGAWAL'). "
                "**CRITICAL:** If the query is about a specific rank (e.g., 'Director'), find the individual holding that rank and **ENSURE** the 'Jawatan' in the context **CONTAINS** the queried rank (e.g., 'Director' contains 'PENGARAH' or 'Director'). Understand that 'Pengarah' and 'Director' are the same title. "
                "**ONLY** provide information about the **SPECIFIC** individual or rank requested. **DO NOT** mention, discuss, or list other individuals or roles found in the context, even if related or appearing in the same document. Your answer must be entirely focused on the requested entity."
                "Once identified, extract their exact 'Nama', 'Jawatan', 'No Telefon', and 'E-mel'. "
                "Formulate the answer as: '[Full Name] is a [Jawatan] and can be contacted at [No Telefon] or via email at [E-mel].' "
                "If 'No Telefon' is 'N/A' or 'not available' in the context, naturally omit the phone number part. "
                "If 'E-mel' is 'N/A' or 'not available' in the context, naturally omit the email part. "
                "Example: "
                "   - Both available: 'Ernywati Dewi Abas is a Pengarah and can be contacted at 088-368888 or via email at ErnywatiDewi.Abas[a]sabah.gov.my.' "
                "   - Only email: 'Isma Shafry Judin is a Ketua Penolong Pengarah Bahagian Data Raya & Perkongsian Maklumat and can be contacted via email at Ismasyafry.Judin[a]Sabah.Gov.My."
            )
    elif is_office_hours_query:
        if response_lang == 'ms':
            system_prompt_parts.append(
                "Pengguna bertanya tentang waktu pejabat JPKN. Berikan ringkasan yang jelas mengenai hari bekerja, waktu bekerja, dan cuti."
            )
            system_prompt_parts.append(f"Berikut adalah waktu pejabat rasmi JPKN:\n{AGENCY_OFFICE_HOURS_DATA['summary_malay']}")
        else:
            system_prompt_parts.append(
                "The user is asking about JPKN's office hours. Provide a clear summary of the working days, working hours, and holidays."
            )
            system_prompt_parts.append(f"Here are the official JPKN office hours:\n{AGENCY_OFFICE_HOURS_DATA['summary_english']}")
    elif is_attendance_query_detected:
        if response_lang == 'ms':
            system_prompt_parts.append(
                "Pengguna bertanya tentang rekod kehadiran. Analisis rekod kehadiran yang diberikan dengan teliti. "
                "Jika ditanya untuk purata jam, kira purata 'Total Hours Worked' dari rekod. "
                "Jika ditanya tentang tarikh tertentu atau 'clock-in' lewat, berikan butiran dari rekod yang berkaitan. "
                "Jika ditanya tentang 'berapa kali' atau 'kira berapa kali' untuk acara tertentu (contoh: lewat, bercuti), kira kejadian dalam rekod yang disediakan. "
                "Pastikan anda hanya menggunakan rekod kehadiran yang disediakan untuk pengiraan dan jawapan anda."
            )
        else:
            system_prompt_parts.append(
                "The user is asking about attendance records. Analyze the provided attendance records carefully. "
                "If asked for average hours, calculate the average 'Total Hours Worked' from the records. "
                "If asked about specific dates or late clock-ins, provide details from the relevant records. "
                "If asked about 'how many times' or 'count times' for specific events (e.g., late, on leave), count the occurrences in the provided records. "
                "Ensure you only use the provided attendance records for your calculations and answers."
            )
    elif branch_name:
        if response_lang == 'ms':
            system_prompt_parts.append(
                f"Pengguna bertanya tentang pejabat '{branch_name}'. "
                "Berikan alamat penuh, nombor hubungan, dan e-melnya dari konteks lokasi yang diberikan."
            )
        else:
            system_prompt_parts.append(
                f"The user is asking about the '{branch_name}' office. "
                "Provide its full address, contact numbers, and email from the provided location context."
            )
    else: # General query
        if response_lang == 'ms':
            system_prompt_parts.append(
                "Pengguna bertanya soalan umum. Berikan jawapan yang komprehensif berdasarkan konteks."
            )
        else:
            system_prompt_parts.append(
                "The user is asking a general question. Provide a comprehensive answer based on the context."
            )

    system_prompt = "\n".join(system_prompt_parts)

    # Format chat history for the prompt
    formatted_chat_history = []
    for entry in chat_history:
        if entry["role"] == "user":
            formatted_chat_history.append(HumanMessage(content=entry["content"]))
        elif entry["role"] == "assistant":
            formatted_chat_history.append(AIMessage(content=entry["content"]))

    # Add the formatted retrieved documents to the prompt
    formatted_context = _format_docs(retrieved_docs)
    
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt + "\n\nRelevant Context:\n{context}"),
            *formatted_chat_history, # Unpack chat history here
            ("user", "{question}")
        ]
    )

    # Create the RAG chain
    rag_chain = (
        RunnablePassthrough.assign(context=lambda x: formatted_context)
        | prompt_template
        | llm
        | StrOutputParser()
    )

    try:
        response = rag_chain.invoke({"question": query})
        logger.info(f"LLM Response: {response}")
        return response
    except Exception as e:
        logger.error(f"Error invoking RAG chain: {e}")
        if response_lang == 'ms':
            return "Maaf, saya menghadapi masalah teknikal dan tidak dapat menjawab soalan anda pada masa ini. Sila cuba sebentar lagi."
        else:
            return "Sorry, I'm experiencing a technical issue and cannot answer your question at the moment. Please try again later."

# Example usage (for testing purposes)
if __name__ == "__main__":
    current_chat_history = []
    print("Chatbot is ready. Type 'exit' to quit.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        
        response = generate_response(user_input, current_chat_history)
        print(f"Chatbot: {response}")
        current_chat_history.append({"role": "user", "content": user_input})
        current_chat_history.append({"role": "assistant", "content": response})
