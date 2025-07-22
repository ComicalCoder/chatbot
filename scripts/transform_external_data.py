import json
import os
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional
import re
import unicodedata
import logging
from pathlib import Path

# Import agency-specific configurations
from agency_config import AGENCY_EXTERNAL_DATA_TRANSFORMATION_RULES

# --- CONFIGURATION ---
RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "raw_data")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data") 
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs") # New log directory

# --- SETUP LOGGING ---
Path(LOG_DIR).mkdir(parents=True, exist_ok=True) # Ensure log directory exists
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'transform_data.log')), # Log to new directory
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- HELPER FUNCTIONS ---
def normalize_text(text: str) -> str:
    """Normalize text: fix whitespace and Unicode issues."""
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = re.sub(r"\s+", " ", text).strip()
    return text

def sanitize_metadata(metadata: dict) -> dict:
    """Sanitize metadata for ChromaDB compatibility. Converts dicts/lists to JSON strings."""
    sanitized = {}
    for key, value in metadata.items():
        if isinstance(value, dict) or isinstance(value, list):
            # Convert dictionaries and lists to JSON strings
            sanitized[key] = json.dumps(value, ensure_ascii=False)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            sanitized[key] = value
        else:
            # Fallback for any other complex types
            sanitized[key] = str(value)
    return sanitized

def sanitize_filename(text: str, max_length: int = 50) -> str:
    """Sanitize a string for use as a filename."""
    s = text.replace(" ", "_")
    s = re.sub(r"[^\w.-]", "", s)
    s = re.sub(r"[_.-]+", "_", s)
    s = s.strip("_.-")
    return s[:max_length]

# --- TRANSFORMATION LOGIC ---
def process_single_file(input_filepath: str, data_type_key: str, output_dir: str) -> None:
    """
    Reads a single external data file, transforms it, and saves it.
    This is the core logic, now called by the main execution block.
    """
    logger.info(f"Processing '{os.path.basename(input_filepath)}' as data type '{data_type_key}'...")

    if data_type_key not in AGENCY_EXTERNAL_DATA_TRANSFORMATION_RULES:
        logger.error(f"[ERROR] Data type key '{data_type_key}' not found in AGENCY_EXTERNAL_DATA_TRANSFORMATION_RULES. Skipping file.")
        logger.error(f"Available keys: {list(AGENCY_EXTERNAL_DATA_TRANSFORMATION_RULES.keys())}")
        return

    config = AGENCY_EXTERNAL_DATA_TRANSFORMATION_RULES[data_type_key]
    root_key = config.get("root_key")
    page_content_template = config.get("page_content_template")
    page_content_template_fields = config.get("page_content_template_fields", {})
    metadata_keys = config.get("metadata_keys", [])
    source_type = config.get("source_type", "external_data")

    if not root_key or not page_content_template:
        logger.error(f"[ERROR] Missing 'root_key' or 'page_content_template' in config for '{data_type_key}'. Skipping file.")
        return

    try:
        with open(input_filepath, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"[ERROR] Failed to decode JSON from {input_filepath}: {e}. Skipping file.")
        return
    except Exception as e:
        logger.error(f"[ERROR] An error occurred reading {input_filepath}: {e}. Skipping file.")
        return

    records = raw_data.get(root_key, [])
    if not records:
        logger.warning(f"[WARNING] No '{root_key}' array found or it is empty in {input_filepath}. No records to transform.")
        return

    transformed_documents: List[Dict[str, Any]] = []

    for record in records:
        # Create a mutable copy of the record to modify values for page_content formatting
        processed_record = record.copy()

        # Explicitly convert boolean-like values to True/False for page_content and metadata
        # This is CRITICAL for LLM to reliably count "True" occurrences
        for key in ["late_clock_in", "incomplete_clock_in_out", "early_clock_out", "public_holiday", "luar_negeri", "kursus", "cuti", "keluar_pejabat"]:
            original_value = processed_record.get(key)
            if original_value is None or original_value == "" or original_value == 0:
                processed_record[key] = False
            elif original_value == 1:
                processed_record[key] = True
            # If it's already True/False, leave it as is
            
        # Handle IC number placeholder
        if processed_record.get("icno") == "(IC NUMBER)":
            processed_record["icno"] = None # Set to None so it can be handled by icno_display lambda

        template_data = {}
        # First, populate with direct values from processed_record (normalized)
        for k, v in processed_record.items():
            template_data[k] = normalize_text(v)

        # Then, override/add values based on page_content_template_fields
        for field_name, field_template_or_func in page_content_template_fields.items():
            if callable(field_template_or_func):
                template_data[field_name] = field_template_or_func(processed_record)
            else:
                try:
                    # For string templates, format them using already populated template_data
                    # This requires that any placeholders within field_template_or_func
                    # refer to keys already in template_data (from processed_record or earlier lambdas)
                    formatted_field = field_template_or_func.format(**template_data)
                    template_data[field_name] = formatted_field if normalize_text(formatted_field).strip() else ""
                except KeyError as e:
                    template_data[field_name] = ""
                    logger.warning(f"[WARNING] Missing key '{e}' for sub-template '{field_name}' in record from {input_filepath}. Setting to empty.")
                except Exception as e:
                    template_data[field_name] = ""
                    logger.warning(f"[WARNING] Error processing sub-template '{field_name}': {e} for record from {input_filepath}. Setting to empty.")

        try:
            page_content = page_content_template.format(**template_data)
        except KeyError as e:
            logger.error(f"[ERROR] Missing key '{e}' in record for page_content_template from {input_filepath}. Skipping record.")
            continue
        
        page_content = normalize_text(page_content)

        metadata = {
            "source_type": source_type,
            "original_filename": os.path.basename(input_filepath),
            "transformed_at": datetime.utcnow().isoformat(),
        }
        for key in metadata_keys:
            if key in processed_record: # Use processed_record for metadata
                metadata[key] = processed_record[key] # Store the processed boolean values in metadata

        transformed_documents.append({
            "page_content": page_content,
            "metadata": sanitize_metadata(metadata) # Use updated sanitize_metadata
        })

    output_filename = f"{data_type_key}_data_{hashlib.md5(input_filepath.encode()).hexdigest()[:12]}.json"
    output_filepath = os.path.join(output_dir, output_filename)

    try:
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(transformed_documents, f, ensure_ascii=False, indent=2)
        logger.info(f"[SUCCESS] Transformed {len(records)} '{data_type_key}' records to {output_filepath}")
    except Exception as e:
        logger.error(f"[ERROR] Failed to save transformed data to {output_filepath}: {e}")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    logger.info("Starting external data transformation process... (Automated mode)")
    
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True) # Ensure output directory exists
    Path(RAW_DATA_DIR).mkdir(parents=True, exist_ok=True) # Ensure raw_data directory exists
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True) # Ensure log directory exists

    json_files_in_raw_data = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.json')]

    if not json_files_in_raw_data:
        logger.warning(f"No JSON files found in '{RAW_DATA_DIR}'. Nothing to transform.")
    else:
        for filename in json_files_in_raw_data:
            input_filepath = os.path.join(RAW_DATA_DIR, filename)
            
            # Infer data_type_key from filename (e.g., "attendance_raw.json" -> "attendance")
            # This logic assumes filenames are like "data_type_key_something.json" or "data_type_key.json"
            data_type_key = filename.replace('_raw.json', '').replace('.json', '')
            
            # Basic validation: ensure the inferred key exists in our rules
            if data_type_key in AGENCY_EXTERNAL_DATA_TRANSFORMATION_RULES:
                process_single_file(input_filepath, data_type_key, OUTPUT_DIR)
            else:
                logger.error(f"[ERROR] Could not determine a valid data_type_key for file '{filename}'. Skipping.")
                logger.info(f"Please ensure the filename matches a key in AGENCY_EXTERNAL_DATA_TRANSFORMATION_RULES (e.g., 'attendance_raw.json' for 'attendance').")

    logger.info("External data transformation process complete.")
    logger.info("Remember to run create_vector_store.py next to update your vector database.")
