import re
import unicodedata

# --- Scraper Configuration ---
# Base URL for the JPKN website
AGENCY_BASE_URL = "https://jpkn.sabah.gov.my/"

# Mapping of URL paths to more readable page titles for indexing
AGENCY_URL_PAGE_TITLE_MAPPING = {
    "index.php/en/jpkn/profil-korporat/director-s-message": "Corporate Profile - Director's Message",
    "index.php/en/jpkn/profil-korporat/organization-chart": "Corporate Profile - Organization Chart",
    "index.php/en/hubungi-kami/alamat-cawangan": "Contact Us - Office Locations",
    "index.php/en/direktori": "Directory",
    "index.php/en/jpkn/profil-korporat/mission-vision": "Corporate Profile - Mission & Vision",
    "index.php/en/gallery": "Gallery",
    "index.php/en/muat-turun/e-book": "Downloads - E-Book",
    "index.php/en/soalan-lazim-faq": "FAQ",
    "index.php/en/galeri-gambar-dan-video": "Photo & Video Gallery",
    "index.php/ms/jpkn/profil-korporat/perutusan-pengarah": "Profil Korporat - Perutusan Pengarah",
    "index.php/ms/jpkn/profil-korporat/carta-organisasi": "Profil Korporat - Carta Organisasi",
    "index.php/ms/hubungi-kami/alamat-cawangan": "Hubungi Kami - Alamat Cawangan",
    "index.php/ms/direktori": "Direktori",
    "index.php/ms/jpkn/profil-korporat/visi-misi": "Profil Korporat - Visi & Misi",
    "index.php/ms/galeri": "Galeri",
    "index.php/ms/muat-turun/e-buku": "Muat Turun - E-Buku",
    "index.php/ms/soalan-lazim-faq": "Soalan Lazim FAQ",
    "index.php/ms/galeri-gambar-dan-video": "Galeri Gambar dan Video",
}

# CSS selectors for content extraction
AGENCY_CSS_SELECTORS = {
    "main_content": [
        "div.rt-article",     # Main article content
        "div.custom",         # General custom content blocks
        "div.item-page",      # Specific page content wrapper
        "div.page-header",    # Headers, if they contain important context
        "div.rt-container",   # Broader container, use with caution
        "div.rt-grid-9",      # Main content grid
        "div.rt-block"        # General block
    ],
    "header": [
        "div.bannerheader"    # Specific header area
    ],
    "footer": [
        "div#rt-footer"       # Footer area
    ],
    "table_data": [
        "table.table",        # General tables, like directories
        "table.category"      # Specific category tables
    ],
    "contact_info": [
        "div.contact-details", # Specific contact detail blocks
        "ul.contact-info"     # Lists of contact info
    ],
    "office_hours": [
        "div.custom > p:contains('WAKTU PEJABAT')", # Specific paragraph for office hours
        "div.custom > h3:contains('WAKTU PEJABAT') + p",
        "div.module-content:contains('WAKTU PEJABAT')" # Module content containing office hours
    ],
    "collapsible_section": [ # For FAQ or similar collapsible content
        "div.uk-accordion"
    ]
}

# Regex patterns for cleaning extracted text. Applied in order.
AGENCY_CLEANING_REGEX = [
    (r'(\s*\n\s*){2,}', '\n\n'),   # Replace multiple newlines with at most two
    (r'\s{2,}', ' '),              # Replace multiple spaces with a single space
    (r'\t+', ' '),                 # Replace tabs with spaces
    (r'\xa0', ' '),                 # Remove non-breaking spaces
    (r'\[ \d+ \]', ''),            # Remove patterns like [ 1 ]
    (r'\[\d{1,2}/\d{1,2}/\d{4}\]', ''), # Remove date patterns like [01/01/2023]
    (r'\(\s*\)', ''),              # Remove empty parentheses
    (r'No\. Tel/Faks:', 'No Telefon/Faks:'), # Standardize
    (r'Tel/Faks:', 'No Telefon/Faks:'),     # Standardize
    (r'No\. Telefon', 'No Telefon'),        # Standardize
    (r'Emel:', 'Emel: '),                   # Ensure space after Emel:
    (r'Nama Cawangan :', 'Nama Cawangan:'), # Standardize
    (r'Alamat :', 'Alamat:'),               # Standardize
    (r'Ketua Cawangan :', 'Ketua Cawangan:'),
    (r'E-mel :', 'Emel:'),
    (r'e-mel :', 'Emel:'),
    (r'Pegawai :', 'Pegawai:'),
    # --- NEW FIX: Handle concatenated words in job titles ---
    (r'(DIGITAL)(KEMENTERIAN)', r'\1 \2'), # Fixes "DIGITALKEMENTERIAN" to "DIGITAL KEMENTERIAN"
    (r'(TEKNOLOGI)(DAN)', r'\1 \2'), # Fixes "TEKNOLOGIDAN" to "TEKNOLOGI DAN"
    (r'(SAINS)(TEKNOLOGI)', r'\1 \2'), # Fixes "SAINSTEKNOLOGI" to "SAINS TEKNOLOGI"
    # --- END NEW FIX ---
]

# --- LLM and RAG Configuration ---
AGENCY_EMBEDDING_MODEL = "mxbai-embed-large" # Model for embeddings (REVERTED)
AGENCY_LLM_MODEL = "mixtral" # Model for language generation (REVERTED)

# --- CHUNKING STRATEGY (ADDED) ---
AGENCY_CHUNK_SIZE = 1000    # Optimal chunk size for RAG context
AGENCY_CHUNK_OVERLAP = 200  # Overlap to maintain context between chunks

# --- DOCUMENT PROCESSING AND FILTERING FLAGS (ADDED) ---
AGENCY_EXCLUDE_HEADERS_FOOTERS = True
AGENCY_EXCLUDE_COLLAPSIBLE_SECTIONS = True

# Keywords to detect a query about roles/directory (used in context_retrieval for intent detection)
AGENCY_ROLE_KEYWORDS = [
    "pengarah", "director", "ketua", "pegawai", "juruteknik", "pembantu",
    "jurutera", "akauntan", "pendaftar", "setiausaha", "bahagian", "sektor",
    "unit", "jabatan", "timbalan", "pangkat", "jawatan", "ketua penolong",
    "penolong pengarah", "leader",
    # --- NEW FIX: Add more specific parts of complex job titles ---
    "pasukan inovasi digital", "pid", "kementerian sains", "teknologi dan inovasi",
    "pembangunan luar bandar", "korporat dan kualiti", "data raya", "perkongsian maklumat"
    # --- END NEW FIX ---
]

# Keywords to detect a query about contact info or office hours
AGENCY_CONTACT_HOURS_KEYWORDS = [
    "telefon", "phone", "email", "emel", "fax", "faks", "hubungi", "contact",
    "waktu pejabat", "office hours", "masa bekerja", "working hours", "operasi",
    "operating hours", "buka", "tutup", "open", "close"
]

# Keywords to detect a query about locations
AGENCY_LOCATION_KEYWORDS = [
    "alamat", "address", "lokasi", "location", "cawangan", "branch", "pejabat",
    "office", "ibu pejabat", "headquarters", "hq", "wilayah", "region", "di mana"
]

# Keywords to detect a query about attendance
AGENCY_ATTENDANCE_KEYWORDS = [
    "attendance", "clock-in", "clock-out", "hours worked", "leave", "cuti",
    "pergerakan", "emp_id", "att_date", "late_clock_in", "early_clock_out",
    "bilangan", "berapa kali", "how many times", "count times", "work days", "hari bekerja", # Added for counting work days
    "average", "purata", "mean", "rata-rata", "jam bekerja", "work hours"
]

# Mapping of internal page titles to their display names for metadata consistency
AGENCY_PAGE_TITLE_MAPPING = {
    "director_message": "Corporate Profile - Director's Message",
    "organization_chart": "Corporate Profile - Organization Chart",
    "office_locations": "Contact Us - Office Locations",
    "directory": "Directory",
    "mission_vision": "Corporate Profile - Mission & Vision",
    "gallery": "Gallery",
    "e_book": "Downloads - E-Book",
    "faq": "FAQ",
    "photo_video_gallery": "Photo & Video Gallery",
    "corporate_info": "Corporate Profile", # General for Director's Message, Mission/Vision
    "hubungi_kami": "Hubungi Kami" # General for contact us pages
}

# Specific rank titles for more precise extraction and filtering
AGENCY_SPECIFIC_RANKS = [
    "Pengarah",
    "Director", # --- NEW FIX: Added Director for English queries ---
    "Ketua Penolong Pengarah Bahagian Data Raya & Perkongsian Maklumat",
    "Ketua Penolong Pengarah Bahagian Korporat Dan Kualiti",
    "Ketua Penolong Pengarah",
    "Penolong Pengarah",
    "Pegawai Teknologi Maklumat",
    "Pen. Pegawai Teknologi Maklumat",
    "Juruteknik Komputer",
    "Pembantu Tadbir",
    "Pembantu Tadbir (Unit Pengurusan Dan Pentadbiran)",
    "Pasukan Inovasi Digital (PID)",
    "Ketua Pasukan Inovasi Digital",
    "Digital Apprentice",
    "Pembantu Khidmat Am",
    # --- NEW FIX: Added more specific complex ranks ---
    "PID Kementerian Sains, Teknologi Dan Inovasi",
    "Ketua Pasukan Inovasi Digital Kementerian Pembangunan Luar Bandar"
    # --- END NEW FIX ---
]

# Common words to filter out when extracting person names (lower case)
AGENCY_COMMON_NON_NAMES_LOWER = {
    "siapa", "who", "apakah", "what", "mengenai", "tentang", "nama", "adakah",
    "is", "are", "bagaimana", "how", "mana", "bila", "kenapa", "mengapa",
    "adakah", "dan", "atau", "dengan", "dari", "kepada", "ini", "itu",
    "untuk", "for", "dalam", "in", "pada", "at", "yang", "which", "ialah", "is", "adalah", "is",
    "beliau", "he", "she", "mereka", "they", "anda", "you", "kami", "we",
    "saya", "i", "dia", "he", "she", "mereka", "they", "jpkn", "jabatan",
    "perkhidmatan", "komputer", "negeri", "sabah", "bahagian", "unit", "sektor",
    "cawangan", "wilayah", "pejabat", "timbalan", "ketua", "penolong", "pegawai",
    "juruteknik", "pembantu", "pengarah", "urusan", "bagi", "setia", "urusan",
    "tentang", "mengenai", "untuk", "di", "apakah", "kenapa", "mengapa", "bagaimana",
    "bilakah", "siapakah", "bolehkah", "tolong", "bantu", "maklumat", "info",
    "contact", "hubungi", "nombor", "number", "email", "emel", "alamat", "address",
    "lokasi", "location", "waktu", "hours", "hari", "day", "cuti", "public",
    "bekerja", "working", "rasmi", "official", "waktu", "masa", "masa pejabat",
    "jumlah", "berapa", "bilangan", "berapa ramai", "berapa banyak", "pangkat", 
    "jawatan" 
}

# Query expansions for better retrieval (e.g., "tel" expands to "telefon")
AGENCY_QUERY_EXPANSIONS = {
    "tel": "telefon",
    "email": "emel",
    "hq": "ibu pejabat",
    "oc": "carta organisasi",
    "office hours": "waktu pejabat masa bekerja",
    "contact": "hubungi",
    "rank": "pangkat jawatan", 
    "jawatan": "pangkat role", 
    "director": "pengarah",
    "timbalan": "deputy",
    "jumlah": "bilangan berapa how many", 
    "berapa": "jumlah bilangan how many",
    "late": "lewat lambat",
    "leave": "cuti",
    "leader": "ketua",
    "how many": "jumlah bilangan berapa" 
}

# Keywords for Malay language detection (lower case)
AGENCY_MALAY_KEYWORDS = {
    "siapa", "apakah", "bagaimana", "mana", "bila", "kenapa", "mengapa",
    "adakah", "dan", "atau", "dengan", "dari", "kepada", "ini", "itu",
    "untuk", "dalam", "pada", "yang", "ialah", "adalah", "beliau", "mereka",
    "anda", "kami", "saya", "dia", "jpkn", "jabatan", "perkhidmatan", "komputer",
    "negeri", "sabah", "bahagian", "unit", "sektor", "cawangan", "wilayah", "pejabat",
    "timbalan", "ketua", "penolong", "pegawai", "juruteknik", "pembantu", "pengarah",
    "urusan", "bagi", "setia", "tentang", "mengenai", "nombor", "alamat",
    "waktu", "hari", "cuti", "bekerja", "rasmi", "masa", "jumlah", "berapa",
    "bilangan", "pangkat", "jawatan", "kehadiran", "lewat", "lambat", "telefon",
    "emel", "faks", "hubungi", "perkhidmatan", "komputer", "perutusan", "visi",
    "misi", "carta", "organisasi", "galeri", "muat", "turun", "e-buku", "soalan",
    "lazim", "gambar", "video", "perkara", "mengenai", "senarai", "maklumat",
    "berhubung", "setiausaha", "pejabat", "yg" 
}

# New: Keywords for English language detection (lower case) - prioritized over Malay
AGENCY_ENGLISH_KEYWORDS = {
    "how many", "who is", "what is", "tell me about", "where is", "when is",
    "contact", "phone", "email", "address", "office hours", "working hours",
    "does", "have", "has", "can you", "could you", "please", "rank", "name", 
    "full name", "leader", "about", "a", "an", "the", "of", "in", "on", "at",
    "my", "your", "his", "her", "their", "our", "find", "show", "list"
}

# --- External Data Transformation Rules ---
# Define how different external JSON structures should be transformed into
# 'page_content' and 'metadata' for ingestion into the vector store.
# Each key in this dictionary represents a 'data_type_key' that you would pass
# to the transform_external_data.py script.
AGENCY_EXTERNAL_DATA_TRANSFORMATION_RULES = {
    "attendance": {
        "root_key": "attendances", # This must match the key in your raw JSON
        "page_content_template": (
            "Attendance record for Employee ID {emp_id} (IC: {icno_display}) on {att_date}: "
            "Clock-in: {clock_in}, Clock-out: {clock_out}, Total hours worked: {total_hours}. "
            "Late Clock In: {late_clock_in_status}. "
            "Incomplete Clock In/Out: {incomplete_clock_in_out_status}. "
            "Early Clock Out: {early_clock_out_status}. "
            "{public_holiday_info}{luar_negeri_info}{kursus_info}{cuti_info}{keluar_pejabat_info}{catatan_info}"
        ),
        "page_content_template_fields": {
            # These are for formatting the page_content string based on original values
            "icno_display": lambda record: record.get("icno") if record.get("icno") and record.get("icno") != "(IC NUMBER)" else "N/A",
            "late_clock_in_status": lambda record: "True" if record.get("late_clock_in") else "False",
            "incomplete_clock_in_out_status": lambda record: "True" if record.get("incomplete_clock_in_out") else "False",
            "early_clock_out_status": lambda record: "True" if record.get("early_clock_out") else "False",
            "public_holiday_info": lambda record: f"Public holiday: {record.get('public_holiday')}. " if record.get('public_holiday') else "",
            "luar_negeri_info": lambda record: f"Out of state/country: {record.get('luar_negeri')}. " if record.get('luar_negeri') else "",
            "kursus_info": lambda record: f"Course/Training: {record.get('kursus')}. " if record.get('kursus') else "",
            "cuti_info": lambda record: f"On leave: {record.get('cuti')} (Type: {record.get('cuti_type')}). " if record.get('cuti') else "",
            "keluar_pejabat_info": lambda record: f"Out of office: {record.get('keluar_pejabat')} (Type: {record.get('jenis_pergerakan')}, Purpose: {record.get('tujuan')}). " if record.get('keluar_pejabat') else "",
            "catatan_info": lambda record: f"Notes: {record.get('catatan')}. " if record.get('catatan') else ""
        },
        "metadata_keys": [ # Fields from the original record to include directly in metadata
            "id", "emp_id", "icno", "dept_code", "att_date", "clock_in", "clock_out",
            "total_hours",
            # These will be explicitly converted to boolean in transform_external_data.py
            "incomplete_clock_in_out", "late_clock_in", "early_clock_out",
            "mtas", "public_holiday", "luar_negeri", "kursus", "cuti", "cuti_type",
            "keluar_pejabat", "jenis_pergerakan", "tujuan", "catatan", "departure_time",
            "return_time", "updated_at", "created_at"
        ],
        "source_type": "attendance_record"
    },
    # Add rules for other external data types here if needed in the future
    # "employee_details": {
    #     "root_key": "employees",
    #     "page_content_template": "Employee: {name}, Position: {position}, Department: {department}, Email: {email}.",
    #     "metadata_keys": ["employee_id", "name", "position", "department", "email", "phone"],
    #     "source_type": "employee_record"
    # }
}


# --- Static Office Hours Data (Manually inserted for reliability) ---
AGENCY_OFFICE_HOURS_DATA = {
    "malay": {
        "Hari Bekerja": "Isnin – Jumaat",
        "Masa Bekerja": "7:30 pagi – 5:00 petang",
        "Waktu Rehat": "1:00 petang – 2:00 petang (Isnin - Khamis), 12:00 tengah hari - 2:00 petang (Jumaat)",
        "Cuti": "Sabtu, Ahad, dan Cuti Umum Negeri Sabah serta Cuti Umum Persekutuan."
    },
    "english": {
        "Working Days": "Monday – Friday",
        "Working Hours": "7:30 AM – 5:00 PM",
        "Break Time": "1:00 PM – 2:00 PM (Monday - Thursday), 12:00 PM – 2:00 PM (Friday)",
        "Holidays": "Saturday, Sunday, and Public Holidays of Sabah State and Federal Public Holidays."
    },
    "summary_malay": "Waktu pejabat JPKN adalah Isnin hingga Jumaat, dari 7:30 pagi hingga 5:00 petang. Waktu rehat adalah dari 1:00 tengah hari hingga 2:00 petang (Isnin - Khamis) dan 12:00 tengah hari hingga 2:00 petang (Jumaat). JPKN bercuti pada hari Sabtu, Ahad, dan Cuti Umum Negeri Sabah serta Cuti Umum Persekutuan.",
    "summary_english": "JPKN office hours are Monday to Friday, from 7:30 AM to 5:00 PM. Break time is from 1:00 PM to 2:00 PM (Monday - Thursday) and 12:00 PM to 2:00 PM (Friday). JPKN is closed on Saturdays, Sundays, and Public Holidays of Sabah State and Federal Public Holidays."
}

# --- Content Formatting Function ---
def agency_format_content_case(text: str) -> str:
    """
    Applies specific capitalization rules for JPKN content.
    Ensures 'JPKN' is always uppercase, 'Sabah' is capitalized,
    and common role titles are capitalized.
    """
    if not isinstance(text, str):
        return text

    # Apply general title casing (first letter of each word capitalized)
    # Exclude common small words unless they are at the beginning of a sentence
    # This is a heuristic and might not be perfect for all cases
    def title_case_except_small_words(input_text):
        words = input_text.split()
        transformed_words = []
        small_words = {'dan', 'yang', 'dengan', 'untuk', 'di', 'pada', 'ini', 'itu', 'atau', 'dalam'} # Add more as needed
        
        for i, word in enumerate(words):
            if i == 0 or word.lower() not in small_words:
                transformed_words.append(word.capitalize())
            else:
                transformed_words.append(word.lower())
        return ' '.join(transformed_words)

    formatted_text = title_case_except_small_words(text.lower()) # Start with lowercasing and then apply custom title case

    # Specific replacements to ensure correct capitalization for known entities/terms
    # Use a list of (pattern, replacement) tuples for ordered application
    replacements_patterns = [
        (r'\bjpkn\b', 'JPKN'),
        (r'\bsabah\b', 'Sabah'),
        (r'\bpengarah\b', 'Pengarah'),
        (r'\bdirector\b', 'Director'), # Added for consistency
        (r'\btimbalan pengarah\b', 'Timbalan Pengarah'),
        (r'\bketua penolong pengarah\b', 'Ketua Penolong Pengarah'),
        (r'\bpenolong pengarah\b', 'Penolong Pengarah'),
        (r'\bpegawai teknologi maklumat\b', 'Pegawai Teknologi Maklumat'),
        (r'\bpen\. pegawai teknologi maklumat\b', 'Pen. Pegawai Teknologi Maklumat'),
        (r'\bjuruteknik komputer\b', 'Juruteknik Komputer'),
        (r'\bpembantu tadbir\b', 'Pembantu Tadbir'),
        (r'\bakauntan\b', 'Akauntan'),
        (r'\bjurutera\b', 'Jurutera'),
        (r'\bketua unit\b', 'Ketua Unit'),
        (r'\bjuruaudit\b', 'Juruaudit'),
        (r'\bpegawai undang-undang\b', 'Pegawai Undang-Undang'),
        (r'\bpegawai khidmat pelanggan\b', 'Pegawai Khidmat Pelanggan'),
        (r'\bpenolong jurutera\b', 'Penolong Jurutera'),
        (r'\bpenolong juruteknik komputer\b', 'Penolong Juruteknik Komputer'),
        (r'\bpembantu juruteknik\b', 'Pembantu Juruteknik'),
        (r'\bpembantu akauntan\b', 'Pembantu Akauntan'),
        (r'\bketua pembantu tadbir\b', 'Ketua Pembantu Tadbir'),
        (r'\bibu pejabat\b', 'Ibu Pejabat'),
        (r'\balamat cawangan\b', 'Alamat Cawangan'),
        (r'\btelefon\b', 'Telefon'),
        (r'\bemel\b', 'Emel'),
        (r'\bfaks\b', 'Faks'),
        (r'\bwaktu pejabat\b', 'Waktu Pejabat'),
        (r'\bhari bekerja\b', 'Hari Bekerja'),
        (r'\bcuti umum\b', 'Cuti Umum'),
        # --- NEW FIX: Handle concatenated words in job titles (more generic) ---
        (r'(digital)(kementerian)', r'Digital Kementerian', re.IGNORECASE), # Fixes "DIGITALKEMENTERIAN"
        (r'(teknologi)(dan)', r'Teknologi Dan', re.IGNORECASE),      # Fixes "TEKNOLOGIDAN"
        (r'(sains)(teknologi)', r'Sains Teknologi', re.IGNORECASE),    # Fixes "SAINSTEKNOLOGI"
        (r'(pasukan)(\s*)(inovasi)(\s*)(digital)(\s*)(kementerian)(\s*)(sains)(,?)(\s*)(teknologi)(,?)(\s*)(dan)(\s*)(inovasi)', r'Pasukan Inovasi Digital Kementerian Sains, Teknologi Dan Inovasi', re.IGNORECASE),
        (r'(ketua)(\s*)(pasukan)(\s*)(inovasi)(\s*)(digital)(\s*)(kementerian)(\s*)(pembangunan)(\s*)(luar)(\s*)(bandar)', r'Ketua Pasukan Inovasi Digital Kementerian Pembangunan Luar Bandar', re.IGNORECASE),
        (r'\bpid\b', 'PID'), # Ensure PID is capitalized
        # --- END NEW FIX ---
    ]

    for pattern_str, replacement_str, *flags in replacements_patterns:
        flags = flags[0] if flags else 0 # Get flags if present, else 0
        formatted_text = re.sub(pattern_str, replacement_str, formatted_text, flags=flags)

    # Ensure full sentences start with a capital letter
    formatted_text = re.sub(r'([.!?]\s*)([a-z])', lambda pat: pat.group(1) + pat.group(2).upper(), formatted_text)
    
    # Capitalize first letter of the whole text if not already capitalized
    if formatted_text and formatted_text[0].islower():
        formatted_text = formatted_text[0].upper() + formatted_text[1:]

    return formatted_text
