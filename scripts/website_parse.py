import os
import json
import hashlib
import re
import unicodedata
from collections import deque
from urllib.parse import urljoin, urlparse
import time

from playwright.sync_api import sync_playwright, Page, Locator
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- CONFIGURATION ---
BASE_URL = "https://jpkn.sabah.gov.my/"  # Starting URL for the crawler
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data") # Directory to save JSON output
MAX_PAGES_TO_CRAWL = 100 # Limit the number of pages to crawl
CRAWL_DELAY_SECONDS = 1 # Delay between page visits to be polite to the server
PLAYWRIGHT_TIMEOUT_MS = 90000 # Increased timeout to 90 seconds

# Text splitting configuration for extracted content
TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)

# File extensions to ignore during crawling (Playwright cannot parse these as HTML)
IGNORED_FILE_EXTENSIONS = (
    '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', 
    '.zip', '.rar', '.7z', '.tar', '.gz', '.jpg', '.jpeg', '.png', 
    '.gif', '.bmp', '.svg', '.mp3', '.mp4', '.avi', '.mov', '.webp'
)

# --- HELPER FUNCTIONS ---

def normalize_text(text: str) -> str:
    """Normalize text: fix whitespace and Unicode issues."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def sanitize_metadata(metadata: dict) -> dict:
    """
    Sanitizes metadata dictionary to ensure all values are simple types
    (str, int, float, bool, or None) for ChromaDB compatibility.
    Converts lists to comma-separated strings.
    """
    sanitized = {}
    for key, value in metadata.items():
        if isinstance(value, list):
            sanitized[key] = ", ".join(map(str, value))
        elif isinstance(value, (str, int, float, bool)) or value is None:
            sanitized[key] = value
        else:
            sanitized[key] = str(value) # Convert any other complex types to string
    return sanitized

def sanitize_filename(text: str, max_length: int = 50) -> str:
    """
    Sanitizes a string to be used as a filename.
    Removes invalid characters, replaces spaces, and truncates.
    """
    # Replace spaces with underscores
    s = text.replace(" ", "_")
    # Remove characters that are not alphanumeric, underscore, or hyphen
    s = re.sub(r"[^\w.-]", "", s)
    # Replace multiple underscores/hyphens with a single one
    s = re.sub(r"[_.-]+", "_", s)
    # Trim leading/trailing underscores/hyphens
    s = s.strip("_.-")
    # Truncate to max_length
    return s[:max_length]

def save_documents_to_json(documents: list[Document], page_url: str, page_title: str):
    """
    Saves a list of Document objects to a JSON file.
    Each original page will have its own JSON file.
    The filename will be based on the page title and a hash of the URL.
    """
    if not documents:
        return

    sanitized_title = sanitize_filename(page_title)
    page_url_short_hash = hashlib.md5(page_url.encode()).hexdigest()[:12] # Use first 12 chars

    filename = f"{sanitized_title}_{page_url_short_hash}.json"
    filepath = os.path.join(OUTPUT_DIR, filename)

    doc_dicts = []
    for doc in documents:
        doc_dicts.append({
            "page_content": doc.page_content,
            "metadata": sanitize_metadata(doc.metadata) # This line calls sanitize_metadata
        })

    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(doc_dicts, f, ensure_ascii=False, indent=2)
        print(f"  Saved {len(documents)} document chunks to {filepath}")
    except Exception as e:
        print(f"  [ERROR] Failed to save documents to {filepath}: {e}")

def extract_text_and_metadata(page: Page, url: str) -> list[Document]:
    """
    Extracts all visible text from the page, including handling dropdowns/collapsible sections,
    and captures header/footer content with specific metadata.
    """
    documents = []
    page_title = page.title() if page.title() else "No Title"
    current_url_hash = hashlib.md5(url.encode()).hexdigest()

    print(f"  Extracting content from: {url}")
    print(f"  Page Title: {page_title}")

    # --- Extract Header Content ---
    header_selectors = [
        "header", 
        "[role='banner']", 
        "#header", 
        ".main-header", 
        ".site-header",
        ".header-area", 
        ".top-bar" 
    ]
    header_content_extracted = False
    for selector in header_selectors:
        try:
            header_locator = page.locator(selector)
            if header_locator.count() > 0:
                header_text_list = header_locator.all_text_contents()
                header_text = normalize_text(" ".join(header_text_list))
                
                if header_text:
                    documents.extend(TEXT_SPLITTER.create_documents([header_text], metadatas=[{
                        "url": url,
                        "page_title": page_title,
                        "type": "header",
                        "source_file": current_url_hash,
                        "selector_used": selector
                    }]))
                    print(f"    Extracted header content using '{selector}'. Length: {len(header_text)} chars.")
                    header_content_extracted = True
                    break 
                else:
                    print(f"    [DEBUG] Header selector '{selector}' found, but no text content. Inner HTML (first 200 chars): {header_locator.inner_html()[:200]}...")
            else:
                print(f"    [DEBUG] Header selector '{selector}' not found.")
        except Exception as e:
            print(f"    [Warning] Error with header selector '{selector}': {e}")
    if not header_content_extracted:
        print("    No header content extracted using available selectors.")

    # --- Extract Footer Content ---
    footer_selectors = [
        "footer", 
        "[role='contentinfo']", 
        "#footer", 
        ".main-footer", 
        ".site-footer",
        ".footer-area", 
        ".contact-info", 
        ".widget-area" 
    ]
    footer_content_extracted = False
    for selector in footer_selectors:
        try:
            footer_locator = page.locator(selector)
            if footer_locator.count() > 0:
                footer_text_list = footer_locator.all_text_contents()
                footer_text = normalize_text(" ".join(footer_text_list))
                
                if footer_text:
                    documents.extend(TEXT_SPLITTER.create_documents([footer_text], metadatas=[{
                        "url": url,
                        "page_title": page_title,
                        "type": "footer",
                        "source_file": current_url_hash,
                        "selector_used": selector
                    }]))
                    print(f"    Extracted footer content using '{selector}'. Length: {len(footer_text)} chars.")
                    footer_content_extracted = True
                    break 
                else:
                    print(f"    [DEBUG] Footer selector '{selector}' found, but no text content. Inner HTML (first 200 chars): {footer_locator.inner_html()[:200]}...")
            else:
                print(f"    [DEBUG] Footer selector '{selector}' not found.")
        except Exception as e:
            print(f"    [Warning] Error with footer selector '{selector}': {e}")
    if not footer_content_extracted:
        print("    No footer content extracted using available selectors.")


    # --- Handle Dropdowns/Collapsible Sections ---
    dropdown_selectors = [
        "[data-toggle='collapse']", 
        "[aria-expanded='false']",  
        ".accordion-header",        
        ".collapsible-btn",         
        "button:has-text('Read More')", 
        "a:has-text('View Details')"
    ]
    
    for selector in dropdown_selectors:
        try:
            collapsible_elements = page.locator(selector).all()
            if collapsible_elements:
                print(f"    Found {len(collapsible_elements)} potential collapsible elements with selector '{selector}'.")
                for i, element in enumerate(collapsible_elements):
                    if element.is_visible():
                        try:
                            element.click(timeout=3000) 
                            page.wait_for_timeout(500) 
                            
                            expanded_text_area_list = page.locator("body").all_text_contents()
                            expanded_content = normalize_text(" ".join(expanded_text_area_list))

                            if expanded_content:
                                documents.extend(TEXT_SPLITTER.create_documents([expanded_content], metadatas=[{
                                    "url": url,
                                    "page_title": page_title,
                                    "type": "collapsible_section_text", 
                                    "section_title": normalize_text(element.text_content()[:100]), 
                                    "source_file": current_url_hash
                                }]))
                                print(f"      Expanded and extracted content from element {i+1} using '{selector}'.")

                        except Exception as click_err:
                            print(f"      [Warning] Could not click or extract from collapsible element {i+1} with selector '{selector}': {click_err}")
        except Exception as sel_err:
            print(f"    [Warning] Error finding elements with selector '{selector}': {sel_err}")

    # --- Extract Table Data ---
    tables = page.locator("table").all()
    if tables:
        print(f"    Found {len(tables)} tables.")
        for i, table in enumerate(tables):
            try:
                headers = [normalize_text(th.text_content()) for th in table.locator("th").all()]
                rows = table.locator("tbody tr").all()
                for r_idx, row in enumerate(rows):
                    cells = [normalize_text(td.text_content()) for td in row.locator("td").all()]
                    
                    row_data_str = ""
                    if headers and len(headers) == len(cells):
                        row_data_str = ", ".join(f"{h}: {c}" for h, c in zip(headers, cells))
                    else:
                        row_data_str = ", ".join(cells) 

                    if row_data_str:
                        documents.extend(TEXT_SPLITTER.create_documents([row_data_str], metadatas=[{
                            "url": url,
                            "page_title": page_title,
                            "type": "table_data",
                            "table_index": i,
                            "row_index": r_idx,
                            "table_headers": ", ".join(headers) if headers else None,
                            "source_file": current_url_hash
                        }]))
                print(f"      Extracted content from table {i+1}.")
            except Exception as e:
                print(f"      [Warning] Could not extract content from table {i+1}: {e}")

    # --- Extract Main Body Text ---
    try:
        main_content_locator = page.locator("main, article, .main-content, #content")
        if main_content_locator.count() > 0:
            main_text_list = main_content_locator.all_text_contents()
            main_text = normalize_text(" ".join(main_text_list))
            if main_text:
                documents.extend(TEXT_SPLITTER.create_documents([main_text], metadatas=[{
                    "url": url,
                    "page_title": page_title,
                    "type": "main_content",
                    "source_file": current_url_hash
                }]))
                print("    Extracted main content.")
        else:
            body_text_list = page.locator("body").all_text_contents()
            body_text = normalize_text(" ".join(body_text_list))
            if body_text:
                documents.extend(TEXT_SPLITTER.create_documents([body_text], metadatas=[{
                    "url": url,
                    "page_title": page_title,
                    "type": "full_page_content",
                    "source_file": current_url_hash
                }]))
                print("    Extracted full page content (no specific main area found).")

    except Exception as e:
        print(f"    [Warning] Could not extract main/full page content: {e}")

    return documents

def get_internal_links(page: Page, base_url: str) -> set[str]:
    """
    Extracts all unique internal links from the current page,
    ignoring specific file extensions.
    """
    internal_links = set()
    base_netloc = urlparse(base_url).netloc

    for link_locator in page.locator("a[href]").all():
        try:
            href = link_locator.get_attribute("href")
            if href:
                full_url = urljoin(page.url, href)
                parsed_full_url = urlparse(full_url)

                if parsed_full_url.path.lower().endswith(IGNORED_FILE_EXTENSIONS):
                    print(f"    [DEBUG] Skipping link to ignored file type: {full_url}")
                    continue

                if (parsed_full_url.netloc == base_netloc and
                    parsed_full_url.scheme in ['http', 'https'] and
                    not parsed_full_url.fragment and 
                    not full_url.startswith("mailto:") and
                    not full_url.startswith("tel:")):
                    
                    normalized_url = parsed_full_url._replace(query="", fragment="").geturl()
                    internal_links.add(normalized_url)
        except Exception as e:
            pass 

    return internal_links

# --- MAIN CRAWLER FUNCTION ---

def crawl_website(base_url: str, output_dir: str, max_pages: int, delay: int, timeout_ms: int):
    """
    Crawls a website, extracts content, and saves it to JSON files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    else:
        print(f"Output directory already exists: {output_dir}")
        print("Clearing existing JSON files in output directory.")
        for file in os.listdir(output_dir):
            if file.endswith(".json"):
                os.remove(os.path.join(output_dir, file))


    visited_urls = set()
    urls_to_visit = deque([base_url])
    crawled_page_count = 0

    # base_url_hash is no longer used for filename, but kept for metadata consistency if needed
    base_url_hash = hashlib.md5(base_url.encode()).hexdigest() 

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True) 
        context = browser.new_context()
        page = context.new_page()

        print(f"Starting crawl from: {base_url}")

        while urls_to_visit and crawled_page_count < max_pages:
            current_url = urls_to_visit.popleft()

            if current_url in visited_urls:
                print(f"Skipping already visited URL: {current_url}")
                continue

            print(f"\nCrawling page {crawled_page_count + 1}/{max_pages}: {current_url}")
            visited_urls.add(current_url)
            crawled_page_count += 1

            try:
                page.goto(current_url, wait_until="load", timeout=timeout_ms) 
                page.wait_for_timeout(3000) 
                
                documents = extract_text_and_metadata(page, current_url)
                
                page_title_for_filename = page.title() if page.title() else "no_title"
                save_documents_to_json(documents, current_url, page_title_for_filename)

                new_links = get_internal_links(page, base_url)
                for link in new_links:
                    if link not in visited_urls:
                        urls_to_visit.append(link)
                
                print(f"  Found {len(new_links)} internal links. {len(urls_to_visit)} links in queue.")

                time.sleep(delay) 
            except Exception as e:
                print(f"[ERROR] Failed to crawl {current_url}: {e}")
                # Optionally, re-add to end of queue or log for later retry

        browser.close()
        print(f"\nCrawl finished. Visited {crawled_page_count} pages.")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    if os.path.exists(OUTPUT_DIR):
        print(f"Clearing existing data in {OUTPUT_DIR}...")
        for file in os.listdir(OUTPUT_DIR):
            if file.endswith(".json"):
                os.remove(os.path.join(OUTPUT_DIR, file))
        print("Existing JSON files cleared.")
    else:
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

    crawl_website(BASE_URL, OUTPUT_DIR, MAX_PAGES_TO_CRAWL, CRAWL_DELAY_SECONDS, PLAYWRIGHT_TIMEOUT_MS)
    print("\nWebsite parsing complete. Run create_vector_store.py next to update your vector database.")
