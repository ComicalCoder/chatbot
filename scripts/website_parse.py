import os
import json # Added import for json
import hashlib
import re
from collections import deque
from urllib.parse import urljoin, urlparse
import time
from typing import List, Dict, Any
import asyncio
from datetime import datetime
from pathlib import Path
import logging

from playwright.async_api import async_playwright, Page, Locator
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langdetect import detect

# Import the new universal configuration system and normalize_text
from universal_scraper_config import get_website_config, WebsiteConfig, normalize_text

# Import agency-specific page title mappings (still needed for metadata tagging)
from agency_config import AGENCY_PAGE_TITLE_MAPPING

# --- CONFIGURATION ---
# BASE_URL will now be determined by the config object
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")  # Directory to save JSON output
MAX_PAGES_TO_CRAWL = 100  # Limit the number of pages to crawl
CRAWL_DELAY_SECONDS = 1  # Delay between page visits
PLAYWRIGHT_TIMEOUT_MS = 90000  # Timeout for page loads
MAX_RETRIES = 5  # Number of retries for failed page loads
PARALLEL_PAGES = 3  # Number of parallel page crawls

# Text splitting configuration for extracted content
TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)

# --- SETUP LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(OUTPUT_DIR, 'scraper.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- HELPER FUNCTIONS ---
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

def save_documents_to_json(documents: List[Document], page_url: str, page_title: str, manifest: Dict[str, Any]):
    """Save documents to JSON and update manifest."""
    if not documents:
        return

    sanitized_title = sanitize_filename(page_title)
    page_url_short_hash = hashlib.md5(page_url.encode()).hexdigest()[:12]
    filename = f"{sanitized_title}_{page_url_short_hash}.json"
    filepath = os.path.join(OUTPUT_DIR, filename)

    doc_dicts = []
    for doc in documents:
        doc_dicts.append({
            "page_content": doc.page_content,
            "metadata": sanitize_metadata(doc.metadata) # Use updated sanitize_metadata
        })

    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(doc_dicts, f, ensure_ascii=False, indent=2)
        manifest[page_url] = {"filename": filename, "title": page_title, "doc_count": len(documents)}
        logger.info(f"Saved {len(documents)} document chunks to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save documents to {filepath}: {e}")

def save_manifest(manifest: Dict[str, Any]):
    """Save the manifest file."""
    manifest_path = os.path.join(OUTPUT_DIR, "manifest.json")
    try:
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved manifest to {manifest_path}")
    except Exception as e:
        logger.error(f"Failed to save manifest: {e}")

def _clean_html(html_content: str, config: WebsiteConfig) -> str:
    """Remove script, style, and boilerplate from HTML using website-specific configurations."""
    soup = BeautifulSoup(html_content, 'html.parser')

    # Remove elements specified in config.boilerplate_selectors
    for selector in config.boilerplate_selectors:
        for elem in soup.select(selector):
            elem.decompose()
            
    # Get text from the body (or remaining HTML). If body doesn't exist, get from soup directly.
    text_content = soup.body.get_text(separator=' ', strip=True) if soup.body else soup.get_text(separator=' ', strip=True)
    text_content = normalize_text(text_content) # Use imported normalize_text

    # Apply aggressive regex for remaining JavaScript/config fragments and boilerplate phrases
    for pattern in config.boilerplate_patterns:
        text_content = re.sub(pattern, '', text_content, flags=re.DOTALL).strip()

    return normalize_text(text_content) # Use imported normalize_text

def compute_text_similarity(text1: str, text2: str) -> float:
    """Compute cosine similarity between two texts."""
    vectorizer = TfidfVectorizer()
    # Handle empty strings for TF-IDF
    if not text1 or not text2:
        return 0.0
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

async def extract_text_and_metadata(page: Page, url: str, depth: int, config: WebsiteConfig) -> List[Document]:
    """Extract text and metadata, handling dynamic content using the provided config."""
    documents = []
    page_title = await page.title() or "No Title"
    current_url_hash = hashlib.md5(url.encode()).hexdigest()
    crawl_timestamp = datetime.utcnow().isoformat()
    
    # Get full HTML content for BeautifulSoup cleaning
    raw_html_content = await page.content()
    cleaned_html_text_for_general_use = _clean_html(raw_html_content, config) # Pass config to _clean_html

    try:
        language = detect(await page.text_content("body") or "")
    except Exception:
        language = "unknown"

    logger.info(f"Extracting content from: {url} (Depth: {depth})")

    # Header
    for selector in config.header_selectors: # Use config's selectors
        try:
            header_locator = page.locator(selector)
            if await header_locator.count() > 0:
                header_text = normalize_text(await header_locator.first.text_content()) # Use imported normalize_text
                if header_text:
                    documents.extend(TEXT_SPLITTER.create_documents([header_text], metadatas=[{
                        "url": url, "page_title": page_title, "type": "header",
                        "source_file": current_url_hash, "selector_used": selector,
                        "depth": depth, "crawl_timestamp": crawl_timestamp, "language": language
                    }]))
                    logger.info(f"Extracted header using '{selector}'. Length: {len(header_text)} chars.")
                    break
        except Exception as e:
            logger.warning(f"Error with header selector '{selector}': {e}")

    # Footer
    for selector in config.footer_selectors: # Use config's selectors
        try:
            footer_locator = page.locator(selector)
            if await footer_locator.count() > 0:
                footer_text = normalize_text(await footer_locator.first.text_content()) # Use imported normalize_text
                if footer_text:
                    documents.extend(TEXT_SPLITTER.create_documents([footer_text], metadatas=[{
                        "url": url, "page_title": page_title, "type": "footer",
                        "source_file": current_url_hash, "selector_used": selector,
                        "depth": depth, "crawl_timestamp": crawl_timestamp, "language": language
                    }]))
                    logger.info(f"Extracted footer using '{selector}'. Length: {len(footer_text)} chars.")
                    break
        except Exception as e:
            logger.warning(f"Error with footer selector '{selector}': {e}")

    # Contact Info (Triggered by page title mapping and config method)
    if page_title == AGENCY_PAGE_TITLE_MAPPING.get("office_locations"): # Still use agency_config for page title mapping
        contact_details = config.extract_contact_info(page) # Call config's method
        if contact_details:
            contact_content = "Contact Information:\n" + "\n".join([f"{k}: {v}" for k, v in contact_details.items()])
            documents.extend(TEXT_SPLITTER.create_documents([contact_content], metadatas=[{
                "url": url, "page_title": page_title, "type": "contact_info",
                "source_file": current_url_hash, "depth": depth,
                "crawl_timestamp": crawl_timestamp, "language": language,
                **contact_details
            }]))
            logger.info("Extracted contact info using config method.")

    # Office Hours: No longer scraped by website_parse.py. It's now a static document added by create_vector_store.py
    # The config.extract_office_hours method exists but will return empty dict for SabahGovConfig
    # if page_title == AGENCY_PAGE_TITLE_MAPPING.get("office_locations"):
    #     office_hours_details = config.extract_office_hours(page)
    #     if office_hours_details:
    #         hours_content = "Office Hours:\n" + "\n".join([f"{k}: {v}" for k, v in office_hours_details.items()])
    #         documents.extend(TEXT_SPLITTER.create_documents([hours_content], metadatas=[{
    #             "url": url, "page_title": page_title, "type": "office_hours",
    #             "source_file": current_url_hash, "depth": depth,
    #             "crawl_timestamp": crawl_timestamp, "language": language,
    #             **office_hours_details
    #         }]))
    #         logger.info("Extracted office hours using config method.")


    # Dynamic Content (Dropdowns, Accordions, Modals)
    for selector in config.dropdown_selectors: # Use config's selectors
        try:
            elements = await page.locator(selector).all()
            if elements:
                logger.info(f"Found {len(elements)} collapsible elements with selector '{selector}'.")
                for i, element in enumerate(elements):
                    if await element.is_visible():
                        try:
                            await element.click(timeout=3000)
                            await page.wait_for_timeout(1000)  # Wait for content to load
                            
                            expanded_html_snippet = await element.evaluate_handle('(el) => { return el.parentElement.innerHTML; }')
                            expanded_html_content = await expanded_html_snippet.json_value()
                            
                            expanded_text = normalize_text(_clean_html(expanded_html_content, config)) # Use imported normalize_text and pass config
                            
                            if expanded_text and len(expanded_text) > 50:
                                documents.extend(TEXT_SPLITTER.create_documents([expanded_text], metadatas=[{
                                    "url": url, "page_title": page_title, "type": "collapsible_section",
                                    "section_title": normalize_text(await element.text_content() or "")[:100], # Use imported normalize_text
                                    "source_file": current_url_hash, "depth": depth,
                                    "crawl_timestamp": crawl_timestamp, "language": language
                                }]))
                                logger.info(f"Extracted collapsible content {i+1} using '{selector}'.")
                            else:
                                logger.info(f"Expanded content from element {i+1} was too short or empty after cleaning.")
                        except Exception as e:
                            logger.warning(f"Could not process collapsible element {i+1} with selector '{selector}': {e}")
        except Exception as e:
            logger.warning(f"Error finding elements with selector '{selector}': {e}")

    # Table Data
    tables = await page.locator("table").all()
    if tables:
        logger.info(f"Found {len(tables)} tables.")
        for i, table in enumerate(tables):
            try:
                headers = [normalize_text(await th.text_content()) for th in await table.locator("th").all()] # Use imported normalize_text
                rows = await table.locator("tbody tr").all()
                for r_idx, row in enumerate(rows):
                    cells = [normalize_text(await td.text_content()) for td in await row.locator("td").all()] # Use imported normalize_text
                    
                    row_data = ""
                    if headers and len(headers) == len(cells):
                        row_data = ", ".join(f"{h.strip()}: {c.strip()}" for h, c in zip(headers, cells) if h.strip() and c.strip())
                    else:
                        row_data = ", ".join(f"Column {j+1}: {c.strip()}" for j, c in enumerate(cells) if c.strip())

                    if row_data:
                        documents.extend(TEXT_SPLITTER.create_documents([row_data], metadatas=[{
                            "url": url, "page_title": page_title, "type": "table_data",
                            "table_index": i, "row_index": r_idx, "table_headers": ", ".join(headers) if headers else None,
                            "source_file": current_url_hash, "depth": depth,
                            "crawl_timestamp": crawl_timestamp, "language": language
                        }]))
                logger.info(f"Extracted table {i+1}.")
            except Exception as e:
                logger.warning(f"Could not extract table {i+1}: {e}")

    # General Main Content
    # Use the pre-cleaned text from the entire page for general content
    if cleaned_html_text_for_general_use and len(cleaned_html_text_for_general_use) > 100:
        is_duplicate = any(compute_text_similarity(cleaned_html_text_for_general_use, doc.page_content) > 0.9 for doc in documents)
        if not is_duplicate:
            documents.extend(TEXT_SPLITTER.create_documents([cleaned_html_text_for_general_use], metadatas=[{
                "url": url, "page_title": page_title, "type": "main_content",
                "source_file": current_url_hash, "depth": depth,
                "crawl_timestamp": crawl_timestamp, "language": language
            }]))
            logger.info("Extracted general main content.")
        else:
            logger.info("Main content skipped due to similarity with structured data.")
    else:
        logger.info("General main content too short or empty after cleaning, skipping.")

    # Custom content extractors defined in the config
    for custom_extractor in config.custom_content_extractors():
        try:
            # Pass normalize_text to custom extractors if they need it
            custom_extracted_data = await custom_extractor(page, normalize_text)
            if custom_extracted_data:
                if isinstance(custom_extracted_data, list):
                    documents.extend(custom_extracted_data)
                elif isinstance(custom_extracted_data, Document):
                    documents.append(custom_extracted_data)
                else:
                    logger.warning(f"Custom extractor returned unexpected type: {type(custom_extracted_data)}")
                logger.info(f"Executed custom extractor: {custom_extractor.__name__}")
        except Exception as e:
            logger.error(f"Error executing custom extractor {custom_extractor.__name__}: {e}")

    return documents

async def get_internal_links(page: Page, base_url: str, config: WebsiteConfig) -> set[str]:
    """Extract internal links, filtering out external domains and ignored extensions using config."""
    internal_links = set()
    base_netloc = urlparse(base_url).netloc

    for link in await page.locator("a[href]").all():
        try:
            href = await link.get_attribute("href")
            if href:
                full_url = urljoin(page.url, href)
                parsed_url = urlparse(full_url)
                if (parsed_url.netloc == base_netloc and
                    parsed_url.scheme in ['http', 'https'] and
                    not parsed_url.fragment and
                    not full_url.startswith(("mailto:", "tel:")) and
                    not parsed_url.path.lower().endswith(config.ignored_extensions) and # Use config's ignored_extensions
                    not any(domain in parsed_url.netloc for domain in config.external_domains_to_skip)): # Use config's external_domains_to_skip
                    normalized_url = parsed_url._replace(query="", fragment="").geturl()
                    internal_links.add(normalized_url)
        except Exception:
            pass
    return internal_links

async def crawl_page(page: Page, url: str, base_url: str, visited_urls: set, urls_to_visit: deque, manifest: Dict[str, Any], depth: int, config: WebsiteConfig):
    """Crawl a single page with retries using the provided config."""
    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"Crawling (Attempt {attempt+1}/{MAX_RETRIES}): {url}")
            await page.goto(url, wait_until="load", timeout=PLAYWRIGHT_TIMEOUT_MS)
            await page.wait_for_timeout(3000)
            documents = await extract_text_and_metadata(page, url, depth, config) # Pass config
            page_title = await page.title() or "No Title"
            save_documents_to_json(documents, url, page_title, manifest)
            new_links = await get_internal_links(page, base_url, config) # Pass config
            for link in new_links:
                if link not in visited_urls:
                    urls_to_visit.append((link, depth + 1))
            logger.info(f"Found {len(new_links)} internal links. Queue size: {len(urls_to_visit)}")
            return True
        except Exception as e:
            logger.error(f"Failed to crawl {url} (Attempt {attempt+1}): {e}")
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
    return False

async def crawl_website(start_url: str, output_dir: str, max_pages: int, delay: int, timeout_ms: int):
    """Crawl the website with parallel page processing, using dynamic config."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for file in Path(output_dir).glob("*.json"):
        file.unlink()
    logger.info(f"Cleared existing JSON files in {output_dir}")

    # Get the initial configuration based on the start_url
    initial_config = get_website_config(start_url)
    base_url_for_crawl = initial_config.base_url # Use the base_url from the determined config

    visited_urls = set()
    urls_to_visit = deque([(start_url, 0)])  # (url, depth)
    crawled_page_count = 0
    manifest = {}

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()

        async def process_page_task():
            nonlocal crawled_page_count
            page = await context.new_page()
            while urls_to_visit and crawled_page_count < max_pages:
                try:
                    url, depth = urls_to_visit.popleft()
                except IndexError:
                    await asyncio.sleep(0.1) # Wait a bit if queue is temporarily empty
                    continue

                if url in visited_urls:
                    logger.info(f"Skipping visited URL: {url}")
                    continue
                
                # Get config for the current URL (could be different if crawling multiple domains)
                current_config = get_website_config(url)

                # Check if it's an ignored extension before adding to visited
                if any(url.lower().endswith(ext) for ext in current_config.ignored_extensions):
                    logger.info(f"Skipping ignored file type: {url}")
                    visited_urls.add(url) # Mark as visited to avoid re-queuing
                    continue

                visited_urls.add(url)
                crawled_page_count += 1
                success = await crawl_page(page, url, base_url_for_crawl, visited_urls, urls_to_visit, manifest, depth, current_config) # Pass current_config
                if success:
                    await asyncio.sleep(delay)
            await page.close()

        tasks = [process_page_task() for _ in range(min(PARALLEL_PAGES, max_pages))]
        await asyncio.gather(*tasks)
        await browser.close()

    save_manifest(manifest)
    logger.info(f"Crawl finished. Visited {crawled_page_count} pages.")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Use the base_url from the SabahGovConfig as the starting point
    initial_config = get_website_config("https://jpkn.sabah.gov.my/")
    BASE_URL_FOR_CRAWL = initial_config.base_url

    asyncio.run(crawl_website(BASE_URL_FOR_CRAWL, OUTPUT_DIR, MAX_PAGES_TO_CRAWL, CRAWL_DELAY_SECONDS, PLAYWRIGHT_TIMEOUT_MS))
    logger.info("Website parsing complete. Run create_vector_store.py next.")
