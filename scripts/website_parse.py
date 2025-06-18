import requests
import hashlib
import os
import json
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from multiprocessing import Pool
from time import sleep

# --- CONFIG ---
BASE_URL = "https://jpkn.sabah.gov.my/"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
MAX_DEPTH = 5
REQUEST_DELAY = 1  # polite delay in seconds
CHUNK_SIZE = 500
NUM_WORKERS = 6  # adjust based on CPU cores
os.makedirs(OUTPUT_DIR, exist_ok=True)

def chunk_text(text, chunk_size=CHUNK_SIZE):
    """Split text into RAG-friendly chunks."""
    words = text.split()
    chunks = []
    current = []
    for word in words:
        current.append(word)
        if len(" ".join(current)) >= chunk_size:
            chunks.append(" ".join(current))
            current = []
    if current:
        chunks.append(" ".join(current))
    return chunks

def scrape_page(url):
    """Extract and return structured content from a page."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup(["script", "style", "header", "footer", "nav", "form", "svg"]):
            tag.decompose()

        main_content = soup.find("main") or soup.find("div", {"id": "content"}) or soup.body
        if not main_content:
            return None

        sections = []
        for tag in main_content.find_all(["h1", "h2", "h3", "h4", "p", "li", "table", "address"], recursive=True):
            text = tag.get_text(strip=True)
            if text:
                sections.append(text)

        full_text = "\n".join(sections)
        chunks = chunk_text(full_text)

        return {
            "url": url,
            "title": soup.title.string.strip() if soup.title else "",
            "raw_text": full_text,
            "chunks": chunks,
        }
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None

def save_text_as_json(url, data):
    """Save page content to a JSON file."""
    if not data:
        return
    filename_hash = hashlib.md5(url.encode()).hexdigest()
    filename = url.replace("https://", "").replace("/", "_").replace("?", "_").replace("&", "_")
    if len(filename) > 100:
        filename = filename_hash
    filepath = os.path.join(OUTPUT_DIR, f"{filename}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def get_links(url):
    """Extract internal links from a page."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        links = set()
        for a in soup.find_all("a", href=True):
            href = urljoin(BASE_URL, a["href"])
            if href.startswith(BASE_URL) and not href.endswith((".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xlsx")):
                links.add(href)
        return links
    except Exception as e:
        print(f"Error getting links from {url}: {e}")
        return set()

def process_url(args):
    """Process a single URL: scrape, save, and extract links."""
    url, depth, max_depth = args
    if depth > max_depth:
        return []

    print(f"[Depth {depth}] Scraping: {url}")
    data = scrape_page(url)
    save_text_as_json(url, data)
    sleep(REQUEST_DELAY)
    links = get_links(url)
    return [(link, depth + 1, max_depth) for link in links]

def parallel_crawl(start_url, max_depth=MAX_DEPTH):
    seen = set()
    queue = [(start_url, 0, max_depth)]

    with Pool(NUM_WORKERS) as pool:
        while queue:
            queue = [item for item in queue if item[0] not in seen]
            for item in queue:
                seen.add(item[0])

            print(f"\n🔄 Processing {len(queue)} URLs in parallel...\n")
            results = pool.map(process_url, queue)
            queue = [item for sublist in results for item in sublist]

def main():
    parallel_crawl(BASE_URL)

if __name__ == "__main__":
    main()