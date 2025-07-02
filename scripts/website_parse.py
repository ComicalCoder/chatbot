import requests
import hashlib
import os
import json
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from multiprocessing import Pool
from time import sleep
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import io
import re
import unicodedata

# --- CONFIG ---
BASE_URL = "https://jpkn.sabah.gov.my/"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
MAX_DEPTH = 20
REQUEST_DELAY = 1
CHUNK_SIZE = 500
NUM_WORKERS = 6
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

def clean_ocr_text(text):
    """Clean OCR text to fix spacing, errors, and remove noise."""
    # Remove footer and noise
    text = re.sub(r"(?i)(SABAH\.?GOV\.?MY|Tingkat 6 & 7|Menara Kinabalu|Teluk Likas|88400 Kota Kinabalu|No Tel/faks|WAKTU PEJABAT|Hari Bekerja|Loading\.\.\.|SABAH MAJU JAYA|© Hak Cipta|JPKN.*BAYU|Pelawat Hari Ini|Jumlah Pelawat|Jumlah Capaian|Last updated|Dasar Privasi|Notis Penafian|Webmail|PAKSi|E-Circular)", "", text)
    text = re.sub(r"[©\(\)\*\!\|\[\]\{\}]", "", text)
    # Fix OCR errors for /188-2/
    replacements = {
        r"\bVIS\b|\bV\s*i\s*s\s*i\b|\bV1S1\b": "Visi",
        r"\bMIS\b|\bM\s*i\s*s\s*i\b|\bM1S1\b": "Misi",
        r"\bMOTTO\b|\bMotoo\b|\bM0T0\b": "Moto",
        r"\bDAARUji\b|\bDasar\s*k\s*u\s*a\s*l\s*i\s*t\s*i\b|\bD\s*a\s*s\s*a\s*r\b": "Dasar Kualiti",
        r"\bOGlg\b|\bO\s*b\s*j\s*e\s*k\s*t\s*i\s*f\b|\b0bj3kt1f\b": "Objektif"
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    # Normalize whitespace and add spacing after headers
    text = re.sub(r"(?i)(Visi|Misi|Moto|Dasar Kualiti|Objektif)(?=\S)", r"\1 ", text)
    text = re.sub(r"\s+", " ", text).strip()
    print(f"[Debug] Cleaned OCR text: {text[:200]}...")
    return text

def extract_image_text(image_url):
    """Extract text from an image using enhanced OCR."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(image_url, headers=headers, timeout=10)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content))
        texts = []
        # Multi-scale preprocessing
        for scale in [1.0, 1.5]:
            scaled_image = image.resize((int(image.width * scale), int(image.height * scale)), Image.Resampling.LANCZOS)
            for threshold in [120, 140, 160]:
                gray = scaled_image.convert("L")
                enhanced = ImageEnhance.Contrast(gray).enhance(3.0)
                enhanced = enhanced.filter(ImageFilter.SHARPEN)
                binary = enhanced.point(lambda x: 0 if x < threshold else 255)
                config = "--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789:.- "
                text = pytesseract.image_to_string(binary, lang="eng+msa", config=config)
                if text.strip():
                    texts.append(clean_ocr_text(text))
        combined_text = " ".join([t for t in texts if t])
        print(f"[Debug] Combined OCR text for {image_url}: {combined_text[:200]}...")
        return combined_text.strip()
    except Exception as e:
        print(f"[ERROR] Failed to extract text from image {image_url}: {e}")
        return ""

def is_valid_url(url):
    """Check if URL is likely valid before scraping."""
    return url.startswith(BASE_URL) and not any(x in url for x in ["#", ".pdf", ".jpg", ".jpeg", ".png"])

def normalize_text(text):
    """Normalize text: fix whitespace and Unicode issues."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def scrape_page(url):
    """Extract and return structured content from a page."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        print(f"[Debug] Page fetched for {url}")

        # Remove scripts and styles
        for tag in soup(["script", "style"]):
            tag.decompose()

        # Remove footer
        for footer in soup.find_all("footer"):
            footer.decompose()

        # Target main content
        main_content = (soup.find("div", id="content") or
                        soup.find("div", class_=re.compile("entry-content|post-content")) or
                        soup.find("main") or
                        soup.body)
        if not main_content:
            print(f"[Debug] No main content found for {url}")
            return None

        print(f"[Debug] Main content tags for {url}: {main_content.name}, classes: {main_content.get('class', [])}")

        sections = []
        table_data = []
        structured_data = []

        # Handle accordion-based directory
        if "direktori" in url:
            accordions = main_content.find_all("div", class_=re.compile("et_pb_toggle|et_pb_accordion_item"))
            for accordion in accordions:
                title = accordion.find("h5", class_="et_pb_toggle_title")
                section_title = title.get_text(strip=True) if title else "Unknown Section"
                tables = accordion.find_all("table")
                for table in tables:
                    rows = table.find_all("tr")
                    for row in rows:
                        cols = row.find_all("td")
                        if len(cols) >= 4:
                            name = cols[1].get_text(strip=True) if len(cols) > 1 else ""
                            jawatan = cols[2].get_text(strip=True) if len(cols) > 2 else ""
                            phone = cols[3].get_text(strip=True) if len(cols) > 3 else ""
                            email = cols[4].get_text(strip=True).replace("[a]", "@") if len(cols) > 4 else ""
                            if name and re.search(r"[A-Za-z]", name) and not re.match(r"^(Facebook|RSS|Twitter|YouTube|DIREKTORI|KETUA-KETUA|BAHAGIAN)$", name, re.IGNORECASE):
                                table_data.append({
                                    "name": name,
                                    "jawatan": jawatan,
                                    "phone": phone,
                                    "email": email,
                                    "section": section_title
                                })
                                sections.append(f"{section_title}: {name}, {jawatan}, {phone}, {email}")
                    print(f"[Debug] Table rows in {section_title} for {url}: {[row.get_text(strip=True) for row in rows]}")

        # Handle mission/vision page (/188-2/)
        elif "188-2" in url:
            # Target specific containers
            containers = main_content.find_all("div", class_=re.compile("et_pb_text|et_pb_module|et_pb_section"))
            img_texts = []
            expected_headers = ["Visi", "Misi", "Moto", "Dasar Kualiti", "Objektif"]
            header_index = 0
            for container in containers:
                images = container.find_all("img", src=True)
                print(f"[Debug] Found {len(images)} images in container for {url}")
                for img in images:
                    img_url = urljoin(url, img["src"])
                    if img_url.endswith((".png", ".jpg", ".jpeg")) and "logojpkn" not in img_url and "banner" not in img_url:
                        print(f"[Debug] Processing image: {img_url}")
                        img_text = extract_image_text(img_url)
                        if img_text and header_index < len(expected_headers):
                            # Assign header based on expected order
                            header = expected_headers[header_index]
                            content = img_text.strip()
                            if content:
                                structured_data.append({
                                    "header": header,
                                    "content": content
                                })
                                sections.append(f"{header}: {content}")
                                print(f"[Debug] Assigned header {header} with content: {content[:100]}...")
                                header_index += 1
                            img_texts.append(img_text)
            combined_text = " ".join(img_texts)
            print(f"[Debug] Combined OCR text for {url}: {combined_text[:200]}...")
            # Fallback regex splitting if structured_data is incomplete
            if len(structured_data) < len(expected_headers):
                headers = expected_headers
                pattern = r"(?i)\b(" + "|".join(headers) + r")\b\s*[:.]?\s*"
                parts = re.split(pattern, combined_text)
                current_header = None
                for i in range(0, len(parts), 2):
                    part = parts[i].strip()
                    if i + 1 < len(parts):
                        header = parts[i + 1].strip()
                        if header.lower() in [h.lower() for h in headers]:
                            if current_header and part:
                                structured_data.append({
                                    "header": current_header,
                                    "content": part
                                })
                                sections.append(f"{current_header}: {part}")
                                print(f"[Debug] Fallback content for {current_header} at {url}: {part[:100]}...")
                            current_header = header
                        elif current_header and part:
                            structured_data.append({
                                "header": current_header,
                                "content": part
                            })
                            sections.append(f"{current_header}: {part}")
                            print(f"[Debug] Fallback content for {current_header} at {url}: {part[:100]}...")
                if current_header and parts[-1].strip():
                    structured_data.append({
                        "header": current_header,
                        "content": parts[-1].strip()
                    })
                    sections.append(f"{current_header}: {parts[-1].strip()}")
                    print(f"[Debug] Fallback content for {current_header} at {url}: {parts[-1][:100]}...")

        # General text extraction (excluding headers)
        for tag in main_content.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "div", "span"]):
            text = normalize_text(tag.get_text(strip=True))
            if text and not any(h in text for h in ["Visi", "Misi", "Moto", "Dasar Kualiti", "Objektif"]):
                sections.append(text)
        print(f"[Debug] Extracted {len(sections)} text sections for {url}")

        full_text = "\n".join(sections)
        chunks = chunk_text(full_text)

        print(f"[Debug] Final extracted text for {url}: {full_text[:200]}...")

        return {
            "url": url,
            "title": soup.title.string.strip() if soup.title else "",
            "raw_text": full_text,
            "chunks": chunks,
            "table_data": table_data if table_data else None,
            "structured_data": structured_data if structured_data else None
        }
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error scraping {url}: {e}")
        return None
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
            if not href.startswith(BASE_URL):
                continue
            if any(x in href for x in ["uploads/", "#", ".pdf", ".jpg", ".jpeg", ".png", "galeri", "Text_Pic_Single.php", "hello-world", "author/"]):
                continue
            if is_valid_url(href):
                links.add(href)
        print(f"[Debug] Found {len(links)} links for {url}")
        return links
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error getting links from {url}: {e}")
        return set()
    except Exception as e:
        print(f"Error getting links from {url}: {e}")
        return set()

def process_url(args):
    """Process a single URL: scrape, save, and extract links."""
    url, depth, max_depth = args
    if depth > max_depth:
        return []
    try:
        print(f"[Depth {depth}] Scraping: {url}")
        data = scrape_page(url)
        save_text_as_json(url, data)
        sleep(REQUEST_DELAY)
        links = get_links(url)
        return [(link, depth + 1, max_depth) for link in links]
    except Exception as e:
        print(f"Error processing {url}: {e}")
        return []

def parallel_crawl(start_url, max_depth=MAX_DEPTH):
    """Crawl the website starting from start_url."""
    seen = set()
    queue = [(start_url, 0, max_depth)]
    iteration = 0

    with Pool(NUM_WORKERS) as pool:
        while queue:
            queue = [item for item in queue if item[0] not in seen]
            if not queue:
                break
            for item in queue:
                seen.add(item[0])

            print(f"\n🔄 Iteration {iteration}: Processing {len(queue)} URLs...\n")
            results = pool.map(process_url, queue)
            queue = [item for sublist in results for item in sublist if item[0] not in seen]
            iteration += 1

def main():
    parallel_crawl(BASE_URL)

if __name__ == "__main__":
    main()