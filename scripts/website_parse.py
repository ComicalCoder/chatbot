import requests
from bs4 import BeautifulSoup
import os
from urllib.parse import urljoin
from time import sleep

BASE_URL = "https://jpkn.sabah.gov.my/"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
MAX_DEPTH = 3  # Limit crawl depth to avoid excessive scraping
REQUEST_DELAY = 1  # Seconds between requests to be polite

def scrape_page(url):
      """Extract text content from a single page."""
      try:
          headers = {"User-Agent": "Mozilla/5.0"}
          response = requests.get(url, headers=headers, timeout=10)
          response.raise_for_status()
          soup = BeautifulSoup(response.text, "html.parser")
          # Extract text from paragraphs, headings, and other relevant tags
          text = " ".join([elem.get_text(strip=True) for elem in soup.find_all(["p", "h1", "h2", "h3", "h4", "li", "span"])])
          return text
      except requests.RequestException as e:
          print(f"Error scraping {url}: {e}")
          return ""

def save_text(url, text):
      """Save extracted text to a file."""
      if not text:
          return
      filename = url.replace("https://", "").replace("/", "_").replace("?", "_").replace("&", "_") + ".txt"
      filepath = os.path.join(OUTPUT_DIR, filename)
      os.makedirs(OUTPUT_DIR, exist_ok=True)
      with open(filepath, "w", encoding="utf-8") as f:
          f.write(text)

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
      except requests.RequestException as e:
          print(f"Error fetching links from {url}: {e}")
          return set()

def crawl(url, visited, depth=0):
      """Recursively crawl pages up to MAX_DEPTH."""
      if depth > MAX_DEPTH or url in visited:
          return
      print(f"Scraping (depth {depth}): {url}")
      visited.add(url)
      text = scrape_page(url)
      save_text(url, text)
      sleep(REQUEST_DELAY)  # Polite delay
      links = get_links(url)
      for link in links:
          crawl(link, visited, depth + 1)

def main():
      visited = set()
      crawl(BASE_URL, visited)

if __name__ == "__main__":
      main()