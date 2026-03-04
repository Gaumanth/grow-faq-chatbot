"""
Scraper for Mutual Fund FAQ Chatbot
Reads URLs from sources.csv, scrapes each page, and saves clean text to scraped_data/
"""

import csv
import os
import re
import time
import hashlib
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

SOURCES_FILE = "sources.csv"
OUTPUT_DIR = "scraped_data"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

TAGS_TO_REMOVE = [
    "script", "style", "nav", "footer", "header", "aside",
    "form", "button", "iframe", "noscript", "svg", "img",
    "video", "audio", "canvas", "map",
]


def clean_text(raw_html: str) -> str:
    """Extract and clean readable text from raw HTML."""
    soup = BeautifulSoup(raw_html, "html.parser")

    for tag in soup.find_all(TAGS_TO_REMOVE):
        tag.decompose()

    for element in soup.find_all(attrs={"class": re.compile(r"(cookie|popup|modal|banner|ad[s]?|sidebar)", re.I)}):
        element.decompose()

    text = soup.get_text(separator="\n")

    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and len(stripped) > 2:
            lines.append(stripped)
    text = "\n".join(lines)

    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def make_filename(url: str, title: str) -> str:
    """Create a safe filename from URL and title."""
    domain = urlparse(url).netloc.replace("www.", "").replace(".", "_")
    slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")[:60]
    short_hash = hashlib.md5(url.encode()).hexdigest()[:6]
    return f"{domain}_{slug}_{short_hash}.txt"


def scrape_url(url: str, timeout: int = 15) -> str | None:
    """Scrape a single URL and return cleaned text, or None on failure."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=timeout)
        response.raise_for_status()
        return clean_text(response.text)
    except requests.RequestException as e:
        print(f"  [ERROR] Failed to scrape {url}: {e}")
        return None


def load_sources(csv_path: str) -> list[dict]:
    """Read sources.csv and return list of {url, title, category} dicts."""
    sources = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sources.append({
                "url": row["url"].strip(),
                "title": row["title"].strip(),
                "category": row["category"].strip(),
            })
    return sources


def run_scraper():
    """Main scraping pipeline: read sources.csv -> scrape -> save text files."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    sources = load_sources(SOURCES_FILE)
    print(f"Loaded {len(sources)} URLs from {SOURCES_FILE}\n")

    results = {"success": 0, "failed": 0, "skipped": 0}

    for i, source in enumerate(sources, 1):
        url = source["url"]
        title = source["title"]
        filename = make_filename(url, title)
        filepath = os.path.join(OUTPUT_DIR, filename)

        if os.path.exists(filepath):
            print(f"[{i}/{len(sources)}] SKIP (already scraped): {title}")
            results["skipped"] += 1
            continue

        print(f"[{i}/{len(sources)}] Scraping: {title}")
        print(f"  URL: {url}")

        content = scrape_url(url)

        if content and len(content) > 50:
            header = f"Source: {url}\nTitle: {title}\nCategory: {source['category']}\n\n"
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(header + content)
            print(f"  Saved: {filename} ({len(content)} chars)")
            results["success"] += 1
        elif content:
            print(f"  [WARN] Content too short ({len(content)} chars), may be JS-rendered.")
            header = f"Source: {url}\nTitle: {title}\nCategory: {source['category']}\n\n"
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(header + content)
            results["success"] += 1
        else:
            results["failed"] += 1

        time.sleep(1.5)

    print(f"\n{'='*50}")
    print(f"Scraping complete!")
    print(f"  Success: {results['success']}")
    print(f"  Skipped: {results['skipped']}")
    print(f"  Failed:  {results['failed']}")
    print(f"  Output:  {os.path.abspath(OUTPUT_DIR)}/")

    if results["failed"] > 0:
        print(f"\nTip: For failed/JS-rendered pages, manually create a .txt file")
        print(f"in '{OUTPUT_DIR}/' with the format:")
        print(f"  Source: <url>")
        print(f"  Title: <page title>")
        print(f"  Category: <category>")
        print(f"  <paste content here>")


if __name__ == "__main__":
    run_scraper()
