import argparse
import csv
import json
import os
import re
import time
import asyncio
from urllib.parse import urljoin, urlparse, urldefrag
 
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter, Retry
from urllib import robotparser
 
 
def make_session():
    s = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD", "OPTIONS"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=10)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.headers.update({
        "User-Agent": "YallakoraCrawler/1.0 (+https://example.com/contact) python-requests"
    })
    return s
 
 
class KooraCrawler:
    def __init__(self, base_url, max_pages=200, delay=1.0, timeout=15):
        if not base_url.startswith("http"):
            base_url = "https://" + base_url
        self.base_url = base_url.rstrip("/")
        self.domain = urlparse(self.base_url).netloc.lower()
        self.max_pages = max_pages
        self.delay = delay
        self.timeout = timeout
 
        self.session = make_session()
        self.rp = robotparser.RobotFileParser()
        self.rp.set_url(urljoin(self.base_url + "/", "robots.txt"))
        try:
            self.rp.read()
        except Exception:
            # If robots.txt fetch fails, proceed cautiously
            pass
 
        os.makedirs("data", exist_ok=True)
        self.visited = set()
        self.queue = [self.base_url]
 
        self.csv_path = "data/yallakora_pages.csv"
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["url", "title", "status", "content_type"])
 
        self.jsonl_path = "data/yallakora_articles.jsonl"
        open(self.jsonl_path, "w", encoding="utf-8").close()
 
    # ---------- HTTP / Robots ----------
    def allowed(self, url: str) -> bool:
        try:
            ua_val = self.session.headers.get("User-Agent", "*")
            ua: str
            if isinstance(ua_val, (bytes, bytearray, memoryview)):
                try:
                    ua = bytes(ua_val).decode("latin-1", errors="ignore") or "*"
                except Exception:
                    ua = "*"
            else:
                ua = (ua_val if isinstance(ua_val, str) else str(ua_val)) or "*"
            return self.rp.can_fetch(ua, url)
        except Exception:
            return True
 
    def fetch(self, url: str) -> requests.Response | None:
        if not self.allowed(url):
            print(f"[robots] Disallowed: {url}")
            return None
        try:
            resp = self.session.get(url, timeout=self.timeout)
            return resp
        except requests.RequestException as e:
            print(f"[error] {e} -> {url}")
            return None
 
    # ---------- URL & Link Handling ----------
    def normalize_link(self, link: str, page_url: str) -> str | None:
        if not link:
            return None
        link = urldefrag(link)[0]  # strip #fragments
        link = urljoin(page_url, link)  # make absolute
 
        parsed = urlparse(link)
        if parsed.scheme not in ("http", "https"):
            return None
        if parsed.netloc.lower() != self.domain:
            return None
 
        # Skip common non-HTML resources
        if re.search(r"\.(jpg|jpeg|png|gif|webp|svg|ico|css|js|pdf|mp4|m3u8|zip|rar)(\?|$)", parsed.path, re.I):
            return None
 
        return link
 
    def extract_links(self, soup: BeautifulSoup, page_url: str) -> list[str]:
        links = set()
        for a in soup.select("a[href]"):
            href_attr = a.get("href")
            if isinstance(href_attr, (list, tuple)):
                if not href_attr:
                    continue
                href = href_attr[0]
            else:
                href = href_attr
            if not isinstance(href, str):
                continue
            norm = self.normalize_link(href, page_url)
            if norm and norm not in self.visited:
                links.add(norm)
        return list(links)
 
    # ---------- Article Heuristics ----------
    def looks_like_article(self, url: str) -> bool:
        """
        Simple heuristic: path contains article-ish words or numeric IDs,
        or deep-ish paths.
        """
        path = urlparse(url).path.lower()
        if any(k in path for k in ["news", "article", "sport", "sports", "details", "story", "content", "match"]):
            return True
        # numeric id in path segment
        if re.search(r"/\d{4,}[-/]?", path):
            return True
        # deep path (e.g., /category/sub/slug)
        if path.count("/") >= 3 and len(path) > 20:
            return True
        return False
 
    def parse_article(self, soup: BeautifulSoup) -> dict:
        # Title candidates
        title = None
        cands = [
            ("h1", "text"),
            ("meta[property='og:title']", "content"),
            ("meta[name='title']", "content"),
            ("title", "text"),
        ]
        for sel, attr in cands:
            el = soup.select_one(sel)
            if el:
                title = el.get_text(strip=True) if attr == "text" else el.get("content")
                if title:
                    break
 
        # Publish time candidates
        published = None
        time_cands = [
            "time[datetime]",
            "meta[property='article:published_time']",
            "meta[name='pubdate']",
            "meta[itemprop='datePublished']",
            "span.time",
            "div.time",
        ]
        for sel in time_cands:
            el = soup.select_one(sel)
            if el:
                published = el.get("datetime") or el.get("content") or el.get_text(strip=True)
                if published:
                    break
 
        # Content candidates
        paragraphs = []
        containers = soup.select("article, .article, .post, .news, .content, #content, .post-content, .entry-content")
        if not containers:
            containers = [soup]
        for container in containers[:3]:
            # prioritize paragraphs and text-heavy divs
            for p in container.find_all(["p", "div"], recursive=True):
                txt = p.get_text(" ", strip=True)
                # skip nav/boilerplate
                if not txt or len(txt) < 40:
                    continue
                if any(bad in txt.lower() for bad in ["javascript", "cookie", "subscribe", "accept", "login"]):
                    continue
                paragraphs.append(txt)
            if paragraphs:
                break
 
        text = "\n".join(paragraphs[:120])  # cap to keep things tidy
        return {"title": title, "published": published, "text": text}
 
    # ---------- Persistence ----------
    def save_page_row(self, url, title, status, content_type):
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([url, title or "", status, content_type or ""])
 
    def save_article(self, url, art: dict):
        if not art.get("title") and not art.get("text"):
            return
        record = {
            "url": url,
            "title": art.get("title"),
            "published": art.get("published"),
            "text": art.get("text"),
        }
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
 
    # ---------- Crawl Loop ----------
    def crawl(self):
        pages_count = 0
        while self.queue and pages_count < self.max_pages:
            url = self.queue.pop(0)
            if url in self.visited:
                continue
            self.visited.add(url)
 
            print(f"[{pages_count+1}/{self.max_pages}] GET {url}")
            resp = self.fetch(url)
            time.sleep(self.delay)
 
            if resp is None:
                continue
 
            ctype = (resp.headers.get("Content-Type") or "").lower()
            status = resp.status_code
 
            title = None
            links = []
            if status == 200 and "text/html" in ctype:
                soup = BeautifulSoup(resp.text, "lxml")
                # title for CSV
                t_el = soup.select_one("title")
                title = t_el.get_text(strip=True) if t_el else None
 
                # queue internal links
                links = self.extract_links(soup, url)
                self.queue.extend([l for l in links if l not in self.visited])
 
                # If looks like an article, try to parse and save it
                if self.looks_like_article(url):
                    art = self.parse_article(soup)
                    self.save_article(url, art)
 
            self.save_page_row(url, title, status, ctype)
            pages_count += 1
 
        print(f"Done. Crawled {pages_count} pages.")
        print(f"- Pages CSV: {self.csv_path}")
        print(f"- Articles JSONL: {self.jsonl_path}")
 
    # ---------- Crawl with Depth (for RAG) ----------
    async def crawl_with_depth(self, start_url: str | None = None, max_depth: int = 1, max_pages: int | None = None) -> list[dict]:
        """
        BFS crawl up to max_depth and return a list of document dicts:
        { "url": str, "title": str, "published": str|None, "text": str }
        Intended for use by arabic_chatbot_final.rag.index_documents.
        """
        start = (start_url or self.base_url).rstrip("/")
        visited = set()
        queue: list[tuple[str, int]] = [(start, 0)]
        results: list[dict] = []
        pages_count = 0

        while queue:
            url, depth = queue.pop(0)
            if url in visited or depth > max_depth:
                continue
            visited.add(url)

            resp = self.fetch(url)
            # polite delay even on errors
            try:
                await asyncio.sleep(self.delay)
            except Exception:
                pass

            if resp is None:
                continue

            ctype = (resp.headers.get("Content-Type") or "").lower()
            status = resp.status_code
            if status != 200 or "text/html" not in ctype:
                continue

            soup = BeautifulSoup(resp.text, "lxml")
            # base title
            t_el = soup.select_one("title")
            base_title = t_el.get_text(strip=True) if t_el else None

            # always parse content (works for article and generic pages)
            art = self.parse_article(soup)
            title = art.get("title") or base_title
            text = (art.get("text") or "").strip()

            # fallback extraction if parse_article found nothing
            if not text:
                for s in soup(["script", "style", "noscript", "nav", "footer", "header", "aside"]):
                    s.decompose()
                text = " ".join(soup.get_text(separator=" ").split())

            if text and len(text) > 120:
                results.append({
                    "url": url,
                    "title": title or "",
                    "published": art.get("published"),
                    "text": text
                })
                pages_count += 1

            # stop early if page limit reached
            if max_pages is not None and pages_count >= max_pages:
                break

            # enqueue children if depth allows
            if depth < max_depth:
                links = self.extract_links(soup, url)
                for lnk in links:
                    if lnk not in visited:
                        queue.append((lnk, depth + 1))

        return results
 
 
def main():
    parser = argparse.ArgumentParser(description="Polite crawler for Yallakora Match Center.")
    parser.add_argument(
        "--base",
        default="https://www.yallakora.com/match-center",
        help="Base URL (default: https://www.yallakora.com/match-center)"
    )
    parser.add_argument("--max-pages", type=int, default=200, help="Max pages to crawl (default: 200)")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay (seconds) between requests (default: 1.0)")
    parser.add_argument("--timeout", type=int, default=15, help="HTTP timeout seconds (default: 15)")
    args = parser.parse_args()
 
    crawler = KooraCrawler(args.base, max_pages=args.max_pages, delay=args.delay, timeout=args.timeout)
    crawler.crawl()
 
 
if __name__ == "__main__":
    main()