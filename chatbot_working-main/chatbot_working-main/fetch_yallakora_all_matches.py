import argparse
import json
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup

BASE = "https://www.yallakora.com"
MC_URL = f"{BASE}/match-center"

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"

def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def daterange(start: datetime, end: datetime):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)

def get_soup(session: requests.Session, url: str) -> BeautifulSoup | None:
    try:
        r = session.get(url, timeout=20)
        r.raise_for_status()
        return BeautifulSoup(r.text, "lxml")
    except Exception:
        return None

def discover_matches_for_date(session: requests.Session, date_str: str) -> set[str]:
    # Many sites support ?date=YYYY-MM-DD on match-center. Try that first; fall back to root.
    urls = set()
    for u in (f"{MC_URL}?date={date_str}", MC_URL):
        soup = get_soup(session, u)
        if not soup:
            continue
        for a in soup.find_all("a", href=True):
            import bs4
            if isinstance(a, bs4.element.Tag):
                href = a.get("href")
                if href and "/match/" in str(href):
                    full = urljoin(BASE, str(href))
                    urls.add(full.split("#", 1)[0])
        # If we found some, good enough for this date
        if urls:
            break
    return urls

def extract_page(session: requests.Session, url: str) -> dict | None:
    soup = get_soup(session, url)
    if not soup:
        return None
    # Try to get a clean title (match pages usually have "مباراة X و Y")
    title = ""
    h1 = soup.find("h1")
    if h1 and normalize_ws(h1.get_text()):
        title = normalize_ws(h1.get_text())
    if not title:
        t = soup.find("title")
        if t:
            title = normalize_ws(t.get_text())
    # Pull all visible text (good enough for regex-based time extraction later)
    text = normalize_ws(soup.get_text(separator=" "))
    # A quick attempt to extract time like "18:00"
    m = re.search(r"\b(\d{1,2}:\d{2})\b", text)
    published = m.group(1) if m else None
    return {
        "url": url,
        "title": title,
        "published": published,
        "text": text,
        "fetched_at": datetime.utcnow().isoformat() + "Z",
    }

def load_seen(path: Path) -> set[str]:
    seen = set()
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                u = line.strip()
                if u:
                    seen.add(u)
    return seen

def append_seen(path: Path, url: str):
    with path.open("a", encoding="utf-8") as f:
        f.write(url + "\n")

def main():
    ap = argparse.ArgumentParser(description="Crawl Yallakora match pages across a date range.")
    ap.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    ap.add_argument("--out", default=str(Path("data") / "yallakora_articles.jsonl"), help="Output JSONL path")
    ap.add_argument("--seen", default=str(Path("data") / "seen_urls.txt"), help="Path to store seen match URLs")
    ap.add_argument("--delay", type=float, default=0.8, help="Delay between requests (seconds)")
    ap.add_argument("--max-per-day", type=int, default=500, help="Safety cap for match links per day")
    args = ap.parse_args()

    out_path = Path(args.out)
    seen_path = Path(args.seen)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen = load_seen(seen_path)

    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")

    session = requests.Session()
    session.headers.update({"User-Agent": UA})

    total_new = 0
    with out_path.open("a", encoding="utf-8") as fout:
        for d in daterange(start, end):
            date_str = d.strftime("%Y-%m-%d")
            print(f"[{date_str}] discovering...")
            links = list(discover_matches_for_date(session, date_str))[: args.max_per_day]
            print(f"[{date_str}] found {len(links)} match links")
            for i, url in enumerate(links, 1):
                if url in seen:
                    continue
                rec = extract_page(session, url)
                if rec:
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    total_new += 1
                seen.add(url)
                append_seen(seen_path, url)
                if args.delay:
                    time.sleep(args.delay)
            # be polite between days
            if args.delay:
                time.sleep(min(args.delay * 2, 2.0))

    print(f"Done. New pages added: {total_new}. Output: {out_path}")

if __name__ == "__main__":
    main()