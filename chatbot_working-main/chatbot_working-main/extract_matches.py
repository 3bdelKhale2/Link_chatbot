import argparse
import csv
import json
import re
from pathlib import Path
from datetime import datetime

AR_MONTHS = {
    "يناير": 1, "فبراير": 2, "مارس": 3,
    "أبريل": 4, "ابريل": 4,
    "مايو": 5, "يونيو": 6, "يوليو": 7,
    "أغسطس": 8, "اغسطس": 8,
    "سبتمبر": 9, "أكتوبر": 10, "اكتوبر": 10,
    "نوفمبر": 11, "ديسمبر": 12,
}

def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def parse_ar_date(text: str) -> str | None:
    # e.g. "14 أغسطس 2025"
    text = normalize_ws(text)
    m = re.search(r"(\d{1,2})\s+(يناير|فبراير|مارس|أبريل|ابريل|مايو|يونيو|يوليو|أغسطس|اغسطس|سبتمبر|أكتوبر|اكتوبر|نوفمبر|ديسمبر)\s+(\d{4})", text)
    if not m:
        return None
    day = int(m.group(1))
    month = AR_MONTHS[m.group(2)]
    year = int(m.group(3))
    try:
        return f"{year:04d}-{month:02d}-{day:02d}"
    except Exception:
        return None

def extract_competition_from_url(url: str) -> str:
    # https://www.yallakora.com/<slug>/... -> take slug
    try:
        slug = url.split("://", 1)[1].split("/", 1)[1].split("/", 1)[0]
    except Exception:
        return ""
    mapping = {
        "egyptian-league": "الدوري المصري",
        "qatar-league": "دوري نجوم قطر",
        "fiba-afrobasket": "بطولة الأفرو باسكت",
        "handball-": "كرة اليد",
        "uael-league": "الدوري الإماراتي",
    }
    return mapping.get(slug, slug.replace("-", " "))

def parse_teams_from_title(title: str) -> tuple[str, str] | None:
    # Match "مباراة X و Y"
    t = normalize_ws(title)
    m = re.match(r"^\s*مباراة\s+(.+?)\s+و\s+(.+)$", t)
    if not m:
        return None
    def clean_team(x: str) -> str:
        x = normalize_ws(x)
        # remove extra trailing "التفاصيل" or similar fragments if any
        x = re.sub(r"\s*التفاصيل\s*$", "", x)
        return x
    return clean_team(m.group(1)), clean_team(m.group(2))

def extract_time(rec: dict) -> str | None:
    pub = normalize_ws(rec.get("published") or "")
    if re.fullmatch(r"\d{1,2}:\d{2}", pub):
        return pub
    text = rec.get("text") or ""
    mt = re.search(r"(\d{1,2}:\d{2})", text)
    return mt.group(1) if mt else None

def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue

def extract_matches(paths: list[Path]) -> list[dict]:
    seen_urls = set()
    rows: list[dict] = []
    for p in paths:
        if not p.exists():
            continue
        for rec in load_jsonl(p):
            url = rec.get("url") or ""
            if not url or url in seen_urls:
                continue
            title = normalize_ws(rec.get("title") or "")
            if not title.startswith("مباراة"):
                # skip non-match pages
                continue
            teams = parse_teams_from_title(title)
            if not teams:
                continue
            time = extract_time(rec)
            if not time:
                continue
            date_iso = parse_ar_date(rec.get("text") or "")
            comp = extract_competition_from_url(url)
            home, away = teams
            rows.append({
                "date": date_iso or "",
                "time": time,
                "home": home,
                "away": away,
                "competition": comp,
                "title": title,
                "url": url,
            })
            seen_urls.add(url)
    # sort by date+time if available, else by title
    def sort_key(r):
        try:
            dt = datetime.strptime((r["date"] or "9999-12-31") + " " + r["time"], "%Y-%m-%d %H:%M")
        except Exception:
            dt = datetime(9999, 12, 31, 23, 59)
        return (dt, r["title"])
    rows.sort(key=sort_key)
    return rows

def write_csv(rows: list[dict], out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["date", "time", "home", "away", "competition", "title", "url"])
        w.writeheader()
        w.writerows(rows)

def write_jsonl(rows: list[dict], out_jsonl: Path):
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser(description="Extract clean matches (title, time, teams, competition, date) from Yallakora dataset.")
    ap.add_argument("--in", dest="inputs", nargs="*", default=[
        str(Path("data") / "yallakora_articles.jsonl"),
        str(Path("data") / "yallakora_articles_chunked.jsonl"),
    ], help="Input JSONL files.")
    ap.add_argument("--out-csv", default=str(Path("data") / "matches_clean.csv"), help="Output CSV path.")
    ap.add_argument("--out-jsonl", default=str(Path("data") / "matches_clean.jsonl"), help="Output JSONL path.")
    args = ap.parse_args()

    input_paths = [Path(p) for p in args.inputs]
    rows = extract_matches(input_paths)
    write_csv(rows, Path(args.out_csv))
    write_jsonl(rows, Path(args.out_jsonl))
    print(f"Wrote {len(rows)} matches to {args.out_csv} and {args.out_jsonl}")

if __name__ == "__main__":
    main()