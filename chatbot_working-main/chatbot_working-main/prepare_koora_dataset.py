import argparse
import json
import re
import sys
from pathlib import Path
from hashlib import sha1

# Reuse the project's chunking to stay consistent with RAG
# arabic_chatbot_final/rag.py -> chunk_text
try:
    from arabic_chatbot_final.rag import chunk_text  # type: ignore
except Exception:
    # Fallback if import context differs
    def chunk_text(text: str, max_chars: int = 800):
        words = text.split()
        chunks = []
        cur = []
        cur_len = 0
        for w in words:
            cur.append(w)
            cur_len += len(w) + 1
            if cur_len > max_chars:
                chunks.append(" ".join(cur))
                cur = []
                cur_len = 0
        if cur:
            chunks.append(" ".join(cur))
        return chunks


def normalize_ws(text: str) -> str:
    # Collapse whitespace and trim
    return re.sub(r"\s+", " ", text or "").strip()


def is_boilerplate(text: str, bad_repeat_threshold: int) -> bool:
    # Heuristic: heavy navigation/schedule phrases repeated many times -> skip
    triggers = ["مباريات الأمس", "مباريات اليوم", "مباريات الغد"]
    repeats = sum(text.count(t) for t in triggers)
    return repeats >= bad_repeat_threshold


def process(input_path: Path, output_path: Path, chunk_size: int, min_chars: int,
            bad_repeat_threshold: int, dedupe: bool) -> dict:
    kept_articles = 0
    skipped_short = 0
    skipped_boiler = 0
    written_chunks = 0

    # Deduplicate exact chunk texts using hash
    seen_hashes = set()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            url = rec.get("url")
            title = normalize_ws(rec.get("title") or "")
            published = rec.get("published")
            text = normalize_ws(rec.get("text") or "")
            if not text or len(text) < min_chars:
                skipped_short += 1
                continue
            if is_boilerplate(text, bad_repeat_threshold):
                skipped_boiler += 1
                continue

            kept_articles += 1
            chunks = chunk_text(text, max_chars=chunk_size)
            for idx, ch in enumerate(chunks):
                ch = normalize_ws(ch)
                if not ch:
                    continue
                if dedupe:
                    h = sha1(ch.encode("utf-8")).hexdigest()
                    if h in seen_hashes:
                        continue
                    seen_hashes.add(h)
                out = {
                    "url": url,
                    "title": title,
                    "published": published,
                    "chunk_index": idx,
                    "text": ch,
                }
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                written_chunks += 1

    return {
        "kept_articles": kept_articles,
        "skipped_short": skipped_short,
        "skipped_boiler": skipped_boiler,
        "written_chunks": written_chunks,
        "output": str(output_path),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Filter and chunk Yallakora articles JSONL.")
    parser.add_argument("--in", dest="inp", default=str(Path("data") / "yallakora_articles.jsonl"),
                        help="Input JSONL path (one article per line).")
    parser.add_argument("--out", dest="out", default=str(Path("data") / "yallakora_articles_chunked.jsonl"),
                        help="Output JSONL path for chunked records.")
    parser.add_argument("--chunk-size", type=int, default=800, help="Max characters per chunk.")
    parser.add_argument("--min-chars", type=int, default=200, help="Minimum characters to keep an article.")
    parser.add_argument("--bad-repeat-threshold", type=int, default=6,
                        help="Skip if boilerplate triggers appear at least this many times.")
    parser.add_argument("--dedupe", action="store_true", help="Deduplicate identical chunks across dataset.")
    args = parser.parse_args(argv)

    stats = process(
        input_path=Path(args.inp),
        output_path=Path(args.out),
        chunk_size=args.chunk_size,
        min_chars=args.min_chars,
        bad_repeat_threshold=args.bad_repeat_threshold,
        dedupe=bool(args.dedupe),
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

