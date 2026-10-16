# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Cache public Google Scholar counts; failed or incomplete fetches never overwrite them."""

import argparse
import json
import re
import unicodedata
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qs, urljoin, urlparse
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
PROFILE_ID = "_vZWAnkAAAAJ"
PROFILE_URL = f"https://scholar.google.com/citations?user={PROFILE_ID}&hl=en&pagesize=100"


class ScholarParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows = []
        self.row = None
        self.field = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = attrs.get("class", "").split()
        if tag == "tr" and "gsc_a_tr" in classes:
            self.row = {"title": "", "count": "", "id": "", "url": "", "count_cell": False}
        if self.row is None:
            return
        if tag == "td" and "gsc_a_c" in classes:
            self.row["count_cell"] = True
            self.field = "count"
        if tag == "a" and "gsc_a_at" in classes:
            query = parse_qs(urlparse(attrs.get("href", "")).query)
            self.row["id"] = query.get("citation_for_view", [""])[0]
            self.field = "title"
        if tag == "a" and "gsc_a_ac" in classes:
            self.row["url"] = attrs.get("href", "")

    def handle_data(self, text):
        if self.row is not None and self.field:
            self.row[self.field] += text

    def handle_endtag(self, tag):
        if tag in {"a", "td"}:
            self.field = None
        if tag == "tr" and self.row is not None:
            self.rows.append(self.row)
            self.row = None


def normalized_title(title):
    return "".join(c for c in unicodedata.normalize("NFKC", title).casefold() if c.isalnum())


def parse_profile(html):
    parser = ScholarParser()
    parser.feed(html)
    if not parser.rows:
        raise ValueError("No Scholar records found; cache was not changed (possible access block).")
    records = {}
    for row in parser.rows:
        identifier = row["id"]
        count = row["count"].strip().replace(",", "")
        if not identifier.startswith(f"{PROFILE_ID}:") or not row["title"].strip() or not row["count_cell"]:
            raise ValueError("Unexpected Scholar row structure; refusing a partial update.")
        if count and not re.fullmatch(r"\d+\*?", count):
            raise ValueError(f"Unexpected citation count for {identifier}: {count!r}")
        if identifier in records:
            raise ValueError(f"Duplicate Scholar record: {identifier}")
        records[identifier] = {**row, "count": int(count.rstrip("*") or "0")}
    return records


def citation_url(record):
    candidate = urljoin(PROFILE_URL, record["url"])
    parsed = urlparse(candidate)
    cites = parse_qs(parsed.query).get("cites", [""])[0]
    if parsed.hostname == "scholar.google.com" and cites.isdigit():
        return f"https://scholar.google.com/scholar?cites={cites}"
    return f"https://scholar.google.com/citations?view_op=view_citation&user={PROFILE_ID}&citation_for_view={record['id']}"


def matched_counts(papers, records):
    counts = {}
    used = set()
    for paper in papers:
        identifier = paper.get("scholarId")
        candidates = [r for r in records.values() if normalized_title(r["title"]) == normalized_title(paper["title"])]
        record = records.get(identifier)
        if record is None and len(candidates) == 1:
            record = candidates[0]
        if record is None:
            if identifier:
                raise ValueError(f"Tracked paper missing: {paper['id']}; old cache retained.")
            continue
        if record["id"] in used or normalized_title(record["title"]) != normalized_title(paper["title"]):
            raise ValueError(f"Ambiguous or conflicting Scholar match: {paper['id']}")
        used.add(record["id"])
        counts[paper["id"]] = {"count": record["count"], "url": citation_url(record)}
    return counts


def update_cache(html, metadata_path, cache_path):
    papers = json.loads(metadata_path.read_text())["papers"]
    counts = matched_counts(papers, parse_profile(html))
    if not counts:
        raise ValueError("No matching papers; old cache retained.")
    snapshot = {
        "updatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "profileUrl": PROFILE_URL,
        "papers": counts,
    }
    temporary = cache_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(cache_path)
    print(f"Updated {len(counts)} matched papers from Google Scholar.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html", type=Path, help="Parse a saved public profile for offline verification.")
    args = parser.parse_args()
    if args.html:
        html = args.html.read_text()
    else:
        request = Request(PROFILE_URL, headers={"User-Agent": "TianyuXieHomepage/1.0"})
        with urlopen(request, timeout=30) as response:
            html = response.read().decode(response.headers.get_content_charset() or "utf-8")
    update_cache(html, ROOT / "data/publications.json", ROOT / "data/citations.json")


if __name__ == "__main__":
    main()
