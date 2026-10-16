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
from html import escape
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
        if not re.fullmatch(rf"{PROFILE_ID}:[A-Za-z0-9_-]+", identifier) or not row["title"].strip() or not row["count_cell"]:
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


class DetailParser(HTMLParser):
    """Read full author lists and bibliographic fields without requesting exports."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.fields = {}
        self.label = ""
        self.capture = None
        self.depth = 0
        self.text = ""
        self.source = ""

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = attrs.get("class", "").split()
        if "gsc_oci_title_link" in classes:
            self.source = attrs.get("href", "")
        if self.capture and tag not in {"br", "img", "input", "meta", "link", "hr"}:
            self.depth += 1
        if "gsc_oci_field" in classes or "gsc_oci_value" in classes:
            self.capture = "label" if "gsc_oci_field" in classes else "value"
            self.depth = 1
            self.text = ""

    def handle_data(self, text):
        if self.capture:
            self.text += text

    def handle_endtag(self, tag):
        if not self.capture:
            return
        self.depth -= 1
        if self.depth == 0:
            if self.capture == "label":
                self.label = self.text.strip()
            else:
                self.fields[self.label] = " ".join(self.text.split())
            self.capture = None


def fetch_html(url):
    request = Request(url, headers={"User-Agent": "TianyuXieHomepage/1.0"})
    with urlopen(request, timeout=30) as response:
        return response.read().decode(response.headers.get_content_charset() or "utf-8")


def bibtex_text(text):
    replacements = {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}",
                    "&": r"\&", "%": r"\%", "#": r"\#", "_": r"\_", "$": r"\$"}
    return "".join(replacements.get(char, char) for char in text)


def imported_paper(record, html):
    parser = DetailParser()
    parser.feed(html)
    fields = parser.fields
    authors = fields.get("Authors", "")
    date = fields.get("Publication date", "")
    if not authors or "…" in authors or "..." in authors or not re.match(r"^\d{4}(?:/|$)", date):
        raise ValueError(f"Incomplete metadata for new paper {record['id']}; previous files retained.")
    year = date[:4]
    identifier = "scholar-" + record["id"].split(":", 1)[1]
    source = parser.source
    if urlparse(source).scheme not in {"http", "https"}:
        source = citation_url(record)
    venue = fields.get("Conference") or fields.get("Journal") or fields.get("Book") or ""
    # A Scholar venue string alone does not establish conference acceptance.
    values = [("title", record["title"]), ("author", authors.replace(", ", " and ")),
              ("year", year), ("url", source)]
    if venue:
        values.append(("howpublished", venue))
    bibtex = "@misc{" + identifier + ",\n" + ",\n".join(
        f"  {key} = {{{bibtex_text(value)}}}" for key, value in values) + "\n}"
    return {"id": identifier, "title": record["title"], "scholarId": record["id"],
            "sourceUrl": source, "bibtex": bibtex, "imported": True,
            "authors": authors, "year": year, "venue": venue}


def render_imports(papers, archive):
    start, end = "<!-- scholar-imports:start -->", "<!-- scholar-imports:end -->"
    if archive.count(start) != 1 or archive.count(end) != 1:
        raise ValueError("Missing or duplicate Scholar import markers; archive was not changed.")
    articles = []
    for paper in sorted((p for p in papers if p.get("imported")), key=lambda p: p["year"], reverse=True):
        identifier, title, url, authors, year, venue = (
            escape(str(paper[key]), quote=True) for key in ("id", "title", "sourceUrl", "authors", "year", "venue"))
        filters = "preprints" if "arxiv" in paper["venue"].casefold() else ""
        articles.append(f'''<article class="paper" data-paper-filters="{filters}">
          <div class="paper-symbol"><svg class="ui-icon" aria-hidden="true"><use href="/images/icons.svg#paper"></use></svg></div>
          <div><div class="paper-status"><span class="venue gray">{year}</span></div>
            <h3><a href="{url}">{title}</a> <a class="citation-count" data-citation-count="{identifier}" href="{url}" hidden></a></h3>
            <div class="meta">{authors}</div><p>{venue}</p>
            <div class="actions"><a class="action" href="{url}">Paper</a><button class="cite-button" type="button" data-cite="{identifier}">Cite</button></div>
          </div>
        </article>''')
    before, remainder = archive.split(start)
    _, after = remainder.split(end)
    result = before + start + "\n" + "\n\n".join(articles) + "\n        " + end + after
    return re.sub(r'(<p data-publication-count aria-live="polite">)\d+ entries',
                  lambda match: match[1] + f"{len(papers)} entries", result)


def update_cache(html, metadata_path, cache_path, archive_path=None, fetch_detail=None):
    metadata = json.loads(metadata_path.read_text())
    papers = metadata["papers"]
    records = parse_profile(html)
    if len(records) >= 100:
        raise ValueError("Scholar profile reached the 100-paper page limit; pagination required before updating.")
    counts = matched_counts(papers, records)
    if not counts:
        raise ValueError("No matching papers; old cache retained.")
    # Assign matched IDs before discovery so newly indexed existing entries are not duplicated.
    used = set()
    for paper in papers:
        if paper["id"] not in counts:
            continue
        record = records.get(paper.get("scholarId")) or next(
            r for r in records.values() if normalized_title(r["title"]) == normalized_title(paper["title"]))
        paper["scholarId"] = record["id"]
        used.add(record["id"])
    new_records = [record for identifier, record in records.items() if identifier not in used]
    if new_records and archive_path is not None:
        if fetch_detail is None:
            raise ValueError("New papers require full Scholar detail metadata.")
        for record in new_records:
            detail_url = f"https://scholar.google.com/citations?view_op=view_citation&hl=en&user={PROFILE_ID}&citation_for_view={record['id']}"
            papers.append(imported_paper(record, fetch_detail(detail_url)))
        counts = matched_counts(papers, records)
    snapshot = {"updatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "profileUrl": PROFILE_URL, "papers": counts}
    outputs = {cache_path: json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n"}
    if archive_path is not None:
        outputs[metadata_path] = json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
        outputs[archive_path] = render_imports(papers, archive_path.read_text())
    # Validate and serialize all records before replacing any tracked file.
    for path, content in outputs.items():
        temporary = path.with_suffix(".tmp")
        temporary.write_text(content)
    for path in outputs:
        path.with_suffix(".tmp").replace(path)
    print(f"Updated {len(counts)} matched papers; discovered {len(new_records)} new Scholar records.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html", type=Path, help="Parse a saved public profile for offline verification.")
    args = parser.parse_args()
    if args.html:
        html = args.html.read_text()
    else:
        html = fetch_html(PROFILE_URL)
    update_cache(html, ROOT / "data/publications.json", ROOT / "data/citations.json",
                 ROOT / "publications/index.html", fetch_html)


if __name__ == "__main__":
    main()
