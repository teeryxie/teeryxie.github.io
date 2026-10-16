import json
import tempfile
import unittest
from pathlib import Path

from update_scholar import PROFILE_ID, matched_counts, parse_profile, update_cache


def row(identifier="one", title="Test: Omni &amp; Audio", count="7", count_cell=True):
    count_class = "gsc_a_c" if count_cell else "changed_markup"
    return f'''<tr class="gsc_a_tr"><td><a class="gsc_a_at"
      href="/citations?citation_for_view={PROFILE_ID}:{identifier}">{title}</a></td>
      <td class="{count_class}"><a class="gsc_a_ac" href="/scholar?cites=123">{count}</a></td></tr>'''


class ScholarTests(unittest.TestCase):
    def test_entities_and_explicit_zero(self):
        records = parse_profile(row(count=""))
        self.assertEqual(records[f"{PROFILE_ID}:one"]["title"], "Test: Omni & Audio")
        self.assertEqual(records[f"{PROFILE_ID}:one"]["count"], 0)

    def test_missing_count_markup_is_not_zero(self):
        with self.assertRaises(ValueError):
            parse_profile(row(count_cell=False))

    def test_ids_cannot_override_conflicting_titles(self):
        papers = [{"id": "test", "title": "Different work", "scholarId": f"{PROFILE_ID}:one"}]
        with self.assertRaises(ValueError):
            matched_counts(papers, parse_profile(row()))

    def test_unique_unicode_title_fallback_and_unindexed_paper(self):
        papers = [{"id": "test", "title": "A²RBench: Audio", "scholarId": f"{PROFILE_ID}:old"},
                  {"id": "unindexed", "title": "Another work", "scholarId": None}]
        counts = matched_counts(papers, parse_profile(row(title="A2RBench: Audio", count="1,234")))
        self.assertEqual(counts["test"]["count"], 1234)
        self.assertEqual(counts["test"]["url"], "https://scholar.google.com/scholar?cites=123")
        self.assertNotIn("unindexed", counts)

    def test_blocked_or_partial_response_preserves_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            metadata, cache = Path(folder) / "papers.json", Path(folder) / "cache.json"
            metadata.write_text(json.dumps({"papers": [
                {"id": "test", "title": "Test: Omni & Audio", "scholarId": f"{PROFILE_ID}:one"},
                {"id": "missing", "title": "Missing record", "scholarId": f"{PROFILE_ID}:missing"}]}))
            cache.write_text("previous valid snapshot")
            for html in ["<h1>Verify you are human</h1>", row(), row(count="invalid")]:
                with self.assertRaises(ValueError):
                    update_cache(html, metadata, cache)
                self.assertEqual(cache.read_text(), "previous valid snapshot")

    def test_duplicate_rows_are_rejected(self):
        with self.assertRaises(ValueError):
            parse_profile(row() + row())

class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        root = Path(self.folder.name)
        self.metadata, self.cache, self.archive = (root / name for name in ("papers.json", "counts.json", "index.html"))
        self.metadata.write_text(json.dumps({"papers": [{"id": "old", "title": "Existing paper", "scholarId": f"{PROFILE_ID}:old"}]}))
        self.cache.write_text("old counts")
        self.archive.write_text('<p data-publication-count aria-live="polite">1 entries</p><!-- scholar-imports:start --><!-- scholar-imports:end -->')
        self.profile = row("old", "Existing paper") + row("new", "New &amp; useful paper")
        self.detail = '''<a class="gsc_oci_title_link" href="https://arxiv.org/abs/1234.56789">New paper</a>
        <div class="gsc_oci_field">Authors</div><div class="gsc_oci_value">Tianyu Xie, Someone Else</div>
        <div class="gsc_oci_field">Publication date</div><div class="gsc_oci_value">2027/1/2</div>
        <div class="gsc_oci_field">Journal</div><div class="gsc_oci_value"><span>arXiv preprint</span> &amp; notes</div>'''

    def test_discovery_renders_safe_html_and_citable_metadata_once(self):
        detail_requests = []

        def fetch(url):
            detail_requests.append(url)
            return self.detail

        update_cache(self.profile, self.metadata, self.cache, self.archive, fetch)
        papers = json.loads(self.metadata.read_text())["papers"]
        self.assertEqual(len(papers), 2)
        self.assertIn("Tianyu Xie and Someone Else", papers[1]["bibtex"])
        self.assertIn("@misc{", papers[1]["bibtex"])
        self.assertIn("year = {2027}", papers[1]["bibtex"])
        archive = self.archive.read_text()
        self.assertIn("New &amp; useful paper", archive)
        self.assertIn('data-cite="scholar-new"', archive)
        self.assertIn('data-citation-count="scholar-new"', archive)
        self.assertIn("2 entries", archive)
        self.assertNotIn("Accepted", archive)
        update_cache(self.profile, self.metadata, self.cache, self.archive, fetch)
        self.assertEqual(len(detail_requests), 1)
        self.assertEqual(self.archive.read_text(), archive)

    def test_newly_indexed_existing_manuscript_is_not_duplicated(self):
        metadata = json.loads(self.metadata.read_text())
        metadata["papers"].append({"id": "manuscript", "title": "New & useful paper", "scholarId": None})
        self.metadata.write_text(json.dumps(metadata))
        update_cache(self.profile, self.metadata, self.cache, self.archive,
                     lambda _: self.fail("Existing manuscript should not request new metadata"))
        papers = json.loads(self.metadata.read_text())["papers"]
        self.assertEqual(len(papers), 2)
        self.assertEqual(papers[1]["scholarId"], f"{PROFILE_ID}:new")

    def test_blocked_new_details_preserve_all_files(self):
        before = [path.read_text() for path in (self.metadata, self.cache, self.archive)]
        for detail in ("<h1>Access denied</h1>", self.detail.replace("Tianyu Xie, Someone Else", "Tianyu Xie, ...")):
            with self.assertRaises(ValueError):
                update_cache(self.profile, self.metadata, self.cache, self.archive, lambda _, detail=detail: detail)
            self.assertEqual([path.read_text() for path in (self.metadata, self.cache, self.archive)], before)


if __name__ == "__main__":
    unittest.main()
