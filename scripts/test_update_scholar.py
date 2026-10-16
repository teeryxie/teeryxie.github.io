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


if __name__ == "__main__":
    unittest.main()
