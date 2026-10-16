import io
import traceback
import unittest
from email.message import Message
from unittest.mock import patch
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse

from scholar_http import fetch_html
from update_scholar import PROFILE_ID, PROFILE_URL


class TransportTests(unittest.TestCase):
    def response(self, content=b"<html>Scholar</html>"):
        response = io.BytesIO(content)
        response.headers = Message()
        return response

    @patch.dict("os.environ", {"SERPAPI_API_KEY": "private-test-key"})
    def test_serpapi_routes_profile_and_detail_without_printing_credentials(self):
        urls = [PROFILE_URL, f"https://scholar.google.com/citations?user={PROFILE_ID}&view_op=view_citation&citation_for_view={PROFILE_ID}:one"]
        for url in urls:
            with self.subTest(url=url), patch("scholar_http.urlopen", return_value=self.response()) as opened:
                self.assertEqual(fetch_html(url), "<html>Scholar</html>")
                request = opened.call_args.args[0]
                parsed = urlparse(request.full_url)
                parameters = parse_qs(parsed.query)
                self.assertEqual(parsed.hostname, "serpapi.com")
                self.assertEqual(parameters["engine"], ["google_scholar_author"])
                self.assertEqual(parameters["author_id"], [PROFILE_ID])
                self.assertEqual(parameters["api_key"], ["private-test-key"])
                self.assertEqual(parameters["output"], ["html"])
                if "view_op" in url:
                    self.assertEqual(parameters["view_op"], ["view_citation"])
                    self.assertEqual(parameters["citation_id"], [f"{PROFILE_ID}:one"])

    @patch.dict("os.environ", {"SERPAPI_API_KEY": ""})
    def test_no_key_uses_direct_fetch_and_server_charset(self):
        response = self.response("café".encode("iso-8859-1"))
        response.headers["Content-Type"] = "text/html; charset=iso-8859-1"
        with patch("scholar_http.urlopen", return_value=response) as opened:
            self.assertEqual(fetch_html(PROFILE_URL), "café")
        self.assertEqual(opened.call_args.args[0].full_url, PROFILE_URL)

    @patch.dict("os.environ", {"SERPAPI_API_KEY": "private-test-key"})
    def test_auth_errors_are_not_retried_and_cannot_expose_key(self):
        for code in (401, 403):
            error = HTTPError("https://serpapi.com/search?api_key=private-test-key", code,
                              "secret-response", None, None)
            with patch("scholar_http.urlopen", side_effect=error) as opened, patch("scholar_http.time.sleep") as sleep:
                try:
                    fetch_html(PROFILE_URL)
                except RuntimeError:
                    formatted = traceback.format_exc()
                    self.assertNotIn("private-test-key", formatted)
                    self.assertNotIn("secret-response", formatted)
                    self.assertIn(f"HTTP {code}", formatted)
                else:
                    self.fail("Authentication errors must fail")
                self.assertEqual(opened.call_count, 1)
                sleep.assert_not_called()

    @patch.dict("os.environ", {"SERPAPI_API_KEY": ""})
    def test_direct_403_explains_secret_configuration(self):
        with patch("scholar_http.urlopen", side_effect=HTTPError(PROFILE_URL, 403, "Blocked", None, None)), self.assertRaisesRegex(RuntimeError, "Configure the repository SERPAPI_API_KEY"):
            fetch_html(PROFILE_URL)

    def test_transient_failures_retry_at_most_three_times(self):
        errors = [HTTPError("private-url", 429, "limited", None, None),
                  HTTPError("private-url", 503, "unavailable", None, None),
                  URLError("secret network detail")]
        for error in errors:
            with self.subTest(error=type(error).__name__), patch("scholar_http.urlopen", side_effect=error) as opened, patch("scholar_http.time.sleep") as sleep:
                with self.assertRaises(RuntimeError) as raised:
                    fetch_html(PROFILE_URL)
                self.assertNotIn("private-url", str(raised.exception))
                self.assertNotIn("secret network detail", str(raised.exception))
                self.assertEqual(opened.call_count, 3)
                self.assertEqual([call.args[0] for call in sleep.call_args_list], [5, 20])

    def test_transient_failure_can_recover(self):
        with patch("scholar_http.urlopen", side_effect=[URLError("temporary"), self.response()]) as opened, patch("scholar_http.time.sleep") as sleep:
            self.assertEqual(fetch_html(PROFILE_URL), "<html>Scholar</html>")
            self.assertEqual(opened.call_count, 2)
            sleep.assert_called_once_with(5)


if __name__ == "__main__":
    unittest.main()
