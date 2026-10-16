"""Fetch Scholar HTML directly or through the configured SerpApi account."""

import os
import time
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import Request, urlopen

RETRY_DELAYS = (5, 20)


def request_url(scholar_url):
    api_key = os.environ.get("SERPAPI_API_KEY", "").strip()
    if not api_key:
        return scholar_url, "Google Scholar"
    query = parse_qs(urlparse(scholar_url).query)
    parameters = {"engine": "google_scholar_author", "author_id": query["user"][0],
                  "hl": "en", "num": "100", "output": "html", "api_key": api_key}
    if query.get("view_op") == ["view_citation"]:
        parameters.update(view_op="view_citation", citation_id=query["citation_for_view"][0])
    return "https://serpapi.com/search?" + urlencode(parameters), "SerpApi"


def fetch_html(scholar_url):
    url, provider = request_url(scholar_url)
    request = Request(url, headers={"User-Agent": "TianyuXieHomepage/1.0"})
    for attempt in range(len(RETRY_DELAYS) + 1):
        try:
            with urlopen(request, timeout=60) as response:
                return response.read().decode(response.headers.get_content_charset() or "utf-8")
        except HTTPError as error:
            retryable = error.code == 429 or 500 <= error.code < 600
            message = f"{provider} returned HTTP {error.code}."
            if provider == "Google Scholar" and error.code == 403:
                message += " Configure the repository SERPAPI_API_KEY secret to use SerpApi."
            elif provider == "SerpApi" and error.code in {401, 403}:
                message += " Check the SERPAPI_API_KEY repository secret and account access."
        except (URLError, TimeoutError, OSError):
            retryable, message = True, f"{provider} network request failed."
        except UnicodeError:
            raise RuntimeError(f"{provider} returned unreadable HTML; cached files retained.") from None
        if not retryable or attempt == len(RETRY_DELAYS):
            # URL and exception repr may include the API key, so neither is logged.
            raise RuntimeError(message + " Cached files retained.") from None
        time.sleep(RETRY_DELAYS[attempt])
