# teeryxie.github.io

Personal academic homepage for Tianyu Xie.

Live site: https://teeryxie.github.io/

Shared typography, colors, focus states, and navigation live in `academic.css`.
`site.css` defines subpage components; `home-academic.css` defines the homepage.
See `design.md` before changing page styles.

Blog content lives in `blog/posts/*.md`, with metadata in `blog/blog-data.js`.
Keep each `displayTitle` identical to the Markdown heading, and update the HTML
page title and description when changing its metadata. The index renders from
metadata directly; it must not fetch every article to display the list.

After editing article Markdown or blog metadata, run `node blog/build.mjs`.
It renders the published HTML with the same Markdown renderer used by the
browser and reserves local image dimensions, so opening an article does not
wait for a second content download. Keep the generated article pages in Git.

## Paper citations

`data/publications.json` contains stable paper IDs, verified BibTeX, and Google
Scholar article IDs. Match new HTML buttons (`data-cite`) and count links
(`data-citation-count`) to those IDs on the homepage and publication archive.
Cite copies BibTeX directly; the small number above each title links to its
Google Scholar citations. Do not label submissions as published conference papers in BibTeX.

`data/citations.json` caches counts fetched from the public Scholar profile.
The same job discovers new Scholar records, fetches full author/date/venue
metadata from their detail pages, and adds them to the marked import region of
`publications/index.html` with working Cite and count controls. The homepage
selection remains curated. Imported records use conservative `@misc` BibTeX
and the venue exactly as indexed; they do not infer acceptance or CCF status.
They use a full-width text layout until a real first-page thumbnail is added manually.
Existing records match by Scholar ID, then unique normalized title, so newly
indexed manuscripts are linked rather than duplicated. A profile reaching the
100-record page limit fails explicitly rather than silently importing a partial list.
The Pages workflow refreshes it weekly on **Monday at 00:20 Asia/Shanghai**
(Sunday 16:20 UTC). Scheduled runs skip fetching when a successful
snapshot already exists for that date in Beijing; manual **Run workflow** always
refreshes. GitHub may start scheduled runs late. Public Scholar requests can be blocked; failed,
incomplete, or ambiguous responses fail the run and preserve the last successful
snapshot. The page displays that snapshot's date. A missing record is not zero.

Hosted runners use [SerpApi's Google Scholar Author interface](https://serpapi.com/google-scholar-author-api)
when the repository Actions secret `SERPAPI_API_KEY` is configured. This third-party
service retrieves Scholar's original HTML, so counts, record IDs, full author
metadata, and existing validation all continue to come from Scholar. No OpenAlex
or Semantic Scholar citation totals are substituted. Only new records request
article details; existing papers need one profile request per weekly refresh.
Transient network errors, HTTP 429, and server errors have bounded retries.
Authentication errors do not retry, and errors never print the API key or its URL.

To configure: create a SerpApi account, choose its free plan, and save the private
key under the repository's **Settings → Secrets and variables → Actions → New
repository secret**, named `SERPAPI_API_KEY`. Do not put it in source, JSON, HTML,
or workflow YAML. The [current free plan](https://serpapi.com/pricing) includes 250
searches per month. At the current profile size, normal weekly operation needs
about 4–5 profile searches per month plus one detail search per new paper.
Without a key, local use still attempts the public Scholar page directly and
reports a clear setup instruction if access is refused.

The workflow commits new publications, metadata, and refreshed counts and explicitly deploys the updated site;
it does not rely on bot commits triggering another Pages build. Pages publishing
must use **GitHub Actions** rather than the legacy branch build.

Local refresh and matching checks use uv with Python's standard library only:

```sh
uv run --script scripts/update_scholar.py
uv run --no-project --python 3.12 python -m unittest discover -s scripts -p 'test_*.py'
```

For an offline parser check, use `--html /path/to/saved-public-profile.html`.
