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
Do not label submissions as published conference papers in BibTeX.

`data/citations.json` caches counts fetched from the public Scholar profile.
The Pages workflow refreshes it daily at **00:20 Asia/Shanghai** (16:20 UTC),
and can also be started from GitHub Actions with **Run workflow**. GitHub may
start scheduled runs late. Public Scholar requests can be blocked; failed,
incomplete, or ambiguous responses fail the run and preserve the last successful
snapshot. The page displays that snapshot's date. A missing record is not zero.

The workflow commits refreshed counts and explicitly deploys the updated site;
it does not rely on bot commits triggering another Pages build. Pages publishing
must use **GitHub Actions** rather than the legacy branch build.

Local refresh and matching checks use uv with Python's standard library only:

```sh
uv run --script scripts/update_scholar.py
uv run --no-project --python 3.12 python -m unittest discover -s scripts -p 'test_*.py'
```

For an offline parser check, use `--html /path/to/saved-public-profile.html`.
