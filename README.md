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

The homepage footer uses Flag Counter `rGrE` for the visitor map and country
statistics, starting in October 2026. Keep its image eager-loaded so homepage
visits count even when readers do not scroll to the footer.
