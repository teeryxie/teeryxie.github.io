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
