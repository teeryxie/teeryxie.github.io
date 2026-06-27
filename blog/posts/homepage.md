# Building a Research Homepage That Does Not Feel Like a Template

An academic homepage should answer three questions quickly: who the researcher is, what they work on, and where the strongest artifacts are. It should not force visitors to decode a long paragraph or dig through a CV before seeing the research identity.

This version uses a compact structure: profile, research focus, representative publications, projects, blog, and CV. The blog is Markdown-based so paper introductions and daily writing can be edited without touching layout code.

## Design choices

- Keep the first screen focused on identity and current research direction.
- Use explicit paper status labels so accepted papers, submissions, withdrawn submissions, and preprints remain distinguishable.
- Keep blog entries as one-row list items for fast scanning.
- Render article content from Markdown files so writing stays separate from layout.

## Maintenance pattern

To add a new post, add a Markdown file under `blog/posts/`, add one metadata row in `blog/blog-data.js`, and create a matching route folder under `blog/`.
