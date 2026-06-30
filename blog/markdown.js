function escapeHtml(value) {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function renderInline(value) {
  return escapeHtml(value)
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2">$1</a>');
}

function renderMarkdown(markdown) {
  const lines = markdown.replace(/\r\n/g, "\n").split("\n");
  const html = [];
  let paragraph = [];
  let list = [];
  let code = [];
  let inCode = false;

  function flushParagraph() {
    if (paragraph.length) {
      html.push(`<p>${renderInline(paragraph.join(" "))}</p>`);
      paragraph = [];
    }
  }

  function flushList() {
    if (list.length) {
      html.push(`<ul>${list.map((item) => `<li>${renderInline(item)}</li>`).join("")}</ul>`);
      list = [];
    }
  }

  function flushCode() {
    if (code.length) {
      html.push(`<pre><code>${escapeHtml(code.join("\n"))}</code></pre>`);
      code = [];
    }
  }

  for (const rawLine of lines) {
    const line = rawLine.trimEnd();

    if (line.startsWith("```")) {
      if (inCode) {
        flushCode();
        inCode = false;
      } else {
        flushParagraph();
        flushList();
        inCode = true;
      }
      continue;
    }

    if (inCode) {
      code.push(rawLine);
      continue;
    }

    if (!line.trim()) {
      flushParagraph();
      flushList();
      continue;
    }

    const heading = /^(#{1,3})\s+(.+)$/.exec(line);
    if (heading) {
      flushParagraph();
      flushList();
      const level = heading[1].length + 1;
      html.push(`<h${level}>${renderInline(heading[2])}</h${level}>`);
      continue;
    }

    const image = /^!\[([^\]]*)\]\(([^)\s]+)(?:\s+"([^"]+)")?\)$/.exec(line.trim());
    if (image) {
      flushParagraph();
      flushList();
      const alt = escapeHtml(image[1] || "");
      const src = escapeHtml(image[2]);
      const caption = escapeHtml(image[3] || image[1] || "");
      html.push(`<figure><img src="${src}" alt="${alt}">${caption ? `<figcaption>${caption}</figcaption>` : ""}</figure>`);
      continue;
    }

    const bullet = /^[-*]\s+(.+)$/.exec(line);
    if (bullet) {
      flushParagraph();
      list.push(bullet[1]);
      continue;
    }

    paragraph.push(line.trim());
  }

  flushParagraph();
  flushList();
  flushCode();
  return html.join("\n");
}

function renderBlogIndex() {
  const list = document.querySelector("[data-blog-list]");
  if (!list || !window.BLOG_POSTS) return;

  list.innerHTML = window.BLOG_POSTS.map((post) => `
    <a class="blog-row" href="/blog/${post.slug}/">
      <span class="blog-row-main">
        <strong>${escapeHtml(post.title)}</strong>
        <span>${escapeHtml(post.summary)}</span>
      </span>
      <span class="blog-row-meta">
        <span>${escapeHtml(post.category)}</span>
        <span>${escapeHtml(post.status)}</span>
        <time>${escapeHtml(post.date)}</time>
      </span>
    </a>
  `).join("");
}

async function renderBlogArticle() {
  const article = document.querySelector("[data-markdown-article]");
  if (!article || !window.BLOG_POSTS) return;

  const slug = document.body.dataset.postSlug;
  const post = window.BLOG_POSTS.find((item) => item.slug === slug);
  if (!post) {
    article.innerHTML = "<p>Post not found.</p>";
    return;
  }

  document.title = `${post.title} | Tianyu Xie`;
  const title = document.querySelector("[data-post-title]");
  const meta = document.querySelector("[data-post-meta]");
  if (title) title.textContent = post.title;
  if (meta) meta.textContent = `${post.category} · ${post.status} · ${post.date}`;

  const response = await fetch(`/blog/posts/${slug}.md`, { cache: "no-cache" });
  if (!response.ok) {
    article.innerHTML = "<p>Markdown source failed to load.</p>";
    return;
  }

  article.innerHTML = renderMarkdown(await response.text());
}

document.addEventListener("DOMContentLoaded", () => {
  renderBlogIndex();
  renderBlogArticle();
});
