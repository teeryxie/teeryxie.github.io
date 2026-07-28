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
  let listType = "ul";
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
      html.push(`<${listType}>${list.map((item) => `<li>${renderInline(item)}</li>`).join("")}</${listType}>`);
      list = [];
    }
  }

  function flushCode() {
    if (code.length) {
      html.push(`<pre><code>${escapeHtml(code.join("\n"))}</code></pre>`);
      code = [];
    }
  }

  function tableCells(value) {
    return value.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((cell) => cell.trim());
  }

  function renderTable(startIndex) {
    const headers = tableCells(lines[startIndex]);
    const rows = [];
    let index = startIndex + 2;
    while (index < lines.length && /^\s*\|.*\|\s*$/.test(lines[index])) {
      rows.push(tableCells(lines[index]));
      index += 1;
    }
    const head = headers.map((cell) => `<th>${renderInline(cell)}</th>`).join("");
    const body = rows.map((row) => `<tr>${headers.map((_, cellIndex) => `<td>${renderInline(row[cellIndex] || "")}</td>`).join("")}</tr>`).join("");
    html.push(`<div class="markdown-table-wrap"><table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`);
    return index - 1;
  }

  for (let index = 0; index < lines.length; index += 1) {
    const rawLine = lines[index];
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

    const nextLine = lines[index + 1] || "";
    if (/^\s*\|.*\|\s*$/.test(line) && /^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$/.test(nextLine)) {
      flushParagraph();
      flushList();
      index = renderTable(index);
      continue;
    }

    const quote = /^>\s?(.+)$/.exec(line);
    if (quote) {
      flushParagraph();
      flushList();
      html.push(`<blockquote>${renderInline(quote[1])}</blockquote>`);
      continue;
    }

    const bullet = /^[-*]\s+(.+)$/.exec(line);
    if (bullet) {
      flushParagraph();
      if (list.length && listType !== "ul") flushList();
      listType = "ul";
      list.push(bullet[1]);
      continue;
    }

    const ordered = /^\d+\.\s+(.+)$/.exec(line);
    if (ordered) {
      flushParagraph();
      if (list.length && listType !== "ol") flushList();
      listType = "ol";
      list.push(ordered[1]);
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
    <a class="blog-row" href="/blog/${post.slug}/" data-category="${escapeHtml(post.category)}">
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

function bindBlogFilters() {
  const filter = document.querySelector("[data-blog-filter]");
  const list = document.querySelector("[data-blog-list]");
  if (!filter || !list) return;

  const buttons = [...filter.querySelectorAll("[data-category]")];
  const applyFilter = (category) => {
    for (const button of buttons) {
      button.classList.toggle("active", button.dataset.category === category);
    }
    for (const row of list.querySelectorAll(".blog-row")) {
      row.hidden = category !== "all" && row.dataset.category !== category;
    }
  };

  filter.addEventListener("click", (event) => {
    const button = event.target.closest("[data-category]");
    if (!button) return;
    applyFilter(button.dataset.category);
  });
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
  bindBlogFilters();
  renderBlogArticle();
});
