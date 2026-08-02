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

const SECTION_NAMES = ["Research Notes", "Surveys", "Paper Readings", "Publications", "Logs"];

function splitTags(status) {
  return [...new Set(status.split("·").map((tag) => tag.trim()).filter(Boolean))];
}

function hasChinese(value) {
  return /[\u3400-\u9fff]/.test(value);
}

function extractMarkdownTitle(markdown) {
  const match = /^#\s+(.+)$/m.exec(markdown);
  return match ? match[1].trim() : "";
}

async function loadIndexTitles(posts) {
  const titles = await Promise.all(posts.map(async (post) => {
    try {
      const response = await fetch(`/blog/posts/${post.slug}.md`, { cache: "no-cache" });
      if (!response.ok) return "";
      return extractMarkdownTitle(await response.text());
    } catch (_error) {
      return "";
    }
  }));

  return posts.map((post, index) => ({
    ...post,
    markdownTitle: titles[index],
    tags: splitTags(post.status),
  }));
}

function renderPostRow(post) {
  const markdownTitle = post.markdownTitle || "";
  const useChineseTitle = hasChinese(markdownTitle);
  const primaryTitle = useChineseTitle ? markdownTitle : post.title;
  const secondaryTitle = useChineseTitle && markdownTitle !== post.title ? post.title : "";
  const visibleTags = post.tags.slice(0, 2);

  return `
    <a class="blog-row" href="/blog/${post.slug}/" data-slug="${escapeHtml(post.slug)}">
      <span class="blog-row-main">
        <strong${useChineseTitle ? "" : ' lang="en"'}>${escapeHtml(primaryTitle)}</strong>
        ${secondaryTitle ? `<span class="blog-row-subtitle" lang="en">${escapeHtml(secondaryTitle)}</span>` : ""}
        <span class="blog-row-summary" lang="en">${escapeHtml(post.summary)}</span>
      </span>
      <span class="blog-row-meta">
        <span class="blog-row-category">${escapeHtml(post.category)}</span>
        <time datetime="${escapeHtml(post.date)}">${escapeHtml(post.date)}</time>
        ${visibleTags.map((tag, index) => `<span class="blog-row-tag blog-row-tag-${index + 1}">${escapeHtml(tag)}</span>`).join("")}
      </span>
    </a>
  `;
}

async function renderBlogIndex() {
  const list = document.querySelector("[data-blog-list]");
  if (!list || !window.BLOG_POSTS) return;

  const searchForm = document.querySelector("[data-blog-search]");
  const searchInput = searchForm?.querySelector('input[type="search"]');
  const filter = document.querySelector("[data-blog-filter]");
  const clearButton = document.querySelector("[data-search-clear]");
  const resultCount = document.querySelector("[data-blog-result-count]");
  const emptyState = document.querySelector("[data-blog-empty]");
  const resetButtons = [...document.querySelectorAll("[data-blog-reset]")];
  if (!searchForm || !searchInput || !filter || !resultCount || !emptyState) return;

  const posts = await loadIndexTitles(window.BLOG_POSTS);
  const buttons = [...filter.querySelectorAll("[data-section]")];
  const params = new URLSearchParams(window.location.search);
  const requestedSection = params.get("section") || "all";
  const state = {
    query: params.get("q") || "",
    section: SECTION_NAMES.includes(requestedSection) ? requestedSection : "all",
  };

  for (const button of buttons) {
    const section = button.dataset.section;
    const count = section === "all" ? posts.length : posts.filter((post) => post.section === section).length;
    const counter = button.querySelector("span");
    if (counter) counter.textContent = String(count);
  }

  const searchableText = (post) => [
    post.title,
    post.markdownTitle,
    post.summary,
    post.category,
    post.section,
    ...post.tags,
  ].join(" ").toLocaleLowerCase();

  const updateUrl = () => {
    const nextParams = new URLSearchParams();
    if (state.query) nextParams.set("q", state.query);
    if (state.section !== "all") nextParams.set("section", state.section);
    const queryString = nextParams.toString();
    history.replaceState(null, "", `${window.location.pathname}${queryString ? `?${queryString}` : ""}${window.location.hash}`);
  };

  const render = () => {
    const normalizedQuery = state.query.trim().toLocaleLowerCase();
    const matches = posts.filter((post) => {
      const sectionMatches = state.section === "all" || post.section === state.section;
      return sectionMatches && (!normalizedQuery || searchableText(post).includes(normalizedQuery));
    });

    list.innerHTML = matches.map(renderPostRow).join("");
    list.hidden = matches.length === 0;
    list.setAttribute("aria-busy", "false");
    emptyState.hidden = matches.length !== 0;
    resultCount.textContent = `${matches.length} 篇文章`;
    searchInput.value = state.query;
    clearButton.hidden = !state.query;
    for (const button of buttons) {
      const active = button.dataset.section === state.section;
      button.classList.toggle("active", active);
      button.setAttribute("aria-pressed", String(active));
    }
    resetButtons.forEach((button) => {
      button.hidden = !state.query && state.section === "all";
    });
    updateUrl();
  };

  searchForm.addEventListener("submit", (event) => event.preventDefault());
  searchInput.addEventListener("input", () => {
    state.query = searchInput.value;
    render();
  });
  clearButton.addEventListener("click", () => {
    state.query = "";
    render();
    searchInput.focus();
  });
  filter.addEventListener("click", (event) => {
    const button = event.target.closest("[data-section]");
    if (!button) return;
    state.section = button.dataset.section;
    render();
  });
  resetButtons.forEach((button) => {
    button.addEventListener("click", () => {
      state.query = "";
      state.section = "all";
      render();
      searchInput.focus();
    });
  });

  render();
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
  const breadcrumbSection = document.querySelector("[data-breadcrumb-section]");
  if (breadcrumbSection) breadcrumbSection.textContent = post.section;

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
