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

function plainText(value) {
  return value
    .replace(/!\[([^\]]*)\]\([^)]+\)/g, "$1")
    .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
    .replace(/[*_`~]/g, "")
    .trim();
}

function slugifyHeading(value, counts) {
  const normalized = plainText(value)
    .normalize("NFKC")
    .toLocaleLowerCase()
    .replace(/[^\p{Letter}\p{Number}]+/gu, "-")
    .replace(/^-+|-+$/g, "") || "section";
  const base = /^\d/.test(normalized) ? `section-${normalized}` : normalized;
  const count = (counts.get(base) || 0) + 1;
  counts.set(base, count);
  return count === 1 ? base : `${base}-${count}`;
}

function calculateReadingStats(markdown) {
  const content = markdown
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/!\[[^\]]*\]\([^)]+\)/g, " ")
    .replace(/https?:\/\/\S+/g, " ");
  const chineseCharacters = (content.match(/[\u3400-\u9fff]/g) || []).length;
  const englishWords = (content.replace(/[\u3400-\u9fff]/g, " ").match(/[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*/g) || []).length;
  const readingTime = Math.max(1, Math.ceil(chineseCharacters / 400 + englishWords / 220));
  return {
    readingTime,
    wordCount: chineseCharacters + englishWords,
  };
}

function renderMarkdown(markdown) {
  const lines = markdown.replace(/\r\n/g, "\n").split("\n");
  const html = [];
  const headings = [];
  const headingCounts = new Map();
  let title = "";
  let paragraph = [];
  let list = [];
  let listType = "ul";
  let code = [];
  let codeLanguage = "";
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
    if (code.length || inCode) {
      const languageLabel = codeLanguage ? `<span class="code-language">${escapeHtml(codeLanguage)}</span>` : "";
      html.push(`<div class="code-block">${languageLabel}<button class="code-copy" type="button" aria-label="复制代码" title="复制代码"><span aria-hidden="true"></span></button><pre><code>${escapeHtml(code.join("\n"))}</code></pre></div>`);
      code = [];
      codeLanguage = "";
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
    html.push(`<div class="markdown-table-wrap" tabindex="0"><table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`);
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
        codeLanguage = line.slice(3).trim();
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
      const rawLevel = heading[1].length;
      const headingText = plainText(heading[2]);
      if (rawLevel === 1 && !title) {
        title = headingText;
        continue;
      }
      const level = rawLevel === 1 ? 2 : rawLevel;
      const id = slugifyHeading(headingText, headingCounts);
      headings.push({ id, level, text: headingText });
      html.push(`<h${level} id="${escapeHtml(id)}">${renderInline(heading[2])}</h${level}>`);
      continue;
    }

    const image = /^!\[([^\]]*)\]\(([^)\s]+)(?:\s+"([^"]+)")?\)$/.exec(line.trim());
    if (image) {
      flushParagraph();
      flushList();
      const alt = escapeHtml(image[1] || "");
      const src = escapeHtml(image[2]);
      const caption = escapeHtml(image[3] || image[1] || "");
      html.push(`<figure><button class="figure-zoom" type="button" data-image-src="${src}" data-image-alt="${alt}" aria-label="查看原尺寸图片" title="查看原尺寸图片"><img src="${src}" alt="${alt}" loading="lazy" decoding="async"></button>${caption ? `<figcaption>${caption}</figcaption>` : ""}</figure>`);
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
  return {
    html: html.join("\n"),
    title,
    headings,
    ...calculateReadingStats(markdown),
  };
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

function tocLinks(headings) {
  return headings.map((heading) => `
    <a class="toc-link toc-level-${heading.level}" href="#${escapeHtml(heading.id)}" data-toc-id="${escapeHtml(heading.id)}">
      ${escapeHtml(heading.text)}
    </a>
  `).join("");
}

function settleLayoutBefore(target) {
  if (!target) return Promise.resolve([]);
  const precedingImages = [...document.querySelectorAll(".markdown-body img")].filter((image) => (
    image.compareDocumentPosition(target) & Node.DOCUMENT_POSITION_FOLLOWING
  ));
  return Promise.allSettled(precedingImages.map((image) => image.decode()));
}

async function scrollToHeading(id) {
  const target = document.getElementById(id);
  if (!target) return;
  history.replaceState(null, "", `${window.location.pathname}${window.location.search}#${encodeURIComponent(id)}`);
  target.scrollIntoView({ behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" });
  await settleLayoutBefore(target);
  requestAnimationFrame(() => target.scrollIntoView({ behavior: "auto", block: "start" }));
}

function buildArticleToc(headings) {
  const sidebar = document.querySelector(".blog-post .sidebar");
  const breadcrumb = document.querySelector(".blog-breadcrumb");
  if (!sidebar || !breadcrumb || !headings.length) {
    if (sidebar) sidebar.innerHTML = '<a class="toc-back" href="/blog/">← Back to Blog</a>';
    return;
  }

  const links = tocLinks(headings);
  sidebar.innerHTML = `
    <nav class="article-toc" aria-label="本文目录">
      <a class="toc-back" href="/blog/">← Back to Blog</a>
      <strong>本文目录</strong>
      <div class="toc-links">${links}</div>
    </nav>
  `;

  const mobileToc = document.createElement("details");
  mobileToc.className = "mobile-toc";
  mobileToc.innerHTML = `<summary>本文目录 <span>${headings.length} 节</span></summary><nav aria-label="移动端本文目录">${links}</nav>`;
  breadcrumb.insertAdjacentElement("afterend", mobileToc);

  document.querySelectorAll("[data-toc-id]").forEach((link) => {
    link.addEventListener("click", (event) => {
      event.preventDefault();
      scrollToHeading(link.dataset.tocId);
      mobileToc.open = false;
    });
  });

  const headingElements = headings.map((heading) => document.getElementById(heading.id)).filter(Boolean);
  const observer = new IntersectionObserver((entries) => {
    const visible = entries
      .filter((entry) => entry.isIntersecting)
      .sort((left, right) => left.boundingClientRect.top - right.boundingClientRect.top)[0];
    if (!visible) return;
    document.querySelectorAll("[data-toc-id]").forEach((link) => {
      const active = link.dataset.tocId === visible.target.id;
      link.classList.toggle("active", active);
      if (active) link.setAttribute("aria-current", "location");
      else link.removeAttribute("aria-current");
    });
  }, { rootMargin: "-88px 0px -68% 0px", threshold: [0, 1] });
  headingElements.forEach((heading) => observer.observe(heading));

  if (window.location.hash) {
    const id = decodeURIComponent(window.location.hash.slice(1));
    const target = document.getElementById(id);
    const restoreHashPosition = () => target?.scrollIntoView({ behavior: "auto", block: "start" });
    requestAnimationFrame(restoreHashPosition);
    window.setTimeout(restoreHashPosition, 120);
    settleLayoutBefore(target).then(() => {
      requestAnimationFrame(restoreHashPosition);
    });
  }
}

function buildReadingControls(article) {
  const masthead = document.querySelector(".masthead");
  if (!masthead) return;

  const progress = document.createElement("div");
  progress.className = "reading-progress";
  progress.setAttribute("aria-hidden", "true");
  progress.innerHTML = "<span></span>";
  masthead.appendChild(progress);

  const backToTop = document.createElement("button");
  backToTop.className = "back-to-top";
  backToTop.type = "button";
  backToTop.setAttribute("aria-label", "返回顶部");
  backToTop.title = "返回顶部";
  backToTop.innerHTML = '<span aria-hidden="true"></span>';
  document.body.appendChild(backToTop);
  backToTop.addEventListener("click", () => window.scrollTo({ top: 0, behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" }));

  let scheduled = false;
  const update = () => {
    const start = article.offsetTop;
    const end = Math.max(start + 1, start + article.offsetHeight - window.innerHeight);
    const value = Math.min(1, Math.max(0, (window.scrollY - start) / (end - start)));
    progress.querySelector("span").style.transform = `scaleX(${value})`;
    backToTop.classList.toggle("visible", window.scrollY > window.innerHeight);
    scheduled = false;
  };
  window.addEventListener("scroll", () => {
    if (!scheduled) {
      scheduled = true;
      requestAnimationFrame(update);
    }
  }, { passive: true });
  window.addEventListener("resize", update);
  new ResizeObserver(update).observe(article);
  update();
}

function buildImageLightbox(article) {
  const dialog = document.createElement("dialog");
  dialog.className = "image-lightbox";
  dialog.innerHTML = `
    <div class="lightbox-toolbar"><span data-lightbox-caption></span><button type="button" data-lightbox-close aria-label="关闭图片" title="关闭图片">×</button></div>
    <div class="lightbox-stage"><img alt=""></div>
  `;
  document.body.appendChild(dialog);
  const image = dialog.querySelector("img");
  const caption = dialog.querySelector("[data-lightbox-caption]");
  let trigger = null;

  const close = () => dialog.close();
  dialog.querySelector("[data-lightbox-close]").addEventListener("click", close);
  dialog.addEventListener("click", (event) => {
    if (event.target === dialog) close();
  });
  dialog.addEventListener("close", () => {
    document.body.classList.remove("dialog-open");
    trigger?.focus();
  });

  article.querySelectorAll(".figure-zoom").forEach((button) => {
    button.addEventListener("click", () => {
      trigger = button;
      const sourceImage = button.querySelector("img");
      image.src = sourceImage.currentSrc || sourceImage.src;
      image.alt = sourceImage.alt;
      image.classList.toggle("complex-svg", /\.svg(?:$|[?#])/i.test(image.src));
      caption.textContent = button.closest("figure")?.querySelector("figcaption")?.textContent || sourceImage.alt;
      document.body.classList.add("dialog-open");
      dialog.showModal();
      dialog.querySelector("[data-lightbox-close]").focus();
    });
  });
}

function bindCodeCopy(article) {
  const liveRegion = document.createElement("span");
  liveRegion.className = "sr-only";
  liveRegion.setAttribute("aria-live", "polite");
  article.appendChild(liveRegion);

  article.querySelectorAll(".code-copy").forEach((button) => {
    button.addEventListener("click", async () => {
      const code = button.closest(".code-block").querySelector("code").textContent;
      try {
        let copied = false;
        if (navigator.clipboard?.writeText) {
          try {
            await navigator.clipboard.writeText(code);
            copied = true;
          } catch (_clipboardError) {
            copied = false;
          }
        }
        if (!copied) {
          const textarea = document.createElement("textarea");
          textarea.value = code;
          textarea.setAttribute("readonly", "");
          textarea.style.position = "fixed";
          textarea.style.opacity = "0";
          document.body.appendChild(textarea);
          textarea.select();
          copied = document.execCommand("copy");
          textarea.remove();
        }
        if (!copied) throw new Error("Copy command failed");
        button.classList.add("copied");
        button.setAttribute("aria-label", "已复制");
        button.title = "已复制";
        liveRegion.textContent = "代码已复制";
        window.setTimeout(() => {
          button.classList.remove("copied");
          button.setAttribute("aria-label", "复制代码");
          button.title = "复制代码";
        }, 1600);
      } catch (_error) {
        liveRegion.textContent = "复制失败，请手动选择代码";
      }
    });
  });
}

function buildArticleFooter(post) {
  const index = window.BLOG_POSTS.findIndex((item) => item.slug === post.slug);
  const previous = index > 0 ? window.BLOG_POSTS[index - 1] : null;
  const next = index < window.BLOG_POSTS.length - 1 ? window.BLOG_POSTS[index + 1] : null;
  const related = window.BLOG_POSTS.filter((item) => item.slug !== post.slug && item.section === post.section).slice(0, 3);
  const link = (item, label) => item ? `<a href="/blog/${item.slug}/"><span>${label}</span><strong>${escapeHtml(item.title)}</strong><small>${escapeHtml(item.category)} · ${escapeHtml(item.date)}</small></a>` : '<span class="article-nav-empty" aria-hidden="true"></span>';

  const footer = document.createElement("footer");
  footer.className = "article-footer-nav";
  footer.innerHTML = `
    <nav class="article-sequence" aria-label="上一篇和下一篇">
      ${link(previous, "上一篇")}
      ${link(next, "下一篇")}
    </nav>
    ${related.length ? `<section class="related-posts"><h2>相关文章</h2><div>${related.map((item) => link(item, item.category)).join("")}</div></section>` : ""}
  `;
  document.querySelector(".blog-post .content")?.appendChild(footer);
}

function renderArticleError(article, retry) {
  article.innerHTML = `
    <div class="article-error" role="alert">
      <strong>文章内容加载失败</strong>
      <p>Markdown 文件暂时无法读取，请重试或返回博客索引。</p>
      <div><button type="button" data-article-retry>重试</button><a href="/blog/">返回 Blog</a></div>
    </div>
  `;
  article.querySelector("[data-article-retry]").addEventListener("click", retry);
}

async function renderBlogArticle() {
  const article = document.querySelector("[data-markdown-article]");
  if (!article || !window.BLOG_POSTS) return;

  const slug = document.body.dataset.postSlug;
  const post = window.BLOG_POSTS.find((item) => item.slug === slug);
  if (!post) {
    renderArticleError(article, () => window.location.reload());
    return;
  }

  document.title = `${post.title} | Tianyu Xie`;
  const title = document.querySelector("[data-post-title]");
  const meta = document.querySelector("[data-post-meta]");
  const breadcrumbSection = document.querySelector("[data-breadcrumb-section]");
  if (breadcrumbSection) breadcrumbSection.textContent = post.section;

  const load = async () => {
    article.innerHTML = '<div class="article-skeleton" aria-label="正在加载文章"><span></span><span></span><span></span><span></span><span></span></div>';
    try {
      const response = await fetch(`/blog/posts/${slug}.md`, { cache: "no-cache" });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const rendered = renderMarkdown(await response.text());
      const primaryTitle = rendered.title || post.title;
      const showEnglishSubtitle = hasChinese(primaryTitle) && primaryTitle !== post.title;
      if (title) {
        title.textContent = primaryTitle;
        title.lang = hasChinese(primaryTitle) ? "zh-CN" : "en";
      }
      document.querySelector("[data-post-subtitle]")?.remove();
      if (showEnglishSubtitle && title) {
        const subtitle = document.createElement("p");
        subtitle.className = "post-subtitle";
        subtitle.dataset.postSubtitle = "";
        subtitle.lang = "en";
        subtitle.textContent = post.title;
        title.insertAdjacentElement("afterend", subtitle);
      }
      if (meta) meta.textContent = `${post.category} · ${post.date} · ${rendered.readingTime} min read · ${rendered.wordCount.toLocaleString()} 字词`;
      article.innerHTML = rendered.html;
      buildArticleToc(rendered.headings);
      buildReadingControls(article);
      buildImageLightbox(article);
      bindCodeCopy(article);
      buildArticleFooter(post);
    } catch (_error) {
      renderArticleError(article, load);
    }
  };

  await load();
}

document.addEventListener("DOMContentLoaded", () => {
  renderBlogIndex();
  renderBlogArticle();
});
