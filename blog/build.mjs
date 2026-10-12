import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('../', import.meta.url));
const renderer = { document: { addEventListener() {} }, window: {} };
vm.runInNewContext(fs.readFileSync(path.join(root, 'blog/markdown.js'), 'utf8'), renderer);
vm.runInNewContext(fs.readFileSync(path.join(root, 'blog/blog-data.js'), 'utf8'), renderer);
const sectionLabels = vm.runInNewContext('SECTION_LABELS', renderer);

function imageDimensions(src) {
  const asset = path.resolve(root, `.${src}`);
  if (!asset.startsWith(path.join(root, 'images') + path.sep) || !fs.existsSync(asset)) return null;
  const bytes = fs.readFileSync(asset);
  if (src.endsWith('.svg')) {
    const viewBox = /viewBox=["']([^"']+)["']/.exec(bytes.toString('utf8'));
    const coordinates = viewBox?.[1].trim().split(/[\s,]+/).map(Number);
    return coordinates?.length === 4 ? coordinates.slice(2) : null;
  }
  if (bytes.length >= 24 && bytes.subarray(0, 8).toString('hex') === '89504e470d0a1a0a') {
    return [bytes.readUInt32BE(16), bytes.readUInt32BE(20)];
  }
  return null;
}

function reserveImageSpace(html) {
  return html.replace(/<img src="([^"]+)"/g, (tag, src) => {
    const dimensions = imageDimensions(src);
    if (!dimensions || !dimensions.every(n => Number.isFinite(n) && n > 0)) return tag;
    return `${tag} width="${dimensions[0]}" height="${dimensions[1]}"`;
  });
}

function articleHeader(post, title) {
  const chinese = renderer.hasChinese(title);
  const category = chinese ? sectionLabels[post.section] : post.section;
  return `<header class="inner-hero article-hero"><nav class="blog-breadcrumb" aria-label="Breadcrumb"><a href="/blog/">← ${chinese ? '研究笔记' : 'Research notes'}</a></nav><h1 class="page-title" data-post-title>${renderer.escapeHtml(title)}</h1><p class="lead" data-post-meta>${renderer.escapeHtml(post.date)} · ${renderer.escapeHtml(category)}</p></header>`;
}

function desktopContents(headings) {
  return `<nav class="article-toc" aria-label="本文目录"><strong>本文目录</strong><div class="toc-links">${renderer.tocLinks(headings)}</div></nav>`;
}

function mobileContents(headings) {
  return `<details class="mobile-toc"><summary>本文目录</summary><nav aria-label="移动端本文目录">${renderer.tocLinks(headings)}</nav></details>`;
}

function buildArticle(post) {
  const markdown = fs.readFileSync(path.join(root, 'blog/posts', `${post.slug}.md`), 'utf8');
  const article = renderer.renderMarkdown(markdown);
  if (article.title !== post.displayTitle) throw new Error(`Title mismatch: ${post.slug}`);
  const htmlPath = path.join(root, 'blog', post.slug, 'index.html');
  let page = fs.readFileSync(htmlPath, 'utf8');
  page = page.replace(/<article class="content">[\s\S]*?<\/article>/, `<article class="content">${articleHeader(post, article.title)}${mobileContents(article.headings)}<div class="markdown-body" data-markdown-article data-rendered="true">${reserveImageSpace(article.html)}</div><script type="application/json" data-article-headings>${JSON.stringify(article.headings).replaceAll('<', '\\u003c')}</script></article>`);
  page = page.replace(/<nav class="article-toc"[\s\S]*?<\/nav>/, '');
  page = page.replace('</section></aside>', `</section>${desktopContents(article.headings)}</aside>`);
  page = page.replace(/<meta name="description"[^>]*>/, `<meta name="description" content="${renderer.escapeHtml(post.summary)}">`);
  fs.writeFileSync(htmlPath, page);
}

renderer.window.BLOG_POSTS.forEach(buildArticle);
console.log(`Rendered ${renderer.window.BLOG_POSTS.length} articles with reserved image space.`);
