# Academic website design

Applies to the homepage, publication archive, projects, CV, blog index, and all blog articles. The public reference is https://howardli1984.github.io/. Use its restrained academic structure with Tianyu Xie's own content and images.

## Shared system

`academic.css` owns the palette, Chinese and English font stack, global focus treatment, and navigation. `site.css` owns subpage components; `home-academic.css` owns the homepage layout. Do not add page-specific font families, remote font imports, hero backgrounds, gradients, or decorative cards.

- White background, neutral text, underlined blue content links, thin grey rules.
- Body: Trebuchet MS with PingFang SC, Microsoft YaHei, Noto Sans CJK SC, Helvetica and Arial fallbacks. Code: SFMono-Regular, Menlo, Consolas and Liberation Mono.
- Body 15px; long-form text 16px / 1.85; mobile long-form text 15px.
- Page titles 32px / 27px on mobile; section headings 23px; article subheadings 19px.
- Shared navigation and personal profile. Articles replace the profile with a table of contents.
- Maximum page width 1360px, 210px profile column, 48px gap. Article prose is at most 820px wide.
- Publication and project rows use research images beside text. Blog index uses compact text rows. CV uses plain sections.
- On small screens: stack profile, content, and research images; wrap navigation and filters while keeping each navigation label on one line.

## Content and behavior

Use direct descriptions of the research question, method, result, and limitation. Preserve technical details, dates, authors, sources, paper status, and quantitative evidence. Use Chinese summaries for Chinese research notes and English summaries for English paper descriptions. Avoid duplicate titles in the blog list.

Keep existing URLs, data attributes, search, filters, article loading and error handling, table-of-contents navigation, table scrolling, image previews, code copying, focus indicators, and reduced-motion behavior. `[hidden]` must remain effective on grid/flex components. Project redirects remain redirects.
