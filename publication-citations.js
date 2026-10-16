(function () {
  "use strict";

  async function readJson(path) {
    const response = await fetch(path);
    if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
    return response.json();
  }

  function createCitationDialog() {
    const dialog = document.createElement("dialog");
    dialog.className = "citation-dialog";
    dialog.setAttribute("aria-labelledby", "citation-dialog-title");
    dialog.innerHTML = '<header><h2 id="citation-dialog-title"></h2><button type="button" data-citation-close aria-label="Close citation">Close</button></header><pre tabindex="0"><code></code></pre><footer><button type="button" data-citation-copy>Copy BibTeX</button><span role="status" aria-live="polite"></span></footer>';
    document.body.append(dialog);
    dialog.querySelector("[data-citation-close]").addEventListener("click", () => dialog.close());
    dialog.querySelector("[data-citation-copy]").addEventListener("click", () => copyCitation(dialog));
    dialog.addEventListener("click", (event) => {
      if (event.target !== dialog) return;
      const bounds = dialog.getBoundingClientRect();
      if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) dialog.close();
    });
    return dialog;
  }

  async function copyCitation(dialog) {
    const code = dialog.querySelector("code");
    const status = dialog.querySelector('[role="status"]');
    try {
      await navigator.clipboard.writeText(code.textContent);
      status.textContent = "BibTeX copied.";
    } catch (error) {
      const range = document.createRange();
      range.selectNodeContents(code);
      window.getSelection().removeAllRanges();
      window.getSelection().addRange(range);
      status.textContent = "Copy unavailable. The BibTeX is selected; copy it manually.";
    }
  }

  function showCitation(dialog, paper, trigger) {
    dialog.querySelector("h2").textContent = paper.title;
    dialog.querySelector("code").textContent = paper.bibtex;
    dialog.querySelector('[role="status"]').textContent = "";
    dialog.addEventListener("close", () => trigger.focus(), { once: true });
    dialog.showModal();
  }

  function bindCitations(publications) {
    if (!Array.isArray(publications.papers)) throw new Error("Invalid publication metadata");
    const papers = new Map(publications.papers.filter((paper) =>
      typeof paper.id === "string" && typeof paper.title === "string" && typeof paper.bibtex === "string" && paper.bibtex.trim()
    ).map((paper) => [paper.id, paper]));
    const dialog = createCitationDialog();
    document.querySelectorAll("[data-cite]").forEach((button) => {
      const paper = papers.get(button.dataset.cite);
      button.disabled = !paper;
      if (!paper) { button.title = "Citation unavailable"; return; }
      button.removeAttribute("title");
      button.addEventListener("click", () => showCitation(dialog, paper, button));
    });
  }

  function scholarUrl(url) {
    try {
      const parsed = new URL(url);
      return parsed.protocol === "https:" && parsed.hostname === "scholar.google.com" ? parsed.href : null;
    } catch (error) {
      if (error instanceof TypeError) return null;
      throw error;
    }
  }

  function updateCitationCounts(citations) {
    if (!citations.papers || typeof citations.papers !== "object" || Array.isArray(citations.papers)) throw new Error("Invalid citation counts");
    const updated = new Date(citations.updatedAt || NaN);
    const date = Number.isNaN(updated.getTime()) ? null : updated.toISOString().slice(0, 10);
    document.querySelectorAll("[data-citation-count]").forEach((link) => {
      const citation = citations.papers[link.dataset.citationCount];
      if (!citation || !Number.isSafeInteger(citation.count) || citation.count < 0) return;
      const url = scholarUrl(citation.url);
      if (!url) return;
      link.textContent = `Cited by ${citation.count}`;
      link.href = url;
      if (date) link.title = `Google Scholar citations · Updated ${date}`;
    });
    if (date) document.querySelectorAll("[data-citations-updated]").forEach((label) => {
      label.textContent = `Citation counts updated ${date}`;
    });
  }

  function initPublicationCitations() {
    const buttons = [...document.querySelectorAll("[data-cite]")];
    buttons.forEach((button) => { button.disabled = true; button.title = "Loading citation…"; });
    if (buttons.length) readJson("/data/publications.json").then(bindCitations).catch((error) => {
      buttons.forEach((button) => { button.title = "Citation unavailable. Please try again later."; });
      console.warn("Unable to load publication citations:", error);
    });
    if (document.querySelector("[data-citation-count]")) {
      readJson("/data/citations.json").then(updateCitationCounts).catch((error) => {
        console.warn("Unable to refresh citation counts; Scholar links are still available:", error);
      });
    }
  }

  document.addEventListener("DOMContentLoaded", initPublicationCitations);
})();
