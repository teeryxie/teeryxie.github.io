(function () {
  "use strict";

  async function readJson(path) {
    const response = await fetch(path);
    if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
    return response.json();
  }

  async function copyCitation(button, bibtex) {
    button.disabled = true;
    try {
      await navigator.clipboard.writeText(bibtex);
    } catch (error) {
      window.prompt("Copy BibTeX manually:", bibtex);
      button.disabled = false;
      return;
    }
    const originalLabel = button.textContent;
    button.textContent = "Copied";
    window.setTimeout(() => {
      button.textContent = originalLabel;
      button.disabled = false;
    }, 1400);
  }

  function bindCitations(publications) {
    if (!Array.isArray(publications.papers)) throw new Error("Invalid publication metadata");
    const papers = new Map(publications.papers.filter((paper) =>
      typeof paper.id === "string" && typeof paper.title === "string" && typeof paper.bibtex === "string" && paper.bibtex.trim()
    ).map((paper) => [paper.id, paper]));
    document.querySelectorAll("[data-cite]").forEach((button) => {
      const paper = papers.get(button.dataset.cite);
      button.disabled = !paper;
      if (!paper) { button.title = "Citation unavailable"; return; }
      button.title = "Copy BibTeX";
      button.setAttribute("aria-live", "polite");
      button.addEventListener("click", () => copyCitation(button, paper.bibtex));
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
      link.textContent = String(citation.count);
      link.hidden = false;
      link.setAttribute("aria-label", `${citation.count} Google Scholar citations`);
      link.href = url;
      link.title = `Google Scholar citations${date ? ` · Updated ${date}` : ""}`;
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
