(function () {
  "use strict";

  const filters = new Set(["all", "first-author", "ccf-a", "preprints", "earlier-work"]);

  function initPublications() {
    const toolbar = document.querySelector("[data-publication-filters]");
    const papers = [...document.querySelectorAll("[data-paper-filters]")];
    const resultCount = document.querySelector("[data-publication-count]");
    if (!toolbar || !papers.length || !resultCount) return;

    const params = new URLSearchParams(window.location.search);
    let active = filters.has(params.get("filter")) ? params.get("filter") : "all";

    const apply = (filter, updateUrl = true) => {
      active = filters.has(filter) ? filter : "all";
      let visible = 0;
      papers.forEach((paper) => {
        const values = paper.dataset.paperFilters.split(/\s+/);
        const show = active === "all" || values.includes(active);
        paper.hidden = !show;
        if (show) visible += 1;
      });
      toolbar.querySelectorAll("button[data-filter]").forEach((button) => {
        const selected = button.dataset.filter === active;
        button.classList.toggle("active", selected);
        button.setAttribute("aria-pressed", String(selected));
      });
      resultCount.textContent = `${visible} publication${visible === 1 ? "" : "s"}`;

      if (updateUrl) {
        const next = new URL(window.location.href);
        if (active === "all") next.searchParams.delete("filter");
        else next.searchParams.set("filter", active);
        history.replaceState(null, "", `${next.pathname}${next.search}${next.hash}`);
      }
    };

    toolbar.addEventListener("click", (event) => {
      const button = event.target.closest("button[data-filter]");
      if (button) apply(button.dataset.filter);
    });
    window.addEventListener("popstate", () => {
      const filter = new URLSearchParams(window.location.search).get("filter") || "all";
      apply(filter, false);
    });
    apply(active, false);
  }

  document.addEventListener("DOMContentLoaded", initPublications);
})();
