document.addEventListener("click", async (event) => {
  const button = event.target.closest(".cite-action");
  if (!button) return;

  const bib = button.dataset.bib || "";
  if (!bib) return;

  const original = button.textContent;
  button.textContent = "Copied";

  try {
    await navigator.clipboard.writeText(bib);
  } catch {
    window.prompt("Copy BibTeX", bib);
  }

  window.setTimeout(() => {
    button.textContent = original;
  }, 1400);
});
