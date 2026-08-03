(function () {
  "use strict";

  let dialog;
  let dialogImage;
  let dialogCaption;
  let trigger;

  function ensureDialog() {
    if (dialog) return dialog;

    dialog = document.createElement("dialog");
    dialog.className = "image-lightbox";
    dialog.setAttribute("aria-label", "Image preview");
    dialog.innerHTML = `
      <div class="lightbox-toolbar">
        <span data-lightbox-caption></span>
        <button type="button" data-lightbox-close aria-label="Close image" title="Close image">×</button>
      </div>
      <div class="lightbox-stage"><img alt=""></div>
    `;
    document.body.appendChild(dialog);
    dialogImage = dialog.querySelector("img");
    dialogCaption = dialog.querySelector("[data-lightbox-caption]");

    dialog.querySelector("[data-lightbox-close]").addEventListener("click", () => dialog.close());
    dialog.addEventListener("click", (event) => {
      if (event.target === dialog) dialog.close();
    });
    dialog.addEventListener("close", () => {
      document.body.classList.remove("dialog-open");
      trigger?.focus();
    });
    return dialog;
  }

  function openLightbox(source, sourceImage) {
    ensureDialog();
    trigger = source;
    dialogImage.src = source.dataset.lightboxSrc || sourceImage.currentSrc || sourceImage.src;
    dialogImage.alt = sourceImage.alt;
    dialogImage.classList.toggle("complex-svg", /\.svg(?:$|[?#])/i.test(dialogImage.src));
    dialogImage.classList.toggle("document-preview", source.matches(".paper-thumb, .project-media"));
    dialogCaption.textContent = source.dataset.caption
      || source.closest("figure")?.querySelector("figcaption")?.textContent
      || sourceImage.alt;
    document.body.classList.add("dialog-open");
    dialog.showModal();
    dialog.querySelector("[data-lightbox-close]").focus();
  }

  function bindLightboxes(root = document) {
    root.querySelectorAll("[data-lightbox], .figure-zoom").forEach((source) => {
      if (source.dataset.siteUiBound === "true") return;
      const sourceImage = source.querySelector("img");
      if (!sourceImage) return;
      source.dataset.siteUiBound = "true";
      source.addEventListener("click", (event) => {
        event.preventDefault();
        openLightbox(source, sourceImage);
      });
    });
  }

  window.SiteUI = { bindLightboxes };
  document.addEventListener("DOMContentLoaded", () => bindLightboxes());
})();
