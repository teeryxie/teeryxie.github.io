(function () {
  "use strict";

  const body = document.body;
  const header = document.querySelector("[data-home-header]");
  const toggle = document.querySelector("[data-menu-toggle]");
  const closeButton = document.querySelector("[data-menu-close]");
  const menu = document.querySelector("[data-mobile-menu]");
  const backdrop = document.querySelector("[data-menu-backdrop]");
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  let menuTrigger = null;

  function setHeaderState() {
    header?.classList.toggle("is-scrolled", window.scrollY > 24);
  }

  function openMenu() {
    if (!toggle || !menu || !backdrop) return;
    menuTrigger = document.activeElement;
    toggle.setAttribute("aria-expanded", "true");
    toggle.setAttribute("aria-label", "Close navigation");
    menu.setAttribute("aria-hidden", "false");
    backdrop.hidden = false;
    body.classList.add("menu-open");
    requestAnimationFrame(() => {
      menu.classList.add("is-open");
      backdrop.classList.add("is-visible");
      window.setTimeout(() => closeButton?.focus(), 0);
    });
  }

  function closeMenu() {
    if (!toggle || !menu || !backdrop) return;
    toggle.setAttribute("aria-expanded", "false");
    toggle.setAttribute("aria-label", "Open navigation");
    menu.setAttribute("aria-hidden", "true");
    menu.classList.remove("is-open");
    backdrop.classList.remove("is-visible");
    body.classList.remove("menu-open");
    window.setTimeout(() => {
      backdrop.hidden = true;
      menuTrigger?.focus();
    }, reduceMotion.matches ? 0 : 280);
  }

  toggle?.addEventListener("click", () => {
    if (toggle.getAttribute("aria-expanded") === "true") closeMenu();
    else openMenu();
  });
  closeButton?.addEventListener("click", closeMenu);
  backdrop?.addEventListener("click", closeMenu);
  menu?.querySelectorAll("a").forEach((link) => link.addEventListener("click", closeMenu));
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && toggle?.getAttribute("aria-expanded") === "true") closeMenu();
  });

  setHeaderState();
  window.addEventListener("scroll", setHeaderState, { passive: true });

  const section = document.querySelector("[data-belief-section]");
  const arc = document.querySelector("[data-belief-arc]");
  const leftCloud = document.querySelector("[data-cloud-left]");
  const rightCloud = document.querySelector("[data-cloud-right]");
  const video = document.querySelector(".hero-video");

  if (reduceMotion.matches) {
    video?.pause();
    return;
  }

  let currentArcY = 110;
  let currentLeftX = -200;
  let currentRightX = 200;
  let currentCloudY = 0;
  let targetArcY = 110;
  let targetLeftX = -200;
  let targetRightX = 200;
  let targetCloudY = 0;
  let frame = 0;

  const lerp = (current, target, factor) => current + (target - current) * factor;
  const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

  function updateTargets() {
    if (!section) return;
    const rect = section.getBoundingClientRect();
    const progress = clamp((window.innerHeight - rect.top) / (window.innerHeight + rect.height), 0, 1);
    const cloudProgress = clamp((progress - 0.12) / 0.8, 0, 1);
    targetArcY = 120 + (-280 * progress);
    targetLeftX = -200 + (200 * cloudProgress);
    targetRightX = 200 - (200 * cloudProgress);
    targetCloudY = progress * -50;
    if (!frame) frame = requestAnimationFrame(animate);
  }

  function animate() {
    currentArcY = lerp(currentArcY, targetArcY, 0.06);
    currentLeftX = lerp(currentLeftX, targetLeftX, 0.05);
    currentRightX = lerp(currentRightX, targetRightX, 0.05);
    currentCloudY = lerp(currentCloudY, targetCloudY, 0.05);

    if (arc) arc.style.transform = `translate3d(0, ${currentArcY}px, 0)`;
    if (leftCloud) {
      leftCloud.style.transform = `translate3d(${currentLeftX}px, ${currentCloudY}px, 0)`;
      leftCloud.style.opacity = String(clamp(1 - Math.abs(currentLeftX) / 210, 0, 0.72));
    }
    if (rightCloud) {
      rightCloud.style.transform = `translate3d(${currentRightX}px, ${currentCloudY}px, 0) scaleX(-1)`;
      rightCloud.style.opacity = String(clamp(1 - Math.abs(currentRightX) / 210, 0, 0.72));
    }

    const moving = Math.abs(currentArcY - targetArcY) > 0.1
      || Math.abs(currentLeftX - targetLeftX) > 0.1
      || Math.abs(currentRightX - targetRightX) > 0.1;
    frame = moving ? requestAnimationFrame(animate) : 0;
  }

  updateTargets();
  window.addEventListener("scroll", updateTargets, { passive: true });
  window.addEventListener("resize", updateTargets);
})(window);
