/* Light/dark: the OS preference by default, overridable for a projector.
   The choice is stamped on <html data-theme>, which the stylesheet's
   [data-theme] blocks win with; canvas painters watch the same attribute. */

const KEY = "polsess-theme";
const ORDER = ["auto", "light", "dark"];
const LABEL = { auto: "◐ auto", light: "☀ jasny", dark: "☾ ciemny" };

function stored() {
  try {
    const value = localStorage.getItem(KEY);
    return ORDER.includes(value) ? value : "auto";
  } catch (err) {
    return "auto";
  }
}

function apply(mode) {
  if (mode === "auto") document.documentElement.removeAttribute("data-theme");
  else document.documentElement.setAttribute("data-theme", mode);
}

/** Apply the stored preference. Safe to call before the DOM is ready. */
export function initTheme() {
  apply(stored());
}

/** A small cycling button for the top bar. */
export function themeButton() {
  const button = document.createElement("button");
  button.className = "themebtn";
  button.type = "button";
  button.title = "Motyw: auto / jasny / ciemny";
  let mode = stored();
  const paint = () => { button.textContent = LABEL[mode]; };
  paint();
  button.addEventListener("click", () => {
    mode = ORDER[(ORDER.indexOf(mode) + 1) % ORDER.length];
    try { localStorage.setItem(KEY, mode); } catch (err) { /* private mode */ }
    apply(mode);
    paint();
  });
  return button;
}

/** Run `callback` whenever the effective theme changes (attribute or OS). */
export function onThemeChange(callback) {
  const observer = new MutationObserver(callback);
  observer.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  const media = window.matchMedia("(prefers-color-scheme: dark)");
  const handler = () => callback();
  if (media.addEventListener) media.addEventListener("change", handler);
  else if (media.addListener) media.addListener(handler);
  return () => {
    observer.disconnect();
    if (media.removeEventListener) media.removeEventListener("change", handler);
    else if (media.removeListener) media.removeListener(handler);
  };
}
