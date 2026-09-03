/**
 * Light/dark: the OS preference by default, overridable for a projector.
 * Port of `webapp/static/theme.js` (Road 1) — same storage key, same three-way
 * cycle, same `<html data-theme>` contract the stylesheet's `[data-theme]`
 * blocks win with. The canvas painters have to repaint on a theme change
 * because their colours come from CSS custom properties.
 */

const KEY = 'polsess-theme';
const ORDER = ['auto', 'light', 'dark'];
const LABEL = { auto: '◐ auto', light: '☀ jasny', dark: '☾ ciemny' };

function stored() {
  try {
    const value = localStorage.getItem(KEY);
    return ORDER.includes(value) ? value : 'auto';
  } catch (err) {
    return 'auto'; // private mode
  }
}

function apply(mode) {
  if (mode === 'auto') document.documentElement.removeAttribute('data-theme');
  else document.documentElement.setAttribute('data-theme', mode);
}

/**
 * Apply the stored preference and wire the existing top-bar button to cycle it.
 * @param {HTMLElement} button
 */
export function initTheme(button) {
  let mode = stored();
  apply(mode);
  button.textContent = LABEL[mode];
  button.addEventListener('click', () => {
    mode = ORDER[(ORDER.indexOf(mode) + 1) % ORDER.length];
    try { localStorage.setItem(KEY, mode); } catch (err) { /* private mode */ }
    apply(mode);
    button.textContent = LABEL[mode];
  });
}

/** Run `callback` whenever the effective theme changes (attribute or OS). */
export function onThemeChange(callback) {
  new MutationObserver(callback).observe(document.documentElement, {
    attributes: true,
    attributeFilter: ['data-theme'],
  });
  const media = window.matchMedia('(prefers-color-scheme: dark)');
  if (media.addEventListener) media.addEventListener('change', callback);
  else if (media.addListener) media.addListener(callback);
}
