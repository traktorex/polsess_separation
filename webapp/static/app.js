/* Entry point: pick the page module from `#app[data-page]` and mount it.

   Every template ships the same three things (top bar, mount div, this
   module); all page logic lives in ES modules. No build step, no external
   resources — the app works with the machine fully offline. */

import { initTheme, themeButton } from "./theme.js";
import { el } from "./dom.js";
import * as home from "./home.js";
import * as job from "./job.js";
import * as examples from "./examples.js";

initTheme();

const PAGES = { index: home, job: job, examples: examples };

function boot() {
  const slot = document.getElementById("theme-slot");
  if (slot) slot.appendChild(themeButton());

  const root = document.getElementById("app");
  if (!root) return;
  const page = PAGES[root.dataset.page];
  if (!page) return;
  try {
    page.mount(root);
  } catch (err) {
    root.appendChild(el("div", { class: "card errcard" }, [
      el("h2", { text: "Błąd interfejsu" }),
      el("div", { class: "emsg", text: String(err && err.message ? err.message : err) }),
    ]));
    throw err;
  }
}

if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
else boot();
