/* The job page: one URL, progress morphing into results.

   Polls `GET /api/jobs/{id}` about once a second while the job is queued or
   running, with a guard against overlapping requests and a back-off (plus a
   visible notice) after three consecutive failures. Polling stops the moment
   the job reaches a terminal state. */

import { el, clear, chip } from "./dom.js";
import { getJSON } from "./api.js";
import { secs } from "./format.js";
import { createProgressView } from "./progress.js";
import { renderResults } from "./results.js";

const POLL_MS = 1000;
const POLL_MS_SLOW = 2000;
const ERRORS_BEFORE_BACKOFF = 3;

export function mount(root) {
  const pollUrl = root.dataset.pollUrl;
  const logUrl = root.dataset.logUrl;

  clear(root);
  const progress = createProgressView();
  const resultsMount = el("div");
  root.appendChild(progress.node);
  root.appendChild(resultsMount);

  let inflight = false;
  let errors = 0;
  let delay = POLL_MS;
  let stopped = false;
  let view = null;

  function showResults(state) {
    progress.destroy();
    progress.node.remove();
    if (view) view.destroy();
    const chips = [];
    if (state.elapsed_s) chips.push(chip(secs(state.elapsed_s), "przetwarzania"));
    view = renderResults(resultsMount, {
      result: state.result,
      title: state.filename,
      chips,
      banners: state.warnings || [],
      logUrl,
      stagesFallback: state.stages,
    });
  }

  function apply(state) {
    if (state.status === "done" && state.result) {
      stopped = true;
      showResults(state);
      return;
    }
    if (state.status === "failed") stopped = true;
    progress.update(state, logUrl);
  }

  async function poll() {
    if (inflight || stopped) return;
    inflight = true;
    try {
      const state = await getJSON(pollUrl);
      if (errors) progress.setNote("");
      errors = 0;
      delay = POLL_MS;
      apply(state);
    } catch (err) {
      errors += 1;
      if (errors >= ERRORS_BEFORE_BACKOFF) {
        delay = POLL_MS_SLOW;
        progress.setNote(`Brak połączenia z serwerem (${errors} nieudanych prób) — ponawiam co 2 s.`);
      }
    } finally {
      inflight = false;
    }
  }

  async function loop() {
    while (!stopped) {
      await poll();
      if (stopped) break;
      await new Promise((resolve) => setTimeout(resolve, delay));
    }
  }

  loop();
}
