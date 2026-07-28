/* The examples gallery: frozen v41_merge results, GT and scores (which appear
   only here), plus "uruchom ponownie" — the same recording submitted as an
   ordinary new job, with no GT and no scoring on that path (design §5.5).

   The detail view reuses `renderResults`, the exact renderer the job page
   uses; only the data source differs. */

import { el, clear, card, chip } from "./dom.js";
import { getJSON, createJob } from "./api.js";
import { clock, num, plural } from "./format.js";
import { renderResults, scoreChips } from "./results.js";

let rows = [];
let view = null;
let examplesApi = "/api/examples";

function rerunButton(exampleId, klass = "btn") {
  const button = el("button", { class: klass, type: "button", text: "Uruchom ponownie" });
  button.addEventListener("click", async (event) => {
    event.stopPropagation();
    button.disabled = true;
    const original = button.textContent;
    button.textContent = "Wysyłanie…";
    try {
      const jobId = await createJob({ sourceExample: exampleId });
      window.location.href = `/j/${jobId}`;
    } catch (err) {
      button.disabled = false;
      button.textContent = original;
      window.alert(`Nie udało się uruchomić zadania: ${err.message}`);
    }
  });
  return button;
}

function exampleCard(row, onOpen) {
  const node = el("div", {
    class: "excard", role: "button", tabindex: "0",
    "aria-label": `Przykład ${row.id}`,
  }, [
    el("div", { class: "exid", text: row.id }),
    el("div", { class: "exrow" }, [
      el("span", { class: `badge ${row.split || ""}`, text: String(row.split || "").toUpperCase() }),
      el("span", { class: "muted", style: "font-size:12.5px", text: clock(row.duration_s) }),
      el("span", {
        class: "muted", style: "font-size:12.5px",
        text: `${row.n_overlap_regions || 0} ${plural(row.n_overlap_regions || 0, "region", "regiony", "regionów")}`,
      }),
    ]),
    el("div", { class: "exrow" }, [
      row.cpwer === null || row.cpwer === undefined ? null : chip(num(row.cpwer, 1), "cpWER"),
      row.cpcer === null || row.cpcer === undefined ? null : chip(num(row.cpcer, 1), "cpCER"),
    ]),
    el("div", { class: "exrow" }, [rerunButton(row.id, "btn small")]),
  ]);
  node.addEventListener("click", () => onOpen(row.id));
  node.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      onOpen(row.id);
    }
  });
  return node;
}

function renderList(root, onOpen) {
  clear(root);
  if (view) { view.destroy(); view = null; }

  const search = el("input", { type: "search", placeholder: "Filtruj po identyfikatorze…", "aria-label": "Filtr" });
  const split = el("select", { "aria-label": "Podzbiór" }, [
    el("option", { value: "", text: "wszystkie" }),
    el("option", { value: "dev", text: "DEV" }),
    el("option", { value: "test", text: "TEST" }),
  ]);
  const count = el("span", { class: "muted", style: "font-size:12.5px" });
  const grid = el("div", { class: "exgrid" });

  const paint = () => {
    const needle = search.value.trim().toLowerCase();
    const wanted = split.value;
    clear(grid);
    let shown = 0;
    for (const row of rows) {
      if (wanted && row.split !== wanted) continue;
      if (needle && !String(row.id).toLowerCase().includes(needle)) continue;
      grid.appendChild(exampleCard(row, onOpen));
      shown += 1;
    }
    count.textContent = `${shown} z ${rows.length}`;
    if (!shown) grid.appendChild(el("div", { class: "lead", text: "Nic nie pasuje do filtra." }));
  };
  search.addEventListener("input", paint);
  split.addEventListener("change", paint);

  root.appendChild(card("Przykłady — wyniki zamrożone (v41_merge)", [
    el("p", { class: "lead", style: "margin-top:0" , text:
      "Gotowe wyniki fragmentów CLARIN: te same liczby, które raportuje praca. "
      + "Referencyjna transkrypcja i metryki cpWER/cpCER pokazywane są wyłącznie tutaj. "
      + "„Uruchom ponownie” wysyła to samo nagranie jako zwykłe nowe zadanie — bez GT i bez metryk." }),
    el("div", { class: "exfilter" }, [search, split, count]),
    grid,
  ]));
  paint();
}

async function renderDetail(root, exampleId, onBack) {
  clear(root);
  if (view) { view.destroy(); view = null; }
  const head = el("div", { class: "exhead" }, [
    el("button", { class: "backlink", type: "button", text: "← wszystkie przykłady", onclick: onBack }),
    el("span", { class: "spacer" }),
    rerunButton(exampleId, "btn primary"),
  ]);
  root.appendChild(head);
  const body = el("div", {}, [el("div", { class: "lead", text: "Wczytywanie przykładu…" })]);
  root.appendChild(body);

  let payload;
  try {
    payload = await getJSON(`${examplesApi}?ids=${encodeURIComponent(exampleId)}`);
  } catch (err) {
    clear(body);
    body.appendChild(el("div", { class: "card errcard" }, [
      el("h2", { text: "Nie udało się wczytać przykładu" }),
      el("div", { class: "emsg", text: err.message }),
    ]));
    return;
  }
  const row = (payload || [])[0];
  if (!row || !row.job_like) {
    clear(body);
    body.appendChild(el("div", { class: "card" }, [
      el("div", { class: "lead", text: `Brak zamrożonego wyniku dla przykładu ${exampleId}.` }),
    ]));
    return;
  }
  clear(body);
  view = renderResults(body, {
    result: row.job_like,
    title: row.title || row.id,
    chips: scoreChips(row),
    gt: row.gt || null,
  });
}

export function mount(root) {
  examplesApi = root.dataset.examplesApi || "/api/examples";
  const showList = () => {
    renderList(root, (id) => {
      history.pushState({ id }, "", `#${id}`);
      renderDetail(root, id, () => { history.pushState({}, "", "#"); showList(); });
    });
  };

  const openFromHash = () => {
    const id = decodeURIComponent((location.hash || "").replace(/^#/, ""));
    if (id && rows.some((row) => row.id === id)) {
      renderDetail(root, id, () => { history.pushState({}, "", "#"); showList(); });
    } else {
      showList();
    }
  };

  clear(root);
  root.appendChild(el("div", { class: "lead", text: "Wczytywanie galerii…" }));
  getJSON(`${examplesApi}?light=1`)
    .then((payload) => {
      rows = payload || [];
      openFromHash();
    })
    .catch((err) => {
      clear(root);
      root.appendChild(el("div", { class: "card errcard" }, [
        el("h2", { text: "Nie udało się wczytać galerii" }),
        el("div", { class: "emsg", text: err.message }),
      ]));
    });

  window.addEventListener("popstate", () => {
    if (rows.length) openFromHash();
  });
}
