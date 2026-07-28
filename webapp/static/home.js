/* Home: upload (drag-and-drop + picker) and the recent-jobs list. */

import { el, clear, card } from "./dom.js";
import { createJob, getJSON } from "./api.js";
import { clock, statusLabel, when } from "./format.js";

const ACCEPT = ".wav,.mp3,.flac,.m4a,.ogg,.opus,.webm,.mp4,audio/*";

function uploadCard(onFile) {
  const input = el("input", { type: "file", accept: ACCEPT, id: "file-input" });
  const pick = el("button", { class: "btn primary", type: "button", text: "Wybierz plik" });
  const zone = el("div", { class: "drop" }, [
    el("div", { class: "big", text: "Przeciągnij nagranie tutaj" }),
    el("div", { class: "hint", text: "albo wskaż plik — WAV, MP3, FLAC, M4A, OGG. Serwer konwertuje wejście do 16 kHz mono." }),
    pick,
    input,
  ]);

  pick.addEventListener("click", () => input.click());
  input.addEventListener("change", () => {
    if (input.files && input.files[0]) onFile(input.files[0]);
    input.value = "";
  });
  for (const type of ["dragenter", "dragover"]) {
    zone.addEventListener(type, (event) => {
      event.preventDefault();
      zone.classList.add("over");
    });
  }
  for (const type of ["dragleave", "dragend"]) {
    zone.addEventListener(type, () => zone.classList.remove("over"));
  }
  zone.addEventListener("drop", (event) => {
    event.preventDefault();
    zone.classList.remove("over");
    const files = event.dataTransfer && event.dataTransfer.files;
    if (files && files[0]) onFile(files[0]);
  });

  return { node: zone, pick };
}

function jobRow(row) {
  const status = String(row.status || "");
  return el("div", { class: "row" }, [
    el("div", { class: "nm" }, [
      el("a", { href: `/j/${row.id}`, text: row.filename || row.id }),
      row.audio_duration_s
        ? el("span", { class: "muted", text: `  ·  ${clock(row.audio_duration_s)}` })
        : null,
    ]),
    el("span", { class: `pill ${status}`, text: statusLabel(status) }),
    el("span", { class: "when", text: when(row.submitted_at) }),
  ]);
}

export function mount(root) {
  clear(root);
  const status = el("div", { class: "lead", style: "margin-top:14px" });
  const errorBox = el("div");

  const upload = uploadCard(async (file) => {
    clear(errorBox);
    upload.pick.disabled = true;
    status.textContent = `Wysyłanie: ${file.name}…`;
    try {
      const jobId = await createJob({ file });
      status.textContent = "Przyjęto — otwieram stronę zadania…";
      window.location.href = `/j/${jobId}`;
    } catch (err) {
      status.textContent = "";
      upload.pick.disabled = false;
      errorBox.appendChild(el("div", { class: "card errcard" }, [
        el("h2", { text: "Nie udało się przyjąć pliku" }),
        el("div", { class: "emsg", text: err.message }),
      ]));
    }
  });

  root.appendChild(card("Nowe nagranie", [
    upload.node,
    status,
    el("p", {
      class: "lead",
      text: "Pipeline jest przygotowany na polską mowę i dwóch mówców w jednym kanale. "
          + "Nagrania dłuższe niż 4 minuty przechodzą na strumieniowy model diaryzacji — zobaczysz o tym ostrzeżenie.",
    }),
  ]));
  root.appendChild(errorBox);

  const list = el("div", { class: "joblist" }, [el("div", { class: "lead", text: "Wczytywanie…" })]);
  root.appendChild(card("Ostatnie zadania", [list]));

  const jobsApi = root.dataset.jobsApi || "/api/jobs";
  let lastSignature = "";
  const refresh = async () => {
    try {
      const rows = await getJSON(`${jobsApi}?limit=10`);
      const signature = JSON.stringify(rows);
      if (signature === lastSignature) return;
      lastSignature = signature;
      clear(list);
      if (!rows.length) {
        list.appendChild(el("div", { class: "lead", text: "Jeszcze nic tu nie ma — wgraj pierwsze nagranie." }));
        return;
      }
      for (const row of rows) list.appendChild(jobRow(row));
    } catch (err) {
      clear(list);
      list.appendChild(el("div", { class: "lead", text: `Nie udało się pobrać listy zadań: ${err.message}` }));
    }
  };
  refresh();
  setInterval(refresh, 5000);
}
