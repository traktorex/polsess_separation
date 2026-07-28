/* Home: upload (drag-and-drop + picker) and the recent-jobs list.

   Picking a file does NOT start the run: the choice lands in a pending card
   that shows what the server is about to receive (name, size, duration probed
   from the file itself) and waits for an explicit "Start". A wrong drag is a
   click to undo, not a GPU minute to wait out. */

import { el, clear, card } from "./dom.js";
import { createJob, deleteJobs, getJSON } from "./api.js";
import { clock, num, plural, statusLabel, when } from "./format.js";

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

/** The pending-file card: what we are about to send, and the two ways out.
 *  Duration is probed locally through an object URL — the server also probes
 *  it at upload, this is only so the user can sanity-check the file first. */
function pendingCard(file, { onStart, onCancel }) {
  const durationLabel = el("span", { class: "mono", text: "…" });
  const startBtn = el("button", { class: "btn primary", type: "button", text: "Start" });
  const cancelBtn = el("button", { class: "btn", type: "button", text: "Anuluj" });
  const errorLine = el("div", { class: "lead hidden" });

  const node = el("div", { class: "pending" }, [
    el("div", { class: "pmeta" }, [
      el("span", { class: "filename", text: file.name }),
      el("span", { class: "chip" }, [el("b", { text: `${num(file.size / (1024 * 1024), 1)} MB` })]),
      el("span", { class: "chip" }, [durationLabel]),
    ]),
    el("div", { class: "pactions" }, [startBtn, cancelBtn]),
    errorLine,
  ]);

  // -- local duration probe ---------------------------------------------
  let objectUrl = null;
  const probe = el("audio", { preload: "metadata" });
  probe.style.display = "none";
  const releaseUrl = () => {
    if (!objectUrl) return;
    URL.revokeObjectURL(objectUrl);
    objectUrl = null;
  };
  try {
    objectUrl = URL.createObjectURL(file);
    probe.addEventListener("loadedmetadata", () => {
      durationLabel.textContent =
        isFinite(probe.duration) && probe.duration > 0 ? clock(probe.duration) : "—";
      releaseUrl();
    });
    probe.addEventListener("error", () => {
      durationLabel.textContent = "—";
      releaseUrl();
    });
    probe.src = objectUrl;
    node.appendChild(probe);
  } catch (err) {
    durationLabel.textContent = "—";
  }

  const teardown = () => {
    probe.removeAttribute("src");
    releaseUrl();
  };

  startBtn.addEventListener("click", async () => {
    startBtn.disabled = true;
    cancelBtn.disabled = true;
    startBtn.textContent = "Wysyłanie…";
    errorLine.classList.add("hidden");
    try {
      await onStart(file);
    } catch (err) {
      startBtn.disabled = false;
      cancelBtn.disabled = false;
      startBtn.textContent = "Start";
      errorLine.classList.remove("hidden");
      errorLine.textContent = `Nie udało się przyjąć pliku: ${err.message}`;
    }
  });
  cancelBtn.addEventListener("click", () => {
    teardown();
    onCancel();
  });

  return { node, teardown };
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
  const pendingBox = el("div");

  // -- upload -------------------------------------------------------------
  let pending = null;
  const dropPending = () => {
    if (!pending) return;
    pending.teardown();
    pending = null;
    clear(pendingBox);
  };

  const upload = uploadCard((file) => {
    dropPending();
    pending = pendingCard(file, {
      onStart: async (chosen) => {
        const jobId = await createJob({ file: chosen });
        window.location.href = `/j/${jobId}`;
      },
      onCancel: dropPending,
    });
    pendingBox.appendChild(pending.node);
  });

  root.appendChild(card("Nowe nagranie", [
    upload.node,
    pendingBox,
    el("p", {
      class: "lead",
      text: "Pipeline jest przygotowany na polską mowę i dwóch mówców w jednym kanale. "
          + "Nagrania dłuższe niż 4 minuty przechodzą na strumieniowy model diaryzacji — zobaczysz o tym ostrzeżenie.",
    }),
  ]));

  // -- recent jobs --------------------------------------------------------
  const list = el("div", { class: "joblist" }, [el("div", { class: "lead", text: "Wczytywanie…" })]);
  const clearNote = el("span", { class: "clearnote" });
  const clearBtn = el("button", { class: "quietbtn", type: "button", text: "Wyczyść historię" });
  const jobsCard = card("Ostatnie zadania", [list]);
  const heading = jobsCard.querySelector("h2");
  heading.classList.add("h2row");
  heading.appendChild(el("span", { class: "h2gap" }));
  heading.appendChild(clearNote);
  heading.appendChild(clearBtn);
  root.appendChild(jobsCard);

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

  clearBtn.addEventListener("click", async () => {
    if (!window.confirm("Usunąć wszystkie zakończone zadania i ich pliki z dysku?")) return;
    clearBtn.disabled = true;
    clearNote.textContent = "";
    try {
      const { removed, skipped } = await deleteJobs(jobsApi);
      lastSignature = "";
      await refresh();
      let text = `Usunięto ${removed} ${plural(removed, "zadanie", "zadania", "zadań")}.`;
      if (skipped > 0) text += ` Pominięto ${skipped} (w toku).`;
      clearNote.textContent = text;
    } catch (err) {
      clearNote.textContent = `Nie udało się wyczyścić: ${err.message}`;
    } finally {
      clearBtn.disabled = false;
    }
  });

  refresh();
  setInterval(refresh, 5000);
}
