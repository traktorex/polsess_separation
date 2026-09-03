# webapp API contract (v1)

Binding contract between `webapp/` backend and frontend. Both are implemented against THIS file;
change it deliberately and update both sides together (design accepted 2026-07-28).

## HTML routes

| Route | Page |
|---|---|
| `GET /` | Home: upload (drag-and-drop + file picker) form, short recent-jobs list, link to examples |
| `GET /j/{job_id}` | Job page — progress morphing into results; client polls the JSON API ~1 Hz |
| `GET /examples` | Pre-processed gallery (frozen v41_merge results + GT + scores; "uruchom ponownie" button per example) |

## JSON API

### `POST /api/jobs`
Multipart form: `file=<audio>`. Also accepts `source_example=<example_id>` instead of a file
(the "uruchom ponownie" path — server copies the example's mixture; NO GT/scoring attaches to the new job).
Non-WAV input is converted to 16 kHz mono WAV via ffmpeg at upload time (before enqueue).
Response `200 {"job_id": str}`. Errors: `400` (no/invalid file, ffmpeg failure — message says why).

### `GET /api/jobs/{id}` → JobState

```jsonc
{
  "id": str,
  "filename": str,                  // original upload name
  "status": "queued" | "running" | "done" | "failed",
  "queue_position": int,            // 0 = running/next; only meaningful while queued
  "submitted_at": str,              // ISO 8601
  "audio_duration_s": float|null,   // probed at upload
  "stages": [                       // exactly the enabled stages, pipeline order (8 for v41_merge)
    { "stage": str,                 // "diarization" | ... | "transcription"
      "state": "pending"|"running"|"done",
      "load_s": float|null, "run_s": float|null,          // from stage_end
      "progress": {"done": int, "total": int} | null }     // from stage_progress events (hook 2)
  ],
  "eta_s": float|null,              // duration-weighted; refined after routing (overlap-aware)
  "elapsed_s": float|null,
  "warnings": [str],                // WARN lines from debug log + weak_anchor + streaming-diarizer swap
  "error": {"type": str, "message": str} | null,   // status=failed only; no partial-success framing
  "partial": null | {               // progressive disclosure (design §5.2): present ONLY while status=running,
                                    // once the per-job spill files exist; null otherwise
    "diarization": {"turns": [{"speaker","start","end"}], "overlaps": [{"start","end","duration"}]} | null,
    "routing": {"overlap_regions": [{"start": float, "end": float}]} | null
  },                                // served from <job>/spill/{diarization.json, overlap_regions.json};
                                    // spill segments are mapped to the eval-facing `turns` shape so the
                                    // frontend consumes ONE shape pre- and post-completion
  "result": null | {
    "speakers": [str], "spk_to_label": {str: str}, "weak_anchor": bool,
    "total_duration_s": float, "n_overlap_regions": int, "overlap_total_s": float,
    "provenance": str,              // compact line built from metadata.config (enh=… · sep=… · asr=…)
    "diarization": {"turns": [{"speaker": str, "start": float, "end": float}]},
    "routing": {"overlap_regions": [{"start": float, "end": float}]},
    "transcripts": { "<label>": {    // label = "A","B",… + optional "mixture"
        "segments": [{"id": str, "start": float, "end": float, "text": str,
                       "words": [{"word": str, "start": float, "end": float, "score": float}]}] } },
    // segments carry stable ids (ELAN-editor future-proofing, design §5.3)
    "files": {                       // URLs under /api/jobs/{id}/files/…
      "mixture": str, "stream_A": str, "stream_B": str,
      "transcript_A_txt": str, "transcript_B_txt": str, "eaf": str, "metadata": str },
    "peaks": {"mixture": [int], "A": [int], "B": [int]},   // ~800 buckets, 0–100 peak envelope (v1 simple form);
                                                            // schema leaves room for additive minmax/rms fields later
    "stage_timings": [{"stage": str, "load_s": float, "run_s": float}],
    "assembly_diag": [ ... ] | null,        // hook 1 payload, verbatim from metadata.json (diagnostics drawer)
    "diarization_diag": { ... } | null      // sortformer census (diagnostics drawer)
  }
}
```

### `GET /api/jobs/{id}/log?offset=N`
Incremental debug-log tail: `{"offset": int, "lines": [str]}` — live during the run (reads
`$ASR_PIPELINE_DEBUG_LOG`, default `/tmp/asr_pipeline_debug.log`, which is one run's worth);
served from the job's copied `debug.log` after completion.

### `GET /api/jobs/{id}/files/{name}`
`FileResponse` (HTTP Range works) from the job's output dir. **Whitelist only**:
`mixture.wav`, `stream_A.wav`, `stream_B.wav`, `transcript_*.txt|json`, `annotation.eaf`,
`metadata.json`, `debug.log`, `enhanced_full.wav` (served from `<job>/spill/` — enhancement
A/B diagnostic panel). Reject anything else (404) — no path traversal.

## v1.1 ratified deltas (orchestrator, 2026-07-28 night)

Implemented additions accepted into the contract:
- `GET /api/jobs?limit=N` — compact recent-jobs rows (feeds the home page list).
- `GET /api/examples/{example_id}/files/{name}` — same whitelist; serves frozen example audio.
- `GET /api/examples?ids=a,b&light=1` — optional filters; `light=1` omits `job_like` for instant gallery render.
- Example rows carry `"gt"` (parsed fragment-level `annotation.eaf` tiers) in addition to `gt_available`;
  `light=1` nulls `gt` too (gallery needs only the flag; the detail `ids=` fetch carries the payload).
- `result.files` carries `"enhanced_full"` when `<job>/spill/enhanced_full.wav` exists (jobs yes, frozen examples no).
- `result.diarization` additionally carries `"overlaps"` (hook 3; `null` when absent).
- **Interpretations (binding):** `eta_s` = seconds REMAINING (monotone-decreasing; full estimate while queued;
  `null` on terminal jobs). `queue_position` = queued jobs ahead (0 = running/next). Log `offset` = LINE index.
- **Per-job spill (binding):** every job runs with `spill_intermediate: true`, `artifact_dir = <job_dir>/spill`
  — the source of `partial` and `enhanced_full.wav`. Deferred-to-later diagnostics (per-overlap separation
  plots, BWE spectrograms) can mine the same spill dir in a future rev without contract changes.

### `GET /api/examples`
`[{"id": str, "title": str, "duration_s": float, "split": "dev"|"test",
   "cpwer": float|null, "cpcer": float|null, "n_overlap_regions": int,
   "gt_available": bool, "job_like": { …same shape as JobState.result… }}]`
Backed by a manifest built once from the frozen eval tree (`~/datasets/eval/clarin_fragments/<frag>/sweep/v41_merge/`);
GT transcripts and scores appear ONLY here.

## Server internals (binding decisions)

- Jobs root: `$WEBAPP_JOBS_ROOT` (default `/home/user/webapp_jobs`); layout = eval-tree-shaped,
  `<root>/<job_id>/<job_id>.wav` + `<root>/<job_id>/pipeline/…` (what `run_batch` writes).
  Job registry rebuilt from disk at startup (status from `metadata.json` sentinel).
- One worker thread + `queue.Queue`, FIFO, concurrency 1. Worker calls
  `asr_pipeline.batch.run_batch(cfg, [(job_id, wav_path)], out_root=jobs_root, subdir="pipeline", on_event=…, skip_existing=False)`.
  The pipeline-facing call sits behind a small `Runner` seam so tests inject a fake.
- Config: `asr_pipeline/configs/sweep_best_e31_refineplus.yaml` loaded once at startup;
  `check_preflight` runs at startup (fail loud, before serving).
- ETA: per-stage table {load_s median (constant), run_s per audio-second median}, seeded from
  constants measured on ~90 s fragments, then updated from every finished job
  (`<jobs_root>/timings.jsonl`); separation/post-sep re-estimated after routing using overlap seconds.
- No silent substitution anywhere (SCOPE §4.1); job failure = error card, batch semantics (§4.2);
  warnings must surface (§4.3).
- Polish UI copy; stage names and load/run stay English. Speaker lanes rendered from `speakers` — no hardcoded 2.

## v1.2 ratified deltas (orchestrator, 2026-07-28 — author feedback round 1)

Ratified after live headless-browser verification. The sections above still describe v1.1 verbatim;
where they conflict (the `~800 buckets` note), this section wins.

### 1. Example rows carry `stratum`, `metrics`, `gt_swapped`

`GET /api/examples` rows gain three fields, served **in both `light=1` and full mode** (they are a few
bytes each, and the gallery filters/sorts on them — unlike `gt`, which stays nulled under `light=1`):

```jsonc
{
  "stratum": "LOW" | "MID" | "HIGH" | null,   // recording-level acoustic-complexity tertile,
                                              // verbatim from the frozen rescore sheets
  "gt_swapped": bool,                         // GT tiers were relabelled to line up with the
                                              // pipeline's stream A/B (display only, see 2.)
  "metrics": null | {                         // all values percent, 1 decimal; null = not available
    "cpwer": float|null, "tcpwer": float|null, "orcwer": float|null,
    "attr_gap": float|null,                   // cpWER − MIMO-WER, in POINTS
    "cpcer": float|null, "mimower": float|null, "mimocer": float|null, "orccer": float|null,
    "floor_orcwer": float|null, "floor_mimower": float|null,
    "floor_cpcer": float|null, "floor_mimocer": float|null
  }
}
```

`floor_*` is the no-pipeline baseline: one Whisper pass over the raw mixture (`transcript_mixture.txt`),
scored against the same GT. `floor_cpcer` is that floor's **time-ordered** merged-reference CER (the
review page's `_mixture_metrics["cer"]`); `floor_mimocer` is its order-forgiving MIMO counterpart.
The whole `metrics` object is `null` for a fragment with no scores at all; individual entries are `null`
when a fragment trips the eval's ORC/MIMO combinatorial blow-up guard and no sheet column covers it.

Provenance (binding): seven of the twelve values come straight from the frozen per-fragment rescore
sheets (`cp_wer`, `cp_cer`, `orc_wer`, `mimo_wer`, `orc_cer`, `mix_mimo_wer`, `mix_mimo_cer`) — those are
the numbers the thesis reports and they always win. The rest are computed at manifest-build time with the
same `asr_pipeline.eval` functions the sheets were produced with, and every overlapping value is
recomputed purely to cross-check the sheet (disagreement > 0.05 pt = loud warning, sheet kept).
Confirmed 0 disagreements over all 141 fragments at the 2026-07-28 rebuild.

### 2. `gt_swapped` is a display alignment, never a score

The pipeline names its two streams without knowing which GT tier is "A", so for 65 of the 141 frozen
fragments its stream A is the GT-B speaker. The manifest detects this (normalised-text similarity,
straight vs crossed pairing, `difflib`) and relabels the two GT tiers in the emitted `gt` so tier A sits
beside stream A. Every metric picks its own optimal speaker assignment and is permutation-invariant, so
**no number changes** — the flag exists so the UI can say why the reference was re-lettered.

### 3. `DELETE /api/jobs`

Removes every **terminal** job (`done` / `failed`): registry entry and `<jobs_root>/<job_id>/` on disk.
Queued and running jobs are never touched. A directory that cannot be removed keeps its registry entry.

Response `200 {"removed": int, "skipped": int}` — `skipped` counts jobs still in flight plus failed
deletions. Idempotent: a second call returns `{"removed": 0, "skipped": 0}`.

### 4. `result.peaks` resolution: 800 → 2400 buckets

`render.DEFAULT_BUCKETS` is now 2400 (the timeline zooms to 8×; 800 buckets read blocky when zoomed).
Shape is unchanged — still a flat list of 0–100 ints per stream, ~3 KB more per stream.
