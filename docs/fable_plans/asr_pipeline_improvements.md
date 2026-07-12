# ASR Pipeline Improvements — Plan

Drafted 2026-07-12 (Claude Fable 5, from a survey session; conversation export:
`pipeline_improvements.md` at repo root). Status: **proposed, not started**.
Read `asr_pipeline/SCOPE.md` before executing any part of this — every item
below is designed to fit inside it (build for Life 1, don't block Life 2;
no platform-grade engineering; UNDECIDED items stay the author's).

Sites are named by symbol, not line number (SCOPE §5 convention).

---

## 0. Findings the plan is built on (survey 2026-07-12)

**What the package already owns (good):**
- Single-recording orchestration: `Pipeline` with phase-major stage execution,
  `load_signature()`/`_ensure_loaded` reload-skipping, one-model-on-GPU.
- Output schema: `io.write_pipeline_outputs` (stable per-recording layout).
- Config baseline policy: `eval/config_presets.fresh_eval_cfg` (already
  extracted so drivers can't drift).
- Unified metric set: `eval/metrics.per_fragment_metrics` (one home for the
  meeteval-call loop; cpWER/tcpWER/cpCER/ORC/MIMO/orccer + mixture floors).
- Load-once discipline on the *scoring* side: `eval/run.evaluate_many`
  (SQUIM loaded once, shared across recordings).

**Gaps found:**
1. **CLI is a stub.** `asr_pipeline/__main__.py` exposes only `run` with
   `--config/--input/--output`; it does not call `write_pipeline_outputs`,
   does not score, has no knob overrides, no batch, no preflight.
2. **Preflight lives only in a notebook cell** (`asr/explore_pipeline.ipynb`,
   the cell after config build). A CLI run discovers a missing
   `$SORTFORMER_VENV_PY` only after minutes of model loading — SCOPE §4
   (fail loud) satisfied late instead of early.
3. **The batch loop is copy-pasted three times** with diverging semantics:
   - `scripts/sweep_pipeline.py` `run_config` — config-major outer loop,
     per-recording `except Exception` → batch continues; skip sentinel =
     `<id>/sweep/<name>/transcript_A.txt` exists.
   - `scripts/run_pipeline_on_recording.py` `run_one_mode` — **no** per-mode
     exception containment (a failure aborts remaining modes — violates
     SCOPE §4.2 batch semantics); skip = target dir non-empty.
   - `scripts/batch_pipeline_noenh.py` `_run_one` — third copy, own sentinel,
     hardcoded `EVAL_ROOT` (ignores `eval_harness.eval_root()`), and imports
     `_fresh_cfg` through `scripts.run_pipeline_on_recording` instead of the
     package's `fresh_eval_cfg` (the exact cross-script hack the extraction
     was meant to end).
   The GPU teardown block (`unload / del / gc.collect / empty_cache`) is
   copy-pasted in all three; `Pipeline.run` only frees the final stage.
4. **Models reload per recording per config arm.** Every driver builds a
   fresh `Pipeline(cfg)` per recording; the interactive API that keeps a
   stage's model resident across contexts is never used by any batch driver.
5. **No cross-config reuse.** ASR-only sweep arms recomputed byte-identical
   upstream streams (diarization/enh/sep) dozens of times. The
   "hold streams fixed, vary the transcriber" pattern exists only in
   `scripts/compare_asr.py` and was never generalized.
6. **Timing is coarse.** `run_meta.json` = total seconds only; load-vs-inference
   attribution is impossible (this fed the 2026-07 timing-mirage incident).
7. **`eval/layer3.py` does not compute cpCER** — the campaign's primary
   metric. `compute_layer3` calls `cpwer_meeteval` directly instead of
   `per_fragment_metrics`, so `evaluate_recording()` / `summarize_layer3()`
   produce a table missing the headline number. cpCER only flows through the
   sweep script's own scorer.
8. **The campaign statistics are trapped in a script.**
   Recording-clustered bootstrap CIs, paired deltas, Holm-Bonferroni,
   Benjamini-Hochberg, strata handling — all private functions in
   `scripts/rescore_stratified.py` (~600 lines), zero unit tests.
9. **Three scorers over the same tree** (`sweep_pipeline.score_configs`,
   `rescore_stratified`, `dump_sweep_results`); the sweep's own bootstrap
   helpers are self-labeled "descriptive only, not for inference" —
   superseded, deletable under the lifetime rule (SCOPE §6).
10. **Ablation-mode logic exists in three encodings**: `MODES` setattr-lambdas
    (`run_pipeline_on_recording`), `CONFIGS` dotted-override rows (sweep),
    and hand-rolled `enhancement.enabled=False` (batch_noenh). Two override
    mechanisms for one job.

---

## Work Package A — CLI front-end + Tier-1 batch consolidation  **[do first]**

Goal: `python -m asr_pipeline` becomes the pipeline's front door, in the
spirit of `python train.py --config ...`. The notebook stays a first-class
consumer; scripts become thin wrappers or die. No pipeline behavior changes;
the shipped best config (`configs/sweep_best_e31_refineplus.yaml`) untouched.

### A1. `--set` dotted overrides (package-level)
- Promote the sweep's `_apply` (dotted-path walker, fails loud on unknown
  leaf via `AttributeError`, re-runs `__post_init__`) into
  `asr_pipeline/config.py` as `apply_overrides(cfg, {"stage.knob": value})`.
- `run`/`batch` accept repeated `--set stage.knob=value` (YAML-parse the
  value string for typing: `true`, `0.5`, quoted strings).
- Sweep script switches to importing it (one override mechanism, one
  fail-loud policy).

### A2. Preflight extraction
- New `asr_pipeline/preflight.py`: `preflight(cfg) -> list[str]` of failures;
  raise on any (SCOPE §4). Port the notebook preflight cell's checks:
  `$SORTFORMER_VENV_PY` when `diarization.backend == "sortformer"`,
  `$HF_TOKEN` for pyannote, `$AP_BWE_CHECKPOINT` when
  `post_separation_processing.backend == "ap_bwe"`, `$COHEREX_VENV_PY` when
  `transcription.backend == "coherex"`, separator checkpoint path exists,
  `num2words` importability (warn-only — hard-dep ruling is SCOPE §10 q2,
  author's).
- Called automatically at the top of CLI `run` and `batch`, **before** any
  model load. Notebook cell becomes a one-line call.

### A3. `run` subcommand upgrade
- Keep current flags; add `--set`, and `--write-outputs <eval_root>`
  (or `--out-dir`) that calls `write_pipeline_outputs` + copies the mixture
  to `<id>/<id>.wav` (the eval-discovery requirement currently hand-rolled
  in the notebook save cell).
- Print per-stage progress + timings (see A6).

### A4. `batch` subcommand + `asr_pipeline/batch.py`
- `run_batch(cfg, recordings, out_root, subdir_name, *, skip_existing,
  on_event) -> BatchReport`; the CLI feeds it a manifest file, a directory
  glob, or a split name (`clarin_dev`/`clarin_test` via
  `eval/clarin_split.csv` — reuse `eval_harness` loaders).
- Semantics per SCOPE §4.2: one recording's failure is caught, recorded
  (`failures.csv` with traceback summary), batch continues. Centralize the
  GPU teardown block here — one home.
- **One skip sentinel**: completion = `metadata.json` present in the target
  subdir (it is written last by `write_pipeline_outputs`), not
  `transcript_A.txt`. `--force` re-runs.
- Ablation modes become config presets applied via `apply_overrides`
  (kills the `MODES` lambda encoding): `--mode full|no_sep|no_enh|minimal`.
- `run_meta.json` gains per-stage seconds (A6) alongside total.

### A5. `score` subcommand
- Walk the eval tree (`walk_eval_tree`), run `evaluate_many`, emit the
  summary tables (`summarize_layer2_*`, `summarize_layer3`) as CSV.
  Notebook-free scoring for supervisors.

### A6. Stage progress/timing events
- `Pipeline` gets an optional `on_event` callback (stage_start/stage_end with
  wall seconds, load seconds vs run seconds split). Consumed by CLI progress
  output and written into `metadata.json` / `run_meta.json`.
- This is the instrumentation that decides Work Package B Tier 2 — and it
  retires the timing-mirage class of bugs (stale `run_meta` timings).

### A7. Script consolidation (after A4 lands)
- `batch_pipeline_noenh.py`: delete (fully subsumed).
- `run_pipeline_on_recording.py`: thin wrapper over `run_batch` with the
  four modes, or delete once callers migrate.
- `sweep_pipeline.py`: keeps its registry/GROUPS/scoring identity but its
  run loop delegates to `run_batch`.
- Candidates to review for deletion (lifetime rule): `transcribe_mixtures.py`,
  `sweep_asr_decode.py`, `sweep_asr_prompt.py` (arms now expressible as
  `CONFIGS` rows). Author call.

Effort: 1–2 focused sessions. Tests: extend `tests/test_pipeline_config.py`
(overrides), new `test_pipeline_preflight.py`, `test_pipeline_batch.py`
(sentinel, failure isolation, ledger) with stage stubs — the orchestrator
tests already show the stubbing pattern.

---

## Work Package C — Evaluation module  **[do second; C1+C2 are the core]**

### C1. cpCER (+ full metric set) into layer3
- Route `compute_layer3` through `eval/metrics.per_fragment_metrics` instead
  of direct `cpwer_meeteval` calls; surface `cpcer` (and keep existing keys
  byte-compatible for `summarize_layer3` / notebook consumers).
- `summarize_layer3` gains `full_cpcer` / per-mode cpCER columns.
- Guard: the ORC/MIMO combinatorial blow-up cap from the explore notebook's
  scoring cell (DP-table size estimate → skip with a printed note) belongs in
  the package with the metrics, not in a notebook cell — fold it into
  `per_fragment_metrics` or a thin wrapper.

### C2. `eval/stats.py` — extract the campaign statistics
- Move from `scripts/rescore_stratified.py`: recording-clustered paired
  bootstrap (`cluster_boot_paired*`, `_boot_ci`, `boot_pvalue`),
  `holm_bonferroni`, `benjamini_hochberg`, micro-average helpers, strata
  loading. Public, documented, unit-tested (fixed-seed golden tests).
- `rescore_stratified.py` becomes a thin driver. Thesis tables become
  regenerable through package code.

### C3. ScoreCard caching (optional)
- L2 chunked PESQ/STOI/SQUIM is the slow half and recomputes every notebook
  run. Cache per (recording id, mode, GT hash, metric version) → parquet
  under the eval root; follow the provenance-hash pattern of the sweep
  ledger (`_gt_snapshot_hash`, `_configs_hash`).

### C4. Scorer consolidation (optional)
- Delete the sweep's superseded descriptive bootstrap
  (`bootstrap_microavg_ci`, `bootstrap_paired_diff_ci`) once C2 exists;
  `score_configs` and `dump_sweep_results` share one scoring path.

### C5. NOT in scope without author ruling
- `num2words` hard-dep (SCOPE §10 q2 — UNDECIDED). Preflight warns only.

Effort: C1 small; C2 medium (the tests are most of it); C3/C4 small each.

---

## Work Package B — Batch rework Tier 2  **[decide only after A6 timing data]**

### B1. Stage-major batch executor (the "major rework")
- Invert the loop: load diarizer once → diarize all pending recordings →
  swap to enhancer → ... Amortizes ~7 model loads across N recordings
  instead of 7×N. Directly enabled by existing `load_signature` /
  `_ensure_loaded` machinery; `evaluate_many` is the in-repo precedent.
- Costs to solve: N contexts in RAM or a spill/restore format (current
  spill is write-only debugging output); (recording, stage)-granular
  failure isolation and resume; more complex progress reporting.
- **Gate:** only build if A6 timings show model-load is a dominant share of
  batch wall time. Campaign test budget is spent — the payoff is future
  sweeps and Life-2 throughput, not thesis numbers.

### B2. Cross-config stream reuse (cheaper, most of the win for sweeps)
- Generalize `compare_asr.py`'s fixed-audio ASR-only swap: hash the
  upstream-relevant config slice (everything above the varied stage);
  arms sharing the hash reuse `stream_{A,B}.wav` + upstream JSONs from the
  first arm that produced them. Targets the single biggest observed waste
  (ASR/decode-only arms) without inverting the loop.
- Needs a principled config-slice partition per stage boundary — document
  which keys feed which stage (SWEEP_KNOBS.md already maps this).

### B3. Optional hardening
- `--isolate`: subprocess-per-recording for crash containment. Real observed
  trigger exists (CUDA context corruption, e14aa22f OOM 2026-07-06) —
  satisfies SCOPE §7's "real observed trigger" rule. Off by default.
- `--shard i/N`: trivial deterministic sharding for splitting a batch across
  the two machines. No coordination, no queue (that would be Life-2 platform
  work — out of scope).

---

## Explicitly out of scope (SCOPE §1/§9)

- Job queues, REST APIs, service daemons, auth — Life-2 deployment work,
  starts as its own effort.
- Gradio demo app — nice Life-2-flavored follow-up (POC heritage in
  `asr/archive/` makes it cheap), but the CLI is the spine; build after A.
- N-speaker support, new backends, new fallback branches (any new fallback
  joins the SCOPE §5 ledger or doesn't merge).

## Sequencing

1. **A** (CLI + preflight + `run_batch` + timings) — one coherent block.
2. **C1 + C2** (cpCER into layer3; stats extraction) — small, high thesis value.
3. Re-evaluate **B** with A6 timing data: pick B2 (stream reuse) and/or B1
   (stage-major) only if the numbers justify them. B3 flags are cheap add-ons
   to A4 whenever wanted.
