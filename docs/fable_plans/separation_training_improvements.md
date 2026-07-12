# Separation / Training Codebase Improvements — Plan

Drafted 2026-07-12 (Claude Fable 5, blue-sky survey session; five parallel
Opus subagents over training infra, models, data, eval/tests, and
cross-pollination + hygiene, plus a direct read of `train.py`/`trainer.py`).
Status: **proposed, not started**. Sibling plan: `asr_pipeline_improvements.md`
(same directory) — that one covers `asr_pipeline/`; this one covers the main
speech-separation training/eval side.

Sites are named by symbol, not line number. Priorities are calibrated to the
actual timeline: hand-in ~Aug–Sep 2026, ~1 month of compute left on 2 GPUs,
remaining work = a few training runs (SPMamba full-ks8 curve etc.) + thesis
tables + making the repo presentable to supervisors. Thesis code principles
apply throughout: clarity over cleverness, no speculative abstraction.

---

## 0. Findings the plan is built on (survey 2026-07-12)

**Verified GOOD — worth not touching (and worth citing in the thesis):**
- **MM-IPC arithmetic is numerically exact.** On real `PolSESS_C_new_64` data,
  the ES+C reconstruction (`mix` minus all non-kept layers) recovers the dry
  `speaker1` target to ~1e-4 RMS — the 16-bit PCM quantization floor. The
  indoor "subtract both dry `speaker2` and `sp2_reverb`" is correct additive
  layer algebra, not double subtraction.
- **Split leakage: train is clean.** Zero overlap of source speech/scene/event
  paths between train↔val and train↔test on every `*OryginalPath` column.
  (Caveat: val∩test share 137 scene source files — harmless for test validity,
  should be documented.)
- Eval-side correctness that holds: PIT reordering before PESQ/STOI (SB path),
  PESQ-`nb`@8000 is the right mode for natively-8kHz PolSESS, fp32 eval,
  per-variant SI-SDRi baselines. Benchmark scripts (warmup, CUDA sync,
  median-of-N, Mamba MAC hooks) are solid.
- Param counts in docs verified by instantiation: ConvTasNet 8.64M,
  DPRNN 2.61M, SepFormer 25.68M, MossFormer2 26.41M/55.74M — all match prose.
- The training side already embodies the ASR pipeline's disciplines where they
  matter: full config snapshot inside every checkpoint + sibling
  `config.yaml`, unconditional cuDNN determinism in `set_seed`, file logging,
  isolated venv for the one conflicting dep. **No structural back-port from
  `asr_pipeline/` is warranted** — that cross-pollination has effectively
  already happened.

**Gaps found (numbered; the work packages reference these):**
1. **`evaluation_results.csv` is schema-corrupted.** Rows 1–155 use a
   26-column schema, rows 156+ use 35 columns, and line 156 is a second header
   row — pandas cannot parse the file. `evaluate_all.flatten_results` changed
   shape and the append path (`--resume` header logic) never reconciled.
   No test pins the column set.
2. **Eval rows have no provenance.** No git SHA, no eval-dataset identity, and
   the CSV's `segment_length`/`sample_rate` columns come from the *training*
   config of the checkpoint, not the eval run — actively misleading when
   checkpoints trained on the old faulty 8k set and the new 64k set sit in the
   same table.
3. **Mean-of-batch-means bias.** `evaluate_model` appends batch-averaged
   SI-SDR/SI-SDRi/STOI scalars and averages over batches; the final partial
   batch is over-weighted, and the number depends on `batch_size` (default 4).
   PESQ (SB path) is per-sample and correct.
4. **No per-sample scores, no CIs.** Nothing downstream can compute a
   confidence interval for the architecture-comparison tables; per-sample
   values are discarded at aggregation.
5. **Zero run provenance in training.** Checkpoints/W&B capture config only —
   no git SHA, dirty flag, package versions, CUDA/GPU, hostname, argv. With
   two venvs (`venv` vs `venv_mamba3`) and an uncommitted feature branch,
   "which code/env produced this checkpoint" is genuinely ambiguous.
6. **Determinism is half-configured.** `set_seed` forces cuDNN determinism but
   omits `torch.use_deterministic_algorithms` + `CUBLAS_WORKSPACE_CONFIG`;
   TF32 is enabled (`set_float32_matmul_precision('high')`); train DataLoaders
   pass no `generator=`/`worker_init_fn`, and training-time MM-IPC variant
   choice uses the global `random` module inside workers. Reproducibility
   currently rides on incidental RNG ordering, and the claim is overstated.
   Flip side: permanently-on `cudnn.benchmark=False` forgoes a free speedup on
   fixed-length conv-heavy runs.
7. **Legacy-resume ordering bug.** `Trainer.load_checkpoint` sets
   `scheduler.best = self.best_val_sisdr` *before* `best_val_sisdr` is loaded
   from the checkpoint → `scheduler.best = -inf` — exactly the "first epoch
   after resume looks like an improvement" the comment says it prevents.
8. **Resume-state gaps (verified in `trainer.py` directly):**
   (a) GradScaler state is never saved/restored — fp16 models resume at
   `init_scale=256`; (b) a NaN-skipped batch calls `optimizer.zero_grad()`,
   discarding gradients accumulated mid-window when
   `grad_accumulation_steps>1`; (c) the grad-accum tail window divides by the
   full `accum_steps` (under-weights the epoch tail); (d) early-stopping
   patience counter resets on resume.
9. **Silent/misleading knobs in models.** `stride` is passed to Decoders but
   SpeechBrain Encoders hardcode `kernel_size//2` (5 models — latent shape bug
   for any non-default stride); SPMamba's `lstm_hidden_units` is a **no-op**
   (docstring claims it's the Mamba hidden dim — this touches the capacity
   -experiment narrative: hidden-units was never a lever); SepFormer's
   `hop_size` is inert yet actively written by the sweep plumbing
   (`load_config_for_run`); SPMamba's `window` param is ignored (hann
   hardcoded).
10. **No dataclass↔constructor parity guard.** `factory.py` does
    `model_class(**vars(params))`; for GPU-only models a field/ctor mismatch
    only explodes at run start on the training box. Checkable on CPU via
    `inspect.signature` in <1s.
11. **`train.py` / `train_sweep.py` are ~90% duplicated** (dataloader
    construction, per-variant val block, model/compile/wandb/trainer setup) —
    live drift risk, and the most visible smell for a supervisor.
12. **SI-SDRi baseline logic duplicated** between `evaluate.py` and
    `Trainer._compute_sisdri` (docstring: "per evaluate.py") — an edit to one
    silently desyncs `val_sisdr` from eval-table numbers.
13. **torch-2.8 `weights_only` landmine.** `config.py` loads checkpoints with
    explicit `weights_only=False`; `utils/model_utils.load_checkpoint_file`
    passes nothing (default flipped to True in torch 2.6+). Works today,
    version-sensitive.
14. **W&B resume orphans runs.** `wandb.init(resume="allow")` without `id=`
    always starts a fresh run; resumed training fragments one experiment
    across two W&B runs (display names already collide across runs).
15. **Sweep overrides silently drop unknown keys.** `load_config_for_run`
    covers ConvTasNet/DPRNN/SepFormer/MossFormer2 knobs; SPMamba/Mamba-family
    architecture keys in a sweep YAML would be ignored with no warning.
16. **README.md is materially stale** (clone-visible): broken link to
    `sweeps/EXPERIMENT_LOG.md` (only `_monolithic` exists), lists four
    deleted `asr/*.py` files, claims "264 tests / 17 files" (actual: 47 files,
    ~1150 tests), never mentions MossFormer2 / `asr_pipeline/` / `scripts/`.
17. **Hygiene smalls:** `plot_notebook.ipynb` is 4.4 MB of embedded outputs
    (largest tracked file); blanket `*.csv` gitignore vs force-tracked sweep
    CSVs (new result CSVs silently un-addable); `requirements.txt` covers only
    the training stack with no pointer to the ASR pipeline's venvs;
    `sweeps/all_runs.csv` has no column dictionary (contrast
    `SWEEP_RESULTS_SCHEMA.md`); `utils/common.py` sets a blanket
    `PYTHONWARNINGS=ignore::UserWarning` that undoes the careful
    `warning_filters.py`; EPS patch only affects ConvTasNet despite generic
    naming; `test_libri2mix.py` init tests are try/except-pass no-ops; val
    variant-selection determinism is untested.
18. **Curriculum × persistent_workers landmine (latent).** Curriculum mutates
    `train_loader.dataset.allowed_variants` in place; it works only because
    workers are re-forked each epoch. Anyone enabling `persistent_workers=True`
    silently trains every epoch on stage-0 variants. Nothing documents this.

---

## Work Package A — Thesis-results integrity  **[do first — feeds the tables]**

The eval layer produces the headline thesis numbers and is currently the
weakest link. Do this *before* final tables are generated/frozen, because A3
shifts numbers in the 3rd decimal place.

### A1. Repair + guard `evaluation_results.csv`  (gap 1) — S
- One-time surgery: split the existing file at the stray header into the
  26-col and 35-col eras, archive both, regenerate what's needed.
- Guard: on append, read the existing header and assert field-set equality —
  refuse (or write a new dated file) on mismatch.
- Golden test on `flatten_results` asserting the exact column tuple (this is
  the test that would have prevented the corruption).

### A2. Provenance columns in eval output  (gap 2) — S
- Add to `flatten_results`: `git_sha` + `git_dirty`, `eval_dataset_name`
  (= `Path(data_root).name`), `eval_data_root`, `eval_subset`,
  `eval_batch_size`, `torch_version`.
- Rename the train-derived columns `segment_length`/`sample_rate` →
  `train_segment_length`/`train_sample_rate`.

### A3. Per-sample scoring, exact means, CIs  (gaps 3, 4) — M
- Force `batch_size=1` in `evaluate_model` (the Libri2Mix path already does) —
  kills the mean-of-batch-means bias and makes every score per-sample.
- Persist long-format per-sample CSV (`run, variant, sample_idx, si_sdr,
  si_sdri, pesq, stoi`) next to the aggregate.
- Small helper: mean ± 95% bootstrap CI per (model, variant) for thesis
  tables. Report PESQ failure counts per variant instead of dropping silently
  at debug level.

### A4. One tracked eval manifest  (part of gap 17) — S/M
- Consolidate `checkpoints_for_eval{,2,3}.csv` into one committed manifest
  (`experiments/thesis_eval_manifest.csv`: display_name, checkpoint_path,
  dataset, notes) + gitignore exception. `evaluate_all` reads it by default.
  The "which checkpoints constitute the thesis comparison" question becomes
  answerable from the repo.

### A5. Shared SI-SDRi helper  (gap 12) — S
- Extract `compute_sisdr_and_sisdri(estimates, clean, mix, task)` into
  `utils/`; call from both `Trainer` and `evaluate.py`. Verify parity on one
  checkpoint before/after.

### A6. Eval determinism insurance  (part of gap 6) — S
- `set_seed()` at top of `evaluate.py:main`; extend the dataset's
  deterministic variant branch from `subset == "val"` to `("val", "test")`.
  (Safe today only because eval always forces a single variant.)

*(Blue-sky umbrella, only if time allows: a single `scripts/eval_thesis.py`
that walks the A4 manifest, evaluates with fixed seed + bs=1, and emits
per-sample parquet + CI table + append-only provenance ledger — subsumes
A1–A4. L effort; the itemized versions deliver most of the value.)*

---

## Work Package B — Run provenance + determinism  **[do before the remaining training runs]**

### B1. Run manifest  (gap 5) — S/M
- `utils.collect_run_manifest()`: git SHA + dirty flag, torch/CUDA/cuDNN
  versions, `mamba_ssm`/`triton` versions when present, GPU name, hostname,
  seed, `sys.argv`. Embed under `"provenance"` in every checkpoint, write a
  human-readable `run_manifest.yaml` next to `config.yaml`, pass to W&B
  config. One call site each in `train.py`/`train_sweep.py`.
- Highest thesis value per line of code in this plan; do before the SPMamba
  full-ks8 runs so *they* are captured.

### B2. Coherent determinism story  (gaps 6, 18) — S
- `training.deterministic: bool` flag (mirror `PipelineConfig.deterministic`):
  when True, current behavior + `torch.use_deterministic_algorithms(True,
  warn_only=True)` + `CUBLAS_WORKSPACE_CONFIG`; when False,
  `cudnn.benchmark=True` for the free conv-autotune speedup. Default =
  current behavior; document the TF32 trade-off in the docstring.
- Canonical DataLoader RNG plumbing: `generator=torch.Generator().manual_seed(
  seed)` + `worker_init_fn` seeding `random` from `torch.initial_seed()` —
  makes the MM-IPC variant stream reproducible by contract, not by accident.
- Write down the curriculum × `persistent_workers` incompatibility at both the
  DataLoader site and `_update_training_variants` (one comment each);
  optionally gate `persistent_workers` on `curriculum is None`.

### B3. Small resume/robustness fixes  (gaps 7, 8, 13, 14) — S each
- Fix `scheduler.best = -inf` ordering in `load_checkpoint` (move the
  `best_val_sisdr` load above the scheduler block).
- Save/restore `scaler.state_dict()` in checkpoints (fp16 models).
- NaN-skip during grad accumulation: don't `zero_grad()` mid-window (or
  document that a NaN batch discards its window).
- Grad-accum tail window: divide by actual micro-batch count (or document the
  approximation; check whether any reported run used `accum_steps>1`).
- `load_checkpoint_file`: explicit `weights_only=False` (match `config.py`).
- W&B resume: persist run id in checkpoint (part of B1's manifest), pass
  `id=..., resume="must"` when resuming; or document that resume ≠ same run.
- Persist `epochs_without_improvement` if resume is used for any reported run;
  otherwise skip.

---

## Work Package C — Model-layer guards + honest knobs  **[cheap, before any late sweep]**

### C1. Dataclass↔constructor parity test  (gap 10) — S, do-now
- Parametrized CPU test over all eight `(Params dataclass, model class)`
  pairs: dataclass field names ⊆ `inspect.signature(cls.__init__)` params.
  Catches the most likely silent breakage of the remaining month, GPU-free.

### C2. `stride == kernel_size // 2` guard  (gap 9) — S, do-now
- One assertion in ConvTasNet/DPRNN/SepFormer/MambaTasNet/DPMamba
  constructors with a message naming the SpeechBrain Encoder lock. (Or drop
  the param; the guard is lower-churn.)

### C3. Truth in knob advertising  (gap 9) — S
- SPMamba: fix the `lstm_hidden_units` docstring to "Unused; kept for API
  symmetry" (spmamba3 already says this) — **and add one line to the SPMamba
  capacity write-up**: hidden-units was never a lever; capacity = emb_dim /
  emb_ks / n_layers only. This is thesis-narrative-relevant, not cosmetics.
- SepFormer: remove `hop_size` (param + `SepFormerParams` + the
  `load_config_for_run` line that writes it) or comment it inert — verify no
  saved checkpoint config round-trips the field first.
- SPMamba/SPMamba3: assert `window == "hann"` or honor the param.
- Sweep overrides (gap 15): after applying known overrides, warn on any
  unconsumed non-W&B key. Only needed if more sweeps are planned.

### C4. SPMamba↔SPMamba3 drift-guard test  (side project, low stakes) — S
- `inspect.getsource` equality test over the intentionally-mirrored symbols
  (`LayerNormalization4D*`, `GridNetBlock`, `forward` bodies). Preserves the
  diffable-mirror design; do NOT extract a shared module.

### C5. Task→C forcing visibility  (CLAUDE.md pitfall #3) — S
- Log one line when `Config.__post_init__` actually changes `C`/`n_srcs`
  ("task=ES forces C 2→1"); add a docstring. Behavior is correct and tested,
  just invisible. Optional: collapse the 35-line if-elif ladder to a
  `{model_type: attr_name}` map.

---

## Work Package D — Thesis-artifact scripts  **[high citation value, small effort]**

### D1. `scripts/audit_mmipc.py` — MM-IPC reconstruction audit — S/M
- For K random rows per split: run every variant through `_apply_mmipc`,
  assert the ES+C residual vs dry `speaker1` is below the PCM-quantization
  threshold, print per-variant residual table. The survey already proved
  ~1e-4 RMS on real data — codify it as a regression guard + a citable
  "augmentation is provably lossless" line for the methods chapter.

### D2. `scripts/audit_split_leakage.py` — split-disjointness audit — S
- Pairwise set intersections of every `*OryginalPath` column across
  train/val/test; exit non-zero on any train↔{val,test} overlap; report the
  known val∩test 137-scene-file sharing as informational. Pre-empts the
  reviewer's first dataset-integrity question with numbers.

### D3. Variant-algebra documentation block — S
- Comment block in `polsess_dataset.py` with the mix decomposition
  (`mix = sp_dry + sp_reverb_tail + scene + event_dry + event_reverb_tail`)
  and a per-variant removed-layers truth table; cross-reference D1. Feeds the
  methods chapter almost verbatim and kills the "is this double-subtracting?"
  reviewer flag.

### D4. Generated param-count manifest — M
- `--params-only` mode on `benchmark_inference.py` (or a 30-line
  `scripts/model_manifest.py`): iterate the thesis configs, build each model,
  dump `{model, config, params}` to CSV/Markdown. Thesis tables cite the
  artifact instead of hand-transcribed counts (the Mamba XS/S/M/L "2.2–59.6M"
  range currently lives only in prose + YAMLs).
- Thesis prose companion: add a mask-vs-mapping column to the architecture
  table (SPMamba/SPMamba3 are TF-domain *mapping* models; the rest mask) —
  paper-faithful, but a methodologist will ask.

### D5. Optional: surface CSV metadata for stratified results — S/M
- Expose `sceneClass`/`eventClass`/`SSR` from the corpus CSV (sidecar
  accessor, not wired into training) → per-condition SI-SDRi breakdowns for
  the analysis chapter, essentially free.

---

## Work Package E — Structure + presentability  **[before supervisor review]**

### E1. De-duplicate `train.py` / `train_sweep.py`  (gap 11) — M
- Extract `build_dataloaders(config, summary_info)` +
  `build_trainer(config, ...)` into `training/setup.py`; both mains shrink to
  config-loading + builders + `trainer.train()`. ~90 duplicated lines gone,
  drift risk dead. Existing Trainer tests make this low-risk.

### E2. README repair  (gap 16) — M, do-now
- Fix the `EXPERIMENT_LOG.md` link (→ `_monolithic`); regenerate the project
  structure block (drop archived `asr/*.py`, add `asr_pipeline/`, `scripts/`,
  `models/mossformer2/`); fix or drop the test count; mention MossFormer2 (and
  that SPMamba3 exists off-thesis); caveat the March-2026 results table as
  "early baselines; see thesis for final numbers". Highest
  credibility-per-hour item in the whole plan — it's the front door.

### E3. Hygiene batch  (gap 17) — S each
- Strip outputs from / relocate `plot_notebook.ipynb` (4.4 MB → ~40 KB).
- Narrow the blanket `*.csv` gitignore to the actual noise files, or add an
  explicit `!sweeps/**/*.csv` so the tracked-CSV policy is visible.
- One-line header in `requirements.txt` pointing at CLAUDE.md's Virtual
  Environments section for the ASR-pipeline stack.
- `sweeps/RESULTS_SCHEMA.md`: ~30-line column dictionary for `all_runs.csv`
  + the W&B export command that regenerates it (the one worthwhile
  cross-pollination from the ASR side).
- Drop the blanket `PYTHONWARNINGS=ignore::UserWarning` env var; rely on the
  targeted `warning_filters.py` (watch for re-surfaced third-party noise).
- Rename/scope `apply_eps_patch` → ConvTasNet-only in name or docstring;
  align the CLAUDE.md description.
- Defensive `_orig_mod` prefix strip in `load_model_for_inference` (the
  CLAUDE.md claim currently describes save-side behavior).

### E4. Test-suite gaps worth filling — M
- Val variant-selection determinism test (the most reproducibility-critical
  untested branch).
- Replace `test_libri2mix.py`'s try/except-pass no-op tests with a `tmp_path`
  fixture actually exercising `__getitem__`.
- One end-to-end `evaluate_model` test on a 2-sample synthetic set with
  hand-computed SI-SDRi (pins baseline + PIT reordering + aggregation).
- Delete or implement the permanently-skipped test in `test_dataset.py` and
  the commented-out one in `test_evaluation.py`.

---

## Explicitly NOT doing (over-engineering, or wrong side of the deadline)

- **No SCOPE.md / knob-inventory / eval-package port** to the training side —
  surveyed and rejected; the light equivalents already exist and a second
  scope contract is ceremony for a closed research codebase.
- **No per-variant precomputed augmentation cache** — augmentation is a few
  tensor subtractions (~free); any loader bottleneck would be decode-bound,
  and the remaining heavy-model runs are GPU-bound anyway. Measure before
  touching (`pin_memory=True` is the only free add).
- **No polsess↔libri2mix base-class extraction** — different contracts, tiny
  overlap; would violate the no-speculative-abstraction rule.
- **No fused/cached per-variant validation rework** (8× val re-render) unless
  a measurement shows val is a meaningful fraction of remaining wall-clock —
  it touches the path that feeds thesis numbers and must prove byte-identical.
- **No bf16 validation for non-MossFormer2 models** — parity of val with
  `evaluate.py` (fp32) is worth more than the speedup pre-hand-in.
- **Local working-dir clutter** (`model_debug.log` 19.7 MB, `evaluate/`,
  `runs/`, `wandb/`, `mobile/` 88 MB) — all gitignored, invisible in a clone;
  tidy opportunistically, not as planned work.

## Sequencing

1. **A1–A6** (results integrity) — before generating/freezing any thesis
   table; A3 changes 3rd-decimal numbers, so it must land first.
2. **B1–B2** (provenance + determinism) — before launching the remaining
   SPMamba/scaling runs, so those runs are born captured.
3. **C1–C3 + B3** (guards, honest knobs, small fixes) — one short session,
   mostly one-liners with outsized silent-failure protection.
4. **D1–D4** (audit scripts + generated tables) — alongside methods-chapter
   writing; each produces a citable artifact.
5. **E1–E4** (dedup, README, hygiene, tests) — before sharing the repo with
   supervisors; E2 (README) can be done any time and should not wait.
