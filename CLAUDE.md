# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

PyTorch implementation of speech separation using ConvTasNet, SepFormer, DPRNN, SPMamba, Mamba-TasNet, and DPMamba architectures on the PolSESS dataset. Part of a master's thesis on speech separation for downstream Polish ASR preprocessing.

Thesis prose and experiment logs live in `thesis/` — a symlink to an Obsidian vault on Windows, tracked by its own git repo (ignored here). Read freely; when editing thesis content, `cd thesis/` so `thesis/CLAUDE.md` stacks in on top of this file.

## Style

Reply in English unless the user specifies otherwise.
The amount of thinking should be proportional to the complexity of the task you're given.
Avoid unnecessary verbosity by using CoT to structure your response.

Important: when launching subagents, use only Opus agents (unless the user specifies differently). You may decide to use Sonnet subagents for the easiest work. Never launch Fable subagents — with one standing exception (approved 2026-07-25): agents whose job is synthesis/adjudication or premium prose editing (`review-synthesizer`, `code-review-synthesizer`, `redaktor`) pin `model: fable` in their frontmatter, because deciding between conflicting reviewers merits the strongest reasoning. Do not extend the exception to other agents without asking.

## Dataset Variants

- `PolSESS_C_both` = `C_both_16k_faulty` — old 8k-effective dataset (half was duplicated). Used for early baselines, HPO, and HPO validation runs.
- `PolSESS_C_new_64` = `C_new_64` — correct 64k dataset generated 2026-04-15. Use `train_max_samples=16000` / `32000` / full for 16k / 32k / 64k scaling experiments.
- `PolSESS_C_final_128_v2` - 128k dataset for final training runs. contains other languages speech alongside Polish.

## W&B Projects

- `polsess-separation` — standalone runs (baselines, HPO validation), uses `PolSESS_C_both`.
- `polsess-thesis-experiments` — sweeps.
- `polsess-separation-real16k` / `-32k` / `-64k` — scaling runs on subsets of `PolSESS_C_new_64`.
- `polsess-separation-128k` - final runs on 128k dataset.

## Thesis Code Principles

This code will be reviewed by academic supervisors. Prioritize: clarity over cleverness, simplicity over abstraction, reproducibility. Prefer explicit implementations that match cited papers. Don't over-engineer — no factories for <4 variants, no deep inheritance, no speculative features. Experiment logging is handled outside this repo.

## Keeping This File Current

After any substantial change — new top-level script or subsystem, new env var, new dataset/model/task variant, new gotcha worth flagging, removed commands, or changed config precedence — propose a targeted edit to this CLAUDE.md. Update in place; don't rewrite from scratch. Skip for routine bugfixes, refactors, or one-off experiments.

## Key Commands

**Training:**
```bash
python train.py --config experiments/dprnn/dprnn_baseline.yaml
python train.py --config experiments/spmamba/spmamba_baseline.yaml --no-wandb --seed 123
python train.py --resume checkpoints/dprnn/SB/run_name/dprnn_SB_best.pt
```

**Evaluation:**
```bash
# PolSESS (all MM-IPC variants)
python evaluate.py --checkpoint checkpoints/dprnn/SB/run_name/dprnn_SB_best.pt
# Specific variant
python evaluate.py --checkpoint path/to/model.pt --variant SER
# Fast (skip PESQ/STOI)
python evaluate.py --checkpoint path/to/model.pt --no-pesq --no-stoi
# Libri2Mix
python evaluate.py --checkpoint path/to/model.pt --dataset librimix --librimix-root /home/user/datasets/LibriMix/Libri2Mix
# Save CSV
python evaluate.py --checkpoint path/to/model.pt --output results.csv
# Batch eval of the thesis checkpoint set (reads experiments/thesis_eval_manifest.csv by default;
# bs=1 exact per-sample scoring → aggregate CSV + <stem>_per_sample.csv, provenance columns)
python evaluate_all.py --resume
```

**Testing:**
```bash
pytest
pytest --cov=. --cov-report=html
pytest tests/test_model.py -v
```

**Sweeps:**
```bash
# Launch a W&B sweep (the sweep YAML points at train_sweep.py as the program)
wandb sweep sweeps/3-hyperparam-opt/dprnn/stage1/dprnn.yaml
# then run agents against the returned sweep ID
```

**Benchmarks (thesis data — the *objective* axes of the ch5 multi-axis table; quality/epochs/wall-clock-to-convergence are run-result facts and come from W&B, not from here):**
```bash
python scripts/benchmark_inference.py --cross-check   # MACs, latency+IQR, RTF, peak infer VRAM
python scripts/benchmark_training.py                  # train-only throughput @ trained bs AND bs=1, peak train VRAM
```
Both read one shared architecture list (`scripts/benchmark_models.py`) and write to `docs/generated/benchmark_{inference,training}.csv` with `gpu/torch/cuda/ptflops` provenance columns; the 2026-04 CSVs in `scripts/` are kept as the provenance of the numbers currently in the chapter. MAC counting needs three corrections ptflops does not make on its own — `nn.MultiheadAttention` double count, invisible `einsum`, Mamba scan FLOPs-vs-MACs — all documented in the inference script's docstring and audited in `thesis/thesis-log/sweep_plan/ch5_test_evals/BENCHMARK_AUDIT.md`. Counting is device-independent (`--device cpu` works for non-Mamba models); everything timed must run on one machine in one session.

**Thesis audit artifacts (citable, CPU-only):**
```bash
python scripts/audit_mmipc.py           # MM-IPC reconstruction lossless to the 16-bit PCM floor (exit≠0 on violation)
python scripts/audit_split_leakage.py   # train↔{val,test} path disjointness (val∩test scene sharing reported informational)
python scripts/model_manifest.py        # per-config param counts → docs/generated/model_manifest.{csv,md}
python scripts/squim_validation.py all  # B7 SQUIM-vs-true-metrics validation over a separator quality ladder (GPU, ~30 min;
                                        #   prereg + outputs: thesis/thesis-log/sweep_plan/{B7_SQUIM_PREREG.md,b7_squim/})
```

**Interactive:**
```bash
jupyter notebook test_model_interactive.ipynb
jupyter notebook asr/explore_pipeline.ipynb   # interactive frontend for the asr_pipeline/ package (see ASR section)
```

**ASR pipeline (CLI front door — preflights env/checkpoints before any model load):**
```bash
python -m asr_pipeline run --config asr_pipeline/configs/sweep_best_e31_refineplus.yaml \
    --input rec.wav --write-outputs <eval_root> --set diarization.backend=pyannote
python -m asr_pipeline batch --split clarin_dev --mode no_enh        # or --manifest/--glob/--inputs + --out-root
python -m asr_pipeline score --eval-root <eval_root> --out-dir <csv_dir>
```

**Showcase webapp (`webapp/` — thesis-defense demo UI over the ASR pipeline):**
```bash
./webapp/run.sh                                  # real server, port ${WEBAPP_PORT:-8871}, HF_HUB_OFFLINE=1; preflight at startup
venv/bin/python -m webapp.dev_server --port 8899 # GPU-free dev harness (FakeRunner replays a timed run; --fail-stage for error UI)
venv/bin/python -m webapp.examples_build         # regenerate examples_manifest.json (gitignored) from the frozen v41_merge eval tree
pytest tests/test_webapp_backend.py              # backend suite (no GPU, no asr_pipeline import)
```
FastAPI + vanilla ES modules (no build step, no CDN); a pure READ-ONLY consumer of `asr_pipeline` (SCOPE §1 — the job queue lives here, never in the package). One worker thread, one job at a time (12 GB GPU, phase-major). Every job runs with per-job `spill_intermediate` → `<job>/spill` (source of mid-run `partial` results + the enhancement A/B panel). Contract = `webapp/API.md` (binding, orchestrator-owned); design + decision log = `docs/fable_plans/frontends_road1_webapp.md`. Jobs land in `$WEBAPP_JOBS_ROOT` (default `~/webapp_jobs`). Polish UI, English technical terms; GT/scores appear ONLY on the examples page.

**On-device demo (`webapp_ondevice/` — Road 2: OSD-gated separation fully client-side, GitHub Pages-ready):**
```bash
python3 webapp_ondevice/devcheck/serve.py webapp_ondevice --port 8123    # dev: serve the ROOT (test assets reachable); deploy publishes site/ alone
~/playwright_venv/bin/python webapp_ondevice/devcheck/check_page.py <url> --shot s.png   # headless verify: screenshots + console errors + exit codes (HARNESS.md)
CUDA_VISIBLE_DEVICES="" venv/bin/python webapp_ondevice/build/export_separators.py && CUDA_VISIBLE_DEVICES="" venv/bin/python webapp_ondevice/build/verify_models.py   # regenerate site/models/
```
Static site, zero server: pyannote-seg-3.0 fp16 ONNX (3 MB) detects overlap in-browser, routing (ported from `asr_pipeline/stages/routing.py`, expand-to-4 s like the shipped configs) sends ONLY overlap regions to the separator; "Separuj mimo wszystko" override makes the ch6 M2 phantom-stream failure audible. Design mockup in `design/` is author-accepted and binding; plan = `docs/fable_plans/frontends_road2_ondevice.md`. Separators: SepFormer-128k int8 (default — WASM gate: RTF 1.46 vs MF2's 22.4 single-threaded) + MF2-128k **e46** int8 MatMul-only ("safe" recipe transparent on e23 degrades to 23 dB on e46 — `build/NOTES.md`). Gotchas: ORT graph-opt level `'all'` **segfaults** on the fp16 pyannote model (use `'extended'`); OSD sliding-window hop must be 79920 samples (whole frames), not 80000; Chromium's AudioBuffer resampling does NO anti-alias filtering on decimation (hand-rolled sinc in `resample.js`); `site/models/` + `site/vendor/` are gitignored but MUST ship on deploy (~90 MB). Engine parity proof page: `webapp_ondevice/test.html` (drive with `--js "window.__done"`, ~6.5 min). OSD Python reference + JS-parity vectors: `reference/`.

## Architecture Overview

**Configuration (`config.py`):** Dataclasses (`DataConfig`, `ModelConfig`, `TrainingConfig`) with nested model/dataset params. Priority: defaults < env vars < YAML < CLI args. Use `get_config_from_args()` for CLI, `load_config_for_run(wandb.config)` for sweeps.

**Model Registry (`models/__init__.py`):** Dict-based. `get_model("name")` returns class. Mamba models auto-excluded without `mamba-ssm`.
- `convtasnet` (~8.7M for SB/C=2; the oft-quoted 8.64M is the ES/C=1 build — see `docs/generated/model_manifest.md`), `sepformer` (~26M), `mossformer2` (matched ~26M / full ~55.7M), `dprnn` (~2-3M) — cross-platform
  - `mossformer2` = MossFormer2 (Zhao et al. 2023, arXiv:2312.11825): transformer + gated-FSMN hybrid. The model files in `models/mossformer2/` are **vendored** from ClearerVoice-Studio (`train/speech_separation/models/mossformer2/`); `models/mossformer2/__init__.py` is the project wrapper. Pure PyTorch (deps: `einops`, `rotary-embedding-torch`), so cross-platform. Single `N` knob = encoder dim = transformer dim (the two must match upstream); `num_blocks` is GFSMN depth (24 = paper full, 11 ≈ SepFormer-matched); `attn_dropout` (default 0.1, upstream hard-coded) covers attention-path dropout — FSMN-gate dropout stays fixed at 0.1. Sweep override key `dropout` routes to `attn_dropout`. Configs: `experiments/mossformer2/mossformer2_{matched,full}.yaml`.
- `spmamba` (~1.2M), `mamba_tasnet` (XS/S/M/L: 2.2-59.0M), `dpmamba` (XS/S/M/L: 2.3-59.8M) — Linux + CUDA only (measured builds: `docs/generated/model_manifest.md`)

**Dataset Registry (`datasets/__init__.py`):** Dict-based. `get_dataset("name")` returns class. Supports: `polsess`, `libri2mix`, `echoset`.
- `echoset` (added 2026-07-31) — reverberant 2-speaker corpus (Matterport3D RIRs), 16 kHz, 6 s items. **CORPUS DELETED 2026-08-03 (author-approved, 26 GB reclaimed): the loader and registry entry are retained as the record of a negative result, but `~/datasets/EchoSet/` no longer exists — re-download before any use.** Why it was dropped: it ships only `spk{1,2}_reverb.wav`, i.e. reverberant targets, while PolSESS SB targets are dry (`_compute_clean` returns the `clean/` speech), so a PolSESS-trained model — which dereverberates — is penalised for it: a probe scores SI-SDRi ≈ **−2.92 dB** where the same code path scores +19.79 on Libri2Mix. Disqualified for generalization numbers, not merely caveated. (Unrelated and unaffected: `asr_pipeline/configs/b1_tiger_echoset.yaml` + `vendor/tiger/` concern the **TIGER checkpoint trained on EchoSet**, `JusperLee/TIGER-speech`, not this corpus.) See `thesis/thesis-log/sweep_plan/ch5_test_evals/CH5_TEST_EVALS_NOTES.md`.

**ASR subsystem (`asr/`):** Notebooks driving the productionised CLARIN pipeline. The pre-CLARIN one-shot REAL-M/LibriMix eval flow (`evaluate_asr.py` + intrusive variant + `dataset.py`/`transcribe.py`/`metrics.py` + its `test_asr.py`) is **archived** under `asr/archive/old-asr/`; the original Gradio POC notebook (`asr_pipeline.ipynb`) + early LibriMix-prep scripts sit in `asr/archive/`. Archived code is parked — its imports reference the old top-level `asr.` package layout and would need rewiring to run.
- `clarin_fragments.ipynb` / `clarin_subset_review.ipynb`: select + review the CLARIN test fragments (uses `scripts/clarin_fragment_finder.py`).
- `explore_pipeline.ipynb`: interactive frontend for the productionised `asr_pipeline/` package — per-stage knobs, re-run any stage in isolation, one model on GPU at a time.
- `evaluate_pipeline.ipynb`: two-layer evaluation (L2 audio quality + L3 WER) of `asr_pipeline/` output against the CLARIN debleed (oracle) channels, backed by `asr_pipeline/eval/`.

**`asr_pipeline/` package** — productionised pipeline. **Before changing code here, read `asr_pipeline/SCOPE.md`** — the scope contract (purpose, error philosophy, fallback ledger, rules for agents); it overrides reviewer instincts, and its `UNDECIDED` items are reserved for the author. `Pipeline` orchestrator runs seven stages in fixed order:
1. **diarization** — `pyannote` (dataclass/`default.yaml` default: `speaker-diarization-3.1`, HF token via `$HF_TOKEN`) or `sortformer` (NVIDIA Sortformer v1 offline EEND via isolated NeMo venv subprocess `scripts/sortformer_worker.py`, reached through `$SORTFORMER_VENV_PY` — no default, missing → loud crash). `num_speakers=2`, mono 16 kHz. The **shipped best config** (`configs/sweep_best_e31_refineplus.yaml`, adopted 2026-07-04) uses `backend: sortformer` + `sortformer_head_policy: merge` (the "fold": surplus-head runs re-assigned to the top-2 speakers by ECAPA2 match instead of discarded); streaming v2.1 model ids get the offline very-high-latency preset automatically in the worker (Life-2 track, NVIDIA Open license). Long recordings (> `sortformer_long_audio_threshold_s`, default 240 s) are auto-routed from the O(T²)-memory offline v1 model to the streaming `sortformer_long_audio_model_id` (default `diar_streaming_sortformer_4spk-v2.1`) via the same worker, with a loud warning — v1 OOMs past ~5-6 min on 12 GB; set the threshold to 0 to disable.
2. **routing** — split overlap vs solo regions.
3. **enhancement** — ClearerVoice backends: `frcrn_se_16k` (interim default, SCOPE §10 q7), `mossformer_gan_se_16k`. (Vendored MP-SENet backend removed 2026-06-11; ModelScope ZipEnhancer + the invalid `mossformer2_se_48k` dropped 2026-07-06; final default ruling deferred.)
4. **separation** — MossFormer2 matched-128k checkpoint by default (`checkpoints/mossformer2/SB/mossformer2_matched_128k_final_42_e46/mossformer2_SB_best_e46.pt` — NB the `_e46` sibling dir, epoch 45, val_sisdr 17.04; the older `final_42/` dir holds the e23 checkpoint and is NOT what ships — a stale reference here misdirected the 2026-08-05 B9 spike. Family swapped in 2026-06-13, ~0.6 dB SI-SDRi over the prior SepFormer 128k on val; runs at `separator_sample_rate=8000` like its predecessor). Dataclass defaults and `configs/default.yaml` agree; a pin test enforces that *consistency*, not the literal checkpoint — swappable by editing both the dataclass default and `default.yaml` (the generic `load_model_for_inference` reads `model_type` from the checkpoint config, so any trained architecture loads). `separator_backend` (default `repo`) additionally accepts `speechbrain` (checkpoint_path = HF id, cached under `checkpoints/external/speechbrain/`), `clearvoice` (model name, e.g. `MossFormer2_SS_16K` — 2 s one-pass window, adapter refuses longer input), `sr_corrnet` (HF id; `sr-corrnet-ss` pip pkg; NB its WHAMR checkpoint degenerates on clean input by domain prior — smoke with noisy audio), `tf_locoformer` (LOCAL .pth; model vendored at `asr_pipeline/vendor/tf_locoformer/`), `tiger` (HF id, 16 kHz; vendored at `vendor/tiger/`) and `mossformer2_dp` (HF id; dual-path variant vendored at `vendor/mossformer2_dp/` — different arch from `models/mossformer2`) for the B1 external-separator experiment; arm configs `configs/b1_*.yaml`, plan `docs/fable_plans/b1_external_separator_swap.md`. Runs on overlap fragments only. Sweepable-knob inventory across all stages: `asr_pipeline/SWEEP_KNOBS.md`.
5. **post_separation_processing** — VAD mask + optional BWE (`naive` / `ap_bwe`; FlowHigh dropped 2026-07-06). Always-on (downstream depends on its `_gated` arrays); set `backend: naive` to apply only the mask. `configs/default.yaml` ships `backend: ap_bwe` (dataclass default is `naive`).
6. **assembly** — stitch per-speaker streams, ECAPA anchor for speaker identity across pieces.
7. **transcription** — `whisperx` / `coherex` backends (base openai-`whisper` dropped 2026-07-06); default = WhisperX `large-v2`. Alignment is per-language: `align_model_name=None` (default) lets WhisperX pick its per-language wav2vec2 default — for `pl` that is `jonatasgrosman/wav2vec2-large-xlsr-53-polish` (unchanged), set explicitly to override. English preset at `asr_pipeline/configs/english.yaml`. (rationale in `asr_pipeline/configs/README.md`). The `coherex` backend runs Cohere Transcribe via Diffio-AI/CohereX in an **isolated venv subprocess** (`scripts/coherex_worker.py`, reached via `$COHEREX_VENV_PY` — its deps conflict with the main venv, same isolation as Brouhaha; `model_name` = the Cohere model id, e.g. `CohereLabs/cohere-transcribe-03-2026`). It loads the model per `transcribe` call, so it targets interactive/single-recording use (`explore_pipeline`), not batch eval. Investigation verdict: WhisperX-large-v2 still beats Cohere ceiling-vs-ceiling on the dev set (~3.3 cpWER); the backend exists for choosability, not because Cohere won.

Phase-major execution (one model on GPU at a time). Config via nested dataclasses + YAML. `PipelineConfig.deterministic` (default `true`) forces deterministic cuDNN algorithms at `Pipeline.__init__` — the enhancement conv stage is otherwise the pipeline's *only* run-to-run nondeterminism source (≈1e-7 float noise in `enhanced_full` that WhisperX can amplify into a flipped token; every other stage is deterministic given fixed input). Costs a ~2× enhancement-stage slowdown (no conv autotuning); set `false` for non-reproducible-but-faster dev runs. Configs in `asr_pipeline/configs/`: `default.yaml` (POC-equivalent), `p4_fixed_pad.yaml` / `p5_full_length.yaml` (ablation knobs). Debug log at `/tmp/asr_pipeline_debug.log` (override `ASR_PIPELINE_DEBUG_LOG`) — survives the WSL stdout bridge dropping. Config serializers (`save_pipeline_config_to_yaml`, the `metadata.json` snapshot in `io.write_pipeline_outputs`) mask `diarization.hf_token` as `REDACTED` so live tokens never land in output files.

The package has a CLI front door: `python -m asr_pipeline run|batch|score`. `run`/`batch` call `asr_pipeline/preflight.py` (fail-loud env/checkpoint checks — missing `$SORTFORMER_VENV_PY` etc. fails in seconds, **before** any model load; `num2words` warn-only, SCOPE §10 q2) and accept repeatable `--set stage.knob=value` dotted overrides (`config.apply_overrides` — the one override mechanism; the sweep script imports it). `asr_pipeline/batch.py` `run_batch` owns the batch loop: per-recording failure isolation (`failures.csv`, batch continues — SCOPE §4.2), completion sentinel = `metadata.json` in the target subdir (`--force` re-runs), `--mode full|no_sep|no_enh|minimal` ablation presets, and the single home of the GPU-teardown block. `scripts/sweep_pipeline.py`'s run loop delegates to it while pinning its legacy `transcript_A.txt` sentinel, so completed sweep trees never recompute. `Pipeline(config, on_event=...)` emits per-stage timing events (load vs run seconds split, measured around the actual calls) that land in `metadata.json`/`run_meta.json` — the data that gates the Tier-2 stage-major batch rework.

**`asr_pipeline/eval/`** — two-layer scoring (L2 + L3). `evaluate_recording(rec) → ScoreCard` runs both layers for one recording; `evaluate_many` batches with SQUIM loaded once; `walk_eval_tree` yields `Recording` per directory under the eval root. (L1/DER retired 2026-06-11, SCOPE §10 q8: no valid reference diarization exists — `eval/layer1.py` + the `compute_der`/`parse_rttm` plumbing deleted.)
- **L2 audio quality** — intrusive SI-SDR / PESQ-WB / STOI (chunked, median-aggregated, speech-presence filtered) when oracle audio is available; non-intrusive TorchAudio-SQUIM (chunked, mean-aggregated) always.
- **L3 ASR** — cpWER + tcpWER **+ cpCER** (the campaign's primary metric) per ablation mode (full / no-sep / no-enh), ORC-WER on the mixture baseline. Backed by `meeteval`; `compute_layer3` routes through `eval/metrics.per_fragment_metrics`, which also owns the ORC/MIMO combinatorial blow-up guard (long recordings skip those metrics with a printed note instead of hanging — the guard formerly lived only in the explore notebook).
- **Campaign statistics** — `eval/stats.py`: recording-clustered paired bootstrap, Holm-Bonferroni, Benjamini-Hochberg, micro-averages, strata assignment — extracted bit-identically (fixed-seed golden tests) from `scripts/rescore_stratified.py`, which is now a thin driver with unchanged CLI/output.

Low-level helpers exported for notebook use: `parse_gt_txt`, `parse_transcript_file`, `cpwer_meeteval`, `orc_wer_meeteval`.

**ASR datasets**
- `~/datasets/clarin_gotowy/gotowy/` — CLARIN debleed eval set (oracle per-speaker channels). Root = `<id>.wav` stereo inputs; `debleed/<id>_{L,R}.wav` = oracle channels; `debleed_enhanced/` = MossFormerGAN-enhanced oracles; `after_pipeline/<id>_{s1,s2}.wav` = pipeline outputs; `transcripts/<id>.txt` = pipeline transcripts; `eval_cache/` = cached references.
- `~/datasets/clarin_all_2speakers/` — full CLARIN 2-speaker download (no oracle channels). `clarin_download/<id>.wav` raw inputs (+ `Korpus.csv`, `Korpus_with_filename.csv`); `diarization/<id>.json` pyannote outputs; `enhanced_mossformer/<id>.wav` MossFormerGAN-enhanced; `auto_transcription_raw/<id>.{txt,json}` and `auto_transcription_enhanced_mossformer/<id>.{txt,json}` WhisperX transcripts.

**ASR helper scripts (`scripts/`)**
- `run_pipeline_on_recording.py` — full pipeline on one recording in three ablation modes (`pipeline` / `pipeline_nosep` / `pipeline_noenh`); drives the L3 WER table. Now a thin wrapper over `asr_pipeline.batch.run_batch` (same CLI). (`batch_pipeline_noenh.py` deleted 2026-07-12 — subsumed by `python -m asr_pipeline batch --mode no_enh`.)
- `prepare_eval_references.py` — cache enhanced oracles + GT-style transcripts for the eval module.
- `enhance_clarin_debleed.py` — batch MossFormerGAN_SE_16K on oracle debleed channels.
- `diarize_clarin_2speakers.py` — pyannote over the full 2-speaker download → `diarization/<id>.json`.
- `transcribe_clarin_2speakers.py` — WhisperX over the full 2-speaker download, raw and MossFormerGAN-enhanced.
- `score_fragment_acoustics.py` — objective acoustic-complexity scorer for the 128 CLARIN eval fragments (SQUIM, DNSMOS ONNX, Brouhaha SNR/C50, WADA-SNR, LUFS, clipping), calibrated vs the author's 16 by-ear grades; writes `acoustic_scores.csv` + `ACOUSTIC_SCORES_REPORT.md` beside the fragments. Brouhaha runs in an isolated venv (`/tmp/brouhaha_venv`, override `BROUHAHA_VENV_PY`/`BROUHAHA_CKPT`) because its pins (numpy 1.x, pyannote.audio ≤3.3.0) conflict with the main venv; if absent, the script falls back to WADA-SNR and says so in the report.
- `pixit_worker.py` — B3 PixIT (joint diarization+separation, `pyannote/speech-separation-ami-1.0`) subprocess worker in the isolated `~/pixit_venv` (`$PIXIT_VENV_PY`; pin recipe in the docstring). UNTESTED until the author accepts the model's HF gate (token 403s as of 2026-07-18); plan: `docs/fable_plans/b3_pixit_arm.md`.
- `compare_asr.py` — held-out WhisperX-vs-Cohere comparison via a **fixed-audio ASR-only swap** (holds the dr_refineplus per-speaker streams constant, varies only the transcriber). Prereq: `sweep_pipeline.py --configs dr_refineplus --recordings <ids>` to produce the streams + WhisperX transcripts; then `compare_asr.py --split test` (needs `$COHEREX_VENV_PY`) adds the Cohere pass (reuses the wired `_CohereXBackend`), dumps per-fragment bundles (`whisperx_/cohere_/gt_{A,B}.txt`) for the per-transcript eyeball pass, and scores cpWER/cpCER per-fragment + micro-avg under `<eval>/_forensics/asr_compare/`. `--gt2-root` adds a second, differently-seeded GT for the cross-seed read. Background: cross-arch WER is **reference-seed biased (~±4 pp)** — each hand-corrected GT mildly flatters the ASR it was seeded from (`_forensics/ANCHORING_EYEBALL_SYNTHESIS.md`); read numbers two ways + use the reference-free eyeball as tiebreak.

**Training Flow:** `train.py` → config → `training/setup.py` builders (`build_dataloaders` / `build_trainer`, shared with `train_sweep.py` — the two mains keep only their genuine differences) → `create_model_from_config()` → optional `torch.compile()` → `Trainer` (AMP, grad accumulation, checkpointing, curriculum learning). Every checkpoint embeds a provenance manifest (`utils.collect_run_manifest`: git SHA+dirty, torch/CUDA/cuDNN/mamba-ssm versions, GPU, hostname, seed, argv, W&B run id) and writes a human-readable `run_manifest.yaml` beside `config.yaml`; on `--resume` the saved W&B run id is reused (`resume="must"`). Old checkpoints without these keys load fine.

**Checkpoints:** Saved to `checkpoints/{model_type}/{task}/{run_name}/`. Run name comes from W&B when available, otherwise timestamp. Each directory includes `config.yaml` for reproducibility. By default only the best checkpoint is kept; `save_all_checkpoints: true` keeps every improvement.

## Experiment Configs

Each model architecture has its own YAML configs in `experiments/`. When creating new experiments, always base them on an existing YAML file from `experiments/` — do not write configs from scratch. The YAML structure mirrors the config dataclasses:

```yaml
data:
  dataset_type: polsess
  batch_size: 16
  task: SB                    # ES=enhance 1 speaker, EB=enhance both, SB=separate both

model:
  model_type: dprnn           # Must match a key in the model registry
  dprnn:                      # Nested params matching the model's param dataclass
    N: 64
    hidden_size: 128
    num_layers: 6

training:
  num_epochs: 100
  lr: 0.00015
  use_amp: true
  use_wandb: true
  curriculum_learning:        # Optional: progressive variant introduction
    - epoch: 1
      variants: ["C", "R"]
    - epoch: 8
      variants: ["R", "SR", "S", "SE", "ER", "E", "SER"]
      lr_scheduler: start     # Optional: gate LR scheduler to this stage
```

## Key Technical Details

- **AMP:** Enabled by default. SpeechBrain EPS patched from 1e-8 to 1e-4 in `utils/common.py` to prevent float16 underflow (`apply_eps_patch` — patches **ConvTasNet's SpeechBrain lobe only**; other architectures are unaffected by it). Most models use float16 + GradScaler; Mamba models **and MossFormer2** use bfloat16 without GradScaler (dispatch on `model_type` in `training/trainer.py:_setup_amp`). MossFormer2's squared-ReLU attention overflows fp16 once activations sharpen — first seen as NaN val SI-SDR (fixed by fp32 validation), then as training NaNs at low `attn_dropout` / higher LR in the 128k sweep.
- **Training determinism (`training.deterministic`, tri-state):** unset/`null` (default) = legacy behavior — cuDNN deterministic + no benchmark, `use_deterministic_algorithms` NOT called; `true` = strict opt-in (`torch.use_deterministic_algorithms(warn_only=True)` + `CUBLAS_WORKSPACE_CONFIG`); `false` = `cudnn.benchmark` conv-autotune speedup, non-deterministic. Train DataLoaders use a seeded `torch.Generator` + `worker_init_fn`, so the MM-IPC variant stream is reproducible by contract. Gotcha: curriculum learning mutates the dataset's `allowed_variants` in place and only works because workers re-fork each epoch — never enable `persistent_workers` with a curriculum active.
- **torch.compile:** Auto-applied on Linux for ~10-20% speedup. Checkpoint loading handles `_orig_mod` prefix. Skipped for Mamba models. `mossformer2` is compiled with `dynamic=False` (per-shape static specialization): its vendored rotary block disables the seq-len cache (`cache_if_possible=False`) and its token-shift/group-rearrange can't be lowered under symbolic shapes — so fixed-length crops compile once, new lengths trigger a one-time static recompile. Per-architecture dispatch lives in `compile_for_model_type` (`utils/model_utils.py`), shared by `train.py` and `train_sweep.py`.
- **MM-IPC (Mix Modification by Inverted Phase Cancellation):** Augmentation that randomly varies background complexity during training by subtracting audio layers from the full mix. Indoor variants (with reverb): SER/SR/ER/R. Outdoor variants (no reverb): SE/S/E/C. Letters indicate what's present: S=scene, E=event, R=reverb, C=clean. Implemented via lazy loading in `datasets/polsess_dataset.py`. Validation uses deterministic selection (seeded by sample index).
- **Curriculum Learning:** Configure in YAML `training.curriculum_learning`. Progressive variant introduction + optional LR scheduler gating. Note: when curriculum learning is active, the LR scheduler is **disabled by default** until a curriculum entry includes `lr_scheduler: start` — omitting this key means the scheduler never runs.
- **Gradient Accumulation:** `training.grad_accumulation_steps` for effective batch scaling.
- **Mamba Models (SPMamba, Mamba-TasNet, DPMamba):** Require Linux + CUDA + `mamba-ssm`. AMP uses bfloat16 (no GradScaler) — Mamba CUDA kernels run float32 internally. Mamba-TasNet/DPMamba come in XS/S/M/L size configs. `models/mamba/` contains BiMamba building blocks adapted from xi-j/Mamba-TasNet.

## PolSESS Dataset Structure

```
PolSESS/
├── train/
│   ├── clean/         # Clean speech (target for ES task)
│   ├── event/         # Event sounds
│   ├── mix/           # Full mixed audio
│   ├── scene/         # Background scene
│   ├── sp1_reverb/    # Speaker 1 with reverb
│   ├── sp2_reverb/    # Speaker 2 with reverb
│   ├── ev_reverb/     # Event with reverb
│   └── corpus_PolSESS_C_in_train_final.csv
├── val/
└── test/
```

MM-IPC works by subtracting layers from the full mix using inverted phase cancellation. For example, the "SR" variant (scene + reverb) is created by removing the event layer from the full SER mix.

## Common Pitfalls

1. **NaN in SI-SDR:** AMP underflow — EPS patch should handle it. If not, `use_amp: false`. The trainer skips NaN/Inf batches; after 1000 consecutive NaN batches it aborts the run (`ConsecutiveNaNError` → `SystemExit(1)`, sweep-friendly — see `MAX_CONSECUTIVE_NAN_BATCHES` in `training/trainer.py`).
2. **Memory overflow:** Reduce `batch_size`, use `grad_accumulation_steps` to compensate.
3. **Config precedence:** CLI > YAML > env vars > defaults. Additionally, `Config.__post_init__` forces the model's output source count (`C`/`n_srcs`) to match the task (ES→1, SB/EB→2), overriding whatever the YAML says — it prints a line when it actually changes the value.
4. **MambaTasNet NaN:** Deep configs need `residual_in_fp32: true`.  `grad_clip_norm: 1.0` (not 5.0) might help too. Historical: every Mamba run before 2026-04-19 silently trained *and validated* in fp16 — torch.compile's `OptimizedModule` wrapper defeated the old class-name AMP dispatch (fixed in `2327c89`; compile disabled for Mamba in `c99b56c`) — so pre-fix NaN lore (including this pitfall's origin) dates from that regime.
5. **Mamba on Windows:** Requires WSL2 + CUDA toolkit 12.4+. Non-Mamba models work natively.
6. **Sweep config access:** `load_config_for_run(wandb.config)` uses `getattr`, not dict access.
7. **`--resume` extends the epoch budget:** `Trainer.train` iterates `range(current_epoch, current_epoch + num_epochs)`, so `num_epochs` is *additional* epochs, not the total. A run resumed at epoch 30 with `num_epochs: 80` runs to epoch 110, and its early-stopping patience counter restarts at 0. There is no `--epochs` CLI override. For seed replicates or any budget-matched comparison, stop the resumed run at the intended total and report best-within-budget. Note also that `Trainer` logs with an **explicit step** (`wandb.log(metrics, step=self.current_epoch)`). W&B steps are unique and must increase, so when a resume restarts below the highest step already logged, those overlapping epochs add **no new rows** — the history keeps the pre-crash values there and the chart only moves again once the epoch counter passes the previous maximum. A run resumed from an early "best" checkpoint after running well past it can therefore look frozen on W&B for many epochs while training normally. Separately, W&B does not clear the `crashed` badge on `resume="must"`, and the resumed run's console pane is overwritten from line 1 (fresh stdout capture) — both cosmetic.

## Virtual Environments

**Main (`venv/`)** — all models except SPMamba3. Alias: `polsess_venv`.

**SPMamba3 (`venv_mamba3/`)** — torch 2.11.0+cu130, triton 3.6.0, Mamba-3 kernels. Clone of main venv with Mamba-3 files manually copied from bare repo clone of `state-spaces/mamba`. Additional deps: `tilelang`, `quack-kernels`, `cuda-bindings`, `nvidia-cutlass-dsl`.

## Cloud Setup

`setup.sh` provisions a fresh cloud GPU instance (Vast.ai / RunPod): clones the repo (`REPO_BRANCH`/`REPO_URL` env-overridable), installs deps, sets env vars; `--rclone` additionally configures the Google Drive remote (headless token flow). Start from a cu128 PyTorch image (`pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel`) — Blackwell GPUs (sm_120) need exactly that; the `-devel` variant has nvcc for a from-source `MAMBA_FORCE_BUILD=TRUE` fallback if the mamba-ssm wheel lacks the arch. `download_dataset.sh` fetches one corpus by name — `DATASET_NAME=PolSESS_C_new_64 ./download_dataset.sh` (default `PolSESS_C_final_128_v2`; pilot `PolSESS_C_both` is a .tar.gz) — and wires `POLSESS_DATA_ROOT` to the directory containing `train/` (auto-detects the double-nested layouts of C_new_64/C_both vs the flat 128_v2). With several corpora on one box, pass `--data-root` per launch rather than trusting the env var. Neither script logs into W&B — run `wandb login` before training (setup.sh's verification warns if credentials are missing).

## Environment Variables

- `POLSESS_DATA_ROOT` — PolSESS dataset path (default in `config.py`)
- `REALM_DATA_ROOT` — REAL-M dataset (default: `~/datasets/REAL-M-v0.1.0/`)
- `LIBRIMIX_ASR_ROOT` — LibriSpeechMixASR (default: `~/datasets/LibriSpeechMixASR/`)
- `HF_TOKEN` — HuggingFace token for the ASR pipeline's pyannote diarization stage
- `AP_BWE_CHECKPOINT` — ASR pipeline post-separation AP-BWE backend (default: `~/AP-BWE/checkpoints/8kto16k/g_8kto16k.zip`). FlowHigh needs no env var (auto-downloads).
- `COHEREX_VENV_PY` — path to the isolated CohereX venv's python (e.g. `~/asr_model_compare/coherex_venv/bin/python`), required by `transcription.backend: coherex`. No default; missing → loud crash (SCOPE §4, no silent fall-back).
- `SORTFORMER_VENV_PY` — path to the isolated NeMo venv's python (`~/sortformer_venv/bin/python`), required by `diarization.backend: sortformer` — which the shipped best ASR-pipeline config uses. No default; missing → loud crash (SCOPE §4).
- `PIXIT_VENV_PY` — path to the isolated PixIT venv's python (`~/pixit_venv/bin/python`), used by `scripts/pixit_worker.py` (B3 arm; not consumed by `asr_pipeline/` itself).
