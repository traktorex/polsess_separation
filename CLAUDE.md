# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Repository Overview

PyTorch speech separation on the PolSESS dataset — ConvTasNet, SepFormer, DPRNN, MossFormer2, TF-MossFormer, SPMamba, Mamba-TasNet, DPMamba — published for consumers as the `polsess-models` package. Master's thesis on speech separation as preprocessing for Polish ASR. `README.md` is the reader-facing overview; this file holds what an agent needs beyond it. The thesis has already been handed in — we're in post-thesis development era.

The ASR pipeline (thesis ch6) and its showcase webapp (ch7) **are no longer in this repository** (removed 2026-10-23, author's ruling: the paper's experiments are redone from scratch). They develop in `~/DGSS-ASR` (github.com/traktorex/DGSS-ASR), which consumes separators from here as inference bundles. The thesis-era instrument is tag `pre-asr-removal-2026-10-23` (`git show <tag>:asr_pipeline/...`); the hand-in state is tag `handin-2026-09-05`. Do not re-add any of it here; a question about pipeline behaviour is a DGSS-ASR question.

Thesis prose and experiment logs live in `thesis/` — a symlink to an Obsidian vault on Windows with its own git repo (ignored here). Read freely; when editing thesis content, `cd thesis/` so `thesis/CLAUDE.md` stacks on top of this file.

The on-device browser demo lives on branch `experiment/webapp-ondevice`, is not in the thesis, and must not be re-added to `main`.

Thesis-era local artifacts (old evaluation CSVs, the legacy `checkpoints_for_eval*.csv` lists, raw W&B exports, superseded benchmark tables, the June deep-research notes) live in `archive/thesis_era/` — ignored by git, indexed by its `README.md`. New result files go to an explicit `--output` path, not the repository root.

## Style

Reply in English unless the user specifies otherwise.
The amount of thinking should be proportional to the complexity of the task you're given.
Avoid unnecessary verbosity by using CoT to structure your response.

Subagent models and the host rules are in `~/.claude/CLAUDE.md`. This repository's standing exception to "never Fable subagents" (approved 2026-07-25): `review-synthesizer`, `code-review-synthesizer` and `redaktor` pin `model: fable` in their frontmatter, because adjudicating between conflicting reviewers and premium prose editing merit the strongest reasoning. Do not extend it to other agents without asking.

Review gates (author, 2026-10-07): `/deep-review` runs only when asked. (The earlier rule about bugfixes to the frozen `asr_pipeline/` copy is moot since its removal on 2026-10-23.)

## Thesis Code Principles

This code will be reviewed by academic supervisors. Prioritize: clarity over cleverness, simplicity over abstraction, reproducibility. Prefer explicit implementations that match cited papers. Don't over-engineer — no deep inheritance, no speculative features. Experiment logging is handled outside this repo.

## Keeping This File Current

After any substantial change — new top-level script or subsystem, new env var, new dataset/model/task variant, new gotcha worth flagging, removed commands, or changed config precedence — propose a targeted edit to this CLAUDE.md. Update in place; don't rewrite from scratch. Skip for routine bugfixes, refactors, one-off experiments, or a single-setting change in one function — even when it changes observable behaviour. The bar is: would a fresh session make a wrong decision without this line? If the answer is only "it wouldn't know the history", the note belongs in the commit message, not here. Experiment results, decision history and "X was removed on DATE" notes belong in the thesis log and git history, not here.

## Dataset Variants

- `PolSESS_C_both` — old 8k-effective dataset (half was duplicated). Used for early baselines, HPO, and HPO validation runs.
- `PolSESS_C_new_64` = `C_new_64` — correct 64k dataset generated 2026-04-15. Use `train_max_samples=16000` / `32000` / full for 16k / 32k / 64k scaling experiments.
- `PolSESS_C_final_128_v2` — 128k dataset for final training runs. Contains other-language speech alongside Polish.
- `PolSESS_C_128_16kHz` — 16 kHz twin of 128_v2 (same six-run recipe, `dataSources_v3`, `outputFreq = 16000`; generated 2026-09-07/08). Same layout: `train/` 128,000 rows interleaved indoor/outdoor, `val/` = 1,000-item stratified subset hardlinked from `val_big/` (12,800), `test/` 12,800. `download_dataset.sh` does not know this corpus. `val_big/` exists only on the generating box (3080); copies elsewhere (the 4070) carry only `train/`/`val/`/`test/`.

## W&B Projects

- `polsess-separation` — standalone runs (baselines, HPO validation), uses `PolSESS_C_both`.
- `polsess-thesis-experiments` — sweeps.
- `polsess-separation-real16k` / `polsess-separation-32k` / `polsess-separation-64k` — scaling runs on subsets of `PolSESS_C_new_64` (NB only the 16k project carries the `real` prefix).
- `polsess-separation-128k` — final runs on the 128k dataset.
- `polsess-separation-16kHz-128k` — 128k finals on the 16 kHz corpus (`PolSESS_C_128_16kHz`). Separate project on purpose: SI-SDR at 16 kHz is a different number on a different signal, not comparable with the 8 kHz rows.

## Key Commands

**Training:**
```bash
python train.py --config experiments/dprnn/dprnn_baseline.yaml
python train.py --config experiments/spmamba/spmamba_sb_reduced.yaml --no-wandb --seed 123
python train.py --config experiments/tf_mossformer/s_8k.yaml   # paper recipe: adamw + warmup
python train.py --resume checkpoints/dprnn/SB/run_name/dprnn_SB_best.pt
```

**Evaluation:**
```bash
python evaluate.py --checkpoint checkpoints/dprnn/SB/run_name/dprnn_SB_best.pt   # PolSESS, all MM-IPC variants
python evaluate.py --checkpoint path/to/model.pt --variant SER
python evaluate.py --checkpoint path/to/model.pt --no-pesq --no-stoi               # fast
python evaluate.py --checkpoint path/to/model.pt --dataset librimix --librimix-root /home/user/datasets/LibriMix/Libri2Mix
python evaluate.py --checkpoint path/to/model.pt --output results.csv
# Batch eval of the thesis checkpoint set (experiments/thesis_eval_manifest.csv by default;
# bs=1 exact per-sample scoring → aggregate CSV + <stem>_per_sample.csv, provenance columns)
python evaluate_all.py --resume
```

**Testing:**
```bash
pytest
pytest --cov=. --cov-report=html
pytest tests/test_model.py -v
```
The suite runs on a CPU-only machine without `mamba-ssm`: Mamba-family, CUDA and dataset-dependent tests skip instead of erroring.

**Sweeps:**
```bash
wandb sweep sweeps/3-hyperparam-opt/dprnn/stage1/dprnn.yaml   # sweep YAML points at train_sweep.py; then run agents against the sweep ID
```

**Benchmarks (objective axes of the ch5 multi-axis table; quality/epochs/wall-clock come from W&B, not from here):**
```bash
python scripts/benchmark_inference.py --cross-check   # MACs, latency+IQR, RTF, peak infer VRAM
python scripts/benchmark_training.py                  # train-only throughput @ trained bs AND bs=1, peak train VRAM
```
Both read the shared architecture list in `scripts/benchmark_models.py` and write `docs/generated/benchmark_{inference,training}.csv` with provenance columns. ptflops needs four corrections (MHA double count, `einsum`, `F.scaled_dot_product_attention` incl. banded cost under a boolean mask, Mamba scan FLOPs-vs-MACs) — documented in the inference script's docstring, tested in `tests/test_benchmark_macs.py`, audited in `thesis/thesis-log/sweep_plan/ch5_test_evals/BENCHMARK_AUDIT.md`. Counting is device-independent (`--device cpu` for non-Mamba); everything timed must run on one machine in one session. The superseded 2026-04 CSVs moved to the local `archive/thesis_era/`.

**Thesis audit artifacts (citable, CPU-only unless noted):**
```bash
python scripts/audit_mmipc.py           # MM-IPC reconstruction lossless to the 16-bit PCM floor (exit≠0 on violation)
python scripts/audit_split_leakage.py   # train↔{val,test} path disjointness
python scripts/model_manifest.py        # per-config param counts → docs/generated/model_manifest.{csv,md}
```

**Interactive:**
```bash
jupyter notebook test_model_interactive.ipynb
```

**Exporting a separator for DGSS-ASR:** `python scripts/export_separator.py <checkpoint.pt> <bundle_dir>` writes `weights.safetensors` + `separator.json`; release rules in `models/README.md`.

## Architecture Overview

**Configuration (`config.py`):** Dataclasses (`DataConfig`, `ModelConfig`, `TrainingConfig`) with nested model/dataset params. Priority: defaults < env vars < YAML < CLI args. Use `get_config_from_args()` for CLI, `load_config_for_run(wandb.config)` for sweeps.

**Model Registry (`models/__init__.py`):** Dict-based. `get_model("name")` returns class. Mamba models auto-excluded without `mamba-ssm`. Measured param counts per config: `docs/generated/model_manifest.md`.
- Cross-platform: `convtasnet` (~8.7M for SB; the oft-quoted 8.64M is the ES/C=1 build), `sepformer` (~26M), `dprnn` (~2-3M), `mossformer2` (matched ~26M / full ~55.7M), `tf_mossformer` (S/M/L: 5.92 / 17.34 / 26.00M).
- Linux + CUDA only: `spmamba` (~1.2M), `mamba_tasnet` and `dpmamba` (XS/S/M/L: ~2.2-60M). `models/mamba/` holds BiMamba blocks adapted from xi-j/Mamba-TasNet.
- `mossformer2` (Zhao et al. 2023, arXiv:2312.11825): files in `models/mossformer2/` are **vendored** from ClearerVoice-Studio; `__init__.py` is the project wrapper. Single `N` knob = encoder dim = transformer dim (must match upstream); `num_blocks` = GFSMN depth (24 paper-full, 11 ≈ SepFormer-matched); `attn_dropout` covers the attention path only (FSMN-gate dropout fixed at 0.1). Sweep key `dropout` routes to `attn_dropout`. Configs: `experiments/mossformer2/`.
- `tf_mossformer` (Zhao et al. 2026, arXiv:2607.21128): TF separator with convolution-gated local + global attention. **Re-implemented from the paper — no upstream code exists**; never call it "vendored". `tf_locoformer_blocks.py` is a copy of the Apache-2.0 TF-Locoformer file (the byte-identical vendored original now lives in DGSS-ASR; `models/` depends on nothing outside itself); `local_global_attention.py` and `__init__.py` are original. Our sizes run ~2.5% above the paper's M/L rows because the paper's own rows are mutually inconsistent — quote the caveat wherever the size appears; reconciliation in `thesis/x_notes/tf_mossformer/tf_mossformer_spec.md` §4. Knobs follow paper Table 1 (`D`, `num_blocks`, `ffn_hidden_dim`, `conv_kernel_size`/`conv_stride`, `n_heads`, `num_groups`, `window_t`/`window_f`, `gate_kernel_size`, `n_fft`/`hop_length` in **samples**, `attn_dropout`; sweep key `dropout` → `attn_dropout`). Its configs (`experiments/tf_mossformer/{s,m,l}_8k.yaml` 16k rows; `{s,m}_128k_final.yaml` the 8 kHz 128k finals and `s_128k_16kHz_final.yaml` the 16 kHz one, all on the all-8 no-curriculum protocol with ES patience 8 instead of the paper's 10) are the only ones on the paper's recipe (`adamw`, `warmup_steps: 4000`, wd 1e-2). Plateau patience: the paper's is 3, but every run since commit c111f8d (2026-09-08) has `lr_patience: 1` — YAML headers that still say "patience 3" are stale (only `16k_tf_mossformer_s_42_originalLR` ran with 3). The paper's bs 4 / lr 1e-3 doesn't fit 12 GB, so they run the batch that fits with LR scaled linearly (S bs 2 / 5e-4, M and L bs 1 / 2.5e-4, S at 16 kHz bs 1 / 2.5e-4, no accumulation — author ruling 2026-09-07). 4070: S ≈ 4.4 GiB and ~29 min per 16k epoch, M 6.1 GiB, L 9.2 GiB. 3080 at 16 kHz: S bs 1 compiled 4.4 GiB, ~6.5 h per 128k epoch (bs 2 spills past 10 GB).

**Inference package (`pyproject.toml`, `models/inference.py`):** `models/` is also published as the pip-installable `polsess-models` distribution, import name `polsess_models` — hatchling maps the directory, nothing is copied, and code in this repository keeps importing `models` from the working directory. Other projects (so far only DGSS-ASR, the extracted ASR pipeline) consume *inference bundles* (`weights.safetensors` + `separator.json`) written by `scripts/export_separator.py` and read by `load_separator`; `load_model_for_inference` (training checkpoints) lives in the same file and is re-exported by `utils`. Code under `models/` uses relative imports only and nothing from `config` / `utils` / `training` (`tests/test_polsess_models_package.py` enforces it); the Mamba family and `factory.py` are excluded from the wheel. A consumer-visible change in `models/` means bumping `models.__version__` and tagging `polsess-models-vX.Y.Z` — rules in `models/README.md`.

**Dataset Registry (`datasets/__init__.py`):** Dict-based. `get_dataset("name")` returns class. Supports `polsess`, `libri2mix`, `echoset`. The EchoSet corpus was deleted 2026-08-03 (loader kept as the record of a negative result): its targets are reverberant while PolSESS SB targets are dry, so PolSESS-trained models score ≈ −2.9 dB SI-SDRi there — disqualified for generalization numbers. Notes: `thesis/thesis-log/sweep_plan/ch5_test_evals/CH5_TEST_EVALS_NOTES.md`. (Unrelated: DGSS-ASR's `b1_tiger_echoset.yaml` is the TIGER checkpoint *trained on* EchoSet.)

**Training Flow:** `train.py` → config → `training/setup.py` builders (`build_dataloaders` / `build_trainer`, shared with `train_sweep.py`) → `create_model_from_config()` → optional `torch.compile()` → `Trainer` (AMP, grad accumulation, checkpointing, curriculum learning). Every checkpoint embeds a provenance manifest (`utils.collect_run_manifest`: git SHA+dirty, library versions, GPU, hostname, seed, argv, W&B run id) and writes `run_manifest.yaml` beside `config.yaml`; `--resume` reuses the saved W&B run id (`resume="must"`). Old checkpoints without these keys load fine.

**Checkpoints:** `checkpoints/{model_type}/{task}/{run_name}/`, run name from W&B when available, otherwise timestamp. Each directory includes `config.yaml`. Only the best checkpoint is kept unless `save_all_checkpoints: true`.

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

- **Corpus sampling rate (`data.sample_rate`, default 8000):** loaders and models are rate-agnostic (kernel/stride/n_fft in samples, chunk sizes in frames), so a 16 kHz PolSESS render with the same layout trains without code changes. The field is provenance plus a guard: `PolSESSDataset` checks the first mix file's header at construction and raises on mismatch; `evaluate.py` / `evaluate_all.py` take the rate from the checkpoint (pre-field checkpoints = 8 kHz) to choose PESQ nb/wb, the STOI rate and the Libri2Mix `wav8k/`/`wav16k/` folder. Nothing rescales geometry: at 16 kHz a clip has 2× the samples, so keeping the frame rate (32/16 encoders, 512/128 STFT) is a per-architecture choice in the 16 kHz YAMLs. (Downstream, DGSS-ASR reads the rate from the exported bundle's `separator.json`; a 16 kHz separator must not be followed by AP-BWE there.) 16 kHz configs: `experiments/tf_mossformer/s_128k_16kHz_final.yaml`, `experiments/mossformer2/6-final-training/128k_{matched,full}_16kHz_final.yaml`, and `s_16k.yaml` (16k-subset row; the three OPEN QUESTIONS in its header — W&B project choice, `window_f` at F=129, ASR handoff — are still live judgement calls). The committed `config.py` default `data_root` is `~/datasets/PolSESS_C_128_16kHz/` on **every** machine, so the 16 kHz configs need no `--data-root` and the **8 kHz configs do** (`--data-root ~/datasets/PolSESS_C_final_128_v2` or `POLSESS_DATA_ROOT`). TF-MossFormer at 16 kHz needs torch ≥ 2.14: on the 4070 run it from `venv_t214/`.
- **SI-SDRi is zero-mean on both terms since 2026-09-22** (`utils/metrics.py` refuses a `zero_mean=False` metric): the same checkpoint now scores ~0.07 dB lower than in thesis tables and in its own saved `val_sisdr`. A constant offset, not a regression — just don't mix pre- and post-fix numbers in one comparison.
- **AMP:** on by default. SpeechBrain EPS patched 1e-8 → 1e-4 in `utils/common.py` (`apply_eps_patch`, ConvTasNet's SpeechBrain lobe only). Most models: float16 + GradScaler. Mamba models, MossFormer2 and TF-MossFormer: bfloat16 without GradScaler (dispatch on `model_type` in `training/trainer.py:_setup_amp`) — MossFormer2's squared-ReLU attention and TF-MossFormer's gated products overflow fp16. TF-MossFormer casts to fp32 around `torch.stft` / `torch.complex` / `torch.istft` inside its forward, like SPMamba (`torch.complex` rejects bf16). Validation runs in fp32.
- **Training determinism (`training.deterministic`, tri-state):** unset/`null` = cuDNN deterministic + no benchmark, `use_deterministic_algorithms` not called; `true` = strict (`torch.use_deterministic_algorithms(warn_only=True)` + `CUBLAS_WORKSPACE_CONFIG`); `false` = `cudnn.benchmark`, non-deterministic. Train DataLoaders use a seeded `torch.Generator` + `worker_init_fn`, so the MM-IPC variant stream is reproducible. Gotcha: curriculum learning mutates `allowed_variants` in place and only works because workers re-fork each epoch — never enable `persistent_workers` with a curriculum active.
- **torch.compile:** auto on Linux (~10-20% speedup); loading handles the `_orig_mod` prefix; skipped for Mamba models. `mossformer2` and `tf_mossformer` compile with `dynamic=False` (their rotary block can't be lowered under symbolic shapes) — fixed-length crops compile once, a new length triggers a one-time recompile. Dispatch: `compile_for_model_type` in `utils/model_utils.py`. The eval-mode / no_grad / fp32 validation forward is a *second* compiled graph; `Trainer._preflight_validation` runs it on one batch per validation loader before epoch 1, so a compile failure there costs seconds instead of an epoch (torch 2.11's Inductor could not compile TF-MossFormer's eval graph at 16 kHz geometry; 2.14 can).
- **MM-IPC (Mix Modification by Inverted Phase Cancellation):** augmentation that varies background complexity by subtracting layers from the full mix. Indoor (reverb): SER/SR/ER/R/C. Outdoor: SE/S/E/C. Letters indicate what's present: S=scene, E=event, R=reverb (C=clean speech only). Lazy loading in `datasets/polsess_dataset.py`; validation picks variants deterministically (seeded by sample index).
- **Curriculum Learning:** `training.curriculum_learning` — progressive variant introduction + optional LR-scheduler gating. With a curriculum active the LR scheduler is **disabled** until an entry includes `lr_scheduler: start`.
- **Optimizer + warmup (`training.optimizer`, `training.warmup_steps`):** `adam` (default, every pre-2026-09 run) or `adamw` (decoupled `weight_decay`); `warmup_steps` (default 0) ramps the LR linearly to `training.lr`, then `ReduceLROnPlateau` takes over. Added for TF-MossFormer's published recipe; sweep-overridable; defaults are byte-compatible with every existing config.
- **LR scheduler + early stopping:** `ReduceLROnPlateau(mode=max, threshold=1e-4 rel)` cuts after `lr_patience+1` non-improving epochs. Logged `train_lr` at epoch e is the post-step value (the LR of e+1) and `epochs_no_improvement` lags one epoch — verify W&B epoch alignment against box logs. Rule from the 419-run analysis (`thesis/x_notes/lr_schedule_analysis/FINDINGS.md`): set `early_stopping_patience = 2·(lr_patience+1)` (6 / 4) instead of 15; never 0; patience 1 vs 2 is open.
- **Gradient Accumulation:** `training.grad_accumulation_steps`.
- **Mamba Models:** Linux + CUDA + `mamba-ssm` (Windows: WSL2 + CUDA toolkit 12.4+). Kernels run float32 internally. Deep Mamba-TasNet configs need `residual_in_fp32: true`; `grad_clip_norm: 1.0` may help. Every Mamba run before 2026-04-19 silently trained and validated in fp16 (compile wrapper defeated the AMP dispatch; fixed in `2327c89`), so pre-fix NaN lore dates from that regime.

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

MM-IPC subtracts layers from the full mix by inverted phase cancellation, e.g. "SR" (scene + reverb) = full SER mix minus the event layer.

## Common Pitfalls

1. **NaN in SI-SDR:** AMP underflow — the EPS patch should handle it; otherwise `use_amp: false`. The trainer skips NaN/Inf batches and aborts after 1000 consecutive ones (`ConsecutiveNaNError` → `SystemExit(1)`; `MAX_CONSECUTIVE_NAN_BATCHES` in `training/trainer.py`).
2. **Memory overflow:** reduce `batch_size` or compensate with `grad_accumulation_steps`.
3. **Source count follows the task:** `Config.__post_init__` forces the model's source count (`C`/`n_srcs`) to match the task (ES→1, SB/EB→2) and prints a line when it changes the value.
4. **Sweep config access:** `load_config_for_run(wandb.config)` uses `getattr`, not dict access.
5. **`--resume` extends the epoch budget:** `num_epochs` is *additional* epochs (a run resumed at 30 with `num_epochs: 80` runs to 110) and the early-stopping counter restarts at 0; there is no `--epochs` override. For budget-matched comparisons stop at the intended total and report best-within-budget. `Trainer` logs with `step=epoch`, so a resume that restarts below the highest logged step adds no W&B rows until it passes it — the run looks frozen on W&B while training normally.

## Virtual Environments

**Main (`venv/`)** — torch 2.14.0+cu130, triton 3.8, cuDNN 9.24, speechbrain 1.1.1, mamba-ssm 2.3.2.post1 and causal-conv1d 1.7.0 built from source. All models except SPMamba3. Alias: `polsess_venv`. That describes the 3080. The 4070 machine deliberately keeps both and has not repeated the swap (author ruling 2026-09-22, not to be re-proposed): `venv/` (torch 2.8.0+cu128, 356 packages) still carries the former ASR-pipeline extras (whisperx, faster-whisper, ctranslate2, clearvoice, python-pptx, jupyter) that the 2.14 build lacks; since the pipeline's removal it backs only the deck build (`scripts/build_prezentacja.py`), the notebooks and the default `pytest` run, while `venv_t214/` (torch 2.14.0+cu130, 199 packages) is the one to use for new training there — including anything at 16 kHz geometry. `requirements.txt` is the installable core set (optional extras — mamba-ssm, the figure/deck-build packages — are commented blocks); `requirements-freeze.txt` is the exact `pip freeze` of the 3080 venv. Installed on top of the core set there: the former webapp/ASR deps (fastapi, uvicorn, python-multipart, pyannote.audio, meeteval, num2words — nothing here imports them since 2026-10-23), ipykernel/ipywidgets for the notebooks, jiwer, openai-whisper, lxml, black, mypy. Deliberately not carried over from the old venv (nothing in the repo imports them): torchvision, onnx, GitPython, pymupdf/pdfplumber/pypdf. Third-party attribution and vendored-code licences: `THIRD_PARTY.md`. Build gotchas for torch ≥ 2.14: the mamba-ssm / causal-conv1d sdists hard-code `-std=c++17` and ATen now requires C++20 — patch `setup.py` to `-std=c++20` and install with `--no-build-isolation` (official wheels stop at torch 2.10; `setup.sh`'s `MAMBA_FORCE_BUILD` fallback hits the same wall). Both `setup.py` files also hard-code a sm_75…sm_121 `-gencode` list that makes `torch.utils.cpp_extension` ignore `TORCH_CUDA_ARCH_LIST`; on the 3080 (nvcc 13.0.88) that all-arch build segfaulted in `cicc` — delete the list so the env var (`8.6`) is honoured, which also cuts each build to ~2 min.

**SPMamba3** — side project, not in the thesis; its `venv_mamba3/` no longer exists (removed 2026-09-10). Rebuild recipe if ever needed: copy of the main venv + mamba-ssm built from source (`state-spaces/mamba`@0048fbf2) + triton ≥ 3.5 (the SISO kernel needs it); the model code lives on branch `feature/spmamba3-rebuild`.

## Cloud Setup

`setup.sh` provisions a fresh cloud GPU instance (Vast.ai / RunPod): clones the repo (`REPO_BRANCH`/`REPO_URL` env-overridable), installs deps, sets env vars; `--rclone` additionally configures the Google Drive remote. Start from `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel` — Blackwell (sm_120) needs exactly that, and `-devel` has nvcc for a `MAMBA_FORCE_BUILD=TRUE` from-source fallback. `DATASET_NAME=PolSESS_C_new_64 ./download_dataset.sh` fetches one corpus (default `PolSESS_C_final_128_v2`) and wires `POLSESS_DATA_ROOT` to the directory containing `train/` (auto-detects the nested layouts). With several corpora on one box, pass `--data-root` per launch rather than trusting the env var. Neither script logs into W&B — run `wandb login` first.

## Environment Variables

- `POLSESS_DATA_ROOT` — PolSESS dataset path (default in `config.py`).
- The pipeline's variables (`HF_TOKEN`, `AP_BWE_CHECKPOINT`, `SORTFORMER_VENV_PY`, `COHEREX_VENV_PY`, …) belong to DGSS-ASR now.
