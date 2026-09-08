# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Repository Overview

PyTorch speech separation on the PolSESS dataset — ConvTasNet, SepFormer, DPRNN, MossFormer2, TF-MossFormer, SPMamba, Mamba-TasNet, DPMamba — plus an ASR pipeline (`asr_pipeline/`) and a defense-demo webapp (`webapp/`) that consume the trained separators. Master's thesis on speech separation as preprocessing for Polish ASR. `README.md` is the reader-facing overview; this file holds what an agent needs beyond it. The thesis has already been handed in — we're in post-thesis development era.

Thesis prose and experiment logs live in `thesis/` — a symlink to an Obsidian vault on Windows with its own git repo (ignored here). Read freely; when editing thesis content, `cd thesis/` so `thesis/CLAUDE.md` stacks on top of this file.

Subsystem guidance lives beside the code and loads when you work there: `asr_pipeline/CLAUDE.md` (stages, configs, CLI, eval, CLARIN datasets, helper scripts) and `webapp/CLAUDE.md`.

The on-device browser demo lives on branch `experiment/webapp-ondevice`, is not in the thesis, and must not be re-added to `main`.

## Style

Reply in English unless the user specifies otherwise.
The amount of thinking should be proportional to the complexity of the task you're given.
Avoid unnecessary verbosity by using CoT to structure your response.

Important: when launching subagents, use only Opus agents (unless the user specifies differently). You may decide to use Sonnet subagents for the easiest work. Never launch Fable subagents — with one standing exception (approved 2026-07-25): agents whose job is synthesis/adjudication or premium prose editing (`review-synthesizer`, `code-review-synthesizer`, `redaktor`) pin `model: fable` in their frontmatter, because deciding between conflicting reviewers merits the strongest reasoning. Do not extend the exception to other agents without asking.

## Thesis Code Principles

This code will be reviewed by academic supervisors. Prioritize: clarity over cleverness, simplicity over abstraction, reproducibility. Prefer explicit implementations that match cited papers. Don't over-engineer — no deep inheritance, no speculative features. Experiment logging is handled outside this repo.

## Keeping This File Current

After any substantial change — new top-level script or subsystem, new env var, new dataset/model/task variant, new gotcha worth flagging, removed commands, or changed config precedence — propose a targeted edit to this CLAUDE.md. Update in place; don't rewrite from scratch. Skip for routine bugfixes, refactors, or one-off experiments. Subsystem detail goes into the nested `asr_pipeline/CLAUDE.md` / `webapp/CLAUDE.md`; experiment results, decision history and "X was removed on DATE" notes belong in the thesis log and git history, not here.

## Dataset Variants

- `PolSESS_C_both` — old 8k-effective dataset (half was duplicated). Used for early baselines, HPO, and HPO validation runs.
- `PolSESS_C_new_64` = `C_new_64` — correct 64k dataset generated 2026-04-15. Use `train_max_samples=16000` / `32000` / full for 16k / 32k / 64k scaling experiments.
- `PolSESS_C_final_128_v2` — 128k dataset for final training runs. Contains other-language speech alongside Polish.
- `PolSESS_C_128_16kHz` — 16 kHz twin of 128_v2 (same six-run recipe, `dataSources_v3`, `outputFreq = 16000`; generated 2026-09-07/08). Same layout: `train/` 128,000 rows interleaved indoor/outdoor, `val/` = 1,000-item stratified subset hardlinked from `val_big/` (12,800), `test/` 12,800. (currently only on the 3080 machine)

## W&B Projects

- `polsess-separation` — standalone runs (baselines, HPO validation), uses `PolSESS_C_both`.
- `polsess-thesis-experiments` — sweeps.
- `polsess-separation-real16k` / `polsess-separation-32k` / `polsess-separation-64k` — scaling runs on subsets of `PolSESS_C_new_64` (NB only the 16k project carries the `real` prefix).
- `polsess-separation-128k` — final runs on the 128k dataset.

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
Both read the shared architecture list in `scripts/benchmark_models.py` and write `docs/generated/benchmark_{inference,training}.csv` with provenance columns. ptflops needs four corrections (MHA double count, `einsum`, `F.scaled_dot_product_attention` incl. banded cost under a boolean mask, Mamba scan FLOPs-vs-MACs) — documented in the inference script's docstring, tested in `tests/test_benchmark_macs.py`, audited in `thesis/thesis-log/sweep_plan/ch5_test_evals/BENCHMARK_AUDIT.md`. Counting is device-independent (`--device cpu` for non-Mamba); everything timed must run on one machine in one session. The 2026-04 CSVs in `scripts/` are superseded.

**Thesis audit artifacts (citable, CPU-only unless noted):**
```bash
python scripts/audit_mmipc.py           # MM-IPC reconstruction lossless to the 16-bit PCM floor (exit≠0 on violation)
python scripts/audit_split_leakage.py   # train↔{val,test} path disjointness
python scripts/model_manifest.py        # per-config param counts → docs/generated/model_manifest.{csv,md}
python scripts/squim_validation.py all  # B7 SQUIM-vs-true-metrics validation (GPU, ~30 min; thesis/thesis-log/sweep_plan/b7_squim/)
```

**Interactive:**
```bash
jupyter notebook test_model_interactive.ipynb
jupyter notebook asr/explore_pipeline.ipynb   # per-stage frontend for asr_pipeline/
```

**ASR pipeline** (details in `asr_pipeline/CLAUDE.md`; read `asr_pipeline/SCOPE.md` before changing code there):
```bash
python -m asr_pipeline run --config asr_pipeline/configs/sweep_best_e31_refineplus.yaml \
    --input rec.wav --write-outputs <eval_root> --set diarization.backend=pyannote
python -m asr_pipeline batch --split clarin_dev --mode no_enh
python -m asr_pipeline score --eval-root <eval_root> --out-dir <csv_dir>
```

**Showcase webapp** (details in `webapp/CLAUDE.md`):
```bash
./webapp/run.sh                                  # real server, port ${WEBAPP_PORT:-8871}
venv/bin/python -m webapp.dev_server --port 8899 # GPU-free dev harness
pytest tests/test_webapp_backend.py
```

## Architecture Overview

**Configuration (`config.py`):** Dataclasses (`DataConfig`, `ModelConfig`, `TrainingConfig`) with nested model/dataset params. Priority: defaults < env vars < YAML < CLI args. Use `get_config_from_args()` for CLI, `load_config_for_run(wandb.config)` for sweeps.

**Model Registry (`models/__init__.py`):** Dict-based. `get_model("name")` returns class. Mamba models auto-excluded without `mamba-ssm`. Measured param counts per config: `docs/generated/model_manifest.md`.
- Cross-platform: `convtasnet` (~8.7M for SB; the oft-quoted 8.64M is the ES/C=1 build), `sepformer` (~26M), `dprnn` (~2-3M), `mossformer2` (matched ~26M / full ~55.7M), `tf_mossformer` (S/M/L: 5.92 / 17.34 / 26.00M).
- Linux + CUDA only: `spmamba` (~1.2M), `mamba_tasnet` and `dpmamba` (XS/S/M/L: ~2.2-60M). `models/mamba/` holds BiMamba blocks adapted from xi-j/Mamba-TasNet.
- `mossformer2` (Zhao et al. 2023, arXiv:2312.11825): files in `models/mossformer2/` are **vendored** from ClearerVoice-Studio; `__init__.py` is the project wrapper. Single `N` knob = encoder dim = transformer dim (must match upstream); `num_blocks` = GFSMN depth (24 paper-full, 11 ≈ SepFormer-matched); `attn_dropout` covers the attention path only (FSMN-gate dropout fixed at 0.1). Sweep key `dropout` routes to `attn_dropout`. Configs: `experiments/mossformer2/`.
- `tf_mossformer` (Zhao et al. 2026, arXiv:2607.21128): TF separator with convolution-gated local + global attention. **Re-implemented from the paper — no upstream code exists**; never call it "vendored". `tf_locoformer_blocks.py` is a deliberate separate copy of the Apache-2.0 TF-Locoformer file in `asr_pipeline/vendor/` (`models/` must not import from the liftable pipeline package); `local_global_attention.py` and `__init__.py` are original. Our sizes run ~2.5% above the paper's M/L rows because the paper's own rows are mutually inconsistent — quote the caveat wherever the size appears; reconciliation in `thesis/x_notes/tf_mossformer/tf_mossformer_spec.md` §4. Knobs follow paper Table 1 (`D`, `num_blocks`, `ffn_hidden_dim`, `conv_kernel_size`/`conv_stride`, `n_heads`, `num_groups`, `window_t`/`window_f`, `gate_kernel_size`, `n_fft`/`hop_length` in **samples**, `attn_dropout`; sweep key `dropout` → `attn_dropout`). Its configs (`experiments/tf_mossformer/{s,m,l}_8k.yaml`) are the only ones on the paper's recipe (`adamw`, `warmup_steps: 4000`, wd 1e-2, plateau patience 3); the paper's bs 4 / lr 1e-3 doesn't fit 12 GB, so they run the batch that fits with LR scaled linearly (S bs 2 / 5e-4, M and L bs 1 / 2.5e-4, no accumulation — author ruling 2026-09-07). 4070: S ≈ 4.4 GiB and ~29 min per 16k epoch, M 6.1 GiB, L 9.2 GiB.

**Dataset Registry (`datasets/__init__.py`):** Dict-based. `get_dataset("name")` returns class. Supports `polsess`, `libri2mix`, `echoset`. The EchoSet corpus was deleted 2026-08-03 (loader kept as the record of a negative result): its targets are reverberant while PolSESS SB targets are dry, so PolSESS-trained models score ≈ −2.9 dB SI-SDRi there — disqualified for generalization numbers. Notes: `thesis/thesis-log/sweep_plan/ch5_test_evals/CH5_TEST_EVALS_NOTES.md`. (Unrelated: `asr_pipeline/configs/b1_tiger_echoset.yaml` is the TIGER checkpoint *trained on* EchoSet.)

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

- **Corpus sampling rate (`data.sample_rate`, default 8000):** loaders and models are rate-agnostic (kernel/stride/n_fft in samples, chunk sizes in frames), so a 16 kHz PolSESS render with the same layout trains without code changes. The field is provenance plus a guard: `PolSESSDataset` checks the first mix file's header at construction and raises on mismatch; `evaluate.py` / `evaluate_all.py` take the rate from the checkpoint (pre-field checkpoints = 8 kHz) to choose PESQ nb/wb, the STOI rate and the Libri2Mix `wav8k/`/`wav16k/` folder. Nothing rescales geometry: at 16 kHz a clip has 2× the samples, so keeping the frame rate (32/16 encoders, 512/128 STFT) is a per-architecture choice in the 16 kHz YAMLs. In the ASR pipeline a 16 kHz separator needs `separation.separator_sample_rate: 16000` **and** `post_separation_processing.backend: naive`. `experiments/tf_mossformer/s_16k.yaml` is the only `sample_rate: 16000` config and is **blocked** until the 16 kHz corpus is on this machine — read its header before unblocking.
- **AMP:** on by default. SpeechBrain EPS patched 1e-8 → 1e-4 in `utils/common.py` (`apply_eps_patch`, ConvTasNet's SpeechBrain lobe only). Most models: float16 + GradScaler. Mamba models, MossFormer2 and TF-MossFormer: bfloat16 without GradScaler (dispatch on `model_type` in `training/trainer.py:_setup_amp`) — MossFormer2's squared-ReLU attention and TF-MossFormer's gated products overflow fp16. TF-MossFormer casts to fp32 around `torch.stft` / `torch.complex` / `torch.istft` inside its forward, like SPMamba (`torch.complex` rejects bf16). Validation runs in fp32.
- **Training determinism (`training.deterministic`, tri-state):** unset/`null` = cuDNN deterministic + no benchmark, `use_deterministic_algorithms` not called; `true` = strict (`torch.use_deterministic_algorithms(warn_only=True)` + `CUBLAS_WORKSPACE_CONFIG`); `false` = `cudnn.benchmark`, non-deterministic. Train DataLoaders use a seeded `torch.Generator` + `worker_init_fn`, so the MM-IPC variant stream is reproducible. Gotcha: curriculum learning mutates `allowed_variants` in place and only works because workers re-fork each epoch — never enable `persistent_workers` with a curriculum active.
- **torch.compile:** auto on Linux (~10-20% speedup); loading handles the `_orig_mod` prefix; skipped for Mamba models. `mossformer2` and `tf_mossformer` compile with `dynamic=False` (their rotary block can't be lowered under symbolic shapes) — fixed-length crops compile once, a new length triggers a one-time recompile. Dispatch: `compile_for_model_type` in `utils/model_utils.py`.
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
3. **Config precedence:** CLI > YAML > env vars > defaults. `Config.__post_init__` forces the model's source count (`C`/`n_srcs`) to match the task (ES→1, SB/EB→2) and prints a line when it changes the value.
4. **Sweep config access:** `load_config_for_run(wandb.config)` uses `getattr`, not dict access.
5. **`--resume` extends the epoch budget:** `num_epochs` is *additional* epochs (a run resumed at 30 with `num_epochs: 80` runs to 110) and the early-stopping counter restarts at 0; there is no `--epochs` override. For budget-matched comparisons stop at the intended total and report best-within-budget. `Trainer` logs with `step=epoch`, so a resume that restarts below the highest logged step adds no W&B rows until it passes it — the run looks frozen on W&B while training normally.

## Virtual Environments

**Main (`venv/`)** — all models except SPMamba3. Alias: `polsess_venv`. `requirements.txt` is the installable core set (optional extras — mamba-ssm, ASR pipeline, B1 git dependency — are commented blocks); `requirements-freeze.txt` is the exact `pip freeze`. Third-party attribution and vendored-code licences: `THIRD_PARTY.md`.

**SPMamba3 (`venv_mamba3/`)** — torch 2.11.0+cu130, triton 3.6.0, Mamba-3 kernels; clone of the main venv with Mamba-3 files copied from a bare clone of `state-spaces/mamba`. Extra deps: `tilelang`, `quack-kernels`, `cuda-bindings`, `nvidia-cutlass-dsl`. Side project, not in the thesis.

## Cloud Setup

`setup.sh` provisions a fresh cloud GPU instance (Vast.ai / RunPod): clones the repo (`REPO_BRANCH`/`REPO_URL` env-overridable), installs deps, sets env vars; `--rclone` additionally configures the Google Drive remote. Start from `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel` — Blackwell (sm_120) needs exactly that, and `-devel` has nvcc for a `MAMBA_FORCE_BUILD=TRUE` from-source fallback. `DATASET_NAME=PolSESS_C_new_64 ./download_dataset.sh` fetches one corpus (default `PolSESS_C_final_128_v2`) and wires `POLSESS_DATA_ROOT` to the directory containing `train/` (auto-detects the nested layouts). With several corpora on one box, pass `--data-root` per launch rather than trusting the env var. Neither script logs into W&B — run `wandb login` first.

## Environment Variables

- `POLSESS_DATA_ROOT` — PolSESS dataset path (default in `config.py`).
- `HF_TOKEN` — HuggingFace token for the ASR pipeline's pyannote diarization stage.
- `AP_BWE_CHECKPOINT` — AP-BWE checkpoint for the ASR post-separation stage (default `~/AP-BWE/checkpoints/8kto16k/g_8kto16k`).
- `SORTFORMER_VENV_PY` — python of the isolated NeMo venv (`~/sortformer_venv/bin/python`); required by `diarization.backend: sortformer`, which the shipped best ASR config uses. No default; missing → loud crash.
- `COHEREX_VENV_PY` — python of the isolated CohereX venv; required by `transcription.backend: coherex`. No default; missing → loud crash.
