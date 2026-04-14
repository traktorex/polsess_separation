# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

PyTorch implementation of speech separation using ConvTasNet, SepFormer, DPRNN, SPMamba, Mamba-TasNet, and DPMamba architectures on the PolSESS dataset. Part of a master's thesis on speech separation for downstream Polish ASR preprocessing.

## Thesis Code Principles

This code will be reviewed by academic supervisors. Prioritize: clarity over cleverness, simplicity over abstraction, reproducibility. Prefer explicit implementations that match cited papers. Don't over-engineer — no factories for <4 variants, no deep inheritance, no speculative features. Experiment logging is handled outside this repo.

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
```

**ASR Evaluation:**
```bash
# Separation on REAL-M (WER/CER + SQUIM)
python asr/evaluate_asr.py --checkpoint path/to/model.pt --dataset realm --mode separation --whisper-model large
# Mixture baseline
python asr/evaluate_asr.py --dataset realm --mode mixture
# Clean source baseline on LibriSpeech
python asr/evaluate_asr.py --dataset librispeech --mode baseline
# Disable SQUIM (WER/CER only)
python asr/evaluate_asr.py --dataset realm --mode mixture --no-squim
```

**Testing:**
```bash
pytest
pytest --cov=. --cov-report=html
pytest tests/test_model.py -v
```

**Interactive:**
```bash
jupyter notebook test_model_interactive.ipynb
```

## Architecture Overview

**Configuration (`config.py`):** Dataclasses (`DataConfig`, `ModelConfig`, `TrainingConfig`) with nested model/dataset params. Priority: defaults < env vars < YAML < CLI args. Use `get_config_from_args()` for CLI, `load_config_for_run(wandb.config)` for sweeps.

**Model Registry (`models/__init__.py`):** Dict-based. `get_model("name")` returns class. Mamba models auto-excluded without `mamba-ssm`.
- `convtasnet` (~8M), `sepformer` (~26M), `dprnn` (~2-3M) — cross-platform
- `spmamba` (~1.2M), `mamba_tasnet` (XS/S/M/L: 2.2-59.6M), `dpmamba` (XS/S/M/L: 2.3-59.8M) — Linux + CUDA only

**Dataset Registry (`datasets/__init__.py`):** Dict-based. `get_dataset("name")` returns class. Supports: `polsess`, `libri2mix`.

**Training Flow:** `train.py` → config → dataloaders → `create_model_from_config()` → optional `torch.compile()` → `Trainer` (AMP, grad accumulation, checkpointing, curriculum learning).

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

- **AMP:** Enabled by default. SpeechBrain EPS patched from 1e-8 to 1e-4 in `utils/common.py` to prevent float16 underflow. Non-Mamba models use float16 + GradScaler; Mamba models use bfloat16 without GradScaler (detected by model class name in `trainer.py`).
- **torch.compile:** Auto-applied on Linux for ~10-20% speedup. Checkpoint loading handles `_orig_mod` prefix.
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

1. **NaN in SI-SDR:** AMP underflow — EPS patch should handle it. If not, `use_amp: false`.
2. **Memory overflow:** Reduce `batch_size`, use `grad_accumulation_steps` to compensate.
3. **Config precedence:** CLI > YAML > env vars > defaults. Additionally, `Config.__post_init__` silently forces the model's output source count (`C`/`n_srcs`) to match the task (ES→1, SB/EB→2), overriding whatever the YAML says.
4. **MambaTasNet NaN:** Deep configs need `residual_in_fp32: true`.  `grad_clip_norm: 1.0` (not 5.0) might help too.
5. **Mamba on Windows:** Requires WSL2 + CUDA toolkit 12.4+. Non-Mamba models work natively.
6. **Sweep config access:** `load_config_for_run(wandb.config)` uses `getattr`, not dict access.

## Virtual Environments

**Main (`venv/`)** — all models except SPMamba3. Alias: `polsess_venv`.

**SPMamba3 (`venv_mamba3/`)** — torch 2.11.0+cu130, triton 3.6.0, Mamba-3 kernels. Clone of main venv with Mamba-3 files manually copied from bare repo clone of `state-spaces/mamba`. Additional deps: `tilelang`, `quack-kernels`, `cuda-bindings`, `nvidia-cutlass-dsl`.

## Cloud Setup

`setup.sh` automates fresh cloud GPU instance setup (Vast.ai / RunPod): clones repo, installs deps, downloads PolSESS from Google Drive, configures env vars. Use `--no-data` if dataset is already mounted.

## Environment Variables

- `POLSESS_DATA_ROOT` — PolSESS dataset path (default in `config.py`)
- `REALM_DATA_ROOT` — REAL-M dataset (default: `~/datasets/REAL-M-v0.1.0/`)
- `LIBRIMIX_ASR_ROOT` — LibriSpeechMixASR (default: `~/datasets/LibriSpeechMixASR/`)
