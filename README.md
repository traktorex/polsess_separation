# PolSESS Speech Separation for Polish ASR Preprocessing

PyTorch implementation of speech separation on the PolSESS dataset using multiple architectures (ConvTasNet, DPRNN, SepFormer, SPMamba). Trained models will be used for preprocessing Polish ASR on real conversational speech (CLARIN corpus).

**Key Feature**: PolSESS includes realistic acoustic conditions (reverb + scene sounds + events), unlike synthetic-only datasets (LibriMix), leading to better generalization on real speech.

## Current Performance (February 2026)

**Baseline Experiments Complete** — SB Task (2-Speaker Separation):

| Model | Avg SI-SDR | Runtime (3 seeds) | Notes |
|-------|------------|-------------------|-------|
| **SPMamba** 🏆 | **5.56 dB** | ~90 hours | Best performer, SSM architecture |
| SepFormer | 5.10 dB | ~54 hours | Transformer-based, 2nd best |
| DPRNN | 3.03 dB | ~11 hours | RNN baseline |
| ConvTasNet | 2.95 dB | ~32 hours | CNN baseline |

**DPRNN Hyperparameter Optimization Complete** — best config (fancy-sweep-62):

| Approach | Best SI-SDR | vs. Baseline | Compute |
|----------|-------------|--------------|---------|
| 3-Stage progressive scaling 🏆 | **4.67 dB** | +1.64 dB (+54%) | 322h |
| Exp B (proxy + LR sweep) | 4.42 dB | +1.39 dB | ~123h |
| Exp A (one-stage 8K) | 4.37 dB | +1.34 dB | ~105h |

**Key insight**: Weight decay is the strongest predictor (correlation −0.27). Optimal range: 1e-6 to 5e-5.

**ConvTasNet & SPMamba HPO**: Stage 2 sweeps currently running.

See [`sweeps/EXPERIMENT_LOG.md`](sweeps/EXPERIMENT_LOG.md) for full experimental details.

## Features

- **Automatic Mixed Precision (AMP):** 30–40% faster training with no quality loss
- **MM-IPC Augmentation:** Randomly varies background complexity (SER, SR, ER, R for indoor; SE, S, E for outdoor)
- **torch.compile Support:** Further speedup on Linux with PyTorch 2.0+
- **W&B Hyperparameter Sweeps:** Bayesian optimization with Hyperband early termination
- **Curriculum Learning:** Progressive variant scheduling over epochs
- **Early Stopping:** Automatic termination when validation plateaus
- **Gradient Accumulation:** Effective batch size scaling without extra VRAM
- **Config-Driven:** YAML configs for reproducible experiments

## Project Structure

```
polsess_separation/
├── models/                    # Model architectures
│   ├── factory.py            # Config-driven model instantiation
│   ├── conv_tasnet.py        # Conv-TasNet implementation
│   ├── dprnn.py              # Dual-Path RNN
│   ├── sepformer.py          # SepFormer (Transformer-based)
│   └── spmamba.py            # SPMamba (State-space model)
│
├── datasets/                  # Dataset handling
│   ├── polsess_dataset.py    # PolSESS with MM-IPC augmentation
│   ├── libri2mix_dataset.py  # LibriMix for comparison
│   └── __init__.py           # Dataset registry
│
├── training/                  # Training infrastructure
│   └── trainer.py            # Training loop with curriculum learning
│
├── utils/                     # Utilities
│   ├── common.py             # Seed, EPS patch, device setup
│   ├── model_utils.py        # Parameter counting, checkpointing
│   ├── wandb_logger.py       # Experiment tracking
│   └── logger.py             # Logging setup
│
├── config.py                  # Configuration dataclasses
├── train.py                   # Training entry point
├── train_sweep.py             # W&B sweep entry point
├── evaluate.py                # Evaluation entry point
├── experiments/               # YAML experiment configs
│   ├── convtasnet/
│   ├── dprnn/
│   ├── sepformer/
│   └── spmamba/
│
├── sweeps/                    # W&B sweep configurations and logs
│   ├── EXPERIMENT_LOG.md     # Complete experimental results
│   ├── 1-baselines-SB/       # Baseline sweep configs
│   └── 3-hyperparam-opt/     # Hyperparameter optimization sweeps
│
└── tests/                     # Comprehensive test suite (264 tests, 17 files)
    ├── test_model.py
    ├── test_model_factory.py
    ├── test_dataset.py
    ├── test_mmipc.py
    ├── test_config_yaml.py
    ├── test_evaluation.py
    └── ...
```

### Key Design Decisions

- **Model Factory Pattern**: Justified for comparing 4 different architectures
- **Direct DataLoader Creation**: Explicit and easy to modify (no dataset factory)
- **Single `evaluate.py`**: All evaluation logic in one file (standard research pattern)
- **Comprehensive Tests**: 264 tests ensuring correctness and reproducibility
- **Config-Driven**: YAML configs for reproducible experiments

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Training

```bash
# Train with YAML config (recommended for experiments)
python train.py --config experiments/dprnn/dprnn_baseline.yaml

# Override model or task at the CLI
python train.py --config experiments/baseline.yaml --model-type spmamba
python train.py --config experiments/baseline.yaml --task SB

# Disable W&B logging
python train.py --config experiments/baseline.yaml --no-wandb

# Resume from checkpoint
python train.py --config experiments/baseline.yaml --resume checkpoints/dprnn/SB/run_name/dprnn_SB_best.pt

# See all options
python train.py --help
```

**Configuration priority:** defaults → env vars (`POLSESS_DATA_ROOT`) → YAML → CLI args (highest)

### W&B Sweeps

```bash
# Register a sweep
wandb sweep sweeps/3-hyperparam-opt/dprnn/stage1.yaml

# Run an agent (use /run-sweep workflow for tmux crash-resistance)
wandb agent <sweep_id>
```

### Evaluation

```bash
# Evaluate on all MM-IPC variants
python evaluate.py --checkpoint checkpoints/dprnn/SB/run_name/dprnn_SB_best.pt

# Evaluate on specific variant only
python evaluate.py --checkpoint checkpoints/dprnn/SB/run_name/dprnn_SB_best.pt --variant SER

# Fast evaluation (skip PESQ and STOI)
python evaluate.py --checkpoint checkpoints/dprnn/SB/run_name/dprnn_SB_best.pt --no-pesq --no-stoi

# Save results to CSV
python evaluate.py --checkpoint checkpoints/dprnn/SB/run_name/dprnn_SB_best.pt --output results.csv

# See all options
python evaluate.py --help
```

### Testing

```bash
pytest
pytest -v
pytest tests/test_config_yaml.py
pytest --cov=. --cov-report=html
```

## Technical Details

### Automatic Mixed Precision (AMP)

Uses float16 for forward/backward passes and float32 for weights/optimizer. SpeechBrain's `EPS=1e-8` underflows to 0 in float16, causing NaN. We patch it to `1e-4` (safe for float16's minimum ~6e-5):

```python
from utils import apply_eps_patch
apply_eps_patch(1e-4)  # Called automatically if AMP is enabled
```

### MM-IPC Augmentation

Randomly varies background complexity during training:

- **Indoor:** SER (speech + event + reverb), SR (speech + reverb), ER (event + reverb), R (reverb only)
- **Outdoor:** SE (speech + event), S (speech only), E (event only), C (clean)

Implemented in [`datasets/polsess_dataset.py`](datasets/polsess_dataset.py) via lazy loading — only loads audio layers needed for the randomly selected variant. Validation uses deterministic variant selection (seeded by sample index).

**Controlling Variants:**
```python
# Training: use all variants (default)
dataset = PolSESSDataset(..., allowed_variants=None)

# Evaluation: use specific variant
dataset = PolSESSDataset(..., allowed_variants=['SER'])

# Curriculum: use only some variants (updated per epoch by Trainer)
dataset = PolSESSDataset(..., allowed_variants=['SER', 'SR', 'ER'])
```

### Configuration

All configuration is centralized in [`config.py`](config.py) with three sections:

**DataConfig:** dataset type, batch size, task (ES/EB/SB), sample limits, PolSESS data root

**ModelConfig:** model type selector + nested params for each architecture:
- `ConvTasNetParams`: N, kernel_size, stride, B, H, P, X, R, C, norm_type, mask_nonlinear
- `DPRNNParams`: N, kernel_size, stride, C, num_layers, chunk_size, rnn_type, hidden_size, bidirectional
- `SepFormerParams`: N, kernel_size, stride, C, num_blocks, num_layers, d_model, nhead, d_ffn, chunk_size
- `SPMambaParams`: n_fft, stride, n_layers, lstm_hidden_units, attn_n_head, n_srcs

**TrainingConfig:** lr, weight_decay, grad_clip_norm, lr_factor, lr_patience, num_epochs, use_amp, seed, curriculum_learning, early_stopping_patience, grad_accumulation_steps, use_wandb, resume_from

### Evaluation Metrics

- **SI-SDR (Scale-Invariant Signal-to-Distortion Ratio):** Separation quality in dB (higher is better)
- **PESQ (Perceptual Evaluation of Speech Quality):** Perceptual quality, range 1–5 (higher is better)
- **STOI (Short-Time Objective Intelligibility):** Intelligibility, range 0–1 (higher is better)

Evaluation can be performed on all MM-IPC variants or specific ones:
- **Indoor (with reverb):** SER, SR, ER, R
- **Outdoor (no reverb):** SE, S, E, C

## Dataset Structure Expected

```
PolSESS/
├── train/
│   ├── clean/         # Clean speech
│   ├── event/         # Event sounds
│   ├── mix/           # Mixed audio
│   ├── scene/         # Background scene
│   ├── sp1_reverb/    # Speaker 1 with reverb
│   ├── sp2_reverb/    # Speaker 2 with reverb
│   ├── ev_reverb/     # Event with reverb
│   └── corpus_PolSESS_C_in_train_final.csv
├── val/
│   └── ...
└── test/
    └── corpus_PolSESS_C_in_test_final.csv
```

## Hardware Requirements

- **GPU:** 12GB VRAM (tested on RTX 4070)
- **RAM:** 16GB+ recommended

With batch_size=4:
- **GPU VRAM:** ~3.5 GB (DPRNN/ConvTasNet)
- **Training speed:** ~0.3s per batch

**SPMamba** (FP32 for stability):
- **GPU VRAM:** ~11.3 GB (batch_size=1)
- **Training speed:** ~1.25 hours per epoch
- **Recommendation:** Disable AMP (`use_amp: false`) for numerical stability

## Troubleshooting

### NaN in SI-SDR
- **Cause:** SpeechBrain's EPS=1e-8 underflows in float16
- **Fix:** Automatically applied via `apply_eps_patch()` when AMP is enabled

### GPU Memory Overflow
- **Symptom:** Training becomes very slow (~35s per batch)
- **Cause:** Batch size too large, overflowing to system RAM
- **Fix:** Reduce `batch_size` in YAML config, or use `grad_accumulation_steps` to maintain effective batch size

### SPMamba on Windows
- **Cause:** `mamba-ssm` requires Linux + CUDA
- **Fix:** Use WSL2 with CUDA toolkit 12.4+

## References

- **ConvTasNet:** [Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking for Speech Separation](https://arxiv.org/abs/1809.07454)
- **DPRNN:** [Dual-Path RNN: Efficient Long Sequence Modeling for Time-Domain Single-Channel Speech Separation](https://arxiv.org/abs/1910.06379)
- **SepFormer:** [Attention is All You Need in Speech Separation](https://arxiv.org/abs/2010.13154)
- **SPMamba:** [SPMamba: State-Space Model is All You Need in Speech Separation](https://arxiv.org/abs/2404.02063)
- **SpeechBrain:** [SpeechBrain: A PyTorch-based Speech Toolkit](https://github.com/speechbrain/speechbrain)
- **MM-IPC Augmentation:** Based on Klec et al.'s approach for PolSESS

## License

This project uses SpeechBrain components (Apache 2.0 License).
