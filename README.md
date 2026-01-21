# PolSESS Speech Separation for Polish ASR Preprocessing

PyTorch implementation of speech separation on the PolSESS dataset using multiple architectures (ConvTasNet, DPRNN, SepFormer, SPMamba). Trained models will be used for preprocessing Polish ASR on real conversational speech (CLARIN corpus).

**Key Feature**: PolSESS includes realistic acoustic conditions (reverb + scene sounds + events), unlike synthetic-only datasets (LibriMix), leading to better generalization on real speech.

## Current Performance (January 2026)

**Baseline Experiments Complete** - SB Task (2-Speaker Separation):

| Model | Avg SI-SDR | Runtime (3 seeds) | Notes |
|-------|------------|-------------------|-------|
| **SPMamba** 🏆 | **5.56 dB** | ~90 hours | Best performer, SSM architecture |
| SepFormer | 5.10 dB | ~54 hours | Transformer-based, 2nd best |
| DPRNN | 3.03 dB | ~11 hours | RNN baseline |
| ConvTasNet | 2.95 dB | ~32 hours | CNN baseline |

**Next**: Hyperparameter optimization in progress (target: +0.2-0.5 dB per model)

See [`sweeps/EXPERIMENT_LOG.md`](sweeps/EXPERIMENT_LOG.md) for full experimental details.

## Features

- **Automatic Mixed Precision (AMP):** 30-40% faster training with no quality loss
- **MM-IPC Augmentation:** Randomly varies background complexity (SER, SR, ER, R for indoor; SE, S, E for outdoor)
- **torch.compile Support:** Further speedup on Linux with PyTorch 2.0+
- **W&B Hyperparameter Sweeps:** Bayesian optimization with Hyperband early termination
- **Early Stopping:** Automatic termination when validation plateaus
- **Configurable:** CLI arguments or config file
- **Modular Architecture:** Clean separation of concerns

## Project Architecture

```
polsess_separation/
├── models/                    # Model architectures
│   ├── factory.py            # Model factory (4 architectures)
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
├── evaluate.py                # Evaluation entry point (420 lines)
├── experiments/               # YAML experiment configs
│   ├── baseline.yaml
│   ├── convtasnet/
│   ├── dprnn/
│   ├── sepformer/
│   └── spmamba/
│
├── sweeps/                     # W&B sweep configurations and logs
│   ├── EXPERIMENT_LOG.md      # Complete experimental results
│   ├── 1-baselines-SB/        # Baseline sweep configs
│   └── 3-hyperparam-opt/      # Hyperparameter optimization sweeps
│
└── tests/                     # Comprehensive test suite (228 tests)
    ├── test_model.py         # Model architecture tests
    ├── test_model_factory.py # Factory pattern tests
    ├── test_dataset.py       # Dataset tests
    ├── test_mmipc.py         # MM-IPC augmentation tests
    ├── test_evaluation.py    # Evaluation pipeline tests
    └── ...
```

### Key Design Decisions

- **Model Factory Pattern**: Justified for comparing 4 different architectures
- **Direct DataLoader Creation**: Explicit and easy to modify (no dataset factory)
- **Single `evaluate.py`**: All evaluation logic in one file (standard research pattern)
- **Comprehensive Tests**: 228 tests ensuring correctness and reproducibility
- **Config-Driven**: YAML configs for reproducible experiments

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Training

```bash
# Train with default settings
python train.py

# Train with YAML config (recommended for experiments)
python train.py --config experiments/baseline.yaml

# Train with custom settings
python train.py --batch-size 8 --epochs 20

# Train with YAML + CLI overrides
python train.py --config experiments/baseline.yaml --lr 0.0001 --epochs 50

# Train with larger model
python train.py --model-size large

# Train on EB task (2 speakers)
python train.py --task EB

# See all options
python train.py --help
```

**Available experiment configs:**

- `experiments/baseline.yaml` - Configuration that achieved 9.84 dB
- `experiments/large_model.yaml` - Larger model (~34M params)
- `experiments/small_fast.yaml` - Smaller model for quick tests
- `experiments/eb_task.yaml` - Enhance both speakers task
- `experiments/lr_sweep.yaml` - Example for hyperparameter tuning

See [experiments/README.md](experiments/README.md) for details.

### Evaluation

```bash
# Evaluate on all MM-IPC variants
python evaluate.py --checkpoint checkpoints/{model}/{task}/{run_name}_{timestamp}/model.pt

# Evaluate on specific variant only
python evaluate.py --checkpoint checkpoints/{model}/{task}/{run_name}_{timestamp}/model.pt --variant SER

# Fast evaluation (skip PESQ and STOI)
python evaluate.py --checkpoint checkpoints/{model}/{task}/{run_name}_{timestamp}/model.pt --no-pesq --no-stoi

# Save results to CSV
python evaluate.py --checkpoint checkpoints/{model}/{task}/{run_name}_{timestamp}/model.pt --output results.csv

# See all options
python evaluate.py --help
```

### Configuration Options

**Data:**

- `--data-root`: Path to PolSESS dataset
- `--task`: ES (single speaker) or EB (both speakers)
- `--batch-size`: Physical batch size (default: 4)
- `--num-workers`: DataLoader workers (default: 4)

**Model:**

- `--model-size`: small/default/large (default: default)

**Training:**

- `--epochs`: Number of epochs (default: 10)
- `--lr`: Learning rate (default: 0.001)
- `--no-amp`: Disable automatic mixed precision

## Project Structure

```
polsess_separation/
├── models/              # Model architectures
│   ├── __init__.py
│   └── conv_tasnet.py   # ConvTasNet implementation
├── training/            # Training logic
│   ├── __init__.py
│   └── trainer.py       # Trainer with AMP
├── data/                # Data utilities
│   ├── __init__.py
│   └── collate.py       # Custom collate function
├── datasets/            # Dataset loaders
│   ├── __init__.py
│   ├── polsess_dataset.py   # PolSESS dataset loader with MM-IPC
│   └── libri2mix_dataset.py # Libri2Mix cross-dataset evaluation
├── utils/               # Utilities
│   ├── __init__.py
│   ├── amp_patch.py     # SpeechBrain AMP compatibility patch
│   ├── common.py        # Common utilities (seeds, warnings, device setup)
│   ├── logger.py        # Colored logging setup
│   └── wandb_logger.py  # Weights & Biases integration
├── tests/               # Test suite
│   ├── conftest.py      # Pytest fixtures
│   ├── test_dataset.py
│   ├── test_model.py
│   ├── test_trainer.py
│   └── test_utils.py
├── docs/                # Documentation
│   ├── CONFIG_GUIDE.md
│   ├── WANDB_GUIDE.md
│   ├── HYPERPARAMETER_SWEEP_GUIDE.md
│   ├── PROJECT_REFLECTION.md
│   ├── FUTURE_ENHANCEMENTS.md
│   └── SPMAMBA_INTEGRATION_PLAN.md
├── experiments/         # Experiment configurations (YAML)
│   ├── README.md
│   ├── baseline.yaml
│   ├── large_model.yaml
│   ├── small_fast.yaml
│   ├── eb_task.yaml
│   ├── klec_replication.yaml
│   ├── wandb_sweep.yaml
│   └── klec_sweep.yaml
├── scripts/             # Convenience scripts
├── archive/             # Archived analysis documents
├── config.py            # Configuration management
├── train.py             # Main training script
├── train_dual_corpus.py # Dual corpus training (Klec et al. replication)
├── train_sweep.py       # W&B hyperparameter sweep entry point
├── evaluate.py          # Evaluation script with per-variant metrics
└── README.md
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
- **Outdoor:** SE (speech + event), S (speech only), E (event only)

Implemented in [`datasets/polsess_dataset.py`](datasets/polsess_dataset.py) via lazy loading - only loads audio layers needed for the randomly selected variant.

**Controlling Variants:**
The dataset supports an `allowed_variants` parameter:

```python
# Training: use all variants (default)
dataset = PolSESSDataset(..., allowed_variants=None)

# Evaluation: use specific variant
dataset = PolSESSDataset(..., allowed_variants=['SER'])

# Training subset: use only some variants
dataset = PolSESSDataset(..., allowed_variants=['SER', 'SR', 'ER'])
```

### ConvTasNet Architecture

- **Encoder:** 1D convolution (kernel=16, stride=8) → 256 filters
- **Separation:** 4 × 8 = 32 temporal convolutional blocks
- **Decoder:** Transposed convolution to reconstruct waveform
- **Normalization:** GlobalLayerNorm with patched EPS for float16 safety

### Evaluation Metrics

The evaluation script computes three standard speech quality metrics:

- **SI-SDR (Scale-Invariant Signal-to-Distortion Ratio):** Measures separation quality in dB (higher is better)
- **PESQ (Perceptual Evaluation of Speech Quality):** Perceptual quality metric, range 1-5 (higher is better)
- **STOI (Short-Time Objective Intelligibility):** Speech intelligibility metric, range 0-1 (higher is better)

Evaluation can be performed on all MM-IPC variants or specific ones:

- **Indoor (with reverb):** SER, SR, ER, R
- **Outdoor (no reverb):** SE, S, E

## Dataset Structure Expected

```
PolSESS/
├── train/
│   ├── clean/         # Clean speech (ES task)
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

Each subset has its own metadata CSV file.

## Training Results

### Baseline Experiments (SB Task - Complete)

**SPMamba** - Best Performer:
```
Seed 42:  Epoch 19: Val SI-SDR = 5.68 dB (diverged @ 21 due to AMP)
Seed 123: Epoch 29: Val SI-SDR = 5.45 dB
Seed 456: Epoch 26: Val SI-SDR = 5.55 dB
Average: 5.56 dB
```

**SepFormer** - 2nd Place:
```
Seed 42:  Epoch 45: Val SI-SDR = 5.14 dB
Seed 123: Epoch 42: Val SI-SDR = 5.26 dB
Seed 456: Epoch 42: Val SI-SDR = 4.89 dB
Average: 5.10 dB
```

**Key Findings**:
- SPMamba outperforms all models despite being "reduced" architecture
- State Space Models + selective attention excel at speech separation
- Consistent performance across seeds (std dev: 0.12 dB for SPMamba)

See full results: [`sweeps/EXPERIMENT_LOG.md`](sweeps/EXPERIMENT_LOG.md)

### Hyperparameter Optimization (In Progress)

W&B sweeps running for all 4 models to optimize:
- Learning rate (1e-4 to 1e-2)
- Weight decay (1e-6 to 1e-4)
- Gradient clipping (1.0 to 10.0)
- LR scheduler factor (0.3 to 0.8)

## Troubleshooting

### NaN in SI-SDR

- **Cause:** SpeechBrain's EPS=1e-8 underflows in float16
- **Fix:** Automatically applied via `apply_eps_patch()` when AMP is enabled

### GPU Memory Overflow

- **Symptom:** Training becomes very slow (~35s per batch)
- **Cause:** Batch size too large, overflowing to system RAM
- **Fix:** Reduce `--batch-size`

### Unstable Validation Metrics

- **Cause:** Validation set too small (< 50 samples)
- **Fix:** Use larger validation set or test set for validation

## Configuration Details

All configuration is centralized in [`config.py`](config.py) with three sections:

**DataConfig:**

- Data paths, CSV filenames, batch size, task (ES/EB)

**ModelConfig:**

- ConvTasNet architecture parameters (N, B, H, P, X, R, C)

**TrainingConfig:**

- Learning rate, weight decay, grad clipping, epochs, AMP settings

**Configuration Priority:**

1. Hardcoded defaults in config.py
2. Environment variables (e.g., `$POLSESS_DATA_ROOT`)
3. YAML config files (e.g., `--config experiments/baseline.yaml`)
4. CLI arguments (highest priority)

See [CONFIG_GUIDE.md](CONFIG_GUIDE.md) for detailed configuration documentation.

## Hardware Requirements

- **GPU:** 12GB VRAM (tested on RTX 4070)
- **RAM:** 16GB+ recommended
- **Disk:** Depends on dataset size

With batch_size=4, the model uses:

- **GPU VRAM:** ~3.5 GB
- **System RAM:** ~2 GB
- **Training speed:** 0.3s per batch

**SPMamba Specific** (FP32 for stability):
- **GPU VRAM:** ~11.3 GB (batch_size=1)
- **Training speed:** ~1.25 hours per epoch (30 epochs = ~37.5 hours)
- **Recommendation:** Disable AMP (`use_amp: false`) for numerical stability

## References

- **ConvTasNet:** [Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking for Speech Separation](https://arxiv.org/abs/1809.07454)
- **SpeechBrain:** [SpeechBrain: A PyTorch-based Speech Toolkit](https://github.com/speechbrain/speechbrain)
- **MM-IPC Augmentation:** Based on Klec et al.'s approach for PolSESS

## License

This project uses SpeechBrain components (Apache 2.0 License).

## Tests (with UI)

You can run the project's tests using pytest. If you'd like a graphical or interactive UI for running/exploring tests, install a pytest UI plugin such as `pytest-ui` (or any other test-runner UI you prefer) and then run pytest as usual. Example:

```powershell
# install dependencies (including pytest-ui)
pip install -r requirements.txt

# run pytest (plugin may expose additional flags; consult the plugin docs)
pytest
```

The repository already contains a `pytest.ini` which points pytest to the `tests/` directory.
