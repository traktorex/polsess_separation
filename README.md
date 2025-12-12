# PolSESS Speech Enhancement with ConvTasNet

PyTorch implementation of speech enhancement on the PolSESS dataset using ConvTasNet with MM-IPC augmentation.

## Performance

- **Best SI-SDR:** 9.84 dB (ES task, test set)
- **Training time:** ~0.3s per batch with AMP enabled
- **Model size:** 8.64M parameters

## Features

- **Automatic Mixed Precision (AMP):** 30-40% faster training with no quality loss
- **MM-IPC Augmentation:** Randomly varies background complexity (SER, SR, ER, R for indoor; SE, S, E for outdoor)
- **torch.compile Support:** Further speedup on Linux with PyTorch 2.0+
- **Configurable:** CLI arguments or config file
- **Modular Architecture:** Clean separation of concerns

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
python evaluate.py --checkpoint checkpoints/best_model.pt

# Evaluate on specific variant only
python evaluate.py --checkpoint checkpoints/best_model.pt --variant SER

# Fast evaluation (skip PESQ and STOI)
python evaluate.py --checkpoint checkpoints/best_model.pt --no-pesq --no-stoi

# Save results to CSV
python evaluate.py --checkpoint checkpoints/best_model.pt --output results.csv

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

Training on ES task with test set as validation (val set only had 20 samples):

```
Epoch 1: Val SI-SDR = 6.79 dB
Epoch 2: Val SI-SDR = 7.52 dB
Epoch 3: Val SI-SDR = 8.10 dB
...
Epoch 9: Val SI-SDR = 9.84 dB (best)
Epoch 10: Val SI-SDR = 9.71 dB
```

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
