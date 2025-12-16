# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a speech processing research repository focused on speech separation and enhancement for Polish speech datasets. The main active project is in `MAG2/polsess_separation/`.

## 🎓 Master's Thesis Context

**CRITICAL: This project is part of a master's thesis submission.**

### Primary Goals (in order of priority):

1. **Code Clarity & Readability** - The code will be reviewed by academic supervisors and potentially published. Every line should be clean, well-documented, and easy to understand.

2. **Simplicity over Cleverness** - Prefer straightforward, explicit implementations over complex abstractions. The thesis examiner should be able to follow the logic without deep software engineering expertise.

3. **Reproducibility** - All experiments must be fully reproducible. Configuration, random seeds, and environment details must be meticulously documented.

4. **Academic Standards** - Follow research code conventions (similar to conference paper repositories) rather than production software patterns.

### What This Means in Practice:

**DO:**
- ✅ Write self-documenting code with clear variable/function names
- ✅ Add docstrings explaining the "why" not just the "what"
- ✅ Keep functions focused and short (prefer 20-30 lines over 100+)
- ✅ Use explicit, direct implementations that match paper descriptions
- ✅ Consolidate related utilities to reduce file count
- ✅ Maintain comprehensive tests for reproducibility verification
- ✅ Document design decisions inline when they deviate from papers

**DON'T:**
- ❌ Over-engineer with excessive abstraction layers
- ❌ Create factory patterns unless handling 4+ variants (we have model factory for 4 models - justified)
- ❌ Split simple functions into separate files "for organization"
- ❌ Use clever tricks that require explanation
- ❌ Add features "for future extensibility" - implement what's needed now
- ❌ Create deep inheritance hierarchies

### Recent Simplifications (Based on Thesis Best Practices):

1. **Evaluation Module** - Merged split modules back into single `evaluate.py` (498→420 lines)
   - *Rationale*: All top speech separation papers use single evaluation scripts
   
2. **Dataset Factory Removed** - Direct DataLoader instantiation
   - *Rationale*: Only 2 datasets, factory pattern unnecessary for thesis scope
   
3. **Utils Consolidated** - Reduced from 7→4 files by merging tiny utilities
   - *Rationale*: Fewer files = easier navigation for reviewers

4. **Model Factory Retained** - Kept for 4 model architectures
   - *Rationale*: Config-driven model selection is standard for multi-model comparison papers

### Code Review Mindset:

When writing or reviewing code, ask:
1. "Would a thesis examiner understand this without my explanation?"
2. "Is this the simplest way to achieve the goal?"
3. "Does this match how it's done in the cited papers?"
4. "Could I explain this in 2 sentences in the thesis?"

If the answer to any is "no", simplify.

## Primary Project: PolSESS Speech Separation

Located in `MAG2/polsess_separation/`, this is a PyTorch implementation of speech enhancement using ConvTasNet, SepFormer, DPRNN, and SPMamba architectures on the PolSESS dataset.

**Thesis Focus**: Comparing 4 speech separation architectures on the PolSESS dataset with Mix Modification by Inverted Phase Cancellation (MM-IPC) augmentation for Polish speech enhancement.

### Key Commands

**Training:**

```bash
# Navigate to project directory
cd MAG2/polsess_separation

# Train with YAML config (recommended)
python train.py --config experiments/baseline.yaml

# Train with custom parameters
python train.py --batch-size 8 --epochs 20 --lr 0.0001

# Train on different task (EB = both speakers)
python train.py --task EB

# Hyperparameter sweep with Weights & Biases
python train_sweep.py --config experiments/wandb_sweep.yaml
```

**Evaluation:**

```bash
# Evaluate on all MM-IPC variants (using latest symlink)
python evaluate.py --checkpoint checkpoints/convtasnet/ES/latest/best_model.pt

# Evaluate specific variant only
python evaluate.py --checkpoint checkpoints/spmamba/ES/latest/best_model.pt --variant SER

# Fast evaluation (skip PESQ and STOI)
python evaluate.py --checkpoint checkpoints/sepformer/EB/latest/best_model.pt --no-pesq --no-stoi

# Save results to CSV
python evaluate.py --checkpoint checkpoints/convtasnet/ES/latest/best_model.pt --output results.csv

# Or use specific run_id instead of latest
python evaluate.py --checkpoint checkpoints/spmamba/ES/run_2024-12-11_14-30-00/best_model.pt
```

**Testing:**

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_model.py

# Run with verbose output
pytest -v
```

**Interactive Model Testing:**

```bash
# Launch Jupyter and open test_model_interactive.ipynb
jupyter notebook test_model_interactive.ipynb

# Features:
# - Auto-discovers all checkpoints from hierarchical structure
# - Dropdown selectors for model, task, variant, and sample
# - Single-button model loading and testing
# - Real-time SI-SDR metrics and waveform visualization
# - Audio playback for mix, clean, and estimate
```

### Architecture Overview

The project follows a modular architecture with clear separation of concerns:

**Configuration System (`config.py`):**

- Centralized configuration using dataclasses
- Three main config sections: `DataConfig`, `ModelConfig`, `TrainingConfig`
- Each model/dataset has nested parameter classes (e.g., `ConvTasNetParams`, `PolSESSParams`)
- Configuration priority: defaults → env vars → YAML → CLI args (highest priority)
- Use `get_config_from_args()` to parse and merge all config sources

**Model Registry (`models/__init__.py`):**

- Models are registered with `@register_model("model_name")`
- Retrieve with `get_model("model_name")` which returns the model class
- Currently supports: `convtasnet`, `sepformer`, `dprnn`, `spmamba`
  - **convtasnet**: Time-domain convolutional architecture (~8M params)
  - **sepformer**: Transformer-based separation (~26M params)
  - **dprnn**: Dual-path RNN with intra/inter-chunk processing
  - **spmamba**: State-space model with Mamba blocks (requires Linux + CUDA)
- Add new models by decorating with `@register_model()` and implementing the class

**Dataset Registry (`datasets/__init__.py`):**

- Datasets registered with `@register_dataset("dataset_name")`
- Retrieve with `get_dataset("dataset_name")` which returns dataset class
- Currently supports: `polsess`, `libri2mix`
- All datasets must support `allowed_variants` parameter for filtering data

**Training Flow:**

1. `train.py` loads config via `get_config_from_args()`
2. Creates dataloaders using `get_dataset()` registry
3. Creates model using `get_model()` registry with nested params
4. Optionally applies `torch.compile()` for speedup (PyTorch 2.0+, Linux only)
5. Instantiates `Trainer` with model, dataloaders, config
6. Trainer handles: AMP, checkpointing, logging, curriculum learning
7. Curriculum learning: can progressively add data variants over epochs
8. Checkpoints saved to: `checkpoints/{model}/{task}/{run_id}/best_model.pt`
9. Symlink created at: `checkpoints/{model}/{task}/latest/` for easy access

**Key Technical Details:**

- **AMP (Automatic Mixed Precision):** Enabled by default for 30-40% speedup

  - SpeechBrain's `EPS=1e-8` causes float16 underflow → patched to `1e-4`
  - Patch applied automatically in `utils/amp_patch.py`

- **torch.compile:** Automatic model compilation for speedup (PyTorch 2.0+)

  - Applied automatically in `train.py` when available
  - ~10-20% training speedup on Linux systems
  - Proper checkpoint handling for compiled models (unwraps `_orig_mod` prefix)
  - No-op on Windows (compilation not supported)

- **Hierarchical Checkpoint Structure:** Organized checkpoint storage

  - Structure: `checkpoints/{model}/{task}/{run_id}/best_model.pt`
  - Each checkpoint directory includes `config.yaml` for easy viewing
  - Automatic symlink creation at `checkpoints/{model}/{task}/latest/`
  - Windows fallback uses junction instead of symlink
  - Run ID includes wandb run name when available

- **MM-IPC Augmentation:** Randomly varies background complexity during training

  - Indoor: SER, SR, ER, R (with reverb)
  - Outdoor: SE, S, E (no reverb)
  - Implemented via lazy loading in `datasets/polsess_dataset.py`

- **Curriculum Learning:** Optional progressive training strategy

  - Configure in YAML: `training.curriculum_learning`
  - Each epoch can specify: epoch number, variants, learning rate
  - Dataset variants automatically updated between epochs
  - LR scheduler can be enabled/disabled at specific epochs

- **SPMamba Requirements:** Linux + CUDA only

  - Requires `mamba-ssm` library (not available on Windows native)
  - Windows users must use WSL2 with CUDA toolkit 12.4+
  - Frequency-domain processing using STFT/iSTFT
  - O(N) complexity for efficient long-sequence processing

### Module Structure

```
MAG2/polsess_separation/
├── config.py                      # Configuration dataclasses and parsing
├── train.py                       # Main training entry point
├── train_sweep.py                 # Weights & Biases sweep entry point
├── evaluate.py                    # Evaluation with per-variant metrics
├── test_model_interactive.ipynb   # Interactive model testing with dropdowns
├── CHANGELOG.md                   # Project changelog
├── models/
│   ├── __init__.py                # Model registry (register_model, get_model)
│   ├── conv_tasnet.py             # ConvTasNet architecture
│   ├── sepformer.py               # SepFormer architecture
│   ├── dprnn.py                   # Dual-path RNN architecture
│   └── spmamba.py                 # SPMamba with Mamba blocks (Linux + CUDA only)
├── datasets/
│   ├── __init__.py                # Dataset registry (register_dataset, get_dataset)
│   ├── polsess_dataset.py         # PolSESS with MM-IPC augmentation
│   └── libri2mix_dataset.py       # Cross-dataset evaluation
├── training/
│   └── trainer.py                 # Trainer with AMP and checkpointing
├── utils/
│   ├── amp_patch.py               # SpeechBrain float16 compatibility
│   ├── common.py                  # Seeds, warnings, device setup
│   ├── logger.py                  # Colored logging
│   └── wandb_logger.py            # Weights & Biases integration
├── tests/                         # Pytest test suite
│   ├── conftest.py                # Shared fixtures
│   ├── test_dataset.py
│   ├── test_model.py
│   ├── test_trainer.py
│   └── test_utils.py
├── experiments/                   # YAML experiment configs
│   ├── baseline.yaml              # ConvTasNet baseline (9.84 dB SI-SDR)
│   ├── large_model.yaml           # Large ConvTasNet (~34M params)
│   ├── eb_task.yaml               # Both speakers task
│   ├── spmamba_baseline.yaml      # SPMamba configuration
│   └── wandb_sweep.yaml           # Hyperparameter sweep config
├── checkpoints/                   # Hierarchical checkpoint storage
│   ├── convtasnet/
│   │   └── ES/
│   │       ├── latest/            # Symlink to most recent run
│   │       └── run_{timestamp}/
│   │           ├── best_model.pt
│   │           └── config.yaml
│   ├── spmamba/
│   │   └── ES/...
│   └── sepformer/...
└── docs/                          # Detailed documentation
    ├── CONFIG_GUIDE.md
    ├── WANDB_GUIDE.md
    └── HYPERPARAMETER_SWEEP_GUIDE.md
```

### Adding New Models

1. Create model file in `models/` directory
2. Implement model class (typically inheriting from `nn.Module`)
3. Add model-specific parameters dataclass to `config.py` (e.g., `NewModelParams`)
4. Add field to `ModelConfig` for the new params
5. Register model with `@register_model("model_name")` decorator in `models/__init__.py`
6. Create experiment YAML in `experiments/` with model config
7. Add conditional logic in `train.py`'s `create_model()` if needed

Example structure:

```python
# models/new_model.py
from models import register_model
import torch.nn as nn

@register_model("newmodel")
class NewModel(nn.Module):
    def __init__(self, param1, param2):
        super().__init__()
        # implementation

# config.py
@dataclass
class NewModelParams:
    param1: int = 128
    param2: str = "default"

@dataclass
class ModelConfig:
    model_type: str = "convtasnet"
    newmodel: Optional[NewModelParams] = None
    # ... other models
```

### Adding New Datasets

1. Create dataset file in `datasets/` directory
2. Implement dataset class inheriting from `torch.utils.data.Dataset`
3. Must accept `allowed_variants` parameter for filtering
4. Add dataset-specific parameters dataclass to `config.py`
5. Add field to `DataConfig` for the new params
6. Register with `@register_dataset("dataset_name")` decorator
7. Update `train.py`'s `create_dataloaders()` with conditional logic
8. May need custom collate function if data structure differs

### Working with Configuration

**YAML Configuration Structure:**

```yaml
data:
  dataset_type: polsess
  batch_size: 4
  task: ES
  polsess:
    data_root: /path/to/dataset

model:
  model_type: convtasnet
  convtasnet:
    N: 256
    B: 256
    # ... other params

training:
  num_epochs: 10
  lr: 0.001
  use_amp: true
  curriculum_learning:
    - epoch: 0
      variants: ["SER", "SR"]
      lr: 0.001
    - epoch: 5
      variants: null  # all variants
      lr: 0.0005
```

**CLI Overrides:**
CLI arguments override YAML values. Use kebab-case for nested params:

```bash
python train.py --config baseline.yaml --lr 0.0001 --batch-size 8
```

### Dependencies

Key dependencies (from `MAG2/polsess_separation/requirements.txt`):

- `torch>=2.0.0`, `torchaudio>=2.0.0` - Core ML framework
- `speechbrain>=1.0.0` - Speech processing toolkit (source of models)
- `wandb>=0.16.0` - Experiment tracking
- `pytest>=7.4.0` - Testing framework
- `torchmetrics>=1.0.0` - Metrics (SI-SDR, PESQ, STOI)
- `pyyaml>=6.0` - Config parsing

### Environment Variables

- `POLSESS_DATA_ROOT` - Path to PolSESS dataset (default hardcoded in config.py)
- `TF_ENABLE_ONEDNN_OPTS=0` - Disable TensorFlow warnings (set in train.py)

### Dataset Structure Expected

PolSESS dataset should follow this structure:

```
PolSESS/
├── train/
│   ├── clean/              # Clean speech (target for ES task)
│   ├── event/              # Event sounds
│   ├── mix/                # Mixed audio
│   ├── scene/              # Background scene
│   ├── sp1_reverb/         # Speaker 1 with reverb
│   ├── sp2_reverb/         # Speaker 2 with reverb
│   ├── ev_reverb/          # Event with reverb
│   └── corpus_PolSESS_C_in_train_final.csv
├── val/
│   └── ...
└── test/
    └── corpus_PolSESS_C_in_test_final.csv
```

### Common Pitfalls

1. **NaN in SI-SDR:** Usually caused by AMP underflow. AMP patch should handle this automatically.

2. **Memory overflow:** Reduce batch size if training slows dramatically (~35s per batch indicates RAM overflow). Note: gradient accumulation has been removed, so physical batch size = effective batch size.

3. **Config changes not applied:** Remember CLI args > YAML > env vars > defaults. Check precedence.

4. **Missing variants:** If evaluation fails, ensure `allowed_variants` parameter is correctly set in dataset instantiation.

5. **Registry errors:** Models/datasets must be imported in `__init__.py` to trigger decorator registration.

6. **SPMamba on Windows:** mamba-ssm requires Linux + CUDA. Use WSL2 with CUDA toolkit 12.4+ on Windows.

7. **torch.compile checkpoint loading:** Old checkpoints from compiled models may have `_orig_mod.` prefix issues. New trainer automatically handles this.

8. **Checkpoint path changes:** New hierarchical structure uses `checkpoints/{model}/{task}/{run_id}/best_model.pt`. Use `latest` symlink for convenience.

9. **WSL2 CUDA setup:** Ensure CUDA toolkit is installed in WSL2, not just on Windows host. Check with `nvcc --version` inside WSL.

## Other Directories

You should not concern yourself with what's there. Only look into MAG2.

- **Archive/**: Old coursework exercises (TEG) - largely inactive
- **WUM/**: Coursework exercises - largely inactive
- **MAG/**: Earlier experiments with speech separation (predecessor to MAG2)
- **speechbrain/**: Cloned SpeechBrain toolkit repository
- **wandb/**: Local Weights & Biases run data

## Virtual Environment

The repository uses a virtual environment at `.venv/`. Activate before running:

```bash
# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

## Development Workflow

1. Create/modify experiment YAML in `experiments/`
2. Run training with config: `python train.py --config experiments/your_config.yaml`
3. Monitor on Weights & Biases dashboard
4. Test models interactively: `jupyter notebook test_model_interactive.ipynb`
   - Use dropdown widgets to select checkpoint (including `latest` symlink)
   - Choose task, variant, and sample ID
   - Get real-time SI-SDR metrics and audio playback
5. Evaluate checkpoints: `python evaluate.py --checkpoint checkpoints/{model}/{task}/latest/best_model.pt`
6. Run tests after code changes: `pytest`
7. For new features: add tests in `tests/`, update README.md, CHANGELOG.md and docs in `docs/` if needed

## Recent Changes

See [CHANGELOG.md](MAG2/polsess_separation/CHANGELOG.md) for detailed project history, including:
- SPMamba model implementation (Linux + CUDA only)
- torch.compile support for training speedup
- Hierarchical checkpoint structure with symlinks
- Interactive testing notebook with dropdown widgets
- Removal of gradient accumulation (simplified training loop)
