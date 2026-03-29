# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a speech processing research repository focused on speech separation and enhancement for Polish speech datasets. The main active project is in `polsess_separation/`.

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
- ❌ Create factory patterns unless handling 4+ variants (we have model factory for 6+ models — justified)
- ❌ Split simple functions into separate files "for organization"
- ❌ Use clever tricks that require explanation
- ❌ Add features "for future extensibility" — implement what's needed now
- ❌ Create deep inheritance hierarchies

### Recent Simplifications (Based on Thesis Best Practices):

1. **Evaluation Module** — Merged split modules back into single `evaluate.py`
   - *Rationale*: All top speech separation papers use single evaluation scripts

2. **Dataset Factory Removed** — Direct DataLoader instantiation
   - *Rationale*: Only 2 datasets, factory pattern unnecessary for thesis scope

3. **Utils Consolidated** — Reduced from 7→4 files by merging tiny utilities
   - *Rationale*: Fewer files = easier navigation for reviewers

4. **Model Factory Retained** — Kept for 4 model architectures
   - *Rationale*: Config-driven model selection is standard for multi-model comparison papers

### Code Review Mindset:

When writing or reviewing code, ask:
1. "Would a thesis examiner understand this without my explanation?"
2. "Is this the simplest way to achieve the goal?"
3. "Does this match how it's done in the cited papers?"
4. "Could I explain this in 2 sentences in the thesis?"

If the answer to any is "no", simplify.

## Primary Project: PolSESS Speech Separation

Located in `polsess_separation/`, this is a PyTorch implementation of speech separation using ConvTasNet, SepFormer, DPRNN, SPMamba, SPMamba3, Mamba-TasNet, and DPMamba architectures on the PolSESS dataset.

**Thesis Focus**: Training robust speech separation models on the PolSESS dataset for downstream Polish ASR preprocessing.

## Current State (March 2026)

### Baseline Experiments Complete (SB Task):

| Model | Avg SI-SDR | Best Run | Status |
|-------|------------|----------|--------|
| **SPMamba** 🏆 | **5.56 dB** | 5.68 dB | ✅ Baseline complete |
| SepFormer | 5.10 dB | 5.26 dB | ✅ Baseline complete |
| DPRNN | 3.03 dB | 3.20 dB | ✅ Baseline complete |
| ConvTasNet | 2.95 dB | 3.28 dB | ✅ Baseline complete |

**Full Results**: See [`sweeps/EXPERIMENT_LOG.md`](sweeps/EXPERIMENT_LOG.md)

### Research Pipeline:

**✅ Phase 1A Complete** — Baseline Performance Established
- Trained all 4 architectures for 3 seeds each (~187 GPU hours)
- Identified SPMamba as best model (5.56 dB average SI-SDR)

**✅ Phase 1B Complete (DPRNN)** — Hyperparameter Optimization
- 3-stage progressive scaling: **4.67 dB** (+1.64 dB vs. baseline, +54%)
- Best HPs: LR=0.00125, WD=2.1e-5, grad_clip=2.76, lr_factor=0.863, lr_patience=3
- Key finding: weight decay is the strongest predictor (correlation −0.27)

**✅ Phase 1B Complete (ConvTasNet)** — 2-Stage HPO
- Stage 2 sweeps + 16K validation complete
- Best: stilted-sweep-16 → **3.68 dB** (+0.73 dB vs baseline, +25%)

**✅ Phase 1B Complete (SPMamba)** — 2-Stage HPO
- Stage 2 sweeps + 16K validation complete (2 seeds per config)
- Best: glowing-sweep-9 → **5.94 dB** (+0.38 dB vs baseline, +7%)

**🔄 Phase 1B In Progress (SepFormer)** — 2-Stage HPO
- Stage 1 (2K) uninformative — SepFormer overfits severely on small subsets
- Stage 2 (8K) sweep complete — best: dutiful-sweep-9 → **4.30 dB**
- Validation (16K, 3 seeds) pending

**🔄 Phase 1C In Progress** — Architecture Variants
- SPMamba3 implemented: SPMamba with Mamba-3 SSM blocks (novel — first Mamba-3 audio application)
- Requires `venv_mamba3` (torch 2.11.0+cu130, triton 3.6.0, mamba-ssm from git with Mamba-3 code)
- Config: `experiments/spmamba3/spmamba3_baseline.yaml`; all 11 unit tests passing

**🔄 Phase 2 In Progress** — ASR Integration
- Evaluate separation as ASR preprocessing via Whisper WER/CER
- Two evaluation datasets: LibriSpeechMixASR (synthetic) and REAL-M (real-world)
- SPMamba results: Libri2Mix WER=18.80%, REAL-M WER=63.31% (Whisper large)

### Key Commands

**Training:**

```bash
cd polsess_separation

# Train with YAML config (recommended)
python train.py --config experiments/dprnn/dprnn_baseline.yaml

# Override model type or task
python train.py --config experiments/baseline.yaml --model-type spmamba
python train.py --config experiments/baseline.yaml --task SB

# Disable W&B logging
python train.py --config experiments/baseline.yaml --no-wandb

# Resume from checkpoint
python train.py --config experiments/baseline.yaml --resume checkpoints/dprnn/SB/run_name/dprnn_SB_best.pt

# Override seed
python train.py --config experiments/baseline.yaml --seed 123
```

**Evaluation:**

```bash
# Evaluate on all MM-IPC variants (PolSESS)
python evaluate.py --checkpoint checkpoints/dprnn/SB/run_name/dprnn_SB_best.pt

# Evaluate specific variant only
python evaluate.py --checkpoint checkpoints/dprnn/SB/run_name/dprnn_SB_best.pt --variant SER

# Fast evaluation (skip PESQ and STOI)
python evaluate.py --checkpoint checkpoints/dprnn/SB/run_name/dprnn_SB_best.pt --no-pesq --no-stoi

# Evaluate on Libri2Mix (both Libri2Mix-Clean and Libri2Mix-Noisy)
python evaluate.py --checkpoint path/to/model.pt \
  --dataset librimix --librimix-root /home/user/datasets/LibriMix/Libri2Mix --no-pesq --no-stoi

# Evaluate on specific Libri2Mix variant
python evaluate.py --checkpoint path/to/model.pt \
  --dataset librimix --librimix-root /home/user/datasets/LibriMix/Libri2Mix --mix-type mix_clean

# Save results to CSV
python evaluate.py --checkpoint checkpoints/dprnn/SB/run_name/dprnn_SB_best.pt --output results.csv
```

**Testing:**

```bash
pytest
pytest --cov=. --cov-report=html
pytest tests/test_model.py
pytest -v
```

**ASR Evaluation:**

```bash
# Evaluate separation model on REAL-M (WER/CER)
python asr/evaluate_asr.py --checkpoint checkpoints/spmamba/SB/.../best.pt \
    --dataset realm --mode separation --whisper-model large

# Mixture baseline (no separation)
python asr/evaluate_asr.py --dataset realm --mode mixture

# Clean source baseline on LibriSpeech
python asr/evaluate_asr.py --dataset librispeech --mode baseline

# Custom Whisper model and device
python asr/evaluate_asr.py --checkpoint path/to/model.pt \
    --dataset librispeech --mode separation --whisper-model base.en --whisper-device cuda
```

**Interactive Model Testing:**

```bash
jupyter notebook test_model_interactive.ipynb
# - Auto-discovers all checkpoints from hierarchical structure
# - Dropdown selectors for model, task, variant, and sample
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
- Use `get_config_from_args()` for CLI training, `load_config_for_run(wandb.config)` for sweeps

**Model Registry (`models/__init__.py`):**

- Simple dict-based registry: `MODELS = {"convtasnet": ConvTasNet, ...}`
- Retrieve with `get_model("model_name")` which returns the model class
- Currently supports: `convtasnet`, `sepformer`, `dprnn`, `spmamba`, `spmamba3`, `mamba_tasnet`, `dpmamba`
  - **convtasnet**: Time-domain convolutional architecture (~8M params)
  - **sepformer**: Transformer-based separation (~26M params)
  - **dprnn**: Dual-path RNN with intra/inter-chunk processing (~2-3M params)
  - **spmamba**: State-space model with Mamba-1 blocks (~1.2M params, requires Linux + CUDA)
  - **spmamba3**: SPMamba with Mamba-3 blocks — exp-trapezoidal discretization, complex RoPE states (~1.5M params, requires `venv_mamba3`) (spmamba3 is on its own branch right now)
  - **mamba_tasnet**: Single-path BiMamba separator (Jiang et al., 2024). LayerNorm → bottleneck → MambaBlocksSequential → mask. Configs: XS/S/M/L (2.2M–59.6M params). Requires Linux + CUDA.
  - **dpmamba**: Dual-path BiMamba separator (Jiang et al., 2024). Uses SpeechBrain's Dual_Path_Model with MambaBlocksSequential for intra/inter paths. `n_mamba_dp` is total blocks across both paths (each gets half). Configs: XS/S/M/L (2.3M–59.8M params). Requires Linux + CUDA.

**Dataset Registry (`datasets/__init__.py`):**

- Simple dict-based registry: `DATASETS = {"polsess": PolSESSDataset, ...}`
- Retrieve with `get_dataset("dataset_name")` which returns dataset class
- Currently supports: `polsess`, `libri2mix`

**Training Flow:**

1. `train.py` loads config via `get_config_from_args()` (or `load_config_for_run(wandb.config)` for sweeps)
2. Creates dataloaders directly (no factory — only 2 datasets)
3. Creates model via `create_model_from_config()` in `models/factory.py`
4. Optionally applies `torch.compile()` for speedup (PyTorch 2.0+, Linux only)
5. Instantiates `Trainer` with model, dataloaders, config
6. Trainer handles: AMP, gradient accumulation, checkpointing, logging, curriculum learning
7. Curriculum learning: progressively adds data variants over epochs
8. Checkpoints saved to: `checkpoints/{model}/{task}/{run_name}/model.pt`

**Key Technical Details:**

- **AMP (Automatic Mixed Precision):** Enabled by default for 30–40% speedup
  - SpeechBrain's `EPS=1e-8` causes float16 underflow → patched to `1e-4` in `utils/common.py`

- **torch.compile:** Automatic model compilation for speedup (PyTorch 2.0+)
  - Applied automatically in `train.py` when available
  - ~10–20% training speedup on Linux systems
  - Proper checkpoint handling for compiled models (unwraps `_orig_mod` prefix)
  - No-op on Windows (compilation not supported)

- **Checkpoint Structure:** `checkpoints/{model}/{task}/{run_name}/`
  - Each checkpoint directory includes `config.yaml` for easy viewing
  - `save_all_checkpoints: true` keeps every improvement; default overwrites single best file
  - Run name comes from W&B run name when available, falls back to timestamp

- **MM-IPC Augmentation:** Randomly varies background complexity during training
  - Indoor: SER, SR, ER, R (with reverb)
  - Outdoor: SE, S, E, C (no reverb)
  - Implemented via lazy loading in `datasets/polsess_dataset.py`
  - Validation uses deterministic selection (seeded by sample index)

- **Curriculum Learning:** Optional progressive training strategy
  - Configure in YAML: `training.curriculum_learning`
  - Each entry specifies: `epoch`, `variants`, optionally `lr_scheduler: start`
  - Dataset variants automatically updated between epochs
  - LR scheduler can be gated to start at a specific curriculum stage

- **Early Stopping:** Optional training termination on plateau
  - Configure in YAML: `training.early_stopping_patience`
  - Monitors validation SI-SDR every epoch
  - Stops if no improvement for N epochs

- **Gradient Accumulation:** Effective batch size scaling
  - Configure in YAML: `training.grad_accumulation_steps`
  - Effective batch = `batch_size × grad_accumulation_steps`
  - Useful when VRAM limits physical batch size

- **Mamba Models (SPMamba, Mamba-TasNet, DPMamba):** Linux + CUDA only
  - Require `mamba-ssm` library (not available on Windows native)
  - Windows users must use WSL2 with CUDA toolkit 12.4+
  - AMP uses bfloat16 (no GradScaler) — Mamba CUDA kernels run float32 internally

### Module Structure

```
polsess_separation/
├── config.py                      # Configuration dataclasses and parsing
├── train.py                       # Main training entry point
├── train_sweep.py                 # Weights & Biases sweep entry point
├── evaluate.py                    # Evaluation with per-variant metrics
├── test_model_interactive.ipynb   # Interactive model testing with dropdowns
├── asr/
│   ├── __init__.py                # Public API
│   ├── dataset.py                 # LibriSpeechMixDataset + RealMDataset
│   ├── evaluate_asr.py            # Unified ASR evaluation (separation/mixture/baseline)
│   ├── metrics.py                 # WER/CER via jiwer (micro-averaged)
│   ├── transcribe.py              # WhisperTranscriber wrapper
│   └── archive/                   # One-time data prep scripts (already run)
├── models/
│   ├── __init__.py                # Model registry (dict-based, get_model)
│   ├── factory.py                 # Config-driven model instantiation
│   ├── conv_tasnet.py             # ConvTasNet architecture
│   ├── sepformer.py               # SepFormer architecture
│   ├── dprnn.py                   # Dual-path RNN architecture
│   ├── spmamba.py                 # SPMamba with Mamba-1 blocks (Linux + CUDA only)
│   ├── spmamba3.py                # SPMamba3 with Mamba-3 blocks (requires venv_mamba3)
│   ├── mamba_tasnet.py            # Mamba-TasNet: single-path BiMamba separator
│   ├── dpmamba.py                 # DPMamba: dual-path BiMamba separator
│   └── mamba/                     # Mamba building blocks (adapted from xi-j/Mamba-TasNet)
│       ├── __init__.py            # Exports MambaBlocksSequential, MAMBA_AVAILABLE
│       ├── selective_scan_interface.py  # MambaInnerFnNoOutProj (derived from mamba-ssm 2.3.1)
│       ├── bimamba.py             # BiMamba v2 mixer (bidirectional SSM)
│       └── mamba_blocks.py        # MambaBlocksSequential (N Block layers + GPT-2 init)
├── datasets/
│   ├── __init__.py                # Dataset registry (dict-based, get_dataset)
│   ├── polsess_dataset.py         # PolSESS with MM-IPC augmentation
│   └── libri2mix_dataset.py       # Libri2Mix 2-speaker separation (mix_clean + mix_both)
├── training/
│   └── trainer.py                 # Trainer with AMP, grad accumulation, checkpointing
├── utils/
│   ├── common.py                  # Seeds, EPS patch, warnings, device setup
│   ├── model_utils.py             # torch.compile, param counting, checkpoint loading
│   ├── logger.py                  # Colored logging
│   └── wandb_logger.py            # Weights & Biases integration
├── tests/                         # Pytest test suite (264 tests, 17 files)
│   ├── conftest.py                # Shared fixtures
│   ├── test_config_yaml.py        # Config loading, saving, summary
│   ├── test_dataset.py
│   ├── test_mmipc.py
│   ├── test_model.py
│   ├── test_trainer.py
│   └── ...
├── experiments/                   # YAML experiment configs
│   ├── convtasnet/
│   ├── dprnn/
│   ├── sepformer/
│   ├── spmamba/
│   ├── mamba_tasnet/              # XS/S/M/L configs
│   └── dpmamba/                   # XS/S/M/L configs
└── sweeps/                        # W&B sweep configs and experiment log
    ├── EXPERIMENT_LOG.md
    ├── 1-baselines-SB/
    └── 3-hyperparam-opt/
```

### Adding New Models

1. Create model file in `models/` directory
2. Implement model class inheriting from `nn.Module`
3. Add model-specific parameters dataclass to `config.py` (e.g., `NewModelParams`)
4. Add `Optional[NewModelParams]` field to `ModelConfig`
5. Add entry to `MODELS` dict in `models/__init__.py`
6. Add initialization logic to `ModelConfig.__post_init__`
7. Create experiment YAML in `experiments/` with model config

Example:

```python
# models/new_model.py
import torch.nn as nn

class NewModel(nn.Module):
    def __init__(self, param1, param2):
        super().__init__()
        # implementation

# models/__init__.py
from models.new_model import NewModel
MODELS = {
    "newmodel": NewModel,
    # ... existing models
}

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
3. Add dataset-specific parameters dataclass to `config.py`
4. Add `Optional[NewDatasetParams]` field to `DataConfig`
5. Add entry to `DATASETS` dict in `datasets/__init__.py`
6. Update `train.py`'s dataloader creation with conditional logic
7. Add custom collate function if data structure differs

### Working with Configuration

**YAML Configuration Structure:**

```yaml
data:
  dataset_type: polsess
  batch_size: 4
  task: SB

model:
  model_type: dprnn
  dprnn:
    N: 64
    hidden_size: 128
    num_layers: 6

training:
  num_epochs: 80
  lr: 0.00125
  weight_decay: 2.1e-5
  grad_clip_norm: 2.76
  lr_factor: 0.863
  lr_patience: 3
  use_amp: true
  seed: 42
  use_wandb: true
  wandb_project: polsess-separation
  curriculum_learning:
    - epoch: 1
      variants: ["C", "R"]
    - epoch: 3
      variants: ["C", "R", "SR", "S"]
    - epoch: 5
      variants: ["R", "SR", "S", "SE", "ER", "E", "SER"]
      lr_scheduler: start
  validation_variants: ["SER", "SE"]
```

**CLI Overrides** (only a subset of params are exposed as CLI args):

```bash
python train.py --config baseline.yaml --model-type spmamba --task SB --no-wandb --seed 123
```

### Dependencies

Key dependencies (from `requirements.txt`):

- `torch>=2.0.0`, `torchaudio>=2.0.0` — Core ML framework
- `speechbrain>=1.0.0` — Speech processing toolkit (source of models)
- `wandb>=0.16.0` — Experiment tracking
- `pytest>=7.4.0` — Testing framework
- `torchmetrics>=1.0.0` — Metrics (SI-SDR, PESQ, STOI)
- `pyyaml>=6.0` — Config parsing
- `jiwer` — WER/CER computation (ASR evaluation)
- `openai-whisper` — ASR transcription (ASR evaluation)
- `mamba-ssm` — Required for SPMamba (Linux + CUDA only)

### Environment Variables

- `POLSESS_DATA_ROOT` — Path to PolSESS dataset (default hardcoded in `config.py`)
- `REALM_DATA_ROOT` — Path to REAL-M dataset (default: `~/datasets/REAL-M-v0.1.0/`)
- `LIBRIMIX_ASR_ROOT` — Path to LibriSpeechMixASR dataset (default: `~/datasets/LibriSpeechMixASR/`)
- `TF_ENABLE_ONEDNN_OPTS=0` — Disable TensorFlow warnings (set in `train.py`)

### Dataset Structure Expected

**PolSESS** (training + evaluation):
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

**Libri2Mix** (evaluation only — SI-SDR separation benchmark):
```
Libri2Mix/wav8k/min/test/
├── mix_clean/    # Libri2Mix-Clean: s1 + s2
├── mix_both/     # Libri2Mix-Noisy: s1 + s2 + WHAM noise
├── s1/           # Speaker 1 target
├── s2/           # Speaker 2 target
└── noise/        # Noise component
```
Generated via `LibriMix/generate_librimix.sh` (test subset only, 3000 samples, 8kHz, min mode).

**LibriSpeechMixASR** (ASR evaluation — synthetic 2-speaker mixes with transcriptions):
```
LibriSpeechMixASR/
├── {split}/                        # dev, test
│   ├── mix/                        # 2-speaker mixtures at 16kHz
│   ├── s1/, s2/                    # Clean source audio
│   └── metadata.csv                # sample_id, transcript1, transcript2, s1_path, s2_path, mix_path
```

**REAL-M** (ASR evaluation — 1,436 real-world 2-speaker mixtures):
```
REAL-M-v0.1.0/
├── audio_files_converted_8000Hz/   # Real 2-speaker mixtures at 8kHz
│   └── {session_id}/mixture_*.wav
├── transcriptions/                 # Per-speaker text transcripts (CSV)
│   └── {session_id}.csv            # Two CSV formats (early vs newer collections)
└── separations/                    # Pre-computed SepFormer separations
```

### Common Pitfalls

1. **NaN in SI-SDR:** Usually caused by AMP underflow. AMP patch should handle this automatically. If it persists, disable AMP (`use_amp: false`).

2. **Memory overflow:** Reduce `batch_size` in YAML if training slows dramatically (~35s per batch indicates RAM overflow). Use `grad_accumulation_steps` to maintain effective batch size.

3. **Config changes not applied:** Remember CLI args > YAML > env vars > defaults. Check precedence.

4. **Missing variants:** If evaluation fails, ensure `allowed_variants` parameter is correctly set in dataset instantiation.

5. **SPMamba on Windows:** `mamba-ssm` requires Linux + CUDA. Use WSL2 with CUDA toolkit 12.4+ on Windows.

6. **torch.compile checkpoint loading:** Old checkpoints from compiled models may have `_orig_mod.` prefix issues. `load_model_from_checkpoint()` in `utils/model_utils.py` handles this automatically.

7. **WSL2 CUDA setup:** Ensure CUDA toolkit is installed in WSL2, not just on Windows host. Check with `nvcc --version` inside WSL.

8. **Sweep config access:** `load_config_for_run(wandb.config)` uses `getattr(sweep_config, key)` — not dict access — for compatibility with `wandb.config` objects.


## Virtual Environments

The repository uses two virtual environments:

**Main (`venv/`)** — all models except SPMamba3:
```bash
polsess_venv
```

**SPMamba3 (`venv_mamba3/`)** — required for SPMamba3 only (torch 2.11.0+cu130, triton 3.6.0, Mamba-3 kernels):
```bash
source venv_mamba3/bin/activate
```
This venv is a clone of the main venv with Mamba-3 support added. The Mamba-3 Python files (not included in the standard mamba-ssm pip package) are manually copied from a bare repo clone of `state-spaces/mamba` at `/tmp/mamba_repo` into the installed package. Additional deps: `tilelang`, `quack-kernels`, `cuda-bindings`, `nvidia-cutlass-dsl`. To rebuild: clone main venv → fix all shebang lines in `bin/` to point to `venv_mamba3` → re-clone bare repo → extract files → install 4 deps → downgrade numpy to `<2.4`.

## Development Workflow

1. Create/modify experiment YAML in `experiments/`
2. Run training: `python train.py --config experiments/your_config.yaml`
3. Monitor on Weights & Biases dashboard
4. Test models interactively: `jupyter notebook test_model_interactive.ipynb`
5. Evaluate checkpoints: `python evaluate.py --checkpoint checkpoints/{model}/{task}/{run_name}/model.pt`
6. Run tests after code changes: `pytest`
7. For new features: add tests in `tests/`, update `README.md` and `CLAUDE.md` if needed
