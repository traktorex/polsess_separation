# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **SPMamba Model Support**: Implemented SPMamba (State-Space Model) architecture for speech separation
  - New model file: `models/spmamba.py`
  - GridNet-style architecture with Mamba blocks replacing LSTMs
  - Frequency-domain processing using STFT/iSTFT
  - O(N) complexity for efficient long-sequence processing
  - Requires `mamba-ssm` library (Linux + CUDA only)
  - Configuration: `experiments/spmamba_baseline.yaml`

- **torch.compile Support**: Automatic model compilation for speedup on PyTorch 2.0+ (Linux)
  - Automatically applied in `train.py` when available
  - ~10-20% training speedup on compatible systems
  - Proper checkpoint handling for compiled models (unwraps `_orig_mod` prefix)

- **Interactive Model Testing**: New Jupyter notebook with dropdown widgets
  - File: `test_model_interactive.ipynb`
  - Auto-discovers checkpoints from new directory structure
  - Interactive dropdowns for checkpoint, task, variant, and sample selection
  - Single-button model loading and testing
  - Real-time SI-SDR metrics and waveform visualization
  - Audio playback for mix, clean, and estimate

- **Hierarchical Checkpoint Structure**: Improved checkpoint organization
  - New structure: `checkpoints/{model}/{task}/{run_name}_{timestamp}/model.pt`
  - Separate `config.yaml` saved alongside each checkpoint
  - Automatic symlink creation for `latest` checkpoint per model/task
  - Windows junction fallback for cross-platform compatibility
  - Better organization by architecture and task

### Changed

- **Simplified Training Loop**: Removed gradient accumulation functionality
  - Removed `accumulation_steps` parameter from `config.py` TrainingConfig
  - Simplified `trainer.py` training loop (optimizer step every batch)
  - Removed conditional update logic based on accumulation
  - Updated all experiment YAML files to remove `accumulation_steps`
  - Updated README.md to reflect changes

- **Checkpoint Saving**: Enhanced checkpoint metadata and organization
  - Saves both `.pt` checkpoint and `.yaml` config file
  - Includes wandb run name in run_id when available
  - Better logging of checkpoint paths and symlinks
  - Proper handling of torch.compile wrapped models

### Fixed

- **mamba-ssm Integration Issues**: Fixed multiple import and initialization errors
  - Corrected `Block` import path: `mamba_ssm.modules.block` (not `mamba_simple`)
  - Fixed `RMSNorm` import: `mamba_ssm.ops.triton.layer_norm` (underscore, not dot)
  - Added required `mlp_cls=nn.Identity` parameter to Block initialization
  - Fixed dynamic module access using `getattr()` instead of subscript notation

- **torch.compile Checkpoint Compatibility**: Fixed state_dict key mismatch
  - Unwrap `_orig_mod` prefix when saving compiled model checkpoints
  - Ensures checkpoints can be loaded in notebooks without compilation
  - Backward compatible with old checkpoints

- **WSL2 Setup**: Documented Linux environment setup for mamba-ssm
  - CUDA 12.4 toolkit installation in WSL2
  - Virtual environment setup with proper PyTorch versions
  - Resolved torchaudio compatibility issues

- **torch.compile Windows Compatibility**: Disabled on Windows platform
  - Added platform check to skip torch.compile on non-Linux systems
  - torch.compile requires Triton which is only available on Linux
  - Prevents lazy compilation errors during first forward pass
  - Logs info message when skipping compilation

- **mamba-ssm Warning Spam**: Fixed repeated warning messages
  - Added module-level flag to show mamba-ssm import warning only once
  - Prevents warning from appearing on every model import

### Removed

- **Gradient Accumulation**: Completely removed accumulation_steps feature
  - Removed from configuration system (`config.py`)
  - Removed from trainer implementation (`training/trainer.py`)
  - Removed from all experiment YAML files
  - Removed from documentation (README.md)
  - Simplified training loop for better maintainability

## [0.1.0] - Previous Release

### Added

- ConvTasNet model implementation
- SepFormer model implementation
- DPRNN model implementation
- PolSESS dataset loader with MM-IPC augmentation
- Libri2Mix dataset loader for cross-dataset evaluation
- Automatic Mixed Precision (AMP) training support
- Curriculum learning support
- Weights & Biases integration
- Comprehensive test suite with pytest
- YAML-based configuration system
- Model and dataset registry pattern
- Per-variant evaluation script
- Multiple experiment configurations

### Features

- **AMP Training**: 30-40% speedup with float16/float32 mixed precision
- **MM-IPC Augmentation**: Lazy loading of audio variants (SER, SR, ER, R, SE, S, E)
- **Curriculum Learning**: Progressive training with variant scheduling
- **Configurable Architecture**: Easy model/dataset swapping via registry
- **Comprehensive Testing**: Unit tests for all major components

## Migration Notes

### Upgrading from Previous Version

1. **Gradient Accumulation Removal**:
   - If your YAML configs include `accumulation_steps`, remove this line
   - Physical batch size is now the effective batch size
   - If you need larger effective batch sizes, increase `batch_size` parameter
   - GPU memory requirements may increase; reduce batch_size if OOM occurs

2. **Checkpoint Structure**:
   - Old checkpoints: `checkpoints/{task}_{model}_{timestamp}.pt`
   - New checkpoints: `checkpoints/{model}/{task}/{run_name}_{timestamp}/model.pt`
   - Old checkpoints still loadable but not auto-discovered in interactive notebook
   - Use symlink `checkpoints/{model}/{task}/latest/` to access most recent model

3. **torch.compile Checkpoints**:
   - New checkpoints automatically handle torch.compile wrapping
   - Old checkpoints from compiled models may have `_orig_mod.` prefix issues
   - Solution: Retrain model or manually strip prefix when loading

4. **SPMamba Requirements**:
   - Requires Linux + CUDA for mamba-ssm library
   - Windows users must use WSL2 with CUDA toolkit
   - See README for WSL2 setup instructions

## Known Issues

- mamba-ssm library only works on Linux with CUDA (WSL2 required for Windows)
- torch.compile only provides speedup on Linux (no-op on Windows)
- Old checkpoints from compiled models may need manual prefix stripping
- Symlink creation may fail on Windows without admin rights (uses junction fallback)

## Acknowledgments

- SPMamba architecture based on: "SPMamba: State-space model is all you need in speech separation"
- mamba-ssm library: [https://github.com/state-spaces/mamba](https://github.com/state-spaces/mamba)
- ConvTasNet, SepFormer implementations adapted from SpeechBrain toolkit
