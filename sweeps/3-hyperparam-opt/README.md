# Series 3: Hyperparameter Optimization

This directory contains W&B sweep configurations for optimizing training hyperparameters.

## Overview

**Goal**: Find optimal learning rate, weight decay, gradient clipping, and LR scheduler settings for each model.

**Baseline Results** (to beat):
- SPMamba: 5.56 dB
- SepFormer: 5.10 dB  
- DPRNN: 3.03 dB
- ConvTasNet: 2.95 dB

## Strategy

- **Method**: Bayesian optimization with Hyperband early termination
- **Scope**: Training hyperparameters only (architecture fixed)
- **Seeds**: Single seed (42) for search, best config re-run with 3 seeds
- **Budget**: ~110 runs total across 4 models

## Hyperparameters Being Optimized

| Parameter | Range | Distribution |
|-----------|-------|--------------|
| `lr` | 1e-4 to 1e-2 | log-uniform |
| `weight_decay` | 1e-6 to 1e-4 | log-uniform |
| `grad_clip_norm` | 1.0 to 10.0 | uniform |
| `lr_factor` | 0.3 to 0.8 | uniform |

**Fixed Parameters**:
- `lr_patience`: 2 (always)
- Architecture parameters (deferred to separate sweeps)
- Curriculum learning schedule
- Validation variants: ["SER", "SE"]

## Sweep Configurations

### 1. ConvTasNet (`convtasnet.yaml`)
- **Runs**: 20
- **Epochs**: 50
- **Early stop**: After 15 epochs if poor

### 2. DPRNN (`dprnn.yaml`)
- **Runs**: 25
- **Epochs**: 100  
- **Early stop**: After 20 epochs if poor

### 3. SepFormer (`sepformer.yaml`)
- **Runs**: 30
- **Epochs**: 50
- **Early stop**: After 15 epochs if poor

### 4. SPMamba (`spmamba.yaml`)
- **Runs**: 35 (highest potential)
- **Epochs**: 30
- **Early stop**: After 10 epochs if poor

## How to Run

### Launch a sweep:
```bash
# 1. Create sweep on W&B
wandb sweep sweeps/3-hyperparam-opt/spmamba.yaml

# 2. Start agent (returns sweep ID)
wandb agent <your-entity>/<project>/<sweep-id>
```

### Monitor progress:
- Check W&B dashboard for parallel coordinate plots
- Best run will be highlighted automatically
- Hyperband will terminate poor runs early

## Expected Timeline

| Model | Runtime/Run | Total Time | Expected Improvement |
|-------|-------------|------------|---------------------|
| ConvTasNet | ~9h | ~180h (7.5 days) | +0.2-0.4 dB |
| DPRNN | ~2.5h | ~62h (2.6 days) | +0.3-0.5 dB |
| SepFormer | ~12h | ~360h (15 days) | +0.3-0.5 dB |
| SPMamba | ~25h | ~875h (36 days) | +0.2-0.4 dB |

**Total**: ~60 days sequential, ~15 days if run in parallel

> **Note**: Early stopping will reduce actual runtime significantly.

## Next Phase: Architecture Optimization

Once training hyperparameters are optimized, we can run separate sweeps to test:
- SPMamba: 4 vs 5 vs 6 layers
- SepFormer: 2 vs 4 vs 6 transformer layers
- DPRNN: Different hidden dimensions

This allows us to analyze performance vs model size independently.

## Validation Strategy

After finding best hyperparameters:
1. Re-run best config with seeds [42, 123, 456]
2. Compare to baseline (3 seeds average)
3. Update experiment log with improvements
