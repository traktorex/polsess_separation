# Thesis Experiment Log

## Overview

**Project**: PolSESS Speech Separation for Polish ASR Preprocessing  
**Focus**: SB (Separate Both) task - 2-speaker separation  
**Models**: ConvTasNet, DPRNN, SepFormer, SPMamba

---

## Series 1: Baseline Experiments

**Status**: Not Started  
**Purpose**: Establish reference performance for each model

| Sweep | Status | Runs | Runtime | Best SI-SDR | W&B Link | Notes |
|-------|--------|------|---------|-------------|----------|-------|
| convtasnet | ⬜ | 0/3 | - | - | - | N=256, B=256 (optimized) |
| dprnn | ⬜ | 0/3 | - | - | - | Paper spec ✅ |
| sepformer | ⬜ | 0/3 | - | - | - | Paper spec ✅ |
| spmamba | ⬜ | 0/3 | - | - | - | Reduced arch (memory) |

**Summary**: -

---

## Series 2: Model Comparison

**Status**: Not Started  
**Purpose**: Statistical comparison to identify best model for ASR preprocessing

| Metric | Result | Notes |
|--------|--------|-------|
| Best Model | - | - |
| SI-SDR Ranking | - | - |
| Statistical Significance | - | - |

---

## Series 3: Hyperparameter Optimization

**Status**: Not Started  
**Purpose**: Optimize training hyperparameters (lr, weight_decay, grad_clip_norm)

| Model | Status | Best Config | SI-SDR Improvement |
|-------|--------|-------------|-------------------|
| convtasnet | ⬜ | - | - |
| dprnn | ⬜ | - | - |
| sepformer | ⬜ | - | - |
| spmamba | ⬜ | - | - |

---

## Key Findings

1. _To be filled after experiments_
2. _..._

---

## Notes

- All experiments use curriculum learning (standardized schedule)
- Validation on ["SER", "SE"] variants
- ConvTasNet uses optimized N=256, B=256 (not paper N=512, B=128)
- SPMamba uses reduced architecture for 12GB GPU compatibility

---

## Timeline

| Date | Event |
|------|-------|
| 2025-12-23 | Created sweep configurations |
| TBD | Start Series 1 baselines |
| TBD | Complete Series 1 |
| TBD | Analyze baselines |
| TBD | Start Series 2 comparison |
| TBD | Complete all experiments |
