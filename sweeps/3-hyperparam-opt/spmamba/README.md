# SPMamba 2-Stage Hyperparameter Optimization

## Overview

2-stage approach based on DPRNN learnings:
- **Stage 1**: Wide search on 2K samples (fast exploration)
- **Stage 2**: Refined search on 8K samples (focused optimization)
- **Validation**: Top 3-5 configs on 16K samples with 3 seeds

## Stage 1: Wide Search (2K samples)

**Config**: `experiments/spmamba/3-hyperparamopt/spmamba_2000.yaml`
**Sweep**: `stage1.yaml`

### Search Ranges

| Parameter | Range | Rationale |
|-----------|-------|-----------|
| LR | [3e-4, 3e-3] | SPMamba paper uses 1e-3, search around it |
| Weight Decay | [1e-6, 1e-4] | DPRNN showed very low WD is optimal |
| Grad Clip | [0.5, 10.0] | SPMamba baseline uses 2.0, search wider |
| LR Factor | [0.3, 0.95] | Paper uses 0.5, but DPRNN showed ~0.86 worked well |
| LR Patience | [1, 2, 3, 4, 5] | Grid search |

### To Run Stage 1

```bash
wandb sweep sweeps/3-hyperparam-opt/spmamba/stage1.yaml
wandb agent <sweep_id>
```

---

## Stage 2: Refined Search (8K samples)

**Config**: `experiments/spmamba/3-hyperparamopt/spmamba_8000.yaml`
**Sweep**: `stage2.yaml`

The Stage 2 ranges in `stage2.yaml` were narrowed from the Stage 1 results; the
annotated parameter block in that file records which Stage 1 runs each bound came
from. Narrowing covered LR, weight decay, grad clip and LR factor.

---

## Validation (16K samples, 3 seeds)

The top Stage 2 configs were re-run on 16K samples with multiple seeds. The
validation configs are in
[`experiments/spmamba/3-hyperparamopt-stage2-vals/`](../../../experiments/spmamba/3-hyperparamopt-stage2-vals/);
the resulting numbers are in [`EXPERIMENT_LOG_monolithic.md`](../../EXPERIMENT_LOG_monolithic.md)
and in the thesis.

---

## Expected Compute

| Stage | Samples | Epochs | Runs | Est. Time |
|-------|---------|--------|------|-----------|
| Stage 1 | 2K | 50 | 60 | ~40-50h |
| Stage 2 | 8K | 80 | 50 | ~80-100h |
| Validation | 16K | 80 | 9-15 | ~30-50h |
| **Total** | | | | **~150-200h** |

Note: SPMamba is ~3x slower than DPRNN per epoch (batch_size=1 vs 16).
