# SPMamba 2-Stage Hyperparameter Optimization

## Overview

2-stage approach based on DPRNN learnings:
- **Stage 1**: Wide search on 2K samples (fast exploration)
- **Stage 2**: Refined search on 8K samples (focused optimization)
- **Validation**: Top 3-5 configs on 16K samples with 3 seeds

## Stage 1: Wide Search (2K samples)

**Config**: `experiments/spmamba/spmamba_2000.yaml`
**Sweep**: `stage1_2k.yaml`

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
wandb sweep sweeps/spmamba-2stage/stage1_2k.yaml
wandb agent <sweep_id>
```

---

## Stage 2: Refined Search (8K samples)

**Config**: `experiments/spmamba/spmamba_8000.yaml`
**Sweep**: `stage2_8k.yaml`

⚠️ **IMPORTANT**: Update `stage2_8k.yaml` ranges after analyzing Stage 1 results!

### After Stage 1, analyze and update:

1. Check best LR values → narrow range
2. Check weight_decay → likely very low (1e-6 to ~5e-5)
3. Check grad_clip_norm → may narrow significantly
4. Check lr_factor → likely higher values (0.6+) work better

---

## Validation (16K samples, 3 seeds)

After Stage 2:
1. Select top 3-5 configs
2. Create validation config files (similar to DPRNN approach)
3. Run with seeds 42, 123, 456
4. Report mean ± std SI-SDR

---

## Expected Compute

| Stage | Samples | Epochs | Runs | Est. Time |
|-------|---------|--------|------|-----------|
| Stage 1 | 2K | 50 | 60 | ~40-50h |
| Stage 2 | 8K | 80 | 50 | ~80-100h |
| Validation | 16K | 80 | 9-15 | ~30-50h |
| **Total** | | | | **~150-200h** |

Note: SPMamba is ~3x slower than DPRNN per epoch (batch_size=1 vs 16).
