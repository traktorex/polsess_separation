# Baseline Sweeps - README

## Series 1: Baseline Experiments (SB Task)

**Purpose**: Establish reference performance for each model using paper-recommended hyperparameters

### Sweeps

| Model | Sweep Name | Runs | Epochs | Training Time |
|-------|-----------|------|--------|---------------|
| ConvTasNet | 1-baseline-SB-convtasnet | 3 | 100 | ~21 hours |
| DPRNN | 1-baseline-SB-dprnn | 3 | 100 | ~15 hours |
| SepFormer | 1-baseline-SB-sepformer | 3 | 50 | ~30 hours |
| SPMamba | 1-baseline-SB-spmamba | 3 | 30 | ~90 hours |

**Total Estimated Time**: ~156 GPU-hours

### Configuration Details

#### Common Across All Models:
- **Task**: SB (Separate Both speakers)
- **Seeds**: [42, 123, 456]
- **Curriculum Learning**: Standardized schedule
  ```yaml
  - epoch 1: ["C", "R"]
  - epoch 3: ["C", "R", "SR", "S"]
  - epoch 5+: ["R", "SR", "S", "SE", "ER", "E", "SER"] + LR scheduler starts
  ```
- **Validation**: ["SER", "SE"]

#### Model-Specific:

**ConvTasNet**:
- Architecture: N=256, B=256 (optimized, not paper N=512, B=128)
- LR: 0.001, weight_decay: 0.0
- LR scheduler: 0.5 factor, patience 2

**DPRNN**:
- Architecture: Paper specification ✅
- LR: 0.00015, weight_decay: 0.0
- LR scheduler: 0.98 factor, patience 2 (paper spec)

**SepFormer**:
- Architecture: Paper specification ✅
- LR: 0.00015, weight_decay: 0.0
- LR scheduler: 0.5 factor, patience 3

**SPMamba**:
- Architecture: Reduced (memory-constrained) ⚠️
  - 4 layers (vs. paper 6)
  - 192 hidden (vs. paper 256)
  - 2 heads (vs. paper 4)
- LR: 0.001, weight_decay: 0.0
- LR scheduler: 0.5 factor, patience 2

### How to Run

#### 1. Test with short runs (4 epochs):
```bash
# Test ConvTasNet first
wandb sweep sweeps/1-baselines-SB/convtasnet.yaml
wandb agent <sweep_id> --count 1
```

#### 2. Run all baselines:
```bash
# Run all 4 sweeps
for model in convtasnet dprnn sepformer spmamba; do
    echo "Starting sweep for $model..."
    sweep_id=$(wandb sweep sweeps/1-baselines-SB/${model}.yaml 2>&1 | grep "wandb agent" | awk '{print $3}')
    wandb agent $sweep_id --count 3
done
```

#### 3. Monitor progress:
- W&B Dashboard: https://wandb.ai/<your-entity>/polsess-thesis-experiments
- Look for sweeps: `1-baseline-SB-*`

### Expected Results

After completion, you should have:
- 12 total runs (4 models × 3 seeds)
- Mean±std SI-SDR for each model
- Baseline table for thesis Chapter 4.2

### Next Steps

1. Analyze results → Create baseline comparison table
2. Identify best-performing model(s)
3. Proceed to Series 2 (Model Comparison) for statistical validation
