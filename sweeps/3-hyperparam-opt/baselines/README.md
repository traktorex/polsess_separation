# Baseline Comparison Experiments

## Purpose

These experiments provide **control baselines** to demonstrate the effectiveness of the multi-stage progressive data-scaling approach used in the main hyperparameter optimization.

**Research Question**: Does progressive data scaling (2K→4K→8K→16K) find better hyperparameters more efficiently than simpler approaches?

---

## Experiments

### Experiment A: One-Stage Baseline (8K samples)

**Config**: [`dprnn_onestage_8k.yaml`](dprnn_onestage_8k.yaml)

**Strategy**: Single-stage sweep with wide search space directly on 8K samples

**Comparison**: vs. Multi-Stage (Stage 1+2+3)
- Same final dataset size (8K)
- Same wide search ranges (Stage 1)
- Similar compute budget (~120 runs)

**Key Difference**: No progressive refinement of search space

**To Run**:
```bash
wandb sweep sweeps/3-hyperparam-opt/baselines/dprnn_onestage_8k.yaml
wandb agent <sweep-id>
```

**Expected Outcome**: 
- Likely finds decent configs but misses optimal regions
- Less efficient than multi-stage (more wasted runs on poor configs)
- Demonstrates value of progressive search space narrowing

---

### Experiment B: Direct Full-Data Sweep (16K samples)

**Config**: [`dprnn_fulldata_16k.yaml`](dprnn_fulldata_16k.yaml)

**Strategy**: Skip progressive scaling entirely, sweep directly on full 16K dataset

**Comparison**: vs. Multi-Stage (all stages)
- Same final dataset (16K)
- Same wide search space (Stage 1)
- Fewer runs (~80) due to computational cost

**Key Difference**: No data-scaling progression

**To Run**:
```bash
wandb sweep sweeps/3-hyperparam-opt/baselines/dprnn_fulldata_16k.yaml
wandb agent <sweep-id>
```

**Expected Outcome**:
- Very expensive per run (~1.5h each)
- Hyperband will aggressively kill runs
- May find good configs but less exploration than multi-stage
- Tests hypothesis: "Can we skip progressive scaling?"

---

## Comparison Matrix

| Approach | Dataset Progression | Search Space | Runs | Total Compute | Best Expected SI-SDR |
|----------|---------------------|--------------|------|---------------|----------------------|
| **Multi-Stage** (Main) | 2K→4K→8K→16K | Wide→Narrow→Refined | 332 (155 finished) | 274h | **4.08 dB** |
| **Experiment A** | 8K only | Wide (constant) | ~120 | ~100-120h | TBD |
| **Experiment B** | 16K only | Wide (constant) | ~80 | ~120-150h | TBD |

---

## Analysis Plan

After running both experiments:

1. **Compute Efficiency**: 
   - Time to find best config
   - Number of runs needed to converge
   - Hyperparameter space coverage

2. **Final Performance**:
   - Best SI-SDR from each approach
   - Mean of top 5 configs
   - Gap vs. multi-stage best (4.08 dB)

3. **Search Quality**:
   - Diversity of top configs
   - Percentage of "wasted" runs (very low performance)
   - Convergence speed

4. **Thesis Contribution**:
   - Quantify benefit of progressive data scaling
   - Demonstrate search space refinement value
   - Cost-benefit analysis for different budgets

---

## Expected Findings

Based on hyperparameter optimization literature:

**Experiment A (One-Stage 8K)**:
- ✅ Pro: Simpler to implement
- ✅ Pro: No risk of subset bias from small data
- ❌ Con: Wastes compute on poor hyperparameter regions
- ❌ Con: No guidance from smaller-scale experiments
- **Prediction**: Finds ~3.5-3.8 dB (vs. 4.08 dB multi-stage)

**Experiment B (Direct 16K)**:
- ✅ Pro: No data-scaling assumptions needed
- ✅ Pro: Directly optimizes on target distribution
- ❌ Con: Extremely expensive per run
- ❌ Con: Hyperband kills runs very aggressively
- ❌ Con: Limited exploration due to budget
- **Prediction**: Finds ~3.3-3.6 dB (less exploration, early termination issues)

**Multi-Stage (Main)**:
- ✅ Pro: Efficient exploration with small data first
- ✅ Pro: Progressive search space refinement
- ✅ Pro: Better hyperparameter space coverage
- ✅ Pro: Insights from each stage inform next
- ❌ Con: More complex to implement
- ⚠️ Con: Risk of subset bias (if small data misleads)
- **Achieved**: **4.08 dB**

---

## Thesis Write-Up Suggestions

### Section: "Hyperparameter Optimization Methodology Comparison"

1. **Introduction**: Explain why hyperparameter optimization is critical
2. **Approaches**: Describe three strategies tested
3. **Results Table**: Compare final performance and efficiency
4. **Analysis**: 
   - Discuss trade-offs
   - Quantify efficiency gains
   - Show convergence plots
5. **Conclusion**: Recommend multi-stage for DPRNN-class models

### Potential Metrics to Report

- **Sample efficiency**: Best SI-SDR per compute hour
- **Convergence speed**: Runs until top-10% performance
- **Robustness**: Std dev of top 5 configs
- **Search coverage**: Unique hyperparameter regions explored

---

## Quick Start

### Option 1: Run Experiment A only
If compute-limited, run Experiment A as the main baseline:
```bash
wandb sweep sweeps/3-hyperparam-opt/baselines/dprnn_onestage_8k.yaml
wandb agent <sweep-id>
# Let run for ~100-120 runs, then analyze
```

### Option 2: Run both experiments
For comprehensive thesis evidence:
```bash
# Start both sweeps in parallel (if you have multiple GPUs)
wandb sweep sweeps/3-hyperparam-opt/baselines/dprnn_onestage_8k.yaml
wandb sweep sweeps/3-hyperparam-opt/baselines/dprnn_fulldata_16k.yaml

# Then launch agents for each sweep
wandb agent <sweep-id-a>  # On GPU 1
wandb agent <sweep-id-b>  # On GPU 2
```

### Option 3: Skip baselines
If compute is very limited and multi-stage results are strong enough, you can:
- Document the baseline experiments as "potential future work"
- Argue theoretically why multi-stage should be better
- Reference existing literature on progressive training

The multi-stage results (4.08 dB, +1.05 dB vs. default) are already strong evidence.

---

## Files Created

- [`dprnn_onestage_8k.yaml`](dprnn_onestage_8k.yaml) - Experiment A sweep config
- [`dprnn_fulldata_16k.yaml`](dprnn_fulldata_16k.yaml) - Experiment B sweep config  
- [`README.md`](README.md) - This documentation
