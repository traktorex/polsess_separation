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

**Two Variants Available**:

#### B1: Proxy-Based (Recommended)

**Config**: [`dprnn_fulldata_16k_proxy.yaml`](dprnn_fulldata_16k_proxy.yaml)

**Strategy**: Quick proxy evaluation with short epochs, then validate top configs

**Setup**:
- 60-80 runs with **20 epochs each** (proxy evaluation)
- Same wide search space as Stage 1
- After sweep completes, select top 5 configs
- Validate those 5 configs with full 80 epochs × 3 seeds

**Comparison**: vs. Multi-Stage (all stages)
- Same final dataset (16K)
- Uses epochs as proxy rather than data size
- Much faster per run (~40min vs ~2.5h)
- Tests hypothesis: "Can short runs on full data replace progressive scaling?"

**To Run**:
```bash
wandb sweep sweeps/3-hyperparam-opt/baselines/dprnn_fulldata_16k_proxy.yaml
wandb agent <sweep-id>
```

**Expected Outcome**:
- Proxy run cost: ~40-53 hours (60-80 runs × 40min)
- Validation cost: ~11 hours (5 configs × 3 seeds × 45min)
- **Total: ~51-64 hours**
- Likely finds decent configs but less exploration than multi-stage
- May miss configs that need more epochs to show promise

---

#### B2: Full Epochs (Original - Very Expensive)

**Config**: [`dprnn_fulldata_16k.yaml`](dprnn_fulldata_16k.yaml)

**Strategy**: Direct sweep with full 80 epochs per run

**Cost**: ~120-160 hours (60-80 runs × 2.5h each)

**Note**: **Not recommended** unless you have massive compute budget. Use B1 (proxy) instead for realistic comparison.

---

## Comparison Matrix

| Approach | Dataset Progression | Epochs/Proxy | Runs | Total Compute | Best Expected SI-SDR |
|----------|---------------------|--------------|------|---------------|----------------------|
| **Multi-Stage** (Main) | 2K→4K→8K→16K | Full (varied) | 347 (170 finished) | 322h | **4.67 dB** ✅ |
| **Experiment A** | 8K only | Full (80) | ~120 | ~100-120h | TBD 🔄 |
| **Experiment B (Proxy)** | 16K only | **20 epochs** | ~70 | ~51-64h | TBD |
| Experiment B (Full) | 16K only | Full (80) | ~80 | ~120-160h | ❌ Not recommended |

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
   - Gap vs. multi-stage best (4.67 dB)

3. **Search Quality**:
   - Diversity of top configs
   - Percentage of "wasted" runs (very low performance)
   - Convergence speed

4. **Thesis Contribution**:
   - Quantify benefit of progressive data scaling
   - Demonstrate search space refinement value
   - Cost-benefit analysis for different budgets
   - Compare proxy approaches (data vs epochs)

---

## Expected Findings

Based on hyperparameter optimization literature:

**Experiment A (One-Stage 8K)**:
- ✅ Pro: Simpler to implement
- ✅ Pro: No risk of subset bias from small data
- ❌ Con: Wastes compute on poor hyperparameter regions
- ❌ Con: No guidance from smaller-scale experiments
- **Prediction**: Finds ~3.5-3.8 dB (vs. 4.67 dB multi-stage)

**Experiment B (Proxy-Based 16K, 20 epochs)**:
- ✅ Pro: Tests on target data distribution
- ✅ Pro: More efficient than full-epoch sweep
- ✅ Pro: 20 epochs may be enough to rank configs
- ❌ Con: Short epochs may not reveal long-term performance
- ❌ Con: May miss configs that improve after epoch 20
- ❌ Con: Still expensive compared to multi-stage early stages
- **Prediction**: Finds ~3.8-4.1 dB (better than Exp A, but still below multi-stage due to limited epochs)

**Multi-Stage (Main)**:
- ✅ Pro: Efficient exploration with small data first
- ✅ Pro: Progressive search space refinement
- ✅ Pro: Better hyperparameter space coverage
- ✅ Pro: Insights from each stage inform next
- ✅ Pro: Full epochs at final stage
- ❌ Con: More complex to implement
- ⚠️ Con: Risk of subset bias (if small data misleads)
- **Achieved**: **4.67 dB** 🏆

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
