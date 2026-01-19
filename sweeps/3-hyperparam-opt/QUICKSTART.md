# Series 3: Hyperparameter Optimization - Quick Start Guide

## ✅ What's Ready

4 W&B sweep configurations in `sweeps/3-hyperparam-opt/`:
- ✅ `convtasnet.yaml` (20 runs)
- ✅ `dprnn.yaml` (25 runs)
- ✅ `sepformer.yaml` (30 runs)
- ✅ `spmamba.yaml` (35 runs)

**Total**: 110 runs optimizing `lr`, `weight_decay`, `grad_clip_norm`, `lr_factor`

---

## 🚀 How to Launch

### Option 1: Start with SPMamba (Best Model)
```bash
cd /home/user/polsess_separation

# Create sweep
wandb sweep sweeps/3-hyperparam-opt/spmamba.yaml

# Start agent (copy sweep ID from output)
wandb agent s17060-polsko-japo-ska-akademia-technik-komputerowych/polsess-thesis-experiments/<sweep-id>
```

### Option 2: Run All in Parallel (Fastest)
Open 4 terminals and run each sweep simultaneously:
```bash
# Terminal 1
wandb sweep sweeps/3-hyperparam-opt/convtasnet.yaml
wandb agent <sweep-id-1>

# Terminal 2  
wandb sweep sweeps/3-hyperparam-opt/dprnn.yaml
wandb agent <sweep-id-2>

# Terminal 3
wandb sweep sweeps/3-hyperparam-opt/sepformer.yaml
wandb agent <sweep-id-3>

# Terminal 4
wandb sweep sweeps/3-hyperparam-opt/spmamba.yaml
wandb agent <sweep-id-4>
```

### Option 3: Sequential (Simplest)
Run one at a time, starting with highest priority:
1. SPMamba (35 runs, ~36 days)
2. SepFormer (30 runs, ~15 days)  
3. DPRNN (25 runs, ~2.6 days)
4. ConvTasNet (20 runs, ~7.5 days)

---

## 📊 Monitoring

1. **W&B Dashboard**: View parallel coordinate plots showing hyperparameter relationships
2. **Early Stopping**: Hyperband will automatically kill poor runs
3. **Best So Far**: W&B highlights the current best configuration

**Check**: `best_val_sisdr` metric to compare against baselines

---

## 🎯 Expected Outcomes

| Model | Baseline | Target | Expected Gain |
|-------|----------|--------|---------------|
| SPMamba | 5.56 dB | 5.8-6.0 dB | +0.24-0.44 dB |
| SepFormer | 5.10 dB | 5.4-5.6 dB | +0.30-0.50 dB |
| DPRNN | 3.03 dB | 3.3-3.5 dB | +0.27-0.47 dB |
| ConvTasNet | 2.95 dB | 3.2-3.4 dB | +0.25-0.45 dB |

---

## ✅ After Finding Best Hyperparameters

For each model:
1. **Validate**: Re-run best config with 3 seeds [42, 123, 456]
2. **Compare**: Calculate improvement vs baseline
3. **Document**: Update `EXPERIMENT_LOG.md`
4. **(Optional) Architecture Search**: Test larger model variants with optimized training params

---

## 💡 Tips

- **Start small**: Launch SPMamba first to validate setup
- **Monitor memory**: SPMamba uses ~11.3GB, others use less
- **Early stop is your friend**: Don't waste time on poor configs
- **W&B parallel plots**: Great for understanding hyperparameter interactions

---

## 🚨 Troubleshooting

**If OOM errors**: Early stopping or reduce batch size in base config  
**If NaNs appear**: Bayesian search should avoid high LR automatically  
**If slow progress**: Check if early termination is working

---

Ready to launch sweeps! 🚀
