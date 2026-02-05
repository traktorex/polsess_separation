# DPRNN Final Validation - Quick Start Guide

## ✅ Setup Complete

All 5 validation config files and run script are ready!

## 🚀 Running All 15 Experiments

Simply execute:

```bash
./run_validation.sh
```

This will run **15 experiments sequentially** (5 configs × 3 seeds: 42, 123, 456).

## 📝 Configuration Files

| Config | Name | Expected SI-SDR | Key Hyperparameters |
|--------|------|-----------------|---------------------|
| 1 | fancy-sweep-62 | **4.08 dB** | LR: 0.00125, WD: 2.1e-5, GC: 2.76 |
| 2 | rose-sweep-41 | 3.84 dB | LR: 0.00150, WD: 4.4e-5, GC: 2.29 |
| 3 | spring-sweep-67 | 3.78 dB | LR: 0.00114, WD: 2.1e-6, GC: 13.86 (diverse) |
| 4 | exalted-sweep-12 | 3.74 dB | LR: 0.00085, WD: 4.7e-5, GC: 3.80 |
| 5 | sunny-sweep-2 | 3.72 dB | LR: 0.00082, WD: 4.9e-5, GC: 2.58 |

## 🎯 Running Individual Configs

To run a specific config with a specific seed:

```bash
# Config 1, seed 42
python train.py --config experiments/dprnn/validation_config1.yaml --seed 42

# Config 1, seed 123
python train.py --config experiments/dprnn/validation_config1.yaml --seed 123

# Config 2, seed 42
python train.py --config experiments/dprnn/validation_config2.yaml --seed 42
```

## ⏱️ Expected Runtime

- **Per run**: ~1.5 hours (80 epochs on 16K samples)
- **Total**: ~22-23 hours for all 15 runs

## 📊 After Completion

1. Check WandB project: `polsess-separation`
2. Group runs by config (run tags/groups)
3. For each config, calculate **mean SI-SDR across 3 seeds**
4. Select config with **highest mean** as final hyperparameters
5. Document best config in thesis

## 🔧 CLI Seed Override

The `--seed` argument was added to `config.py`:
- Overrides the seed specified in YAML config
- Allows running same config with different seeds
- No need to create temporary config files

## 📁 Files Created

- `experiments/dprnn/validation_config1.yaml` through `validation_config5.yaml`
- `run_validation.sh` - automated runner script
- `experiments/dprnn/dprnn_16000.yaml` - base config (not used directly)

## 🎓 Next Steps

After validation completes:
1. Analyze results in WandB
2. Select best config
3. (Optional) Run Stage 3 sweeps for other models (ConvTasNet, SepFormer)
4. Compare multi-stage optimization vs. simple baselines for thesis
