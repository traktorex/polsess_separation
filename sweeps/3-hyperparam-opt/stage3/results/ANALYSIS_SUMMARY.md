# DPRNN Stage 3 Analysis Summary

## Overview
Combined analysis of 69 finished runs from hyperband sweep and 40 finished runs from conservative sweep.

## Top 5 Selected Configurations

### Selection Criteria
1. **Primary**: Validation SI-SDR performance
2. **Secondary**: Hyperparameter diversity (to avoid redundant configs)
3. **Balance**: Mix of both sweep types

### Selected Configurations

#### 🥇 Config 1: fancy-sweep-62 (Hyperband)
- **Validation SI-SDR**: 4.08 dB
- **Performance**: Best overall config
- **Hyperparameters**:
  - LR: 0.001253
  - Weight Decay: 2.11e-05
  - Grad Clip: 2.76
  - LR Factor: 0.8633
  - LR Patience: 3
- **Notes**: Trained for full 80 epochs, no early stopping

#### 🥈 Config 2: rose-sweep-41 (Hyperband)
- **Validation SI-SDR**: 3.84 dB
- **Hyperparameters**:
  - LR: 0.001497
  - Weight Decay: 4.41e-05
  - Grad Clip: 2.29
  - LR Factor: 0.7994
  - LR Patience: 5
- **Notes**: Completed 80 epochs, higher LR patience

#### 🥉 Config 3: spring-sweep-67 (Hyperband)
- **Validation SI-SDR**: 3.78 dB
- **Hyperparameters**:
  - LR: 0.001135
  - Weight Decay: 2.07e-06 (very low!)
  - Grad Clip: 13.86 (very high!)
  - LR Factor: 0.5422 (aggressive decay)
  - LR Patience: 5
- **Notes**: Most diverse config - explores very different hyperparameter region

#### Config 4: exalted-sweep-12 (Conservative)
- **Validation SI-SDR**: 3.74 dB
- **Hyperparameters**:
  - LR: 0.000847 (lower)
  - Weight Decay: 4.75e-05
  - Grad Clip: 3.80
  - LR Factor: 0.6949
  - LR Patience: 3
- **Notes**: From conservative sweep, early stopped at epoch 79

#### Config 5: sunny-sweep-2 (Conservative)
- **Validation SI-SDR**: 3.72 dB
- **Hyperparameters**:
  - LR: 0.000817
  - Weight Decay: 4.85e-05
  - Grad Clip: 2.58
  - LR Factor: 0.7675
  - LR Patience: 2 (most aggressive)
- **Notes**: Early stopped at epoch 69

## Key Insights

### Performance Distribution
- **Best overall**: 4.08 dB (fancy-sweep-62)
- **Mean**: 3.46 dB
- **Std**: 0.41 dB
- **Range**: 2.86 - 4.08 dB

### Sweep Type Comparison
- **Hyperband**: Best 4.08 dB, Mean 3.44 dB
- **Conservative**: Best 3.74 dB, Mean 3.49 dB
- **Winner**: Hyperband produced the absolute best config, but conservative had slightly higher mean

### Hyperparameter Patterns (Top 10 configs)
1. **Learning Rate**: 0.0008 - 0.0020 (most important)
2. **Weight Decay**: 2e-6 to 5e-5 (very small values work best)
3. **Grad Clip**: Wide range (0.9 - 14.0) - not critical
4. **LR Factor**: 0.38 - 0.90 (moderate decay preferred)
5. **LR Patience**: Mixed (2-5), no clear winner

### Correlations with Performance
- **Positive**: LR (0.12), LR Factor (0.23)
- **Negative**: Weight Decay (-0.27), Grad Clip (-0.01)
- **Interpretation**: Higher LR and gentler decay helps, but keep weight decay very low

## Next Steps: Final Validation

### Plan
Run 15 total experiments (5 configs × 3 seeds):
- Seeds: 42, 123, 456
- Dataset: Full 16,000 training samples
- Epochs: 80
- Early stopping patience: 15

### Expected Outcomes
- **Best case**: Config 1 maintains ~4.1 dB on full dataset
- **Realistic**: 3.8-4.0 dB range for top configs
- **Goal**: Select config with highest **mean** across 3 seeds

### Files Generated
1. `results/top5_configs_for_validation.csv` - Quick reference table
2. `results/validation_configs.txt` - Ready-to-use training commands
3. `results/analysis_plots.png` - Visualizations (4 plots)

## Run Commands for Validation

First, create `experiments/dprnn/dprnn_16000.yaml` with 16,000 training samples, then:

```bash
# Config 1 - fancy-sweep-62
python train.py --config experiments/dprnn/dprnn_16000.yaml --lr 0.00125331 --weight-decay 2.11e-05 --grad-clip-norm 2.76 --lr-factor 0.8633 --lr-patience 3 --seed 42
python train.py --config experiments/dprnn/dprnn_16000.yaml --lr 0.00125331 --weight-decay 2.11e-05 --grad-clip-norm 2.76 --lr-factor 0.8633 --lr-patience 3 --seed 123
python train.py --config experiments/dprnn/dprnn_16000.yaml --lr 0.00125331 --weight-decay 2.11e-05 --grad-clip-norm 2.76 --lr-factor 0.8633 --lr-patience 3 --seed 456

# Repeat for configs 2-5...
```

See `results/validation_configs.txt` for all commands.
