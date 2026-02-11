# DPRNN Hyperparameter Optimization - Experiment Log

## Overview

**Project**: PolSESS Speech Separation for Polish ASR  
**Task**: SB (Separate Both) - 2-speaker separation  
**Model**: DPRNN (Dual-Path RNN)

---

## Series 1: Baseline Experiments

**Purpose**: Establish reference performance for each model architecture

| Model | Avg SI-SDR | Best Run | Std Dev | Runtime | Links |
|-------|------------|----------|---------|---------|-------|
| **SPMamba** 🏆 | **5.56 dB** | 5.68 dB | 0.12 dB | ~90h | [fast-sweep-1](https://wandb.ai/s17060-polsko-japo-ska-akademia-technik-komputerowych/polsess-thesis-experiments/runs/fast-sweep-1), [20t4fjqw](https://wandb.ai/s17060-polsko-japo-ska-akademia-technik-komputerowych/polsess-thesis-experiments/runs/20t4fjqw), [eky4hhl7](https://wandb.ai/s17060-polsko-japo-ska-akademia-technik-komputerowych/polsess-thesis-experiments/runs/eky4hhl7) |
| **SepFormer** | 5.10 dB | 5.26 dB | 0.19 dB | ~54h | [dbuvdjbs](https://wandb.ai/s17060-polsko-japo-ska-akademia-technik-komputerowych/polsess-thesis-experiments/runs/dbuvdjbs), [lqef1eqq](https://wandb.ai/s17060-polsko-japo-ska-akademia-technik-komputerowych/polsess-thesis-experiments/runs/lqef1eqq), [oc3sfnig](https://wandb.ai/s17060-polsko-japo-ska-akademia-technik-komputerowych/polsess-thesis-experiments/runs/oc3sfnig) (resumed from chckpnt) |
| **DPRNN** | 3.03 dB | 3.20 dB | 0.17 dB | ~11h | [zgmviikh](https://wandb.ai/s17060-polsko-japo-ska-akademia-technik-komputerowych/polsess-thesis-experiments/runs/zgmviikh), [lw6vv676](https://wandb.ai/s17060-polsko-japo-ska-akademia-technik-komputerowych/polsess-thesis-experiments/runs/lw6vv676), [67lul3x0](https://wandb.ai/s17060-polsko-japo-ska-akademia-technik-komputerowych/polsess-thesis-experiments/runs/67lul3x0) |
| **ConvTasNet** | 2.95 dB | 3.28 dB | 0.29 dB | ~32h | [g3ldnlbd](https://wandb.ai/s17060-polsko-japo-ska-akademia-technik-komputerowych/polsess-thesis-experiments/runs/g3ldnlbd), [z14ednx3](https://wandb.ai/s17060-polsko-japo-ska-akademia-technik-komputerowych/polsess-thesis-experiments/runs/z14ednx3), [g3ldnlbd](https://wandb.ai/s17060-polsko-japo-ska-akademia-technik-komputerowych/polsess-thesis-experiments/runs/g3ldnlbd) |

**Key Finding**: SPMamba outperforms all models, but DPRNN selected for hyperparameter optimization due to computational efficiency and thesis scope.

---

## Series 2: Model Comparison

| Rank | Model | Avg SI-SDR | Improvement vs ConvTasNet |
|------|-------|------------|---------------------------|
| 🥇 | SPMamba | 5.56 dB | +2.61 dB (+88%) |
| 🥈 | SepFormer | 5.10 dB | +2.15 dB (+73%) |
| 🥉 | DPRNN | 3.03 dB | +0.08 dB (+3%) |
| 4️⃣ | ConvTasNet | 2.95 dB | Baseline |

---

## Series 3: Hyperparameter Optimization (DPRNN)

**Goal**: Optimize DPRNN hyperparameters to maximize SI-SDR performance

**Baseline Performance**: 3.03 dB (default hyperparameters)

### Three Optimization Strategies Compared

| Approach | Dataset Progression | Compute | Best SI-SDR | Status |
|----------|---------------------|---------|-------------|--------|
| **3-Stage** | 2K→4K→8K→16K val | 322h | **4.67 dB** ✅ | Complete |
| **Exp A** | 8K→16K val | ~105h | **TBD** 🔄 | Validating |
| **Exp B** | 16K proxy→8K LR sweep→16K val | ~123h | **TBD** 🔄 | In progress |

---

### Approach 1: Multi-Stage Progressive Scaling (Main)

**Strategy**: Progressive data scaling with search space refinement

**WandB Project**: [polsess-thesis-experiments](https://wandb.ai/s17060-polsko-japo-ska-akademia-technik-komputerowych/polsess-thesis-experiments)

#### Stage 1: Wide Search (2K samples)

**Config**: [`dprnn.yaml`](sweeps/3-hyperparam-opt/stage1/dprnn.yaml) | **Sweep**: [ocjl0lhr](https://wandb.ai/s17060-polsko-japo-ska-akademia-technik-komputerowych/polsess-thesis-experiments/sweeps/ocjl0lhr)

| Metric | Value |
|--------|-------|
| Runs | 91 (42 finished) |
| Best SI-SDR | **1.75 dB** |
| Runtime | 19.8h |
| Early Termination | Hyperband (s=2, min_iter=10) |

**Search Space**:
- LR: [3e-4, 3e-3]
- Weight Decay: [1e-6, 1e-4]
- Grad Clip: [0.5, 20.0]
- LR Factor: [0.3, 0.95]
- LR Patience: [1, 5]

**Outcome**: Identified higher LR (~1e-3) and very low weight decay as promising

---

#### Stage 2: Narrowed Search (4K samples)

**Config**: [`dprnn.yaml`](sweeps/3-hyperparam-opt/stage2/dprnn.yaml) | **Sweep**: [va7wk46n](https://wandb.ai/s17060-polsko-japo-ska-akademia-technik-komputerowych/polsess-thesis-experiments/sweeps/va7wk46n)

| Metric | Value |
|--------|-------|
| Runs | 131 (48 finished) |
| Best SI-SDR | **3.06 dB** (+1.31 dB vs Stage 1) |
| Runtime | 48.7h |

**Refined Search Space** (based on Stage 1):
- LR: [5e-4, 2e-3] (narrowed)
- Weight Decay: [1e-6, 5e-5] (much lower max)
- Grad Clip: [0.5, 20.0] (kept wide)
- LR Factor: [0.38, 0.95] (slight adjustment)
- LR Patience: [1, 5] (unchanged)

**Key Finding**: Weight decay must be very low (<5e-5)

---

#### Stage 3: Final Refinement (8K samples)

Two parallel strategies:

**3a - Hyperband**: [hj7sbz6c](https://wandb.ai/s17060-polsko-japo-ska-akademia-technik-komputerowych/polsess-thesis-experiments/sweeps/hj7sbz6c) | 69 runs (25 finished) | **4.08 dB** best | 47.6h |
**Config**: [`sweeps/3-hyperparam-opt/stage3/dprnn.yaml`](sweeps/3-hyperparam-opt/stage3/dprnn.yaml)  

**3b - Conservative**: [1wtvbmiu](https://wandb.ai/s17060-polsko-japo-ska-akademia-technik-komputerowych/polsess-thesis-experiments/sweeps/1wtvbmiu) | 41 runs (40 finished) | 3.74 dB best | 57.1h |
**Config**: [`sweeps/3-hyperparam-opt/stage3/dprnn_conservative.yaml`](sweeps/3-hyperparam-opt/stage3/dprnn_conservative.yaml) 

**Stage 3 Search Space** (further narrowed):
- LR: [7e-4, 2e-3]
- Weight Decay: [1e-6, 5e-5]
- Grad Clip: [0.5, 20.0]
- LR Factor: [0.38, 0.95]
- LR Patience: [2, 5]

**Winner**: Hyperband found best config (4.08 dB)

---

#### Final Validation (16K samples, 3 seeds)

**Top 5 configs** selected from Stage 3 based on performance and diversity.
**Configs**: (experiments/dprnn/3-hyperparamopt-3stage-vals) 

| Rank | Strategy | Config | Mean SI-SDR | Std | 8K→16K Gain | Individual Results | Links |
|------|----------|--------|-------------|-----|-------------|-------------------|-------|
| **🥇** | **Hyperband** | **fancy-sweep-62** | **4.67 dB** | **0.07** | **+0.59 dB** | 4.65, 4.62, 4.75 dB | [722e8mux](https://wandb.ai/s17060-polsko-japo-ska-akademia-technik-komputerowych/polsess-separation/runs/722e8mux), [8vda24q0](https://wandb.ai/s17060-polsko-japo-ska-akademia-technik-komputerowych/polsess-separation/runs/8vda24q0), [dzfpezdn](https://wandb.ai/s17060-polsko-japo-ska-akademia-technik-komputerowych/polsess-separation/runs/dzfpezdn) |
| 🥈 | Hyperband | rose-sweep-41 | 4.29 dB | 0.07 | +0.45 dB | 4.21, 4.31, 4.34 dB | [2pfhn3vw](https://wandb.ai/s17060-polsko-japo-ska-akademia-technik-komputerowych/polsess-separation/runs/2pfhn3vw), [qz2y4s67](https://wandb.ai/s17060-polsko-japo-ska-akademia-technik-komputerowych/polsess-separation/runs/qz2y4s67), [v9k57wnb](https://wandb.ai/s17060-polsko-japo-ska-akademia-technik-komputerowych/polsess-separation/runs/v9k57wnb) |
| 🥉 | Hyperband | spring-sweep-67 | 4.28 dB | 0.12 | +0.50 dB | 4.24, 4.42, 4.19 dB | [d5puian1](https://wandb.ai/s17060-polsko-japo-ska-akademia-technik-komputerowych/polsess-separation/runs/d5puian1), [hjpbov0z](https://wandb.ai/s17060-polsko-japo-ska-akademia-technik-komputerowych/polsess-separation/runs/hjpbov0z), [3efoaqli](https://wandb.ai/s17060-polsko-japo-ska-akademia-technik-komputerowych/polsess-separation/runs/3efoaqli) |
| 4 | Conservative | exalted-sweep-12 | 4.24 dB | 0.13 | +0.50 dB | 4.09, 4.35, 4.27 dB | [ka2iinu4](https://wandb.ai/s17060-polsko-japo-ska-akademia-technik-komputerowych/polsess-separation/runs/ka2iinu4), [aigaiez1](https://wandb.ai/s17060-polsko-japo-ska-akademia-technik-komputerowych/polsess-separation/runs/aigaiez1), [bbc7repc](https://wandb.ai/s17060-polsko-japo-ska-akademia-technik-komputerowych/polsess-separation/runs/bbc7repc) |
| 5 | Conservative | sunny-sweep-2 | 4.17 dB | 0.10 | +0.45 dB | 4.06, 4.25, 4.21 dB | [htyogmgo](https://wandb.ai/s17060-polsko-japo-ska-akademia-technik-komputerowych/polsess-separation/runs/htyogmgo), [4z7p2xjd](https://wandb.ai/s17060-polsko-japo-ska-akademia-technik-komputerowych/polsess-separation/runs/4z7p2xjd), [8vcr3smx](https://wandb.ai/s17060-polsko-japo-ska-akademia-technik-komputerowych/polsess-separation/runs/8vcr3smx)  |

**Winner**: fancy-sweep-62

**Final Hyperparameters**:
- LR: 0.00125 | WD: 2.1e-5 | Grad Clip: 2.76 | LR Factor: 0.863 | LR Patience: 3

**Improvement**: **+1.64 dB** vs baseline (3.03 → 4.67 dB = 54% gain)

---

### Approach 2: One-Stage Baseline (Experiment A)

**Strategy**: Single wide search on 8K samples (no progressive scaling)

**Config**: [`dprnn_onestage_8k.yaml`](sweeps/3-hyperparam-opt/baselines/dprnn_onestage_8k.yaml)  
**Sweep**: [zp95xdye](https://wandb.ai/s17060-polsko-japo-ska-akademia-technik-komputerowych/polsess-thesis-experiments/sweeps/zp95xdye)

| Metric | Value |
|--------|-------|
| Runs | 130 finished |
| Runtime | ~105h |
| Best SI-SDR (8K) | **3.88 dB** |
| Early Termination | Hyperband (s=2, eta=3) |

**Search Space**: Same wide ranges as 3-Stage Stage 1

#### Top 3 Configs Selected for Validation

| Rank | Config | 8K SI-SDR | Validation Status | 16K Mean SI-SDR | Individual Results | Links |
|------|--------|-----------|-------------------|-----------------|-------|-------|
| 1 | lively-sweep-34 | 3.88 dB | ✅ Complete   | 4.22 dB | 4.17, 4.40, 4.11 dB | [ppl269tg](https://wandb.ai/s17060-polsko-japo-ska-akademia-technik-komputerowych/polsess-separation/runs/ppl269tg), [1havizhg](https://wandb.ai/s17060-polsko-japo-ska-akademia-technik-komputerowych/polsess-separation/runs/1havizhg), [1jlxplbd](https://wandb.ai/s17060-polsko-japo-ska-akademia-technik-komputerowych/polsess-separation/runs/1jlxplbd) |
| 2 | glowing-sweep-124 | 3.87 dB | ✅ Complete | 4.37 dB | 4.40, 4.30, 4.41 dB | [ym3u4pm8](https://wandb.ai/s17060-polsko-japo-ska-akademia-technik-komputerowych/polsess-separation/runs/ym3u4pm8), [x620gqb7](https://wandb.ai/s17060-polsko-japo-ska-akademia-technik-komputerowych/polsess-separation/runs/x620gqb7), [bzwnzgi9](https://wandb.ai/s17060-polsko-japo-ska-akademia-technik-komputerowych/polsess-separation/runs/bzwnzgi9) |
| 3 | ruby-sweep-116 | 3.86 dB | ✅ Complete | 4.24 dB | 4.47, 4.10, 4.14 dB | [i5i35bwg](https://wandb.ai/s17060-polsko-japo-ska-akademia-technik-komputerowych/polsess-separation/runs/i5i35bwg), [7mgb85uj](https://wandb.ai/s17060-polsko-japo-ska-akademia-technik-komputerowych/polsess-separation/runs/7mgb85uj), [wn99baaj](https://wandb.ai/s17060-polsko-japo-ska-akademia-technik-komputerowych/polsess-separation/runs/wn99baaj) |


**Comparison vs 3-Stage**: Tests whether progressive scaling is necessary or if one-stage on medium data (8K) is sufficient

---

### Approach 3: Proxy-Based Full-Data (Experiment B)

**Strategy**: Quick proxy evaluation (20 epochs) on full data (16K), then LR scheduler optimization on 8K, then validation on 16K

**Phase 1 - Proxy Sweep** (20 epochs, 16K samples)

**Config**: [`dprnn_fulldata_16k_proxy.yaml`](sweeps/3-hyperparam-opt/baselines/dprnn_fulldata_16k_proxy.yaml)  
**Sweep**: [igozsq0r](https://wandb.ai/s17060-polsko-japo-ska-akademia-technik-komputerowych/polsess-thesis-experiments/sweeps/igozsq0r)

| Metric | Value |
|--------|-------|
| Runs | ~70 finished |
| Runtime | ~40h |
| Search Params | LR, weight_decay, grad_clip_norm (NO lr_factor/lr_patience) |
| Early Termination | Hyperband (min_iter=6, s=2, eta=3) |

**Top 3 Configs**:
1. kind-sweep-68: **3.41 dB**
2. wise-sweep-64: **3.36 dB**
3. prime-sweep-35: **3.34 dB**


**Rationale**: 20 epochs too short for LR scheduler to show effect; tune LR/WD/GC first, then optimize LR scheduler separately

---

**Phase 2 - LR Scheduler Grid Search** (80 epochs, 8K samples)

For each top-3 config, run grid search over LR scheduler params:
- **lr_factor**: [0.40, 0.55, 0.70, 0.85] (4 values)
- **lr_patience**: [2, 3, 4, 5] (4 values)
- **Total**: 16 runs per config × 3 configs = 48 runs


**kind-sweep-68 LR sweep**: [xfcscfml](https://wandb.ai/s17060-polsko-japo-ska-akademia-technik-komputerowych/polsess-thesis-experiments/sweeps/xfcscfml) ✅ Complete
- Best [lr_factor, lr patience]: [0.55, 5]

**wise-sweep-64 LR sweep**: [5ojzuq7z](https://wandb.ai/s17060-polsko-japo-ska-akademia-technik-komputerowych/polsess-thesis-experiments/sweeps/5ojzuq7z) 🔄 Running

**prime-sweep-35 LR sweep**: ⏳ Pending

**Phase 3 - Final Validation**: Best LR params from each config × 3 seeds × 80 epochs on 16K

| Rank | Config | 2K SI-SDR | Validation Status | 16K Mean SI-SDR | Individual Results | Links |
|------|--------|-----------|-------------------|-----------------|-------|-------|
| 1 | kind-sweep-68 | 3.41 dB | ✅ Complete   | 4.42 dB | 4.35, 4.69, 4.22 dB | [ff2i3l0v](https://wandb.ai/s17060-polsko-japo-ska-akademia-technik-komputerowych/polsess-separation/runs/ff2i3l0v), [bk0dmhqj](https://wandb.ai/s17060-polsko-japo-ska-akademia-technik-komputerowych/polsess-separation/runs/bk0dmhqj), [wmvnhx84](https://wandb.ai/s17060-polsko-japo-ska-akademia-technik-komputerowych/polsess-separation/runs/wmvnhx84) |
| 2 | wise-sweep-64 | 3.36 dB | ✅ Complete | 4.16 dB | 4.17, 4.19, 4.11 dB | [l2xzaoon](https://wandb.ai/s17060-polsko-japo-ska-akademia-technik-komputerowych/polsess-separation/runs/l2xzaoon), [1bcg8abx](https://wandb.ai/s17060-polsko-japo-ska-akademia-technik-komputerowych/polsess-separation/runs/1bcg8abx), [305dk3qx](https://wandb.ai/s17060-polsko-japo-ska-akademia-technik-komputerowych/polsess-separation/runs/305dk3qx) |
| 3 | prime-sweep-35 | 3.34 dB | ✅ Complete | 3.96 dB | 4.00, 3.91, 3.96 dB | [xoaottd1](https://wandb.ai/s17060-polsko-japo-ska-akademia-technik-komputerowych/polsess-separation/runs/xoaottd1), [oanobnwm](https://wandb.ai/s17060-polsko-japo-ska-akademia-technik-komputerowych/polsess-separation/runs/oanobnwm), [w5pcz9oe](https://wandb.ai/s17060-polsko-japo-ska-akademia-technik-komputerowych/polsess-separation/runs/w5pcz9oe) |

**Comparison vs 3-Stage**: Tests whether epoch-based proxy can replace data-based progressive scaling

---

## Summary: Optimization Strategies

| Approach | Key Idea | Pros | Cons | Status |
|----------|----------|------|------|--------|
| **3-Stage** | Progressive data scaling | Efficient search space refinement, excellent results | More complex | ✅ **4.67 dB** |
| **Exp A** | One-stage wide search | Simpler, tests single-stage viability | No guidance from smaller data | 🔄 Validating |
| **Exp B** | Epoch proxy + LR sweep | Tests proxy hypothesis | Two-phase optimization | 🔄 In progress |

---

## Key Hyperparameter Insights

From 330+ runs across all 3-stage sweeps:

1. **Learning Rate**: Optimal **1e-3 to 1.5e-3** (correlation: +0.12)
2. **Weight Decay**: Keep **very low** 1e-6 to 5e-5 (correlation: **-0.27** - strongest predictor)
3. **Gradient Clipping**: Wide range works 0.9-14.0 (correlation: -0.01, minimal impact)
4. **LR Factor**: Gentler decay 0.54-0.86 preferred (correlation: +0.23)
5. **LR Patience**: Mixed results 2-5 epochs

**Critical Insight**: Weight decay is the most important hyperparameter for DPRNN; high values significantly hurt performance.

---

## Final Results

**3-Stage Winner**: **4.67 dB** (fancy-sweep-62)  
**Improvement over baseline**: **+1.64 dB** (54% gain)  
**Total improvement from Stage 1**: **+2.92 dB** (167% gain)

**Experiment A**: Results pending (validation in progress)  
**Experiment B**: Results pending (LR sweeps in progress)

---

## Tools & Scripts

- **Analysis**: [`analyze_results.py`](sweeps/3-hyperparam-opt/stage3/analyze_results.py)
- **Validation scripts**: [`run_validation.sh`](experiments/dprnn/3-hyperparamopt-3stage-vals/run_validation.sh), [`run_validation.sh`](experiments/dprnn/3-hyperparamopt-expA-vals/run_validation.sh)
- **Baseline configs**: [`baselines/`](sweeps/3-hyperparam-opt/baselines/)
- **Config generator**: [`generate_lr_sweeps.py`](sweeps/3-hyperparam-opt/baselines/generate_lr_sweeps.py)

---

## Notes

- All experiments use curriculum learning with ["SER", "SE"] validation
- DPRNN architecture: N=64, kernel=16, stride=8, 6 layers, chunk=100, LSTM hidden=128
- Hyperparameter optimization: Bayesian search with Hyperband early termination
- Final comparison will be completed once Experiments A and B finish validation
