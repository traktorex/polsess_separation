# Hyperparameter Sweep Guide

How to find the best hyperparameters using W&B Sweeps with Bayesian optimization.

---

## Quick Start

### TL;DR - 3 Commands

**Single Corpus:**
```bash
# 1. Create sweep
wandb sweep experiments/wandb_sweep.yaml

# 2. Run agent (copy command from step 1 output)
wandb agent your-username/polsess-separation/abc123xyz

# 3. Watch results at the URL from step 1
```

**Dual Corpus (Klec et al.):**
```bash
# 1. Create sweep
wandb sweep experiments/klec_sweep.yaml

# 2. Run agent
wandb agent your-username/polsess-separation/abc123xyz
```

---

## Background

Klec et al. (2024) specified some hyperparameters but left others unspecified:

### Fixed (from paper - DON'T change):
- ✅ Input filter length: 20 samples (N=256 at 16kHz)
- ✅ TCN blocks: 32 total (R=4, X=8)
- ✅ Initial LR: 0.001
- ✅ Optimizer: Adam
- ✅ LR scheduler: ×0.95 every 2 epochs without improvement
- ✅ Epochs: 50
- ✅ Loss: SI-SDR

### Variable (NOT specified - CAN sweep):
- ❓ Model capacity (B, H - bottleneck and conv channels)
- ❓ Weight decay
- ❓ Gradient clipping
- ❓ Batch size / accumulation steps

Our sweep optimizes these unspecified parameters!

---

## Single Corpus Sweep

### Configuration File

**File:** `experiments/wandb_sweep.yaml`

**Method:** Bayesian optimization (smarter than grid search)

**Metric:** Validation SI-SDR (maximize)

**Early stopping:** HyperBand (stops bad runs early to save compute)

**Parameters being swept:**

| Parameter | Values | Why Sweep? |
|-----------|--------|------------|
| `model_B` | 128, 256, 512 | Not specified by Klec et al. |
| `model_H` | 256, 512, 1024 | Not specified by Klec et al. |
| `weight_decay` | 0.00001 - 0.001 | Not mentioned in paper |
| `grad_clip_norm` | 1.0, 5.0, 10.0, 20.0 | Not mentioned in paper |
| `batch_size` | 2, 4, 8 | Not specified |
| `accumulation_steps` | 3, 6, 12 | Not mentioned |

### Running the Sweep

#### 1. Login to W&B (one time)

```bash
wandb login
```

#### 2. Create Sweep

```bash
wandb sweep experiments/wandb_sweep.yaml
```

This will output something like:
```
wandb: Created sweep with ID: abc123xyz
wandb: View sweep at: https://wandb.ai/your-username/polsess-separation/sweeps/abc123xyz
wandb: Run sweep agent with: wandb agent your-username/polsess-separation/abc123xyz
```

#### 3. Run Sweep Agent

```bash
# Copy the command from step 2
wandb agent your-username/polsess-separation/abc123xyz
```

The agent will:
1. Pick hyperparameters using Bayesian optimization
2. Train a model with those parameters
3. Run on train dataset
4. Report results to W&B
5. Repeat with smarter parameter choices

#### 4. Run Multiple Agents (Optional - Parallel Training)

Open multiple terminals and run the same agent command:

```bash
# Terminal 1
wandb agent your-username/polsess-separation/abc123xyz

# Terminal 2 (simultaneously)
wandb agent your-username/polsess-separation/abc123xyz

# Terminal 3 (simultaneously)
wandb agent your-username/polsess-separation/abc123xyz
```

Each agent will train different hyperparameter combinations in parallel!

#### 5. Monitor Results

Visit the sweep URL from step 2 to see:
- 📊 **Parallel coordinates:** Hyperparameter relationships
- 🏆 **Best run:** Current winner
- 📈 **Parameter importance:** Which params matter most
- ⚡ **Live updates:** Real-time training progress

---

## Dual Corpus Sweep (Klec et al.)

### Configuration Files

- **`experiments/klec_sweep.yaml`** - Full Bayesian optimization sweep
- **`experiments/klec_sweep_quick.yaml`** - Quick grid search for testing

### Sweep Parameters

#### Full Sweep (`klec_sweep.yaml`)

**Model Capacity:**
- `model_B`: [128, 256, 512] - Bottleneck channels
- `model_H`: [256, 512, 1024] - Conv block channels

**Regularization:**
- `weight_decay`: [0.00001 to 0.001] - Log-uniform
- `grad_clip_norm`: [1.0, 5.0, 10.0]

**Batch Size:**
- `accumulation_steps`: [6, 12, 18, 24]
- Effective batch = 2 × accumulation_steps (batch_size fixed at 2)

**Training:**
- `lr`: 0.001 (fixed for Klec replication)
- `epochs`: 50 (fixed for Klec replication)

**Early Termination:**
- Hyperband algorithm stops poorly performing runs after 10 epochs

#### Quick Sweep (`klec_sweep_quick.yaml`)

**Simplified grid search:**
- Model B: [256, 512]
- Model H: [512, 1024]
- LR: [0.0005, 0.001, 0.002]
- Epochs: 20 (faster testing)

**Total runs:** 2 × 2 × 3 = 12 runs

### Running Dual Corpus Sweep

#### 1. Initialize Sweep

```bash
# For full sweep (Bayesian optimization)
wandb sweep experiments/klec_sweep.yaml

# For quick testing (grid search, shorter training)
wandb sweep experiments/klec_sweep_quick.yaml
```

This will output a sweep ID like: `your-entity/polsess-separation/abc123xyz`

#### 2. Run Sweep Agent

```bash
# Run sweep agent (single GPU)
wandb agent your-entity/polsess-separation/abc123xyz

# Run multiple agents in parallel (if you have multiple GPUs)
# Terminal 1:
wandb agent your-entity/polsess-separation/abc123xyz

# Terminal 2 (different GPU):
CUDA_VISIBLE_DEVICES=1 wandb agent your-entity/polsess-separation/abc123xyz
```

### Key Differences from Single Corpus

1. **Dataset:** Uses both C_in (indoor) and C_out (outdoor) corpora
2. **Batch size:** Fixed at 2 (1 indoor + 1 outdoor per batch)
3. **Base config:** Uses `klec_replication.yaml` by default
4. **Architecture:** kernel_size=20, stride=10 (from base config)

### Dataset Details

- Each sweep run trains on 4000 C_in + 4000 C_out samples (8000 pairs total)
- Validation uses 100 C_in (SER) + 100 C_out (SE) samples
- Best model is saved based on validation SI-SDR
- All runs log to the same W&B project for comparison

### Expected Runtime

**Full sweep:**
- ~50 runs (Bayesian optimization)
- ~2-3 hours per run (50 epochs)
- Total: ~100-150 GPU hours

**Quick sweep:**
- 12 runs (grid search)
- ~45 minutes per run (20 epochs)
- Total: ~9 GPU hours

---

## How Bayesian Optimization Works

Unlike grid search (tries all combinations) or random search, Bayesian optimization:

1. **Starts** with a few random configurations
2. **Builds a model** of which hyperparameters work well
3. **Picks next config** that's likely to improve (exploration vs exploitation)
4. **Updates model** after each result
5. **Converges faster** to good hyperparameters

**Result:** Find optimal params with ~10-20 runs instead of 100+ grid search runs!

---

## How Many Runs?

- **Quick test:** 10-15 runs (~few hours)
- **Good results:** 20-30 runs (~overnight)
- **Exhaustive:** 50+ runs (~1-2 days)

Bayesian optimization is smart - diminishing returns after 30 runs.

---

## Understanding Results

### Parallel Coordinates Plot
Shows how hyperparameters relate to performance. Lines that converge at high SI-SDR show winning combinations.

### Parameter Importance
W&B will rank which parameters matter most:
- High importance: This param significantly affects results
- Low importance: This param doesn't matter much (use default)

### Best Run
W&B shows the best configuration found. Use it for your final training!

---

## What Gets Logged

For each sweep run, W&B tracks:
- All hyperparameters tested
- Training SI-SDR per epoch
- Validation SI-SDR per epoch
- Learning rate changes
- Model size (parameters)
- GPU utilization
- Training duration

---

## After the Sweep

Once you find the best hyperparameters:

### 1. Create final config:

```yaml
# experiments/best_params.yaml
# Best hyperparameters from sweep abc123xyz

data:
  batch_size: <best_value>

model:
  B: <best_value>
  H: <best_value>

training:
  weight_decay: <best_value>
  grad_clip_norm: <best_value>
  accumulation_steps: <best_value>
  num_epochs: 50  # Full training
```

### 2. Train final model:

```bash
python train.py --config experiments/best_params.yaml
```

### 3. Evaluate:

```bash
python evaluate.py --checkpoint checkpoints/model.pt
```

---

## Customizing the Sweep

### Change number of runs:

Add to sweep YAML:
```yaml
run_cap: 30  # Stop after 30 runs
```

### Change search space:

Edit parameter ranges in sweep YAML:
```yaml
parameters:
  weight_decay:
    distribution: log_uniform_values
    min: 0.000001  # Smaller range
    max: 0.0001
```

### Use grid search instead:

Change method in sweep YAML:
```yaml
method: grid  # Try all combinations (slower but exhaustive)
```

### Use random search:

```yaml
method: random  # Random sampling (faster but less smart)
```

---

## Stopping a Sweep

```bash
# Stop agent: Ctrl+C in terminal

# Stop entire sweep (prevents new agents from starting)
wandb sweep --stop <sweep-id>

# Pause sweep
wandb sweep --pause <sweep-id>
```

---

## Tips

1. **Start with quick sweep:** Run 10-15 runs to get intuition
2. **Check early:** Look at results after 5 runs, may already see trends
3. **Run overnight:** Bayesian optimization gets smarter with more data
4. **Use multiple GPUs:** Run agents on different machines
5. **Save best config:** Create a new YAML with winning hyperparameters

---

## Troubleshooting

### "wandb: ERROR Error uploading"
- Check internet connection
- Try: `wandb login --relogin`

### Agent picks same params repeatedly
- Check metric name in sweep config matches logged metric
- Verify metric is being logged: `wandb.log({'val/sisdr': value})`

### Runs failing immediately
- Test script manually first:
  - Single corpus: `python train_sweep.py`
  - Dual corpus: `python train_sweep.py --dual-corpus`
- Check W&B logs for error messages

### Out of memory
- Reduce batch size range in sweep config
- Increase accumulation steps
- Use gradient checkpointing

### "RuntimeError: CUDA out of memory" (dual corpus)
- Reduce `accumulation_steps` range
- Use gradient checkpointing

### "No module named 'train_dual_corpus'" (dual corpus)
- Ensure you're running from the project root directory

---

## References

- [W&B Sweeps Documentation](https://docs.wandb.ai/guides/sweeps)
- [Bayesian Optimization Explained](https://arxiv.org/abs/1807.02811)
- Klec et al. (2024): "Polish emotional speech database with speech separation and enhancement for multimodal systems"
