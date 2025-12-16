# Weights & Biases (W&B) Integration Guide

How W&B handles hyperparameters and integrates with your configuration system.

---

## How W&B Handles Hyperparameters

### TL;DR
**W&B is config-agnostic** - it logs whatever you pass to `wandb.init(config=...)`. It works with:
- ✅ CLI arguments
- ✅ YAML files
- ✅ Python dictionaries
- ✅ Dataclass objects
- ✅ Argparse namespaces

**You choose the source, W&B just logs it.**

---

## Integration Patterns

### Pattern 1: Log CLI Arguments (Simplest)

```python
# train.py
import wandb
from config import get_config_from_args

def main():
    # Get config from CLI/YAML/env vars (your existing system)
    config = get_config_from_args()

    # Initialize W&B with the config
    wandb.init(
        project='polsess-separation',
        name='experiment-1',
        config={
            # Log all hyperparameters
            'data_root': config.data.data_root,
            'batch_size': config.data.batch_size,
            'task': config.data.task,
            'lr': config.training.lr,
            'epochs': config.training.num_epochs,
            'model_N': config.model.N,
            'model_B': config.model.B,
            # ... etc
        }
    )

    # Train model
    trainer.train()
```

**W&B Dashboard shows:**
```
Hyperparameters:
  batch_size: 4
  lr: 0.001
  task: ES
  epochs: 10
  model_N: 256
```

### Pattern 2: Log Entire Config Object (Better)

W&B can automatically convert dataclasses to dictionaries:

```python
# train.py
import wandb
from dataclasses import asdict
from config import get_config_from_args

def main():
    config = get_config_from_args()

    # Convert config to dict and log it
    wandb.init(
        project='polsess-separation',
        config={
            'data': asdict(config.data),
            'model': asdict(config.model),
            'training': asdict(config.training)
        }
    )
```

**W&B Dashboard shows (nested):**
```
Hyperparameters:
  data:
    batch_size: 4
    task: ES
    data_root: /path/to/data
  model:
    N: 256
    B: 256
    H: 512
  training:
    lr: 0.001
    epochs: 10
    use_amp: true
```

### Pattern 3: Use W&B's Config Object (Most Flexible)

W&B has its own config object you can use:

```python
import wandb

wandb.init(project='polsess-separation')

# Access config (can be set via CLI, YAML, or web interface)
batch_size = wandb.config.batch_size
lr = wandb.config.lr

# Or set defaults
wandb.config.update({
    'batch_size': 4,
    'lr': 0.001,
    'epochs': 10
})
```

**But this replaces your config system** - not recommended for you since you already have a good one.

---

## Integration with Your Current System

### Your Config Flow

```
Environment Var → Config Defaults → YAML File → CLI Args
                                                    ↓
                                              Final Config
                                                    ↓
                                               W&B logs it
```

### Recommended Integration

```python
# train.py
import wandb
from dataclasses import asdict
from config import get_config_from_args

def main():
    # 1. Get config from your system (env + yaml + cli)
    config = get_config_from_args()

    # 2. Initialize W&B and log the config
    run = wandb.init(
        project='polsess-separation',
        name=f'{config.data.task}_lr{config.training.lr}_bs{config.data.batch_size}',
        config={
            'data': asdict(config.data),
            'model': asdict(config.model),
            'training': asdict(config.training)
        },
        tags=[config.data.task, f'model_size_{config.model.N}']
    )

    # 3. Access config normally in your code
    trainer = Trainer(model, train_loader, val_loader, config)

    # 4. Log metrics during training
    for epoch in range(config.training.num_epochs):
        train_sisdr = trainer.train_epoch()
        val_sisdr = trainer.validate()

        # Log to W&B
        wandb.log({
            'epoch': epoch,
            'train_sisdr': train_sisdr,
            'val_sisdr': val_sisdr,
            'lr': trainer.optimizer.param_groups[0]['lr']
        })

    # 5. Save model artifact
    wandb.save('checkpoints/model.pt')

    wandb.finish()
```

---

## CLI Arguments with W&B

W&B doesn't care about CLI args directly. Your workflow:

```bash
# You run with your CLI args
python train.py --lr 0.001 --batch-size 8 --epochs 50

# Inside train.py:
# 1. Your config system parses CLI args
config = get_config_from_args()  # ← Your system

# 2. W&B logs whatever config you give it
wandb.init(config=asdict(config))  # ← Just logs it
```

**Result on W&B dashboard:**
```
Run: eager-mountain-42
Hyperparameters:
  lr: 0.001
  batch_size: 8
  epochs: 50
```

---

## YAML Files with W&B

Same story - W&B logs whatever you pass:

```bash
# You run with YAML
python train.py --config experiments/baseline.yaml

# Inside train.py:
config = load_config_from_yaml(args.config)  # ← Your system
wandb.init(config=asdict(config))  # ← Just logs it
```

**W&B doesn't read the YAML** - your code does, then W&B logs the result.

---

## W&B Sweeps (Hyperparameter Search)

W&B has a built-in hyperparameter sweep feature:

### Step 1: Define Sweep Config (YAML)

```yaml
# sweep.yaml
program: train.py
method: bayes  # or 'grid', 'random'
metric:
  name: val_sisdr
  goal: maximize
parameters:
  lr:
    values: [0.0001, 0.001, 0.01]
  batch_size:
    values: [4, 8, 16]
  accumulation_steps:
    values: [1, 2, 4, 6]
  model.N:
    values: [128, 256, 512]
```

### Step 2: Modify train.py to Use W&B Config

```python
# train.py
def main():
    # Initialize W&B (sweep will inject config)
    wandb.init()

    # Get sweep config from W&B
    config = Config(
        data=DataConfig(
            batch_size=wandb.config.batch_size
        ),
        model=ModelConfig(
            N=wandb.config.get('model.N', 256)
        ),
        training=TrainingConfig(
            lr=wandb.config.lr,
            accumulation_steps=wandb.config.accumulation_steps
        )
    )

    # Train and log results
    trainer = Trainer(model, train_loader, val_loader, config)
    val_sisdr = trainer.train(config.training.num_epochs)

    # W&B automatically logs this
    wandb.log({'val_sisdr': val_sisdr})
```

### Step 3: Run Sweep

```bash
# Initialize sweep
wandb sweep sweep.yaml
# Output: Created sweep with ID: abc123

# Run agents (can run multiple in parallel)
wandb agent your-username/polsess-separation/abc123
```

W&B will try different hyperparameter combinations and find the best one.

---

## Comparison: Your Config System vs W&B Config

### Your System (Keep This!)
```python
# Flexible sources
config = get_config_from_args()  # Reads: env vars, YAML, CLI

# Use your config
model = ConvTasNet(
    N=config.model.N,
    B=config.model.B
)
```

**Pros:**
- ✅ Multiple config sources
- ✅ Type safety (dataclasses)
- ✅ Validation
- ✅ Works offline
- ✅ Already implemented

### W&B Config (Optional for Sweeps)
```python
# W&B controls config
wandb.init()
lr = wandb.config.lr
batch_size = wandb.config.batch_size
```

**Pros:**
- ✅ Built-in hyperparameter search
- ✅ Remote config via web interface
- ✅ Automatic logging

**Cons:**
- ⚠️ Replaces your system
- ⚠️ Requires W&B account
- ⚠️ No offline mode

---

## Recommended Hybrid Approach

**Use your config system, log to W&B:**

```python
# train.py
def main():
    # 1. Your config system (primary)
    config = get_config_from_args()  # env + YAML + CLI

    # 2. W&B (for logging only)
    wandb.init(
        project='polsess-separation',
        config=asdict(config),  # Just logs your config
        mode='online'  # or 'offline' for local-only
    )

    # 3. Use your config everywhere
    model = ConvTasNet(
        N=config.model.N,
        B=config.model.B,
        H=config.model.H
    )

    trainer = Trainer(model, train_loader, val_loader, config)

    # 4. Log metrics to W&B
    for epoch in range(config.training.num_epochs):
        train_sisdr = trainer.train_epoch()
        val_sisdr = trainer.validate()

        wandb.log({
            'epoch': epoch,
            'train_sisdr': train_sisdr,
            'val_sisdr': val_sisdr
        })

    wandb.finish()
```

**For hyperparameter sweeps:**
```python
def main():
    # Check if running as W&B sweep
    if os.getenv('WANDB_SWEEP_ID'):
        # W&B sweep mode: override config with W&B values
        wandb.init()
        config = Config(
            training=TrainingConfig(
                lr=wandb.config.lr,
                epochs=wandb.config.epochs
            )
        )
    else:
        # Normal mode: use your config system
        config = get_config_from_args()
        wandb.init(config=asdict(config))
```

---

## Example: Full Integration

```python
# train.py
import wandb
from dataclasses import asdict
from config import get_config_from_args

def main():
    # Parse config from env/YAML/CLI
    config = get_config_from_args()

    # Initialize W&B
    run = wandb.init(
        project='polsess-separation',
        name=f'exp_{config.data.task}_{config.training.lr}',
        config={
            'data': asdict(config.data),
            'model': asdict(config.model),
            'training': asdict(config.training)
        },
        tags=[config.data.task]
    )

    # Your normal training code
    model = ConvTasNet(...)
    trainer = Trainer(model, train_loader, val_loader, config)

    # Wrap train loop to log metrics
    for epoch in range(config.training.num_epochs):
        train_sisdr = trainer.train_epoch()
        val_sisdr = trainer.validate()

        # Log to W&B
        wandb.log({
            'epoch': epoch,
            'train/sisdr': train_sisdr,
            'val/sisdr': val_sisdr,
            'train/lr': trainer.optimizer.param_groups[0]['lr']
        })

        # Log best model
        if val_sisdr > trainer.best_val_sisdr:
            wandb.log({'best_val_sisdr': val_sisdr})

    # Log final artifacts
    wandb.save('checkpoints/model.pt')
    wandb.save('training.log')

    # Create summary
    wandb.summary['best_val_sisdr'] = trainer.best_val_sisdr
    wandb.summary['final_epoch'] = config.training.num_epochs

    wandb.finish()

if __name__ == '__main__':
    main()
```

---

## Running with Different Config Sources

### With CLI Args
```bash
python train.py --lr 0.001 --batch-size 8 --epochs 50
```
W&B logs: `lr=0.001, batch_size=8, epochs=50`

### With YAML
```bash
python train.py --config experiments/baseline.yaml
```
W&B logs: everything from `baseline.yaml`

### With YAML + CLI Override
```bash
python train.py --config baseline.yaml --lr 0.0001
```
W&B logs: YAML values with `lr=0.0001` overridden

### With Environment Variable
```bash
export POLSESS_DATA_ROOT=/data/polsess
python train.py
```
W&B logs: `data_root=/data/polsess` (from env var)

---

## What W&B Dashboard Shows

After running experiments, you'll see:

### Runs Table
| Name | Status | lr | batch_size | train_sisdr | val_sisdr |
|------|--------|----|-----------:|------------:|----------:|
| exp_1 | ✓ | 0.001 | 4 | 9.52 | 9.84 |
| exp_2 | ✓ | 0.0001 | 8 | 8.21 | 8.45 |
| exp_3 | ✓ | 0.01 | 4 | NaN | NaN |

### Charts
- Line plot: SI-SDR vs Epoch (all runs overlaid)
- Scatter plot: Learning Rate vs Final SI-SDR
- Parallel coordinates: Hyperparameters → Performance

### Filters
- Filter by: `lr > 0.001`
- Group by: `task` (ES vs EB)
- Sort by: `val_sisdr` (descending)

---

## Key Takeaways

1. **W&B is config-agnostic** - it logs whatever you give it
2. **Keep your config system** - W&B just logs the result
3. **Your workflow unchanged** - CLI/YAML/env vars work as before
4. **W&B adds visualization** - See results in beautiful dashboard
5. **For sweeps** - W&B can inject config, but it's optional

---

## Minimal Integration (5 minutes)

Want the absolute minimum to try W&B?

```python
# Add to train.py (3 lines)
import wandb

wandb.init(project='polsess-separation')
# ... your training code ...
wandb.log({'val_sisdr': val_sisdr})  # In validation loop
```

```bash
# Install
pip install wandb

# Login (one time)
wandb login

# Run normally
python train.py --lr 0.001
```

Then open https://wandb.ai to see your results!

---

## Summary

**Question:** How does W&B handle hyperparameters?

**Answer:** W&B logs whatever you pass to `wandb.init(config=...)`. It doesn't care if it comes from CLI args, YAML files, or Python code.

**Your workflow:**
```
Your Config System → Final Config → W&B logs it → Dashboard visualizes it
(env + YAML + CLI)
```

**You control the config, W&B just logs and visualizes it.**
