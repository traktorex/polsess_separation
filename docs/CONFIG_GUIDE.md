# Configuration Guide

This guide explains how to configure training and evaluation using environment variables, CLI arguments, and code.

## Table of Contents

1. [YAML Configuration Files](#yaml-configuration-files)
2. [Environment Variables](#environment-variables)
3. [Command Line Arguments](#command-line-arguments)
4. [Programmatic Configuration](#programmatic-configuration)
5. [Priority Order](#priority-order)
6. [Examples](#examples)

---

## YAML Configuration Files

YAML config files provide a clean, organized way to manage experiments. This is especially useful for hyperparameter sweeps and reproducibility.

### Basic Usage

```bash
# Train with YAML config
python train.py --config experiments/baseline.yaml

# Override specific values with CLI
python train.py --config experiments/baseline.yaml --lr 0.0001 --epochs 50
```

### Creating a YAML Config

**Example: experiments/baseline.yaml**

```yaml
data:
  batch_size: 4
  task: ES
  num_workers: 4

model:
  N: 256
  B: 256
  H: 512
  P: 3
  X: 8
  R: 4
  C: 1
  norm_type: gLN

training:
  num_epochs: 20
  lr: 0.001
  weight_decay: 0.0001
  grad_clip_norm: 5.0
  use_amp: true
  amp_eps: 0.0001
  device: cuda
  save_dir: checkpoints/baseline
```

### Partial YAML Configs

You don't need to specify all fields - missing fields use defaults:

```yaml
# Minimal config - only specify what changes
data:
  batch_size: 8

training:
  lr: 0.0001
  num_epochs: 50
```

### Available Experiment Configs

The `experiments/` folder contains pre-configured YAML files:

- **baseline.yaml** - Default config that achieved 9.84 dB SI-SDR
- **large_model.yaml** - Larger model (~34M params) for better performance
- **small_fast.yaml** - Smaller model (~2M params) for quick experiments
- **eb_task.yaml** - Configuration for EB task (enhance both speakers)
- **lr_sweep.yaml** - Template for learning rate experiments

See [experiments/README.md](experiments/README.md) for details.

### Loading and Saving in Python

```python
from config import load_config_from_yaml, save_config_to_yaml

# Load from YAML
config = load_config_from_yaml('experiments/baseline.yaml')

# Modify
config.training.lr = 0.0005

# Save for reproducibility
save_config_to_yaml(config, 'experiments/my_experiment.yaml')
```

### Why Use YAML?

**Advantages:**

- **Reproducibility:** Save exact config with results
- **Organization:** Group related experiments in one file
- **Version control:** Easy to track changes in git
- **Clarity:** Easier to read than long CLI commands
- **Reusability:** Share configs between team members

**When to use:**

- Running multiple experiments
- Hyperparameter sweeps
- Documenting results
- Sharing configurations

**When CLI is better:**

- Quick one-off experiments
- Overriding a single value
- Interactive testing

---

## Environment Variables

Environment variables provide default values that persist across sessions. They're read when the config is created.

### How `default_factory` Works

```python
data_root: str = field(
    default_factory=lambda: os.getenv(
        'POLSESS_DATA_ROOT',
        'F:\\PolSMSE\\EksperymentyMOWA\\BAZY\\MOWA\\PolSESS_C_in\\PolSESS_C_in'
    )
)
```

**Breakdown:**

- `default_factory=lambda:` - Creates a function that runs when the dataclass is instantiated
- `os.getenv('POLSESS_DATA_ROOT', ...)` - Looks for environment variable `POLSESS_DATA_ROOT`
  - If found: Uses that value
  - If not found: Uses the fallback value (the long path string)
- This allows you to set a persistent default without hardcoding paths

### Setting Environment Variables

**Windows (PowerShell):**

```powershell
# Temporary (current session only)
$env:POLSESS_DATA_ROOT = "D:\Datasets\PolSESS"

# Permanent (user-level)
[System.Environment]::SetEnvironmentVariable('POLSESS_DATA_ROOT', 'D:\Datasets\PolSESS', 'User')

# Verify
echo $env:POLSESS_DATA_ROOT
```

**Windows (CMD):**

```cmd
# Temporary (current session only)
set POLSESS_DATA_ROOT=D:\Datasets\PolSESS

# Permanent
setx POLSESS_DATA_ROOT "D:\Datasets\PolSESS"
```

**Linux/Mac:**

```bash
# Temporary (current session only)
export POLSESS_DATA_ROOT="/path/to/PolSESS"

# Permanent (add to ~/.bashrc or ~/.zshrc)
echo 'export POLSESS_DATA_ROOT="/path/to/PolSESS"' >> ~/.bashrc
source ~/.bashrc
```

**Available Environment Variables:**

- `POLSESS_DATA_ROOT` - Path to PolSESS dataset directory

---

## Command Line Arguments

CLI arguments override both environment variables and config defaults.

### Training (train.py)

**Data Settings:**

```bash
python train.py --data-root /path/to/data      # Override dataset path
python train.py --task EB                       # Task: ES or EB
python train.py --batch-size 8                  # Physical batch size
python train.py --num-workers 4                 # DataLoader workers
```

**Model Settings:**

```bash
python train.py --model-size small    # Presets: small, default, large
# small:   N=128, B=128, H=256
# default: N=256, B=256, H=512 (8.64M params)
# large:   N=512, B=512, H=1024
```

**Training Settings:**

```bash
python train.py --epochs 20                     # Number of epochs
python train.py --lr 0.001                      # Learning rate
python train.py --no-amp                        # Disable AMP
python train.py --device cuda                   # Device: cuda or cpu
```

**Full Example:**

```bash
python train.py \
    --data-root /mnt/datasets/PolSESS \
    --task ES \
    --batch-size 4 \
    --epochs 20 \
    --lr 0.001 \
    --model-size default
```

### Evaluation (evaluate.py)

```bash
# Basic evaluation
python evaluate.py --checkpoint checkpoints/best_model.pt

# With YAML config
python evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --config experiments/baseline.yaml

# With custom settings
python evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --data-root /path/to/data \
    --variant SER \
    --batch-size 8 \
    --output results.csv \
    --no-pesq \
    --no-stoi
```

**Arguments:**

- `--checkpoint` - Path to model checkpoint (required)
- `--config` - Path to YAML config file (optional)
- `--data-root` - Dataset path (optional, uses env var or default)
- `--task` - ES or EB (optional, uses config default)
- `--variant` - Specific variant to test (SER, SR, ER, R, SE, S, E)
- `--batch-size` - Batch size for evaluation (optional, uses config default)
- `--device` - cuda or cpu
- `--no-pesq` - Skip PESQ (faster)
- `--no-stoi` - Skip STOI (faster)
- `--output` - Save results to CSV

---

## Programmatic Configuration

You can also create and modify configs in Python code:

### Creating Configs

```python
from config import Config, DataConfig, ModelConfig, TrainingConfig

# Use defaults
config = Config()

# Create with custom values
config = Config(
    data=DataConfig(
        data_root='/path/to/data',
        batch_size=8,
        task='ES'
    ),
    model=ModelConfig(
        N=128,
        B=128,
        H=256
    ),
    training=TrainingConfig(
        lr=0.001,
        num_epochs=20
    )
)

# Print summary
print(config.summary())
```

### Modifying Configs

```python
# After creation
config = Config()
config.data.batch_size = 8
config.training.num_epochs = 20
config.model.N = 512
```

### Using get_config_from_args()

This is what train.py uses:

```python
from config import get_config_from_args

# Parse CLI args and create config
config = get_config_from_args()

# Now use config
model = ConvTasNet(
    N=config.model.N,
    B=config.model.B,
    # ... etc
)
```

---

## Priority Order

Configuration values are applied in this order (later overrides earlier):

1. **Hardcoded defaults** in config.py

   ```python
   batch_size: int = 4
   ```
2. **Environment variables** (if set)

   ```bash
   export POLSESS_DATA_ROOT="/custom/path"
   ```
3. **YAML config file** (if provided)

   ```bash
   python train.py --config experiments/baseline.yaml
   ```
4. **Command line arguments** (highest priority)

   ```bash
   python train.py --batch-size 8 --data-root /override/path
   ```

### Example Priority Flow

```python
# 1. config.py default
data_root: str = field(default_factory=lambda: os.getenv(
    'POLSESS_DATA_ROOT',
    'F:\\PolSMSE\\...'  # <-- Fallback default
))
```

```bash
# 2. Set environment variable (overrides default)
export POLSESS_DATA_ROOT="/env/path"

# 3. Use YAML config (overrides environment variable)
# experiments/my_config.yaml:
#   data:
#     data_root: /yaml/path

python train.py --config experiments/my_config.yaml

# 4. CLI arg overrides everything
python train.py --config experiments/my_config.yaml --data-root /cli/path
```

**Result:** Uses `/cli/path`

### YAML + CLI Override Example

```bash
# baseline.yaml has:
#   training:
#     lr: 0.001
#     num_epochs: 20

# Override LR with CLI, keep other YAML values
python train.py --config experiments/baseline.yaml --lr 0.0001

# Result:
# - Uses lr: 0.0001 (from CLI)
# - Uses num_epochs: 20 (from YAML)
# - Uses batch_size: 4 (from YAML)
```

---

## Examples

### Example 1: Quick Local Training

Use defaults (environment variable or hardcoded):

```bash
python train.py
```

### Example 2: Custom Dataset Location

Set environment variable once:

```powershell
$env:POLSESS_DATA_ROOT = "D:\My Datasets\PolSESS"
```

Then just run:

```bash
python train.py
```

### Example 3: One-Off Custom Path

Override for single run:

```bash
python train.py --data-root /tmp/PolSESS_test
```

### Example 4: YAML-Based Experiments

```bash
# Use baseline config (known to work well)
python train.py --config experiments/baseline.yaml

# Quick test with small model
python train.py --config experiments/small_fast.yaml --epochs 5

# Full training with large model
python train.py --config experiments/large_model.yaml

# Test different task
python train.py --config experiments/eb_task.yaml
```

### Example 5: Hyperparameter Sweep with YAML

```bash
# Create base config, override specific values
python train.py --config experiments/baseline.yaml --lr 0.0001
python train.py --config experiments/baseline.yaml --lr 0.001
python train.py --config experiments/baseline.yaml --lr 0.01

# Or use LR template and override other values
python train.py --config experiments/lr_sweep.yaml --epochs 50
python train.py --config experiments/lr_sweep.yaml --batch-size 8
```

### Example 6: Evaluation Workflow

```bash
# Fast evaluation (SI-SDR only)
python evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --no-pesq --no-stoi

# Full evaluation with all metrics
python evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --output full_results.csv

# Test specific challenging variant
python evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --variant R \
    --output reverb_only_results.csv
```

### Example 7: Multi-Environment Setup

**Development machine:**

```powershell
# .env or PowerShell profile
$env:POLSESS_DATA_ROOT = "D:\Datasets\PolSESS"
```

**Training server:**

```bash
# ~/.bashrc
export POLSESS_DATA_ROOT="/data/shared/PolSESS"
```

**Both machines:**

```bash
# Same command works on both!
python train.py --epochs 100
```

---

## Troubleshooting

### Scripts in subdirectories fail to import

**Problem:**

```bash
PS F:\python\MAG2\polsess_separation\scripts> python .\test_evaluate.py
ModuleNotFoundError: No module named 'evaluate'
```

**Solution 1 - Run from project root:**

```bash
# From polsess_separation directory
python scripts/test_evaluate.py
```

**Solution 2 - Add parent to path (already in test scripts):**

```python
import sys
sys.path.insert(0, '..')

from evaluate import VariantDataset
```

**Solution 3 - Use -m flag:**

```bash
# From project root
python -m scripts.test_evaluate
```

### Environment variable not working

**Check if it's set:**

```powershell
# PowerShell
echo $env:POLSESS_DATA_ROOT

# CMD
echo %POLSESS_DATA_ROOT%

# Linux/Mac
echo $POLSESS_DATA_ROOT
```

**If empty, it's not set. Set it and restart your terminal.**

### Want to see current config

```bash
# Add this temporarily to train.py or evaluate.py
python -c "from config import Config; print(Config().summary())"
```

---

## Best Practices

1. **Development:** Use environment variables for your machine
2. **Production:** Use CLI arguments in scripts/automation
3. **Experiments:** Use CLI arguments to override specific values
4. **Default paths:** Keep reasonable defaults in config.py for quick testing

**Example workflow:**

```bash
# Set environment variable once
export POLSESS_DATA_ROOT="/data/PolSESS"

# Run experiments with CLI overrides
python train.py --epochs 10 --model-size small    # Quick test
python train.py --epochs 100 --model-size large   # Full training
```
