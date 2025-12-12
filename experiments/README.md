# Experiment Configurations

This folder contains YAML configuration files for different experiments.

## Available Configurations

### baseline.yaml

The configuration that achieved **9.84 dB SI-SDR**.

- Model: Default ConvTasNet (8.64M params)
- Task: ES (single speaker)
- Batch: 4

**Usage:**

```bash
python train.py --config experiments/baseline.yaml
```

### large_model.yaml

Larger model for potentially better performance.

- Model: Large ConvTasNet (~34M params)
- N=512, H=1024, R=5
- Slower training but more capacity

**Usage:**

```bash
python train.py --config experiments/large_model.yaml
```

### small_fast.yaml

Smaller model for quick experiments.

- Model: Small ConvTasNet (~2M params)
- N=128, H=256, R=3
- Faster training, good for testing

**Usage:**

```bash
python train.py --config experiments/small_fast.yaml
```

### eb_task.yaml

Enhance Both speakers (remove noise, keep both speakers).

- Task: EB
- Expected: 10-12 dB SI-SDR (easier than ES)

**Usage:**

```bash
python train.py --config experiments/eb_task.yaml
```

### lr_sweep.yaml

Example for hyperparameter search.

- Lower learning rate (0.0001 vs 0.001)
- Use as template for LR experiments

**Usage:**

```bash
python train.py --config experiments/lr_sweep.yaml
```

### klec_replication.yaml

Klec et al. (2024) dual corpus replication.

- Dual corpus: C_in (indoor) + C_out (outdoor)
- Architecture: kernel_size=20, stride=10, R=4, X=8
- Batch size: 2 (1 indoor + 1 outdoor)
- MM-IPC augmentation (multi-variant data)

**Usage:**

```bash
python train_dual_corpus.py --config experiments/klec_replication.yaml
```

### wandb_sweep.yaml

W&B sweep for single corpus hyperparameter optimization.

- Method: Bayesian optimization
- Sweeps: model_B, model_H, weight_decay, grad_clip_norm, batch_size
- Early termination: HyperBand

**Usage:**

```bash
wandb sweep experiments/wandb_sweep.yaml
wandb agent <sweep-id>
```

### klec_sweep.yaml

W&B sweep for dual corpus (Klec et al.) hyperparameter optimization.

- Method: Bayesian optimization
- Sweeps: model_B, model_H, weight_decay, grad_clip_norm
- Epochs: 50 (full training)
- Early termination: HyperBand

**Usage:**

```bash
wandb sweep experiments/klec_sweep.yaml
wandb agent <sweep-id>
```

### klec_sweep_quick.yaml

Quick grid search for dual corpus testing.

- Method: Grid search
- Reduced parameter space (12 runs total)
- Epochs: 20 (faster testing)

**Usage:**

```bash
wandb sweep experiments/klec_sweep_quick.yaml
wandb agent <sweep-id>
```

## Creating Your Own Config

Create a new YAML file:

```yaml
# experiments/my_experiment.yaml

data:
  batch_size: 8
  task: ES
  # data_root: /custom/path  # Optional: override data path

model:
  N: 256
  B: 256
  H: 512
  # ... see baseline.yaml for all options

training:
  num_epochs: 50
  lr: 0.001
  # ... see baseline.yaml for all options
```

Then run:

```bash
python train.py --config experiments/my_experiment.yaml
```

## Overriding Config Values

You can override any config value with CLI arguments:

```bash
# Use baseline config but change LR
python train.py --config experiments/baseline.yaml --lr 0.0001

# Use large model but fewer epochs
python train.py --config experiments/large_model.yaml --epochs 10

# Use baseline but disable AMP
python train.py --config experiments/baseline.yaml --no-amp
```

## Priority Order

Configuration values are applied in this order (later overrides earlier):

1. **YAML file** (`--config experiments/baseline.yaml`)
2. **CLI arguments** (`--lr 0.0001 --epochs 20`)

Example:

```bash
# baseline.yaml has lr: 0.001
# CLI overrides with --lr 0.0001
# Result: Uses lr=0.0001
python train.py --config experiments/baseline.yaml --lr 0.0001
```

## Saving Your Config

After training, you can save the exact config used:

```python
from config import get_config_from_args, save_config_to_yaml

config = get_config_from_args()
save_config_to_yaml(config, 'experiments/my_run.yaml')
```

This is useful for reproducibility!

## Tips

1. **Start with baseline.yaml** - It's known to work well (9.84 dB)
2. **Copy and modify** - Copy baseline.yaml and change specific values
3. **Name meaningfully** - Use descriptive names like `lr_0.0001.yaml`
4. **Save with results** - Keep config with experiment results
5. **Version control** - Commit configs to git for reproducibility
