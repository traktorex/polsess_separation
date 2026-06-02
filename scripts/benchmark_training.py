"""Training time benchmark: wall-clock time per epoch for all architectures.

Runs 3 training epochs + 1 validation per model on the same GPU with batch_size=1
and a fixed dataset subset (2K samples) so results are comparable across models.
No checkpointing, no W&B logging.

Usage:
    python benchmark_training.py
    python benchmark_training.py --n-epochs 5
    python benchmark_training.py --train-samples 4000
"""

import argparse
import gc
import time
import csv
import sys
import logging
import torch
from pathlib import Path

from config import load_config_from_yaml
from models.factory import create_model_from_config
from datasets import get_dataset, polsess_collate_fn
from training.trainer import Trainer
from utils import set_seed, setup_warnings, setup_device_and_amp
from utils.model_utils import count_parameters, format_parameter_count


# Same configs as inference benchmark
CONFIGS = [
    ("ConvTasNet", "experiments/convtasnet/baseline.yaml"),
    ("DPRNN (k=16)", "experiments/dprnn/dprnn_baseline.yaml"),
    ("DPRNN (k=2)", "experiments/dprnn/variants/dprnn_baseline_kernel2.yaml"),
    ("SepFormer", "experiments/sepformer/sepformer_baseline.yaml"),
    ("SepFormer (pos-enc)", "experiments/sepformer/sepformer_baseline_positionalenc.yaml"),
    ("SepFormer (final 128k)", "experiments/sepformer/6-final-training/128k_final.yaml"),
    ("SPMamba (reduced 1.2M)", "experiments/spmamba/spmamba_sb_reduced.yaml"),
    ("SPMamba (unreduced 6M)", "experiments/spmamba/spmamba_sb.yaml"),
    ("MambaTasNet-XS", "experiments/mamba_tasnet/mamba_tasnet_xs.yaml"),
    ("MambaTasNet-S", "experiments/mamba_tasnet/mamba_tasnet_s.yaml"),
    ("MambaTasNet-M", "experiments/mamba_tasnet/mamba_tasnet_m.yaml"),
    ("MambaTasNet-L", "experiments/mamba_tasnet/mamba_tasnet_l.yaml"),
    ("DPMamba-XS", "experiments/dpmamba/dpmamba_xs.yaml"),
    ("DPMamba-S", "experiments/dpmamba/dpmamba_s.yaml"),
    ("DPMamba-M", "experiments/dpmamba/dpmamba_m.yaml"),
    ("DPMamba-L", "experiments/dpmamba/dpmamba_l.yaml"),
]


def benchmark_training(
    name: str,
    config_path: str,
    n_epochs: int,
    train_samples: int,
    val_samples: int,
    device_str: str,
    logger: logging.Logger,
) -> dict | None:
    """Run a short training benchmark for one model. Returns timing dict or None on failure."""
    config_path = Path(config_path)
    if not config_path.exists():
        print(f"  SKIP {name}: config not found")
        return None

    try:
        config = load_config_from_yaml(str(config_path))
    except Exception as e:
        print(f"  SKIP {name}: config load failed ({e})")
        return None

    # Override config for benchmarking
    config.data.batch_size = 1
    config.data.train_max_samples = train_samples
    config.data.val_max_samples = val_samples
    config.data.num_workers = 1
    config.training.num_epochs = n_epochs
    config.training.use_wandb = False
    config.training.seed = 42
    config.training.curriculum_learning = None  # No curriculum — all variants from start
    config.training.early_stopping_patience = None
    config.training.save_dir = "/tmp/benchmark_checkpoints"  # Avoid overwriting real checkpoints

    set_seed(42)

    # Setup device/AMP
    summary_info = {"seed": 42}
    device = setup_device_and_amp(config, summary_info)

    # Create datasets
    dataset_class = get_dataset(config.data.dataset_type)
    data_root = config.data.polsess.data_root

    train_dataset = dataset_class(
        data_root, subset="train", task=config.data.task,
        max_samples=config.data.train_max_samples,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=True,
        num_workers=config.data.num_workers,
        prefetch_factor=config.data.prefetch_factor if config.data.num_workers > 0 else None,
        collate_fn=polsess_collate_fn,
    )

    val_dataset = dataset_class(
        data_root, subset="val", task=config.data.task,
        max_samples=val_samples,
        allowed_variants=config.training.validation_variants,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=config.data.num_workers,
        prefetch_factor=config.data.prefetch_factor if config.data.num_workers > 0 else None,
        collate_fn=polsess_collate_fn,
    )

    # Create model
    try:
        model = create_model_from_config(config.model, summary_info)
    except Exception as e:
        print(f"  SKIP {name}: model creation failed ({e})")
        return None

    num_params = count_parameters(model)

    # Create trainer (no W&B)
    trainer = Trainer(
        model, train_loader, val_loader, config,
        device=device, logger=logger, wandb_logger=None,
    )

    # Track peak memory across entire training run
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    # Run epochs and time them
    train_times = []
    val_times = []

    for epoch in range(1, n_epochs + 1):
        trainer.current_epoch = epoch

        # Train epoch
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        train_sisdr, _ = trainer.train_epoch()
        torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        train_times.append(t1 - t0)

        # Validation
        torch.cuda.synchronize(device)
        t2 = time.perf_counter()
        val_sisdr, _ = trainer.validate()
        torch.cuda.synchronize(device)
        t3 = time.perf_counter()
        val_times.append(t3 - t2)

        print(f"  Epoch {epoch}: train {t1-t0:.1f}s, val {t3-t2:.1f}s | "
              f"SI-SDR train={train_sisdr:.2f} val={val_sisdr:.2f}")

    # Peak memory includes model + optimizer + gradients + activations + AMP scaler
    torch.cuda.synchronize(device)
    peak_memory_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024

    # Save counts before cleanup
    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)

    # Cleanup — ensure GPU memory is fully freed before next model
    del model, trainer, train_loader, val_loader, train_dataset, val_dataset
    gc.collect()
    torch.cuda.empty_cache()

    # Skip first epoch (compilation warmup), average the rest
    if len(train_times) > 1:
        avg_train = sum(train_times[1:]) / len(train_times[1:])
        avg_val = sum(val_times[1:]) / len(val_times[1:])
    else:
        avg_train = train_times[0]
        avg_val = val_times[0]

    return {
        "name": name,
        "params": num_params,
        "params_str": format_parameter_count(num_params),
        "train_samples": n_train,
        "val_samples": n_val,
        "avg_train_epoch_s": avg_train,
        "avg_val_epoch_s": avg_val,
        "peak_memory_mb": peak_memory_mb,
        "all_train_times": train_times,
        "all_val_times": val_times,
    }


def main():
    parser = argparse.ArgumentParser(description="Training time benchmark for all architectures")
    parser.add_argument("--n-epochs", type=int, default=3, help="Epochs per model (default: 3, first is warmup)")
    parser.add_argument("--train-samples", type=int, default=2000, help="Training samples (default: 2000)")
    parser.add_argument("--val-samples", type=int, default=400, help="Validation samples (default: 400)")
    parser.add_argument("--output", default="benchmark_training.csv", help="Output CSV path")
    parser.add_argument("--only", nargs="+", default=None,
                        help="Only benchmark configs whose name contains any of these substrings "
                             "(e.g. --only sepformer spmamba)")
    parser.add_argument("--project-epochs", type=int, default=70,
                        help="Epoch count used to project full-run days for 64k/128k (default: 70)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.")
        sys.exit(1)

    setup_warnings()
    torch.set_float32_matmul_precision('high')

    gpu_name = torch.cuda.get_device_name()
    print(f"GPU: {gpu_name}")
    print(f"Config: {args.n_epochs} epochs, {args.train_samples} train samples, "
          f"{args.val_samples} val samples, batch_size=1")
    print()

    # Quiet logger for trainer (avoid tqdm spam)
    logger = logging.getLogger("polsess")
    logger.setLevel(logging.WARNING)

    configs = CONFIGS
    if args.only:
        needles = [s.lower() for s in args.only]
        configs = [(n, p) for (n, p) in CONFIGS if any(s in n.lower() for s in needles)]
        if not configs:
            print(f"No configs match --only {args.only}.")
            print("Available:", ", ".join(n for n, _ in CONFIGS))
            sys.exit(1)

    results = []
    for name, config_path in configs:
        print(f"Benchmarking {name}...")
        try:
            result = benchmark_training(
                name, config_path,
                n_epochs=args.n_epochs,
                train_samples=args.train_samples,
                val_samples=args.val_samples,
                device_str="cuda",
                logger=logger,
            )
        except RuntimeError as e:
            torch.cuda.empty_cache()
            if "out of memory" in str(e).lower():
                print(f"  → OOM: {name} does not fit at batch_size=1 on this GPU "
                      "(try a shorter training segment).")
            else:
                print(f"  → FAILED: {str(e)[:100]}")
            result = None
        if result:
            print(f"  → Avg: {result['avg_train_epoch_s']:.1f}s/train epoch, "
                  f"{result['avg_val_epoch_s']:.1f}s/val epoch, "
                  f"{result['peak_memory_mb']:.0f} MB peak")
            results.append(result)
        print()

    # Summary table
    print("=" * 105)
    print(f"{'Model':<25} {'Params':>10} {'Train (s/ep)':>14} {'Val (s/ep)':>14} {'Total (s/ep)':>14} {'Peak Mem (MB)':>15}")
    print("-" * 105)
    for r in results:
        total = r["avg_train_epoch_s"] + r["avg_val_epoch_s"]
        print(f"{r['name']:<25} {r['params_str']:>10} "
              f"{r['avg_train_epoch_s']:>13.1f} {r['avg_val_epoch_s']:>13.1f} {total:>13.1f} {r['peak_memory_mb']:>14.0f}")
    print("=" * 105)
    print(f"GPU: {gpu_name} | {args.train_samples} train + {args.val_samples} val samples | batch_size=1")

    # Project full-run wall-clock. batch_size=1 → train time scales linearly with sample count.
    # Approximate: train-only (ignores validation overhead, which adds a bit each epoch).
    if results:
        print(f"\nProjected TRAIN wall-clock @ {args.project_epochs} epochs "
              f"(scaled from {args.train_samples} samples; excludes validation):")
        print(f"{'Model':<25} {'64k':>10} {'128k':>10}")
        print("-" * 47)
        for r in results:
            train_s_per_sample = r["avg_train_epoch_s"] / r["train_samples"]
            d64 = train_s_per_sample * 64000 * args.project_epochs / 86400
            d128 = train_s_per_sample * 128000 * args.project_epochs / 86400
            print(f"{r['name']:<25} {d64:>8.1f}d {d128:>8.1f}d")

    # Save CSV
    output_path = Path(args.output)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "params", "train_samples", "val_samples",
            "avg_train_s", "avg_val_s", "peak_memory_mb", "epoch1_train_s", "gpu",
        ])
        for r in results:
            writer.writerow([
                r["name"], r["params"], r["train_samples"], r["val_samples"],
                f"{r['avg_train_epoch_s']:.2f}",
                f"{r['avg_val_epoch_s']:.2f}",
                f"{r['peak_memory_mb']:.1f}",
                f"{r['all_train_times'][0]:.2f}",
                gpu_name,
            ])
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
