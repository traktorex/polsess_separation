#!/usr/bin/env python
"""Training-cost benchmark: training throughput and peak training VRAM.

WHAT THIS PROVES / PRODUCES
---------------------------
How much GPU time and GPU memory one *training* pass over a fixed number of
PolSESS mixtures costs per architecture, on one machine, under one protocol. The
companion to `benchmark_inference.py`; together they are the objective half of the
thesis's multi-axis comparison. Quality, epochs-to-convergence and
wall-clock-to-convergence are run-result facts and belong in a separate table
sourced from W&B / the experiment logs — see
`thesis/thesis-log/sweep_plan/ch5_test_evals/BENCHMARK_AUDIT.md`.

WHAT CHANGED ON 2026-07-30 (and why the old numbers are not these numbers)
-------------------------------------------------------------------------
1. **Validation is out of the timed region.** The old script timed
   `train_epoch()` and `validate()` separately but the thesis table quoted their
   *sum* as "training time per epoch" (ConvTasNet 200.22 + 11.83 = 212.0 s).
   Validation cost depends on the validation-set size and the metric set, neither
   of which is a property of the architecture. This script times training only;
   a stub validation loader exists purely because `Trainer.__init__` requires
   one, and `validate()` is never called.
2. **Warm-up is excluded explicitly** rather than by "drop epoch 1". A short
   warm-up pass absorbs torch.compile, cuDNN autotuning and allocator growth;
   only the second pass over a fixed sample count is timed.
3. **torch.compile is applied**, through the same `compile_for_model_type` that
   `train.py` and `train_sweep.py` use. The old benchmark ran uncompiled while
   every real run was compiled, so it mis-stated the cost of exactly the
   architectures compile helps most.
4. **Two batch sizes, reported side by side** (see the batch-size policy below).
5. Throughput (samples/s) replaces s/epoch as the stored quantity, so the reader
   can project any corpus size: s/epoch = corpus_size / samples_per_s.

BATCH-SIZE POLICY
-----------------
Benchmarking everything at batch_size=1 is not neutral: at bs=1 a small model is
launch-latency-bound and never saturates the GPU, while a large one already is
compute-bound, so bs=1 flatters big models and penalises small ones. But no
single batch size fits all sixteen configs on 12 GB either. So both ends are
measured and reported:

  * **`trained` (primary)** — each architecture's own `data.batch_size` from its
    experiment YAML. These were memory-driven constants spanning 1-16, never
    swept, i.e. part of how each architecture has to be run on this hardware.
    This is the cost the campaign actually paid, and the number RQ5 is about.
  * **`one` (secondary)** — batch_size=1 for every architecture. The controlled
    setting: the only one in which every config fits, and continuous with the
    pre-2026-07 figures.

If the two disagree on the ranking, the bs=1 ranking was a saturation artifact —
and reporting both is what lets the thesis say so rather than guess.

KNOWN LIMITS
------------
  * Everything must run on **one** machine to be comparable. Historical h/epoch
    figures in the logs mix rented cloud GPUs with two local cards and are not
    comparable with each other or with this table.
  * A real dataloader is used, so the pass includes MM-IPC lazy layer loading.
    With too few workers a cheap model becomes I/O-bound and its throughput
    measures the disk, not the architecture. `--num-workers` defaults to 4; if a
    row's throughput barely moves between `one` and `trained`, suspect I/O.
  * Peak memory is `torch.cuda.max_memory_allocated()` — tensor allocations only,
    excluding the CUDA context. Relative comparison, not "GPU needed".
  * Curriculum learning is disabled and all MM-IPC variants are allowed from the
    start, so the variant mix is the same for every architecture (seeded, hence
    reproducible).

HOW TO CITE
-----------
    scripts/benchmark_training.py -> docs/generated/benchmark_training.csv
    Training throughput: fixed 1000-mixture training pass (4.000 s clips,
    8 kHz), warm-up excluded, no validation, torch.compile as in production,
    AMP per the architecture's own policy.

USAGE
-----
    python scripts/benchmark_training.py
    python scripts/benchmark_training.py --only sepformer mossformer2
    python scripts/benchmark_training.py --batch-modes trained --train-samples 2000
"""

import argparse
import csv
import gc
import logging
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling benchmark_models

from benchmark_models import MODELS, select  # noqa: E402
from config import load_config_from_yaml  # noqa: E402
from models.factory import create_model_from_config  # noqa: E402
from training.setup import build_dataloaders  # noqa: E402
from training.trainer import Trainer  # noqa: E402
from utils import set_seed, setup_device_and_amp, setup_warnings  # noqa: E402
from utils.model_utils import compile_for_model_type, count_parameters  # noqa: E402
from utils.model_utils import format_parameter_count  # noqa: E402

SEED = 42


def _sibling_loader(loader, indices, batch_size):
    """A DataLoader over a subset of `loader`'s dataset, same collate/worker setup.

    Used to give the warm-up pass its own slice of mixtures so the timed pass
    starts from a warm compile cache and a grown allocator without having paid
    for a second full pass.
    """
    return torch.utils.data.DataLoader(
        torch.utils.data.Subset(loader.dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=loader.num_workers,
        prefetch_factor=loader.prefetch_factor if loader.num_workers > 0 else None,
        collate_fn=loader.collate_fn,
    )


def benchmark_training(name, size, config_path, batch_mode, args, logger):
    """Time one training pass for one config at one batch size. None on failure."""
    path = REPO_ROOT / config_path
    if not path.exists():
        print(f"  SKIP {name}: config not found ({config_path})")
        return None
    config = load_config_from_yaml(str(path))

    batch_size = 1 if batch_mode == "one" else config.data.batch_size
    total_samples = args.warmup_samples + args.train_samples

    # Benchmark overrides. train_max_samples covers warm-up + timed; the stub
    # validation set exists only to satisfy Trainer's "exactly one of
    # val_loader / per_variant_val_loaders" assertion — validate() is never run.
    config.data.batch_size = batch_size
    config.data.train_max_samples = total_samples
    config.data.val_max_samples = max(batch_size, 2)
    config.data.num_workers = args.num_workers
    config.training.use_wandb = False
    config.training.seed = SEED
    config.training.curriculum_learning = None  # all MM-IPC variants from step 1
    config.training.early_stopping_patience = None
    config.training.per_variant_validation = False
    # Pin the determinism policy so it is identical across architectures rather
    # than whatever each YAML happened to leave unset (tri-state: None = legacy
    # cuDNN-deterministic, no benchmark autotuning).
    config.training.deterministic = None
    config.training.save_dir = str(Path(args.scratch_dir) / "benchmark_checkpoints")

    set_seed(SEED)
    summary_info = {"seed": SEED}
    device = setup_device_and_amp(config, summary_info)
    on_cuda = device.startswith("cuda")

    train_loader, val_loader, per_variant = build_dataloaders(config, summary_info, logger)
    if len(train_loader.dataset) < total_samples:
        print(f"  SKIP {name}: dataset has only {len(train_loader.dataset)} mixtures, "
              f"need {total_samples}")
        return None

    warmup_loader = _sibling_loader(train_loader, range(args.warmup_samples), batch_size)
    timed_loader = _sibling_loader(
        train_loader, range(args.warmup_samples, total_samples), batch_size)

    model = create_model_from_config(config.model, summary_info)
    params = count_parameters(model)
    compiled = False
    if args.compile:
        before = type(model)
        model = compile_for_model_type(model, config.model.model_type, logger=logger)
        compiled = type(model) is not before

    trainer = Trainer(model, warmup_loader, val_loader, config, device=device,
                      logger=logger, wandb_logger=None,
                      per_variant_val_loaders=per_variant)

    # Warm-up: compile, cuDNN autotune, allocator growth. Not timed.
    trainer.current_epoch = 1
    trainer.train_epoch()

    # Timed pass. Swapping the loader keeps the training step itself the
    # production `Trainer.train_epoch`, so the benchmark cannot drift from what
    # real training does.
    trainer.train_loader = timed_loader
    trainer.current_epoch = 2
    if on_cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    train_sisdr, _ = trainer.train_epoch()
    if on_cuda:
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    peak_mem = torch.cuda.max_memory_allocated(device) / 1024**2 if on_cuda else 0.0

    amp_dtype = str(getattr(trainer, "amp_dtype", "")).replace("torch.", "") \
        if config.training.use_amp else "off"

    del model, trainer, train_loader, val_loader, warmup_loader, timed_loader
    gc.collect()
    if on_cuda:
        torch.cuda.empty_cache()

    return {
        "name": name, "size": size, "config": config_path, "params": params,
        "batch_mode": batch_mode, "batch_size": batch_size,
        "train_samples": args.train_samples, "warmup_samples": args.warmup_samples,
        "compiled": compiled, "amp_dtype": amp_dtype,
        "train_s": elapsed,
        "samples_per_s": args.train_samples / elapsed,
        "peak_train_mem_mb": peak_mem,
        "train_sisdr": train_sisdr,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--train-samples", type=int, default=1000,
                        help="Mixtures in the timed pass (default: 1000)")
    parser.add_argument("--warmup-samples", type=int, default=128,
                        help="Mixtures in the untimed warm-up pass (default: 128)")
    parser.add_argument("--batch-modes", default="trained,one",
                        help="Comma-separated subset of {trained,one} "
                             "(default: trained,one)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Dataloader workers (default: 4). Too few makes cheap "
                             "models I/O-bound")
    parser.add_argument("--no-compile", dest="compile", action="store_false",
                        help="Skip torch.compile (production applies it, so the "
                             "default is on)")
    parser.add_argument("--only", nargs="+", default=None,
                        help="Only models whose name contains any of these substrings")
    parser.add_argument("--scratch-dir", default="/tmp",
                        help="Where the never-written checkpoint dir is rooted")
    parser.add_argument("--project-epoch-samples", type=int, nargs="+",
                        default=[64000, 128000],
                        help="Corpus sizes to project s/epoch for (default: 64000 128000)")
    parser.add_argument("--allow-cpu", action="store_true",
                        help="Run without a GPU. Plumbing smoke-test only — the "
                             "timings and memory figures are meaningless")
    parser.add_argument("--output", default="docs/generated/benchmark_training.csv",
                        help="Output CSV, relative to the repo root")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        if not args.allow_cpu:
            print("ERROR: CUDA not available — a training benchmark needs the GPU. "
                  "Pass --allow-cpu to smoke-test the plumbing anyway.")
            sys.exit(1)
        print("WARNING: no GPU. Timings and memory are meaningless; this is a "
              "plumbing smoke-test only.\n")

    batch_modes = [m.strip() for m in args.batch_modes.split(",") if m.strip()]
    unknown = set(batch_modes) - {"trained", "one"}
    if unknown:
        print(f"ERROR: unknown batch mode(s) {sorted(unknown)}; expected trained/one")
        sys.exit(1)

    setup_warnings()
    torch.set_float32_matmul_precision("high")
    logger = logging.getLogger("polsess")
    logger.setLevel(logging.WARNING)

    gpu = torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"
    print(f"GPU: {gpu}  |  torch {torch.__version__} / cuda {torch.version.cuda}")
    print(f"{args.warmup_samples} warm-up + {args.train_samples} timed mixtures "
          f"(4.000 s @ 8 kHz), {args.num_workers} workers, "
          f"torch.compile {'on' if args.compile else 'off'}")
    print(f"batch modes: {', '.join(batch_modes)}")
    print()

    models = select(MODELS, args.only)
    if not models:
        print(f"No models match --only {args.only}. Available:",
              ", ".join(m[0] for m in MODELS))
        sys.exit(1)

    results = []
    for name, size, config_path in models:
        for batch_mode in batch_modes:
            print(f"Benchmarking {name} [{batch_mode}]...")
            try:
                row = benchmark_training(name, size, config_path, batch_mode,
                                         args, logger)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"  -> OOM at batch_mode={batch_mode}")
                row = None
            except RuntimeError as exc:
                torch.cuda.empty_cache()
                detail = "OOM" if "out of memory" in str(exc).lower() else str(exc)[:120]
                print(f"  -> FAILED: {detail}")
                row = None
            if row:
                print(f"  -> bs={row['batch_size']}  {row['train_s']:.1f} s for "
                      f"{row['train_samples']} mixtures  "
                      f"{row['samples_per_s']:.2f} samples/s  "
                      f"{row['peak_train_mem_mb']:.0f} MB peak  "
                      f"(compiled={row['compiled']}, amp={row['amp_dtype']})")
                results.append(row)
            print()

    header = (f"{'Model':<22}{'Params':>9}{'mode':>9}{'bs':>4}"
              f"{'samples/s':>11}{'Peak MB':>9}"
              + "".join(f"{'s/ep ' + str(n // 1000) + 'k':>12}"
                        for n in args.project_epoch_samples))
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        projections = "".join(f"{n / r['samples_per_s']:>12.0f}"
                              for n in args.project_epoch_samples)
        print(f"{r['name']:<22}{format_parameter_count(r['params']):>9}"
              f"{r['batch_mode']:>9}{r['batch_size']:>4}"
              f"{r['samples_per_s']:>11.2f}{r['peak_train_mem_mb']:>9.0f}{projections}")
    print("=" * len(header))
    print(f"GPU: {gpu}. s/ep columns are projections (corpus_size / samples_per_s), "
          "training only — no validation, no checkpointing.")

    output = REPO_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "model", "size", "config", "params", "batch_mode", "batch_size",
            "train_samples", "warmup_samples", "compiled", "amp_dtype",
            "train_s", "train_samples_per_s", "peak_train_mem_mb",
            "projected_s_per_epoch_64k", "projected_s_per_epoch_128k",
            "train_sisdr_last_pass", "gpu", "torch", "cuda",
        ])
        for r in results:
            writer.writerow([
                r["name"], r["size"], r["config"], r["params"], r["batch_mode"],
                r["batch_size"], r["train_samples"], r["warmup_samples"],
                r["compiled"], r["amp_dtype"], f"{r['train_s']:.2f}",
                f"{r['samples_per_s']:.4f}", f"{r['peak_train_mem_mb']:.1f}",
                f"{64000 / r['samples_per_s']:.1f}",
                f"{128000 / r['samples_per_s']:.1f}",
                f"{r['train_sisdr']:.3f}", gpu, torch.__version__,
                torch.version.cuda or "",
            ])
    print(f"\nResults saved to {output}")


if __name__ == "__main__":
    main()
