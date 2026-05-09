"""Batch evaluate all checkpoints and save results to CSV.

Uses local Config defaults for data_root (POLSESS_DATA_ROOT env var, else the
PolSESS_C_new_64 default in config.py); only `task` is inherited from the
checkpoint's embedded config.

Usage:
    python evaluate_all.py
    python evaluate_all.py --resume                  # skip already evaluated
    python evaluate_all.py --max-samples 50          # quick test
    python evaluate_all.py --no-pesq --no-stoi       # SI-SDR only (faster)
"""

import argparse
import csv
import logging
import sys
from datetime import datetime
from pathlib import Path

import torch

from utils import load_model_for_inference, count_parameters


def find_all_checkpoints(checkpoints_dir: str):
    """Find all checkpoint files under checkpoints/."""
    checkpoints_dir = Path(checkpoints_dir)
    checkpoint_files = sorted(checkpoints_dir.glob("**/SB/*/*.pt"))
    return checkpoint_files


def load_checkpoints_from_csv(csv_path: str):
    """Read checkpoint paths from a semicolon-separated CSV with a `path` column."""
    csv_path = Path(csv_path)
    paths = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            path_str = (row.get("path") or "").strip()
            if not path_str:
                continue
            paths.append(Path(path_str))
    return paths


def extract_checkpoint_info(checkpoint_path: Path, checkpoint: dict):
    """Extract metadata from checkpoint path and contents."""
    # Path structure: checkpoints/{model_type}/SB/{run_name}/{model}_{task}_best.pt
    parts = checkpoint_path.relative_to(checkpoint_path.parents[3])
    model_type = parts.parts[0]
    task = parts.parts[1]
    run_name = parts.parts[2]

    config = checkpoint.get("config", {})
    model_config = config.get("model", {}).get(model_type, {})
    training_config = config.get("training", {})
    data_config = config.get("data", {})

    return {
        "model_type": model_type,
        "task": task,
        "run_name": run_name,
        "checkpoint_path": str(checkpoint_path),
        "epoch": checkpoint.get("epoch", ""),
        "val_sisdr": checkpoint.get("val_sisdr", ""),
        "model_config": model_config,
        "training_config": training_config,
        "data_config": data_config,
    }


def run_evaluation(checkpoint_path: str, device: str,
                   max_samples: int = None, no_pesq: bool = False,
                   no_stoi: bool = False):
    """Run evaluate.py logic for a single checkpoint."""
    from config import Config
    from evaluate import evaluate_by_variant

    model, checkpoint = load_model_for_inference(checkpoint_path, device)
    info = extract_checkpoint_info(Path(checkpoint_path), checkpoint)
    num_params = count_parameters(model)

    # Use local Config defaults (data_root from POLSESS_DATA_ROOT env, else
    # PolSESS_C_new_64 from config.py); only inherit `task` from the
    # checkpoint's embedded config. The sidecar config.yaml is ignored —
    # it's byte-identical to the embedded config and on imported checkpoints
    # carries the source PC's data_root (e.g. C:\datasety\...).
    config = Config()
    task = checkpoint.get("config", {}).get("data", {}).get("task")
    if task:
        config.data.task = task

    # Run evaluation by variant
    results = evaluate_by_variant(
        model=model,
        config=config,
        device=device,
        compute_pesq=not no_pesq,
        compute_stoi=not no_stoi,
        max_samples=max_samples,
    )

    return info, num_params, results


def flatten_results(info: dict, num_params: int, variant_results: dict):
    """Flatten evaluation results into a single CSV row per checkpoint.

    Per-variant SI-SDR values are stored in columns like si_sdr_SER, si_sdr_SE, etc.
    """
    model_config = info["model_config"]
    training_config = info["training_config"]
    data_config = info["data_config"]

    # Compute average SI-SDR across all variants
    all_sisdrs = [r["si_sdr"] for r in variant_results.values()]
    avg_sisdr = sum(all_sisdrs) / len(all_sisdrs) if all_sisdrs else 0.0

    row = {
        # Identity
        "model_type": info["model_type"],
        "task": info["task"],
        "run_name": info["run_name"],
        # Checkpoint metadata
        "epoch": info["epoch"],
        "val_sisdr": info["val_sisdr"],
        "num_params": num_params,
        "checkpoint_path": info["checkpoint_path"],
        # Average across all variants
        "avg_sisdr": avg_sisdr,
    }

    # Per-variant SI-SDR columns (si_sdr_SER, si_sdr_SE, etc.)
    all_variants = ["SER", "SR", "ER", "R", "SE", "S", "E", "C"]
    for variant in all_variants:
        if variant in variant_results:
            row[f"si_sdr_{variant}"] = variant_results[variant].get("si_sdr", "")
        else:
            row[f"si_sdr_{variant}"] = ""

    # Training config
    row.update({
        "lr": training_config.get("lr", ""),
        "batch_size": training_config.get("batch_size", ""),
        "epochs": training_config.get("epochs", ""),
        "optimizer": training_config.get("optimizer", ""),
        "scheduler": training_config.get("scheduler", ""),
        "grad_clip": training_config.get("grad_clip", ""),
        # Data config
        "segment_length": data_config.get("segment_length", ""),
        "sample_rate": data_config.get("sample_rate", ""),
        # Model config (flattened)
        "model_config": str(model_config),
        # Metadata
        "evaluated_at": datetime.now().isoformat(),
    })

    return row


def main():
    # Match evaluate.py's logging so per-variant headers and result lines are
    # visible (without this, the `polsess` logger stays at WARNING and only
    # tqdm bars appear).
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Batch evaluate all checkpoints")
    parser.add_argument("--checkpoints-dir", default="checkpoints",
                        help="Root checkpoints directory (default: checkpoints)")
    parser.add_argument("--output", default="evaluation_results.csv",
                        help="Output CSV path (default: evaluation_results.csv)")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit samples per variant (for quick testing)")
    parser.add_argument("--no-pesq", action="store_true", help="Skip PESQ computation")
    parser.add_argument("--no-stoi", action="store_true", help="Skip STOI computation")
    parser.add_argument("--resume", action="store_true",
                        help="Skip checkpoints already present in the output CSV")
    parser.add_argument("--min-val-sisdr", type=float, default=3.0,
                        help="Skip checkpoints with val SI-SDR below this threshold (default: 3.0 dB)")
    parser.add_argument("--csv-list", default=None,
                        help="Evaluate only checkpoints listed in this semicolon-separated CSV "
                             "(requires a `path` column); overrides --checkpoints-dir discovery")
    args = parser.parse_args()

    # Find checkpoints (either from CSV list or by globbing)
    if args.csv_list:
        checkpoint_files = load_checkpoints_from_csv(args.csv_list)
        print(f"Loaded {len(checkpoint_files)} checkpoints from {args.csv_list}")
    else:
        checkpoint_files = find_all_checkpoints(args.checkpoints_dir)
        print(f"Found {len(checkpoint_files)} checkpoints")

    if not checkpoint_files:
        print("No checkpoints found. Check --checkpoints-dir path.")
        return

    # Load already-evaluated checkpoints if resuming
    already_evaluated = set()
    output_path = Path(args.output)
    if args.resume and output_path.exists():
        with open(output_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                already_evaluated.add(row["checkpoint_path"])
        print(f"Resuming: {len(already_evaluated)} checkpoints already evaluated")

    # Determine if we need to write the header
    write_header = not output_path.exists() or not args.resume

    # Evaluate each checkpoint, appending results incrementally
    evaluated = 0
    skipped = 0
    filtered = 0
    failed = 0

    for i, ckpt_path in enumerate(checkpoint_files):
        ckpt_str = str(ckpt_path)

        if ckpt_str in already_evaluated:
            skipped += 1
            continue

        run_name = ckpt_path.parent.name
        model_type = ckpt_path.parts[-4]

        # Quick filter: check val_sisdr from checkpoint metadata without loading model
        ckpt_meta = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        val_sisdr = ckpt_meta.get("val_sisdr", 0.0)
        del ckpt_meta

        if val_sisdr < args.min_val_sisdr:
            filtered += 1
            continue

        print(f"\n[{i+1}/{len(checkpoint_files)}] {model_type}/{run_name} (val_sisdr={val_sisdr:.2f} dB)")

        try:
            info, num_params, results = run_evaluation(
                checkpoint_path=ckpt_str,
                device=args.device,
                max_samples=args.max_samples,
                no_pesq=args.no_pesq,
                no_stoi=args.no_stoi,
            )

            row = flatten_results(info, num_params, results)

            # Append to CSV
            with open(output_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                if write_header:
                    writer.writeheader()
                    write_header = False
                writer.writerow(row)

            print(f"  -> avg SI-SDR: {row['avg_sisdr']:.2f} dB")
            evaluated += 1

        except Exception as e:
            print(f"  -> FAILED: {e}")
            failed += 1

        # Free GPU memory
        torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f"Done. Evaluated: {evaluated}, Skipped (resume): {skipped}, "
          f"Filtered (val_sisdr < {args.min_val_sisdr} dB): {filtered}, Failed: {failed}")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
