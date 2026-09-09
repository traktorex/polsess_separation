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
import contextlib
import csv
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

import torch

from utils import load_model_for_inference, count_parameters, git_provenance

# Default tracked eval manifest (survey gap 17): the single answer to "which
# checkpoints constitute the thesis comparison". Consolidates the legacy
# checkpoints_for_eval{,2,3}.csv lists.
DEFAULT_MANIFEST = "experiments/thesis_eval_manifest.csv"


class _Tee:
    """Write to multiple streams (used to mirror stdout to a per-checkpoint log)."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self._streams:
            s.flush()


@contextlib.contextmanager
def capture_to_file(log_path: Path):
    """Mirror print() and root-logger output to ``log_path`` for the duration.

    Terminal output is preserved (Tee). The log is plain text; tqdm bars
    write to stderr and do not appear in the file (only the per-variant
    summary lines that go through ``logger.info`` are captured).
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(log_path, "w")

    root = logging.getLogger()
    file_handler = logging.StreamHandler(f)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    )
    file_handler.setLevel(logging.INFO)
    root.addHandler(file_handler)

    old_stdout = sys.stdout
    sys.stdout = _Tee(old_stdout, f)
    try:
        yield
    finally:
        sys.stdout = old_stdout
        file_handler.flush()
        root.removeHandler(file_handler)
        f.flush()
        f.close()


def _safe_filename(s: str) -> str:
    """Strip filesystem-unsafe chars from a checkpoint identifier."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", s)


def find_all_checkpoints(checkpoints_dir: str):
    """Find all checkpoint files under checkpoints/."""
    checkpoints_dir = Path(checkpoints_dir)
    checkpoint_files = sorted(checkpoints_dir.glob("**/SB/*/*.pt"))
    return checkpoint_files


def load_checkpoints_from_csv(csv_path: str):
    """Read checkpoint paths from a semicolon-separated CSV with a `path` column.

    Legacy format used by checkpoints_for_eval{,2,3}.csv (kept for --csv-list).
    """
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


def load_checkpoints_from_manifest(manifest_path: str):
    """Read checkpoint paths from the tracked eval manifest.

    Comma-delimited, columns: display_name, checkpoint_path, dataset, notes.
    Only checkpoint_path is used here; the other columns document the run.
    """
    manifest_path = Path(manifest_path)
    paths = []
    with open(manifest_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            path_str = (row.get("checkpoint_path") or "").strip()
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
    from evaluate import evaluate_by_variant, checkpoint_sample_rate

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
    # The sampling rate is a property of the trained model (8 kHz for every
    # checkpoint that predates the field); it picks PESQ nb/wb + STOI rate and
    # makes PolSESSDataset refuse a corpus stored at a different rate.
    config.data.sample_rate = checkpoint_sample_rate(checkpoint)

    # Run evaluation by variant (evaluate_by_variant forces batch_size=1)
    results = evaluate_by_variant(
        model=model,
        config=config,
        device=device,
        compute_pesq=not no_pesq,
        compute_stoi=not no_stoi,
        max_samples=max_samples,
    )

    # Eval-run provenance context for flatten_results (git_sha/git_dirty are
    # merged in by the caller, computed once per invocation).
    eval_context = {
        "eval_data_root": config.data.polsess.data_root,
        "eval_subset": "test",
        "eval_batch_size": 1,
    }

    return info, num_params, results, eval_context


# Exact, ordered column schema of a flatten_results row. Pinned by a golden
# test (tests/test_evaluate_all_flatten.py) — the guard that would have caught
# the schema drift that corrupted evaluation_results.csv. Change this tuple and
# the test in the same commit.
_VARIANTS = ("SER", "SR", "ER", "R", "SE", "S", "E", "C")
FLATTEN_COLUMNS = (
    # Identity
    "model_type", "task", "run_name",
    # Checkpoint metadata
    "epoch", "val_sisdr", "num_params", "checkpoint_path",
    # Averages across all variants
    "avg_sisdr", "avg_sisdri",
    # Per-variant SI-SDR + SI-SDRi
    *sum(([f"si_sdr_{v}", f"si_sdri_{v}"] for v in _VARIANTS), []),
    # Training config (from the checkpoint's embedded config)
    "lr", "batch_size", "epochs", "optimizer", "scheduler", "grad_clip",
    # Data config the checkpoint was TRAINED on (renamed from segment_length /
    # sample_rate so it is never confused with the eval dataset, survey gap 2)
    "train_segment_length", "train_sample_rate",
    "model_config", "evaluated_at",
    # Eval-run provenance (survey gap 2)
    "git_sha", "git_dirty", "torch_version",
    "eval_dataset_name", "eval_data_root", "eval_subset", "eval_batch_size",
)


def flatten_results(info: dict, num_params: int, variant_results: dict, eval_context: dict):
    """Flatten evaluation results into a single CSV row per checkpoint.

    Per-variant SI-SDR/SI-SDRi values are stored in columns like si_sdr_SER,
    si_sdri_SER, etc. `eval_context` supplies the eval-run provenance and must
    carry: git_sha, git_dirty, eval_data_root, eval_subset, eval_batch_size.
    The returned dict's keys are exactly FLATTEN_COLUMNS, in order.
    """
    model_config = info["model_config"]
    training_config = info["training_config"]
    data_config = info["data_config"]

    # Compute averages across all variants
    all_sisdrs = [r["si_sdr"] for r in variant_results.values()]
    all_sisdris = [r.get("si_sdri", 0.0) for r in variant_results.values()]
    avg_sisdr = sum(all_sisdrs) / len(all_sisdrs) if all_sisdrs else 0.0
    avg_sisdri = sum(all_sisdris) / len(all_sisdris) if all_sisdris else 0.0

    eval_data_root = eval_context.get("eval_data_root", "")
    eval_dataset_name = Path(eval_data_root).name if eval_data_root else ""

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
        # Averages across all variants
        "avg_sisdr": avg_sisdr,
        "avg_sisdri": avg_sisdri,
    }

    # Per-variant SI-SDR + SI-SDRi columns
    for variant in _VARIANTS:
        v = variant_results.get(variant, {})
        row[f"si_sdr_{variant}"] = v.get("si_sdr", "")
        row[f"si_sdri_{variant}"] = v.get("si_sdri", "")

    row.update({
        # Training config
        "lr": training_config.get("lr", ""),
        "batch_size": training_config.get("batch_size", ""),
        "epochs": training_config.get("epochs", ""),
        "optimizer": training_config.get("optimizer", ""),
        "scheduler": training_config.get("scheduler", ""),
        "grad_clip": training_config.get("grad_clip", ""),
        # Data config the checkpoint was TRAINED on (NOT the eval dataset)
        "train_segment_length": data_config.get("segment_length", ""),
        "train_sample_rate": data_config.get("sample_rate", ""),
        # Model config (flattened)
        "model_config": str(model_config),
        "evaluated_at": datetime.now().isoformat(),
        # Eval-run provenance
        "git_sha": eval_context.get("git_sha", ""),
        "git_dirty": eval_context.get("git_dirty", ""),
        "torch_version": torch.__version__,
        "eval_dataset_name": eval_dataset_name,
        "eval_data_root": eval_data_root,
        "eval_subset": eval_context.get("eval_subset", ""),
        "eval_batch_size": eval_context.get("eval_batch_size", ""),
    })

    return row


def resolve_output_target(output_path: Path, fieldnames) -> Path:
    """Return the path to append rows to, guarding against schema drift (gap 1).

    If ``output_path`` exists and its header's field-set differs from
    ``fieldnames`` (the current flatten_results schema), appending would produce
    exactly the mixed-schema file that had to be split into eras. In that case
    we refuse and return a new dated file beside it instead. If the schema
    matches (or the file is absent), we return ``output_path`` unchanged.
    """
    if not output_path.exists():
        return output_path
    with open(output_path, newline="") as f:
        try:
            existing = next(csv.reader(f))
        except StopIteration:
            existing = []
    if set(existing) == set(fieldnames):
        return output_path
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dated = output_path.with_name(f"{output_path.stem}_{ts}{output_path.suffix}")
    print(
        f"WARNING: {output_path} has {len(existing)} columns but the current eval "
        f"schema has {len(fieldnames)} — refusing to append (would corrupt it). "
        f"Writing to {dated} instead. Pass --output {dated} to keep appending there."
    )
    return dated


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
    parser.add_argument("--no-repeat", action="store_true",
                        help="Skip checkpoints whose per-checkpoint log already exists in --log-dir")
    parser.add_argument("--min-val-sisdr", type=float, default=3.0,
                        help="Skip checkpoints with val SI-SDR below this threshold (default: 3.0 dB)")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST,
                        help="Tracked eval manifest (display_name,checkpoint_path,dataset,notes). "
                             "Used by default when present; --csv-list overrides it.")
    parser.add_argument("--csv-list", default=None,
                        help="Evaluate only checkpoints listed in this semicolon-separated CSV "
                             "(requires a `path` column); overrides the manifest and --checkpoints-dir discovery")
    parser.add_argument("--log-dir", default="evaluate",
                        help="Directory for per-checkpoint evaluation logs "
                             "(default: evaluate/). One <model_type>__<run_name>.txt per checkpoint.")
    args = parser.parse_args()

    # Find checkpoints: explicit --csv-list wins, else the tracked manifest if it
    # exists, else glob the checkpoints tree.
    if args.csv_list:
        checkpoint_files = load_checkpoints_from_csv(args.csv_list)
        print(f"Loaded {len(checkpoint_files)} checkpoints from {args.csv_list}")
    elif Path(args.manifest).exists():
        checkpoint_files = load_checkpoints_from_manifest(args.manifest)
        print(f"Loaded {len(checkpoint_files)} checkpoints from manifest {args.manifest}")
    else:
        checkpoint_files = find_all_checkpoints(args.checkpoints_dir)
        print(f"Found {len(checkpoint_files)} checkpoints")

    if not checkpoint_files:
        print("No checkpoints found. Check --checkpoints-dir path.")
        return

    from evaluate import PER_SAMPLE_COLUMNS, per_sample_rows

    # Schema guard (gap 1): never append rows with a different column set into an
    # existing file — that is what produced the two-era corruption. On a schema
    # mismatch we divert to a dated sibling file instead of appending.
    output_path = resolve_output_target(Path(args.output), FLATTEN_COLUMNS)
    per_sample_path = output_path.with_name(f"{output_path.stem}_per_sample{output_path.suffix}")

    # Eval-run git provenance, computed once (identical for every checkpoint).
    git_info = git_provenance()

    # Load already-evaluated checkpoints if resuming (from the actual target).
    already_evaluated = set()
    if args.resume and output_path.exists():
        with open(output_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                already_evaluated.add(row["checkpoint_path"])
        print(f"Resuming: {len(already_evaluated)} checkpoints already evaluated")

    # Header is written only when the target file does not exist yet (the schema
    # guard guarantees an existing target already has the matching header).
    write_header = not output_path.exists()
    per_sample_header = not per_sample_path.exists()

    # Evaluate each checkpoint, appending results incrementally
    evaluated = 0
    skipped = 0
    log_skipped = 0
    filtered = 0
    failed = 0

    for i, ckpt_path in enumerate(checkpoint_files):
        ckpt_str = str(ckpt_path)

        if ckpt_str in already_evaluated:
            skipped += 1
            continue

        run_name = ckpt_path.parent.name
        model_type = ckpt_path.parts[-4]

        log_path = Path(args.log_dir) / f"{_safe_filename(model_type)}__{_safe_filename(run_name)}.txt"

        if args.no_repeat and log_path.exists():
            log_skipped += 1
            continue

        # Quick filter: check val_sisdr from checkpoint metadata without loading model
        ckpt_meta = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        val_sisdr = ckpt_meta.get("val_sisdr", 0.0)
        del ckpt_meta

        if val_sisdr < args.min_val_sisdr:
            filtered += 1
            continue

        print(f"\n[{i+1}/{len(checkpoint_files)}] {model_type}/{run_name} (val_sisdr={val_sisdr:.2f} dB)")

        try:
            with capture_to_file(log_path):
                print(f"Checkpoint: {ckpt_str}")
                print(f"Model: {model_type} | Run: {run_name} | Val SI-SDR: {val_sisdr:.2f} dB")
                info, num_params, results, eval_context = run_evaluation(
                    checkpoint_path=ckpt_str,
                    device=args.device,
                    max_samples=args.max_samples,
                    no_pesq=args.no_pesq,
                    no_stoi=args.no_stoi,
                )
                eval_context = {**eval_context, **git_info}

                row = flatten_results(info, num_params, results, eval_context)

                # Append aggregate row to CSV (fieldnames pinned to FLATTEN_COLUMNS
                # so the header order is stable across runs and matches the guard).
                with open(output_path, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=FLATTEN_COLUMNS)
                    if write_header:
                        writer.writeheader()
                        write_header = False
                    writer.writerow(row)

                # Append long-format per-sample rows for confidence intervals (gap 4).
                sample_rows = per_sample_rows(results, info["run_name"])
                if sample_rows:
                    with open(per_sample_path, "a", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=PER_SAMPLE_COLUMNS)
                        if per_sample_header:
                            writer.writeheader()
                            per_sample_header = False
                        writer.writerows(sample_rows)

                print(f"  -> avg SI-SDR: {row['avg_sisdr']:.2f} dB | avg SI-SDRi: {row['avg_sisdri']:.2f} dB")
                print(f"  -> log saved: {log_path}")
            evaluated += 1

        except Exception as e:
            print(f"  -> FAILED: {e}")
            failed += 1

        # Free GPU memory
        torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f"Done. Evaluated: {evaluated}, Skipped (resume): {skipped}, "
          f"Skipped (log exists): {log_skipped}, "
          f"Filtered (val_sisdr < {args.min_val_sisdr} dB): {filtered}, Failed: {failed}")
    print(f"Results saved to: {output_path}")
    if per_sample_path.exists():
        print(f"Per-sample scores: {per_sample_path}")


if __name__ == "__main__":
    main()
