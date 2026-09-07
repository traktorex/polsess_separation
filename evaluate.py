"""Evaluation script for PolSESS speech separation models (SI-SDR, PESQ, STOI)."""

from utils import warning_filters  # noqa: F401  must precede speechbrain imports (registers filters)

import logging
import torch
import argparse
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from tabulate import tabulate
from torchmetrics.audio import (
    ScaleInvariantSignalDistortionRatio,
    PerceptualEvaluationSpeechQuality,
    ShortTimeObjectiveIntelligibility,
)
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr

from datasets import (
    PolSESSDataset,
    Libri2MixDataset,
    polsess_collate_fn,
    libri2mix_collate_fn,
)
from config import Config, load_config_from_yaml
from utils import (
    apply_eps_patch,
    load_model_for_inference,
    count_parameters,
    compute_sisdr_and_sisdri,
    set_seed,
)

logger = logging.getLogger("polsess")

# Sampling rate the checkpoint was trained at, when its embedded config predates
# the `data.sample_rate` field (every checkpoint before 2026-09-07; all 8 kHz).
LEGACY_SAMPLE_RATE = 8000


def pesq_mode_for(sample_rate: int) -> str:
    """PESQ operating mode for a sampling rate: narrowband @ 8 kHz, wideband @ 16 kHz.

    These are the only two rates ITU-T P.862 defines, so anything else is an
    error rather than a silent nearest-match.
    """
    if sample_rate == 8000:
        return "nb"
    if sample_rate == 16000:
        return "wb"
    raise ValueError(f"PESQ is defined for 8 kHz and 16 kHz only, got {sample_rate} Hz")


def checkpoint_sample_rate(checkpoint: dict) -> int:
    """Sampling rate a checkpoint was trained at, from its embedded config."""
    data_cfg = (checkpoint.get("config") or {}).get("data") or {}
    return int(data_cfg.get("sample_rate", LEGACY_SAMPLE_RATE))

# Long-format per-sample CSV schema (one row per evaluated sample).
PER_SAMPLE_COLUMNS = ("run", "variant", "sample_idx", "si_sdr", "si_sdri", "pesq", "stoi")


def evaluate_model(
    model,
    dataloader: DataLoader,
    device: str = "cuda",
    compute_pesq: bool = True,
    compute_stoi: bool = True,
    use_amp: bool = False,
    task: str = "ES",
    sample_rate: int = LEGACY_SAMPLE_RATE,
) -> dict:
    """Evaluate model on a dataset and compute metrics.

    ``sample_rate`` is the rate of the audio the dataloader yields; it selects
    the PESQ mode (nb/wb) and the STOI rate. SI-SDR is rate-free.

    Scores are accumulated per batch and averaged; callers use ``batch_size=1``
    (see ``evaluate_by_variant`` and the Libri2Mix path) so every accumulated
    scalar is a single-sample score and the mean is an exact per-sample mean
    (no mean-of-batch-means bias, survey gap 3). The returned ``per_sample``
    list carries the individual scores for downstream confidence intervals.
    """
    si_sdr_metric = ScaleInvariantSignalDistortionRatio().to(device)

    # For SB task, use PIT-based SI-SDR
    pit_sisdr = None
    if task == "SB":
        pit_sisdr = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx").to(device)

    pesq_metric = None
    stoi_metric = None
    if compute_pesq:
        pesq_metric = PerceptualEvaluationSpeechQuality(
            sample_rate, pesq_mode_for(sample_rate)
        ).to(device)
    if compute_stoi:
        stoi_metric = ShortTimeObjectiveIntelligibility(sample_rate).to(device)

    si_sdr_scores = []
    si_sdri_scores = []
    pesq_scores = []
    stoi_scores = []
    pesqi_scores = []
    stoii_scores = []
    per_sample = []
    pesq_failures = 0

    model.eval()
    with torch.no_grad():
        for sample_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating", leave=False)):
            mix = batch["mix"].to(device)
            clean = batch["clean"].to(device)

            if use_amp and device == "cuda":
                with torch.amp.autocast("cuda"):
                    mix_input = mix.unsqueeze(1)
                    estimates = model(mix_input)
            else:
                mix_input = mix.unsqueeze(1)
                estimates = model(mix_input)

            # Trim to same length
            min_len = min(estimates.shape[-1], clean.shape[-1])
            estimates = estimates[..., :min_len]
            clean = clean[..., :min_len]
            mix_trimmed = mix[..., :min_len]

            # SI-SDR / SI-SDRi via the shared helper (same code the trainer uses).
            # For SB, `aligned` is the PIT-reordered estimates matched to `clean`;
            # for enhancement it is the (channel-squeezed) estimates.
            si_sdr, si_sdri, aligned = compute_sisdr_and_sisdri(
                estimates, clean, mix_trimmed, task, si_sdr_metric, pit_loss=pit_sisdr
            )
            si_sdr_scores.append(si_sdr)
            si_sdri_scores.append(si_sdri)

            sample_pesq = None
            sample_stoi = None

            if task == "SB":
                # PESQ and STOI on PIT-reordered estimates
                if pesq_metric:
                    pesq_sum = 0.0
                    pesq_mix_sum = 0.0
                    pesq_count = 0
                    for spk in range(clean.shape[1]):
                        for est, ref, mx in zip(aligned[:, spk], clean[:, spk], mix_trimmed):
                            try:
                                p = pesq_metric(est.unsqueeze(0), ref.unsqueeze(0))
                                p_mix = pesq_metric(mx.unsqueeze(0), ref.unsqueeze(0))
                                if not (torch.isnan(p) or torch.isinf(p)
                                        or torch.isnan(p_mix) or torch.isinf(p_mix)):
                                    pesq_sum += p.item()
                                    pesq_mix_sum += p_mix.item()
                                    pesq_count += 1
                                else:
                                    pesq_failures += 1
                            except Exception as e:
                                pesq_failures += 1
                                logger.debug(f"PESQ computation failed for sample: {e}")
                    if pesq_count > 0:
                        avg_pesq = pesq_sum / pesq_count
                        avg_pesq_mix = pesq_mix_sum / pesq_count
                        pesq_scores.append(avg_pesq)
                        pesqi_scores.append(avg_pesq - avg_pesq_mix)
                        sample_pesq = avg_pesq

                if stoi_metric:
                    stoi_sum = 0.0
                    stoi_mix_sum = 0.0
                    for spk in range(clean.shape[1]):
                        stoi_sum += stoi_metric(aligned[:, spk], clean[:, spk]).item()
                        stoi_mix_sum += stoi_metric(mix_trimmed, clean[:, spk]).item()
                    avg_stoi = stoi_sum / clean.shape[1]
                    avg_stoi_mix = stoi_mix_sum / clean.shape[1]
                    stoi_scores.append(avg_stoi)
                    stoii_scores.append(avg_stoi - avg_stoi_mix)
                    sample_stoi = avg_stoi
            else:
                # Enhancement: `aligned` is the squeezed estimates; squeeze clean to match.
                clean_sq = clean.squeeze(1) if (clean.dim() == 3 and clean.shape[1] == 1) else clean

                if pesq_metric:
                    for est, ref in zip(aligned, clean_sq):
                        try:
                            pesq = pesq_metric(est.unsqueeze(0), ref.unsqueeze(0))
                            if not torch.isnan(pesq) and not torch.isinf(pesq):
                                pesq_scores.append(pesq.item())
                                sample_pesq = pesq.item()
                            else:
                                pesq_failures += 1
                        except Exception as e:
                            pesq_failures += 1
                            logger.debug(f"PESQ computation failed for sample: {e}")

                if stoi_metric:
                    stoi = stoi_metric(aligned, clean_sq)
                    stoi_scores.append(stoi.item())
                    sample_stoi = stoi.item()

            per_sample.append({
                "sample_idx": sample_idx,
                "si_sdr": si_sdr,
                "si_sdri": si_sdri,
                "pesq": sample_pesq,
                "stoi": sample_stoi,
            })

    results = {
        "si_sdr": sum(si_sdr_scores) / len(si_sdr_scores) if si_sdr_scores else 0,
        "si_sdri": sum(si_sdri_scores) / len(si_sdri_scores) if si_sdri_scores else 0,
        "num_samples": len(dataloader.dataset),
        "pesq_failures": pesq_failures,
        "per_sample": per_sample,
    }

    if pesq_scores:
        results["pesq"] = sum(pesq_scores) / len(pesq_scores)
    if stoi_scores:
        results["stoi"] = sum(stoi_scores) / len(stoi_scores)
    if pesqi_scores:
        results["pesqi"] = sum(pesqi_scores) / len(pesqi_scores)
    if stoii_scores:
        results["stoii"] = sum(stoii_scores) / len(stoii_scores)

    return results


def bootstrap_ci(values, ci: float = 0.95, n_resamples: int = 10000, seed: int = 0):
    """Mean and percentile bootstrap CI for a 1-D sample (thesis-table helper).

    ``None`` entries are dropped first. Returns ``(mean, lo, hi)``:
    ``(nan, nan, nan)`` for an empty sample, ``(mean, mean, mean)`` for n == 1.
    """
    import numpy as np

    vals = np.asarray([v for v in values if v is not None], dtype=float)
    if vals.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(vals.mean())
    if vals.size == 1:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, vals.size, size=(n_resamples, vals.size))
    boot_means = vals[idx].mean(axis=1)
    lo = float(np.percentile(boot_means, (1 - ci) / 2 * 100))
    hi = float(np.percentile(boot_means, (1 + ci) / 2 * 100))
    return mean, lo, hi


def per_sample_rows(results_by_variant: dict, run: str):
    """Flatten ``{variant: eval-result-dict}`` into long-format per-sample rows.

    Each row has the ``PER_SAMPLE_COLUMNS`` fields. Variants whose result dict
    carries no ``per_sample`` list (e.g. older callers) contribute nothing.
    """
    rows = []
    for variant, res in results_by_variant.items():
        for rec in res.get("per_sample", []):
            rows.append({
                "run": run,
                "variant": variant,
                "sample_idx": rec["sample_idx"],
                "si_sdr": rec["si_sdr"],
                "si_sdri": rec["si_sdri"],
                "pesq": rec.get("pesq"),
                "stoi": rec.get("stoi"),
            })
    return rows


def summarize_per_sample(csv_path, metrics=("si_sdr", "si_sdri", "pesq", "stoi"), ci: float = 0.95):
    """Mean ± bootstrap-CI table per (run, variant, metric) from a per-sample CSV.

    Reads the long-format CSV written during evaluation and returns a DataFrame
    with columns ``run, variant, metric, n, mean, ci_lo, ci_hi`` for the thesis
    architecture-comparison tables.
    """
    df = pd.read_csv(csv_path)
    rows = []
    for (run, variant), group in df.groupby(["run", "variant"]):
        for metric in metrics:
            if metric not in group.columns:
                continue
            vals = group[metric].dropna().tolist()
            if not vals:
                continue
            mean, lo, hi = bootstrap_ci(vals, ci=ci)
            rows.append({
                "run": run,
                "variant": variant,
                "metric": metric,
                "n": len(vals),
                "mean": mean,
                "ci_lo": lo,
                "ci_hi": hi,
            })
    return pd.DataFrame(rows)


def evaluate_by_variant(
    model,
    config: Config,
    device: str = "cuda",
    compute_pesq: bool = True,
    compute_stoi: bool = True,
    specific_variant: str = None,
    max_samples: int = None,
) -> dict:
    """Evaluate model on each MM-IPC variant separately.

    Evaluation always runs at ``batch_size=1`` so every metric is a per-sample
    score: this removes the mean-of-batch-means bias (survey gap 3) and yields
    the ``per_sample`` records used for confidence intervals.
    """
    indoor_variants = ["SER", "SR", "ER", "R"]
    outdoor_variants = ["SE", "S", "E", "C"]
    all_variants = indoor_variants + outdoor_variants

    if specific_variant:
        if specific_variant not in all_variants:
            raise ValueError(f"Unknown variant: {specific_variant}")
        variants_to_test = [specific_variant]
    else:
        variants_to_test = all_variants

    if config.data.dataset_type != "polsess":
        raise ValueError(
            f"Dataset {config.data.dataset_type} not supported for variant evaluation"
        )

    data_root = config.data.polsess.data_root
    results = {}

    for variant in variants_to_test:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating variant: {variant}")
        logger.info(f"{'='*60}")

        dataset = PolSESSDataset(
            data_root,
            subset="test",
            task=config.data.task,
            allowed_variants=[variant],
            max_samples=max_samples,
            sample_rate=config.data.sample_rate,  # guard: corpus rate must match
        )

        dataloader = DataLoader(
            dataset,
            batch_size=1,  # per-sample scoring (gap 3); see docstring
            shuffle=False,
            num_workers=config.data.num_workers,
            collate_fn=polsess_collate_fn,
        )

        variant_results = evaluate_model(
            model,
            dataloader,
            device,
            compute_pesq=compute_pesq,
            compute_stoi=compute_stoi,
            use_amp=False,
            task=config.data.task,
            sample_rate=config.data.sample_rate,
        )

        results[variant] = variant_results

        logger.info(f"{variant} Results:")
        logger.info(f"  SI-SDR: {variant_results['si_sdr']:.2f} dB")
        logger.info(f"  SI-SDRi: {variant_results['si_sdri']:.2f} dB")
        if variant_results.get("pesq_failures"):
            logger.warning(
                f"  PESQ failed on {variant_results['pesq_failures']} sample(s) "
                f"in variant {variant} (excluded from the PESQ mean)"
            )
        if "pesq" in variant_results:
            pesq_str = f"  PESQ: {variant_results['pesq']:.2f}"
            if "pesqi" in variant_results:
                pesq_str += f" (PESQi: {variant_results['pesqi']:+.2f})"
            logger.info(pesq_str)
        if "stoi" in variant_results:
            stoi_str = f"  STOI: {variant_results['stoi']:.3f}"
            if "stoii" in variant_results:
                stoi_str += f" (STOIi: {variant_results['stoii']:+.3f})"
            logger.info(stoi_str)
        logger.info(f"  Samples: {variant_results['num_samples']}")

    return results


def print_summary(results: dict):
    """Print summary table of results."""
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)

    has_pesq = any("pesq" in r for r in results.values())
    has_stoi = any("stoi" in r for r in results.values())
    has_pesqi = any("pesqi" in r for r in results.values())
    has_stoii = any("stoii" in r for r in results.values())

    headers = ["Variant", "SI-SDR (dB)", "SI-SDRi (dB)"]
    if has_pesq:
        headers.append("PESQ")
    if has_pesqi:
        headers.append("PESQi")
    if has_stoi:
        headers.append("STOI")
    if has_stoii:
        headers.append("STOIi")
    headers.append("Samples")

    table_data = []
    for variant, metrics in sorted(results.items()):
        row = [variant, f"{metrics['si_sdr']:.2f}", f"{metrics['si_sdri']:.2f}"]
        if has_pesq:
            row.append(f"{metrics['pesq']:.2f}" if "pesq" in metrics else "N/A")
        if has_pesqi:
            row.append(f"{metrics['pesqi']:+.2f}" if "pesqi" in metrics else "N/A")
        if has_stoi:
            row.append(f"{metrics['stoi']:.3f}" if "stoi" in metrics else "N/A")
        if has_stoii:
            row.append(f"{metrics['stoii']:+.3f}" if "stoii" in metrics else "N/A")
        row.append(metrics["num_samples"])
        table_data.append(row)

    if len(results) > 1:
        avg_row = [
            "AVERAGE",
            f"{sum(r['si_sdr'] for r in results.values()) / len(results):.2f}",
            f"{sum(r['si_sdri'] for r in results.values()) / len(results):.2f}",
        ]
        if has_pesq:
            pesq_values = [r["pesq"] for r in results.values() if "pesq" in r]
            avg_row.append(
                f"{sum(pesq_values) / len(pesq_values):.2f}" if pesq_values else "N/A"
            )
        if has_pesqi:
            pesqi_values = [r["pesqi"] for r in results.values() if "pesqi" in r]
            avg_row.append(
                f"{sum(pesqi_values) / len(pesqi_values):+.2f}" if pesqi_values else "N/A"
            )
        if has_stoi:
            stoi_values = [r["stoi"] for r in results.values() if "stoi" in r]
            avg_row.append(
                f"{sum(stoi_values) / len(stoi_values):.3f}" if stoi_values else "N/A"
            )
        if has_stoii:
            stoii_values = [r["stoii"] for r in results.values() if "stoii" in r]
            avg_row.append(
                f"{sum(stoii_values) / len(stoii_values):+.3f}" if stoii_values else "N/A"
            )
        avg_row.append("")
        table_data.append(avg_row)

    print(tabulate(table_data, headers=headers, tablefmt="simple"))
    print("=" * 80)


def save_results_csv(results: dict, output_path: str):
    """Save results to CSV file."""
    rows = []
    for variant, metrics in results.items():
        row = {
            "variant": variant,
            "si_sdr_db": metrics["si_sdr"],
            "si_sdri_db": metrics["si_sdri"],
            "num_samples": metrics["num_samples"],
        }
        if "pesq" in metrics:
            row["pesq"] = metrics["pesq"]
        if "pesqi" in metrics:
            row["pesqi"] = metrics["pesqi"]
        if "stoi" in metrics:
            row["stoi"] = metrics["stoi"]
        if "stoii" in metrics:
            row["stoii"] = metrics["stoii"]
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")


def main():
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(description="Evaluate speech separation model")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--config", default=None, help="Path to YAML config file")
    parser.add_argument("--dataset", default="polsess", choices=["polsess", "librimix"])
    parser.add_argument("--data-root", help="Root directory of dataset")
    parser.add_argument("--task", choices=["ES", "EB", "SB"], help="Task type")
    parser.add_argument("--variant", help="Specific MM-IPC variant to test (polsess only)")
    parser.add_argument("--librimix-root", help="Path to Libri2Mix root directory")
    parser.add_argument("--librimix-subset", default="test", choices=["test", "dev", "train-100"])
    parser.add_argument("--mix-type", choices=["mix_clean", "mix_both"],
                        help="Libri2Mix variant (default: evaluate both)")
    parser.add_argument("--max-samples", type=int, help="Limit number of samples per variant")
    parser.add_argument("--batch-size", type=int,
                        help="(ignored) evaluation always runs at batch_size=1 for per-sample scoring")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--no-pesq", action="store_true", help="Skip PESQ")
    parser.add_argument("--no-stoi", action="store_true", help="Skip STOI")
    parser.add_argument("--output", help="Output CSV file")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Deterministic evaluation: seed all RNGs before any dataset/model work so
    # variant selection and any stochastic op are reproducible across runs
    # (survey gap 6). Eval always forces a single variant per pass, but this
    # also pins the deterministic cuDNN path via set_seed.
    set_seed()

    # Load config
    if args.config:
        logger.info(f"Loading config from: {args.config}")
        config = load_config_from_yaml(args.config)
    else:
        config = Config()

    # Apply CLI overrides
    if args.data_root and args.dataset == "polsess":
        config.data.polsess.data_root = args.data_root
    if args.task:
        config.data.task = args.task
    # --batch-size is intentionally not applied: evaluation forces batch_size=1
    # for per-sample scoring (see evaluate_by_variant / the Libri2Mix path).

    # Apply EPS patch if using AMP
    if config.training.use_amp:
        apply_eps_patch(config.training.amp_eps)

    # Load model
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    model, checkpoint = load_model_for_inference(args.checkpoint, device)

    # Log checkpoint info
    ckpt_config = checkpoint.get("config", {})
    model_type = ckpt_config.get("model", {}).get("model_type", "unknown")
    logger.info(f"Model loaded: {model_type}")
    logger.info(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    if "val_sisdr" in checkpoint:
        logger.info(f"  Validation SI-SDR: {checkpoint['val_sisdr']:.2f} dB")
    logger.info(f"  Parameters: {count_parameters(model) / 1e6:.2f}M")

    # Auto-detect task from checkpoint config if not explicitly set
    if not args.task and not args.config:
        ckpt_task = ckpt_config.get("data", {}).get("task")
        if ckpt_task and ckpt_task != config.data.task:
            logger.info(f"Auto-detected task from checkpoint: {ckpt_task}")
            config.data.task = ckpt_task

    # The sampling rate always follows the checkpoint: it is a property of the
    # trained model, not of the eval invocation. Checkpoints saved before the
    # field existed are all 8 kHz.
    config.data.sample_rate = checkpoint_sample_rate(checkpoint)
    logger.info(f"  Sample rate: {config.data.sample_rate} Hz")

    # Run evaluation
    if args.dataset == "polsess":
        results = evaluate_by_variant(
            model,
            config,
            device,
            compute_pesq=not args.no_pesq,
            compute_stoi=not args.no_stoi,
            specific_variant=args.variant,
            max_samples=args.max_samples,
        )
    else:
        # LibriMix evaluation (2-speaker separation)
        if not args.librimix_root:
            raise ValueError(
                "--librimix-root is required when --dataset librimix is used"
            )

        # Libri2Mix is always 2-speaker separation
        config.data.task = "SB"
        config.data.batch_size = 1

        # Evaluate specified mix_type, or both if not specified
        mix_types = [args.mix_type] if args.mix_type else ["mix_clean", "mix_both"]
        results = {}

        for mix_type in mix_types:
            variant_name = "Libri2Mix-Clean" if mix_type == "mix_clean" else "Libri2Mix-Noisy"
            logger.info(f"\n{'='*60}")
            logger.info(f"Evaluating: {variant_name}")
            logger.info(f"{'='*60}")

            dataset = Libri2MixDataset(
                args.librimix_root,
                subset=args.librimix_subset,
                sample_rate=config.data.sample_rate,  # wav8k/ or wav16k/
                mix_type=mix_type,
                max_samples=args.max_samples,
            )

            dataloader = DataLoader(
                dataset,
                batch_size=config.data.batch_size,
                shuffle=False,
                num_workers=config.data.num_workers,
                collate_fn=libri2mix_collate_fn,
            )

            result = evaluate_model(
                model,
                dataloader,
                device,
                compute_pesq=not args.no_pesq,
                compute_stoi=not args.no_stoi,
                use_amp=False,
                task="SB",
                sample_rate=config.data.sample_rate,
            )

            results[variant_name] = result
            logger.info(f"{variant_name} SI-SDR: {result['si_sdr']:.2f} dB | SI-SDRi: {result['si_sdri']:.2f} dB")

    # Print and save results
    print_summary(results)

    if args.output:
        save_results_csv(results, args.output)


if __name__ == "__main__":
    main()
