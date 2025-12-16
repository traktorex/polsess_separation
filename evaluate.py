"""Evaluation script for PolSESS speech separation models (SI-SDR, PESQ, STOI)."""

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
from models import get_model
from config import Config, load_config_from_yaml
from utils import apply_eps_patch, load_checkpoint_file, count_parameters


def load_model_from_checkpoint(
    checkpoint_path: str, config: Config = None, device: str = "cuda"
) -> torch.nn.Module:
    """Load trained model from checkpoint for evaluation.

    If checkpoint contains config, uses it. Otherwise uses provided config.
    """
    print(f"Loading model from {checkpoint_path}...")

    checkpoint = load_checkpoint_file(checkpoint_path, device)

    # Try to get model config from checkpoint first
    if "config" in checkpoint:
        print("Using model config from checkpoint")
        ckpt_config = checkpoint["config"]
        model_type = ckpt_config.get("model", {}).get("model_type", "convtasnet")
        model_class = get_model(model_type)

        # Get model-specific parameters
        model_params_dict = ckpt_config.get("model", {}).get(model_type, {})
        if not model_params_dict:
            model_params_dict = ckpt_config.get("model", {})  # Legacy flat config

        model = model_class(**model_params_dict)
    else:
        if config is None:
            raise ValueError("No config in checkpoint and no config provided")
        print("Using provided config")
        model_type = config.model.model_type
        model_class = get_model(model_type)
        model_params = getattr(config.model, model_type, None)

        if model_params is None:
            raise ValueError(f"Config doesn't contain parameters for '{model_type}'")

        model_params_dict = vars(model_params)
        model = model_class(**model_params_dict)

    # Load weights and prepare for eval
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Log info
    model_type_str = checkpoint.get("config", {}).get("model", {}).get(
        "model_type", model_type if config else "unknown"
    )
    print(f"Model loaded: {model_type_str}")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    if "val_sisdr" in checkpoint:
        print(f"  Validation SI-SDR: {checkpoint['val_sisdr']:.2f} dB")
    print(f"  Parameters: {count_parameters(model) / 1e6:.2f}M")

    return model


def evaluate_model(
    model,
    dataloader: DataLoader,
    device: str = "cuda",
    compute_pesq: bool = True,
    compute_stoi: bool = True,
    use_amp: bool = False,
    task: str = "ES",
) -> dict:
    """Evaluate model on a dataset and compute metrics."""
    si_sdr_metric = ScaleInvariantSignalDistortionRatio().to(device)

    # For SB task, use PIT-based SI-SDR
    if task == "SB":
        pit_sisdr = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx").to(device)

    pesq_metric = None
    stoi_metric = None
    if compute_pesq:
        pesq_metric = PerceptualEvaluationSpeechQuality(8000, "nb").to(device)
    if compute_stoi:
        stoi_metric = ShortTimeObjectiveIntelligibility(8000).to(device)

    si_sdr_scores = []
    pesq_scores = []
    stoi_scores = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
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

            # Compute SI-SDR based on task
            if task == "SB":
                # Speaker separation: use PIT-based SI-SDR
                loss = pit_sisdr(estimates, clean)
                si_sdr = -loss
                si_sdr_scores.append(si_sdr.item())
            else:
                # Enhancement: standard SI-SDR
                if clean.dim() == 3 and clean.shape[1] == 1:
                    clean = clean.squeeze(1)
                if estimates.dim() == 3 and estimates.shape[1] == 1:
                    estimates = estimates.squeeze(1)

                si_sdr = si_sdr_metric(estimates, clean)
                si_sdr_scores.append(si_sdr.item())

            # PESQ and STOI only for enhancement tasks
            if task != "SB":
                if pesq_metric:
                    for est, ref in zip(estimates, clean):
                        try:
                            pesq = pesq_metric(est.unsqueeze(0), ref.unsqueeze(0))
                            if not torch.isnan(pesq) and not torch.isinf(pesq):
                                pesq_scores.append(pesq.item())
                        except Exception:
                            pass

                if stoi_metric:
                    stoi = stoi_metric(estimates, clean)
                    stoi_scores.append(stoi.item())

    results = {
        "si_sdr": sum(si_sdr_scores) / len(si_sdr_scores) if si_sdr_scores else 0,
        "num_samples": len(dataloader.dataset),
    }

    if pesq_scores:
        results["pesq"] = sum(pesq_scores) / len(pesq_scores)
    if stoi_scores:
        results["stoi"] = sum(stoi_scores) / len(stoi_scores)

    return results


def evaluate_by_variant(
    model,
    config: Config,
    device: str = "cuda",
    batch_size: int = 4,
    compute_pesq: bool = True,
    compute_stoi: bool = True,
    specific_variant: str = None,
) -> dict:
    """Evaluate model on each MM-IPC variant separately."""
    indoor_variants = ["SER", "SR", "ER", "R", "C"]
    outdoor_variants = ["SE", "S", "E", "C"]
    all_variants = indoor_variants + outdoor_variants

    if specific_variant:
        if specific_variant not in all_variants:
            raise ValueError(f"Unknown variant: {specific_variant}")
        variants_to_test = [specific_variant]
    else:
        variants_to_test = all_variants

    results = {}

    for variant in variants_to_test:
        print(f"\n{'='*60}")
        print(f"Evaluating variant: {variant}")
        print(f"{'='*60}")

        # Get data_root
        if config.data.dataset_type == "polsess":
            data_root = config.data.polsess.data_root
        else:
            raise ValueError(
                f"Dataset {config.data.dataset_type} not supported for variant evaluation"
            )

        dataset = PolSESSDataset(
            data_root,
            subset="test",
            task=config.data.task,
            allowed_variants=[variant],
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
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
        )

        results[variant] = variant_results

        print(f"\n{variant} Results:")
        print(f"  SI-SDR: {variant_results['si_sdr']:.2f} dB")
        if "pesq" in variant_results:
            print(f"  PESQ: {variant_results['pesq']:.2f}")
        if "stoi" in variant_results:
            print(f"  STOI: {variant_results['stoi']:.3f}")
        print(f"  Samples: {variant_results['num_samples']}")

    return results


def print_summary(results: dict):
    """Print summary table of results."""
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)

    has_pesq = any("pesq" in r for r in results.values())
    has_stoi = any("stoi" in r for r in results.values())

    headers = ["Variant", "SI-SDR (dB)"]
    if has_pesq:
        headers.append("PESQ")
    if has_stoi:
        headers.append("STOI")
    headers.append("Samples")

    table_data = []
    for variant, metrics in sorted(results.items()):
        row = [variant, f"{metrics['si_sdr']:.2f}"]
        if has_pesq:
            row.append(f"{metrics['pesq']:.2f}" if "pesq" in metrics else "N/A")
        if has_stoi:
            row.append(f"{metrics['stoi']:.3f}" if "stoi" in metrics else "N/A")
        row.append(metrics["num_samples"])
        table_data.append(row)

    if len(results) > 1:
        avg_row = [
            "AVERAGE",
            f"{sum(r['si_sdr'] for r in results.values()) / len(results):.2f}",
        ]
        if has_pesq:
            pesq_values = [r["pesq"] for r in results.values() if "pesq" in r]
            avg_row.append(
                f"{sum(pesq_values) / len(pesq_values):.2f}" if pesq_values else "N/A"
            )
        if has_stoi:
            stoi_values = [r["stoi"] for r in results.values() if "stoi" in r]
            avg_row.append(
                f"{sum(stoi_values) / len(stoi_values):.3f}" if stoi_values else "N/A"
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
            "num_samples": metrics["num_samples"],
        }
        if "pesq" in metrics:
            row["pesq"] = metrics["pesq"]
        if "stoi" in metrics:
            row["stoi"] = metrics["stoi"]
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
    parser.add_argument("--variant", help="Specific MM-IPC variant to test")
    parser.add_argument("--librimix-root", help="Path to Libri2Mix root")
    parser.add_argument("--librimix-subset", default="test", choices=["test", "dev", "train-100"])
    parser.add_argument("--max-samples", type=int, help="Limit number of samples")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--no-pesq", action="store_true", help="Skip PESQ")
    parser.add_argument("--no-stoi", action="store_true", help="Skip STOI")
    parser.add_argument("--output", help="Output CSV file")

    args = parser.parse_args()

    # Load config
    if args.config:
        print(f"Loading config from: {args.config}")
        config = load_config_from_yaml(args.config)
    else:
        config = Config()

    # Apply CLI overrides
    if args.data_root:
        if args.dataset == "polsess":
            config.data.polsess.data_root = args.data_root
        elif args.dataset == "librimix":
            pass  # Handle separately below
    if args.task:
        config.data.task = args.task
    if args.batch_size:
        config.data.batch_size = args.batch_size

    # Apply EPS patch if using AMP
    if config.training.use_amp:
        apply_eps_patch(config.training.amp_eps)

    # Load model
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    model = load_model_from_checkpoint(args.checkpoint, config, device)

    # Determine evaluation mode
    if args.variant or (args.dataset == "polsess" and not args.variant):
        # PolSESS variant evaluation
        results = evaluate_by_variant(
            model,
            config,
            device,
            batch_size=config.data.batch_size,
            compute_pesq=not args.no_pesq,
            compute_stoi=not args.no_stoi,
            specific_variant=args.variant,
        )
    else:
        # Single dataset evaluation (LibriMix or simple PolSESS)
        if args.dataset == "librimix":
            librimix_root = args.librimix_root or config.data.librimix.data_root
            dataset = Libri2MixDataset(
                librimix_root,
                subset=args.librimix_subset,
                max_samples=args.max_samples,
            )
            collate_fn = libri2mix_collate_fn
        else:
            dataset = PolSESSDataset(
                config.data.polsess.data_root,
                subset="test",
                task=config.data.task,
                max_samples=args.max_samples,
            )
            collate_fn = polsess_collate_fn

        dataloader = DataLoader(
            dataset,
            batch_size=config.data.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
            collate_fn=collate_fn,
        )

        result = evaluate_model(
            model,
            dataloader,
            device,
            compute_pesq=not args.no_pesq,
            compute_stoi=not args.no_stoi,
            use_amp=False,
            task=config.data.task,
        )

        results = {args.dataset: result}

    # Print and save results
    print_summary(results)

    if args.output:
        save_results_csv(results, args.output)


if __name__ == "__main__":
    main()
