"""Evaluation script for PolSESS speech separation models (SI-SDR, PESQ, STOI)."""

import torch
import argparse
import pandas as pd
from tqdm import tqdm
from tabulate import tabulate
from torch.utils.data import DataLoader
from torchmetrics.audio import (
    ScaleInvariantSignalDistortionRatio,
    PerceptualEvaluationSpeechQuality,
    ShortTimeObjectiveIntelligibility,
)

from datasets import (
    PolSESSDataset,
    Libri2MixDataset,
    polsess_collate_fn,
    libri2mix_collate_fn,
)
from models import get_model
from config import Config, load_config_from_yaml
from utils import apply_eps_patch


def load_model(checkpoint_path: str, config: Config, device: str = "cuda"):
    """Load trained model from checkpoint.

    If checkpoint contains config, uses it. Otherwise uses provided config.
    """
    print(f"Loading model from {checkpoint_path}...")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "config" in checkpoint:
        print("Using model config from checkpoint")
        ckpt_config = checkpoint["config"]
        model_type = ckpt_config.get("model", {}).get("model_type", "convtasnet")
        model_class = get_model(model_type)

        # Get model-specific parameters from nested config
        model_params_dict = ckpt_config.get("model", {}).get(model_type, {})

        # If nested config doesn't exist, try flat config (legacy)
        if not model_params_dict:
            model_params_dict = ckpt_config.get("model", {})

        # Create model with parameters
        print(f"Creating {model_type} model from checkpoint config")
        model = model_class(**model_params_dict)
    else:
        print("Using provided config (checkpoint doesn't contain config)")
        model_type = config.model.model_type
        model_class = get_model(model_type)

        # Get model-specific parameters
        model_params = getattr(config.model, model_type, None)

        if model_params is None:
            raise ValueError(
                f"Config doesn't contain parameters for model type '{model_type}'. "
                f"Expected config.model.{model_type} to exist."
            )

        # Convert params to dict
        if hasattr(model_params, "__dict__"):
            model_params_dict = vars(model_params)
        else:
            raise ValueError(f"Invalid model params for {model_type}")

        print(f"Creating {model_type} model from provided config")
        model = model_class(**model_params_dict)

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"Model loaded: {model_type}")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    if "val_sisdr" in checkpoint:
        print(f"  Validation SI-SDR: {checkpoint['val_sisdr']:.2f} dB")

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params / 1e6:.2f}M")

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
    """Evaluate model on a dataset.

    Args:
        model: Model to evaluate
        dataloader: DataLoader for evaluation
        device: Device to use
        compute_pesq: Whether to compute PESQ
        compute_stoi: Whether to compute STOI
        use_amp: Whether to use AMP (disabled by default for stability)
        task: Task type ("ES", "EB", "SB") for proper metric computation
    """
    from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr

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

            # Trim to same length (handle encoder-decoder length mismatch)
            min_len = min(estimates.shape[-1], clean.shape[-1])
            estimates = estimates[..., :min_len]
            clean = clean[..., :min_len]

            # Compute SI-SDR based on task
            if task == "SB":
                # Speaker separation: use PIT-based SI-SDR
                # clean: [B, 2, T], estimates: [B, 2, T]
                loss = pit_sisdr(estimates, clean)
                si_sdr = -loss  # Convert negative loss back to positive SI-SDR
                si_sdr_scores.append(si_sdr.item())
            else:
                # Enhancement: standard SI-SDR
                # Handle both [B, T] and [B, 1, T] formats
                if clean.dim() == 3 and clean.shape[1] == 1:
                    clean = clean.squeeze(1)
                if estimates.dim() == 3 and estimates.shape[1] == 1:
                    estimates = estimates.squeeze(1)

                si_sdr = si_sdr_metric(estimates, clean)
                si_sdr_scores.append(si_sdr.item())

            # PESQ and STOI only for enhancement tasks (single channel)
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
            raise ValueError(
                f"Unknown variant: {specific_variant}. "
                f"Valid variants: {', '.join(all_variants)}"
            )
        variants_to_test = [specific_variant]
    else:
        variants_to_test = all_variants

    results = {}

    for variant in variants_to_test:
        print(f"\n{'='*60}")
        print(f"Evaluating variant: {variant}")
        print(f"{'='*60}")

        # Get data_root from nested config
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
            use_amp=False,  # Disabled for stability
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


def create_evaluation_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate speech separation model")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file (optional)"
    )

    # Dataset selection
    parser.add_argument(
        "--dataset",
        type=str,
        default="polsess",
        choices=["polsess", "librimix"],
        help="Dataset to evaluate on: polsess or librimix",
    )

    # PolSESS arguments
    parser.add_argument(
        "--data-root", type=str, default=None, help="Root directory of PolSESS dataset"
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        choices=["ES", "EB", "SB"],
        help="Task: ES (single speaker), EB (both speakers), or SB (separate both)",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Specific MM-IPC variant to test (SER, SR, ER, R, SE, S, E)",
    )

    # Libri2Mix arguments
    parser.add_argument(
        "--librimix-root",
        type=str,
        default=None,
        help="Path to Libri2Mix root (e.g., C:/datasety/LibriMix/generated/Libri2Mix)",
    )
    parser.add_argument(
        "--librimix-subset",
        type=str,
        default="test",
        choices=["test", "dev", "train-100"],
        help="Libri2Mix subset to evaluate",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of samples (for quick testing)",
    )

    # Common arguments
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use",
    )
    parser.add_argument(
        "--no-pesq", action="store_true", help="Skip PESQ computation (faster)"
    )
    parser.add_argument(
        "--no-stoi", action="store_true", help="Skip STOI computation (faster)"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output CSV file for results"
    )

    return parser


def load_config_and_apply_overrides(args: argparse.Namespace) -> tuple[Config, int]:
    """Load config from file or create default, apply CLI overrides."""
    if args.config:
        print(f"Loading config from: {args.config}")
        config = load_config_from_yaml(args.config)
    else:
        config = Config()

    # Apply CLI overrides to nested config
    if args.data_root:
        if config.data.dataset_type == "polsess":
            if config.data.polsess is None:
                from config import PolSESSParams
                config.data.polsess = PolSESSParams()
            config.data.polsess.data_root = args.data_root
    if args.task:
        config.data.task = args.task

    batch_size = args.batch_size if args.batch_size else config.data.batch_size

    return config, batch_size


def create_librimix_dataloader(
    args: argparse.Namespace, batch_size: int
) -> DataLoader:
    """Create DataLoader for Libri2Mix evaluation."""
    dataset = Libri2MixDataset(
        data_root=args.librimix_root,
        subset=args.librimix_subset,
        sample_rate=8000,
        mode="min",
        max_samples=args.max_samples,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=libri2mix_collate_fn,
    )


def run_evaluation(
    model,
    config: Config,
    args: argparse.Namespace,
    device: str,
    batch_size: int,
) -> dict:
    """Run evaluation on selected dataset."""
    if args.dataset == "librimix":
        print(f"\n{'='*80}")
        print(f"LIBRI2MIX CROSS-DATASET EVALUATION")
        print(f"{'='*80}")

        dataloader = create_librimix_dataloader(args, batch_size)

        librimix_results = evaluate_model(
            model,
            dataloader,
            device=device,
            compute_pesq=not args.no_pesq,
            compute_stoi=not args.no_stoi,
            use_amp=False,
        )

        return {f"Libri2Mix_{args.librimix_subset}": librimix_results}
    else:
        return evaluate_by_variant(
            model,
            config,
            device=device,
            batch_size=batch_size,
            compute_pesq=not args.no_pesq,
            compute_stoi=not args.no_stoi,
            specific_variant=args.variant,
        )


def main():
    parser = create_evaluation_parser()
    args = parser.parse_args()

    # Validate dataset-specific arguments
    if args.dataset == "librimix" and not args.librimix_root:
        parser.error("--librimix-root is required when --dataset=librimix")
    if args.dataset == "polsess" and not args.data_root and not args.config:
        parser.error("--data-root or --config is required when --dataset=polsess")

    config, batch_size = load_config_and_apply_overrides(args)

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    apply_eps_patch(1e-4)

    model = load_model(args.checkpoint, config, device)

    results = run_evaluation(model, config, args, device, batch_size)

    print_summary(results)

    if args.output:
        save_results_csv(results, args.output)


if __name__ == "__main__":
    main()
