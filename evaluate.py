"""Evaluation script for PolSESS speech separation models (SI-SDR, PESQ, STOI)."""

import torch
import argparse
from torch.utils.data import DataLoader

from datasets import Libri2MixDataset, libri2mix_collate_fn
from config import Config, load_config_from_yaml, PolSESSParams
from evaluation import (
    load_model_from_checkpoint_for_eval,
    evaluate_model,
    evaluate_by_variant,
    print_summary,
    save_results_csv,
)
from utils import apply_eps_patch


def create_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate speech separation model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file (optional)")
    
    # Dataset selection
    parser.add_argument(
        "--dataset",
        type=str,
        default="polsess",
        choices=["polsess", "librimix"],
        help="Dataset to evaluate on",
    )
    
    # PolSESS arguments
    parser.add_argument("--data-root", type=str, default=None, help="Root directory of PolSESS dataset")
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
    parser.add_argument("--librimix-root", type=str, default=None, help="Path to Libri2Mix root")
    parser.add_argument(
        "--librimix-subset",
        type=str,
        default="test",
        choices=["test", "dev", "train-100"],
        help="Libri2Mix subset",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of samples")
    
    # Common arguments
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use")
    parser.add_argument("--no-pesq", action="store_true", help="Skip PESQ computation (faster)")
    parser.add_argument("--no-stoi", action="store_true", help="Skip STOI computation (faster)")
    parser.add_argument("--output", type=str, default=None, help="Output CSV file for results")
    
    return parser


def load_and_apply_cli_overrides(args: argparse.Namespace) -> tuple[Config, int]:
    """Load config and apply CLI overrides."""
    if args.config:
        print(f"Loading config from: {args.config}")
        config = load_config_from_yaml(args.config)
    else:
        config = Config()
    
    # Apply CLI overrides
    if args.data_root:
        if config.data.polsess is None:
            config.data.polsess = PolSESSParams()
        config.data.polsess.data_root = args.data_root
    if args.task:
        config.data.task = args.task
    
    batch_size = args.batch_size if args.batch_size else config.data.batch_size
    return config, batch_size


def evaluate_librimix(
    model, args: argparse.Namespace, batch_size: int, device: str
) -> dict:
    """Evaluate on Libri2Mix dataset."""
    print(f"\n{'='*80}")
    print("LIBRI2MIX CROSS-DATASET EVALUATION")
    print(f"{'='*80}")
    
    dataset = Libri2MixDataset(
        data_root=args.librimix_root,
        subset=args.librimix_subset,
        sample_rate=8000,
        mode="min",
        max_samples=args.max_samples,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=libri2mix_collate_fn,
    )
    
    results = evaluate_model(
        model,
        dataloader,
        device=device,
        compute_pesq=not args.no_pesq,
        compute_stoi=not args.no_stoi,
        use_amp=False,
    )
    
    return {f"Libri2Mix_{args.librimix_subset}": results}


def main():
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Validate arguments
    if args.dataset == "librimix" and not args.librimix_root:
        parser.error("--librimix-root is required when --dataset=librimix")
    if args.dataset == "polsess" and not args.data_root and not args.config:
        parser.error("--data-root or --config is required when --dataset=polsess")
    
    config, batch_size = load_and_apply_cli_overrides(args)
    
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    apply_eps_patch(1e-4)
    
    model = load_model_from_checkpoint_for_eval(args.checkpoint, config, device)
    
    # Run evaluation
    if args.dataset == "librimix":
        results = evaluate_librimix(model, args, batch_size, device)
    else:
        results = evaluate_by_variant(
            model,
            config,
            device=device,
            batch_size=batch_size,
            compute_pesq=not args.no_pesq,
            compute_stoi=not args.no_stoi,
            specific_variant=args.variant,
        )
    
    print_summary(results)
    
    if args.output:
        save_results_csv(results, args.output)


if __name__ == "__main__":
    main()
