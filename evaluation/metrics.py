"""Metrics computation for model evaluation."""

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchmetrics.audio import (
    ScaleInvariantSignalDistortionRatio,
    PerceptualEvaluationSpeechQuality,
    ShortTimeObjectiveIntelligibility,
)
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr

from config import Config
from datasets import PolSESSDataset, polsess_collate_fn


def evaluate_model(
    model,
    dataloader: DataLoader,
    device: str = "cuda",
    compute_pesq: bool = True,
    compute_stoi: bool = True,
    use_amp: bool = False,
    task: str = "ES",
) -> dict:
    """Evaluate model on a dataset."""
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
