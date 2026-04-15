"""Evaluate speech separation models on downstream ASR performance.

Pipeline: mixture → separate (optional) → transcribe with Whisper → WER/CER.
Optionally computes non-intrusive speech quality metrics (SQUIM) for datasets
without clean source audio (e.g., REAL-M).

Three evaluation modes:
    separation: Load model, separate mixture, transcribe, compute WER/CER.
    mixture:    Transcribe unseparated mixture directly (no model needed).
                Shows how badly ASR performs without separation.
    baseline:   Transcribe clean source audio (LibriSpeech only, no model needed).
                Establishes best achievable WER for the dataset.

Usage (from polsess_separation/ directory):
    # Evaluate separation model on REAL-M (SQUIM metrics computed by default)
    python asr/evaluate_asr.py --checkpoint checkpoints/.../best.pt \\
        --dataset realm --mode separation --whisper-model large

    # Mixture baseline on REAL-M (no model needed)
    python asr/evaluate_asr.py --dataset realm --mode mixture

    # Disable SQUIM metrics
    python asr/evaluate_asr.py --dataset realm --mode mixture --no-squim

    # Clean source baseline on LibriSpeech
    python asr/evaluate_asr.py --dataset librispeech --split dev --mode baseline
"""

import argparse
import json
import sys
from pathlib import Path

# Allow running as `python asr/evaluate_asr.py` from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torchaudio
from tqdm import tqdm

from asr.dataset import LibriSpeechMixDataset, RealMDataset
from asr.metrics import compute_metrics, aggregate_metrics
from asr.transcribe import WhisperTranscriber
from utils.model_utils import load_model_for_inference

# Whisper expects 16kHz input (verified: whisper.audio.SAMPLE_RATE == 16000)
WHISPER_SAMPLE_RATE = 16000

# Our separation models operate at 8kHz
MODEL_SAMPLE_RATE = 8000

# SQUIM requires 16kHz input (same as Whisper — no extra resampling needed)
SQUIM_SAMPLE_RATE = 16000


# --- SQUIM non-intrusive quality metrics ---


def load_squim_model(device="cpu"):
    """Load the SQUIM objective model for non-intrusive quality prediction.

    Returns a model that predicts STOI, PESQ, and SI-SDR without reference audio.
    Model weights are downloaded automatically on first use (~28MB).
    """
    from torchaudio.pipelines import SQUIM_OBJECTIVE

    model = SQUIM_OBJECTIVE.get_model()
    model.eval()
    return model.to(device)


def compute_squim_metrics(squim_model, waveform):
    """Predict speech quality metrics without a reference signal using SQUIM.

    Args:
        squim_model: SQUIM objective model from torchaudio.pipelines.
        waveform: Audio tensor (num_channels, samples) at 16kHz.

    Returns:
        List of dicts (one per channel) with squim_stoi, squim_pesq, squim_si_sdr.
    """
    with torch.no_grad():
        stoi, pesq, si_sdr = squim_model(waveform.to(next(squim_model.parameters()).device))

    results = []
    for i in range(waveform.shape[0]):
        results.append({
            "squim_stoi": stoi[i].item(),
            "squim_pesq": pesq[i].item(),
            "squim_si_sdr": si_sdr[i].item(),
        })
    return results


# --- Core evaluation functions ---


def separate_and_transcribe(model, mix, transcriber, resample_in, resample_out, device):
    """Separate a single mixture and transcribe both outputs.

    Args:
        model: Separation model (expects [B, 1, T] at MODEL_SAMPLE_RATE).
        mix: Mixture audio tensor (1, T) at dataset sample rate.
        transcriber: WhisperTranscriber instance.
        resample_in: Resampler to model SR (or None if already matching).
        resample_out: Resampler from model SR to Whisper SR.
        device: Torch device for model inference.

    Returns:
        Tuple of (transcript_s1, transcript_s2, separated_16k) where
        separated_16k is the (2, T) tensor at 16kHz (reused for SQUIM).
    """
    mix_input = mix.to(device)

    # Resample to model SR if dataset SR differs
    if resample_in is not None:
        mix_input = resample_in(mix_input)

    # Separate: model expects (B, 1, T) → returns (B, 2, T)
    with torch.no_grad():
        separated = model(mix_input.unsqueeze(0))  # (1, 2, T)
        separated = separated.squeeze(0)  # (2, T)

    # Resample to 16kHz for Whisper (and SQUIM)
    separated_16k = resample_out(separated.cpu())

    # Transcribe both speakers
    transcript_s1 = transcriber.transcribe(separated_16k[0].numpy())
    transcript_s2 = transcriber.transcribe(separated_16k[1].numpy())

    return transcript_s1, transcript_s2, separated_16k


def assign_speakers(hyp1, hyp2, ref1, ref2):
    """Permutation-invariant speaker assignment based on WER.

    Tries both (hyp1→ref1, hyp2→ref2) and (hyp1→ref2, hyp2→ref1),
    picks the assignment with lower total WER.

    Returns:
        Tuple of (metrics_speaker1, metrics_speaker2) for the best assignment.
    """
    # Assignment 1: hyp1→ref1, hyp2→ref2
    m1_s1 = compute_metrics(hyp1, ref1)
    m1_s2 = compute_metrics(hyp2, ref2)
    error_1 = m1_s1["wer"] + m1_s2["wer"]

    # Assignment 2: hyp1→ref2, hyp2→ref1
    m2_s1 = compute_metrics(hyp2, ref1)
    m2_s2 = compute_metrics(hyp1, ref2)
    error_2 = m2_s1["wer"] + m2_s2["wer"]

    if error_1 <= error_2:
        return m1_s1, m1_s2
    else:
        return m2_s1, m2_s2


def evaluate_separation(model_checkpoint, dataset, transcriber, device="cuda",
                        num_samples=None, squim_model=None):
    """Evaluate separation model: separate → transcribe → WER/CER.

    Args:
        model_checkpoint: Path to trained separation model checkpoint.
        dataset: Dataset instance (LibriSpeechMixDataset or RealMDataset).
        transcriber: WhisperTranscriber instance.
        device: Torch device for model inference.
        num_samples: Number of samples to evaluate (None = all).
        squim_model: SQUIM objective model (None to skip quality metrics).

    Returns:
        Results dictionary with WER/CER and optional SQUIM metrics.
    """
    model, checkpoint = load_model_for_inference(model_checkpoint, device)
    model_type = checkpoint.get("config", {}).get("model", {}).get("model_type", "unknown")
    n = min(num_samples or len(dataset), len(dataset))

    # Determine dataset sample rate from first sample
    first_sample = dataset[0]
    dataset_sr = first_sample["sample_rate"]

    # Setup resamplers
    resample_in = None
    if dataset_sr != MODEL_SAMPLE_RATE:
        resample_in = torchaudio.transforms.Resample(dataset_sr, MODEL_SAMPLE_RATE).to(device)

    resample_out = torchaudio.transforms.Resample(MODEL_SAMPLE_RATE, WHISPER_SAMPLE_RATE)

    # Resampler for mixture → 16kHz (for SQUIM on raw mixture)
    resample_mix_16k = None
    if squim_model is not None and dataset_sr != SQUIM_SAMPLE_RATE:
        resample_mix_16k = torchaudio.transforms.Resample(dataset_sr, SQUIM_SAMPLE_RATE)

    print(f"Model: {model_type} | Dataset SR: {dataset_sr}Hz | Samples: {n}")
    print(f"Resampling: {dataset_sr}→{MODEL_SAMPLE_RATE} (model) → {WHISPER_SAMPLE_RATE} (Whisper)")
    if squim_model is not None:
        print("SQUIM non-intrusive quality metrics: enabled")

    results_s1, results_s2 = [], []
    squim_separated_results, squim_mixture_results = [], []

    for i in tqdm(range(n), desc="Separating & transcribing"):
        sample = dataset[i]
        hyp1, hyp2, separated_16k = separate_and_transcribe(
            model, sample["mix"], transcriber,
            resample_in, resample_out, device,
        )
        m_s1, m_s2 = assign_speakers(
            hyp1, hyp2, sample["transcription1"], sample["transcription2"],
        )
        results_s1.append(m_s1)
        results_s2.append(m_s2)

        if squim_model is not None:
            # SQUIM on separated outputs (2 speakers)
            squim_separated_results.append(compute_squim_metrics(squim_model, separated_16k))

            # SQUIM on raw mixture (single channel)
            mix_16k = sample["mix"]
            if resample_mix_16k is not None:
                mix_16k = resample_mix_16k(mix_16k)
            squim_mixture_results.append(
                compute_squim_metrics(squim_model, mix_16k)[0]
            )

    return _build_results(
        results_s1, results_s2, model_type, n,
        squim_separated=squim_separated_results or None,
        squim_mixture=squim_mixture_results or None,
    )


def evaluate_mixture(dataset, transcriber, num_samples=None, squim_model=None):
    """Transcribe unseparated mixtures — shows ASR degradation without separation.

    This is expected to produce high WER since two speakers overlap.
    """
    n = min(num_samples or len(dataset), len(dataset))

    # Determine dataset sample rate
    first_sample = dataset[0]
    dataset_sr = first_sample["sample_rate"]

    resample = None
    if dataset_sr != WHISPER_SAMPLE_RATE:
        resample = torchaudio.transforms.Resample(dataset_sr, WHISPER_SAMPLE_RATE)

    print(f"Mode: mixture (no separation) | Samples: {n}")
    if squim_model is not None:
        print("SQUIM non-intrusive quality metrics: enabled")

    results_s1, results_s2 = [], []
    squim_mixture_results = []

    for i in tqdm(range(n), desc="Transcribing mixtures"):
        sample = dataset[i]
        mix = sample["mix"]

        if resample is not None:
            mix = resample(mix)

        # Transcribe the mixture as-is — Whisper will try its best
        hypothesis = transcriber.transcribe(mix.squeeze(0).numpy())

        # Compare against both speaker transcriptions
        m_s1, m_s2 = assign_speakers(
            hypothesis, hypothesis,
            sample["transcription1"], sample["transcription2"],
        )
        results_s1.append(m_s1)
        results_s2.append(m_s2)

        if squim_model is not None:
            # mix is (1, T) at 16kHz after resampling
            squim_mixture_results.append(
                compute_squim_metrics(squim_model, mix)[0]
            )

    return _build_results(
        results_s1, results_s2, "mixture_baseline", n,
        squim_mixture=squim_mixture_results or None,
    )


def evaluate_baseline(dataset, transcriber, num_samples=None, squim_model=None):
    """Transcribe clean source audio — best achievable WER (LibriSpeech only).

    Requires dataset to provide 's1' and 's2' keys (clean source audio).
    When SQUIM is enabled, also reports best achievable quality scores.
    """
    if not hasattr(dataset, "s1_dir"):
        raise ValueError("Baseline mode requires clean source audio (LibriSpeech only)")

    n = min(num_samples or len(dataset), len(dataset))
    print(f"Mode: baseline (clean sources) | Samples: {n}")
    if squim_model is not None:
        print("SQUIM non-intrusive quality metrics: enabled (best achievable scores)")

    # Determine dataset sample rate for SQUIM resampling
    first_sample = dataset[0]
    dataset_sr = first_sample["sample_rate"]

    resample_to_16k = None
    if squim_model is not None and dataset_sr != SQUIM_SAMPLE_RATE:
        resample_to_16k = torchaudio.transforms.Resample(dataset_sr, SQUIM_SAMPLE_RATE)

    results_s1, results_s2 = [], []
    squim_separated_results = []

    for i in tqdm(range(n), desc="Transcribing clean sources"):
        sample = dataset[i]

        hyp1 = transcriber.transcribe(sample["s1"].squeeze(0).numpy())
        hyp2 = transcriber.transcribe(sample["s2"].squeeze(0).numpy())

        m_s1 = compute_metrics(hyp1, sample["transcription1"])
        m_s2 = compute_metrics(hyp2, sample["transcription2"])
        results_s1.append(m_s1)
        results_s2.append(m_s2)

        if squim_model is not None:
            # Stack clean sources: (2, T) at 16kHz
            s1 = sample["s1"]
            s2 = sample["s2"]
            clean_pair = torch.cat([s1, s2], dim=0)  # (2, T)
            if resample_to_16k is not None:
                clean_pair = resample_to_16k(clean_pair)
            squim_separated_results.append(
                compute_squim_metrics(squim_model, clean_pair)
            )

    return _build_results(
        results_s1, results_s2, "clean_baseline", n,
        squim_separated=squim_separated_results or None,
    )


# --- Helper functions ---


def _build_results(results_s1, results_s2, model_type, num_samples,
                    squim_separated=None, squim_mixture=None):
    """Aggregate per-sample metrics into a results dictionary.

    Args:
        results_s1, results_s2: Per-sample WER/CER metrics for each speaker.
        model_type: Model identifier string.
        num_samples: Number of evaluated samples.
        squim_separated: List of [speaker1_metrics, speaker2_metrics] per sample.
        squim_mixture: List of mixture metrics dicts per sample.
    """
    agg_s1 = aggregate_metrics(results_s1)
    agg_s2 = aggregate_metrics(results_s2)
    avg_wer = (agg_s1["wer"] + agg_s2["wer"]) / 2
    avg_cer = (agg_s1["cer"] + agg_s2["cer"]) / 2

    results = {
        "model_type": model_type,
        "num_samples": num_samples,
        "avg_wer": float(avg_wer),
        "avg_cer": float(avg_cer),
        "speaker1": _to_python(agg_s1),
        "speaker2": _to_python(agg_s2),
    }

    if squim_separated is not None:
        squim = {}
        for metric in ("squim_stoi", "squim_pesq", "squim_si_sdr"):
            s1_vals = [s[0][metric] for s in squim_separated]
            s2_vals = [s[1][metric] for s in squim_separated]
            s1_mean = sum(s1_vals) / len(s1_vals)
            s2_mean = sum(s2_vals) / len(s2_vals)
            squim[f"separated_{metric}_s1"] = s1_mean
            squim[f"separated_{metric}_s2"] = s2_mean
            squim[f"separated_{metric}_avg"] = (s1_mean + s2_mean) / 2
        results["squim"] = squim

    if squim_mixture is not None:
        squim = results.get("squim", {})
        for metric in ("squim_stoi", "squim_pesq", "squim_si_sdr"):
            vals = [m[metric] for m in squim_mixture]
            squim[f"mixture_{metric}"] = sum(vals) / len(vals)
        results["squim"] = squim

    return results


def _to_python(obj):
    """Convert numpy types to Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_python(item) for item in obj]
    if hasattr(obj, "item"):
        return obj.item()
    return obj


def print_results(results):
    """Print evaluation results summary."""
    print()
    print("=" * 70)
    print(f"RESULTS: {results['model_type']} ({results['num_samples']} samples)")
    print("=" * 70)
    print(f"  Speaker 1:  WER = {results['speaker1']['wer']:.2f}%,  CER = {results['speaker1']['cer']:.2f}%")
    print(f"  Speaker 2:  WER = {results['speaker2']['wer']:.2f}%,  CER = {results['speaker2']['cer']:.2f}%")
    print(f"  Average:    WER = {results['avg_wer']:.2f}%,  CER = {results['avg_cer']:.2f}%")

    if "squim" in results:
        squim = results["squim"]
        print()
        print("  SQUIM Speech Quality (non-intrusive):")

        has_separated = "separated_squim_stoi_avg" in squim
        has_mixture = "mixture_squim_stoi" in squim

        if has_separated:
            print(f"    Separated — STOI: {squim['separated_squim_stoi_avg']:.3f},"
                  f"  PESQ: {squim['separated_squim_pesq_avg']:.2f},"
                  f"  SI-SDR: {squim['separated_squim_si_sdr_avg']:.1f} dB")
        if has_mixture:
            print(f"    Mixture   — STOI: {squim['mixture_squim_stoi']:.3f},"
                  f"  PESQ: {squim['mixture_squim_pesq']:.2f},"
                  f"  SI-SDR: {squim['mixture_squim_si_sdr']:.1f} dB")
        if has_separated and has_mixture:
            print(f"    Improvement: "
                  f"STOI: {squim['separated_squim_stoi_avg'] - squim['mixture_squim_stoi']:+.3f},"
                  f"  PESQ: {squim['separated_squim_pesq_avg'] - squim['mixture_squim_pesq']:+.2f},"
                  f"  SI-SDR: {squim['separated_squim_si_sdr_avg'] - squim['mixture_squim_si_sdr']:+.1f} dB")

    print("=" * 70)


def save_results(results, output_path):
    """Save results to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")


# --- CLI ---


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate speech separation models on ASR performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Mode
    parser.add_argument("--mode", type=str, default="separation",
                        choices=["separation", "mixture", "baseline"],
                        help="Evaluation mode (default: separation)")

    # Dataset
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["librispeech", "realm"],
                        help="Dataset to evaluate on")
    parser.add_argument("--dataset-dir", type=str, default=None,
                        help="Override dataset root directory")
    parser.add_argument("--split", type=str, default="dev",
                        help="LibriSpeech split: dev, test, long, 10sec (default: dev)")

    # Model (required for separation mode)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to separation model checkpoint (required for separation mode)")

    # Whisper
    parser.add_argument("--whisper-model", type=str, default="tiny.en",
                        help="Whisper model size (default: tiny.en)")
    parser.add_argument("--whisper-device", type=str, default="cuda",
                        help="Device for Whisper")

    # SQUIM
    parser.add_argument("--no-squim", action="store_true",
                        help="Skip SQUIM non-intrusive quality metrics "
                             "(STOI, PESQ, SI-SDR predicted without reference)")

    # General
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Limit number of samples (default: all)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for separation model (default: cuda)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSON file")

    args = parser.parse_args()

    # Validate: separation mode requires a checkpoint
    if args.mode == "separation" and args.checkpoint is None:
        parser.error("--checkpoint is required for separation mode")

    # Validate: baseline mode only for LibriSpeech
    if args.mode == "baseline" and args.dataset == "realm":
        parser.error("Baseline mode requires clean source audio (LibriSpeech only)")

    # Create dataset
    if args.dataset == "librispeech":
        dataset = LibriSpeechMixDataset(
            dataset_root=args.dataset_dir,
            split=args.split,
            max_samples=args.num_samples,
        )
    else:
        dataset = RealMDataset(
            dataset_root=args.dataset_dir,
            max_samples=args.num_samples,
        )

    # Load Whisper
    transcriber = WhisperTranscriber(
        model_name=args.whisper_model,
        device=args.whisper_device,
    )

    # Load SQUIM model (enabled by default)
    squim_model = None
    if not args.no_squim:
        squim_model = load_squim_model(device=args.device)
        print("SQUIM objective model loaded")

    # Run evaluation
    if args.mode == "separation":
        results = evaluate_separation(
            model_checkpoint=args.checkpoint,
            dataset=dataset,
            transcriber=transcriber,
            device=args.device,
            num_samples=args.num_samples,
            squim_model=squim_model,
        )
    elif args.mode == "mixture":
        results = evaluate_mixture(
            dataset=dataset,
            transcriber=transcriber,
            num_samples=args.num_samples,
            squim_model=squim_model,
        )
    else:  # baseline
        results = evaluate_baseline(
            dataset=dataset,
            transcriber=transcriber,
            num_samples=args.num_samples,
            squim_model=squim_model,
        )

    # Output
    print_results(results)

    if args.output:
        save_results(results, args.output)


if __name__ == "__main__":
    main()
