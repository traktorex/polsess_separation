"""ASR evaluation metrics: WER and CER using jiwer."""

from typing import List

import jiwer


def word_error_rate(hypothesis: str, reference: str) -> tuple[float, int, int]:
    """Compute Word Error Rate (WER).

    Args:
        hypothesis: Predicted transcription.
        reference: Ground truth transcription.

    Returns:
        Tuple of (WER as percentage, edit distance, reference length).
    """
    ref_words = reference.upper().split()
    ref_len = len(ref_words)

    if ref_len == 0:
        return 0.0, 0, 0

    out = jiwer.process_words(reference.upper(), hypothesis.upper())
    distance = out.substitutions + out.deletions + out.insertions
    wer = distance / ref_len * 100

    return wer, distance, ref_len


def character_error_rate(hypothesis: str, reference: str) -> tuple[float, int, int]:
    """Compute Character Error Rate (CER).

    Spaces are stripped before comparison.

    Args:
        hypothesis: Predicted transcription.
        reference: Ground truth transcription.

    Returns:
        Tuple of (CER as percentage, edit distance, reference length).
    """
    ref_clean = reference.upper().replace(" ", "")
    hyp_clean = hypothesis.upper().replace(" ", "")
    ref_len = len(ref_clean)

    if ref_len == 0:
        return 0.0, 0, 0

    out = jiwer.process_characters(ref_clean, hyp_clean)
    distance = out.substitutions + out.deletions + out.insertions
    cer = distance / ref_len * 100

    return cer, distance, ref_len


def compute_metrics(hypothesis: str, reference: str) -> dict:
    """Compute both WER and CER.

    Args:
        hypothesis: Predicted transcription.
        reference: Ground truth transcription.

    Returns:
        Dictionary with WER and CER metrics.
    """
    wer, wer_dist, wer_ref_len = word_error_rate(hypothesis, reference)
    cer, cer_dist, cer_ref_len = character_error_rate(hypothesis, reference)

    return {
        "wer": wer,
        "wer_distance": wer_dist,
        "wer_ref_len": wer_ref_len,
        "cer": cer,
        "cer_distance": cer_dist,
        "cer_ref_len": cer_ref_len,
        "hypothesis": hypothesis.upper(),
        "reference": reference.upper(),
    }


def aggregate_metrics(results: List[dict]) -> dict:
    """Aggregate metrics across multiple samples (micro-averaging).

    Args:
        results: List of individual metric dictionaries from compute_metrics.

    Returns:
        Aggregated metrics weighted by reference length.
    """
    total_wer_dist = sum(r["wer_distance"] for r in results)
    total_wer_ref = sum(r["wer_ref_len"] for r in results)
    total_cer_dist = sum(r["cer_distance"] for r in results)
    total_cer_ref = sum(r["cer_ref_len"] for r in results)

    avg_wer = (total_wer_dist / total_wer_ref * 100) if total_wer_ref > 0 else 0.0
    avg_cer = (total_cer_dist / total_cer_ref * 100) if total_cer_ref > 0 else 0.0

    return {
        "wer": avg_wer,
        "cer": avg_cer,
        "total_samples": len(results),
        "total_wer_distance": total_wer_dist,
        "total_wer_ref_len": total_wer_ref,
        "total_cer_distance": total_cer_dist,
        "total_cer_ref_len": total_cer_ref,
    }
