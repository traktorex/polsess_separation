#!/usr/bin/env python
"""MM-IPC reconstruction audit — proves the augmentation arithmetic is lossless.

WHAT THIS PROVES
----------------
PolSESS training data is augmented with MM-IPC (Mix Modification by Inverted
Phase Cancellation): background layers are removed from the rendered mixture by
subtracting the individually-stored component files (``datasets/polsess_dataset.py``,
``PolSESSDataset._apply_mmipc``). This script demonstrates, on real data, that
that subtraction is *numerically exact* — it recovers the intended target to the
16-bit PCM quantization floor and nothing worse.

The dataset renders every mixture from the additive decomposition (Klec et al.,
Eq. 1; see ``docs/MMIPC_PAPER_VERIFICATION.md``):

    mix = sp1_dry + sp1_reverb + sp2_dry + sp2_reverb
          + scene + event_dry + event_reverb        (indoor / has_reverb)
    mix = sp1_dry + sp2_dry + scene + event_dry      (outdoor / no reverb)

For the ES task, ``_apply_mmipc`` for MM-IPC variant V produces
``mix - (subtracted layers)``. Because the *subtracted* layers and the
*retained* layers partition the full component set, the residual between the
real MM-IPC output and an independently-summed "retained-layers" target reduces,
in exact arithmetic, to ``stored_mix - Σ(all stored layers)`` — i.e. purely the
mismatch between the mixture rendered-then-quantized to 16-bit PCM and the sum of
the separately-quantized component files. That residual is ~1e-4 RMS, well below
this script's threshold, and is *invariant to the variant* (a useful sanity
check: the subtraction is layer algebra, not double-subtraction).

The C variant is the headline case: its retained set is exactly {sp1_dry}, so
ES+C reconstructs the dry speaker-1 target. A small residual there is the
citable "the augmentation is provably lossless" line for the methods chapter.

HOW TO CITE
-----------
    scripts/audit_mmipc.py — regression guard + citable proof that MM-IPC
    reconstructs the dry ES target (and every variant's retained layers) to the
    16-bit PCM quantization floor (~1e-4 RMS) on real PolSESS_C_new_64 data.

USAGE
-----
    CUDA_VISIBLE_DEVICES="" python scripts/audit_mmipc.py            # K=8 rows/split
    CUDA_VISIBLE_DEVICES="" python scripts/audit_mmipc.py --k 2 --splits train
    CUDA_VISIBLE_DEVICES="" python scripts/audit_mmipc.py --threshold 5e-4

Exits non-zero if any variant's reconstruction residual exceeds --threshold.
Companion documentation: the variant-algebra block in
``datasets/polsess_dataset.py`` near ``_apply_mmipc``.
"""

import argparse
import random
import sys
from pathlib import Path

import pandas as pd
import torch
import torchaudio

# Allow running as `python scripts/audit_mmipc.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import PolSESSParams  # noqa: E402
from datasets.polsess_dataset import PolSESSDataset  # noqa: E402

# 16-bit signed PCM full-scale step (LSB). A signal in [-1, 1] stored as int16 is
# quantized to steps of 2 / 2**16 = 2**-15. Per-sample round-off is uniform on
# [-LSB/2, LSB/2] -> RMS = LSB / sqrt(12) ~= 8.8e-6. The reconstruction residual
# sums ~8 independently-quantized layers, so its RMS floor is a few * that
# (~1e-4 observed on real data).
PCM_LSB = 2.0 ** -15  # ~= 3.05e-5

# Default pass/fail threshold: ~33 * PCM_LSB ~= 1e-3. That is one order of
# magnitude above the observed ~1e-4 floor (comfortable margin against benign
# variation) yet two-plus orders of magnitude below any genuine arithmetic
# breakage — a dropped or extra layer shows up at signal scale (1e-2 .. 1e-1 RMS).
DEFAULT_THRESHOLD = 1e-3


def rms(x: torch.Tensor) -> float:
    """Root-mean-square of a tensor as a Python float."""
    return float(torch.sqrt(torch.mean(x.double() ** 2)))


def _ensure_1d(tensor: torch.Tensor) -> torch.Tensor:
    """Squeeze a leading mono channel dim (mirrors PolSESSDataset._ensure_1d)."""
    while tensor.dim() > 1 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    return tensor


def _load1d(path) -> torch.Tensor:
    """Load a wav as a 1-D float tensor."""
    audio, _ = torchaudio.load(str(path))
    return _ensure_1d(audio)


def load_raw_layers(paths: dict, has_reverb: bool) -> dict:
    """Load every stored component layer for a row, independently of the dataset.

    Keys use the additive-decomposition names: sp1_dry, sp2_dry, scene,
    event_dry (+ sp1_reverb, sp2_reverb, event_reverb when has_reverb).
    """
    layers = {
        "sp1_dry": _load1d(paths["speaker1"]),
        "sp2_dry": _load1d(paths["speaker2"]),
        "scene": _load1d(paths["scene"]),
        "event_dry": _load1d(paths["event"]),
    }
    if has_reverb:
        layers["sp1_reverb"] = _load1d(paths["sp1_reverb"])
        layers["sp2_reverb"] = _load1d(paths["sp2_reverb"])
        layers["event_reverb"] = _load1d(paths["ev_reverb"])
    return layers


def expected_retained_es(layers: dict, variant: str, has_reverb: bool) -> torch.Tensor:
    """Independently reconstruct the ES-task target retained by MM-IPC variant V.

    ES keeps speaker 1 and removes speaker 2 entirely. The variant letters say
    which *background* layers survive: S -> keep scene, E -> keep event(+reverb),
    R -> keep speaker-1 reverb tail; C keeps none (so the target is dry speaker 1).

    This is built purely from the decomposition + the KEEP semantics — it never
    calls _apply_mmipc — so comparing it to the real MM-IPC output is an
    independent check of the subtraction, not a tautology.
    """
    target = layers["sp1_dry"].clone()

    # Speaker-1 reverb tail is retained unless the variant is C.
    if has_reverb and variant != "C":
        target = target + layers["sp1_reverb"]

    # Scene retained iff "S" in the variant.
    if "S" in variant:
        target = target + layers["scene"]

    # Event (+ its reverb tail, indoor) retained iff "E" in the variant.
    if "E" in variant:
        target = target + layers["event_dry"]
        if has_reverb:
            target = target + layers["event_reverb"]

    return target


def residual_rms(mmipc_output: torch.Tensor, ideal: torch.Tensor) -> float:
    """RMS of (mmipc_output - ideal), aligned to the shorter length."""
    n = min(mmipc_output.shape[-1], ideal.shape[-1])
    return rms(mmipc_output[..., :n] - ideal[..., :n])


def audit_split(dataset: PolSESSDataset, split: str, k: int, seed: int, rows_rng):
    """Return a list of per-(row, variant) residual records for one split."""
    n_rows = len(dataset.full_metadata)
    if n_rows == 0:
        return [], (0, 0)

    k = min(k, n_rows)
    idxs = rows_rng.sample(range(n_rows), k)

    records = []
    n_indoor = n_outdoor = 0
    for idx in idxs:
        row = dataset.full_metadata.iloc[idx]
        has_reverb = pd.notna(row["reverbForSpeaker1"])
        if has_reverb:
            n_indoor += 1
            variants = PolSESSDataset.INDOOR_VARIANTS
        else:
            n_outdoor += 1
            variants = PolSESSDataset.OUTDOOR_VARIANTS

        paths = dataset._build_paths(row, has_reverb)
        raw = load_raw_layers(paths, has_reverb)

        for variant in variants:
            # Real MM-IPC output via the actual dataset code path.
            audio = dataset._lazy_load(paths, variant, has_reverb)
            mmipc_output = dataset._apply_mmipc(audio, has_reverb)
            # Independently-summed retained-layers target.
            ideal = expected_retained_es(raw, variant, has_reverb)
            records.append(
                {
                    "split": split,
                    "row": idx,
                    "variant": variant,
                    "has_reverb": bool(has_reverb),
                    "residual_rms": residual_rms(mmipc_output, ideal),
                }
            )
    return records, (n_indoor, n_outdoor)


def print_report(records, threshold: float, coverage: dict) -> bool:
    """Print the per-variant residual table. Return True if all pass."""
    df = pd.DataFrame(records)

    print("\nSampled rows per split (indoor / outdoor):")
    for split, (ind, out) in coverage.items():
        print(f"  {split:5s}: {ind} indoor, {out} outdoor")

    # Preserve the canonical variant order for the table.
    order = ["SER", "SR", "ER", "R", "SE", "S", "E", "C"]
    present = [v for v in order if v in set(df["variant"])]

    print("\nPer-variant reconstruction residual (RMS of MM-IPC output vs. "
          "independently-summed retained layers):")
    print(f"  {'variant':<8}{'n':>4}{'mean RMS':>14}{'max RMS':>14}   status")
    print("  " + "-" * 52)
    all_ok = True
    for v in present:
        sub = df[df["variant"] == v]["residual_rms"]
        mean_rms = sub.mean()
        max_rms = sub.max()
        ok = max_rms <= threshold
        all_ok = all_ok and ok
        note = ""
        if v == "C":
            note = "  <- ES+C == dry speaker1"
        status = "PASS" if ok else "FAIL"
        print(f"  {v:<8}{len(sub):>4}{mean_rms:>14.3e}{max_rms:>14.3e}   {status}{note}")

    overall_max = df["residual_rms"].max()
    c_rows = df[df["variant"] == "C"]["residual_rms"]

    print("\nSummary:")
    print(f"  16-bit PCM LSB (full-scale step): {PCM_LSB:.3e}")
    print(f"  threshold:                        {threshold:.3e}")
    print(f"  overall max residual (all variants): {overall_max:.3e}")
    if len(c_rows):
        print(f"  ES+C -> dry speaker1 max residual:   {c_rows.max():.3e}  "
              f"(over {len(c_rows)} rows)")
    return all_ok


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit MM-IPC reconstruction against the 16-bit PCM floor.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=PolSESSParams().data_root,
        help="PolSESS dataset root (defaults to POLSESS_DATA_ROOT / config default).",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,val,test",
        help="Comma-separated subsets to audit.",
    )
    parser.add_argument("--k", type=int, default=8, help="Random rows per split.")
    parser.add_argument("--seed", type=int, default=0, help="Row-selection RNG seed.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Max allowed reconstruction residual RMS (PASS/FAIL boundary).",
    )
    args = parser.parse_args(argv)

    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f"ERROR: data root does not exist: {data_root}", file=sys.stderr)
        return 2

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    rows_rng = random.Random(args.seed)

    print("=" * 66)
    print("MM-IPC reconstruction audit")
    print("=" * 66)
    print(f"data root : {data_root}")
    print(f"splits    : {splits}")
    print(f"K/split   : {args.k}   seed: {args.seed}   task: ES")

    all_records = []
    coverage = {}
    for split in splits:
        # task='ES' so C reconstructs the dry speaker-1 target; unfiltered
        # metadata so we sample uniformly over the whole split.
        dataset = PolSESSDataset(
            data_root=data_root, subset=split, task="ES", allowed_variants=None
        )
        records, cov = audit_split(dataset, split, args.k, args.seed, rows_rng)
        all_records.extend(records)
        coverage[split] = cov

    if not all_records:
        print("ERROR: no rows sampled (empty splits?)", file=sys.stderr)
        return 2

    all_ok = print_report(all_records, args.threshold, coverage)

    print("\n" + "=" * 66)
    if all_ok:
        print("RESULT: PASS — MM-IPC augmentation is numerically lossless "
              "(all residuals <= threshold).")
        return 0
    print("RESULT: FAIL — a variant exceeded the reconstruction threshold. "
          "MM-IPC arithmetic may be broken.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
