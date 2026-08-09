"""The one list of architectures the two benchmark scripts measure.

Shared so `benchmark_inference.py` and `benchmark_training.py` can never drift
apart again: before 2026-07-30 each script carried its own copy, the training
copy grew two extra rows and renamed one, and the two stored CSVs could no
longer be joined on the model name.

Every row is a **separation (SB, C=2)** config, because the thesis's multi-axis
comparison is a separation comparison. Two rows the old lists had are gone:

  * ConvTasNet used to be measured through `experiments/convtasnet/baseline.yaml`,
    which is the **ES (C=1)** enhancement config — a single-output model on a row
    whose SI-SDR came from a two-output SB run. Replaced by `sb_task.yaml`.
  * "SepFormer (pos-enc)" and "SepFormer" were separate rows. Sinusoidal
    positional encoding is parameter-free and adds no measurable MACs or latency
    (the 2026-04 CSV reports 258.4822 GMACs / 54.5 ms for both, identical to four
    decimals), so one row covers both. The ±PE difference is a quality and
    convergence finding, not an efficiency one.
  * "SepFormer (final 128k)" is gone: its model block is architecturally
    identical to the baseline's (only `dropout` differs), so it timed the same
    network twice.
"""

# (display name, size label, config path relative to the repo root)
# Order = the order the thesis table lists them in.
MODELS = [
    ("ConvTasNet",      "sb",      "experiments/convtasnet/sb_task.yaml"),
    ("DPRNN (k=16)",    "default", "experiments/dprnn/dprnn_baseline.yaml"),
    ("DPRNN (k=2)",     "k2",      "experiments/dprnn/variants/dprnn_baseline_kernel2.yaml"),
    ("SepFormer",       "default", "experiments/sepformer/sepformer_baseline_positionalenc.yaml"),
    ("MossFormer2-matched", "matched", "experiments/mossformer2/mossformer2_matched.yaml"),
    ("MossFormer2-full",    "full",    "experiments/mossformer2/mossformer2_full.yaml"),
    ("SPMamba (reduced)",   "reduced", "experiments/spmamba/spmamba_sb_reduced.yaml"),
    ("SPMamba (full)",      "full",    "experiments/spmamba/spmamba_sb.yaml"),
    ("MambaTasNet-XS",  "xs",      "experiments/mamba_tasnet/mamba_tasnet_xs.yaml"),
    ("MambaTasNet-S",   "s",       "experiments/mamba_tasnet/mamba_tasnet_s.yaml"),
    ("MambaTasNet-M",   "m",       "experiments/mamba_tasnet/mamba_tasnet_m.yaml"),
    ("MambaTasNet-L",   "l",       "experiments/mamba_tasnet/mamba_tasnet_l.yaml"),
    ("DPMamba-XS",      "xs",      "experiments/dpmamba/dpmamba_xs.yaml"),
    ("DPMamba-S",       "s",       "experiments/dpmamba/dpmamba_s.yaml"),
    ("DPMamba-M",       "m",       "experiments/dpmamba/dpmamba_m.yaml"),
    ("DPMamba-L",       "l",       "experiments/dpmamba/dpmamba_l.yaml"),
]

# Mamba-family models need CUDA + mamba-ssm for their forward pass (they
# construct fine on CPU). Used to print an honest skip message instead of a
# stack trace when the benchmark is run without a GPU.
MAMBA_PREFIXES = ("SPMamba", "MambaTasNet", "DPMamba")


def select(models, only):
    """Filter `models` to rows whose display name contains any `only` substring."""
    if not only:
        return list(models)
    needles = [s.lower() for s in only]
    return [m for m in models if any(s in m[0].lower() for s in needles)]
