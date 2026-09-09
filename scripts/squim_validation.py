"""B7 — SQUIM-on-PolSESS validation (thesis backlog B7, ch7 §7.3 / figure F6).

Validates TorchAudio-SQUIM (SQUIM_OBJECTIVE) against true intrusive metrics
on PolSESS test mixtures separated by a *quality ladder* of systems spanning
noise → near-perfect, so the correlation is measured over the whole plausible
range of separation quality. Pre-registration (frozen before any scoring):
``thesis/thesis-log/sweep_plan/B7_SQUIM_PREREG.md`` — the verdict rule
(pooled Spearman >= 0.6 on real rungs), ladder, protocol, and smell checks
all live there; this script only executes them.

Protocol fidelity: SQUIM and the true PESQ-WB / STOI secondaries are computed
through ``asr_pipeline.eval.layer2``'s own chunked functions (identical
chunker + speech-presence filter as deployment) on 16 kHz sinc-upsampled
audio — the same spectral shape the deployed pipeline feeds SQUIM. True
SI-SDR (the primary truth) follows the repo convention: torchmetrics at
native 8 kHz, PIT-matched with the same ``PITLossWrapper`` object
``evaluate.py`` and the trainer use.

Stages (idempotent, resume-safe at system granularity):

    python scripts/squim_validation.py subset            # freeze the sample subset
    python scripts/squim_validation.py score             # run + score the ladder
    python scripts/squim_validation.py analyze           # report + smell checks
    python scripts/squim_validation.py all               # the three in order
    ... --smoke                                          # 2 samples/family, _smoke outputs
    ... score --systems mossformer2_e46                  # single rung (e.g. adding MF2-full later)

Outputs -> ``thesis/thesis-log/sweep_plan/b7_squim/``:
``b7_subset.csv``, ``b7_scores.csv``, ``b7_provenance.json``,
``B7_SQUIM_REPORT.md`` (smoke runs write ``*_smoke`` siblings).
"""

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from utils import warning_filters  # noqa: F401, E402  must precede speechbrain imports

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torchaudio.functional as AF  # noqa: E402

from torchmetrics.audio import ScaleInvariantSignalDistortionRatio  # noqa: E402
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr  # noqa: E402

from datasets import PolSESSDataset  # noqa: E402
from utils import apply_eps_patch, load_model_for_inference  # noqa: E402
from asr_pipeline.eval.layer2 import (  # noqa: E402
    load_squim_model,
    pesq_wb_chunked,
    squim_chunked,
    stoi_chunked,
)

SR = 8000
SR_SQUIM = 16000
N_PER_FAMILY = 64
SEED = 0
REF20_SNR_DB = 20.0
SPEARMAN_THRESHOLD = 0.6       # konspekt v3 §7.3, prereg'd
WITHIN_SYSTEM_GUARD = 0.3      # prereg'd nuance guard
BOOTSTRAP_DRAWS = 10_000

DATA_ROOT = Path.home() / "datasets" / "PolSESS_C_final_128_v2"
OUT_DIR = REPO / "thesis" / "thesis-log" / "sweep_plan" / "b7_squim"

INDOOR = ["SER", "SR", "ER", "R", "C"]
OUTDOOR = ["SE", "S", "E", "C"]
# Stable per-variant integer for noise seeding (never reorder).
VARIANT_CODE = {v: i for i, v in enumerate(["SER", "SR", "ER", "R", "C", "SE", "S", "E"])}

# (system_id, checkpoint path relative to repo root or None for synthetic).
# Ladder order = intended quality order (smell check: true SI-SDR should be
# broadly monotone along it). See prereg §Systems.
LADDER = [
    ("anchor_noise", None),
    ("anchor_mix", None),
    ("convtasnet_base", "checkpoints/convtasnet/SB/run_2026-03-24_15-04-31/convtasnet_SB_best.pt"),
    ("dprnn_base", "checkpoints/dprnn/SB/run_2026-03-24_15-15-52/dprnn_SB_best.pt"),
    ("spmamba_64k", "checkpoints/spmamba/SB/64k_baseline_87lepmzg/spmamba_SB_best.pt"),
    # Directory renamed 2026-07-30 (checkpoint-zoo disambiguation); the B7 run of
    # 2026-07-18 recorded it under its pre-rename name. Same run: its config.yaml
    # carries wandb_run_name: 16k_baseline_posenc-avgvaltest.
    ("sepformer_16k", "checkpoints/sepformer/SB/16k_baseline_posenc-avgvaltest_DEVRUN_tms4000/sepformer_SB_best.pt"),
    ("sepformer_128k", "checkpoints/sepformer/SB/128_run/sepformer_SB_best_128k_e41.pt"),
    ("mossformer2_e46", "checkpoints/mossformer2/SB/mossformer2_matched_128k_final_42_e46/mossformer2_SB_best_e46.pt"),
    ("anchor_ref20", None),
]

SCORE_COLUMNS = [
    "system", "sample_idx", "family", "variant", "stream", "mix_file",
    "true_sisdr", "mix_sisdr", "true_sisdri",
    "pesq_wb", "pesq_n_scored", "stoi",
    "squim_sisdr", "squim_pesq", "squim_stoi", "squim_n_chunks",
    "est_rms", "est_clip_frac", "est_has_nan", "vol_scale",
]


# ---------------------------------------------------------------------------
# Data plumbing
# ---------------------------------------------------------------------------


def make_dataset() -> PolSESSDataset:
    return PolSESSDataset(data_root=DATA_ROOT, subset="test", task="SB", allowed_variants=None)


def subset_path(smoke: bool) -> Path:
    return OUT_DIR / ("b7_subset_smoke.csv" if smoke else "b7_subset.csv")


def scores_path(smoke: bool) -> Path:
    return OUT_DIR / ("b7_scores_smoke.csv" if smoke else "b7_scores.csv")


def freeze_subset(ds: PolSESSDataset, n_per_family: int, smoke: bool) -> pd.DataFrame:
    """Draw and freeze the sample subset (prereg §Data). Idempotent: an
    existing subset file is loaded, never redrawn."""
    path = subset_path(smoke)
    if path.exists():
        return pd.read_csv(path)

    indoor_mask = ds.metadata["reverbForSpeaker1"].notna()
    indoor_idx = ds.metadata.index[indoor_mask].tolist()
    outdoor_idx = ds.metadata.index[~indoor_mask].tolist()
    rng = random.Random(SEED)
    chosen = sorted(rng.sample(indoor_idx, n_per_family)) + sorted(
        rng.sample(outdoor_idx, n_per_family)
    )

    rows = []
    extra_cols = [c for c in ("SSR", "sceneClass", "eventClass") if c in ds.metadata.columns]
    for idx in chosen:
        row = ds.metadata.iloc[idx]
        family = "indoor" if pd.notna(row["reverbForSpeaker1"]) else "outdoor"
        rec = {"sample_idx": idx, "family": family, "mix_file": row["mixFile"]}
        for c in extra_cols:
            rec[c] = row[c]
        rows.append(rec)
    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[subset] froze {len(df)} samples -> {path}")
    return df


def work_items(subset: pd.DataFrame):
    """Yield (sample_idx, family, variant, mix_file) — every variant compatible
    with each sample's family."""
    for _, row in subset.iterrows():
        variants = INDOOR if row["family"] == "indoor" else OUTDOOR
        for v in variants:
            yield int(row["sample_idx"]), row["family"], v, row["mix_file"]


def load_item(ds: PolSESSDataset, idx: int, variant: str):
    """Materialize (mix, clean[2,T]) for a forced MM-IPC variant — same
    internals as ``__getitem__`` / ``scripts/audit_mmipc.py``, minus the
    per-index variant draw."""
    row = ds.metadata.iloc[idx]
    has_reverb = pd.notna(row["reverbForSpeaker1"])
    paths = ds._build_paths(row, has_reverb)
    audio = ds._lazy_load(paths, variant, has_reverb)
    mix = ds._apply_mmipc(audio, has_reverb)
    clean = ds._compute_clean(audio)
    return mix, clean


# ---------------------------------------------------------------------------
# Estimate generation
# ---------------------------------------------------------------------------


def _seeded_noise(shape, idx: int, variant: str, stream: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(((idx * 8 + VARIANT_CODE[variant]) << 1) | stream)
    return torch.randn(shape, generator=g)


def synth_estimate(system: str, mix: torch.Tensor, clean: torch.Tensor,
                   idx: int, variant: str) -> torch.Tensor:
    """Synthetic anchor outputs, [2, T]."""
    if system == "anchor_mix":
        return torch.stack([mix, mix])
    if system == "anchor_noise":
        mix_rms = mix.pow(2).mean().sqrt().clamp_min(1e-8)
        return torch.stack([
            _seeded_noise(mix.shape, idx, variant, k) * mix_rms for k in (0, 1)
        ])
    if system == "anchor_ref20":
        outs = []
        for k in (0, 1):
            ref = clean[k]
            ref_rms = ref.pow(2).mean().sqrt().clamp_min(1e-8)
            n = _seeded_noise(ref.shape, idx, variant, k)
            n = n / n.pow(2).mean().sqrt().clamp_min(1e-8)
            outs.append(ref + n * ref_rms / (10 ** (REF20_SNR_DB / 20)))
        return torch.stack(outs)
    raise ValueError(f"unknown synthetic system {system}")


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


class Scorer:
    """Holds the metric objects (repo conventions) + the SQUIM model."""

    def __init__(self, device: str):
        self.device = device
        self.sisdr = ScaleInvariantSignalDistortionRatio().to(device)
        self.pit = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx").to(device)
        self.squim, self.squim_device = load_squim_model(device)

    def score(self, system: str, idx: int, family: str, variant: str, mix_file: str,
              est: torch.Tensor, clean: torch.Tensor, mix: torch.Tensor) -> list:
        """est [2,T'] (any device), clean [2,T], mix [T] cpu -> two CSV rows."""
        min_len = min(est.shape[-1], clean.shape[-1], mix.shape[-1])
        est = est[..., :min_len].float().to(self.device)
        clean = clean[..., :min_len].float().to(self.device)
        mix_d = mix[..., :min_len].float().to(self.device)

        # Deployment-parity volume normalization (separation stage
        # `_volume_normalise`, mode `sum_equals_mix` — shipped default): one
        # common scale so RMS(s1+s2) == RMS(mix). SI-SDR/PESQ/STOI are
        # level-invariant; SQUIM is NOT, and deployed streams are normalized
        # before SQUIM ever sees them. Applied uniformly to every system
        # (prereg deviations log, 2026-07-17).
        combined_rms = (est[0] + est[1]).double().pow(2).mean().sqrt()
        mix_rms = mix_d.double().pow(2).mean().sqrt()
        vol_scale = 1.0
        if combined_rms > 1e-9:
            vol_scale = float(mix_rms / combined_rms)
            est = est * vol_scale

        _, reordered = self.pit(est.unsqueeze(0), clean.unsqueeze(0), return_est=True)
        reordered = reordered[0]

        rows = []
        for k in (0, 1):
            e, r = reordered[k], clean[k]
            has_nan = bool(torch.isnan(e).any() or torch.isinf(e).any())
            if has_nan:
                e = torch.nan_to_num(e)
            true_sisdr = self.sisdr(e.unsqueeze(0), r.unsqueeze(0)).item()
            mix_sisdr = self.sisdr(mix_d.unsqueeze(0), r.unsqueeze(0)).item()

            e_cpu = e.detach().cpu()
            r_cpu = r.detach().cpu()
            e16_t = AF.resample(e_cpu, SR, SR_SQUIM)
            r16_t = AF.resample(r_cpu, SR, SR_SQUIM)
            pq = pesq_wb_chunked(e16_t, r16_t, SR_SQUIM)
            st = stoi_chunked(e16_t, r16_t, SR_SQUIM)
            sq = squim_chunked(e16_t.numpy(), SR_SQUIM, self.squim, self.squim_device)

            rows.append({
                "system": system, "sample_idx": idx, "family": family,
                "variant": variant, "stream": k, "mix_file": mix_file,
                "true_sisdr": true_sisdr, "mix_sisdr": mix_sisdr,
                "true_sisdri": true_sisdr - mix_sisdr,
                "pesq_wb": pq["median"], "pesq_n_scored": pq["n_scored"],
                "stoi": st,
                "squim_sisdr": sq["squim_si_sdr"], "squim_pesq": sq["squim_pesq"],
                "squim_stoi": sq["squim_stoi"], "squim_n_chunks": sq["n_chunks"],
                "est_rms": float(e_cpu.pow(2).mean().sqrt()),
                "est_clip_frac": float((e_cpu.abs() >= 0.99).float().mean()),
                "est_has_nan": has_nan,
                "vol_scale": vol_scale,
            })
        return rows


def run_system(system: str, ckpt: str, ds: PolSESSDataset, subset: pd.DataFrame,
               scorer: Scorer, device: str) -> pd.DataFrame:
    model = None
    if ckpt is not None:
        model, _ = load_model_for_inference(str(REPO / ckpt), device=device)
        model.eval()

    items = list(work_items(subset))
    rows, t0 = [], time.time()
    with torch.no_grad():
        for n, (idx, family, variant, mix_file) in enumerate(items, 1):
            mix, clean = load_item(ds, idx, variant)
            if model is None:
                est = synth_estimate(system, mix, clean, idx, variant)
            else:
                est = model(mix.unsqueeze(0).unsqueeze(0).to(device))[0]
            rows.extend(scorer.score(system, idx, family, variant, mix_file, est, clean, mix))
            if n % 100 == 0:
                print(f"  [{system}] {n}/{len(items)} items "
                      f"({time.time() - t0:.0f}s)", flush=True)

    if model is not None:
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    print(f"[score] {system}: {len(df)} rows, true SI-SDR mean "
          f"{df['true_sisdr'].mean():+.2f} dB (sd {df['true_sisdr'].std():.2f}), "
          f"{time.time() - t0:.0f}s", flush=True)
    return df


def stage_score(ds, subset, smoke: bool, only_systems, device: str) -> pd.DataFrame:
    path = scores_path(smoke)
    existing = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=SCORE_COLUMNS)
    expected = sum(1 for _ in work_items(subset)) * 2

    apply_eps_patch(1e-4)  # training-time SpeechBrain EPS (ConvTasNet lobe parity)
    torch.manual_seed(SEED)
    scorer = Scorer(device)

    for system, ckpt in LADDER:
        if only_systems and system not in only_systems:
            continue
        have = int((existing["system"] == system).sum())
        if have == expected:
            print(f"[score] {system}: complete ({have} rows), skipping")
            continue
        if have:
            print(f"[score] {system}: partial ({have}/{expected}), redoing")
            existing = existing[existing["system"] != system]
        df = run_system(system, ckpt, ds, subset, scorer, device)
        existing = pd.concat([existing, df], ignore_index=True)[SCORE_COLUMNS]
        tmp = path.with_suffix(".tmp.csv")
        existing.to_csv(tmp, index=False)
        tmp.replace(path)

    del scorer
    if device == "cuda":
        torch.cuda.empty_cache()
    return existing


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def _spearman(x: pd.Series, y: pd.Series) -> tuple:
    from scipy.stats import spearmanr

    mask = x.notna() & y.notna()
    if mask.sum() < 3:
        return float("nan"), 0
    rho, _ = spearmanr(x[mask], y[mask])
    return float(rho), int(mask.sum())


def _cluster_bootstrap_ci(df: pd.DataFrame, xcol: str, ycol: str,
                          draws: int = BOOTSTRAP_DRAWS, seed: int = SEED) -> tuple:
    """Percentile CI for Spearman rho, clustered by base sample (prereg)."""
    from scipy.stats import spearmanr

    d = df[df[xcol].notna() & df[ycol].notna()]
    clusters = {sid: g[[xcol, ycol]].to_numpy() for sid, g in d.groupby("sample_idx")}
    ids = list(clusters)
    rng = np.random.default_rng(seed)
    rhos = np.empty(draws)
    for b in range(draws):
        picked = rng.choice(len(ids), size=len(ids), replace=True)
        arr = np.concatenate([clusters[ids[i]] for i in picked])
        rhos[b], _ = spearmanr(arr[:, 0], arr[:, 1])
    return float(np.percentile(rhos, 2.5)), float(np.percentile(rhos, 97.5))


def stage_analyze(ds, subset: pd.DataFrame, scores: pd.DataFrame, smoke: bool) -> None:
    real = scores[~scores["system"].str.startswith("anchor_")]
    lines: list[str] = []
    add = lines.append
    problems: list[str] = []

    add("# B7 — SQUIM-on-PolSESS validation: report")
    add("")
    add(f"*Generated {time.strftime('%Y-%m-%d %H:%M')} by `scripts/squim_validation.py"
        f"{' --smoke' if smoke else ''}`. Prereg: `B7_SQUIM_PREREG.md` (same directory) — "
        "verdict rule, ladder, protocol all frozen there. Data: `b7_scores.csv`; "
        "subset: `b7_subset.csv`.*")
    add("")

    # ----- Smell checks (prereg §Smell checks) -----------------------------
    add("## Smell checks")
    add("")

    n_expected = sum(1 for _ in work_items(subset)) * 2
    add(f"- Subset: {len(subset)} samples "
        f"({(subset['family'] == 'indoor').sum()} indoor / "
        f"{(subset['family'] == 'outdoor').sum()} outdoor); "
        f"{n_expected} expected rows per system.")
    counts = scores.groupby("system").size()
    for system, _ in LADDER:
        n = int(counts.get(system, 0))
        if n != n_expected:
            problems.append(f"{system}: {n} rows (expected {n_expected})")
    add(f"- Row counts: {' | '.join(f'{s}={int(counts.get(s, 0))}' for s, _ in LADDER)}")

    if "SSR" in subset.columns and "SSR" in ds.metadata.columns:
        full_ssr = ds.metadata["SSR"]
        add(f"- SSR distribution: subset mean {subset['SSR'].mean():+.2f} dB "
            f"(sd {subset['SSR'].std():.2f}) vs full test {full_ssr.mean():+.2f} "
            f"(sd {full_ssr.std():.2f}).")
        if abs(subset["SSR"].mean() - full_ssr.mean()) > 0.5 * full_ssr.std():
            problems.append("subset SSR mean deviates > 0.5 sd from full test")
    for col in ("sceneClass", "eventClass"):
        if col in subset.columns and col in ds.metadata.columns:
            sub_top = subset[col].value_counts(normalize=True).head(3)
            full_top = ds.metadata[col].value_counts(normalize=True)
            drift = max(abs(sub_top.get(k, 0) - full_top.get(k, 0)) for k in sub_top.index)
            add(f"- {col}: top subset shares "
                f"{ {k: round(float(v), 2) for k, v in sub_top.items()} }; "
                f"max drift vs full test {drift:.2f}.")
            if drift > 0.15:
                problems.append(f"{col} share drifts > 0.15 vs full test")

    add("")
    add("Per-system output & metric sanity:")
    add("")
    add("| system | true SI-SDR mean±sd [min,max] | silent frac | clip frac | NaN | "
        "SQUIM rows scored | n_chunks=1 |")
    add("|---|---|---|---|---|---|---|")
    for system, _ in LADDER:
        g = scores[scores["system"] == system]
        if g.empty:
            continue
        silent = float((g["est_rms"] < 1e-4).mean())
        clipf = float((g["est_clip_frac"] > 0.05).mean())
        nnan = int(g["est_has_nan"].sum())
        sq_ok = int(g["squim_sisdr"].notna().sum())
        chunk1 = float((g["squim_n_chunks"] == 1).mean())
        add(f"| {system} | {g['true_sisdr'].mean():+.2f}±{g['true_sisdr'].std():.2f} "
            f"[{g['true_sisdr'].min():+.1f},{g['true_sisdr'].max():+.1f}] "
            f"| {silent:.3f} | {clipf:.3f} | {nnan} | {sq_ok}/{len(g)} | {chunk1:.3f} |")
        if not system.startswith("anchor_"):
            if silent > 0.02:
                problems.append(f"{system}: silent-stream fraction {silent:.3f} > 2%")
            if clipf > 0.05:
                problems.append(f"{system}: rows with >5% clipped samples: {clipf:.3f}")
            if nnan:
                problems.append(f"{system}: {nnan} NaN/Inf outputs")
            if chunk1 < 0.99:
                problems.append(f"{system}: squim n_chunks==1 on only {chunk1:.3f}")

    am = scores[scores["system"] == "anchor_mix"]
    if not am.empty:
        max_dev = float((am["true_sisdr"] - am["mix_sisdr"]).abs().max())
        add(f"- anchor_mix cross-check: max |SI-SDR − mix baseline| = {max_dev:.4f} dB "
            f"(tolerance 1e-3).")
        if max_dev > 1e-3:
            problems.append(f"anchor_mix SI-SDR deviates from mix baseline by {max_dev:.4f} dB")
    ar = scores[scores["system"] == "anchor_ref20"]
    if not ar.empty:
        m = float(ar["true_sisdr"].mean())
        add(f"- anchor_ref20 cross-check: mean SI-SDR {m:+.2f} dB (expect ≈{REF20_SNR_DB:.0f}±1.5).")
        if abs(m - REF20_SNR_DB) > 1.5:
            problems.append(f"anchor_ref20 mean SI-SDR {m:+.2f} off {REF20_SNR_DB}±1.5")

    valid_squim = scores["squim_sisdr"].dropna()
    stoi_bad = int(((scores["squim_stoi"] < 0) | (scores["squim_stoi"] > 1.0)).sum())
    pesq_bad = int(((scores["squim_pesq"] < 0.9) | (scores["squim_pesq"] > 4.8)).sum())
    add(f"- SQUIM ranges: SI-SDR est [{valid_squim.min():+.1f}, {valid_squim.max():+.1f}] dB; "
        f"STOI out-of-[0,1]: {stoi_bad}; PESQ outside [0.9,4.8]: {pesq_bad}.")
    if stoi_bad or pesq_bad:
        problems.append(f"SQUIM outputs out of valid range (stoi {stoi_bad}, pesq {pesq_bad})")

    add("")
    if problems:
        add("**SMELL CHECK FAILURES:**")
        for p in problems:
            add(f"- ⚠ {p}")
    else:
        add("**All smell checks pass.**")
    add("")

    # ----- Primary + secondary analyses (prereg §Analyses) ------------------
    add("## Primary endpoint")
    add("")
    rho, n = _spearman(real["squim_sisdr"], real["true_sisdr"])
    lo, hi = _cluster_bootstrap_ci(real, "squim_sisdr", "true_sisdr")
    verdict = "PASS" if rho >= SPEARMAN_THRESHOLD else "FAIL"
    add(f"Pooled Spearman ρ(squim_si_sdr, true SI-SDR), real rungs, n={n}: "
        f"**ρ = {rho:.3f}** [95% cluster-bootstrap CI {lo:.3f}, {hi:.3f}] "
        f"vs threshold {SPEARMAN_THRESHOLD} → **{verdict}**.")
    add("")

    add("## Within-system ρ (nuance guard) + calibration")
    add("")
    add("| system | ρ (SI-SDR) | n | mean signed err (SQUIM−true, dB) |")
    add("|---|---|---|---|")
    within = []
    for system, _ in LADDER:
        g = scores[scores["system"] == system]
        if g.empty:
            continue
        r, nn = _spearman(g["squim_sisdr"], g["true_sisdr"])
        bias = float((g["squim_sisdr"] - g["true_sisdr"]).mean())
        if not system.startswith("anchor_"):
            within.append(r)
        add(f"| {system} | {r:.3f} | {nn} | {bias:+.2f} |")
    med_within = float(np.nanmedian(within)) if within else float("nan")
    guard = ("ranking-valid within systems"
             if med_within >= WITHIN_SYSTEM_GUARD
             else "NOT usable for per-fragment ranking (guard tripped)")
    add("")
    add(f"Median within-system ρ across real rungs: **{med_within:.3f}** "
        f"(guard {WITHIN_SYSTEM_GUARD}) → {guard}.")
    add("")

    add("## Per-variant pooled ρ (real rungs)")
    add("")
    add("| variant | ρ (SI-SDR) | n |")
    add("|---|---|---|")
    for v in INDOOR + [x for x in OUTDOOR if x != "C"]:
        g = real[real["variant"] == v]
        r, nn = _spearman(g["squim_sisdr"], g["true_sisdr"])
        add(f"| {v} | {r:.3f} | {nn} |")
    add("")

    add("## Secondary metrics")
    add("")
    for est_col, true_col, label in (
        ("squim_pesq", "pesq_wb", "PESQ"),
        ("squim_stoi", "stoi", "STOI"),
    ):
        r, nn = _spearman(real[est_col], real[true_col])
        w = [
            _spearman(scores[scores["system"] == s][est_col],
                      scores[scores["system"] == s][true_col])[0]
            for s, _ in LADDER if not s.startswith("anchor_")
        ]
        add(f"- {label}: pooled ρ = {r:.3f} (n={nn}); median within-system "
            f"ρ = {float(np.nanmedian(w)):.3f}.")
    r_all, n_all = _spearman(scores["squim_sisdr"], scores["true_sisdr"])
    add(f"- Anchors-included pooled ρ = {r_all:.3f} (n={n_all}) — range-inflated, "
        "never the decision number (prereg §Analyses 6).")
    add("")

    add("## Verdict")
    add("")
    if verdict == "PASS":
        add(f"Primary **PASS** (ρ = {rho:.3f} ≥ {SPEARMAN_THRESHOLD}): SQUIM retained in "
            "pipeline scoring; F6 drawn from `b7_scores.csv`; ch7 §7.7 quantitative "
            "figure unlocked; ch8 SQUIM verdict positive. "
            + ("Within-system guard also clear." if med_within >= WITHIN_SYSTEM_GUARD else
               "**But the within-system guard tripped** — §7.3 must state both numbers; "
               "per-fragment ranking use is NOT licensed."))
    else:
        add(f"Primary **FAIL** (ρ = {rho:.3f} < {SPEARMAN_THRESHOLD}): SQUIM is dropped "
            "from pipeline scoring; ch8 SQUIM-threshold verdict flips negative; ch7 §7.7 "
            "cites the failed validation as further dB≠WER evidence and stays qualitative.")
    if smoke:
        add("")
        add("*(SMOKE RUN — subset too small for any verdict; numbers are plumbing checks only.)*")

    report = OUT_DIR / ("B7_SQUIM_REPORT_smoke.md" if smoke else "B7_SQUIM_REPORT.md")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[analyze] report -> {report}")


def write_provenance(smoke: bool, args) -> None:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                             text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=REPO,
                                    capture_output=True, text=True).stdout.strip())
    except Exception:
        sha, dirty = "unknown", None
    import torchaudio
    prov = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_sha": sha, "git_dirty": dirty,
        "seed": SEED, "n_per_family": N_PER_FAMILY if not smoke else args.n_per_family,
        "data_root": str(DATA_ROOT),
        "ladder": [{"system": s, "checkpoint": c} for s, c in LADDER],
        "torch": torch.__version__, "torchaudio": torchaudio.__version__,
        "argv": sys.argv,
    }
    path = OUT_DIR / ("b7_provenance_smoke.json" if smoke else "b7_provenance.json")
    path.write_text(json.dumps(prov, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=["subset", "score", "analyze", "all"])
    ap.add_argument("--smoke", action="store_true",
                    help="2 samples/family, *_smoke output files")
    ap.add_argument("--systems", nargs="*", default=None,
                    help="restrict scoring to these ladder ids")
    ap.add_argument("--n-per-family", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    n_per_family = args.n_per_family or (2 if args.smoke else N_PER_FAMILY)
    ds = make_dataset()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    subset = freeze_subset(ds, n_per_family, args.smoke)
    if args.stage == "subset":
        return

    if args.stage in ("score", "all"):
        stage_score(ds, subset, args.smoke, args.systems, args.device)
        write_provenance(args.smoke, args)

    if args.stage in ("analyze", "all"):
        path = scores_path(args.smoke)
        if not path.exists():
            sys.exit(f"no scores at {path} — run the score stage first")
        stage_analyze(ds, subset, pd.read_csv(path), args.smoke)


if __name__ == "__main__":
    main()
