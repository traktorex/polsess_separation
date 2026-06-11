"""Sweep ASR-pipeline configurations over a fixed eval set and rank by cpWER.

Runs each named config (a set of overrides on ``default.yaml`` + the eval
overrides from ``run_pipeline_on_recording._fresh_cfg``) across the pilot
recordings, writes outputs to ``<id>/sweep/<config_name>/``, then scores
every config's per-speaker transcripts against the hand-corrected GT EAF
(``<id>/annotation.eaf``) with cpWER / tcpWER and ranks them.

The config registry is **data-driven**: each entry is a dict of dotted
override paths → values, applied on top of the baseline. Add a row to
``CONFIGS`` (and optionally to ``GROUPS``) to explore a new knob — no new
code. Single-knob (OFAT) entries isolate each knob's marginal effect from
the baseline; combine paths in one dict for interactions.

Phase-major: each (config, recording) is a fresh ``Pipeline`` that loads +
unloads its models. That re-runs every stage per config — simple and
correct, but the per-run model-load overhead dominates on short fragments,
so keep the active config set focused.

Usage::

    python scripts/sweep_pipeline.py --groups asr enhance      # run two groups
    python scripts/sweep_pipeline.py --configs baseline enh_mossformer
    python scripts/sweep_pipeline.py --score-only              # re-rank, no runs
    python scripts/sweep_pipeline.py --groups asr --force      # re-run
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from asr_pipeline import Pipeline                                   # noqa: E402
from asr_pipeline.io import write_pipeline_outputs                 # noqa: E402
from asr_pipeline.eval.metrics import (                             # noqa: E402
    cp_cer_meeteval,
    cpwer_meeteval,
    mimo_cer_meeteval,
    mimo_wer_meeteval,
    orc_wer_meeteval,
    orc_wer_multistream,
)
from asr_pipeline.eval.layer3 import read_mixture, read_per_speaker  # noqa: E402
from asr_pipeline.eval.recordings import (                          # noqa: E402
    load_recording,
    load_reference_utterances,
)
from scripts.run_pipeline_on_recording import _fresh_cfg            # noqa: E402


EVAL_ROOT = Path("~/datasets/eval/clarin_fragments").expanduser()
CFG_PATH = REPO_ROOT / "asr_pipeline" / "configs" / "default.yaml"

# The 5 pilot recordings (hand-corrected GT present).
PILOT = [
    "065a9896__seg00", "150d1ccc__seg00", "ccfbb9db__seg00",
    "2bf3474d__seg00", "d1e63652__seg00",
]

# --- Config registry ------------------------------------------------------
# Each value is a dict of "stage.field" -> value, applied on the baseline.
# baseline = default.yaml (full pipeline: enh=mpsenet, sep=128k SepFormer,
# bwe=ap_bwe, whisperx large-v2) + eval overrides (full_length,
# transcribe_mixture, min_overlap_dur=0).
CONFIGS: dict[str, dict] = {
    "baseline":          {},   # full pipeline; transcription = large-v2
    # --- transcription model (the biggest WER lever); large-v2 = baseline ---
    "asr_largev3":       {"transcription.model_name": "large-v3"},
    # --- enhancement backend (solo regions) ---
    "enh_mossformer":    {"enhancement.backend": "mossformer_gan_se_16k"},
    "enh_frcrn":         {"enhancement.backend": "frcrn_se_16k"},
    "enh_none":          {"enhancement.enabled": False},
    # --- bandwidth extension (overlap regions) ---
    "bwe_naive":         {"post_separation_processing.backend": "naive"},
    "bwe_flowhigh":      {"post_separation_processing.backend": "flowhigh"},
    # --- separation knobs ---
    "sep_seam_zc":       {"separation.seam_mode": "zero_crossing"},
    "sep_seam_boundary": {"separation.seam_mode": "overlap_boundary"},
    "sep_vad_strict":    {"separation.vad_threshold": 0.5,
                          "separation.vad_soft_threshold": 0.2},
    "sep_vol_none":      {"separation.volume_normalization": "none"},
    # --- assembly knobs ---
    "asm_perpiece_rms":  {"assembly.per_piece_rms_norm": True},
    # --- ablation corners (2x2 separation x enhancement) ---
    #   baseline   = sep + enh   (full)
    #   enh_none   = sep, no enh (above, in the enhance group)
    #   nosep      = enh, no sep
    #   nosep_noenh = neither
    "nosep":             {"separation.enabled": False},
    "nosep_noenh":       {"separation.enabled": False, "enhancement.enabled": False},
    # --- round 2: re-anchor the promising knobs on FRCRN (round-1 winner) ---
    "frcrn_vad_strict":  {"enhancement.backend": "frcrn_se_16k",
                          "separation.vad_threshold": 0.5,
                          "separation.vad_soft_threshold": 0.2},
    "frcrn_seam_zc":     {"enhancement.backend": "frcrn_se_16k",
                          "separation.seam_mode": "zero_crossing"},
    "frcrn_vad_seam":    {"enhancement.backend": "frcrn_se_16k",
                          "separation.vad_threshold": 0.5,
                          "separation.vad_soft_threshold": 0.2,
                          "separation.seam_mode": "zero_crossing"},
    "frcrn_largev3":     {"enhancement.backend": "frcrn_se_16k",
                          "transcription.model_name": "large-v3"},
    # --- round 3: micro-variants around frcrn_vad_strict (round-2 winner) ---
    # frcrn_vad_strict = frcrn + VAD 0.5/0.2 + attack/release 1/1 (defaults).
    "frcrn_vad_040":     {"enhancement.backend": "frcrn_se_16k",
                          "separation.vad_threshold": 0.4,
                          "separation.vad_soft_threshold": 0.15},
    "frcrn_vad_060":     {"enhancement.backend": "frcrn_se_16k",
                          "separation.vad_threshold": 0.6,
                          "separation.vad_soft_threshold": 0.25},
    "frcrn_vad_strict_ar0": {"enhancement.backend": "frcrn_se_16k",
                             "separation.vad_threshold": 0.5,
                             "separation.vad_soft_threshold": 0.2,
                             "separation.vad_attack_frames": 0,
                             "separation.vad_release_frames": 0},
    "frcrn_vad_strict_ar2": {"enhancement.backend": "frcrn_se_16k",
                             "separation.vad_threshold": 0.5,
                             "separation.vad_soft_threshold": 0.2,
                             "separation.vad_attack_frames": 2,
                             "separation.vad_release_frames": 2},
    # --- BWE decision (SCOPE open question 4): flowhigh vs ap_bwe on the
    # round-2/3 winner, not the mpsenet baseline (where round-1's bwe_*
    # configs were swamped by enhancement errors) ---
    "frcrn_vad_strict_flowhigh": {"enhancement.backend": "frcrn_se_16k",
                                  "separation.vad_threshold": 0.5,
                                  "separation.vad_soft_threshold": 0.2,
                                  "post_separation_processing.backend": "flowhigh"},
    # input_sr A/B: default.yaml ships flowhigh_input_sr=8000 (matches the
    # separator's spectral content); this arm feeds 16 kHz instead.
    "frcrn_vad_strict_flowhigh16": {"enhancement.backend": "frcrn_se_16k",
                                    "separation.vad_threshold": 0.5,
                                    "separation.vad_soft_threshold": 0.2,
                                    "post_separation_processing.backend": "flowhigh",
                                    "post_separation_processing.flowhigh_input_sr": 16_000},
}

# Named groups for --groups selection. "baseline" is always included.
GROUPS: dict[str, list[str]] = {
    "asr":        ["asr_largev3"],
    "enhance":    ["enh_mossformer", "enh_frcrn", "enh_none"],
    "bwe":        ["bwe_naive", "bwe_flowhigh"],
    "separation": ["sep_seam_zc", "sep_seam_boundary", "sep_vad_strict", "sep_vol_none"],
    "assembly":   ["asm_perpiece_rms"],
    "ablation":   ["nosep", "nosep_noenh"],
    # enh_frcrn (round-1 winner) included as the round-2 anchor — already run,
    # so it's skipped on run and just rescored alongside the new variants.
    "round2":     ["enh_frcrn", "frcrn_vad_strict", "frcrn_seam_zc",
                   "frcrn_vad_seam", "frcrn_largev3"],
    # frcrn_vad_strict (round-2 winner) included as the round-3 anchor.
    "round3":     ["frcrn_vad_strict", "frcrn_vad_040", "frcrn_vad_060",
                   "frcrn_vad_strict_ar0", "frcrn_vad_strict_ar2"],
}


def _apply(cfg, overrides: dict):
    """Apply dotted-path overrides onto a config, then re-validate.

    A typo'd path must fail loud (SCOPE §4.2): bare ``setattr`` would create a
    junk attribute, leave the intended knob at its default, and silently run
    the baseline under the typo'd name — a fabricated sweep row with no signal.
    """
    for path, val in overrides.items():
        obj = cfg
        *parents, leaf = path.split(".")
        for p in parents:
            obj = getattr(obj, p)
        if not hasattr(obj, leaf):
            raise AttributeError(f"unknown override path: {path!r}")
        setattr(obj, leaf, val)
    cfg.__post_init__()
    return cfg


def _build_cfg(overrides: dict):
    return _apply(_fresh_cfg(CFG_PATH), overrides)


# --- Run ------------------------------------------------------------------


def run_config(name, overrides, force, eval_root, recordings) -> None:
    """Run one config over the given recordings → ``<id>/sweep/<name>/``."""
    subdir = f"sweep/{name}"
    for fid in recordings:
        rec_dir = eval_root / fid
        audio = rec_dir / f"{fid}.wav"
        target = rec_dir / subdir / "transcript_A.txt"
        if not audio.exists():
            print(f"    {fid}: MISSING audio {audio}")
            continue
        if target.exists() and not force:
            print(f"    {fid}: skip (done)")
            continue
        t0 = time.perf_counter()
        cfg = _build_cfg(overrides)
        p = Pipeline(cfg)
        try:
            ctx = p.run(str(audio))
            write_pipeline_outputs(ctx, rec_dir, config_snapshot=asdict(cfg),
                                   subdir_name=subdir)
            print(f"    {fid}: done in {time.perf_counter()-t0:.1f}s")
        except Exception as e:
            print(f"    {fid}: ERROR {type(e).__name__}: {e}")
        finally:
            p.unload(); del p; gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# --- Score ----------------------------------------------------------------


def score_configs(config_names, eval_root, recordings) -> pd.DataFrame:
    """cpWER / tcpWER / ORC per (config, recording), micro-averaged.

    Micro-average (sum errors / sum reference length) is the standard WER
    aggregation; we also keep the per-recording rates for inspection. GT is
    read via the eval loader — the EAF at ``<id>/annotation.eaf`` if present,
    else ``<id>/reference/speaker_{A,B}.txt`` — so this works for both the
    ELAN-annotated fragments and the older .txt-GT recordings.
    """
    gt = {}
    for fid in recordings:
        rec = load_recording(eval_root / fid)
        gt[fid] = load_reference_utterances(rec) if rec is not None else {}
    rows = []
    for name in config_names:
        cp_err = cp_len = tcp_err = tcp_len = 0
        orc_err = orc_len = 0          # ORC-WER on the 2-stream output
        cer_err = cer_len = 0          # cp-CER on the 2-stream output (cpWER permutation)
        mix_err = mix_len = 0          # mixture floor, ORC-WER (time-fixed merge)
        mmix_err = mmix_len = 0        # mixture floor, MIMO-WER (optimised interleaving)
        mxcer_err = mxcer_len = 0      # mixture floor, MIMO-CER
        per_rec = {}
        n_done = 0
        for fid in recordings:
            d = eval_root / fid / "sweep" / name
            hyp = read_per_speaker(d)
            if hyp is None:
                continue
            ref = {k: v for k, v in gt[fid].items() if v}
            if not ref:
                continue
            r = cpwer_meeteval(ref, hyp, session_id=fid)
            cp_err += r["cp_errors"]; cp_len += r["cp_length"]
            tcp_err += r["tcp_errors"]; tcp_len += r["tcp_length"]
            o = orc_wer_multistream(ref, hyp, session_id=fid)
            orc_err += o["errors"]; orc_len += o["length"]
            cc = cp_cer_meeteval(ref, hyp, session_id=fid)
            cer_err += cc["errors"]; cer_len += cc["length"]
            mix_utts = read_mixture(d)
            if mix_utts is not None:
                m = orc_wer_meeteval(ref, mix_utts, session_id=fid)
                mix_err += m["errors"]; mix_len += m["length"]
                mm = mimo_wer_meeteval(ref, mix_utts, session_id=fid)
                mmix_err += mm["errors"]; mmix_len += mm["length"]
                mxc = mimo_cer_meeteval(ref, mix_utts, session_id=fid)
                mxcer_err += mxc["errors"]; mxcer_len += mxc["length"]
            per_rec[fid] = r["cpwer"]
            n_done += 1
        if n_done == 0:
            continue
        cpwer = 100 * cp_err / cp_len if cp_len else float("nan")
        orcwer = 100 * orc_err / orc_len if orc_len else float("nan")
        row = {
            "config": name,
            "n": n_done,
            "cpWER": cpwer,
            "orcWER": orcwer,
            "attr_gap": cpwer - orcwer,      # speaker-attribution penalty
            "tcpWER": 100 * tcp_err / tcp_len if tcp_len else float("nan"),
            "CER": 100 * cer_err / cer_len if cer_len else float("nan"),
            "mixMIMO": 100 * mmix_err / mmix_len if mmix_len else float("nan"),
            "mixORC": 100 * mix_err / mix_len if mix_len else float("nan"),
            "mixCER": 100 * mxcer_err / mxcer_len if mxcer_len else float("nan"),
        }
        for fid in recordings:
            row[fid[:8]] = round(100 * per_rec[fid], 1) if fid in per_rec else None
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("cpWER").reset_index(drop=True)
    return df


# --- Driver ---------------------------------------------------------------


def _selected_configs(args) -> list[str]:
    if args.configs:
        names = list(args.configs)
    elif args.groups:
        names = []
        for g in args.groups:
            names += GROUPS[g]
    else:
        names = [n for n in CONFIGS if n != "baseline"]  # all but baseline
    # baseline always present for comparison; dedupe, preserve order
    seen, ordered = set(), []
    for n in ["baseline"] + names:
        if n in CONFIGS and n not in seen:
            seen.add(n); ordered.append(n)
    return ordered


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--groups", nargs="+", choices=list(GROUPS),
                    help="Config groups to run (baseline always included).")
    ap.add_argument("--configs", nargs="+", choices=list(CONFIGS),
                    help="Explicit config names (overrides --groups).")
    ap.add_argument("--score-only", action="store_true",
                    help="Skip running; just score + rank existing outputs.")
    ap.add_argument("--force", action="store_true", help="Re-run even if done.")
    ap.add_argument("--eval-root", type=Path, default=EVAL_ROOT,
                    help="Eval-tree root holding <id>/ recording dirs "
                         f"(default: {EVAL_ROOT}).")
    ap.add_argument("--recordings", nargs="+", default=PILOT,
                    help="Recording ids under --eval-root (default: the 5 pilot fragments).")
    ap.add_argument("--csv", type=Path, default=None,
                    help="Results CSV (default: <eval-root>/_sweep_results.csv).")
    args = ap.parse_args()

    eval_root = args.eval_root.expanduser()
    recordings = args.recordings
    csv = args.csv or (eval_root / "_sweep_results.csv")

    names = _selected_configs(args)
    print(f"eval_root: {eval_root}")
    print(f"recordings: {recordings}")
    print(f"configs: {names}\n")

    if not args.score_only:
        for name in names:
            print(f"[{name}]  overrides={CONFIGS[name] or '(baseline)'}")
            run_config(name, CONFIGS[name], args.force, eval_root, recordings)
            print()

    df = score_configs(names, eval_root, recordings)
    pd.set_option("display.width", 200)
    print("\n=== ranked by micro-averaged cpWER (lower is better) ===")
    print(df.to_string(index=False))
    df.to_csv(csv, index=False)
    print(f"\nwrote {csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
