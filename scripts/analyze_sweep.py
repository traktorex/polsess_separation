#!/usr/bin/env python3
"""Correlation / ranking analysis over the tidy sweep-results CSV.

Three views, all micro-averaged correctly (Σerrors / Σref, NOT mean-of-rates):
  1. best overall configs
  2. best configs per stratum (LOW/MID/HIGH)
  3. per-knob × per-stratum effect map (the adaptivity signal)

Point estimates only — for SIGNIFICANCE (recording-clustered CI + Holm/FDR) use
`scripts/rescore_stratified.py`. This tool tells you WHERE to look; the rescorer
tells you whether it's real.

Usage:
  python scripts/analyze_sweep.py                       # dev, cpcer (primary)
  python scripts/analyze_sweep.py --metric cpwer
  python scripts/analyze_sweep.py --csv .../sweep_results_tidy.csv
"""
import argparse
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

EVAL = Path("~/datasets/eval/clarin_fragments").expanduser()
ANCHOR = "dr_refineplus"
NUMERIC_KNOBS = ["oa_ratio", "vad_threshold"]
CAT_KNOBS = ["asr_model", "bwe_backend", "enh_backend", "enh_enabled",
             "diar_embedding", "relabel_source"]
KNOBS = NUMERIC_KNOBS + CAT_KNOBS
REF = {"cpwer": "cpwer_ref_words", "cpcer": "cpcer_ref_chars",
       "tcpwer": "tcpwer_ref_words", "orcwer": "orcwer_ref_words"}


def micro(d: pd.DataFrame, metric: str) -> float:
    """Exact micro-average rate (%) from the count columns."""
    e = d[f"{metric}_errors"].sum()
    r = d[REF[metric]].sum()
    return 100.0 * e / r if r else np.nan


def per_config_micro(df, metric, stratum=None):
    d = df if stratum is None else df[df.stratum == stratum]
    return (d.groupby("config")
             .apply(lambda g: micro(g, metric))
             .rename(metric))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(EVAL / "sweep_results_tidy.csv"))
    ap.add_argument("--metric", default="cpcer", choices=list(REF))
    ap.add_argument("--secondary", default="cpwer", choices=list(REF))
    ap.add_argument("--topn", type=int, default=15)
    args = ap.parse_args()
    M, S = args.metric, args.secondary
    df = pd.read_csv(args.csv)
    anchor_row = df[df.config == ANCHOR].iloc[0]
    anchor_m = micro(df[df.config == ANCHOR], M)
    strata = ["LOW", "MID", "HIGH"]

    # ---- which knob(s) each config changes vs the anchor (OFAT detection) ----
    def diff_knobs(cfg):
        row = df[df.config == cfg].iloc[0]
        return [k for k in KNOBS if str(row[k]) != str(anchor_row[k])]
    configs = sorted(df.config.unique())
    dk = {c: diff_knobs(c) for c in configs}

    # ============ 1. BEST OVERALL ============
    print(f"\n{'='*70}\n1. BEST OVERALL CONFIGS (micro-avg, lower=better; anchor {M}={anchor_m:.2f})\n{'='*70}")
    ov = pd.concat([per_config_micro(df, M), per_config_micro(df, S)], axis=1)
    ov["Δvs_anchor"] = ov[M] - anchor_m
    ov["changes"] = [", ".join(dk[c]) or "(anchor/no-op)" for c in ov.index]
    print(ov.sort_values(M).head(args.topn).to_string(
        float_format=lambda x: f"{x:6.2f}"))

    # ============ 2. BEST PER STRATUM ============
    print(f"\n{'='*70}\n2. BEST CONFIG PER STRATUM (micro-avg within stratum)\n{'='*70}")
    print("⚠ per-stratum n is small (LOW/MID=6 recs, HIGH=8) → point-estimate")
    print("  rankings are NOISY; confirm with rescore_stratified.py Holm/CI.")
    for st in strata:
        a_st = micro(df[(df.config == ANCHOR) & (df.stratum == st)], M)
        r = pd.concat([per_config_micro(df, M, st), per_config_micro(df, S, st)], axis=1)
        r["Δvs_anchor"] = r[M] - a_st
        r["changes"] = [", ".join(dk[c]) or "(anchor/no-op)" for c in r.index]
        print(f"\n--- {st} (anchor {M}={a_st:.2f}) — top 5 ---")
        print(r.sort_values(M).head(5).to_string(float_format=lambda x: f"{x:6.2f}"))

    # ============ 3. KNOB × STRATUM ADAPTIVITY MAP ============
    print(f"\n{'='*70}\n3. KNOB × STRATUM EFFECT MAP — the adaptivity signal\n{'='*70}")
    print("For each knob's OFAT sub-grid (configs that change ONLY that knob),")
    print(f"the best value + Δ{M} vs anchor IN EACH STRATUM. A knob whose best")
    print("value / effect DIFFERS across strata is an adaptivity candidate.\n")
    ofat = {}
    for c in configs:
        if len(dk[c]) == 1:
            ofat.setdefault(dk[c][0], []).append(c)
    rows = []
    for k in KNOBS:
        grid = [ANCHOR] + ofat.get(k, [])
        if len(grid) < 2:
            continue
        rec = {"knob": k, "n_vals": len(grid)}
        for st in strata + ["ALL"]:
            sub = df[df.config.isin(grid)]
            scores = {c: micro(sub[(sub.config == c) & ((sub.stratum == st) if st != "ALL" else True)], M)
                      for c in grid}
            best_c = min(scores, key=lambda c: (scores[c] if not np.isnan(scores[c]) else 1e9))
            best_v = df[df.config == best_c].iloc[0][k]
            a = scores[ANCHOR]
            rec[f"{st}_best"] = f"{best_v}({scores[best_c]-a:+.1f})"
            if k in NUMERIC_KNOBS:
                vals = [(float(df[df.config == c].iloc[0][k]), scores[c])
                        for c in grid if not np.isnan(scores[c])]
                if len(vals) >= 3:
                    rho, p = stats.spearmanr([v for v, _ in vals], [s for _, s in vals])
                    rec[f"{st}_ρ"] = f"{rho:+.2f}"
        rows.append(rec)
    amap = pd.DataFrame(rows).set_index("knob")
    print(amap.to_string())
    print("\nReading it: 'best(Δ)' = best value of the knob in that stratum and its")
    print(f"Δ{M} vs anchor. ρ = Spearman(knob_value, {M}) within the stratum")
    print("(negative ρ = larger knob value → lower error). A knob is an ADAPTIVITY")
    print("candidate iff its best value or effect-sign differs across strata AND the")
    print("effect is non-trivial — then confirm significance with the rescorer.")


if __name__ == "__main__":
    main()
