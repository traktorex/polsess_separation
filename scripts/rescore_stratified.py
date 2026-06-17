"""Stratified, recording-clustered re-score of pipeline configs (dev or test).

The measurement backbone for the forensic-solutions work AND the frozen one-shot
test scoring. For each config it collapses fragments to RECORDINGS (88741282 has
3 segments, 9a651086 has 2 — fragment-level resampling fabricates significance),
micro-averages cpWER / cp-CER per recording, stratifies by acoustic complexity
(composite tertiles, assigned at RECORDING level so a multi-segment recording
lands in exactly ONE stratum), and cluster-bootstraps the PAIRED difference vs an
anchor config (default f_oa03) resampling RECORDINGS with replacement.

CER is co-headline with WER (Polish morphology inflates WER; see forensics).
Positive paired delta = the config is BETTER than the anchor (lower error).

Pick the fragment set with --split (dev|test) or --fragments-file. The headline
number is always computed on the recordings COMMON to every config (paired); a
config missing any recording is reported loudly, never silently intersected away.

    python scripts/rescore_stratified.py --configs t2_attr_margin pl_volnone
    python scripts/rescore_stratified.py --split test --configs f_oa03 --mixture
    python scripts/rescore_stratified.py --anchor f_oa03 --configs cx_pieces --mixture
"""
from __future__ import annotations
import argparse, csv, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from asr_pipeline.eval.metrics import (                                  # noqa: E402
    cpwer_meeteval, cp_cer_meeteval,
    mimo_wer_meeteval, mimo_cer_meeteval)
from asr_pipeline.eval.layer3 import read_per_speaker, read_mixture       # noqa: E402
from asr_pipeline.eval.recordings import (                                # noqa: E402
    load_recording, load_reference_utterances)

EVAL = Path("~/datasets/eval/clarin_fragments").expanduser()
B_DRAWS, SEED = 10000, 0


def _recid(fid): return fid.split("__")[0]


def load_split(args) -> list[str]:
    """The active fragment list — a named split or an explicit file. Both the
    space-separated dev list and the newline-separated test list .split() cleanly."""
    if args.fragments_file:
        path = Path(args.fragments_file).expanduser()
    else:
        path = REPO / "asr_pipeline" / "eval" / f"clarin_{args.split}.txt"
    if not path.exists():
        sys.exit(f"fragment list not found: {path}")
    frags = path.read_text().split()
    if not frags:
        sys.exit(f"no fragments in {path}")
    return frags


def _load_gt(frags):
    gt = {}
    for fid in frags:
        rec = load_recording(EVAL / fid)
        g = load_reference_utterances(rec) if rec is not None else {}
        gt[fid] = {k: v for k, v in g.items() if v}
    return gt


def _strata(frags):
    """Recording-level tertiles. Each recording gets ONE stratum (mean of its
    fragments' composite scores), so a multi-segment recording can never leak
    into two strata (which would double-count it). Returns {recid: stratum}.
    A fragment with no composite score is a hard error — silently sinking it into
    HIGH (the old comp.get(...,9.0) default) would corrupt the one-shot number."""
    with open(EVAL / "composite_scores.csv", newline="") as fh:
        comp = {r["frag_id"]: float(r["composite"]) for r in csv.DictReader(fh)}
    missing = [f for f in frags if f not in comp]
    if missing:
        sys.exit(f"composite_scores.csv missing {len(missing)} fragment(s): "
                 f"{', '.join(missing[:5])}{' ...' if len(missing) > 5 else ''}")
    rec_comp = defaultdict(list)
    for f in frags:
        rec_comp[_recid(f)].append(comp[f])
    rec_mean = {r: sum(v) / len(v) for r, v in rec_comp.items()}
    order = sorted(rec_mean, key=lambda r: rec_mean[r])
    t = len(order) // 3
    strat = {**{r: "LOW" for r in order[:t]},
             **{r: "MID" for r in order[t:2 * t]},
             **{r: "HIGH" for r in order[2 * t:]}}
    assert len(strat) == len(order), "strata must partition the recordings"
    return strat


def per_fragment(cfg, gt, frags):
    """{fid: {cpE,cpL,cerE,cerL, mixE,mixL,mixcE,mixcL}} for a config."""
    out = {}
    for fid in frags:
        d = EVAL / fid / "sweep" / cfg
        hyp = read_per_speaker(d)
        if hyp is None or not gt[fid]:
            continue
        cp = cpwer_meeteval(gt[fid], hyp, session_id=fid)
        cc = cp_cer_meeteval(gt[fid], hyp, session_id=fid)
        row = dict(cpE=cp["cp_errors"], cpL=cp["cp_length"],
                   cerE=cc["errors"], cerL=cc["length"])
        mix = read_mixture(d)
        if mix is not None:
            mw = mimo_wer_meeteval(gt[fid], mix, session_id=fid)
            mc = mimo_cer_meeteval(gt[fid], mix, session_id=fid)
            row.update(mixE=mw["errors"], mixL=mw["length"],
                       mixcE=mc["errors"], mixcL=mc["length"])
        out[fid] = row
    return out


def by_recording(frag):
    """Collapse fragments -> recordings (sum errors/lengths)."""
    rec = defaultdict(lambda: defaultdict(float))
    for fid, r in frag.items():
        R = rec[_recid(fid)]
        for k, v in r.items():
            R[k] += v
    return rec


def micro(rows, eK, lK):
    e = sum(r[eK] for r in rows if lK in r)
    l = sum(r[lK] for r in rows if lK in r)
    return 100 * e / l if l else float("nan")


def cluster_boot_paired(recs_a, recs_b, eK, lK, rng):
    """Paired (anchor - cfg) micro-avg delta, cluster-bootstrap by recording.
    Positive => cfg better (lower error). Recordings with zero reference length
    on either side are excluded up front so a resample can never sum to a 0/0 nan
    draw (one nan draw poisons np.percentile and would flip the significance star
    ON for a meaningless delta)."""
    ids = [r for r in recs_a if r in recs_b
           and recs_a[r].get(lK, 0) > 0 and recs_b[r].get(lK, 0) > 0]
    if not ids:
        return float("nan"), float("nan"), float("nan")

    def delta(sample):
        ae = sum(recs_a[r][eK] for r in sample); al = sum(recs_a[r][lK] for r in sample)
        be = sum(recs_b[r][eK] for r in sample); bl = sum(recs_b[r][lK] for r in sample)
        return (100 * ae / al) - (100 * be / bl) if al and bl else float("nan")

    point = delta(ids)
    idx = np.arange(len(ids))
    draws = [delta([ids[i] for i in rng.choice(idx, len(idx), replace=True)])
             for _ in range(B_DRAWS)]
    draws = [d for d in draws if np.isfinite(d)]   # belt-and-suspenders: drop any nan
    if not draws:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return point, lo, hi


def _sig(lo, hi):
    """Significance star only for a finite CI that excludes 0 (a nan CI is NOT
    significant — the bug this guards against printed `*` on nan <= 0 == False)."""
    return " *" if np.isfinite(lo) and np.isfinite(hi) and not (lo <= 0 <= hi) else ""


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--anchor", default="f_oa03")
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--split", default="dev", choices=["dev", "test"],
                    help="named fragment list asr_pipeline/eval/clarin_<split>.txt")
    ap.add_argument("--fragments-file", default=None,
                    help="explicit fragment-list file (overrides --split)")
    ap.add_argument("--mixture", action="store_true", help="also print mixture floor")
    args = ap.parse_args()

    frags = load_split(args)
    gt = _load_gt(frags)
    strat = _strata(frags)                       # {recid: stratum}
    allcfgs = [args.anchor] + [c for c in args.configs if c != args.anchor]
    perfrag = {c: per_fragment(c, gt, frags) for c in allcfgs}
    recs = {c: by_recording(perfrag[c]) for c in allcfgs}

    # ---- coverage: the headline number must be PAIRED across every config ----
    cov = {c: set(recs[c]) for c in allcfgs}
    common = set.intersection(*cov.values()) if cov else set()
    if not common:
        sys.exit("no recording is covered by ALL configs — nothing to score. "
                 "Did the pipeline run for every config on this split? "
                 + "; ".join(f"{c}:{len(cov[c])} recs" for c in allcfgs))
    seen = set().union(*cov.values())
    if any(cov[c] != common for c in allcfgs):
        for c in allcfgs:
            miss = seen - cov[c]
            if miss:
                print(f"!! WARNING {c}: missing {len(miss)} recording(s): "
                      f"{', '.join(sorted(r[:8] for r in miss))}")
        print(f"!! PAIRED scoring on the {len(common)} recordings common to ALL "
              f"configs (of {len(seen)} seen). Drop is reported above, not silent.")

    strata_order = ["LOW", "MID", "HIGH", "ALL"]

    def recs_in(cfg, st):
        rr = recs[cfg]
        if st == "ALL":
            return [rr[r] for r in common]
        return [rr[r] for r in common if strat.get(r) == st]

    # ---- absolute table (cpWER / cpCER per config x stratum, paired set) ----
    print(f"\n=== Absolute cpWER / cp-CER by stratum "
          f"(split={args.fragments_file or args.split}, anchor={args.anchor}, "
          f"n={len(common)} recs) ===")
    hdr = f"{'config':18}" + "".join(f"{s:>14}" for s in strata_order)
    print(hdr); print("-" * len(hdr))
    for c in allcfgs:
        cells = []
        for st in strata_order:
            rr = recs_in(c, st)
            cells.append(f"{micro(rr,'cpE','cpL'):5.1f}/{micro(rr,'cerE','cerL'):4.1f}")
        print(f"{c:18}" + "".join(f"{x:>14}" for x in cells))
    if args.mixture:
        cells = []
        for st in strata_order:
            rr = recs_in(args.anchor, st)
            cells.append(f"{micro(rr,'mixE','mixL'):5.1f}/{micro(rr,'mixcE','mixcL'):4.1f}")
        print(f"{'MIXTURE floor':18}" + "".join(f"{x:>14}" for x in cells))
    print("(cells = cpWER / cp-CER)")

    # ---- paired cluster-bootstrap vs anchor ----
    print(f"\n=== Paired Δ vs {args.anchor} (cluster-boot by recording; + = better) ===")
    for c in args.configs:
        if c == args.anchor:
            continue
        print(f"\n[{c}]")
        for st in strata_order:
            keep = None if st == "ALL" else {r for r in common if strat.get(r) == st}
            ra = {r: recs[args.anchor][r] for r in common if keep is None or r in keep}
            rb = {r: recs[c][r] for r in common if keep is None or r in keep}
            dW, loW, hiW = cluster_boot_paired(ra, rb, "cpE", "cpL", np.random.default_rng(SEED))
            dC, loC, hiC = cluster_boot_paired(ra, rb, "cerE", "cerL", np.random.default_rng(SEED))
            print(f"  {st:4} ΔcpWER {dW:+5.1f} [{loW:+5.1f},{hiW:+5.1f}]{_sig(loW,hiW):2} | "
                  f"ΔcpCER {dC:+5.1f} [{loC:+5.1f},{hiC:+5.1f}]{_sig(loC,hiC)}")

    # ---- per-recording winner-regression veto (CER) ----
    print(f"\n=== Per-recording cp-CER vs {args.anchor} (veto: winner regressing) ===")
    for c in args.configs:
        if c == args.anchor:
            continue
        worse = []
        for rid in sorted(common):
            a = micro([recs[args.anchor][rid]], "cerE", "cerL")
            b = micro([recs[c][rid]], "cerE", "cerL")
            if b > a + 0.5:
                worse.append((rid, a, b))
        tag = "OK" if not worse else f"{len(worse)} regressed"
        print(f"  {c}: {tag}" + ("" if not worse else
              "  " + ", ".join(f"{r[:8]}({a:.0f}->{b:.0f})" for r, a, b in worse)))


if __name__ == "__main__":
    main()
