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
    mimo_wer_meeteval, mimo_cer_meeteval, orc_wer_multistream,
    orc_cer_multistream)
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


# --normalize: apply the census-vetted Polish scoring normalizer (see
# scripts/polish_scoring_normalizer.py) SYMMETRICALLY to reference and
# hypothesis text before scoring. Secondary metric only — default off, and the
# pre-registered primary numbers are always the unnormalized ones.
NORMALIZE = False


def _maybe_norm(utts):
    if not NORMALIZE or utts is None:
        return utts
    from polish_scoring_normalizer import normalize_text
    return [u._replace(text=normalize_text(u.text)) for u in utts]


def _load_gt(frags):
    gt = {}
    for fid in frags:
        rec = load_recording(EVAL / fid)
        g = load_reference_utterances(rec) if rec is not None else {}
        gt[fid] = {k: _maybe_norm(v) for k, v in g.items() if v}
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


def load_purity():
    """{(frag_id, config): (pure, total)} from score_attribution_purity.py's CSV,
    or {} if absent (the purity table/Δ are then silently omitted; the WER/CER
    gaps still print). Window counts micro-average like errors/length."""
    path = EVAL / "_attribution_purity.csv"
    if not path.exists():
        return {}
    out = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            out[(r["frag_id"], r["config"])] = (int(r["pure"]), int(r["total"]))
    return out


def per_fragment(cfg, gt, frags):
    """{fid: {cpE,cpL,cerE,cerL, mixE,mixL,mixcE,mixcL}} for a config."""
    out = {}
    for fid in frags:
        d = EVAL / fid / "sweep" / cfg
        hyp = read_per_speaker(d)
        if hyp is None or not gt[fid]:
            continue
        hyp = {k: _maybe_norm(v) for k, v in hyp.items()}
        cp = cpwer_meeteval(gt[fid], hyp, session_id=fid)
        cc = cp_cer_meeteval(gt[fid], hyp, session_id=fid)
        # Speaker-agnostic content floor of the PIPELINE output: ORC-WER on the
        # multi-stream hyp charges no attribution (meeteval optimally assigns each
        # reference utterance to a stream). cpWER - ORC = the ATTRIBUTION GAP — the
        # error caused purely by mis-filing content to the wrong speaker, which is
        # exactly what the attribution-fix work targets (and the "ORC" the forensics
        # quote). ORC-WER <= cpWER always, so the gap is non-negative. (A multi-stream
        # ORC/MIMO *CER* helper doesn't exist yet, so the gap is WER-only for now;
        # cpCER stays the headline in the absolute table.)
        orc = orc_wer_multistream(gt[fid], hyp, session_id=fid)
        # MIMO floor too: ORC assigns whole REFERENCE utterances to streams, so
        # it is sensitive to the GT's utterance granularity — a coarse GT can hide
        # an overlap mis-attribution as "content error" (gap understated). MIMO
        # splits a speaker's stream at word level, so it is granularity-robust.
        # The recoverable-attribution truth is bracketed by the two; for fixed-GT
        # config COMPARISONS the Δ is valid either way (the GT bias cancels).
        mw = mimo_wer_meeteval(gt[fid], hyp, session_id=fid)
        # CER content floor for the CER attribution gap. ORC-CER only: char-level
        # MIMO-CER is ~150x slower (meeteval's MIMO assignment explodes on char
        # tokens, ~11 s/fragment) and ORC≈MIMO on this data, so it isn't worth it.
        oc = orc_cer_multistream(gt[fid], hyp, session_id=fid)
        row = dict(cpE=cp["cp_errors"], cpL=cp["cp_length"],
                   cerE=cc["errors"], cerL=cc["length"],
                   ctE=orc["errors"], ctL=orc["length"],
                   mwE=mw["errors"], mwL=mw["length"],
                   ocE=oc["errors"], ocL=oc["length"])
        mix = _maybe_norm(read_mixture(d))
        if mix is not None:
            mw = mimo_wer_meeteval(gt[fid], mix, session_id=fid)
            mc = mimo_cer_meeteval(gt[fid], mix, session_id=fid)
            row.update(mixE=mw["errors"], mixL=mw["length"],
                       mixcE=mc["errors"], mixcL=mc["length"])
        out[fid] = row
    return out


def _check_frag_parity(label, ids_by_cfg):
    """Abort loudly if any config's set of covered fragment ids differs from
    the union across configs.

    ``per_fragment`` silently ``continue``s past a fragment whose hypothesis
    transcript is missing for THAT config. The recording-level coverage check
    further down (``cov``/``common``) cannot see the fallout: a config
    missing SOME fragments of a multi-segment recording still has an entry
    for that recording (just a partial sum), so it still counts as "covered"
    there. Left unchecked, ``by_recording`` would then sum different
    underlying content on the two sides of a paired delta. Every config must
    therefore cover the IDENTICAL set of fragments; a no-op when they do."""
    if not ids_by_cfg:
        return
    union = set().union(*ids_by_cfg.values())
    bad = {c: sorted(union - ids) for c, ids in ids_by_cfg.items() if ids != union}
    if not bad:
        return
    lines = [f"  {c}: missing {len(m)} of {len(union)} fragment(s): {', '.join(m)}"
             for c, m in bad.items()]
    sys.exit(f"fragment-level pairing mismatch ({label}) — every config must cover "
              "the IDENTICAL set of fragments, or a by-recording sum silently "
              "mis-pairs different underlying content:\n" + "\n".join(lines))


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


def cluster_boot_paired_draws(recs_a, recs_b, eK, lK, rng):
    """Like ``cluster_boot_paired`` but ALSO returns the finite bootstrap draws.

    Same point estimate and same resampling — factored out so the Holm/FDR pass
    can derive a bootstrap p-value from the draws (fraction on the wrong side of
    0, two-sided) WITHOUT re-running the bootstrap or changing the existing
    per-stratum CI output (which keeps calling ``cluster_boot_paired``). Returns
    ``(point, draws_array)``; ``draws_array`` is empty when no finite draw exists."""
    ids = [r for r in recs_a if r in recs_b
           and recs_a[r].get(lK, 0) > 0 and recs_b[r].get(lK, 0) > 0]
    if not ids:
        return float("nan"), np.array([])

    def delta(sample):
        ae = sum(recs_a[r][eK] for r in sample); al = sum(recs_a[r][lK] for r in sample)
        be = sum(recs_b[r][eK] for r in sample); bl = sum(recs_b[r][lK] for r in sample)
        return (100 * ae / al) - (100 * be / bl) if al and bl else float("nan")

    point = delta(ids)
    idx = np.arange(len(ids))
    draws = np.array([delta([ids[i] for i in rng.choice(idx, len(idx), replace=True)])
                      for _ in range(B_DRAWS)])
    return point, draws[np.isfinite(draws)]


def cluster_boot_2key(recs, eKa, lKa, eKb, lKb, rng):
    """Paired Δ = micro(a) − micro(b) of TWO metrics over the SAME recordings,
    cluster-bootstrapped by recording. For the separation-vs-mixture contrast a =
    mixture content floor, b = pipeline content floor (both inside recs[anchor]),
    so + = pipeline lower error = separation recovered content. Same recording
    resample + nan-guard as cluster_boot_paired."""
    ids = [r for r in recs if recs[r].get(lKa, 0) > 0 and recs[r].get(lKb, 0) > 0]
    if not ids:
        return float("nan"), float("nan"), float("nan")

    def delta(sample):
        ae = sum(recs[r][eKa] for r in sample); al = sum(recs[r][lKa] for r in sample)
        be = sum(recs[r][eKb] for r in sample); bl = sum(recs[r][lKb] for r in sample)
        return (100 * ae / al) - (100 * be / bl) if al and bl else float("nan")

    point = delta(ids)
    idx = np.arange(len(ids))
    draws = [delta([ids[i] for i in rng.choice(idx, len(idx), replace=True)])
             for _ in range(B_DRAWS)]
    draws = [d for d in draws if np.isfinite(d)]
    if not draws:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return point, lo, hi


def boot_pvalue(draws) -> float:
    """Two-sided bootstrap p-value for H0: paired Δ = 0.

    Standard percentile-bootstrap p: p = 2 * min(frac draws <= 0, frac draws >= 0),
    clipped to [0, 1]. A Δ whose draws sit entirely on one side of 0 gets the
    smallest resolvable p (≈ 2/B_DRAWS, never exactly 0 — the bootstrap cannot
    resolve below its resolution). Empty draws → 1.0 (cannot reject)."""
    draws = np.asarray(draws, dtype=float)
    draws = draws[np.isfinite(draws)]
    n = draws.size
    if n == 0:
        return 1.0
    frac_le = float(np.count_nonzero(draws <= 0)) / n
    frac_ge = float(np.count_nonzero(draws >= 0)) / n
    p = 2.0 * min(frac_le, frac_ge)
    # Floor at the bootstrap resolution so an all-one-side draw isn't reported p=0.
    return float(min(max(p, 1.0 / n), 1.0))


def holm_bonferroni(pvals):
    """Holm-Bonferroni step-down adjusted p-values (family-wise, conservative).

    ``pvals`` is a list of raw p-values; returns adjusted p-values in the SAME
    order. Reject H_i at level α iff adjusted p_i <= α. Monotone by construction
    (cumulative max along the sorted order)."""
    m = len(pvals)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        running = max(running, val)
        adj[i] = min(running, 1.0)
    return adj


def benjamini_hochberg(pvals):
    """Benjamini-Hochberg FDR-adjusted p-values (less conservative than Holm).

    Same order in / out as ``holm_bonferroni``. Standard step-up with the
    monotone (cumulative-min from the largest) enforcement."""
    m = len(pvals)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        val = pvals[i] * m / (rank + 1)
        prev = min(prev, val)
        adj[i] = min(prev, 1.0)
    return adj


def cluster_boot_gap(recs_a, recs_b, eK, lK, gK, gL, rng):
    """Paired Δ of the ATTRIBUTION GAP (cpWER - MIMO content floor), cluster-boot
    by recording. gap = micro(cpWER) - micro(content); Δ = gap_anchor - gap_cfg,
    so positive => cfg has the SMALLER gap (better attribution). (eK,lK) = cpWER
    error/length; (gK,gL) = content-floor error/length. Excludes recordings with
    zero reference length on either side (same nan-guard as cluster_boot_paired)."""
    ids = [r for r in recs_a if r in recs_b
           and recs_a[r].get(lK, 0) > 0 and recs_b[r].get(lK, 0) > 0]
    if not ids:
        return float("nan"), float("nan"), float("nan")

    def gap(recs, sample):
        ce = sum(recs[r][eK] for r in sample); cl = sum(recs[r][lK] for r in sample)
        ge = sum(recs[r][gK] for r in sample); gl = sum(recs[r][gL] for r in sample)
        return (100 * ce / cl - 100 * ge / gl) if cl and gl else float("nan")

    def delta(sample):
        return gap(recs_a, sample) - gap(recs_b, sample)

    point = delta(ids)
    idx = np.arange(len(ids))
    draws = [delta([ids[i] for i in rng.choice(idx, len(idx), replace=True)])
             for _ in range(B_DRAWS)]
    draws = [d for d in draws if np.isfinite(d)]
    if not draws:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return point, lo, hi


def _sig(lo, hi):
    """Significance star only for a finite CI that excludes 0 (a nan CI is NOT
    significant — the bug this guards against printed `*` on nan <= 0 == False)."""
    return " *" if np.isfinite(lo) and np.isfinite(hi) and not (lo <= 0 <= hi) else ""


def dump_per_fragment(perfrag, strat, comp, path):
    """Write per-(config, fragment) metrics so analysis isn't limited to the
    tertile tables (the adaptivity question needs per-fragment granularity).
    cpWER/cpCER carry their error/length so any regrouping (by stratum, overlap,
    recording) re-micro-averages correctly — per-fragment PERCENTAGES must never be
    plain-averaged. Floors / mixture / purity are percentages. Returns (n_cfg, n_rows)."""
    def pct(r, eK, lK):
        return round(100 * r[eK] / r[lK], 2) if r.get(lK) else ""
    cols = ["frag_id", "recid", "config", "stratum", "composite",
            "cp_wer", "cp_cer", "cp_err", "cp_len", "cer_err", "cer_len",
            "orc_wer", "mimo_wer", "orc_cer", "mix_mimo_wer", "mix_mimo_cer",
            "purity_pct"]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for cfg, frd in perfrag.items():
            for fid, r in sorted(frd.items()):
                rid = _recid(fid)
                w.writerow([
                    fid, rid, cfg, strat.get(rid, ""),
                    round(comp[fid], 3) if fid in comp else "",
                    pct(r, "cpE", "cpL"), pct(r, "cerE", "cerL"),
                    r.get("cpE", ""), r.get("cpL", ""),
                    r.get("cerE", ""), r.get("cerL", ""),
                    pct(r, "ctE", "ctL"), pct(r, "mwE", "mwL"), pct(r, "ocE", "ocL"),
                    pct(r, "mixE", "mixL"), pct(r, "mixcE", "mixcL"),
                    (round(100 * r["purE"] / r["purL"], 1) if r.get("purL") else ""),
                ])
    return len(perfrag), sum(len(v) for v in perfrag.values())


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--anchor", default="f_oa03")
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--split", default="dev", choices=["dev", "test"],
                    help="named fragment list asr_pipeline/eval/clarin_<split>.txt")
    ap.add_argument("--fragments-file", default=None,
                    help="explicit fragment-list file (overrides --split)")
    ap.add_argument("--mixture", action="store_true",
                    help="also print the mixture floor + the cluster-boot "
                         "separation-vs-mixture content-floor contrast (per stratum)")
    ap.add_argument("--perfrag-out", default=None,
                    help="per-(config,fragment) metrics CSV (default "
                         "EVAL/_rescore_perfrag_<split>.csv); saves every fragment's "
                         "cpWER/cpCER (+err/len), content floors, mixture floor, "
                         "purity, stratum, composite — so analysis isn't limited to "
                         "the tertile tables.")
    ap.add_argument("--normalize", action="store_true",
                    help="apply the census-vetted Polish scoring normalizer to "
                         "ref+hyp before scoring (SECONDARY metric; primary "
                         "numbers stay unnormalized)")
    args = ap.parse_args()
    if args.normalize:
        global NORMALIZE
        NORMALIZE = True
        print("### NORMALIZED SCORING: polish_scoring_normalizer rules applied "
              "symmetrically to reference and hypothesis (secondary metric) ###")

    frags = load_split(args)
    gt = _load_gt(frags)
    strat = _strata(frags)                       # {recid: stratum}
    allcfgs = [args.anchor] + [c for c in args.configs if c != args.anchor]
    perfrag = {c: per_fragment(c, gt, frags) for c in allcfgs}
    _check_frag_parity("cpWER/cpCER per-fragment scores",
                        {c: set(perfrag[c]) for c in allcfgs})
    if args.mixture:
        _check_frag_parity("mixture-floor per-fragment coverage",
                            {c: {fid for fid, r in perfrag[c].items() if "mixE" in r}
                             for c in allcfgs})
    purity = load_purity()
    if purity:
        for (fid, c), (pure, total) in purity.items():
            if c in perfrag and fid in perfrag[c] and total > 0:
                perfrag[c][fid]["purE"] = pure
                perfrag[c][fid]["purL"] = total
    has_purity = bool(purity)
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

    # ---- separation-vs-mixture content-floor contrast, with CI (idea R3) ----
    # The "separation helps" headline needs its own CI: the per-arm bootstrap below
    # is arm-vs-anchor, not pipeline-vs-mixture. This contrasts the anchor pipeline's
    # MIMO-WER content floor against the raw mixture's MIMO-WER, paired by recording
    # — attribution removed on BOTH sides, so it isolates *content recovery* (did
    # separation let WhisperX hear more words), the assumption-light form of the claim.
    if args.mixture:
        print(f"\n=== Separation vs mixture (MIMO-WER content floor, cluster-boot; "
              f"+ = separation recovers content) ===")
        ra = {r: recs[args.anchor][r] for r in common}
        for st in strata_order:
            sub = ra if st == "ALL" else {r: ra[r] for r in common if strat.get(r) == st}
            dW, loW, hiW = cluster_boot_2key(sub, "mixE", "mixL", "mwE", "mwL",
                                             np.random.default_rng(SEED))
            print(f"  {st:4} Δ {dW:+5.1f} [{loW:+5.1f},{hiW:+5.1f}]{_sig(loW,hiW)}")
        print(f"(Δ = mixture MIMO-WER − {args.anchor} pipeline MIMO-WER, paired by "
              "recording; + = pipeline lower = separation helps. Report per-stratum, "
              "never averaged — the effect inverts. CER floor needs pipeline MIMO-CER "
              "(slow); add if wanted.)")

    # ---- attribution-gap tables (cp{WER,CER} - content floor; ORC and MIMO) ----
    def _gap_table(title, cpe, cpl, oe, ol, me, ml, note):
        print(f"\n=== {title} (paired set; lower = better attribution) ===")
        hdr = f"{'config':18}" + "".join(f"{s:>14}" for s in strata_order)
        print(hdr); print("-" * len(hdr))
        for c in allcfgs:
            cells = []
            for st in strata_order:
                rr = recs_in(c, st)
                go = micro(rr, cpe, cpl) - micro(rr, oe, ol)
                if me is not None:
                    gm = micro(rr, cpe, cpl) - micro(rr, me, ml)
                    cells.append(f"{go:4.1f}/{gm:4.1f}")
                else:
                    cells.append(f"{go:5.1f}")
            print(f"{c:18}" + "".join(f"{x:>14}" for x in cells))
        print(note)

    _gap_table("WER attribution gap = cpWER - content floor",
               "cpE", "cpL", "ctE", "ctL", "mwE", "mwL",
               "(cells = ORCgap / MIMOgap; ORC is GT-granularity-sensitive, MIMO robust)")
    _gap_table("CER attribution gap = cpCER - ORC content floor",
               "cerE", "cerL", "ocE", "ocL", None, None,
               "(cells = cpCER - ORC-CER, character units; MIMO-CER omitted — too slow)")

    # ---- reference-free stream purity (idea #1) ----
    if has_purity:
        print(f"\n=== Stream purity % (reference-free; HIGHER = cleaner attribution) ===")
        hdr = f"{'config':18}" + "".join(f"{s:>14}" for s in strata_order)
        print(hdr); print("-" * len(hdr))
        for c in allcfgs:
            cells = [f"{micro(recs_in(c, st), 'purE', 'purL'):5.1f}"
                     for st in strata_order]
            print(f"{c:18}" + "".join(f"{x:>14}" for x in cells))
        print("(centroid self-consistency of assembled stream_A/B; window-micro %)")

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
            mk = lambda: np.random.default_rng(SEED)   # noqa: E731 (same seed each call)
            dW, loW, hiW = cluster_boot_paired(ra, rb, "cpE", "cpL", mk())
            dC, loC, hiC = cluster_boot_paired(ra, rb, "cerE", "cerL", mk())
            dWo, loWo, hiWo = cluster_boot_gap(ra, rb, "cpE", "cpL", "ctE", "ctL", mk())
            dWm, loWm, hiWm = cluster_boot_gap(ra, rb, "cpE", "cpL", "mwE", "mwL", mk())
            dCo, loCo, hiCo = cluster_boot_gap(ra, rb, "cerE", "cerL", "ocE", "ocL", mk())
            print(f"  {st:4} ΔcpWER {dW:+5.1f} [{loW:+5.1f},{hiW:+5.1f}]{_sig(loW,hiW):2} | "
                  f"ΔcpCER {dC:+5.1f} [{loC:+5.1f},{hiC:+5.1f}]{_sig(loC,hiC)}")
            print(f"       WERgap Δ ORC {dWo:+5.1f} [{loWo:+5.1f},{hiWo:+5.1f}]{_sig(loWo,hiWo):2} | "
                  f"MIMO {dWm:+5.1f} [{loWm:+5.1f},{hiWm:+5.1f}]{_sig(loWm,hiWm)}")
            print(f"       CERgap Δ ORC {dCo:+5.1f} [{loCo:+5.1f},{hiCo:+5.1f}]{_sig(loCo,hiCo)}")
            if has_purity:
                # Higher purity = better, so flip args (cfg - anchor): + = better.
                dP, loP, hiP = cluster_boot_paired(rb, ra, "purE", "purL", mk())
                print(f"       Δpurity {dP:+5.1f} [{loP:+5.1f},{hiP:+5.1f}]{_sig(loP,hiP)}")

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

    # ---- multiple-comparison correction across all paired arms (ALL stratum) ----
    # SWEEP_DESIGN §3.4 pre-registers Holm-Bonferroni as the SELECTION GATE (FDR/BH
    # reported alongside). The per-arm CIs above are the (uncorrected) effect
    # estimates; this table is the family-wise corrected significance over the
    # whole arm set, on the full (ALL) paired set. cpCER is the primary, cpWER the
    # confirming secondary (§3.4). p is the two-sided bootstrap p from the SAME
    # cluster-bootstrap draws as the CIs above; an arm "clears" the gate only when
    # its corrected p stays below the level AND its Δ is favourable (> 0).
    arms = [c for c in args.configs if c != args.anchor]
    if arms:
        ra_all = {r: recs[args.anchor][r] for r in common}
        rows = []
        for c in arms:
            rb_all = {r: recs[c][r] for r in common}
            dC, drC = cluster_boot_paired_draws(
                ra_all, rb_all, "cerE", "cerL", np.random.default_rng(SEED))
            dW, drW = cluster_boot_paired_draws(
                ra_all, rb_all, "cpE", "cpL", np.random.default_rng(SEED))
            rows.append({"cfg": c, "dC": dC, "pC": boot_pvalue(drC),
                         "dW": dW, "pW": boot_pvalue(drW)})
        holmC = holm_bonferroni([r["pC"] for r in rows])
        bhC = benjamini_hochberg([r["pC"] for r in rows])
        holmW = holm_bonferroni([r["pW"] for r in rows])
        bhW = benjamini_hochberg([r["pW"] for r in rows])
        print(f"\n=== Multiple-comparison correction vs {args.anchor} "
              f"(ALL stratum, {len(arms)} arms; + = better) ===")
        print(f"{'config':18}{'ΔcpCER':>8}{'pC':>8}{'HolmC':>8}{'BH_C':>8}"
              f"{'ΔcpWER':>8}{'pW':>8}{'HolmW':>8}{'BH_W':>8}")
        print("-" * 82)
        for i, r in enumerate(rows):
            def star(d, p):   # gate: favourable Δ AND corrected p < 0.05
                return "*" if (d > 0 and p < 0.05) else " "
            print(f"{r['cfg']:18}{r['dC']:+8.1f}{r['pC']:8.3f}"
                  f"{holmC[i]:8.3f}{bhC[i]:8.3f}{star(r['dC'], holmC[i])}"
                  f"{r['dW']:+7.1f}{r['pW']:8.3f}{holmW[i]:8.3f}{bhW[i]:8.3f}"
                  f"{star(r['dW'], holmW[i])}")
        print("(p = two-sided bootstrap p from the cluster-boot draws; Holm = "
              "family-wise, BH = FDR. * = favourable Δ AND Holm-adjusted p < 0.05.)")

    # ---- per-fragment dump (analysis beyond the tertile tables) ----
    with open(EVAL / "composite_scores.csv", newline="") as fh:
        comp = {r["frag_id"]: float(r["composite"]) for r in csv.DictReader(fh)}
    tag = Path(args.fragments_file).stem if args.fragments_file else args.split
    out = (Path(args.perfrag_out).expanduser() if args.perfrag_out
           else EVAL / f"_rescore_perfrag_{tag}.csv")
    nconf, nrows = dump_per_fragment(perfrag, strat, comp, out)
    print(f"\n[per-fragment] {nrows} rows ({nconf} configs) -> {out}")


if __name__ == "__main__":
    main()
