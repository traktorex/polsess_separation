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
from asr_pipeline.eval.metrics import per_fragment_metrics                 # noqa: E402
from asr_pipeline.eval.layer3 import read_per_speaker, read_mixture       # noqa: E402
from asr_pipeline.eval.recordings import (                                # noqa: E402
    load_recording, load_reference_utterances)
# The campaign statistics (recording-clustered bootstrap, Holm/BH gate,
# micro-average, tertile strata) live in the package now (asr_pipeline/eval/
# stats.py), unit-tested there; this script is their CLI driver. Re-imported
# names keep the module's public surface (and tests/test_rescore_stratified.py)
# unchanged.
from asr_pipeline.eval.stats import (                                     # noqa: E402,F401
    B_DRAWS, SEED, assign_strata, benjamini_hochberg, boot_pvalue,
    cluster_boot_2key, cluster_boot_gap, cluster_boot_paired,
    cluster_boot_paired_draws, holm_bonferroni, micro, recording_means,
    significance_star as _sig)
from scripts.eval_harness import eval_root, load_purity, load_split       # noqa: E402

EVAL = eval_root()


def _recid(fid): return fid.split("__")[0]


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
    """Recording-level acoustic-complexity tertiles for this fragment set.

    Loads the per-fragment composite scores from disk, then delegates the pure
    collapse + tertile assignment to the package
    (:func:`asr_pipeline.eval.stats.recording_means` /
    :func:`~asr_pipeline.eval.stats.assign_strata`). A fragment with no composite
    score is a hard error — silently sinking it into HIGH (the old
    comp.get(...,9.0) default) would corrupt the one-shot number. Returns
    {recid: stratum}."""
    with open(EVAL / "composite_scores.csv", newline="") as fh:
        comp = {r["frag_id"]: float(r["composite"]) for r in csv.DictReader(fh)}
    missing = [f for f in frags if f not in comp]
    if missing:
        sys.exit(f"composite_scores.csv missing {len(missing)} fragment(s): "
                 f"{', '.join(missing[:5])}{' ...' if len(missing) > 5 else ''}")
    rec_mean = recording_means({f: comp[f] for f in frags}, _recid)
    return assign_strata(rec_mean)


def per_fragment(cfg, gt, frags):
    """{fid: {cpE,cpL,cerE,cerL, mixE,mixL,mixcE,mixcL}} for a config.

    The meeteval calls live in ``asr_pipeline.eval.per_fragment_metrics`` (shared
    with dump_sweep_results / sweep_pipeline); here we pull out the exact counts
    the rescorer has always used. The content floors: ORC-WER/ORC-CER on the
    multi-stream hyp charge no attribution (meeteval optimally routes each
    reference utterance to a stream), so ``cpWER - ORC`` is the ATTRIBUTION GAP;
    MIMO-WER is the granularity-robust bracket on it. The mixture floor is scored
    with MIMO (mixE/mixcE) — its GT-fault robustness matters for the raw-mix
    baseline. (The shared helper also computes the mixture ORC-WER floor, which
    the rescorer does not report; it is ignored here — no reported number changes.)
    """
    out = {}
    for fid in frags:
        d = EVAL / fid / "sweep" / cfg
        hyp = read_per_speaker(d)
        if hyp is None or not gt[fid]:
            continue
        hyp = {k: _maybe_norm(v) for k, v in hyp.items()}
        mix = _maybe_norm(read_mixture(d))
        m = per_fragment_metrics(gt[fid], hyp, session_id=fid, mix=mix)
        cp, cc, orc, mw, oc = m["cp"], m["cpcer"], m["orc"], m["mimo"], m["orccer"]
        # The meeteval DP-cap guard (eval/metrics.py) can skip any floor metric
        # on long recordings (returns None; cpWER/cpCER always computed).
        # Contribute nothing to that metric's cells rather than crash —
        # loudly, never silently.
        skipped = [n for n, v in (("ORC", orc), ("MIMO", mw), ("ORC-CER", oc))
                   if v is None]
        if skipped:
            print(f"[note] {cfg}/{fid}: {'/'.join(skipped)} skipped by DP-cap "
                  f"guard — those aggregates exclude this fragment")
            zero = {"errors": 0, "length": 0}
            orc, mw, oc = orc or zero, mw or zero, oc or zero
        row = dict(cpE=cp["cp_errors"], cpL=cp["cp_length"],
                   cerE=cc["errors"], cerL=cc["length"],
                   ctE=orc["errors"], ctL=orc["length"],
                   mwE=mw["errors"], mwL=mw["length"],
                   ocE=oc["errors"], ocL=oc["length"])
        if mix is not None:
            mm, mc = m["mix_mimo"], m["mix_cer"]
            row.update(mixE=mm["errors"], mixL=mm["length"],
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

    frags = load_split(args.split, args.fragments_file)
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
    purity = load_purity(EVAL)
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
