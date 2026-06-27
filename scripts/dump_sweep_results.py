"""Dump ALL pipeline-sweep results to a tidy/long source-of-truth CSV (+ pivots).

The reusable, deterministic generator behind ``sweep_results_tidy.csv`` — one row
per (config, fragment), every Layer-3 ASR metric as its own column, plus the raw
error/reference counts so any micro-average reproduces exactly (Σerrors / Σref,
never mean-of-rates). Two wide config×fragment pivots (cpWER, cpCER) are derived
from it, each carrying a trailing ``MICRO_ALL`` column = the exact micro-average
over all fragments from the count columns.

The config set scored = ``GROUPS["definitive"] ∪ GROUPS["phase2"] ∪ {"baseline"}``
from ``scripts/sweep_pipeline.py`` (parsed statically — no torch import). Scoring
*reuses the eval module* (``asr_pipeline.eval.metrics`` / ``layer3`` /
``recordings``) and mirrors ``scripts/rescore_stratified.py:per_fragment`` exactly,
so the per-config micro-averages equal the rescorer's ALL-stratum numbers. The
recording-level acoustic stratum (LOW/MID/HIGH) comes from ``rescore_stratified._strata``.

Outputs (under ``~/datasets/eval/clarin_fragments/`` by default):
  - ``sweep_results_tidy.csv``   — the tidy source of truth.
  - ``sweep_cpwer_wide.csv``     — config×fragment cpWER pivot + MICRO_ALL.
  - ``sweep_cpcer_wide.csv``     — config×fragment cpCER pivot + MICRO_ALL.

Usage::

    python scripts/dump_sweep_results.py                 # dev split (default)
    python scripts/dump_sweep_results.py --split test    # held-out test later
    python scripts/dump_sweep_results.py --validate      # also print micro-avg check
"""
from __future__ import annotations

import argparse
import ast
import csv
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# Eval-module scoring — the SAME functions rescore_stratified.per_fragment uses,
# so aggregates are consistent (mandatory, per the task). meeteval/rapidfuzz only;
# no torch is imported on this path.
from asr_pipeline.eval.metrics import (                                   # noqa: E402
    cpwer_meeteval, cp_cer_meeteval,
    mimo_wer_meeteval, mimo_cer_meeteval,
    orc_wer_meeteval, orc_wer_multistream, orc_cer_multistream,
)
from asr_pipeline.eval.layer3 import read_per_speaker, read_mixture       # noqa: E402
from asr_pipeline.eval.recordings import (                                # noqa: E402
    load_recording, load_reference_utterances,
)
from scripts.rescore_stratified import _strata                            # noqa: E402

EVAL = Path("~/datasets/eval/clarin_fragments").expanduser()
SWEEP_PIPELINE = REPO / "scripts" / "sweep_pipeline.py"


# ---------------------------------------------------------------------------
# Config registry — parsed STATICALLY from sweep_pipeline.py (avoids importing
# the module, which pulls in torch). CONFIGS / GROUPS are plain literal dicts.
# ---------------------------------------------------------------------------
def load_registry() -> tuple[dict, dict]:
    """`(CONFIGS, GROUPS)` from sweep_pipeline.py via ast.literal_eval."""
    tree = ast.parse(SWEEP_PIPELINE.read_text())
    out: dict[str, object] = {}
    for node in tree.body:
        tgt = val = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            tgt, val = node.target.id, node.value
        elif (isinstance(node, ast.Assign) and len(node.targets) == 1
              and isinstance(node.targets[0], ast.Name)):
            tgt, val = node.targets[0].id, node.value
        if tgt in ("CONFIGS", "GROUPS") and val is not None:
            out[tgt] = ast.literal_eval(val)
    if "CONFIGS" not in out or "GROUPS" not in out:
        sys.exit("could not parse CONFIGS/GROUPS from sweep_pipeline.py")
    return out["CONFIGS"], out["GROUPS"]


def selected_configs(groups: dict) -> list[str]:
    """The scored set = definitive ∪ phase2 ∪ {baseline}, baseline first then
    deterministic (sorted) so a re-run gives byte-identical row order."""
    sel = set(groups["definitive"]) | set(groups["phase2"]) | {"baseline"}
    return ["baseline"] + sorted(sel - {"baseline"})


# ---------------------------------------------------------------------------
# Per-config knob extraction. Defaults are the committed default.yaml / dataclass
# values (the pipeline base for an un-overridden knob); a CONFIGS override wins.
# Kept as a static map so this script needs no torch / config-build. Verified
# against asr_pipeline/configs/default.yaml + asr_pipeline/config.py dataclasses.
# ---------------------------------------------------------------------------
# (dotted override path, output column, default value)
KNOBS = [
    ("enhancement.observation_mix_ratio", "oa_ratio", 0.0),
    ("transcription.model_name", "asr_model", "large-v2"),
    ("post_separation_processing.backend", "bwe_backend", "ap_bwe"),
    ("enhancement.backend", "enh_backend", "frcrn_se_16k"),
    ("enhancement.enabled", "enh_enabled", True),
    ("diarization.embedding", "diar_embedding", None),  # None = stock pyannote 3.1
    ("relabel.enabled", "relabel_enabled", False),
    ("relabel.source", "relabel_source", "solos"),
    ("separation.vad_threshold", "vad_threshold", 0.25),
]


def config_knobs(overrides: dict) -> dict:
    """{column: value} for the inventoried knobs, override-then-default.

    `diar_embedding` reports the literal ``None`` default as the string
    ``"stock_3.1"`` so a CSV reader never confuses an empty cell with the stock
    pipeline; `relabel_source` is reported only when relabel is enabled (else the
    field is inert and would mislead a groupby)."""
    out = {}
    for path, col, default in KNOBS:
        out[col] = overrides.get(path, default)
    if out["diar_embedding"] is None:
        out["diar_embedding"] = "stock_3.1"
    if not out["relabel_enabled"]:
        out["relabel_source"] = ""   # inert when relabel is off
    return out


# ---------------------------------------------------------------------------
# Scoring. per_fragment_metrics mirrors rescore_stratified.per_fragment's metric
# set + a couple extras the tidy CSV carries (tcpWER, mixture ORC/MIMO floors).
# ---------------------------------------------------------------------------
def _rate(err, length):
    """100*err/length, or None when there are no reference units (empty GT) —
    a blank rate, never a fabricated 0 (the count columns still record 0/0)."""
    return round(100.0 * err / length, 4) if length else None


def per_fragment_metrics(cfg: str, fid: str, ref: dict) -> dict | None:
    """All L3 metrics + raw counts for one (config, fragment). None when the
    pipeline produced no per-speaker hyp (then the caller records the gap)."""
    d = EVAL / fid / "sweep" / cfg
    hyp = read_per_speaker(d)
    if hyp is None:
        return None

    cp = cpwer_meeteval(ref, hyp, session_id=fid)        # cpWER (+ tcpWER)
    cc = cp_cer_meeteval(ref, hyp, session_id=fid)       # cpCER
    orc = orc_wer_multistream(ref, hyp, session_id=fid)  # WER content floor (ORC)
    mw = mimo_wer_meeteval(ref, hyp, session_id=fid)     # WER content floor (MIMO)
    oc = orc_cer_multistream(ref, hyp, session_id=fid)   # CER content floor (ORC)

    cp_e, cp_l = cp["cp_errors"], cp["cp_length"]
    cer_e, cer_l = cc["errors"], cc["length"]
    tcp_e, tcp_l = cp["tcp_errors"], cp["tcp_length"]
    ct_e, ct_l = orc["errors"], orc["length"]
    mw_e, mw_l = mw["errors"], mw["length"]
    oc_e, oc_l = oc["errors"], oc["length"]

    row = {
        # cpWER / cpCER (speaker-attributed headline) + raw counts.
        "cpwer": _rate(cp_e, cp_l), "cpwer_errors": cp_e, "cpwer_ref_words": cp_l,
        "cpcer": _rate(cer_e, cer_l), "cpcer_errors": cer_e, "cpcer_ref_chars": cer_l,
        # tcpWER (time-constrained; collar 5 s).
        "tcpwer": _rate(tcp_e, tcp_l), "tcpwer_errors": tcp_e, "tcpwer_ref_words": tcp_l,
        # Attribution-blind content floors of the multi-stream pipeline hyp.
        "orcwer": _rate(ct_e, ct_l), "orcwer_errors": ct_e, "orcwer_ref_words": ct_l,
        "mimower": _rate(mw_e, mw_l), "mimower_errors": mw_e, "mimower_ref_words": mw_l,
        "orccer": _rate(oc_e, oc_l), "orccer_errors": oc_e, "orccer_ref_chars": oc_l,
        # Attribution gaps (cp - content floor); blank if either side has no ref.
        "wer_attr_gap_orc": (round(100.0 * cp_e / cp_l - 100.0 * ct_e / ct_l, 4)
                             if cp_l and ct_l else None),
        "wer_attr_gap_mimo": (round(100.0 * cp_e / cp_l - 100.0 * mw_e / mw_l, 4)
                              if cp_l and mw_l else None),
        "cer_attr_gap_orc": (round(100.0 * cer_e / cer_l - 100.0 * oc_e / oc_l, 4)
                             if cer_l and oc_l else None),
    }

    # Mixture baseline floor (single-stream Whisper on the raw mix), when written.
    mix = read_mixture(d)
    if mix is not None:
        m_orc = orc_wer_meeteval(ref, mix, session_id=fid)
        m_mimo = mimo_wer_meeteval(ref, mix, session_id=fid)
        m_cer = mimo_cer_meeteval(ref, mix, session_id=fid)
        row.update(
            mix_orcwer=_rate(m_orc["errors"], m_orc["length"]),
            mix_orcwer_errors=m_orc["errors"], mix_orcwer_ref_words=m_orc["length"],
            mix_mimower=_rate(m_mimo["errors"], m_mimo["length"]),
            mix_mimower_errors=m_mimo["errors"], mix_mimower_ref_words=m_mimo["length"],
            mix_mimocer=_rate(m_cer["errors"], m_cer["length"]),
            mix_mimocer_errors=m_cer["errors"], mix_mimocer_ref_chars=m_cer["length"],
        )
    else:
        for k in ("mix_orcwer", "mix_orcwer_errors", "mix_orcwer_ref_words",
                  "mix_mimower", "mix_mimower_errors", "mix_mimower_ref_words",
                  "mix_mimocer", "mix_mimocer_errors", "mix_mimocer_ref_chars"):
            row[k] = None
    return row


# Optional reference-free attribution purity, joined per (fragment, config) from
# score_attribution_purity.py's CSV when present (else the columns stay blank).
def load_purity() -> dict:
    path = EVAL / "_attribution_purity.csv"
    if not path.exists():
        return {}
    out = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            out[(r["frag_id"], r["config"])] = (int(r["pure"]), int(r["total"]))
    return out


def load_acoustic() -> dict:
    """{frag_id: {brouhaha_snr, dnsmos_ovr, squim_pesq}} cheap acoustic features."""
    path = EVAL / "acoustic_scores.csv"
    if not path.exists():
        return {}
    keep = ("brouhaha_snr", "dnsmos_ovr", "squim_pesq")
    out = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            out[r["frag_id"]] = {k: r.get(k, "") for k in keep}
    return out


def load_composite() -> dict:
    with open(EVAL / "composite_scores.csv", newline="") as fh:
        return {r["frag_id"]: float(r["composite"]) for r in csv.DictReader(fh)}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
# Tidy column order: identifiers, knobs, then metrics (rate, errors, ref).
ID_COLS = ["config", "fragment", "recording", "stratum", "composite",
           "brouhaha_snr", "dnsmos_ovr", "squim_pesq"]
KNOB_COLS = [col for _, col, _ in KNOBS]
METRIC_COLS = [
    "cpwer", "cpwer_errors", "cpwer_ref_words",
    "cpcer", "cpcer_errors", "cpcer_ref_chars",
    "tcpwer", "tcpwer_errors", "tcpwer_ref_words",
    "orcwer", "orcwer_errors", "orcwer_ref_words",
    "mimower", "mimower_errors", "mimower_ref_words",
    "orccer", "orccer_errors", "orccer_ref_chars",
    "wer_attr_gap_orc", "wer_attr_gap_mimo", "cer_attr_gap_orc",
    "purity_pure", "purity_total", "purity_pct",
    "mix_orcwer", "mix_orcwer_errors", "mix_orcwer_ref_words",
    "mix_mimower", "mix_mimower_errors", "mix_mimower_ref_words",
    "mix_mimocer", "mix_mimocer_errors", "mix_mimocer_ref_chars",
]
ALL_COLS = ID_COLS + KNOB_COLS + METRIC_COLS


def load_split(split: str) -> list[str]:
    path = REPO / "asr_pipeline" / "eval" / f"clarin_{split}.txt"
    if not path.exists():
        sys.exit(f"fragment list not found: {path}")
    frags = path.read_text().split()
    if not frags:
        sys.exit(f"no fragments in {path}")
    return frags


def build_rows(configs, frags, gt, strat, comp, acoustic, purity):
    """Tidy rows for every (config, fragment); also collect the missing pairs."""
    rows, missing = [], []
    for cfg in configs:
        overrides = CONFIGS.get(cfg, {})
        knobs = config_knobs(overrides)
        for fid in frags:
            recid = fid.split("__")[0]
            base = {
                "config": cfg, "fragment": fid, "recording": recid,
                "stratum": strat.get(recid, ""), "composite": comp.get(fid, ""),
                **{k: acoustic.get(fid, {}).get(k, "") for k in
                   ("brouhaha_snr", "dnsmos_ovr", "squim_pesq")},
                **knobs,
            }
            ref = gt.get(fid, {})
            metrics = per_fragment_metrics(cfg, fid, ref) if ref else None
            if metrics is None:
                missing.append((cfg, fid))
                # Still emit a row (no silent drop) with blank metrics.
                metrics = {c: None for c in METRIC_COLS}
            else:
                pure_total = purity.get((fid, cfg))
                if pure_total:
                    pure, total = pure_total
                    metrics["purity_pure"] = pure
                    metrics["purity_total"] = total
                    metrics["purity_pct"] = (round(100.0 * pure / total, 4)
                                             if total else None)
                else:
                    metrics["purity_pure"] = metrics["purity_total"] = None
                    metrics["purity_pct"] = None
            rows.append({**base, **{c: metrics.get(c) for c in METRIC_COLS}})
    return rows, missing


def micro_pivot(rows, rate_col, err_col, ref_col, configs, frags):
    """Wide config×fragment pivot of `rate_col` + a trailing exact MICRO_ALL
    column (100*Σerr/Σref over all the config's fragments)."""
    cell = {(r["config"], r["fragment"]): r[rate_col] for r in rows}
    err = {(r["config"], r["fragment"]): r[err_col] for r in rows}
    ref = {(r["config"], r["fragment"]): r[ref_col] for r in rows}
    out = []
    for cfg in configs:
        row = {"config": cfg}
        se = sl = 0
        for fid in frags:
            row[fid] = cell.get((cfg, fid))
            e, l = err.get((cfg, fid)), ref.get((cfg, fid))
            if e is not None and l is not None:
                se += e
                sl += l
        row["MICRO_ALL"] = round(100.0 * se / sl, 4) if sl else None
        out.append(row)
    return out


def write_csv(path: Path, fieldnames, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def validate(rows, configs):
    """Print per-config micro-avg cpWER/cpCER (Σerr/Σref over all frags) for the
    spot-check configs the task names, next to the rescorer's ALL-stratum numbers."""
    targets = {"baseline": (25.6, 16.6), "dr_refineplus": (22.6, 15.3),
               "dr_oa050": (21.8, 14.6), "dr_oa070": (19.6, 13.1)}
    agg = defaultdict(lambda: [0, 0, 0, 0])  # cpwer_e, cpwer_l, cpcer_e, cpcer_l
    for r in rows:
        a = agg[r["config"]]
        if r["cpwer_errors"] is not None:
            a[0] += r["cpwer_errors"]; a[1] += r["cpwer_ref_words"]
            a[2] += r["cpcer_errors"]; a[3] += r["cpcer_ref_chars"]
    print("\n=== VALIDATION: tidy micro-avg vs rescore_stratified ALL-stratum ===")
    print(f"{'config':16}{'cpWER':>8}{'(resc)':>8}{'cpCER':>8}{'(resc)':>8}  match")
    ok = True
    for cfg, (rw, rc) in targets.items():
        a = agg.get(cfg)
        if not a or a[1] == 0:
            print(f"{cfg:16}  MISSING from scored set")
            ok = False
            continue
        w = 100.0 * a[0] / a[1]
        c = 100.0 * a[2] / a[3]
        m = abs(w - rw) <= 0.15 and abs(c - rc) <= 0.15
        ok = ok and m
        print(f"{cfg:16}{w:8.1f}{rw:8.1f}{c:8.1f}{rc:8.1f}  {'OK' if m else 'MISMATCH'}")
    print("ALL MATCH" if ok else "SOME MISMATCH — investigate")
    return ok


CONFIGS, GROUPS = load_registry()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", default="dev", choices=["dev", "test"],
                    help="fragment list asr_pipeline/eval/clarin_<split>.txt")
    ap.add_argument("--out-dir", default=str(EVAL),
                    help="output directory (default: the eval root)")
    ap.add_argument("--validate", action="store_true",
                    help="also print the micro-avg spot-check vs the rescorer")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = selected_configs(GROUPS)
    frags = load_split(args.split)
    strat = _strata(frags)                       # {recid: LOW/MID/HIGH}
    comp = load_composite()
    acoustic = load_acoustic()
    purity = load_purity()

    gt = {}
    for fid in frags:
        rec = load_recording(EVAL / fid)
        g = load_reference_utterances(rec) if rec is not None else {}
        gt[fid] = {k: v for k, v in g.items() if v}

    print(f"scoring {len(configs)} configs × {len(frags)} fragments "
          f"({len(configs) * len(frags)} pairs) on split={args.split} ...",
          flush=True)
    rows, missing = build_rows(configs, frags, gt, strat, comp, acoustic, purity)

    tidy_path = out_dir / "sweep_results_tidy.csv"
    write_csv(tidy_path, ALL_COLS, rows)

    cpwer_wide = micro_pivot(rows, "cpwer", "cpwer_errors", "cpwer_ref_words",
                             configs, frags)
    cpcer_wide = micro_pivot(rows, "cpcer", "cpcer_errors", "cpcer_ref_chars",
                             configs, frags)
    write_csv(out_dir / "sweep_cpwer_wide.csv", ["config", *frags, "MICRO_ALL"],
              cpwer_wide)
    write_csv(out_dir / "sweep_cpcer_wide.csv", ["config", *frags, "MICRO_ALL"],
              cpcer_wide)

    print(f"wrote {tidy_path} ({len(rows)} rows, {len(ALL_COLS)} cols)")
    print(f"wrote sweep_cpwer_wide.csv / sweep_cpcer_wide.csv "
          f"({len(configs)} rows, {len(frags) + 2} cols)")
    if missing:
        print(f"\n!! {len(missing)} (config, fragment) pairs had no pipeline "
              f"per-speaker output (row kept, metrics blank):")
        for cfg, fid in missing:
            print(f"   {cfg}  {fid}")
    else:
        print("coverage: every (config, fragment) pair has pipeline output.")

    if args.validate:
        validate(rows, configs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
