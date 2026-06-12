"""Build the candidate decision menu for the CLARIN eval-set extension.

Reads the candidate manifest + the candidate acoustic scores (produced by
``mine_clarin_candidates.py`` then ``score_fragment_acoustics.py --root
<candidate tree>``), computes a composite that is **directly comparable to the
existing 128-fragment eval set**, and writes the author-facing menu
``CANDIDATES_REPORT.md``.

Composite comparability — the load-bearing design choice
--------------------------------------------------------
The composite is the z-mean of ``-dnsmos_sig`` and ``-brouhaha_snr`` (higher =
harder/noisier: both raw metrics rise with cleaner audio, so we negate them).
The z-normalisation mean/std are taken **from the REFERENCE 128-fragment run**
(``~/datasets/eval/clarin_fragments/acoustic_scores.csv``), NOT re-normalised
over the candidates. A candidate's composite is therefore on the exact same
scale as every existing fragment's — a candidate at composite 1.5 is as hard as
an existing fragment at 1.5. Re-normalising over the candidate pool would make
the numbers look comparable while silently shifting the origin and unit; we do
not do that. (The existing fragments' composites in the comparison histogram are
computed the same way, from those same reference stats, so the two histograms
share an axis.)

This is a pure-analysis step: no audio, no models. It only reads CSVs.

Author policy reflected in the menu (see the task brief):
  - NEW authors are first-class (new speakers) → listed first.
  - USED-author candidates are second-class (new conditions, same speaker) but
    still listed.
  - Overlap is a floor + tie bonus, not the ranking axis. The menu ranks by
    composite (acoustic diversity), and shows overlap so the author can prefer
    higher-overlap candidates among acoustically similar ones.
  - Nothing auto-enters the eval set; the "suggested additions" are explicitly
    labelled as starting points for the author's listening pass.

Usage::

    python scripts/build_candidate_report.py
    python scripts/build_candidate_report.py \\
        --manifest ~/datasets/clarin_fragment_candidates/candidates_manifest.csv \\
        --candidate-scores ~/datasets/clarin_fragment_candidates/acoustic_scores.csv \\
        --reference-scores ~/datasets/eval/clarin_fragments/acoustic_scores.csv \\
        --out ~/datasets/clarin_fragment_candidates/CANDIDATES_REPORT.md
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

OUTPUT_ROOT = Path("~/datasets/clarin_fragment_candidates").expanduser()
DEFAULT_MANIFEST = OUTPUT_ROOT / "candidates_manifest.csv"
DEFAULT_CAND_SCORES = OUTPUT_ROOT / "acoustic_scores.csv"
DEFAULT_REF_SCORES = Path(
    "~/datasets/eval/clarin_fragments/acoustic_scores.csv"
).expanduser()
DEFAULT_OUT = OUTPUT_ROOT / "CANDIDATES_REPORT.md"

# The two raw metrics the composite is built from. Both RISE with cleaner audio,
# so the composite negates them → higher composite = harder.
COMPOSITE_METRICS = ("dnsmos_sig", "brouhaha_snr")


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------


def _to_float(v) -> float:
    try:
        f = float(v)
        return f if math.isfinite(f) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def load_scores(path: Path) -> dict:
    """frag_id -> {metric: float}. Missing/blank cells become nan."""
    out: dict = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            fid = r.get("frag_id", "")
            if not fid:
                continue
            out[fid] = {k: _to_float(v) for k, v in r.items() if k != "frag_id"}
    return out


def load_manifest(path: Path) -> dict:
    """cand_id -> manifest row dict (raw strings)."""
    out: dict = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[r["cand_id"]] = r
    return out


# ---------------------------------------------------------------------------
# Composite — z-normalised against the REFERENCE 128, NOT the candidates
# ---------------------------------------------------------------------------


@dataclass
class RefStats:
    mean: dict           # metric -> reference mean
    std: dict            # metric -> reference std
    n: int               # reference rows used


def reference_stats(ref_scores: dict) -> RefStats:
    """Mean/std of each composite metric over the reference 128 set. These are
    the ONLY normalisation constants used for every composite in this report —
    candidate and reference alike — so all composites share one scale."""
    mean: dict = {}
    std: dict = {}
    n = 0
    for m in COMPOSITE_METRICS:
        vals = np.array([row.get(m, float("nan")) for row in ref_scores.values()])
        vals = vals[np.isfinite(vals)]
        n = max(n, len(vals))
        mean[m] = float(vals.mean()) if vals.size else float("nan")
        std[m] = float(vals.std()) if vals.size else float("nan")
    return RefStats(mean=mean, std=std, n=n)


def composite(row: dict, ref: RefStats) -> float:
    """z-mean of -dnsmos_sig, -brouhaha_snr using the REFERENCE mean/std.

    Higher = harder. A metric whose cell is nan is dropped from that fragment's
    mean (the other still contributes); both nan → nan.
    """
    zs = []
    for m in COMPOSITE_METRICS:
        v = row.get(m, float("nan"))
        mu, sd = ref.mean[m], ref.std[m]
        if not (math.isfinite(v) and math.isfinite(mu) and math.isfinite(sd)) or sd == 0:
            continue
        zs.append(-(v - mu) / sd)  # negate: higher raw = easier → lower composite
    if not zs:
        return float("nan")
    return float(np.mean(zs))


# ---------------------------------------------------------------------------
# Text histogram
# ---------------------------------------------------------------------------


def text_histogram(values, label: str, lo: float, hi: float,
                   nbins: int = 12, width: int = 40) -> list[str]:
    vals = np.array([v for v in values if math.isfinite(v)])
    lines = [f"{label} (n={len(vals)}):"]
    if vals.size == 0:
        return lines + ["  (no finite values)"]
    edges = np.linspace(lo, hi, nbins + 1)
    counts, _ = np.histogram(vals, bins=edges)
    peak = max(counts.max(), 1)
    for i in range(nbins):
        bar = "#" * int(round(width * counts[i] / peak))
        lines.append(f"  [{edges[i]:+5.2f},{edges[i+1]:+5.2f})  "
                     f"{counts[i]:3d} {bar}")
    return lines


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _fmt(v, nd: int = 3) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "nan"
    return f"{v:.{nd}f}"


def build_report(manifest: dict, cand_scores: dict, ref_scores: dict,
                 ref: RefStats) -> tuple[str, list[str]]:
    """Returns (markdown, list_of_exclusion_notes)."""
    exclusions: list[str] = []
    L: list[str] = []
    a = L.append

    # --- Per-candidate composite (reference-normalised) ---
    cand_comp: dict = {}
    scored_but_no_manifest = sorted(set(cand_scores) - set(manifest))
    manifest_but_no_score = sorted(set(manifest) - set(cand_scores))
    for cid in scored_but_no_manifest:
        exclusions.append(f"scored but absent from manifest: {cid} (skipped)")
    for cid in manifest_but_no_score:
        exclusions.append(
            f"in manifest but not scored: {cid} — composite=nan, listed last")

    for cid, mrow in manifest.items():
        srow = cand_scores.get(cid, {})
        comp = composite(srow, ref) if srow else float("nan")
        cand_comp[cid] = comp
        if srow and not math.isfinite(comp):
            exclusions.append(
                f"composite=nan for {cid}: both {COMPOSITE_METRICS} cells "
                "missing/non-finite in the scores CSV")

    # Existing-set composites for the comparison histogram (same ref stats).
    ref_comp = {fid: composite(row, ref) for fid, row in ref_scores.items()}

    # ----------------------------------------------------------------------
    a("# CLARIN candidate decision menu\n")
    a(f"Generated: **{datetime.now().isoformat(timespec='seconds')}**\n")
    a("This menu ranks NEW fragment candidates mined from the CLARIN "
      "recordings the\n"
      "128-fragment eval set has not used. **Nothing here has entered the eval "
      "set.**\n"
      "It is a listening shortlist for the author.\n")

    # --- Composite methodology (stated up front, it is load-bearing) ---
    a("## How the composite is computed (read this before trusting the numbers)\n")
    a(f"Composite = z-mean of `-dnsmos_sig` and `-brouhaha_snr`. Both raw "
      f"metrics\n"
      f"rise with *cleaner* audio, so they are negated → **higher composite = "
      f"harder /\n"
      f"noisier**. The z-normalisation mean/std come from the **reference "
      f"{ref.n}-fragment\n"
      f"eval run**, NOT from the candidate pool, so every composite below is "
      f"on the\n"
      f"exact same scale as the existing set:\n")
    a("| metric | reference mean | reference std |")
    a("|---|---:|---:|")
    for m in COMPOSITE_METRICS:
        a(f"| {m} | {_fmt(ref.mean[m])} | {_fmt(ref.std[m])} |")
    a("")

    # --- Inventory summary ---
    a("## Inventory summary\n")
    by_status = Counter(m["author_status"] for m in manifest.values())
    new_authors = sorted({
        m["Autor"] for m in manifest.values() if m["author_status"] == "new"
    })
    used_authors_in_cands = sorted({
        m["Autor"] for m in manifest.values() if m["author_status"] == "used"
    })
    a(f"- Candidates total: **{len(manifest)}**")
    a(f"- NEW-author candidates: **{by_status.get('new', 0)}** "
      f"(authors: {', '.join(x[:16] + '…' for x in new_authors) or 'none'})")
    a(f"- USED-author candidates (new conditions, same speaker): "
      f"**{by_status.get('used', 0)}** "
      f"(across {len(used_authors_in_cands)} already-used authors)")
    a(f"- UNKNOWN-author candidates (unmatched to Korpus): "
      f"**{by_status.get('unknown', 0)}**\n")

    # Per-new-author breakdown.
    if new_authors:
        a("### New authors found\n")
        a("| author | candidates | recordings |")
        a("|---|---:|---:|")
        for aut in new_authors:
            cids = [c for c, m in manifest.items() if m["Autor"] == aut]
            recs = {manifest[c]["rec"] for c in cids}
            a(f"| {aut[:24]} | {len(cids)} | {len(recs)} |")
        a("")

    # --- Full ranked table: new authors first, then by composite desc within author ---
    a("## Full ranked candidate table\n")
    a("Ordering: **new-author candidates first** (then unknown, then used), "
      "and within each group by **composite descending** (hardest first). "
      "`ovl` = overlap seconds (floor 3 s); `dur` = fragment seconds.\n")
    a("| # | status | author | cand_id | composite | ovl | dur | noise | "
      "environment | device | topic |")
    a("|---|---|---|---|---:|---:|---:|---|---|---|---|")

    status_rank = {"new": 0, "unknown": 1, "used": 2}

    def sort_key(cid):
        m = manifest[cid]
        comp = cand_comp.get(cid, float("nan"))
        comp_key = comp if math.isfinite(comp) else float("-inf")
        return (status_rank.get(m["author_status"], 3),
                m["Autor"],
                -comp_key)

    ordered = sorted(manifest, key=sort_key)
    for i, cid in enumerate(ordered, 1):
        m = manifest[cid]
        comp = cand_comp.get(cid, float("nan"))
        a(f"| {i} | {m['author_status']} | {m['Autor'][:14]} | {cid} | "
          f"{_fmt(comp, 2)} | {_fmt(_to_float(m.get('overlap_s')), 1)} | "
          f"{_fmt(_to_float(m.get('duration')), 1)} | "
          f"{m.get('Poziom szumów', '') or '-'} | "
          f"{m.get('Środowisko', '') or '-'} | "
          f"{m.get('Urządzenie nagrywające', '') or '-'} | "
          f"{(m.get('Temat rozmowy', '') or '-')[:36]} |")
    a("")

    # --- Composite distribution: candidates vs existing 128 ---
    a("## Composite distribution: candidates vs the existing eval set\n")
    cand_vals = [v for v in cand_comp.values() if math.isfinite(v)]
    ref_vals = [v for v in ref_comp.values() if math.isfinite(v)]
    all_vals = cand_vals + ref_vals
    lo = math.floor(min(all_vals)) if all_vals else -2.0
    hi = math.ceil(max(all_vals)) if all_vals else 3.0
    a("```")
    for line in text_histogram(ref_vals, "Existing eval set", lo, hi):
        a(line)
    a("")
    for line in text_histogram(cand_vals, "Candidates", lo, hi):
        a(line)
    a("```")
    if cand_vals and ref_vals:
        a(f"\nCandidate composite range: "
          f"**[{min(cand_vals):+.2f}, {max(cand_vals):+.2f}]** "
          f"(median {np.median(cand_vals):+.2f}).")
        a(f"Existing-set composite range: "
          f"**[{min(ref_vals):+.2f}, {max(ref_vals):+.2f}]** "
          f"(median {np.median(ref_vals):+.2f}).")
        n_beyond = sum(1 for v in cand_vals if v > max(ref_vals))
        n_below = sum(1 for v in cand_vals if v < min(ref_vals))
        a(f"Candidates harder than ANY existing fragment: **{n_beyond}**; "
          f"easier than any existing: **{n_below}**.")
    a("")

    # --- Suggested additions (per new author, then a few standout used-author) ---
    a("## Suggested additions (a listening shortlist — NOT auto-added)\n")
    a("For each NEW author, the 1–3 candidates that would add the most "
      "diversity\n"
      "(spread across the composite axis + distinct conditions). These are "
      "starting\n"
      "points for the author's listening pass, nothing more.\n")
    _suggest_new_authors(a, manifest, cand_comp)
    _suggest_used_standouts(a, manifest, cand_comp, ref_comp)

    # --- Exclusions / failures ---
    a("## Exclusions & failures\n")
    if exclusions:
        for e in exclusions:
            a(f"- {e}")
    else:
        a("- None. Every manifest candidate has a finite composite, and every "
          "scored fragment is in the manifest.")
    a("")

    return "\n".join(L), exclusions


def _suggest_new_authors(a, manifest: dict, cand_comp: dict) -> None:
    new_authors = sorted({
        m["Autor"] for m in manifest.values() if m["author_status"] == "new"
    })
    if not new_authors:
        a("_No new authors among the unused recordings — the candidate pool is "
          "entirely new conditions of already-used speakers. See the used-author "
          "standouts below._\n")
        return
    for aut in new_authors:
        cids = [c for c, m in manifest.items() if m["Autor"] == aut]
        # Spread: hardest, easiest, and a middle one (up to 3, dedup).
        finite = sorted(
            [c for c in cids if math.isfinite(cand_comp.get(c, float("nan")))],
            key=lambda c: cand_comp[c], reverse=True,
        )
        if not finite:
            a(f"### New author `{aut[:20]}`\n- no scored candidate "
              "(composite=nan) — listen anyway if conditions are rare.\n")
            continue
        picks = _spread_pick(finite, cand_comp, k=min(3, len(finite)))
        comps = [cand_comp[c] for c in finite]
        a(f"### New author `{aut[:20]}`  ({len(cids)} candidate(s))\n")
        a(f"_This author is a **new voice** — the single most valuable kind of "
          f"diversity — so its candidates are worth adding largely independent "
          f"of where they land on the acoustic axis. Here their composites span "
          f"[{min(comps):+.2f}, {max(comps):+.2f}] "
          f"({'all on the clean end' if max(comps) < 0 else 'mixed'}); add at "
          f"least one regardless._\n")
        for c in picks:
            m = manifest[c]
            a(f"- **{c}** — composite {_fmt(cand_comp[c], 2)}, "
              f"overlap {_fmt(_to_float(m.get('overlap_s')), 1)}s, "
              f"{m.get('Środowisko', '') or '?'} / "
              f"{m.get('Urządzenie nagrywające', '') or '?'} / "
              f"noise {m.get('Poziom szumów', '') or '?'}. "
              f"_New speaker; {_rationale(cand_comp[c])}._")
        a("")


def _suggest_used_standouts(a, manifest: dict, cand_comp: dict,
                            ref_comp: dict) -> None:
    used = [c for c, m in manifest.items() if m["author_status"] == "used"
            and math.isfinite(cand_comp.get(c, float("nan")))]
    if not used:
        return
    ref_vals = [v for v in ref_comp.values() if math.isfinite(v)]
    ref_max = max(ref_vals) if ref_vals else float("inf")
    # Standouts: candidates harder than the existing-set max, OR rare conditions.
    rare_envs = {"Kawiarnia", "Knajpa", "W samochodzie", "Na zewnątrz",
                 "Na ulicy", "korytarz", "kuchnia"}
    flagged = []
    for c in used:
        m = manifest[c]
        is_hard = cand_comp[c] > ref_max
        is_rare = (m.get("Środowisko", "") in rare_envs)
        if is_hard or is_rare:
            flagged.append((c, is_hard, is_rare))
    flagged.sort(key=lambda t: -cand_comp[t[0]])
    a("### Used-author standouts (new conditions worth a listen)\n")
    if not flagged:
        a("_No used-author candidate is harder than the existing-set max or in "
          "a rare environment; they fill the existing distribution rather than "
          "extending it._\n")
        return
    a("Same speakers as the eval set, but either acoustically harder than "
      "anything\n"
      "currently in it, or recorded in an under-represented environment:\n")
    for c, is_hard, is_rare in flagged[:12]:
        m = manifest[c]
        tags = []
        if is_hard:
            tags.append("harder than existing max")
        if is_rare:
            tags.append(f"rare env: {m.get('Środowisko', '')}")
        a(f"- **{c}** ({m['Autor'][:12]}) — composite "
          f"{_fmt(cand_comp[c], 2)}, overlap "
          f"{_fmt(_to_float(m.get('overlap_s')), 1)}s, "
          f"{m.get('Urządzenie nagrywające', '') or '?'}. _{'; '.join(tags)}._")
    a("")


def _spread_pick(finite_sorted: list, cand_comp: dict, k: int) -> list:
    """Pick up to k candidates spread across the composite range: always the
    hardest and easiest, plus an evenly-spaced middle one when k>=3."""
    if k <= 1 or len(finite_sorted) <= 1:
        return finite_sorted[:k]
    if k == 2 or len(finite_sorted) == 2:
        return [finite_sorted[0], finite_sorted[-1]]
    mid = finite_sorted[len(finite_sorted) // 2]
    out = [finite_sorted[0], mid, finite_sorted[-1]]
    # Dedup preserving order.
    seen = set()
    return [c for c in out if not (c in seen or seen.add(c))]


def _rationale(comp: float) -> str:
    if comp > 1.0:
        return "acoustically hard, extends the noisy end"
    if comp < -1.0:
        return "very clean, anchors the easy end"
    return "mid-range condition, broadens coverage"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--candidate-scores", type=Path, default=DEFAULT_CAND_SCORES)
    ap.add_argument("--reference-scores", type=Path, default=DEFAULT_REF_SCORES)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    manifest = load_manifest(args.manifest)
    cand_scores = load_scores(args.candidate_scores)
    ref_scores = load_scores(args.reference_scores)
    ref = reference_stats(ref_scores)

    print(f"[report] manifest={len(manifest)} candidates, "
          f"cand_scores={len(cand_scores)}, ref={ref.n} reference rows")
    report, exclusions = build_report(manifest, cand_scores, ref_scores, ref)
    args.out.write_text(report, encoding="utf-8")
    print(f"[report] wrote {args.out}")
    if exclusions:
        print(f"[report] {len(exclusions)} exclusion note(s) recorded in report")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
