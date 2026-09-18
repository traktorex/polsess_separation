"""Build `webapp/examples_manifest.json` from the frozen eval tree.

Offline CLI, run once (and re-run whenever the frozen arm changes)::

    venv/bin/python -m webapp.examples_build
    venv/bin/python -m webapp.examples_build --arm v41_merge --limit 12
    venv/bin/python -m webapp.examples_build --ids 005cba37__seg00 065a9896__seg00

It scans ``<root>/<fragment>/sweep/<arm>/`` and writes one light row per
fragment. The heavy ``job_like`` payload (transcripts + peak envelopes) is NOT
stored here — the server assembles and caches it from `pipeline_dir` at request
time, which keeps the manifest small and always in step with the tree.

Ground truth comes from the **fragment-level** ``<fragment>/annotation.eaf`` —
the hand-corrected reference, not the pipeline's own EAF inside the arm
directory. It is parsed with the standard library rather than
`asr_pipeline.eval.transcript_parser.parse_eaf` (identical semantics, mirrored
below) purely to keep this CLI off the heavyweight eval import chain; the two
must stay in step if the EAF tier convention ever changes.

Scores are joined in from the frozen per-fragment rescore CSVs, which stay
authoritative: whatever a sheet carries is what the gallery shows, so the
numbers on screen are exactly the numbers the thesis reports. The sheets do not
carry the whole metric set the examples page wants (no tcpWER, no MIMO-CER for
the arm, no mixture ORC-WER / time-ordered mixture CER), so the missing entries
are **computed here with the identical frozen eval code** (`asr_pipeline.eval`,
the same functions `scripts/rescore_stratified.py` and
`scripts/build_review_page.py` call) and every value that also exists in a sheet
is recomputed purely to cross-check it — a disagreement above
`CROSS_CHECK_TOLERANCE` is reported loudly and the sheet wins.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import soundfile as sf

DEFAULT_ROOT = Path("~/datasets/eval/clarin_fragments").expanduser()
DEFAULT_ARM = "v41_merge"
DEFAULT_OUT = Path(__file__).resolve().parent / "examples_manifest.json"

# Frozen per-fragment scores from the definitive campaign's rescore pass: the
# dev sheet (23 fragments) and the V5 test sheet (118). Rows are per (fragment,
# config); only the shipped arm's rows are read.
DEFAULT_SCORES_CSVS = [
    DEFAULT_ROOT / "_rescore_perfrag_v41_dev.csv",
    DEFAULT_ROOT / "_rescore_perfrag_v5_test.csv",
]

# The metric set every example row carries (API.md v1.2 draft). All values are
# percentages rounded to one decimal, except `attr_gap` (cpWER − MIMO-WER, in
# points). `floor_*` are the no-pipeline baseline: one Whisper pass over the raw
# mixture, scored against the same GT.
METRIC_KEYS = (
    "cpwer", "tcpwer", "orcwer", "attr_gap",
    "cpcer", "mimower", "mimocer", "orccer",
    "floor_orcwer", "floor_mimower", "floor_cpcer", "floor_mimocer",
)

# Metric keys the frozen sheets already carry -> their CSV column. These are the
# authoritative values; a recomputed one is only ever used to cross-check them.
_CSV_METRIC_COLUMNS = {
    "cpwer": "cp_wer",
    "cpcer": "cp_cer",
    "orcwer": "orc_wer",
    "mimower": "mimo_wer",
    "orccer": "orc_cer",
    "floor_mimower": "mix_mimo_wer",
    "floor_mimocer": "mix_mimo_cer",
}

# Cross-check tolerance, in percentage points. Sheet values and recomputed ones
# are compared at the precision they are *computed* at — the sheets round to 2
# decimals, so ≤0.005 pt of rounding noise is expected and the manifest's own
# 1-decimal rounding is applied only afterwards. Anything above this means the
# recomputation and the frozen sheet genuinely disagree.
CROSS_CHECK_TOLERANCE = 0.05

# Frozen thesis fragment lists (the dev/test split of the definitive campaign).
_SPLIT_FILES = {
    "dev": Path(__file__).resolve().parent.parent / "asr_pipeline/eval/clarin_dev.txt",
    "test": Path(__file__).resolve().parent.parent / "asr_pipeline/eval/clarin_test.txt",
}


def load_splits() -> Dict[str, str]:
    """``fragment_id -> "dev" | "test"`` from the frozen split lists.

    Missing list files are not fatal — every example simply gets ``split: null``,
    which the API contract allows for.
    """
    out: Dict[str, str] = {}
    for split, path in _SPLIT_FILES.items():
        if not path.exists():
            print(f"[examples] split list not found: {path} (split will be null)")
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            frag = line.strip()
            if frag and not frag.startswith("#"):
                out[frag] = split
    return out


def load_scores(csv_paths: List[Path], arm: str) -> Dict[str, dict]:
    """``fragment_id -> {"cpwer", "cpcer", "stratum", "metrics"}`` for `arm`.

    Reads the frozen rescore sheets and keeps only the rows whose ``config``
    column matches the arm being frozen, so the gallery cannot accidentally show
    another arm's numbers. The two headline values stay raw floats — formatting
    is the frontend's job; the `metrics` sub-dict holds the sheet columns of
    :data:`_CSV_METRIC_COLUMNS` exactly as written (2 decimals), so the
    cross-check in :func:`merge_metrics` compares full-precision values and only
    the merged result is rounded.
    ``stratum`` is the recording-level acoustic-complexity tertile the campaign
    assigned (LOW/MID/HIGH), verbatim. A missing file is reported and skipped,
    never fatal: the manifest is still usable with the scores null.
    """
    scores: Dict[str, dict] = {}
    for path in csv_paths:
        path = Path(path).expanduser()
        if not path.exists():
            print(f"[examples] scores CSV not found: {path} (scores will be null)")
            continue
        try:
            with open(path, newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
        except (OSError, ValueError) as exc:
            print(f"[examples] unreadable scores CSV {path}: {exc}")
            continue
        kept = 0
        for row in rows:
            if (row.get("config") or "").strip() != arm:
                continue
            frag = (row.get("frag_id") or "").strip()
            if not frag:
                continue
            scores[frag] = {
                "cpwer": _float_or_none(row.get("cp_wer")),
                "cpcer": _float_or_none(row.get("cp_cer")),
                "stratum": (row.get("stratum") or "").strip() or None,
                "metrics": {
                    key: _float_or_none(row.get(column))
                    for key, column in _CSV_METRIC_COLUMNS.items()
                },
            }
            kept += 1
        print(f"[examples] {path.name}: {kept} rows for config={arm!r}")
    return scores


def _float_or_none(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _round1(value: Optional[float]) -> Optional[float]:
    """One decimal, or ``None`` passed through — the manifest's metric precision."""
    return None if value is None else round(value, 1)


# ---------------------------------------------------------------------------
# The metric set: frozen sheets first, the same eval code for the rest
# ---------------------------------------------------------------------------


def _pct(
    sub: Optional[dict], errors_key: str = "errors", length_key: str = "length"
) -> Optional[float]:
    """A meeteval sub-result's error rate as a percentage, **unrounded**.

    Computed from the raw ``errors`` / ``length`` counts, which is exactly how
    `scripts/rescore_stratified.py` writes the frozen sheets — so the two agree
    before rounding rather than by luck, and the cross-check compares like with
    like (rounding happens once, in :func:`merge_metrics`). A sub-result skipped
    by the combinatorial guard (``None``) or one with an empty reference stays
    ``None``: a missing metric is reported as missing, never as a zero.
    """
    if sub is None or sub.get(errors_key) is None:
        return None
    length = sub.get(length_key) or 0
    if not length:
        return None
    return 100.0 * float(sub[errors_key]) / float(length)


def _time_ordered_mixture_cer(gt: dict, mix: list, normalize) -> Optional[float]:
    """The mixture floor's CER against a **time-ordered** merged reference.

    Mirrors ``_mixture_metrics``' ``cer`` in `scripts/build_review_page.py`: a
    single hypothesis stream cannot reorder anything, so both speakers' GT is
    interleaved by start time and compared with one character-level edit
    distance. (The MIMO-flavoured floor CER, which forgives the interleaving
    order, is the separate ``floor_mimocer``.)
    """
    from rapidfuzz.distance import Levenshtein

    ref_utts = sorted((u for spk in gt for u in gt[spk]), key=lambda u: u.start)
    ref_all = normalize(" ".join(u.text for u in ref_utts))
    hyp_all = normalize(" ".join(u.text for u in mix))
    if not ref_all:
        return None
    return 100.0 * Levenshtein.distance(ref_all, hyp_all) / len(ref_all)


def compute_metric_set(frag_dir: Path, arm: str) -> Optional[Dict[str, Optional[float]]]:
    """Recompute one fragment's metric set with the frozen eval code.

    The sheets carry seven of the twelve values (:data:`_CSV_METRIC_COLUMNS`);
    the rest — tcpWER, the arm's MIMO-CER, and the two mixture floors the
    rescorer never dumped — have to come from somewhere, and the only honest
    source is the same `asr_pipeline.eval` code that produced the sheets. The
    seven overlapping values are computed as well, purely so they can be
    cross-checked (see :func:`merge_metrics`).

    Reading matches `scripts/rescore_stratified.py` exactly — `parse_eaf` GT with
    empty tiers dropped, `read_per_speaker` / `read_mixture` for the arm — so a
    recomputed value that disagrees with its sheet column is a real
    disagreement, not a parsing difference.

    Returns ``None`` when the fragment has no GT EAF or no per-speaker
    transcript (nothing to score). Individual values are ``None`` when the
    fragment trips `per_fragment_metrics`' combinatorial blow-up guard.
    Percentages are returned unrounded, and ``attr_gap`` is not computed here —
    both are :func:`merge_metrics`' job, once the sheet values have had their say.

    The imports are function-local on purpose: `asr_pipeline.eval` pulls in torch
    and meeteval, and neither the webapp server nor a metric-free manifest
    rebuild may depend on either.
    """
    eaf = frag_dir / "annotation.eaf"
    pipeline_dir = frag_dir / "sweep" / arm
    if not eaf.exists() or not (pipeline_dir / "transcript_A.txt").exists():
        return None

    from asr_pipeline.eval.layer3 import read_mixture, read_per_speaker
    from asr_pipeline.eval.metrics import (
        _normalize_text, mimo_cer_meeteval, per_fragment_metrics,
    )
    from asr_pipeline.eval.transcript_parser import parse_eaf

    # The GT read here is the on-disk one. The display swap of `detect_gt_swap`
    # must never reach it: every metric below picks its own optimal speaker
    # assignment, so a relabelled reference would change nothing but the audit
    # trail.
    gt = {spk: utts for spk, utts in parse_eaf(eaf).items() if utts}
    hyp = read_per_speaker(pipeline_dir)
    if not gt or hyp is None:
        return None
    mix = read_mixture(pipeline_dir)
    frag = frag_dir.name
    metrics = per_fragment_metrics(gt, hyp, session_id=frag, mix=mix)

    out: Dict[str, Optional[float]] = {key: None for key in METRIC_KEYS}
    out["cpwer"] = _pct(metrics["cp"], "cp_errors", "cp_length")
    out["tcpwer"] = _pct(metrics["cp"], "tcp_errors", "tcp_length")
    out["cpcer"] = _pct(metrics["cpcer"])
    out["orcwer"] = _pct(metrics["orc"])
    out["mimower"] = _pct(metrics["mimo"])
    out["orccer"] = _pct(metrics["orccer"])
    # MIMO-CER for the arm is not part of per_fragment_metrics. It runs the same
    # word-level MIMO table internally, so it inherits `mimo`'s guard verdict
    # instead of re-deriving the cap.
    if metrics["mimo"] is not None:
        out["mimocer"] = _pct(mimo_cer_meeteval(gt, hyp, frag))
    out["floor_orcwer"] = _pct(metrics.get("mix_orc"))
    out["floor_mimower"] = _pct(metrics.get("mix_mimo"))
    out["floor_mimocer"] = _pct(metrics.get("mix_cer"))
    if mix:
        out["floor_cpcer"] = _time_ordered_mixture_cer(gt, mix, _normalize_text)

    skipped = [role for role, value in metrics.items() if value is None]
    if skipped:
        print(f"[examples] {frag}: combinatorial guard skipped "
              f"{', '.join(sorted(skipped))} — those metrics stay null unless "
              f"the frozen sheets carry them")
    return out


def merge_metrics(
    frag: str,
    computed: Optional[Dict[str, Optional[float]]],
    from_csv: Dict[str, Optional[float]],
) -> Tuple[Optional[dict], List[str]]:
    """Combine sheet values and recomputed ones into one metric row.

    The sheet always wins — it is the number the thesis reports. The recomputed
    value's job is to catch a silent drift between the frozen CSVs and today's
    eval code: any pair disagreeing by more than
    :data:`CROSS_CHECK_TOLERANCE` points is returned as a warning line (and
    printed by the caller), with the sheet value still kept. Both sides are
    compared unrounded; the manifest's one decimal is applied once, at the end.

    Returns ``(metrics, warnings)``; `metrics` is ``None`` when neither source
    produced a single value, so a fragment with no scores at all carries
    ``"metrics": null`` rather than a row of nulls.
    """
    warnings: List[str] = []
    if computed is None and not any(v is not None for v in from_csv.values()):
        return None, warnings

    merged: Dict[str, Optional[float]] = {key: None for key in METRIC_KEYS}
    for key, value in (computed or {}).items():
        if key in merged:
            merged[key] = value
    for key, csv_value in from_csv.items():
        if csv_value is None:
            continue
        recomputed = merged.get(key)
        if recomputed is not None and abs(recomputed - csv_value) > CROSS_CHECK_TOLERANCE:
            warnings.append(
                f"[examples] WARNING: {frag} {key}: recomputed {recomputed:.3f} vs "
                f"frozen sheet {csv_value} (Δ {recomputed - csv_value:+.3f} pt) "
                f"— sheet value kept"
            )
        merged[key] = csv_value

    rounded = {key: _round1(value) for key, value in merged.items()}
    # The attribution gap is derived from the two values the row actually shows,
    # so the displayed gap is always the displayed difference.
    if rounded["cpwer"] is not None and rounded["mimower"] is not None:
        rounded["attr_gap"] = round(rounded["cpwer"] - rounded["mimower"], 1)
    return rounded, warnings


# ---------------------------------------------------------------------------
# GT speaker-swap detection (display alignment only)
# ---------------------------------------------------------------------------

# Minimum advantage the crossed pairing needs over the straight one before the
# GT tiers are relabelled. Small, but non-zero: two near-identical similarity
# sums mean the texts do not tell the tiers apart, and the on-disk labels stand.
SWAP_MARGIN = 0.02

_TIMESTAMP_PREFIX = re.compile(r"^\s*\[[^\]]*\]\s*")
_NON_WORD = re.compile(r"[^\w\s]", re.UNICODE)


def _transcript_text(path: Path) -> str:
    """All spoken text of one ``transcript_<label>.txt``, timestamps dropped.

    A local mini-reader rather than
    `asr_pipeline.eval.transcript_parser.parse_gt_txt`, for the same reason the
    EAF parse above is mirrored: swap detection runs for every example on the
    fast path, which must stay off the heavyweight eval import chain. Only the
    words matter here — timings are never consulted.
    """
    if not path.exists():
        return ""
    kept = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("==="):
            continue
        kept.append(_TIMESTAMP_PREFIX.sub("", line))
    return " ".join(kept)


def _normalize_for_match(text: str) -> str:
    """Lowercase, punctuation stripped, whitespace collapsed."""
    return " ".join(_NON_WORD.sub(" ", text.lower()).split())


def detect_gt_swap(gt: Optional[Dict[str, List[dict]]], pipeline_dir: Path) -> bool:
    """True when GT tier order is the mirror image of the pipeline's A/B.

    The pipeline names its two streams without knowing which GT tier is "A", so
    for a good share of fragments its stream A is the GT-B speaker (e.g.
    ``026eafb1__seg00``). Every metric here already picks its own optimal
    speaker assignment and is permutation-invariant, so this is a **display**
    decision and nothing else: it only decides which GT tier is shown beside
    which pipeline stream.

    Both pairings are scored with `difflib.SequenceMatcher` over the normalized,
    concatenated text; the crossed pairing must win by at least
    :data:`SWAP_MARGIN` to be believed. Anything other than exactly two GT tiers
    with both pipeline transcripts present is left alone.
    """
    if not gt or len(gt) != 2:
        return False
    first, second = list(gt)
    pipeline = {}
    for label in ("A", "B"):
        path = pipeline_dir / f"transcript_{label}.txt"
        if not path.exists():
            return False
        pipeline[label] = _normalize_for_match(_transcript_text(path))
    reference = {
        tier: _normalize_for_match(" ".join(u["text"] for u in gt[tier]))
        for tier in (first, second)
    }

    def ratio(a: str, b: str) -> float:
        return difflib.SequenceMatcher(None, a, b).ratio()

    straight = ratio(pipeline["A"], reference[first]) + ratio(pipeline["B"], reference[second])
    crossed = ratio(pipeline["A"], reference[second]) + ratio(pipeline["B"], reference[first])
    return crossed > straight + SWAP_MARGIN


def swap_gt_tiers(gt: Dict[str, List[dict]]) -> Dict[str, List[dict]]:
    """Exchange the two tiers' labels, keeping the original key order."""
    first, second = list(gt)
    return {first: gt[second], second: gt[first]}


def parse_gt_eaf(path: Path) -> Optional[Dict[str, List[dict]]]:
    """ELAN GT -> ``{"A": [{"start", "end", "text"}, ...], "B": [...]}``.

    Mirrors `asr_pipeline.eval.transcript_parser.parse_eaf`: one tier per
    speaker, the ``Speaker_`` prefix stripped so ``Speaker_A`` -> ``A``, only
    aligned annotations with text and resolvable time slots, sorted by start.
    Returns ``None`` when the file is absent or unparsable.
    """
    if not path.exists():
        return None
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as exc:
        print(f"[examples] unreadable GT {path}: {exc}")
        return None

    slots: Dict[str, float] = {}
    for ts in root.iter("TIME_SLOT"):
        sid, value = ts.get("TIME_SLOT_ID"), ts.get("TIME_VALUE")
        if sid is not None and value is not None:
            slots[sid] = float(value) / 1000.0

    tiers: Dict[str, List[dict]] = {}
    for tier in root.iter("TIER"):
        tier_id = tier.get("TIER_ID") or ""
        label = (
            tier_id[len("Speaker_"):] if tier_id.startswith("Speaker_") else tier_id
        )
        utterances = []
        for ann in tier.iter("ALIGNABLE_ANNOTATION"):
            start = slots.get(ann.get("TIME_SLOT_REF1"))
            end = slots.get(ann.get("TIME_SLOT_REF2"))
            value = ann.find("ANNOTATION_VALUE")
            text = (value.text or "").strip() if value is not None else ""
            if text and start is not None and end is not None:
                utterances.append({"start": start, "end": end, "text": text})
        utterances.sort(key=lambda u: u["start"])
        if utterances:
            tiers[label] = utterances
    return tiers or None


def _duration_s(mixture: Path, metadata: dict) -> Optional[float]:
    try:
        info = sf.info(str(mixture))
        if info.samplerate:
            return float(info.frames) / float(info.samplerate)
    except (RuntimeError, OSError):
        pass
    return metadata.get("total_duration_s")


def build_manifest(root: Path, arm: str, ids: Optional[List[str]] = None,
                   limit: Optional[int] = None,
                   scores_csvs: Optional[List[Path]] = None,
                   compute_metrics: bool = True) -> dict:
    """Scan the eval tree and return the manifest dict.

    `compute_metrics` off skips the `asr_pipeline.eval` recomputation entirely —
    the sheets' own columns still land, the rest stay null. It exists for a fast
    rebuild on a machine without the eval dependencies; the shipped manifest is
    always built with it on.
    """
    splits = load_splits()
    scores = load_scores(
        scores_csvs if scores_csvs is not None else DEFAULT_SCORES_CSVS, arm
    )
    mismatches = 0
    rows: List[dict] = []
    for frag_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        frag = frag_dir.name
        if ids and frag not in ids:
            continue
        pipeline_dir = frag_dir / "sweep" / arm
        metadata_path = pipeline_dir / "metadata.json"
        mixture = frag_dir / f"{frag}.wav"
        if not metadata_path.exists() or not mixture.exists():
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"[examples] skipping {frag}: unreadable metadata ({exc})")
            continue
        gt = parse_gt_eaf(frag_dir / "annotation.eaf")
        score = scores.get(frag) or {}
        gt_swapped = detect_gt_swap(gt, pipeline_dir)
        if gt_swapped:
            gt = swap_gt_tiers(gt)
        metrics, warnings = merge_metrics(
            frag,
            compute_metric_set(frag_dir, arm) if compute_metrics else None,
            score.get("metrics") or {},
        )
        for line in warnings:
            print(line)
        mismatches += len(warnings)
        rows.append({
            "id": frag,
            # Human title: the fragment id is what every thesis artifact calls
            # it, so it stays the identity. Enrich here if a nicer label ever
            # exists (CLARIN Korpus.csv has session/author columns).
            "title": frag,
            "duration_s": _duration_s(mixture, metadata),
            "split": splits.get(frag),
            "n_overlap_regions": metadata.get("n_overlap_regions"),
            "cpwer": score.get("cpwer"),
            "cpcer": score.get("cpcer"),
            "stratum": score.get("stratum"),
            "metrics": metrics,
            "gt": gt,
            "gt_swapped": gt_swapped,
            "pipeline_dir": str(pipeline_dir),
            "mixture_path": str(mixture),
        })
        if limit and len(rows) >= limit:
            break
    print(f"[examples] metric cross-check: {mismatches} disagreement(s) above "
          f"{CROSS_CHECK_TOLERANCE} pt between the recomputation and the frozen sheets")
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_root": str(root),
        "arm": arm,
        "examples": rows,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="webapp.examples_build")
    parser.add_argument("--root", default=str(DEFAULT_ROOT),
                        help=f"Eval tree root (default: {DEFAULT_ROOT}).")
    parser.add_argument("--arm", default=DEFAULT_ARM,
                        help=f"Sweep arm directory to freeze (default: {DEFAULT_ARM}).")
    parser.add_argument("--out", default=str(DEFAULT_OUT),
                        help=f"Manifest output path (default: {DEFAULT_OUT}).")
    parser.add_argument("--ids", nargs="+", default=None,
                        help="Restrict to these fragment ids.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Stop after this many examples.")
    parser.add_argument(
        "--scores-csv", dest="scores_csvs", action="append", default=None,
        metavar="PATH",
        help="Per-fragment rescore CSV to join scores from (repeatable). "
             "Default: the frozen dev + V5 test sheets under the eval root.",
    )
    parser.add_argument(
        "--no-metrics", dest="compute_metrics", action="store_false",
        help="Skip the asr_pipeline.eval recomputation: only the columns the "
             "frozen sheets carry land, the rest stay null (fast rebuild).",
    )
    args = parser.parse_args(argv)

    root = Path(args.root).expanduser()
    if not root.exists():
        raise SystemExit(f"eval root not found: {root}")
    scores_csvs = (
        [Path(p).expanduser() for p in args.scores_csvs]
        if args.scores_csvs else None
    )
    manifest = build_manifest(
        root, args.arm, ids=args.ids, limit=args.limit, scores_csvs=scores_csvs,
        compute_metrics=args.compute_metrics,
    )
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    examples = manifest["examples"]
    total = len(examples)
    with_gt = sum(1 for r in examples if r["gt"])
    with_split = sum(1 for r in examples if r["split"])
    scored = sum(1 for r in examples if r["cpwer"] is not None)
    with_stratum = sum(1 for r in examples if r["stratum"])
    full_metrics = sum(
        1 for r in examples
        if r["metrics"] and all(r["metrics"][k] is not None for k in METRIC_KEYS)
    )
    swapped = [r["id"] for r in examples if r["gt_swapped"]]
    print(
        f"wrote {out} — {total} examples "
        f"({with_gt} with GT, {with_split} with a split label, "
        f"{scored} scored, {total - scored} without scores)"
    )
    print(f"  strata: {with_stratum}/{total} · full metric set: "
          f"{full_metrics}/{total} · GT tiers swapped for display: "
          f"{len(swapped)} ({', '.join(swapped) if swapped else 'none'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
