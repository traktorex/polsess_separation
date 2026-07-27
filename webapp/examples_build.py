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

Scores (`cpwer` / `cpcer`) are emitted as ``null``: computing them needs the
eval stack, and wiring the real numbers in is the orchestrator's step.
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import soundfile as sf

DEFAULT_ROOT = Path("~/datasets/eval/clarin_fragments").expanduser()
DEFAULT_ARM = "v41_merge"
DEFAULT_OUT = Path(__file__).resolve().parent / "examples_manifest.json"

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
                   limit: Optional[int] = None) -> dict:
    """Scan the eval tree and return the manifest dict."""
    splits = load_splits()
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
        rows.append({
            "id": frag,
            # Human title: the fragment id is what every thesis artifact calls
            # it, so it stays the identity. Enrich here if a nicer label ever
            # exists (CLARIN Korpus.csv has session/author columns).
            "title": frag,
            "duration_s": _duration_s(mixture, metadata),
            "split": splits.get(frag),
            "n_overlap_regions": metadata.get("n_overlap_regions"),
            "cpwer": None,      # scores are the orchestrator's wiring step
            "cpcer": None,
            "gt": gt,
            "pipeline_dir": str(pipeline_dir),
            "mixture_path": str(mixture),
        })
        if limit and len(rows) >= limit:
            break
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
    args = parser.parse_args(argv)

    root = Path(args.root).expanduser()
    if not root.exists():
        raise SystemExit(f"eval root not found: {root}")
    manifest = build_manifest(root, args.arm, ids=args.ids, limit=args.limit)
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with_gt = sum(1 for r in manifest["examples"] if r["gt"])
    with_split = sum(1 for r in manifest["examples"] if r["split"])
    print(
        f"wrote {out} — {len(manifest['examples'])} examples "
        f"({with_gt} with GT, {with_split} with a split label)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
