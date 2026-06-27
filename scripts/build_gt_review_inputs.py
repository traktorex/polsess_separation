"""Build the per-fragment inputs for the GT second-pass eyeball review.

The author hand-corrects ground-truth transcripts in ELAN on F:
(``/mnt/f/clarin_fragments/<frag>/annotation.eaf``). To support a *second pair
of eyes* per fragment (catching missed/misheard words, malformed ``<nzr>`` tags,
attribution slips, etc.) before the author's own second pass, the reviewer needs
clean ASR readings of the *real* audio to triangulate the GT against — but the
test fragments were never run through the pipeline, so no ASR output exists for
~all of them.

This script produces, for every in-scope fragment, a small text bundle written
*beside the EAF on F:* at ``<frag>/gt_review/``:

1. ``gt.txt``       — the hand-corrected GT, both speaker tiers interleaved
                      chronologically, time- and tier-labelled, with ``<nzr>``
                      kept visible. This is what the reviewer checks.
2. ``asr_lv2.txt``  — WhisperX large-v2 on the raw mono mix. large-v2 is the
                      model the GT was originally seeded from, so it AGREES with
                      the GT by construction — it shows the anchor.
3. ``asr_lv3.txt``  — WhisperX large-v3 on the same audio. An INDEPENDENT second
                      reading: where lv3 confidently diverges from the GT, that
                      is exactly where an lv2-anchored GT may have inherited an
                      error. The more valuable of the two for catching mistakes.

Both ASR readings are *guess menus*, not truth — the author resolves by ear.

Scope is computed from the authoritative sources, not hard-coded:
  in-scope = {role==test in clarin_split.csv}
             ∩ {annotation.eaf present & non-stub}
             − {marked "Not Done" or "DROPPED" in robione.txt}
This auto-includes the ``__cand`` fragments (which robione lists under drifted
ids) and auto-excludes the 15 delegated + 2 dropped fragments. The resolved list
is written to ``<drive>/gt_review_inscope.txt`` for the downstream eyeball pass.

Phase-major execution: GT rendered first (CPU), then large-v2 loaded once and run
over every fragment, unloaded, then large-v3. The GPU never holds two models. A
per-fragment ASR failure is caught, reported, and never aborts the batch.

Usage::

    source venv/bin/activate
    python scripts/build_gt_review_inputs.py                 # all in-scope
    python scripts/build_gt_review_inputs.py --skip-existing # cheap re-run
    python scripts/build_gt_review_inputs.py --fragments 005cba37__seg00 ...
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from asr_pipeline.config import TranscriptionConfig  # noqa: E402
from asr_pipeline.eval.transcript_parser import Utterance, parse_eaf  # noqa: E402
from asr_pipeline.stages.transcription import _WhisperXBackend  # noqa: E402
from scripts.build_nzr_aids import parse_robione  # noqa: E402

SAMPLE_RATE = 16_000
DROVE = "/mnt/f/clarin_fragments"
SPLIT_CSV = "asr_pipeline/eval/clarin_split.csv"
# (label, model_name) — phase-major, one at a time.
ASR_MODELS = (("lv2", "large-v2"), ("lv3", "large-v3"))


# --------------------------------------------------------------------------- #
# Scope resolution (reproducible — no hard-coded fragment lists)
# --------------------------------------------------------------------------- #


def _robione_excluded(text: str) -> set[str]:
    """Fragment ids robione marks as not-eligible (``Not Done`` / ``DROPPED``).

    A line is excluded iff its first token is a fragment-id-shaped token and any
    later token (case-insensitive) is ``dropped`` or the pair ``not done``.
    """
    out: set[str] = set()
    for raw in text.splitlines():
        toks = raw.replace("\r", "").split()
        if len(toks) < 2:
            continue
        frag = toks[0]
        rest = " ".join(toks[1:]).lower()
        if "dropped" in rest or "not done" in rest:
            out.add(frag)
    return out


def resolve_inscope(drive_root: Path, repo_root: Path) -> list[str]:
    """The 103 test fragments eligible for second-pass review."""
    with open(repo_root / SPLIT_CSV) as f:
        test = [row["frag_id"] for row in csv.DictReader(f) if row["role"] == "test"]
    robione = (drive_root / "robione.txt").read_text(encoding="utf-8", errors="replace")
    excluded = _robione_excluded(robione)
    inscope = []
    for frag in sorted(test):
        if frag in excluded:
            continue
        eaf = drive_root / frag / "annotation.eaf"
        if not eaf.exists() or eaf.stat().st_size < 500:
            print(f"  [skip] {frag}: eaf missing/stub")
            continue
        inscope.append(frag)
    return inscope


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def _fmt_time(t: float) -> str:
    m = int(t // 60)
    return f"{m:02d}:{t - 60 * m:05.2f}"


def render_gt(frag: str, by_tier: dict[str, list[Utterance]]) -> str:
    """Both tiers, chronologically interleaved, tier/time-labelled."""
    flat = sorted(
        ((tier, u) for tier, utts in by_tier.items() for u in utts),
        key=lambda tu: (tu[1].start if tu[1].start is not None else 0.0,
                        tu[1].end if tu[1].end is not None else 0.0),
    )
    lines = [
        f"GROUND TRUTH (hand-corrected) — {frag}",
        f"tiers: {', '.join(by_tier)}  |  {len(flat)} utterances",
        "Both speakers interleaved in time. <nzr> = author marked unintelligible.",
        "=" * 70,
    ]
    for tier, u in flat:
        s = _fmt_time(u.start) if u.start is not None else " --:-- "
        e = _fmt_time(u.end) if u.end is not None else " --:-- "
        flag = "  <<< has <nzr>" if "<nzr>" in u.text else ""
        lines.append(f"[{tier} {s}->{e}] {u.text}{flag}")
    return "\n".join(lines) + "\n"


def render_asr(frag: str, label: str, model_name: str, result: dict) -> str:
    """WhisperX result -> time-labelled segment list."""
    segs = result.get("segments", []) or []
    lines = [
        f"ASR READING — {frag}  [WhisperX {model_name}]  (a GUESS MENU, not truth)",
        f"{len(segs)} segments. Single mono mix → both speakers in one stream.",
        "=" * 70,
    ]
    for seg in segs:
        st = seg.get("start")
        txt = (seg.get("text") or "").strip()
        if not txt:
            continue
        lines.append(f"[{_fmt_time(st) if st is not None else ' --:-- '}] {txt}")
    if len(lines) == 3:
        lines.append("(no speech recognised)")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Work items + phases
# --------------------------------------------------------------------------- #


@dataclass
class Item:
    frag: str
    bundle: Path
    audio: np.ndarray
    errors: list[str] = field(default_factory=list)

    def note(self, stage: str, exc: Exception) -> None:
        msg = f"{stage}: {type(exc).__name__}: {exc}"
        self.errors.append(msg)
        print(f"    [FAIL] {self.frag} {msg}")


def build_items(fragments: list[str], drive_root: Path, audio_root: Path,
                skip_existing: bool) -> tuple[list[Item], list[str]]:
    items: list[Item] = []
    skips: list[str] = []
    for frag in fragments:
        eaf = drive_root / frag / "annotation.eaf"
        if not eaf.exists():
            skips.append(f"{frag}: no annotation.eaf")
            continue
        wav = audio_root / frag / f"{frag}.wav"
        if not wav.exists():
            skips.append(f"{frag}: no audio at {wav}")
            continue
        bundle = drive_root / frag / "gt_review"
        bundle.mkdir(parents=True, exist_ok=True)
        # GT is CPU-only — always (re)rendered so it reflects the latest EAF.
        try:
            (bundle / "gt.txt").write_text(
                render_gt(frag, parse_eaf(eaf)), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            skips.append(f"{frag}: GT render failed: {exc}")
            continue
        # ASR is the expensive part — honour --skip-existing per fragment.
        need = [lbl for lbl, _ in ASR_MODELS
                if not (skip_existing and (bundle / f"asr_{lbl}.txt").exists())]
        if not need:
            skips.append(f"{frag}: asr_* exist, skipped")
            continue
        audio, _ = librosa.load(wav, sr=SAMPLE_RATE, mono=True)
        items.append(Item(frag=frag, bundle=bundle, audio=audio.astype(np.float32)))
    return items, skips


def phase_asr(items: list[Item], label: str, model_name: str,
              device: torch.device, skip_existing: bool) -> None:
    """Load one WhisperX model, transcribe every item's mix, unload."""
    cfg = TranscriptionConfig(backend="whisperx", model_name=model_name,
                              language="pl", word_timestamps=True)
    backend = _WhisperXBackend(cfg)
    print(f"[asr:{label}] loading WhisperX {model_name} on {device}")
    t0 = time.perf_counter()
    backend.load(device)
    print(f"[asr:{label}] loaded in {time.perf_counter() - t0:.1f}s; "
          f"transcribing {len(items)} fragment(s)")
    try:
        for it in items:
            out = it.bundle / f"asr_{label}.txt"
            if skip_existing and out.exists():
                continue
            try:
                result = backend.transcribe(it.audio)
                out.write_text(render_asr(it.frag, label, model_name, result),
                               encoding="utf-8")
            except Exception as exc:  # noqa: BLE001 — per-item isolation
                it.note(f"asr[{label}]", exc)
    finally:
        try:
            backend.unload()
        except Exception:  # noqa: BLE001
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--drive-root", default=DROVE)
    p.add_argument("--audio-root",
                   default=str(Path.home() / "datasets/eval/clarin_fragments"))
    p.add_argument("--fragments", nargs="*", default=None,
                   help="Explicit ids (default: all in-scope test fragments).")
    p.add_argument("--skip-existing", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    drive_root = Path(args.drive_root)
    audio_root = Path(args.audio_root).expanduser()
    repo_root = Path(__file__).resolve().parent.parent
    if not drive_root.is_dir():
        raise SystemExit(f"drive root not mounted: {drive_root}")

    fragments = args.fragments or resolve_inscope(drive_root, repo_root)
    if not args.fragments:
        listing = drive_root / "gt_review_inscope.txt"
        listing.write_text("\n".join(fragments) + "\n", encoding="utf-8")
        print(f"[scope] {len(fragments)} in-scope fragment(s) -> {listing}")
    else:
        print(f"[scope] {len(fragments)} explicit fragment(s)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    items, skips = build_items(fragments, drive_root, audio_root, args.skip_existing)
    print(f"[scan] {len(items)} fragment(s) need ASR; {len(skips)} skipped")
    for s in skips:
        print(f"  [skip] {s}")

    if items:
        for label, model_name in ASR_MODELS:
            phase_asr(items, label, model_name, device, args.skip_existing)

    failed = [it for it in items if it.errors]
    print("\n=== summary ===")
    print(f"  fragments processed : {len(items)}")
    print(f"  with failures       : {len(failed)}")
    print(f"  skipped             : {len(skips)}")
    for it in failed:
        print(f"  [FAIL] {it.frag}: {'; '.join(it.errors)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
