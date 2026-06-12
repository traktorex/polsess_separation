"""Mine NEW candidate fragments for the CLARIN eval set from UNUSED recordings.

The live 128-fragment eval set (``~/datasets/eval/clarin_fragments/``) draws on
123 source recordings. The full 2-speaker CLARIN download has more diarized
recordings than that; this script finds the best fragment candidates in the
recordings the eval set has **not** touched, so the author can grow the set
toward more speaker / acoustic-condition diversity.

It does NOT modify the live eval set. Everything it produces lands in a separate
staging tree (``~/datasets/clarin_fragment_candidates/``):

    candidates_manifest.csv          one row per candidate (+ metadata join)
    INVENTORY.md                     the unused-recording inventory + classes
    <rec>__candNN/<rec>__candNN.wav  mono 16 kHz slice per candidate

Pipeline (mirrors ``build_clarin_fragment_set.py`` conventions — same finder,
same slice extraction, same 16 kHz mono):

  1. INVENTORY. Unused = (audio AND diar present) MINUS the eval manifest's
     ``rec`` column. Join each to ``Korpus_with_filename.csv``; classify the
     author as ``new`` (author absent from the eval set), ``used`` (author
     already represented), or ``unknown`` (recording not matchable to a Korpus
     row → author UNKNOWN, still eligible). Unmatchable recordings are reported
     loudly, never silently dropped (SCOPE no-silent-substitution spirit).
  2. FIND. Run ``find_fragments`` with the eval set's selection-time params over
     every unused recording. Keep the top <=2 candidates per recording, ranked
     by the finder's own ``overlap_s * speaker_balance`` score, ties broken by
     higher overlap. Recordings yielding zero candidates are recorded with the
     distributional reason (no overlap >= floor, balance too low, etc.).
  3. EXTRACT. Slice each kept candidate to its own dir; write the manifest.

Scoring + the decision menu are a SEPARATE step
(``scripts/score_fragment_acoustics.py --root ...`` then
``scripts/build_candidate_report.py``) — this script stops at extraction so the
acoustic scorer (GPU) and the test suite never contend.

Author policy encoded here (do not re-litigate — see the task brief):
  - Overlap is a FLOOR + tie-breaker bonus, never the optimisation axis. We keep
    ``min_overlap_s=3.0`` and, among a recording's candidates, prefer higher
    overlap; we do NOT chase overlap at the cost of diversity.
  - Diversity is the goal. NEW authors are first-class; unused recordings of
    already-used authors are second-class (new conditions, same speaker) but
    still listed.
  - The author selects by listening; this only produces the ranked menu.

Idempotent: re-running skips candidates whose WAV already exists unless
``--force``.

Usage::

    python scripts/mine_clarin_candidates.py            # inventory + find + extract
    python scripts/mine_clarin_candidates.py --dry-run  # inventory + find, no audio
    python scripts/mine_clarin_candidates.py --force     # re-extract existing WAVs
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.clarin_fragment_finder import (  # noqa: E402
    Fragment, FragmentParams,
    find_fragments, load_diarization,
)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SOURCE_ROOT = Path("~/datasets/clarin_all_2speakers").expanduser()
AUDIO_DIR = SOURCE_ROOT / "clarin_download"
DIAR_DIR = SOURCE_ROOT / "diarization"
KORPUS_CSV = AUDIO_DIR / "Korpus_with_filename.csv"

EVAL_MANIFEST = Path("~/datasets/eval/clarin_fragments/manifest.csv").expanduser()

OUTPUT_ROOT = Path("~/datasets/clarin_fragment_candidates").expanduser()
CANDIDATES_MANIFEST = OUTPUT_ROOT / "candidates_manifest.csv"
INVENTORY_MD = OUTPUT_ROOT / "INVENTORY.md"


# Selection-time finder params — copied verbatim from the eval tree's
# SELECTION.md so candidate stats are directly comparable to the live set.
SELECTION_PARAMS = FragmentParams(
    target_length_s=90.0, min_length_s=60.0, max_length_s=120.0,
    stride_s=5.0,
    silence_snap_min_s=0.5, silence_snap_search_s=5.0,
    min_overlap_s=3.0,
    min_speaker_balance=0.20,
    min_speech_density=0.55,
    nms_iou_threshold=0.10,
)

MAX_CANDIDATES_PER_REC = 2
OUTPUT_SAMPLE_RATE = 16_000

# Korpus columns carried onto each candidate row (matches the metadata the
# author wants in the menu).
KORPUS_FIELDS = [
    "Autor", "Poziom szumów", "Środowisko", "Urządzenie nagrywające",
    "Temat rozmowy", "Liczba mówców", "Nazwa", "Identyfikator nagrania",
]


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------


@dataclass
class Inventory:
    eligible: list             # recordings with BOTH audio and diar
    used: set                  # eval-manifest rec ids
    unused: list               # eligible - used, sorted
    used_authors: set          # eval-manifest Autor values (non-blank)
    meta: dict                 # rec -> Korpus row dict
    unmatched: list            # unused recs with no Korpus row
    audio_no_diar: list        # reported, excluded from eligible
    diar_no_audio: list        # reported, excluded from eligible
    author_of: dict            # rec -> author string ('UNKNOWN' if unmatched)
    author_status: dict        # rec -> 'new' | 'used' | 'unknown'
    new_authors: list          # distinct new-author ids, sorted


def _rec_id_from_wav(wav_name: str) -> str:
    wav_name = wav_name.strip()
    return wav_name[:-4] if wav_name.endswith(".wav") else wav_name


def load_korpus_meta() -> dict:
    """rec_id -> Korpus row dict, keyed by the 'Nazwa pliku WAV' stem."""
    meta: dict = {}
    with open(KORPUS_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            wav = (r.get("Nazwa pliku WAV") or "").strip()
            if not wav:
                continue
            meta[_rec_id_from_wav(wav)] = r
    return meta


def load_used() -> tuple[set, set]:
    """Return (used_rec_ids, used_author_set) from the eval manifest."""
    used: set = set()
    authors: set = set()
    with open(EVAL_MANIFEST, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            used.add(r["rec"])
            a = (r.get("Autor") or "").strip()
            if a:
                authors.add(a)
    return used, authors


def build_inventory() -> Inventory:
    audio_ids = {p.stem for p in AUDIO_DIR.glob("*.wav")}
    diar_ids = {p.stem for p in DIAR_DIR.glob("*.json")}
    audio_no_diar = sorted(audio_ids - diar_ids)
    diar_no_audio = sorted(diar_ids - audio_ids)
    eligible = sorted(audio_ids & diar_ids)  # need BOTH

    used, used_authors = load_used()
    unused = sorted(set(eligible) - used)

    meta = load_korpus_meta()
    unmatched = [u for u in unused if u not in meta]

    author_of: dict = {}
    author_status: dict = {}
    for u in unused:
        a = (meta.get(u, {}).get("Autor") or "").strip() or "UNKNOWN"
        author_of[u] = a
        if a == "UNKNOWN":
            author_status[u] = "unknown"
        elif a in used_authors:
            author_status[u] = "used"
        else:
            author_status[u] = "new"

    new_authors = sorted({
        author_of[u] for u in unused if author_status[u] == "new"
    })

    return Inventory(
        eligible=eligible, used=used, unused=unused, used_authors=used_authors,
        meta=meta, unmatched=unmatched, audio_no_diar=audio_no_diar,
        diar_no_audio=diar_no_audio, author_of=author_of,
        author_status=author_status, new_authors=new_authors,
    )


# ---------------------------------------------------------------------------
# Candidate finding
# ---------------------------------------------------------------------------


def _finder_score(f: Fragment) -> float:
    """The eval set's own ranking score: balanced fragments with more overlap
    rank first (``overlap_s * speaker_balance``)."""
    return f.overlap_s * f.speaker_balance


@dataclass
class ZeroReason:
    rec: str
    reason: str


def _diagnose_zero(turns, total_dur) -> str:
    """Best-effort distributional reason a recording yielded no candidate, by
    relaxing the finder filters one axis at a time from the floor-only baseline.
    Purely diagnostic — not used for selection."""
    from scripts.clarin_fragment_finder import overlap_intervals

    overlaps = overlap_intervals(turns)
    if not overlaps:
        return "no overlap intervals at all (no 2-speaker simultaneity)"
    total_overlap = sum(e - s for s, e in overlaps)
    max_event = max((e - s for s, e in overlaps), default=0.0)
    if total_overlap < SELECTION_PARAMS.min_overlap_s:
        return (f"total overlap {total_overlap:.1f}s < floor "
                f"{SELECTION_PARAMS.min_overlap_s:.0f}s")
    # Overlap exists and clears the floor somewhere; the window-level filters
    # (speaker balance / speech density / length envelope) must be biting.
    # Re-run with the balance + density filters relaxed to see which.
    relaxed_bal = FragmentParams(
        **{**SELECTION_PARAMS.__dict__, "min_speaker_balance": 0.0}
    )
    if find_fragments(turns, total_dur, relaxed_bal):
        return ("overlap present but no window clears speaker-balance "
                f">= {SELECTION_PARAMS.min_speaker_balance:.2f} (one speaker "
                "dominates the overlap-bearing windows)")
    relaxed_dens = FragmentParams(
        **{**SELECTION_PARAMS.__dict__, "min_speech_density": 0.0}
    )
    if find_fragments(turns, total_dur, relaxed_dens):
        return (f"overlap present but speech density < "
                f"{SELECTION_PARAMS.min_speech_density:.2f} in every "
                "overlap-bearing window (too much silence)")
    return (f"overlap present (total {total_overlap:.1f}s, longest event "
            f"{max_event:.1f}s) but no window of "
            f"[{SELECTION_PARAMS.min_length_s:.0f}, "
            f"{SELECTION_PARAMS.max_length_s:.0f}]s satisfies all filters "
            "jointly")


def find_candidates(inv: Inventory) -> tuple[list[dict], list[ZeroReason]]:
    """Run the finder over every unused recording; keep top <=2 per rec.

    Returns (candidate_rows, zero_reasons). Each candidate row carries the
    finder stats, the candidate id, and the joined Korpus metadata.
    """
    rows: list[dict] = []
    zeros: list[ZeroReason] = []

    for rec in inv.unused:
        diar_path = DIAR_DIR / f"{rec}.json"
        turns, total_dur = load_diarization(diar_path)
        frags = find_fragments(turns, total_dur, SELECTION_PARAMS)
        if not frags:
            zeros.append(ZeroReason(rec, _diagnose_zero(turns, total_dur)))
            continue
        # Rank: finder score desc, ties broken by higher overlap_s.
        frags_ranked = sorted(
            frags, key=lambda f: (_finder_score(f), f.overlap_s), reverse=True
        )
        meta = inv.meta.get(rec, {})
        for i, f in enumerate(frags_ranked[:MAX_CANDIDATES_PER_REC]):
            cand_id = f"{rec}__cand{i:02d}"
            row = {
                "rec": rec,
                "cand_id": cand_id,
                "start": round(f.start, 3),
                "end": round(f.end, 3),
                "duration": round(f.duration, 3),
                "overlap_s": round(f.overlap_s, 3),
                "n_overlap_events": f.n_overlap_events,
                "max_event_s": round(f.max_event_s, 3),
                "speaker_balance": round(f.speaker_balance, 4),
                "speech_density": round(f.speech_density, 4),
                "finder_score": round(_finder_score(f), 4),
                "Autor": inv.author_of[rec],
                "author_status": inv.author_status[rec],
            }
            for k in KORPUS_FIELDS:
                if k == "Autor":
                    continue  # already set (UNKNOWN-aware)
                row[k] = (meta.get(k) or "").strip()
            rows.append(row)
    return rows, zeros


# ---------------------------------------------------------------------------
# Audio extraction (same convention as build_clarin_fragment_set.py)
# ---------------------------------------------------------------------------


def _load_mono_16k(path: Path, target_sr: int) -> np.ndarray:
    arr, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if sr != target_sr:
        import torch
        import torchaudio.functional as AF
        t = torch.from_numpy(arr).unsqueeze(0)
        t = AF.resample(t, sr, target_sr)
        arr = t.squeeze(0).numpy().astype(np.float32)
    return arr


def extract_audio(rows: list[dict], force: bool = False,
                  limit: Optional[int] = None) -> tuple[int, int]:
    """Write one WAV per candidate into ``<root>/<cand_id>/<cand_id>.wav``.

    Returns (n_written, n_skipped). Loads each source recording once.
    """
    if limit is not None:
        rows = rows[:limit]
        print(f"  --limit {limit} -> extracting first {len(rows)} candidate(s)")

    by_rec: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_rec[r["rec"]].append(r)

    n_written = n_skipped = 0
    for rec, group in by_rec.items():
        wav_path = AUDIO_DIR / f"{rec}.wav"
        if not wav_path.exists():
            print(f"  [missing] {wav_path} — skipping its candidates")
            continue
        targets = []
        for r in group:
            dest_dir = OUTPUT_ROOT / r["cand_id"]
            dest = dest_dir / f"{r['cand_id']}.wav"
            if dest.exists() and not force:
                n_skipped += 1
                continue
            targets.append((r, dest_dir, dest))
        if not targets:
            continue
        audio = _load_mono_16k(wav_path, OUTPUT_SAMPLE_RATE)
        sr = OUTPUT_SAMPLE_RATE
        for r, dest_dir, dest in targets:
            dest_dir.mkdir(parents=True, exist_ok=True)
            lo = int(r["start"] * sr)
            hi = int(r["end"] * sr)
            clip = audio[lo:hi].astype(np.float32)
            sf.write(str(dest), clip, sr)
            n_written += 1
        del audio
    return n_written, n_skipped


# ---------------------------------------------------------------------------
# Outputs: manifest + inventory doc
# ---------------------------------------------------------------------------


MANIFEST_COLUMNS = [
    "rec", "cand_id", "start", "end", "duration", "overlap_s",
    "n_overlap_events", "max_event_s", "speaker_balance", "speech_density",
    "finder_score", "Autor", "author_status",
    "Poziom szumów", "Środowisko", "Urządzenie nagrywające",
    "Temat rozmowy", "Liczba mówców", "Nazwa", "Identyfikator nagrania",
]


def write_manifest(rows: list[dict]) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(CANDIDATES_MANIFEST, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        w.writeheader()
        for r in sorted(rows, key=lambda r: r["cand_id"]):
            w.writerow({k: r.get(k, "") for k in MANIFEST_COLUMNS})
    print(f"  wrote {CANDIDATES_MANIFEST}  ({len(rows)} candidates)")


def write_inventory_md(inv: Inventory, rows: list[dict],
                       zeros: list[ZeroReason]) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    L: list[str] = []
    a = L.append
    a("# CLARIN candidate-mining inventory\n")
    a(f"Generated: **{datetime.now().isoformat(timespec='seconds')}**\n")
    a(f"Source: `{DIAR_DIR}` + `{AUDIO_DIR}`")
    a(f"Eval set used as the 'used' reference: `{EVAL_MANIFEST}`\n")

    a("## Recording inventory\n")
    a(f"- Audio files: **{len(list(AUDIO_DIR.glob('*.wav')))}**")
    a(f"- Diarization files: **{len(list(DIAR_DIR.glob('*.json')))}**")
    a(f"- Eligible (audio AND diar): **{len(inv.eligible)}**")
    a(f"- Used by the eval set (`rec` column): **{len(inv.used)}**")
    a(f"- **UNUSED eligible recordings: {len(inv.unused)}**\n")
    if inv.diar_no_audio:
        a(f"- Diar without audio (excluded): "
          f"`{', '.join(inv.diar_no_audio)}`")
    if inv.audio_no_diar:
        a(f"- Audio without diar (excluded): "
          f"`{', '.join(inv.audio_no_diar)}`")
    a("")

    a("## Author classification of the unused recordings\n")
    status_counts = Counter(inv.author_status[u] for u in inv.unused)
    a(f"- NEW-author recordings: **{status_counts.get('new', 0)}** "
      f"(across **{len(inv.new_authors)}** distinct new author(s))")
    a(f"- USED-author recordings (new conditions, same speaker): "
      f"**{status_counts.get('used', 0)}**")
    a(f"- UNKNOWN-author recordings (unmatched to Korpus): "
      f"**{status_counts.get('unknown', 0)}**\n")
    if inv.unmatched:
        a(f"**Unmatched to Korpus (reported, still eligible):** "
          f"`{', '.join(inv.unmatched)}` — author treated as UNKNOWN.\n")

    a("### Unused recordings per author\n")
    per_aut: dict[str, list[str]] = defaultdict(list)
    for u in inv.unused:
        per_aut[inv.author_of[u]].append(u)
    a("| author (truncated) | status | unused recs | candidates kept |")
    a("|---|---|---:|---:|")
    cand_by_rec = Counter(r["rec"] for r in rows)
    for aut in sorted(per_aut, key=lambda x: -len(per_aut[x])):
        recs = per_aut[aut]
        status = inv.author_status[recs[0]]
        n_cand = sum(cand_by_rec.get(r, 0) for r in recs)
        a(f"| {aut[:20]} | {status} | {len(recs)} | {n_cand} |")
    a("")

    a("## Candidate yield\n")
    a(f"- Recordings yielding >=1 candidate: "
      f"**{len({r['rec'] for r in rows})}** / {len(inv.unused)}")
    a(f"- Total candidates kept (<= {MAX_CANDIDATES_PER_REC}/rec): "
      f"**{len(rows)}**")
    a(f"- Recordings yielding ZERO candidates: **{len(zeros)}**\n")
    if zeros:
        reason_counts = Counter(_reason_bucket(z.reason) for z in zeros)
        a("Zero-candidate reasons (bucketed):\n")
        a("| reason bucket | count |")
        a("|---|---:|")
        for reason, n in reason_counts.most_common():
            a(f"| {reason} | {n} |")
        a("")
        a("<details><summary>Per-recording zero reasons</summary>\n")
        a("| rec | reason |")
        a("|---|---|")
        for z in sorted(zeros, key=lambda z: z.rec):
            a(f"| {z.rec} | {z.reason} |")
        a("\n</details>")
    a("")

    a("## Finder parameters (copied from the eval set's SELECTION.md)\n")
    a("```python")
    for k, v in SELECTION_PARAMS.__dict__.items():
        a(f"{k} = {v!r}")
    a("```\n")
    a("> Acoustic scoring + the decision menu are produced separately: run\n"
      "> `score_fragment_acoustics.py --root <this tree> --csv-out "
      "<...> --no-brouhaha? --no-report`\n"
      "> then `build_candidate_report.py`.\n")

    INVENTORY_MD.write_text("\n".join(L), encoding="utf-8")
    print(f"  wrote {INVENTORY_MD}")


def _reason_bucket(reason: str) -> str:
    if "no overlap intervals" in reason:
        return "no 2-speaker overlap at all"
    if "< floor" in reason:
        return "total overlap below floor"
    if "speaker-balance" in reason:
        return "one speaker dominates overlap windows"
    if "speech density" in reason:
        return "speech density too low in overlap windows"
    return "filters not jointly satisfiable"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true",
                    help="inventory + find + write manifest/inventory, skip audio")
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing candidate WAVs")
    ap.add_argument("--limit", type=int, default=None,
                    help="extract only the first N candidates (smoke check)")
    args = ap.parse_args(argv)

    print("=== CLARIN candidate mining ===")
    print(f"  source diar:  {DIAR_DIR}")
    print(f"  source audio: {AUDIO_DIR}")
    print(f"  output:       {OUTPUT_ROOT}\n")

    print("=== Inventory ===")
    inv = build_inventory()
    print(f"  eligible={len(inv.eligible)} used={len(inv.used)} "
          f"unused={len(inv.unused)}")
    print(f"  new authors={len(inv.new_authors)} "
          f"unmatched-to-Korpus={len(inv.unmatched)}")
    if inv.unmatched:
        print(f"  [unmatched, author=UNKNOWN] {', '.join(inv.unmatched)}")

    print("\n=== Finding candidates ===")
    rows, zeros = find_candidates(inv)
    print(f"  {len(rows)} candidate(s) from "
          f"{len({r['rec'] for r in rows})} recording(s); "
          f"{len(zeros)} recording(s) yielded zero")

    print("\n=== Writing manifest + inventory ===")
    write_manifest(rows)
    write_inventory_md(inv, rows, zeros)

    if args.dry_run:
        print("\ndry-run: skipping audio extraction")
        return 0

    print("\n=== Extracting audio ===")
    n_written, n_skipped = extract_audio(rows, force=args.force, limit=args.limit)
    print(f"  wrote {n_written} WAV(s), skipped {n_skipped} existing")
    print(f"\ndone. inspect {INVENTORY_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
