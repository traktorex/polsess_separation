"""Build the CLARIN fragment test set.

Runs `clarin_fragment_finder.find_fragments` over every diarization in
``~/datasets/clarin_all_2speakers/``, applies stratified sampling
(overlap_bin × noise_level), enforces a per-recording quota, extracts
the audio slices, and writes a manifest + decision log.

Output layout::

    ~/datasets/clarin_all_2speaker_fragments/
      SELECTION.md                 decision log (params + counts + date)
      manifest.csv                 one row per selected fragment
      fragments/
        <rec>__seg<idx>.wav        mono 16 kHz slice

The selection plan is committed into ``SELECTION_PLAN`` below; tweak it
here (then re-run) rather than threading every knob through the CLI.
Idempotent: re-running with the same plan skips work that's already on
disk unless ``--force`` is passed.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.clarin_fragment_finder import (
    Fragment, FragmentParams,
    find_fragments, load_diarization,
)


# ---------------------------------------------------------------------------
# Selection plan — the committed design
# ---------------------------------------------------------------------------


SOURCE_ROOT  = Path("~/datasets/clarin_all_2speakers").expanduser()
AUDIO_DIR    = SOURCE_ROOT / "clarin_download"
DIAR_DIR     = SOURCE_ROOT / "diarization"
CSV_PATH     = AUDIO_DIR / "Korpus_with_filename.csv"

OUTPUT_ROOT  = Path("~/datasets/clarin_all_2speaker_fragments").expanduser()


@dataclass(frozen=True)
class SelectionPlan:
    """Everything that determines which fragments end up in the subset."""

    # Fragment-finder params — passed straight through.
    params: FragmentParams = field(default_factory=lambda: FragmentParams(
        target_length_s=90.0, min_length_s=60.0, max_length_s=120.0,
        stride_s=5.0,
        silence_snap_min_s=0.5, silence_snap_search_s=5.0,
        min_overlap_s=3.0,
        min_speaker_balance=0.20,
        min_speech_density=0.55,
        nms_iou_threshold=0.10,
    ))

    # Overlap-bin edges (in seconds of overlap inside the fragment).
    # Bins are: light < edges[0]; moderate < edges[1]; heavy >= edges[1].
    overlap_bin_edges_s: tuple = (6.0, 10.0)

    # Substantive-vs-backchannel discriminator.
    #
    # A fragment is "substantive" if its single longest overlap event lasts
    # at least this many seconds — i.e. the speakers actually talk over each
    # other for a measurable stretch, not just backchannel like "mhm" or
    # "tak". Sub-1s overlaps are usually below the separator's effective
    # time resolution anyway, so they don't really test what we want to test.
    #
    # Two pools per cell:
    # - substantive (max_event_s >= threshold) — the main test set
    # - backchannel (max_event_s <  threshold) — kept ONLY in the cells that
    #   carry a non-zero backchannel target (light + moderate; see
    #   `backchannel_targets`), to surface this case without overweighting it.
    min_substantive_event_s: float = 1.0

    # Per-cell SUBSTANTIVE target counts. Tuned so the row totals match the
    # original plan minus the reserved backchannel slots in that row.
    substantive_targets_per_cell: dict = field(default_factory=lambda: {
        ("light",    "Niski"):  19,   # was 20; 1 slot reserved for backchannel
        ("light",    "Średni"): 19,   # was 20; 1 slot reserved for backchannel
        ("light",    "Wysoki"):  5,
        ("moderate", "Niski"):  19,   # was 20; 1 slot reserved for backchannel
        ("moderate", "Średni"): 20,
        ("moderate", "Wysoki"):  5,
        ("heavy",    "Niski"):  20,
        ("heavy",    "Średni"): 20,
        ("heavy",    "Wysoki"):  5,
    })

    # Per-cell BACKCHANNEL-ONLY target counts. Total 3 fragments across the
    # whole subset: 2 in the light row (Niski + Średni), 1 in the moderate
    # row (Niski). Wysoki rows get none — the cells are already tight on
    # source recordings; spending one of those slots on backchannel-only
    # isn't useful.
    backchannel_targets_per_cell: dict = field(default_factory=lambda: {
        ("light",    "Niski"):  1,
        ("light",    "Średni"): 1,
        ("light",    "Wysoki"): 0,
        ("moderate", "Niski"):  1,
        ("moderate", "Średni"): 0,
        ("moderate", "Wysoki"): 0,
        ("heavy",    "Niski"):  0,
        ("heavy",    "Średni"): 0,
        ("heavy",    "Wysoki"): 0,
    })

    # Per-recording quota by noise level. Wysoki gets a higher quota
    # because its source pool is tiny (~3-5 distinct recordings) and the
    # quota is shared across all overlap bins — without bumping, earlier
    # cells consume the only candidates and the heavy×Wysoki cell falls
    # to zero. Quota=3 means each Wysoki recording can contribute up to
    # one fragment per overlap bin.
    quota_by_noise: dict = field(default_factory=lambda: {
        "Niski":  1, "Średni": 1, "Wysoki": 3, "unknown": 1,
    })

    # Output sample rate. Pipeline expects 16 kHz mono.
    output_sample_rate: int = 16_000


# Module-level singleton (mutate by editing the dataclass defaults above).
PLAN = SelectionPlan()


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def _bin_overlap(overlap_s: float, edges: tuple) -> str:
    if overlap_s < edges[0]:  return "light"
    if overlap_s < edges[1]:  return "moderate"
    return "heavy"


def _score(f: Fragment) -> float:
    """Ranking score: balanced fragments with more overlap come first."""
    return f.overlap_s * f.speaker_balance


def collect_candidates(plan: SelectionPlan, noise_by_rec: dict[str, str]) -> pd.DataFrame:
    """Run the finder on every diarization → DataFrame of candidates."""
    rows = []
    for p in sorted(DIAR_DIR.glob("*.json")):
        rid = p.stem
        turns, dur = load_diarization(p)
        frags = find_fragments(turns, dur, plan.params)
        noise = noise_by_rec.get(rid, "unknown")
        for f in frags:
            rows.append({
                "rec": rid,
                "noise": noise,
                "overlap_bin": _bin_overlap(f.overlap_s, plan.overlap_bin_edges_s),
                "score": _score(f),
                "start": f.start, "end": f.end, "duration": f.duration,
                "overlap_s": f.overlap_s,
                "n_overlap_events": f.n_overlap_events,
                "max_event_s": f.max_event_s,
                "is_substantive": f.max_event_s >= plan.min_substantive_event_s,
                "speech_density": f.speech_density,
                "speaker_balance": f.speaker_balance,
                **{
                    f"spk_{k}_s": v
                    for k, v in sorted(f.speech_s_by_spk.items())
                },
            })
    return pd.DataFrame(rows)


def _take_from_pool(
    cell: pd.DataFrame, target: int, quota: int,
    per_rec_count: dict[str, int],
) -> list[dict]:
    """Pick top-scoring `target` rows from `cell` while respecting the
    shared `per_rec_count` quota. Mutates `per_rec_count` in place."""
    chosen: list[dict] = []
    for _, row in cell.iterrows():
        if len(chosen) >= target:
            break
        rec = row["rec"]
        if per_rec_count.get(rec, 0) >= quota:
            continue
        chosen.append(row.to_dict())
        per_rec_count[rec] = per_rec_count.get(rec, 0) + 1
    return chosen


def select(plan: SelectionPlan, candidates: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Stratified selection in two passes per cell: substantive first, then
    backchannel. Per-recording quota is shared across both passes and across
    all cells, so the same recording can't be picked twice.

    Returns (selected_df, audit_dict) where audit_dict tracks target/got
    for both passes per cell.
    """
    selected_rows: list[dict] = []
    per_rec_count: dict[str, int] = {}
    audit: dict[tuple[str, str], dict] = {}

    cell_keys = list(plan.substantive_targets_per_cell.keys())
    for (bin_, noise) in cell_keys:
        substantive_target = plan.substantive_targets_per_cell[(bin_, noise)]
        backchannel_target = plan.backchannel_targets_per_cell.get((bin_, noise), 0)
        quota = plan.quota_by_noise.get(noise, 1)

        cell_all = candidates[
            (candidates["overlap_bin"] == bin_) & (candidates["noise"] == noise)
        ].sort_values("score", ascending=False)
        cell_sub  = cell_all[cell_all["is_substantive"]]
        cell_back = cell_all[~cell_all["is_substantive"]]

        chosen_sub = _take_from_pool(
            cell_sub, substantive_target, quota, per_rec_count,
        )
        chosen_back = _take_from_pool(
            cell_back, backchannel_target, quota, per_rec_count,
        )

        # Tag each row with which pass picked it.
        for r in chosen_sub:  r["pool"] = "substantive"
        for r in chosen_back: r["pool"] = "backchannel"

        selected_rows.extend(chosen_sub)
        selected_rows.extend(chosen_back)
        audit[(bin_, noise)] = {
            "substantive_target": substantive_target,
            "substantive_got":    len(chosen_sub),
            "backchannel_target": backchannel_target,
            "backchannel_got":    len(chosen_back),
            "available_sub_candidates":  len(cell_sub),
            "available_back_candidates": len(cell_back),
            "available_sub_recordings":  cell_sub["rec"].nunique() if not cell_sub.empty else 0,
            "available_back_recordings": cell_back["rec"].nunique() if not cell_back.empty else 0,
            "recordings_used": sorted({c["rec"] for c in (chosen_sub + chosen_back)}),
        }

    selected = pd.DataFrame(selected_rows)
    if not selected.empty:
        selected = selected.sort_values(["overlap_bin", "noise", "rec", "start"]).reset_index(drop=True)
        # Re-number per recording so frag_id is human-readable.
        idx_by_rec: dict[str, int] = {}
        new_ids = []
        for _, row in selected.iterrows():
            rec = row["rec"]
            i = idx_by_rec.get(rec, 0)
            new_ids.append(f"{rec}__seg{i:02d}")
            idx_by_rec[rec] = i + 1
        selected["frag_id"] = new_ids
    return selected, audit


# ---------------------------------------------------------------------------
# Audio extraction
# ---------------------------------------------------------------------------


def _load_mono_16k(path: Path, target_sr: int) -> np.ndarray:
    arr, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if sr != target_sr:
        # Lazy resample via torchaudio (avoids adding scipy / librosa for this).
        import torch
        import torchaudio.functional as AF
        t = torch.from_numpy(arr).unsqueeze(0)
        t = AF.resample(t, sr, target_sr)
        arr = t.squeeze(0).numpy().astype(np.float32)
    return arr


def extract_audio(selected: pd.DataFrame, plan: SelectionPlan,
                  force: bool = False, limit: Optional[int] = None) -> int:
    """Write one WAV per selected fragment. Returns the number actually
    written (skipped files don't count).

    ``limit`` truncates the work to the first N rows of ``selected`` (in
    the manifest's row order — by overlap_bin / noise / rec / start). The
    manifest itself is unchanged; only audio extraction is limited. Useful
    for spot-checking the first fragment before committing to the full run.
    """
    out_dir = OUTPUT_ROOT / "fragments"
    out_dir.mkdir(parents=True, exist_ok=True)

    if limit is not None:
        selected = selected.head(limit)
        print(f"  --limit {limit} → extracting first {len(selected)} fragment(s)")

    by_rec = selected.groupby("rec")
    n_written = 0
    n_skipped = 0
    t0 = time.perf_counter()
    for rec, group in by_rec:
        wav_path = AUDIO_DIR / f"{rec}.wav"
        if not wav_path.exists():
            print(f"  [missing] {wav_path}")
            continue
        # Decide whether we need to load (any fragment for this rec needs work?)
        targets = []
        for _, row in group.iterrows():
            dest = out_dir / f"{row['frag_id']}.wav"
            if dest.exists() and not force:
                n_skipped += 1
                continue
            targets.append((row, dest))
        if not targets:
            continue

        audio = _load_mono_16k(wav_path, plan.output_sample_rate)
        sr = plan.output_sample_rate
        for row, dest in targets:
            lo = int(row["start"] * sr)
            hi = int(row["end"]   * sr)
            clip = audio[lo:hi].astype(np.float32)
            sf.write(str(dest), clip, sr)
            n_written += 1
        del audio   # free large array between recordings

    print(f"  wrote {n_written} fragment WAV(s), skipped {n_skipped} "
          f"(in {time.perf_counter()-t0:.1f}s)")
    return n_written


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


def write_manifest(selected: pd.DataFrame, csv_df: pd.DataFrame) -> Path:
    """Write manifest.csv. Joins source-recording metadata from the CLARIN CSV
    onto each fragment row so users have all context in one file."""
    csv_keep = [
        "Nazwa pliku WAV", "Nazwa", "Domena", "Styl", "Temat rozmowy",
        "Poziom szumów", "Typ szumów", "Środowisko",
        "Urządzenie nagrywające", "Typ mikrofonu", "Model",
        "Autor", "Sesja", "Identyfikator nagrania", "Status",
    ]
    csv_subset = csv_df[csv_keep].copy()
    csv_subset["rec"] = csv_subset["Nazwa pliku WAV"].str.replace(".wav", "", regex=False)
    csv_subset = csv_subset.drop(columns=["Nazwa pliku WAV"])

    out = selected.merge(csv_subset, on="rec", how="left")
    manifest_path = OUTPUT_ROOT / "manifest.csv"
    out.to_csv(manifest_path, index=False, encoding="utf-8")
    print(f"  wrote {manifest_path}  ({len(out)} rows)")
    return manifest_path


def write_selection_md(plan: SelectionPlan, selected: pd.DataFrame,
                       audit: dict) -> Path:
    """The decision-log document — every parameter + the cell-fill table."""
    md_path = OUTPUT_ROOT / "SELECTION.md"
    n_selected = len(selected)
    total_dur_s = selected["duration"].sum() if not selected.empty else 0
    n_recs = selected["rec"].nunique() if not selected.empty else 0

    lines: list[str] = []
    lines.append("# CLARIN fragment test set — selection log\n\n")
    lines.append(f"Generated: **{datetime.now().isoformat(timespec='seconds')}**\n\n")
    lines.append(f"Source: `{DIAR_DIR}` (diarization) + `{AUDIO_DIR}` (audio)\n")
    lines.append(f"Output: `{OUTPUT_ROOT}`\n\n")

    lines.append("## Result\n\n")
    lines.append(f"- Fragments selected: **{n_selected}**\n")
    lines.append(f"- Total audio: **{total_dur_s/60:.1f} min** ({total_dur_s/3600:.2f} h)\n")
    lines.append(f"- Source recordings represented: **{n_recs}**\n\n")

    lines.append("## FragmentParams (fragment-finder)\n\n")
    lines.append("```python\n")
    for k, v in asdict(plan.params).items():
        lines.append(f"{k} = {v!r}\n")
    lines.append("```\n\n")

    lines.append("## Stratification\n\n")
    lines.append(f"Overlap-bin edges (seconds): `{plan.overlap_bin_edges_s}` → "
                 f"light < {plan.overlap_bin_edges_s[0]:.1f}s, "
                 f"moderate < {plan.overlap_bin_edges_s[1]:.1f}s, "
                 f"heavy ≥ {plan.overlap_bin_edges_s[1]:.1f}s.\n\n")
    lines.append("Per-recording quota by noise level: "
                 + ", ".join(f"{k}={v}" for k, v in plan.quota_by_noise.items())
                 + ".\n\n")
    lines.append("Ranking score within each cell: `overlap_s × speaker_balance` "
                 "(rewards balanced conversations with substantial overlap).\n\n")
    lines.append(f"Substantive-vs-backchannel threshold: "
                 f"`max_event_s >= {plan.min_substantive_event_s:.1f}s`. "
                 "A fragment is *substantive* if its single longest overlap event\n"
                 "lasts at least this long; otherwise it is *backchannel-only*.\n"
                 "Sub-1 s overlaps are usually too short for the separator to\n"
                 "meaningfully act on (well below its training-chunk time\n"
                 "resolution), so they don't test what we mean to test.\n\n")
    lines.append("Backchannel-only fragments are explicitly under-represented "
                 "(small reserved slots in the light and moderate rows only) so the\n"
                 "case is surfaced without dominating the test set:\n\n")
    lines.append("| overlap | noise | substantive target | backchannel target |\n")
    lines.append("| --- | --- | ---:| ---:|\n")
    for (bin_, noise), sub_target in plan.substantive_targets_per_cell.items():
        back_target = plan.backchannel_targets_per_cell.get((bin_, noise), 0)
        lines.append(f"| {bin_} | {noise} | {sub_target} | {back_target} |\n")
    lines.append("\n")

    lines.append("## Cell fill\n\n")
    lines.append("`(target → got)` per pool, plus available candidates and recordings used.\n\n")
    lines.append("| overlap | noise | substantive | backchannel | recordings used | "
                 "sub. avail. recs (cands) | back. avail. recs (cands) |\n")
    lines.append("| --- | --- |:---:|:---:| ---:| ---:| ---:|\n")
    for (bin_, noise), info in audit.items():
        lines.append(
            f"| {bin_} | {noise} | "
            f"{info['substantive_got']}/{info['substantive_target']} | "
            f"{info['backchannel_got']}/{info['backchannel_target']} | "
            f"{len(info['recordings_used'])} | "
            f"{info['available_sub_recordings']} ({info['available_sub_candidates']}) | "
            f"{info['available_back_recordings']} ({info['available_back_candidates']}) |\n"
        )
    lines.append("\n")

    lines.append("## Methodology notes\n\n")
    lines.append("- **GT bootstrap**: per-fragment GT will be produced by running\n")
    lines.append("  the pipeline with `separation.enabled=false` (no separator), then\n")
    lines.append("  hand-correcting the WhisperX-generated per-speaker transcripts.\n")
    lines.append("  Listening pass catches omissions; content is corrected, timestamps\n")
    lines.append("  are inherited from WhisperX (good to ±50 ms with the Polish\n")
    lines.append("  wav2vec2 align model). Document this caveat in any tcpWER claims.\n\n")
    lines.append("- **Status filter dropped**: CSV `Status` field reflects the\n")
    lines.append("  source platform's curation state; the author intends to promote\n")
    lines.append("  every selected source to `gotowy` during the GT-creation pass,\n")
    lines.append("  so all 209 diarized recordings are eligible up front.\n\n")
    lines.append("- **Wysoki cells under-target**: the corpus only contains a handful\n")
    lines.append("  of high-noise recordings (≤ 5 unique). Per-cell counts are reduced\n")
    lines.append("  to 5 each and the per-recording quota is bumped to 3 — without it\n")
    lines.append("  the heavy×Wysoki cell falls to zero because light/moderate cells\n")
    lines.append("  consume the shared source pool first. Sub-claims about high-noise\n")
    lines.append("  robustness will be weak; report Wysoki numbers but don't generalise\n")
    lines.append("  from them.\n\n")

    md_path.write_text("".join(lines), encoding="utf-8")
    print(f"  wrote {md_path}")
    return md_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true",
                        help="Run selection + write manifest/MD, but skip audio extraction.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing fragment WAVs.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Extract audio for only the first N fragments "
                             "(manifest order). Useful for smoke-checking before "
                             "committing to the full extraction.")
    args = parser.parse_args()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"=== CLARIN fragment set build ===")
    print(f"  source diarization: {DIAR_DIR}")
    print(f"  source audio:       {AUDIO_DIR}")
    print(f"  output:             {OUTPUT_ROOT}")
    print()

    csv_df = pd.read_csv(CSV_PATH)
    noise_by_rec = dict(
        zip(csv_df["Nazwa pliku WAV"].str.replace(".wav", "", regex=False),
            csv_df["Poziom szumów"].fillna("unknown"))
    )

    print(f"=== Collecting candidates ===")
    candidates = collect_candidates(PLAN, noise_by_rec)
    print(f"  {len(candidates)} candidate(s) from "
          f"{candidates['rec'].nunique()} recording(s)")
    print()

    print(f"=== Selecting ===")
    selected, audit = select(PLAN, candidates)
    print(f"  selected {len(selected)} fragment(s) from "
          f"{selected['rec'].nunique()} recording(s)")
    print(f"  total audio: {selected['duration'].sum()/60:.1f} min")
    print()

    # Per-cell summary
    print(f"=== Cell fill (substantive | backchannel) ===")
    for (bin_, noise), info in audit.items():
        sub_t = info["substantive_target"]
        sub_g = info["substantive_got"]
        bk_t  = info["backchannel_target"]
        bk_g  = info["backchannel_got"]
        cell_total_t = sub_t + bk_t
        cell_total_g = sub_g + bk_g
        print(f"  {bin_:>8s} × {noise:<7s}  "
              f"sub {sub_g:>2d}/{sub_t:<2d}  back {bk_g}/{bk_t}  "
              f"= {cell_total_g:>2d}/{cell_total_t:<2d}  "
              f"(sub avail: {info['available_sub_recordings']:>3d} recs / "
              f"{info['available_sub_candidates']:>4d} cands; "
              f"back avail: {info['available_back_recordings']:>2d}/"
              f"{info['available_back_candidates']:>3d})")
    print()

    print(f"=== Writing manifest + decision log ===")
    write_manifest(selected, csv_df)
    write_selection_md(PLAN, selected, audit)
    print()

    if args.dry_run:
        print("dry-run: skipping audio extraction")
        return 0

    print(f"=== Extracting audio ===")
    extract_audio(selected, PLAN, force=args.force, limit=args.limit)

    print()
    print(f"done. inspect {OUTPUT_ROOT}/SELECTION.md for the decision log.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
