"""Build a slim 2-speaker LibriCSS test set from the two_speaker_segments.csv.

Reads `<src>/two_speaker_segments.csv` (34 segments where only 2 speakers
participate) and writes a flat, minimal tree under `<dst>` containing
just those 34 segments — discarding the other ~95% of LibriCSS that has
3+ speakers and is irrelevant for our 2-speaker pipeline.

Target layout::

    <dst>/
        record/segments/<key>.wav          # mono 16k, channel 0 of the 7-ch array
        clean/segments/<key>_spkA.wav      # speaker_a clean reference
        clean/segments/<key>_spkB.wav      # speaker_b clean reference
        transcriptions/<key>.txt           # per-segment text with speaker labels
        manifest.csv                       # rows from input CSV + new paths
        README.md

`<key>` = `<cond>_session<N>_seg<idx>` (e.g. `0L_session1_seg3`).

Channel-to-speaker mapping in `clean/each_spk.wav` is undocumented, so we
match channels to speaker IDs empirically per meeting: for each channel we
find the first non-silent sample, and for each speaker we find the first
utterance start time from `meeting_info.txt`; channels are paired to
speakers by nearest first-activity time.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16_000
SILENCE_THRESHOLD = 1e-3   # for channel→speaker matching only


def parse_meeting_info(path: Path) -> list[dict]:
    """Return list of {start_s, end_s, speaker, utterance_id, text}."""
    rows = []
    with path.open() as f:
        lines = f.readlines()[1:]   # skip header
    for ln in lines:
        parts = ln.rstrip("\n").split("\t")
        rows.append({
            "start_s": float(parts[0]),
            "end_s": float(parts[1]),
            "speaker": parts[2],
            "utterance_id": parts[3],
            "text": parts[4],
        })
    return rows


def derive_channel_to_speaker(each_spk: np.ndarray, meeting_info: list[dict]) -> dict[int, str]:
    """Map channel index → speaker id by first-activity-time nearest neighbour."""
    n_ch = each_spk.shape[1]
    ch_first = {}
    for ch in range(n_ch):
        nz = np.where(np.abs(each_spk[:, ch]) > SILENCE_THRESHOLD)[0]
        if len(nz):
            ch_first[ch] = nz[0] / SR
        else:
            ch_first[ch] = float("inf")

    spk_first: dict[str, float] = {}
    for row in meeting_info:
        spk = row["speaker"]
        if spk not in spk_first or row["start_s"] < spk_first[spk]:
            spk_first[spk] = row["start_s"]

    used_speakers: set[str] = set()
    mapping: dict[int, str] = {}
    for ch in sorted(ch_first, key=ch_first.get):
        best_spk = None
        best_delta = float("inf")
        for spk, t in spk_first.items():
            if spk in used_speakers:
                continue
            d = abs(t - ch_first[ch])
            if d < best_delta:
                best_delta = d
                best_spk = spk
        if best_spk is not None:
            mapping[ch] = best_spk
            used_speakers.add(best_spk)
    return mapping


def session_number_from_meeting(meeting: str) -> str:
    """`overlap_ratio_0.0_sil2.9_3.0_session1_actual0.0` -> `session1`."""
    m = re.search(r"session(\d+)", meeting)
    return f"session{m.group(1)}" if m else meeting


def slice_clean_for_speaker(each_spk: np.ndarray, channel: int, start_s: float, end_s: float) -> np.ndarray:
    start = int(start_s * SR)
    end = int(end_s * SR)
    return each_spk[start:end, channel].astype(np.float32)


def utterances_in_segment(meeting_info: list[dict], start_s: float, end_s: float,
                          spk_a: str, spk_b: str) -> list[dict]:
    """Utterances overlapping the segment by speaker_a or speaker_b only."""
    out = []
    for row in meeting_info:
        if row["speaker"] not in (spk_a, spk_b):
            continue
        # Keep utterances whose midpoint is inside [start_s, end_s]
        mid = 0.5 * (row["start_s"] + row["end_s"])
        if start_s <= mid <= end_s:
            label = "A" if row["speaker"] == spk_a else "B"
            out.append({
                "start_s": row["start_s"] - start_s,
                "end_s": row["end_s"] - start_s,
                "speaker_label": label,
                "speaker_id": row["speaker"],
                "utterance_id": row["utterance_id"],
                "text": row["text"],
            })
    out.sort(key=lambda r: r["start_s"])
    return out


def build(src_root: Path, csv_path: Path, dst_root: Path) -> None:
    dst_root.mkdir(parents=True, exist_ok=True)
    (dst_root / "record" / "segments").mkdir(parents=True, exist_ok=True)
    (dst_root / "clean" / "segments").mkdir(parents=True, exist_ok=True)
    (dst_root / "transcriptions").mkdir(parents=True, exist_ok=True)

    with csv_path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    manifest: list[dict] = []

    # Cache per-meeting heavy work (each_spk.wav is up to 600s * 8ch).
    meeting_cache: dict[tuple[str, str], dict] = {}

    for row in rows:
        cond = row["condition"]
        meeting = row["meeting"]
        seg_idx = int(row["segment_idx"])
        start_s = float(row["start_s"])
        end_s = float(row["end_s"])
        spk_a = row["speaker_a"]
        spk_b = row["speaker_b"]

        meeting_dir = src_root / cond / meeting
        record_seg_path = meeting_dir / "record" / "segments" / f"segment_{seg_idx}.wav"
        if not record_seg_path.exists():
            print(f"  SKIP {cond}/{meeting}/segment_{seg_idx}: missing source")
            continue

        cache_key = (cond, meeting)
        if cache_key not in meeting_cache:
            each_spk, sr = sf.read(meeting_dir / "clean" / "each_spk.wav")
            assert sr == SR
            info = parse_meeting_info(meeting_dir / "transcription" / "meeting_info.txt")
            ch_to_spk = derive_channel_to_speaker(each_spk, info)
            spk_to_ch = {spk: ch for ch, spk in ch_to_spk.items()}
            meeting_cache[cache_key] = {
                "each_spk": each_spk,
                "info": info,
                "spk_to_ch": spk_to_ch,
            }
        cache = meeting_cache[cache_key]
        spk_to_ch = cache["spk_to_ch"]
        if spk_a not in spk_to_ch or spk_b not in spk_to_ch:
            print(f"  SKIP {cond}/{meeting}/segment_{seg_idx}: speaker not in channel map "
                  f"(spk_a={spk_a}, spk_b={spk_b}, mapped={list(spk_to_ch)})")
            continue

        # --- key + paths
        key = f"{cond}_{session_number_from_meeting(meeting)}_seg{seg_idx}"
        record_out = dst_root / "record" / "segments" / f"{key}.wav"
        clean_a_out = dst_root / "clean" / "segments" / f"{key}_spkA.wav"
        clean_b_out = dst_root / "clean" / "segments" / f"{key}_spkB.wav"
        trans_out = dst_root / "transcriptions" / f"{key}.txt"

        # --- record (mono ch0 from 7-ch array; the file is already segment-cut)
        record_full, sr = sf.read(record_seg_path)
        assert sr == SR
        if record_full.ndim == 2:
            record_full = record_full[:, 0]
        sf.write(record_out, record_full.astype(np.float32), SR)

        # --- clean per speaker (sliced from meeting-level each_spk.wav)
        clean_a = slice_clean_for_speaker(cache["each_spk"], spk_to_ch[spk_a], start_s, end_s)
        clean_b = slice_clean_for_speaker(cache["each_spk"], spk_to_ch[spk_b], start_s, end_s)
        sf.write(clean_a_out, clean_a, SR)
        sf.write(clean_b_out, clean_b, SR)

        # --- transcript
        utts = utterances_in_segment(cache["info"], start_s, end_s, spk_a, spk_b)
        with trans_out.open("w") as f:
            f.write(f"# segment: {key}\n")
            f.write(f"# condition: {cond}\n")
            f.write(f"# meeting: {meeting}\n")
            f.write(f"# segment_in_meeting: [{start_s:.2f}s, {end_s:.2f}s] (duration {end_s - start_s:.2f}s)\n")
            f.write(f"# speaker_A_id: {spk_a}\n")
            f.write(f"# speaker_B_id: {spk_b}\n")
            f.write("# columns: start_s\tend_s\tspeaker\tutterance_id\ttext\n")
            for u in utts:
                f.write(f"{u['start_s']:.3f}\t{u['end_s']:.3f}\t{u['speaker_label']}\t"
                        f"{u['utterance_id']}\t{u['text']}\n")

        manifest.append({
            "key": key,
            "condition": cond,
            "meeting": meeting,
            "segment_idx": seg_idx,
            "duration_s": end_s - start_s,
            "n_utterances": row["n_utterances"],
            "speaker_a_id": spk_a,
            "speaker_b_id": spk_b,
            "record_path": str(record_out.relative_to(dst_root)),
            "clean_a_path": str(clean_a_out.relative_to(dst_root)),
            "clean_b_path": str(clean_b_out.relative_to(dst_root)),
            "transcript_path": str(trans_out.relative_to(dst_root)),
        })

    manifest_path = dst_root / "manifest.csv"
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        writer.writeheader()
        writer.writerows(manifest)

    readme = dst_root / "README.md"
    readme.write_text(
        "# LibriCSS — 2-speaker slim subset\n\n"
        "Built from the LibriCSS `for_release/` distribution by selecting only the\n"
        "segments in `two_speaker_segments.csv` (segments in which only 2 distinct\n"
        "LibriSpeech speakers are active). The remaining 95% of LibriCSS, which has\n"
        "3+ speakers per segment, is irrelevant for our 2-speaker pipeline and was\n"
        "discarded to save disk.\n\n"
        "## Layout\n\n"
        "```\n"
        "LibriCSS_2spk/\n"
        "├── record/segments/<key>.wav        # mono 16 kHz, channel 0 of the 7-ch array\n"
        "├── clean/segments/<key>_spkA.wav    # clean reference for speaker A (speaker_a in CSV)\n"
        "├── clean/segments/<key>_spkB.wav    # clean reference for speaker B (speaker_b in CSV)\n"
        "├── transcriptions/<key>.txt         # per-segment text with timestamps and A/B labels\n"
        "└── manifest.csv\n"
        "```\n\n"
        "`<key>` = `<cond>_session<N>_seg<idx>`. Conditions: 0L, 0S, OV10, OV20,\n"
        "OV30, OV40 (overlap-ratio categories from LibriCSS).\n\n"
        "## Provenance\n\n"
        "- Source: LibriCSS, Chen et al., ICASSP 2020 (`microsoft/LibriCSS`).\n"
        "- Channel→speaker mapping in `clean/each_spk.wav` is derived empirically\n"
        "  per meeting by matching first-activity time to `meeting_info.txt`\n"
        "  utterance start times. (LibriCSS does not document the channel ordering.)\n"
        "- The `record/segments/<key>.wav` keeps channel 0 of the array (standard\n"
        "  LibriCSS evaluation convention).\n"
    )

    print(f"\nBuilt {len(manifest)} segments in {dst_root}")
    print(f"  manifest: {manifest_path}")
    print(f"  readme:   {readme}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--src", default="/home/user/datasets/LibriCSS",
                   help="LibriCSS root containing for_release/ and two_speaker_segments.csv")
    p.add_argument("--csv", default=None,
                   help="Path to two_speaker_segments.csv (default: <src>/two_speaker_segments.csv)")
    p.add_argument("--dst", default="/home/user/datasets/LibriCSS_2spk",
                   help="Target slim-tree root")
    p.add_argument("--clean-dst", action="store_true",
                   help="rm -rf the target before building (use after a failed run)")
    args = p.parse_args()

    src = Path(args.src)
    csv_path = Path(args.csv) if args.csv else src / "two_speaker_segments.csv"
    dst = Path(args.dst)
    src_release = src / "for_release"

    if args.clean_dst and dst.exists():
        print(f"Removing existing {dst}")
        shutil.rmtree(dst)

    build(src_release, csv_path, dst)


if __name__ == "__main__":
    main()
