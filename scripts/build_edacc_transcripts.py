"""Build per-conversation transcript files for EdAcc from the HuggingFace mirror.

The DataShare zip ships full-conversation audio (`edacc/audios/EAEC-Cxx*.wav`)
but no transcripts. The HF mirror (`edinburghcstr/edacc`) ships per-utterance
rows with `speaker`, `text`, and an audio path like `EDACC-C35_P1-53.wav` — but
no timing columns.

For cpWER we don't need timings: we need per-speaker text in utterance order.
This script:

  1. Downloads every parquet file from the HF mirror.
  2. Extracts (conv_id, speaker_label, utterance_idx, text) per row.
  3. Writes one transcript file per conversation:
       transcripts/<conv_id>.txt
       header lines: # conv: ..., # speaker_A_id: ..., # speaker_B_id: ...
       columns: utterance_idx, speaker, text   (speaker = A | B)
  4. Optionally cleans up the parquet cache.

Conversation ID prefix: HF uses `EDACC-`, DataShare uses `EAEC-`. We canonicalise
to `EAEC-` to match the audio filenames on disk.
"""

from __future__ import annotations

import argparse
import re
import shutil
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

REPO = "edinburghcstr/edacc"


def list_parquets() -> list[str]:
    api = HfApi()
    files = api.list_repo_files(REPO, repo_type="dataset")
    return sorted(f for f in files if f.endswith(".parquet"))


def download_parquets(remote_paths: list[str], cache_dir: Path) -> list[Path]:
    """Download to a real HF cache directory so re-runs hit the cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    local = []
    for i, rp in enumerate(remote_paths):
        print(f"[{i+1}/{len(remote_paths)}] {rp}")
        p = hf_hub_download(REPO, rp, repo_type="dataset", cache_dir=str(cache_dir))
        local.append(Path(p))
    return local


# audio.path examples observed:
#   EDACC-C35_P1-53.wav           -> conv=EDACC-C35_P1  idx=53
#   EDACC-C06-12.wav              -> conv=EDACC-C06     idx=12
_PATH_RE = re.compile(r"^(?P<conv>.+)-(?P<idx>\d+)\.wav$")


def parse_audio_path(path: str) -> tuple[str, int]:
    """`EDACC-C35_P1-53.wav` -> ('EDACC-C35_P1', 53)."""
    m = _PATH_RE.match(path)
    if not m:
        raise ValueError(f"unparsable audio path: {path!r}")
    return m.group("conv"), int(m.group("idx"))


# speaker examples: "EDACC-C06-A", "EDACC-C35-B".
_SPK_RE = re.compile(r"^(?P<conv>.+)-(?P<label>[AB])$")


def parse_speaker(speaker: str) -> tuple[str, str]:
    """`EDACC-C06-A` -> ('EDACC-C06', 'A'). Note: speaker conv doesn't include _Pn."""
    m = _SPK_RE.match(speaker)
    if not m:
        raise ValueError(f"unparsable speaker id: {speaker!r}")
    return m.group("conv"), m.group("label")


def canonicalise_conv(hf_conv: str) -> str:
    """Map HF `EDACC-*` -> DataShare `EAEC-*`."""
    return hf_conv.replace("EDACC-", "EAEC-", 1)


def extract_rows(parquet_path: Path) -> list[dict]:
    """Read text + speaker + audio.path columns only. Drop audio bytes."""
    table = pq.read_table(parquet_path, columns=["speaker", "text", "audio"])
    rows = []
    for spk, text, audio in zip(
        table.column("speaker").to_pylist(),
        table.column("text").to_pylist(),
        table.column("audio").to_pylist(),
    ):
        path = audio.get("path") if isinstance(audio, dict) else None
        if not path or not spk:
            continue
        try:
            audio_conv, idx = parse_audio_path(path)
            spk_conv, label = parse_speaker(spk)
        except ValueError as e:
            print(f"  skip row: {e}")
            continue
        rows.append({
            "audio_conv": canonicalise_conv(audio_conv),
            "speaker_conv": canonicalise_conv(spk_conv),
            "speaker_label": label,
            "utterance_idx": idx,
            "text": text or "",
        })
    return rows


def write_transcripts(rows: list[dict], dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    # Group by audio_conv (this includes part suffix like _P1)
    by_conv: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_conv[r["audio_conv"]].append(r)

    for conv, items in sorted(by_conv.items()):
        items.sort(key=lambda r: r["utterance_idx"])
        # Build speaker_id mapping (A/B -> e.g. C06-A / C06-B); same speakers
        # may appear across parts (_P1 / _P2) for the same base conversation.
        base_conv = re.sub(r"_P\d+$", "", conv)
        spk_ids = {r["speaker_label"]: f"{base_conv}-{r['speaker_label']}" for r in items}

        out = dst_dir / f"{conv}.txt"
        with out.open("w") as f:
            f.write(f"# conversation: {conv}\n")
            f.write(f"# base_conversation: {base_conv}\n")
            f.write(f"# speaker_A_id: {spk_ids.get('A', '<none>')}\n")
            f.write(f"# speaker_B_id: {spk_ids.get('B', '<none>')}\n")
            f.write(f"# n_utterances: {len(items)}\n")
            f.write("# columns: utterance_idx\tspeaker\ttext\n")
            for r in items:
                f.write(f"{r['utterance_idx']}\t{r['speaker_label']}\t{r['text']}\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--cache", default="/tmp/edacc_parquets",
                   help="Parquet download cache (will be filled with ~7 GB).")
    p.add_argument("--out", default="/home/user/datasets/EdAcc/transcripts",
                   help="Output directory for per-conversation transcripts.")
    p.add_argument("--keep-cache", action="store_true",
                   help="Don't delete the parquet cache after extraction.")
    args = p.parse_args()

    cache = Path(args.cache)
    out = Path(args.out)

    print("Listing HF parquet files...")
    remote = list_parquets()
    print(f"  {len(remote)} files")
    local = download_parquets(remote, cache)

    all_rows: list[dict] = []
    for lp in local:
        print(f"Extracting {lp.name}...")
        all_rows.extend(extract_rows(lp))
    print(f"Total rows: {len(all_rows)}")

    write_transcripts(all_rows, out)
    n_files = sum(1 for _ in out.glob("*.txt"))
    print(f"Wrote {n_files} transcript files to {out}")

    if not args.keep_cache:
        print(f"Cleaning parquet cache at {cache}")
        shutil.rmtree(cache, ignore_errors=True)


if __name__ == "__main__":
    main()
