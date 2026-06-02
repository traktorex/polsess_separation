"""Run pyannote diarization on every CLARIN 2-speaker recording.

Reads every `<id>.wav` from
`/home/user/datasets/clarin_all_2speakers/clarin_download/`, runs the
exact same diarization stage the production ASR pipeline uses
(`pyannote/speaker-diarization-3.1`, `num_speakers=2`, mono 16 kHz),
and writes one `<id>.json` per recording to
`/home/user/datasets/clarin_all_2speakers/diarization/`.

The output schema matches `asr_pipeline.stages.diarization.DiarizationStage.spill()`:

    {
      "total_duration_s": float,
      "segments":  [{start, end, duration, speaker}, ...],
      "overlaps":  [{start, end, duration}, ...]
    }

Idempotent: outputs that already exist are skipped, so reruns only fill
gaps. Phase-major in spirit — one model loaded once, every recording
processed in a single pass.

Usage:
    source venv/bin/activate
    HF_TOKEN=... python scripts/diarize_clarin_2speakers.py
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from asr_pipeline.config import DiarizationConfig                # noqa: E402
from asr_pipeline.context import PipelineContext                 # noqa: E402
from asr_pipeline.io import load_audio_as_mono                   # noqa: E402
from asr_pipeline.stages.diarization import DiarizationStage     # noqa: E402


INPUT_DIR = Path("/home/user/datasets/clarin_all_2speakers/clarin_download")
OUTPUT_DIR = Path("/home/user/datasets/clarin_all_2speakers/diarization")
SAMPLE_RATE = 16_000


def _list_inputs(input_dir: Path) -> list[Path]:
    """Return all real wav files, skipping Zone.Identifier and other junk."""
    return sorted(p for p in input_dir.iterdir() if p.suffix == ".wav")


def _write_json(out_path: Path, ctx: PipelineContext) -> None:
    diar = ctx.diarization
    payload = {
        "total_duration_s": diar.total_duration_s,
        "segments": diar.segments_df.to_dict(orient="records"),
        "overlaps": diar.overlaps_df.to_dict(orient="records"),
    }
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    tmp.replace(out_path)


def main() -> int:
    if not INPUT_DIR.is_dir():
        print(f"error: input dir not found: {INPUT_DIR}", file=sys.stderr)
        return 1
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    inputs = _list_inputs(INPUT_DIR)
    print(f"[diarize] found {len(inputs)} wav files in {INPUT_DIR}")
    if not inputs:
        return 0

    cfg = DiarizationConfig()  # defaults: pyannote 3.1, num_speakers=2, HF_TOKEN env
    print(f"[diarize] config: {asdict(cfg) | {'hf_token': '***' if cfg.hf_token else None}}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[diarize] loading {cfg.model_id} on {device}")
    t0 = time.perf_counter()
    stage = DiarizationStage(cfg)
    stage.load(device)
    print(f"[diarize] loaded in {time.perf_counter()-t0:.1f}s")

    total_audio_s = 0.0
    total_wall_s = 0.0
    skipped = 0
    done = 0
    failed: list[tuple[str, str]] = []

    for i, wav_in in enumerate(inputs, 1):
        out_path = OUTPUT_DIR / f"{wav_in.stem}.json"
        if out_path.exists():
            skipped += 1
            print(f"[diarize] ({i}/{len(inputs)}) {wav_in.name} — skip (exists)")
            continue

        try:
            audio = load_audio_as_mono(str(wav_in), target_sr=SAMPLE_RATE)
        except Exception as e:
            failed.append((wav_in.name, f"load: {e}"))
            print(f"[diarize] ({i}/{len(inputs)}) {wav_in.name} — LOAD FAILED: {e}")
            continue

        dur_s = len(audio) / SAMPLE_RATE
        ctx = PipelineContext(audio=audio, sample_rate=SAMPLE_RATE)

        t_diar = time.perf_counter()
        try:
            stage.run(ctx)
        except Exception as e:
            failed.append((wav_in.name, f"diarize: {e}"))
            print(f"[diarize] ({i}/{len(inputs)}) {wav_in.name} — DIARIZE FAILED: {e}")
            continue
        elapsed = time.perf_counter() - t_diar
        rtf = dur_s / elapsed if elapsed > 0 else float("inf")
        total_audio_s += dur_s
        total_wall_s += elapsed

        _write_json(out_path, ctx)
        done += 1

        n_seg = len(ctx.diarization.segments_df)
        n_ovl = len(ctx.diarization.overlaps_df)
        print(
            f"[diarize] ({i}/{len(inputs)}) {wav_in.name} "
            f"— {dur_s:.1f}s audio in {elapsed:.1f}s ({rtf:.1f}× RT), "
            f"{n_seg} segs, {n_ovl} overlaps"
        )

    print()
    print(f"[diarize] done: {done} diarized, {skipped} skipped, {len(failed)} failed")
    if done:
        avg_rtf = total_audio_s / total_wall_s if total_wall_s > 0 else float("inf")
        print(
            f"[diarize] total {total_audio_s:.1f}s audio in "
            f"{total_wall_s:.1f}s wall ({avg_rtf:.1f}× RT avg)"
        )
    if failed:
        print("[diarize] failures:")
        for name, msg in failed:
            print(f"  - {name}: {msg}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
