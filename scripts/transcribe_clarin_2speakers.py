"""WhisperX-transcribe every CLARIN 2-speaker recording, raw and enhanced.

Drives two parallel transcript sets over
`/home/user/datasets/clarin_all_2speakers/clarin_download/*.wav`:

    auto_transcription_raw/                 — WhisperX on the raw download
    auto_transcription_enhanced_mossformer/ — WhisperX on MossFormerGAN_SE_16K-enhanced audio

The enhanced wavs are also kept on disk under
``clarin_all_2speakers/enhanced_mossformer/`` so the second transcription
pass can run after the enhancement model has been dropped from GPU.

Transcription settings mirror `asr_pipeline/configs/default.yaml`:
backend = whisperx, model = large-v2, language = pl, initial prompt
"Rozmowa po polsku.", align model
``jonatasgrosman/wav2vec2-large-xlsr-53-polish``, word timestamps on.
That keeps these transcripts directly comparable to what the production
ASR pipeline writes.

Phase-major to keep peak GPU memory low: enhancement model loaded for
the enhance pass, dropped, then WhisperX loaded once and reused for both
transcription passes.

Per-recording outputs (in each target dir):
  <id>.txt   — `[start → end]  text` lines (same format as the pipeline)
  <id>.json  — full WhisperX result (segments + word timestamps)

Idempotent: a file is skipped if both its .txt and .json already exist.

Usage:
    source venv/bin/activate
    HF_TOKEN=... python scripts/transcribe_clarin_2speakers.py
    # Raw-mixture transcripts only (skip enhancement + enhanced pass):
    HF_TOKEN=... python scripts/transcribe_clarin_2speakers.py --phases tx-raw
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from asr_pipeline.config import TranscriptionConfig                      # noqa: E402
from asr_pipeline.stages.enhancement import (                            # noqa: E402
    _CLEARVOICE_BACKENDS,
    _ClearVoiceBackend,
)
from asr_pipeline.stages.transcription import _WhisperXBackend           # noqa: E402
from asr_pipeline.transcript_format import format_transcript, to_jsonable  # noqa: E402


ROOT = Path("/home/user/datasets/clarin_all_2speakers")
RAW_DIR = ROOT / "clarin_download"
ENH_DIR = ROOT / "enhanced_mossformer"
TX_RAW = ROOT / "auto_transcription_raw"
TX_ENH = ROOT / "auto_transcription_enhanced_mossformer"

SAMPLE_RATE = 16_000
ENH_BACKEND_KEY = "mossformer_gan_se_16k"


def _list_wavs(d: Path) -> list[Path]:
    return sorted(p for p in d.iterdir() if p.suffix == ".wav")


def _load_mono_16k(path: Path) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1).astype(np.float32)
    if sr != SAMPLE_RATE:
        audio = librosa.resample(
            audio, orig_sr=sr, target_sr=SAMPLE_RATE, res_type="soxr_hq"
        ).astype(np.float32)
    return audio


def _outputs_exist(out_dir: Path, stem: str) -> bool:
    return (out_dir / f"{stem}.txt").exists() and (out_dir / f"{stem}.json").exists()


def _write_transcript(out_dir: Path, stem: str, result: dict) -> None:
    txt_path = out_dir / f"{stem}.txt"
    json_path = out_dir / f"{stem}.json"
    tmp_txt = txt_path.with_suffix(".txt.tmp")
    tmp_json = json_path.with_suffix(".json.tmp")
    tmp_txt.write_text(format_transcript(result) + "\n", encoding="utf-8")
    with open(tmp_json, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(result), f, indent=2, ensure_ascii=False)
    tmp_txt.replace(txt_path)
    tmp_json.replace(json_path)


# ---------------------------------------------------------------------------
# Phase 1: enhance every raw wav → enhanced_mossformer/<id>.wav
# ---------------------------------------------------------------------------


def enhance_all(device: torch.device, inputs: list[Path]) -> None:
    ENH_DIR.mkdir(parents=True, exist_ok=True)
    model_name, native_sr = _CLEARVOICE_BACKENDS[ENH_BACKEND_KEY]
    print(f"[enhance] loading {model_name} on {device} (native_sr={native_sr})")
    t0 = time.perf_counter()
    backend = _ClearVoiceBackend(model_name, native_sr)
    backend.load(device)
    print(f"[enhance] loaded in {time.perf_counter()-t0:.1f}s")

    total_audio_s = 0.0
    total_wall_s = 0.0
    done = 0
    skipped = 0
    failed: list[tuple[str, str]] = []

    for i, wav_in in enumerate(inputs, 1):
        wav_out = ENH_DIR / wav_in.name
        if wav_out.exists():
            skipped += 1
            print(f"[enhance] ({i}/{len(inputs)}) {wav_in.name} — skip (exists)")
            continue
        try:
            audio = _load_mono_16k(wav_in)
            dur_s = len(audio) / SAMPLE_RATE
            t_enh = time.perf_counter()
            # Backend resamples internally if SAMPLE_RATE != native_sr;
            # mossformer_gan_se_16k is 16 kHz native, so no resample.
            out = backend.enhance(audio, SAMPLE_RATE)
            elapsed = time.perf_counter() - t_enh
            sf.write(wav_out, out.astype(np.float32), SAMPLE_RATE, subtype="FLOAT")
            done += 1
            total_audio_s += dur_s
            total_wall_s += elapsed
            rtf = dur_s / elapsed if elapsed > 0 else float("inf")
            print(
                f"[enhance] ({i}/{len(inputs)}) {wav_in.name} "
                f"— {dur_s:.1f}s in {elapsed:.1f}s ({rtf:.1f}× RT)"
            )
        except Exception as e:
            failed.append((wav_in.name, str(e)))
            print(f"[enhance] ({i}/{len(inputs)}) {wav_in.name} — FAILED: {e}")

    print()
    print(f"[enhance] done: {done} enhanced, {skipped} skipped, {len(failed)} failed")
    if done:
        rtf = total_audio_s / total_wall_s if total_wall_s > 0 else float("inf")
        print(f"[enhance] total {total_audio_s:.1f}s in {total_wall_s:.1f}s ({rtf:.1f}× RT avg)")
    if failed:
        print("[enhance] failures:")
        for name, msg in failed:
            print(f"  - {name}: {msg}")

    # Drop the enhancement model before we load WhisperX.
    backend.unload()
    del backend
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Phase 2: WhisperX on raw + enhanced
# ---------------------------------------------------------------------------


def _make_whisperx_cfg() -> TranscriptionConfig:
    # Defaults match asr_pipeline/configs/default.yaml.
    return TranscriptionConfig(
        enabled=True,
        backend="whisperx",
        model_name="large-v2",
        language="pl",
        initial_prompt="Rozmowa po polsku.",
        word_timestamps=True,
        align_model_name="jonatasgrosman/wav2vec2-large-xlsr-53-polish",
    )


def _transcribe_dir(
    backend: _WhisperXBackend,
    src_dir: Path,
    out_dir: Path,
    inputs: list[Path],
    tag: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    total_audio_s = 0.0
    total_wall_s = 0.0
    done = 0
    skipped = 0
    failed: list[tuple[str, str]] = []

    for i, ref_wav in enumerate(inputs, 1):
        stem = ref_wav.stem
        wav = src_dir / ref_wav.name
        if not wav.exists():
            failed.append((wav.name, "input wav missing"))
            print(f"[{tag}] ({i}/{len(inputs)}) {wav.name} — MISSING in {src_dir}")
            continue
        if _outputs_exist(out_dir, stem):
            skipped += 1
            print(f"[{tag}] ({i}/{len(inputs)}) {wav.name} — skip (exists)")
            continue
        try:
            audio = _load_mono_16k(wav)
            dur_s = len(audio) / SAMPLE_RATE
            t0 = time.perf_counter()
            result = backend.transcribe(audio)
            elapsed = time.perf_counter() - t0
            _write_transcript(out_dir, stem, result)
            done += 1
            total_audio_s += dur_s
            total_wall_s += elapsed
            rtf = dur_s / elapsed if elapsed > 0 else float("inf")
            n_segs = len(result.get("segments") or [])
            print(
                f"[{tag}] ({i}/{len(inputs)}) {wav.name} "
                f"— {dur_s:.1f}s in {elapsed:.1f}s ({rtf:.1f}× RT), {n_segs} segs"
            )
        except Exception as e:
            failed.append((wav.name, str(e)))
            print(f"[{tag}] ({i}/{len(inputs)}) {wav.name} — FAILED: {e}")

    print()
    print(f"[{tag}] done: {done} transcribed, {skipped} skipped, {len(failed)} failed")
    if done:
        rtf = total_audio_s / total_wall_s if total_wall_s > 0 else float("inf")
        print(f"[{tag}] total {total_audio_s:.1f}s in {total_wall_s:.1f}s ({rtf:.1f}× RT avg)")
    if failed:
        print(f"[{tag}] failures:")
        for name, msg in failed:
            print(f"  - {name}: {msg}")


def transcribe_all(
    device: torch.device,
    inputs: list[Path],
    do_raw: bool,
    do_enh: bool,
) -> None:
    if not (do_raw or do_enh):
        return
    cfg = _make_whisperx_cfg()
    print(f"[whisperx] loading {cfg.backend}/{cfg.model_name} (+ {cfg.align_model_name}) on {device}")
    t0 = time.perf_counter()
    backend = _WhisperXBackend(cfg)
    backend.load(device)
    print(f"[whisperx] loaded in {time.perf_counter()-t0:.1f}s")

    try:
        if do_raw:
            print("\n=== WhisperX on raw audio → auto_transcription_raw/ ===")
            _transcribe_dir(backend, RAW_DIR, TX_RAW, inputs, tag="tx-raw")
        if do_enh:
            print("\n=== WhisperX on enhanced audio → auto_transcription_enhanced_mossformer/ ===")
            _transcribe_dir(backend, ENH_DIR, TX_ENH, inputs, tag="tx-enh")
    finally:
        backend.unload()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


PHASE_CHOICES = ("enhance", "tx-raw", "tx-enh")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--phases", nargs="+", default=list(PHASE_CHOICES), choices=PHASE_CHOICES,
        help="Phases to run (default: all). Order is fixed regardless of order given.",
    )
    args = ap.parse_args()
    phases = set(args.phases)

    if not RAW_DIR.is_dir():
        print(f"error: raw dir not found: {RAW_DIR}", file=sys.stderr)
        return 1

    inputs = _list_wavs(RAW_DIR)
    print(f"[main] {len(inputs)} raw wavs in {RAW_DIR}")
    print(f"[main] phases: {sorted(phases)}")
    if not inputs:
        return 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if "enhance" in phases:
        print("\n=== Phase 1: enhancement (MossFormerGAN_SE_16K) ===")
        enhance_all(device, inputs)
    elif "tx-enh" in phases and not ENH_DIR.is_dir():
        print(
            f"error: phase 'tx-enh' requested but {ENH_DIR} does not exist — "
            "run the 'enhance' phase first.",
            file=sys.stderr,
        )
        return 1

    if "tx-raw" in phases or "tx-enh" in phases:
        print("\n=== Phase 2: WhisperX large-v2 ===")
        transcribe_all(device, inputs, "tx-raw" in phases, "tx-enh" in phases)

    print("\n[main] all done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
