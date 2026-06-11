"""Enhance all CLARIN debleed channels with MossFormerGAN_SE_16K.

Reads every `<id>_{L,R}.wav` from `<gotowy>/debleed/`, runs ClearerVoice
MossFormerGAN_SE_16K on it, and writes the enhanced result to
`<gotowy>/debleed_enhanced/` with the same filename. Mono, 16 kHz,
float32 throughout.

Idempotent: outputs that already exist are skipped, so reruns only fill
gaps.

Usage:
    source venv/bin/activate
    python scripts/enhance_clarin_debleed.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

# Reuse the pipeline's existing ClearerVoice backend wrapper — same chunking,
# same device handling, same resampling logic as the actual pipeline runs.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from asr_pipeline.stages.enhancement import _ClearVoiceBackend, _CLEARVOICE_BACKENDS


GOTOWY = Path("/home/user/datasets/clarin_gotowy/gotowy")
DEBLEED = GOTOWY / "debleed"
OUT = GOTOWY / "debleed_enhanced"
MODEL_KEY = "mossformer_gan_se_16k"


def _list_inputs(debleed: Path) -> list[Path]:
    """Return all real wav files in debleed/, skipping `.asd` auto-save data."""
    wavs = sorted(p for p in debleed.iterdir() if p.suffix == ".wav")
    return wavs


def main() -> None:
    if not DEBLEED.is_dir():
        raise SystemExit(f"debleed directory not found: {DEBLEED}")
    OUT.mkdir(parents=True, exist_ok=True)

    inputs = _list_inputs(DEBLEED)
    print(f"[enhance] found {len(inputs)} wav files in {DEBLEED}")
    if not inputs:
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name, native_sr = _CLEARVOICE_BACKENDS[MODEL_KEY]
    print(f"[enhance] loading {model_name} on {device} (native_sr={native_sr})")
    t0 = time.perf_counter()
    backend = _ClearVoiceBackend(model_name, native_sr)
    backend.load(device)
    print(f"[enhance] loaded in {time.perf_counter()-t0:.1f}s")

    total_audio_s = 0.0
    total_wall_s = 0.0
    skipped = 0
    done = 0

    for i, wav_in in enumerate(inputs, 1):
        wav_out = OUT / wav_in.name
        if wav_out.exists():
            skipped += 1
            print(f"[enhance] ({i}/{len(inputs)}) {wav_in.name} — skip (exists)")
            continue

        audio, sr = sf.read(wav_in, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            # CLARIN debleed channels should already be mono — guard anyway.
            audio = audio.mean(axis=1).astype(np.float32)
        if sr != native_sr:
            audio = librosa.resample(
                audio, orig_sr=sr, target_sr=native_sr, res_type="soxr_hq"
            ).astype(np.float32)
            sr = native_sr

        dur_s = len(audio) / sr
        t_enh = time.perf_counter()
        out = backend.enhance(audio, sr)
        elapsed = time.perf_counter() - t_enh
        rtf = dur_s / elapsed if elapsed > 0 else float("inf")
        total_audio_s += dur_s
        total_wall_s += elapsed

        sf.write(wav_out, out.astype(np.float32), sr, subtype="FLOAT")
        done += 1
        print(
            f"[enhance] ({i}/{len(inputs)}) {wav_in.name} "
            f"— {dur_s:.1f}s audio in {elapsed:.1f}s ({rtf:.1f}× RT)"
        )

    print()
    print(f"[enhance] done: {done} enhanced, {skipped} skipped")
    if done:
        rtf = total_audio_s / total_wall_s if total_wall_s > 0 else float("inf")
        print(
            f"[enhance] total {total_audio_s:.1f}s audio in "
            f"{total_wall_s:.1f}s wall ({rtf:.1f}× RT avg)"
        )


if __name__ == "__main__":
    main()
