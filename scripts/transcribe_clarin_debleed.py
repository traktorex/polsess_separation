"""Whisper-transcribe all enhanced CLARIN debleed channels.

Each debleed channel is one speaker's audio, but with low-level residue
of the *other* speaker that the debleed process couldn't fully remove.
A plain VAD trigger picks up that residue too — so the transcript ends
up with stray words from the wrong speaker. Whisper itself is also
prone to repetition-cascade hallucination on long silent stretches.

The fix mirrors what `asr/explore_pipeline.ipynb` does for the main
pipeline: run pyannote speaker diarization on the channel, identify the
**main speaker** as the one with the highest total energy on that
channel (the loud one — by construction the residue is the quiet one),
and zero out everything outside that speaker's turns. Whisper then sees
a stream that's exact zeros everywhere except where the channel's main
speaker is talking — which matches the audio shape pyannote-trained
Whisper handles cleanly (silences classified, speech transcribed).

We also pass `condition_on_previous_text=False` as a belt-and-suspenders
defence against any remaining hallucination — if a single window does
misfire, it won't cascade through the rest of the file.

Output: `<gotowy>/debleed_enhanced_transcripts/<stem>.txt`, one segment
per line in `[HH:MM:SS.cc → HH:MM:SS.cc] text` form.

Idempotent: outputs that already exist are skipped.

Usage:
    source venv/bin/activate
    python scripts/transcribe_clarin_debleed.py
    python scripts/transcribe_clarin_debleed.py --only 442dd69e_R
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import whisper

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from asr_pipeline.config import load_pipeline_config_from_yaml


GOTOWY = Path("/home/user/datasets/clarin_gotowy/gotowy")
ENH = GOTOWY / "debleed_enhanced"
OUT = GOTOWY / "debleed_enhanced_transcripts"
PIPELINE_CONFIG = Path(__file__).resolve().parent.parent / "asr_pipeline" / "configs" / "default.yaml"

WHISPER_MODEL = "large-v3"
WHISPER_LANG = "pl"
WHISPER_PROMPT = "Rozmowa po polsku."

# pyannote.audio is hit-or-miss when forced to find exactly 2 speakers in
# debleed channels where the residue is sometimes too quiet to detect.
# min=1/max=2 gives it the option to report a single speaker, in which case
# we keep everything — equivalent to "no residue detected".
DIAR_MIN_SPEAKERS = 1
DIAR_MAX_SPEAKERS = 2


def _fmt_ts(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"


def _load_diarizer(model_id: str, hf_token: str, device: torch.device):
    from pyannote.audio import Pipeline as PyannotePipeline
    pipeline = PyannotePipeline.from_pretrained(model_id, token=hf_token).to(device)
    return pipeline


def _main_speaker_mask(
    audio: np.ndarray, sr: int, diarizer
) -> tuple[np.ndarray, dict[str, float], str | None]:
    """Diarize the channel; build a mask keeping only the highest-energy speaker.

    Returns ``(mask, rms_by_speaker, main_speaker_label)``. If diarization
    finds nothing, `main_speaker_label` is None and the mask is all zeros.
    """
    waveform = torch.from_numpy(audio).unsqueeze(0)
    diar = diarizer(
        {"waveform": waveform, "sample_rate": sr},
        min_speakers=DIAR_MIN_SPEAKERS,
        max_speakers=DIAR_MAX_SPEAKERS,
    )
    # Pyannote 3.1 sometimes returns a `SpeakerDiarization` wrapper.
    diar = diar.speaker_diarization if hasattr(diar, "speaker_diarization") else diar

    spk_segments: dict[str, list[tuple[float, float]]] = {}
    for turn, _, spk in diar.itertracks(yield_label=True):
        spk_segments.setdefault(spk, []).append((turn.start, turn.end))

    if not spk_segments:
        return np.zeros(len(audio), dtype=np.float32), {}, None

    # Per-speaker RMS over the speaker's own segments.
    rms_by_spk: dict[str, float] = {}
    for spk, segs in spk_segments.items():
        sq_sum = 0.0
        n = 0
        for s, e in segs:
            i = max(0, int(s * sr))
            j = min(len(audio), int(e * sr))
            if j > i:
                seg = audio[i:j]
                sq_sum += float((seg ** 2).sum())
                n += (j - i)
        rms_by_spk[spk] = math.sqrt(sq_sum / max(n, 1))

    main_spk = max(rms_by_spk, key=rms_by_spk.get)
    mask = np.zeros(len(audio), dtype=np.float32)
    for s, e in spk_segments[main_spk]:
        i = max(0, int(s * sr))
        j = min(len(audio), int(e * sr))
        mask[i:j] = 1.0
    return mask, rms_by_spk, main_spk


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None,
                    help="Substring match against the stem (e.g. `442dd69e_R`).")
    args = ap.parse_args()

    if not ENH.is_dir():
        raise SystemExit(f"debleed_enhanced not found: {ENH}")
    OUT.mkdir(parents=True, exist_ok=True)

    inputs = sorted(p for p in ENH.iterdir() if p.suffix == ".wav")
    if args.only:
        inputs = [p for p in inputs if args.only in p.stem]
    print(f"[transcribe] {len(inputs)} wav files"
          + (f" (filtered by --only {args.only!r})" if args.only else ""))
    if not inputs:
        return

    # Borrow the diarization model id + HF token from the pipeline config so
    # we stay in sync if the user rotates the token.
    cfg = load_pipeline_config_from_yaml(str(PIPELINE_CONFIG))
    diar_model_id = cfg.diarization.model_id
    hf_token = cfg.diarization.hf_token
    if not hf_token:
        raise SystemExit("HF token empty in pipeline config; set $HF_TOKEN or edit default.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[transcribe] loading pyannote diarization ({diar_model_id}) on {device}")
    t0 = time.perf_counter()
    diarizer = _load_diarizer(diar_model_id, hf_token, device)
    print(f"[transcribe] diarizer ready in {time.perf_counter()-t0:.1f}s")

    print(f"[transcribe] loading whisper {WHISPER_MODEL} on {device}")
    t0 = time.perf_counter()
    asr_model = whisper.load_model(WHISPER_MODEL, device=str(device))
    print(f"[transcribe] whisper ready in {time.perf_counter()-t0:.1f}s")

    done = 0
    skipped = 0
    for i, wav in enumerate(inputs, 1):
        txt_out = OUT / f"{wav.stem}.txt"
        if txt_out.exists():
            skipped += 1
            print(f"[transcribe] ({i}/{len(inputs)}) {wav.name} — skip (exists)")
            continue

        audio, sr = sf.read(str(wav), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1).astype(np.float32)
        if sr != 16_000:
            raise RuntimeError(f"expected 16 kHz, got {sr} Hz: {wav}")

        t_d = time.perf_counter()
        mask, rms_by_spk, main_spk = _main_speaker_mask(audio, sr, diarizer)
        elapsed_d = time.perf_counter() - t_d

        if main_spk is None:
            print(f"[transcribe] ({i}/{len(inputs)}) {wav.name} — diarization empty, skipping Whisper")
            txt_out.write_text("", encoding="utf-8")
            continue

        speech_frac = float(mask.mean())
        rms_summary = ", ".join(
            f"{s}={rms_by_spk[s]:.4f}{'*' if s == main_spk else ''}"
            for s in sorted(rms_by_spk, key=lambda x: -rms_by_spk[x])
        )
        gated = (audio * mask).astype(np.float32)

        t_w = time.perf_counter()
        result = asr_model.transcribe(
            gated,
            language=WHISPER_LANG,
            initial_prompt=WHISPER_PROMPT,
            word_timestamps=True,
            temperature=0.0,
            condition_on_previous_text=False,
            verbose=False,
            fp16=(device.type == "cuda"),
        )
        elapsed_w = time.perf_counter() - t_w

        lines = []
        for seg in result["segments"]:
            text = seg["text"].strip()
            if not text:
                continue
            lines.append(f"[{_fmt_ts(seg['start'])} → {_fmt_ts(seg['end'])}] {text}")
        txt_out.write_text("\n".join(lines) + "\n", encoding="utf-8")

        n_segs = len([s for s in result["segments"] if s["text"].strip()])
        print(
            f"[transcribe] ({i}/{len(inputs)}) {wav.name} "
            f"— diar: {elapsed_d:.1f}s, main={main_spk} (RMS {rms_summary}), "
            f"kept {speech_frac*100:.1f}% of audio, "
            f"whisper: {n_segs} segs ({elapsed_w:.1f}s) → {txt_out.name}"
        )
        done += 1

    print()
    print(f"[transcribe] done: {done} transcribed, {skipped} skipped")


if __name__ == "__main__":
    main()
