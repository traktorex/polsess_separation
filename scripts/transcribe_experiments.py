"""Compare transcription pipelines on 442dd69e_R against the human GT.

Round 2 — dimensions beyond volume processing (the volume sweep from
round 1 showed baseline already wins; quiet-utterance failures aren't
fixed by amplification). This round varies:

  1. baseline           — Whisper, diarization-gated (full_length style)
  2. shortened_mode     — same but concat the main-speaker segments with
                          a 0.3 s silence gap (the assembly stage's
                          `shortened` output_mode equivalent)
  3. temperature_fb     — Whisper's default temperature ramp + fallback
                          (vs the round-1 temperature=0.0 only)
  4. no_prompt          — drop the `Rozmowa po polsku.` initial_prompt
  5. expanded_mask      — pad each pyannote turn ±0.3 s in the mask, to
                          catch words pyannote clipped at the edges
  6. faster_whisper     — same gated audio, faster-whisper backend
                          (loaded after unloading OpenAI whisper)
  7. raw_debleed        — feed *raw* (non-enhanced) debleed audio through
                          the same gating; tests whether MossFormer-GAN
                          enhancement is net-positive for ASR

Output: per-method transcript in `<gotowy>/experiments/<method>/442dd69e_R.txt`
plus a WER table sorted ascending. Same audio + cached diarization mask
across all methods so results are directly comparable.

Caveat: N=1 GT — winner is the local optimum for this recording.
"""
from __future__ import annotations

import gc
import math
import sys
import time
from pathlib import Path

import jiwer
import numpy as np
import soundfile as sf
import torch
import whisper

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from asr_pipeline.config import load_pipeline_config_from_yaml
from asr_pipeline.eval.metrics import _normalize_text
from asr_pipeline.eval.transcript_parser import parse_gt_txt


GOTOWY = Path("/home/user/datasets/clarin_gotowy/gotowy")
GT_PATH = GOTOWY / "true_transcripts" / "442dd69e_R.txt"
ENH_WAV = GOTOWY / "debleed_enhanced" / "442dd69e_R.wav"
RAW_WAV = GOTOWY / "debleed" / "442dd69e_R.wav"
EXP_ROOT = GOTOWY / "experiments"
PIPELINE_CONFIG = Path(__file__).resolve().parent.parent / "asr_pipeline" / "configs" / "default.yaml"

WHISPER_MODEL = "large-v3"
WHISPER_LANG = "pl"
WHISPER_PROMPT = "Rozmowa po polsku."
DIAR_MIN_SPEAKERS = 1
DIAR_MAX_SPEAKERS = 2
SHORTENED_GAP_S = 0.3      # matches asr_pipeline default `silence_separator_s`
EXPAND_MASK_S = 0.3        # padding around each pyannote turn for #5
WHISPER_TEMP_RAMP = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)  # OpenAI whisper default


# ---------------------------------------------------------------------------
# Helpers (shared with the prior script, slimmed down)
# ---------------------------------------------------------------------------

def _fmt_ts(s: float) -> str:
    if s < 0: s = 0.0
    h = int(s // 3600); m = int((s % 3600) // 60); rest = s % 60
    return f"{h:02d}:{m:02d}:{rest:05.2f}"


def _load_audio(path: Path, expected_sr: int = 16_000) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1).astype(np.float32)
    if sr != expected_sr:
        raise SystemExit(f"expected {expected_sr} Hz, got {sr}: {path}")
    return audio


def _diarize_main_speaker(audio, sr, diarizer):
    waveform = torch.from_numpy(audio).unsqueeze(0)
    diar = diarizer(
        {"waveform": waveform, "sample_rate": sr},
        min_speakers=DIAR_MIN_SPEAKERS, max_speakers=DIAR_MAX_SPEAKERS,
    )
    diar = diar.speaker_diarization if hasattr(diar, "speaker_diarization") else diar
    spk_segs: dict[str, list[tuple[float, float]]] = {}
    for turn, _, spk in diar.itertracks(yield_label=True):
        spk_segs.setdefault(spk, []).append((turn.start, turn.end))
    if not spk_segs:
        return np.zeros(len(audio), dtype=np.float32), []
    rms_by_spk = {}
    for spk, segs in spk_segs.items():
        sq, n = 0.0, 0
        for s, e in segs:
            i, j = max(0, int(s * sr)), min(len(audio), int(e * sr))
            sq += float((audio[i:j] ** 2).sum())
            n += j - i
        rms_by_spk[spk] = math.sqrt(sq / max(n, 1))
    main_spk = max(rms_by_spk, key=rms_by_spk.get)
    print(f"  main speaker: {main_spk}  (RMS by speaker: {rms_by_spk})")
    segs = spk_segs[main_spk]
    mask = np.zeros(len(audio), dtype=np.float32)
    for s, e in segs:
        i, j = max(0, int(s * sr)), min(len(audio), int(e * sr))
        mask[i:j] = 1.0
    return mask, segs


def _expand_mask(segs, mask_len, sr, pad_s):
    out = np.zeros(mask_len, dtype=np.float32)
    expanded = []
    for s, e in segs:
        s = max(0.0, s - pad_s)
        e = e + pad_s
        i, j = max(0, int(s * sr)), min(mask_len, int(e * sr))
        out[i:j] = 1.0
        expanded.append((i / sr, j / sr))
    return out, expanded


def _shortened_concat(audio, sr, segs, gap_s):
    """Concat each segment back-to-back with `gap_s` silence between.

    Returns (concat_audio, remap) where remap is a list of
    (concat_offset_s, orig_start_s, orig_end_s) entries — one per
    contributed segment — for mapping ASR-emitted timestamps back to
    the original timeline.
    """
    gap_samples = int(gap_s * sr)
    pieces = []
    remap = []
    cursor = 0.0
    for s, e in segs:
        i, j = max(0, int(s * sr)), min(len(audio), int(e * sr))
        if j <= i:
            continue
        seg = audio[i:j].astype(np.float32)
        remap.append((cursor, s, e))
        pieces.append(seg)
        cursor += (j - i) / sr
        pieces.append(np.zeros(gap_samples, dtype=np.float32))
        cursor += gap_s
    if not pieces:
        return np.zeros(0, dtype=np.float32), []
    return np.concatenate(pieces).astype(np.float32), remap


def _remap_segments(asr_segs, remap):
    """Map ASR (concat-time) segments back to original-time."""
    if not remap:
        return asr_segs
    out = []
    for as_s, as_e, txt in asr_segs:
        # Find which chunk contains as_s; remap is in order of cursor.
        chunk = None
        for k, (concat_start, orig_s, orig_e) in enumerate(remap):
            next_concat = remap[k + 1][0] if k + 1 < len(remap) else float("inf")
            if concat_start <= as_s < next_concat:
                chunk = (concat_start, orig_s, orig_e)
                break
        if chunk is None:
            chunk = remap[-1]
        concat_start, orig_s, orig_e = chunk
        offset = max(0.0, min(as_s - concat_start, orig_e - orig_s))
        new_start = orig_s + offset
        new_end = new_start + (as_e - as_s)
        out.append((new_start, new_end, txt))
    return out


def asr_openai_whisper(model, audio, sr, **overrides):
    kwargs = dict(
        language=WHISPER_LANG,
        initial_prompt=WHISPER_PROMPT,
        word_timestamps=True,
        temperature=0.0,
        condition_on_previous_text=False,
        verbose=False,
        fp16=torch.cuda.is_available(),
    )
    kwargs.update(overrides)
    res = model.transcribe(audio, **kwargs)
    return [(float(s["start"]), float(s["end"]), s["text"].strip())
            for s in res["segments"] if s["text"].strip()]


def asr_faster_whisper(model, audio, sr):
    seg_iter, _ = model.transcribe(
        audio,
        language=WHISPER_LANG,
        initial_prompt=WHISPER_PROMPT,
        word_timestamps=True,
        temperature=0.0,
        condition_on_previous_text=False,
    )
    return [(float(s.start), float(s.end), s.text.strip())
            for s in seg_iter if s.text.strip()]


def save_transcript(segs, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"[{_fmt_ts(s)} → {_fmt_ts(e)}] {t}" for s, e, t in segs]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def wer_against(gt_path, hyp_path):
    gt = _normalize_text(" ".join(u.text for u in parse_gt_txt(gt_path)))
    hyp = _normalize_text(" ".join(u.text for u in parse_gt_txt(hyp_path)))
    out = jiwer.process_words(gt, hyp)
    return {
        "wer": float(out.wer), "ref_words": len(gt.split()), "hyp_words": len(hyp.split()),
        "sub": int(out.substitutions), "ins": int(out.insertions), "del": int(out.deletions),
    }


# ---------------------------------------------------------------------------
# Per-method runners. Each returns (segs, n_input_seconds).
# ---------------------------------------------------------------------------

def m_baseline(audio, sr, mask, segs, whisper_m):
    x = (audio * mask).astype(np.float32)
    return asr_openai_whisper(whisper_m, x, sr), len(x) / sr


def m_shortened_mode(audio, sr, mask, segs, whisper_m):
    x, remap = _shortened_concat(audio, sr, segs, SHORTENED_GAP_S)
    asr_segs = asr_openai_whisper(whisper_m, x, sr)
    return _remap_segments(asr_segs, remap), len(x) / sr


def m_temperature_fb(audio, sr, mask, segs, whisper_m):
    x = (audio * mask).astype(np.float32)
    return asr_openai_whisper(whisper_m, x, sr, temperature=WHISPER_TEMP_RAMP), len(x) / sr


def m_no_prompt(audio, sr, mask, segs, whisper_m):
    x = (audio * mask).astype(np.float32)
    return asr_openai_whisper(whisper_m, x, sr, initial_prompt=None), len(x) / sr


def m_expanded_mask(audio, sr, mask, segs, whisper_m):
    expanded_mask, _ = _expand_mask(segs, len(audio), sr, EXPAND_MASK_S)
    x = (audio * expanded_mask).astype(np.float32)
    return asr_openai_whisper(whisper_m, x, sr), len(x) / sr


def m_raw_debleed(audio_unused, sr, mask_unused, segs_unused, whisper_m, *, diarizer):
    """Re-diarize on raw debleed and run baseline-style transcription."""
    raw = _load_audio(RAW_WAV, expected_sr=sr)
    mask_raw, segs_raw = _diarize_main_speaker(raw, sr, diarizer)
    x = (raw * mask_raw).astype(np.float32)
    return asr_openai_whisper(whisper_m, x, sr), len(x) / sr


def m_faster_whisper(audio, sr, mask, segs, faster_m):
    x = (audio * mask).astype(np.float32)
    return asr_faster_whisper(faster_m, x, sr), len(x) / sr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not GT_PATH.exists():
        raise SystemExit(f"GT not found: {GT_PATH}")
    if not ENH_WAV.exists():
        raise SystemExit(f"Enhanced wav not found: {ENH_WAV}")

    cfg = load_pipeline_config_from_yaml(str(PIPELINE_CONFIG))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[setup] device={device}")
    audio = _load_audio(ENH_WAV)
    sr = 16_000
    print(f"[setup] audio: {len(audio)/sr:.1f}s ({ENH_WAV.name})")

    print(f"[setup] loading pyannote ({cfg.diarization.model_id})")
    t0 = time.perf_counter()
    from pyannote.audio import Pipeline as PyannotePipeline
    diarizer = PyannotePipeline.from_pretrained(
        cfg.diarization.model_id, token=cfg.diarization.hf_token
    ).to(device)
    print(f"[setup] diarizer ready ({time.perf_counter()-t0:.1f}s)")

    print(f"[setup] running diarization on enhanced (cached)")
    t0 = time.perf_counter()
    mask, segs = _diarize_main_speaker(audio, sr, diarizer)
    print(f"[setup] done ({time.perf_counter()-t0:.1f}s, kept {mask.mean()*100:.1f}%, {len(segs)} turns)")

    print(f"[setup] loading OpenAI whisper {WHISPER_MODEL}")
    t0 = time.perf_counter()
    whisper_m = whisper.load_model(WHISPER_MODEL, device=str(device))
    print(f"[setup] whisper ready ({time.perf_counter()-t0:.1f}s)")

    results = []

    def run(label, fn, asr_kind="whisper", **kw):
        print(f"\n[exp] === {label} ===")
        t0 = time.perf_counter()
        try:
            asr_model = whisper_m if asr_kind == "whisper" else kw.pop("faster_m")
            output = fn(audio, sr, mask, segs, asr_model, **kw)
            run_segs, input_dur = output
        except Exception as exc:
            import traceback; traceback.print_exc()
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            return
        elapsed = time.perf_counter() - t0
        out_path = EXP_ROOT / label / "442dd69e_R.txt"
        save_transcript(run_segs, out_path)
        wer = wer_against(GT_PATH, out_path)
        print(f"  input audio: {input_dur:.1f}s; {len(run_segs)} ASR segs; {elapsed:.1f}s wall")
        print(f"  WER {wer['wer']*100:.2f}%  ({wer['sub']} sub / {wer['ins']} ins / {wer['del']} del, "
              f"of {wer['ref_words']} ref words)")
        results.append({"label": label, **wer, "time_s": elapsed})

    # --- OpenAI whisper experiments first (so we can free whisper before loading faster-whisper)
    run("baseline",        m_baseline)
    run("shortened_mode",  m_shortened_mode)
    run("temperature_fb",  m_temperature_fb)
    run("no_prompt",       m_no_prompt)
    run("expanded_mask",   m_expanded_mask)
    run("raw_debleed",     m_raw_debleed, diarizer=diarizer)

    # --- Free OpenAI whisper before loading faster-whisper.
    print("\n[setup] unloading OpenAI whisper before loading faster-whisper")
    whisper_m = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        from faster_whisper import WhisperModel
        print(f"[setup] loading faster-whisper {WHISPER_MODEL}")
        t0 = time.perf_counter()
        faster_m = WhisperModel(
            WHISPER_MODEL,
            device=str(device).split(":")[0],
            compute_type="float16",
        )
        print(f"[setup] faster-whisper ready ({time.perf_counter()-t0:.1f}s)")
        run("faster_whisper", m_faster_whisper, asr_kind="faster", faster_m=faster_m)
    except Exception as exc:
        print(f"[setup] faster-whisper skipped: {type(exc).__name__}: {exc}")

    # --- Print sorted summary
    print()
    print("=" * 84)
    print(f"{'method':<22}  {'WER':>8}  {'sub':>5}  {'ins':>5}  {'del':>5}  {'hyp/ref':>10}  {'wall(s)':>9}")
    print("-" * 84)
    for r in sorted(results, key=lambda x: x["wer"]):
        print(f"{r['label']:<22}  {r['wer']*100:7.2f}%  {r['sub']:5d}  {r['ins']:5d}  {r['del']:5d}  "
              f"{r['hyp_words']:5d}/{r['ref_words']:<4d}  {r['time_s']:9.1f}")


if __name__ == "__main__":
    main()
