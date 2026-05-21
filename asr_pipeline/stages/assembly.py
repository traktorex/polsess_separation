"""Stage 4 — per-speaker stream assembly via ECAPA anchors + TimestampMap.

For each speaker (from pyannote's labels):

1. Build an ECAPA-TDNN *anchor* embedding from the speaker's concatenated
   enhanced solo audio. If the speaker has < `min_solo_for_anchor_s` of
   solo audio, mark the recording as weak-anchor and fall back to a fixed
   straight-through assignment for overlaps.
2. For each overlap, ECAPA-embed each of the two separator outputs (the
   full padded gated stream) and pick the speaker pairing
   (s1->A,s2->B vs s1->B,s2->A) with the higher summed cosine similarity
   against the anchors.
3. Slice each overlap's gated stream to its `emit_start`/`emit_end` region
   (from Phase 4), so we only emit the assembler-relevant slice — not the
   entire padded separator output.
4. Concatenate per speaker in `output_mode`:
     - "shortened"   -> speech-only concat with `silence_separator_s` gaps
     - "full_length" -> stream length = input length; pieces placed at
                        their original timestamps; gaps filled with silence
5. Build a `TimestampMap` that records the (concat_*) -> (orig_*) mapping
   for every piece in every speaker's stream.

Speaker-assignment behaviour is hard-coded to the POC's ECAPA-per-overlap
approach. If a second variant is ever introduced (e.g. global PIT, or
trusting pyannote's own embeddings), refactor to a strategy interface.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch

from asr_pipeline.config import AssemblyConfig
from asr_pipeline.context import (
    PipelineContext,
    TimestampMap,
    TimestampMapEntry,
)
from asr_pipeline.stages.base import Stage


# ---------------------------------------------------------------------------
# ECAPA embedding helpers
# ---------------------------------------------------------------------------


@torch.no_grad()
def _ecapa_embed(
    audio_16k: np.ndarray, ecapa, device: torch.device, sample_rate: int
) -> torch.Tensor:
    """Return a unit-norm ECAPA embedding for the audio (1-D float32)."""
    min_len = sample_rate // 4   # 0.25 s — POC's lower bound
    if len(audio_16k) < min_len:
        audio_16k = np.pad(audio_16k, (0, min_len - len(audio_16k)))
    audio = torch.from_numpy(audio_16k).unsqueeze(0).to(device)
    emb = ecapa.encode_batch(audio).squeeze(0).squeeze(0)
    emb = emb / (emb.norm() + 1e-8)
    return emb


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a * b).sum())


# ---------------------------------------------------------------------------
# Emit-region slicing
# ---------------------------------------------------------------------------


def _slice_emit(
    gated_audio: np.ndarray,
    pad_start_s: float,
    emit_start_s: float,
    emit_end_s: float,
    sample_rate: int,
) -> np.ndarray:
    """Slice the gated padded separator output to the assembler emit region."""
    offset = int((emit_start_s - pad_start_s) * sample_rate)
    length = int((emit_end_s - emit_start_s) * sample_rate)
    if offset < 0:
        offset = 0
    if offset + length > len(gated_audio):
        length = len(gated_audio) - offset
    if length <= 0:
        return np.zeros(0, dtype=np.float32)
    return gated_audio[offset : offset + length].astype(np.float32)


# ---------------------------------------------------------------------------
# Optional per-piece RMS normalisation
# ---------------------------------------------------------------------------


def _rms(arr: np.ndarray) -> float:
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(arr.astype(np.float64) ** 2)))


def _rms_normalise(
    pieces: list[np.ndarray], target_rms: Optional[float]
) -> list[np.ndarray]:
    """Rescale each piece so its RMS matches `target_rms`.

    When `target_rms` is None, the median RMS across non-empty pieces is
    used as the target. Empty pieces are returned untouched.
    """
    rms_values = [_rms(p) for p in pieces if p.size > 0]
    if not rms_values:
        return pieces
    if target_rms is None:
        target_rms = float(np.median(rms_values))
    out: list[np.ndarray] = []
    for p in pieces:
        r = _rms(p)
        if r < 1e-9 or p.size == 0:
            out.append(p)
        else:
            out.append(p * (target_rms / r))
    return out


# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------


class AssemblyStage(Stage):
    name = "assembly"

    def __init__(self, config: AssemblyConfig) -> None:
        super().__init__(enabled=config.enabled)
        self.config = config
        self._ecapa = None
        self._device: Optional[torch.device] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def load(self, device: torch.device) -> None:
        from speechbrain.inference.speaker import EncoderClassifier

        cache_dir = Path.cwd() / ".cache" / "ecapa"
        cache_dir.mkdir(parents=True, exist_ok=True)
        ecapa = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": str(device)},
            savedir=str(cache_dir),
        )
        ecapa.eval()
        self._ecapa = ecapa
        self._device = device

    def unload(self) -> None:
        self._ecapa = None
        self._device = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def run(self, ctx: PipelineContext) -> None:
        if self._ecapa is None or self._device is None:
            raise RuntimeError("AssemblyStage.run called before load().")
        if not ctx.speakers:
            # Nothing to assemble — keep ctx fields at their defaults.
            ctx.timestamp_map = TimestampMap(weak_anchor=False, per_speaker={})
            return

        cfg = self.config
        sr = ctx.sample_rate
        speakers = ctx.speakers
        spk_to_label = {spk: chr(ord("A") + i) for i, spk in enumerate(speakers)}

        # -- Anchors per speaker ---------------------------------------
        solo_by_spk: dict[str, list[dict]] = {spk: [] for spk in speakers}
        for seg in sorted(ctx.solo_enhanced.values(), key=lambda x: x["start"]):
            if seg["speaker"] in solo_by_spk:
                solo_by_spk[seg["speaker"]].append(seg)

        anchors: dict[str, Optional[torch.Tensor]] = {}
        solo_durations: dict[str, float] = {}
        for spk in speakers:
            if solo_by_spk[spk]:
                concat = np.concatenate([s["enhanced"] for s in solo_by_spk[spk]])
            else:
                concat = np.zeros(16, dtype=np.float32)
            solo_durations[spk] = len(concat) / sr
            if len(concat) >= sr // 4:
                anchors[spk] = _ecapa_embed(
                    concat, self._ecapa, self._device, sr
                ).cpu()
            else:
                anchors[spk] = None

        weak_anchor = any(d < cfg.min_solo_for_anchor_s for d in solo_durations.values())

        # -- Per-overlap speaker assignment ----------------------------
        assignments: list[dict] = []
        for ovl in ctx.overlap_separated:
            if len(ovl["s1_gated"]) < sr // 10:
                continue
            emb1 = _ecapa_embed(ovl["s1_gated"], self._ecapa, self._device, sr).cpu()
            emb2 = _ecapa_embed(ovl["s2_gated"], self._ecapa, self._device, sr).cpu()
            if len(speakers) >= 2 and all(anchors.get(s) is not None for s in speakers[:2]):
                a, b = speakers[0], speakers[1]
                straight = _cos(emb1, anchors[a]) + _cos(emb2, anchors[b])
                swapped = _cos(emb1, anchors[b]) + _cos(emb2, anchors[a])
                if straight >= swapped:
                    stream_for = {a: ovl["s1_gated"], b: ovl["s2_gated"]}
                    pairing = "straight"
                else:
                    stream_for = {a: ovl["s2_gated"], b: ovl["s1_gated"]}
                    pairing = "swapped"
            else:
                # Weak-anchor fallback: fixed assignment.
                stream_for = {}
                if len(speakers) >= 1:
                    stream_for[speakers[0]] = ovl["s1_gated"]
                if len(speakers) >= 2:
                    stream_for[speakers[1]] = ovl["s2_gated"]
                pairing = "arbitrary (weak anchor)"

            # Slice each speaker's stream down to the emit region.
            emit_pieces = {
                spk: _slice_emit(
                    audio,
                    ovl["pad_start"],
                    ovl["emit_start"],
                    ovl["emit_end"],
                    sr,
                )
                for spk, audio in stream_for.items()
            }
            assignments.append({
                "orig_start": float(ovl["emit_start"]),
                "orig_end": float(ovl["emit_end"]),
                "pairing": pairing,
                "emit_pieces": emit_pieces,
            })

        # -- Build per-speaker events list (sorted by original time) ----
        events: dict[str, list[dict]] = {spk: [] for spk in speakers}
        for spk in speakers:
            for seg in solo_by_spk[spk]:
                events[spk].append({
                    "orig_start": float(seg["start"]),
                    "orig_end": float(seg["end"]),
                    "audio": seg["enhanced"].astype(np.float32),
                    "kind": "solo",
                })
        for a in assignments:
            for spk, audio in a["emit_pieces"].items():
                if audio.size == 0:
                    continue
                events[spk].append({
                    "orig_start": a["orig_start"],
                    "orig_end": a["orig_end"],
                    "audio": audio,
                    "kind": "overlap",
                })
        for spk in speakers:
            events[spk].sort(key=lambda e: e["orig_start"])

        # -- Optional per-piece RMS normalisation -----------------------
        if cfg.per_piece_rms_norm:
            for spk in speakers:
                rescaled = _rms_normalise(
                    [e["audio"] for e in events[spk]], cfg.target_rms
                )
                for e, r in zip(events[spk], rescaled):
                    e["audio"] = r.astype(np.float32)

        # -- Concatenate (mode-dependent) -------------------------------
        assembled: dict[str, np.ndarray] = {}
        timestamp_map = TimestampMap(weak_anchor=weak_anchor, per_speaker={})
        if cfg.output_mode == "shortened":
            gap_samples = int(cfg.silence_separator_s * sr)
            for spk in speakers:
                pieces: list[np.ndarray] = []
                tmap: list[TimestampMapEntry] = []
                cursor = 0.0
                for ev in events[spk]:
                    if pieces:
                        pieces.append(np.zeros(gap_samples, dtype=np.float32))
                        cursor += gap_samples / sr
                    pieces.append(ev["audio"])
                    dur = len(ev["audio"]) / sr
                    tmap.append(TimestampMapEntry(
                        concat_start=cursor,
                        concat_end=cursor + dur,
                        orig_start=ev["orig_start"],
                        orig_end=ev["orig_end"],
                        kind=ev["kind"],
                    ))
                    cursor += dur
                assembled[spk] = (
                    np.concatenate(pieces).astype(np.float32)
                    if pieces
                    else np.zeros(sr, dtype=np.float32)
                )
                timestamp_map.per_speaker[spk] = tmap

        elif cfg.output_mode == "full_length":
            total_dur = ctx.diarization.total_duration_s if ctx.diarization else 0.0
            total_samples = int(total_dur * sr)
            for spk in speakers:
                stream = np.zeros(total_samples, dtype=np.float32)
                tmap: list[TimestampMapEntry] = []
                for ev in events[spk]:
                    start_idx = int(ev["orig_start"] * sr)
                    if start_idx >= total_samples:
                        continue
                    end_idx = min(total_samples, start_idx + len(ev["audio"]))
                    audio = ev["audio"][: end_idx - start_idx]
                    stream[start_idx:end_idx] = audio
                    tmap.append(TimestampMapEntry(
                        concat_start=ev["orig_start"],
                        concat_end=ev["orig_start"] + len(audio) / sr,
                        orig_start=ev["orig_start"],
                        orig_end=ev["orig_end"],
                        kind=ev["kind"],
                    ))
                assembled[spk] = stream
                timestamp_map.per_speaker[spk] = tmap
        else:
            # Should be unreachable: PipelineConfig.__post_init__ already
            # validates the enum-string.
            raise ValueError(f"Unknown output_mode: {cfg.output_mode!r}")

        ctx.assembled = assembled
        ctx.spk_to_label = spk_to_label
        ctx.timestamp_map = timestamp_map
        ctx.weak_anchor = weak_anchor

    # ------------------------------------------------------------------
    # Spill
    # ------------------------------------------------------------------
    def spill(self, ctx: PipelineContext, artifact_dir: Path) -> None:
        if not ctx.assembled:
            return
        for spk, audio in ctx.assembled.items():
            label = ctx.spk_to_label.get(spk, spk)
            sf.write(
                artifact_dir / f"assembled_{label}.wav",
                audio.astype(np.float32),
                ctx.sample_rate,
            )
        if ctx.timestamp_map is not None:
            payload = {
                "weak_anchor": ctx.timestamp_map.weak_anchor,
                "spk_to_label": ctx.spk_to_label,
                "output_mode": self.config.output_mode,
                "per_speaker": {
                    spk: [
                        {
                            "concat_start": e.concat_start,
                            "concat_end": e.concat_end,
                            "orig_start": e.orig_start,
                            "orig_end": e.orig_end,
                            "kind": e.kind,
                        }
                        for e in entries
                    ]
                    for spk, entries in ctx.timestamp_map.per_speaker.items()
                },
            }
            with open(artifact_dir / "timestamp_map.json", "w") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
