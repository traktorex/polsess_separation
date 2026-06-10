"""Stage 5 — Whisper ASR per assembled per-speaker stream.

Two backends, same per-speaker output shape::

    {"text": str, "segments": [{"start": float, "end": float,
                                "text": str, "words": [...optional...]}],
     "language": str}

- ``whisper``: vanilla openai-whisper. Fast to set up, no wav2vec2 alignment.
- ``whisperx``: WhisperX = faster-whisper + wav2vec2 forced alignment.
  Word-level timestamps to ±50 ms. Supports arbitrary HF Whisper model ids.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from asr_pipeline.config import TranscriptionConfig
from asr_pipeline.context import PipelineContext
from asr_pipeline.debug_log import dlog
from asr_pipeline.stages.base import Stage
from asr_pipeline.transcript_format import format_transcript, to_jsonable


# Minimum stream length worth sending to Whisper. SR-relative so it tracks
# ctx.sample_rate rather than baking in 16 kHz (POC's lower bound).
_MIN_TRANSCRIBE_DURATION_S = 0.5
# Peak-amplitude floor below which a stream is treated as silent and skipped.
# The assembler emits all-zeros sentinels for no-event speakers (assembly.py:
# _concat_shortened/_concat_full_length); those clear the duration gate above,
# and Whisper hallucinates phantom Polish on pure silence — which would then be
# spilled and scored as insertions in the L3 WER table. Value mirrors the
# silence floor in eval/layer2.py (duplicated, not imported: stages must not
# depend on eval).
_SILENCE_FLOOR = 1e-4


def _log(msg: str) -> None:
    """Transcription debug log — visible in the notebook and durable on disk.

    Routed through `dlog` (not `print`) so messages survive the WSL stdout
    bridge dropping; matters for the multi-minute one-time CT2 conversion.
    """
    dlog("transcription", msg)


# ---------------------------------------------------------------------------
# Output formatting helpers (shared by both backends)
# ---------------------------------------------------------------------------


def _empty_result(language: str) -> dict:
    """The stage's empty-transcript contract — a fresh dict every call.

    Emitted for a speaker with no assembled audio, a stream shorter than the
    duration floor, or a silent stream. Returned fresh (never a shared literal)
    so the per-speaker / mixture branches can't alias one `segments` list.
    """
    return {"text": "", "segments": [], "language": language}


def _normalise_result(result: dict, language: str) -> dict:
    """Ensure both backends emit ``text`` and ``language`` at the top level."""
    out = dict(result)
    segments = out.get("segments") or []
    if "text" not in out:
        out["text"] = " ".join((s.get("text") or "").strip() for s in segments).strip()
    if "language" not in out:
        out["language"] = language
    return out


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class _WhisperBackend:
    """openai-whisper. Canonical OpenAI checkpoints only."""

    def __init__(self, cfg: TranscriptionConfig) -> None:
        self.cfg = cfg
        self._model = None
        self._device: Optional[torch.device] = None

    def load(self, device: torch.device) -> None:
        import whisper
        self._model = whisper.load_model(self.cfg.model_name, device=str(device))
        self._device = device

    def transcribe(self, audio: np.ndarray) -> dict:
        result = self._model.transcribe(
            audio.astype(np.float32),
            language=self.cfg.language,
            initial_prompt=self.cfg.initial_prompt,
            word_timestamps=self.cfg.word_timestamps,
            verbose=False,
        )
        return _normalise_result(result, self.cfg.language)

    def unload(self) -> None:
        self._model = None
        self._device = None


_CT2_CACHE_ROOT = Path.home() / "models" / "ct2-whisper"


def _ensure_ct2_model(model_name: str) -> str:
    """Resolve a Whisper identifier to a local CT2-converted path.

    - OpenAI short names (``large-v3``, ``large-v2``, …) pass through unchanged;
      faster-whisper handles them natively.
    - HuggingFace ids (anything containing ``/``) are converted on first use
      and cached under ``~/models/ct2-whisper/<safe-name>/``. Subsequent calls
      reuse the cache directory.
    - Existing local paths pass through unchanged.

    The conversion is two-step: first re-materialise the HF model with the
    *fast* tokenizer so a unified ``tokenizer.json`` exists, then run
    ``ct2-transformers-converter`` on the local dir. The intermediate dir
    is removed at the end. This handles older HF Whisper finetunes that
    ship only the split tokenizer files
    (``vocab.json`` + ``merges.txt`` + …) — without the re-materialise step,
    the converter fails on a missing ``tokenizer.json``.
    """
    if "/" not in model_name:
        return model_name
    if Path(model_name).exists():
        return model_name
    safe_name = model_name.replace("/", "-")
    cache_dir = _CT2_CACHE_ROOT / safe_name
    if (cache_dir / "model.bin").exists():
        return str(cache_dir)

    import shutil
    import subprocess
    import tempfile
    from transformers import (
        WhisperForConditionalGeneration,
        WhisperProcessor,
        WhisperTokenizerFast,
    )

    cache_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"hf-{safe_name}-") as tmp:
        _log(f"Re-materialising {model_name} with fast tokenizer "
             f"at {tmp} (one-time)...")
        model = WhisperForConditionalGeneration.from_pretrained(model_name)
        tok = WhisperTokenizerFast.from_pretrained(model_name)
        proc = WhisperProcessor.from_pretrained(model_name)
        model.save_pretrained(tmp)
        tok.save_pretrained(tmp)
        proc.save_pretrained(tmp)
        del model, tok, proc

        _log(f"Converting → CTranslate2 at {cache_dir}...")
        result = subprocess.run(
            ["ct2-transformers-converter",
             "--model", tmp,
             "--output_dir", str(cache_dir),
             "--copy_files", "tokenizer.json", "preprocessor_config.json",
             "--quantization", "float16",
             "--force"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            shutil.rmtree(cache_dir, ignore_errors=True)
            raise RuntimeError(
                f"ct2-transformers-converter failed for {model_name!r}:\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
    return str(cache_dir)


class _WhisperXBackend:
    """WhisperX = faster-whisper + wav2vec2 forced alignment.

    The alignment step is what makes word timestamps trustworthy. ``model_name``
    can be a canonical OpenAI short name (``large-v3``), an arbitrary HF
    Whisper model id, or a local CT2-converted dir. HF ids
    are auto-converted to CT2 format on first use via ``_ensure_ct2_model``
    and cached under ``~/models/ct2-whisper/``.
    """

    def __init__(self, cfg: TranscriptionConfig) -> None:
        self.cfg = cfg
        self._asr = None
        self._align_model = None
        self._align_metadata: Optional[dict] = None
        self._device_str: Optional[str] = None

    def load(self, device: torch.device) -> None:
        import whisperx
        device_str = "cuda" if device.type == "cuda" else "cpu"
        # bf16/fp16 unsafe on CPU; let WhisperX pick a sane default.
        compute_type = "float16" if device_str == "cuda" else "int8"
        model_path = _ensure_ct2_model(self.cfg.model_name)
        self._asr = whisperx.load_model(
            model_path,
            device=device_str,
            compute_type=compute_type,
            language=self.cfg.language,
            asr_options={"initial_prompt": self.cfg.initial_prompt},
        )
        # Always load the wav2vec2 align model when using WhisperX — it
        # also catches hallucinations (words that can't be aligned to actual
        # audio are filtered out) and re-segments the output to actual word
        # boundaries, so its value is more than just word-level timestamps.
        # If you actually want no alignment, pick `backend: whisper` instead.
        self._align_model, self._align_metadata = whisperx.load_align_model(
            language_code=self.cfg.language,
            device=device_str,
            model_name=self.cfg.align_model_name,
        )
        self._device_str = device_str

    def transcribe(self, audio: np.ndarray) -> dict:
        import whisperx
        audio = audio.astype(np.float32)
        result = self._asr.transcribe(audio, language=self.cfg.language)
        # `result` has segments with .text / .start / .end but no word-level
        # timing. Alignment adds word timestamps from wav2vec2.
        if self.cfg.word_timestamps and result.get("segments"):
            aligned = whisperx.align(
                result["segments"],
                self._align_model,
                self._align_metadata,
                audio,
                self._device_str,
                return_char_alignments=False,
            )
            result = {**result, **aligned}
        return _normalise_result(result, self.cfg.language)

    def unload(self) -> None:
        self._asr = None
        self._align_model = None
        self._align_metadata = None
        self._device_str = None


# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------


class TranscriptionStage(Stage):
    name = "transcription"

    def __init__(self, config: TranscriptionConfig) -> None:
        super().__init__(enabled=config.enabled)
        self.config = config
        self._backend: Optional[_WhisperBackend | _WhisperXBackend] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def load(self, device: torch.device) -> None:
        if self.config.backend == "whisper":
            self._backend = _WhisperBackend(self.config)
        elif self.config.backend == "whisperx":
            self._backend = _WhisperXBackend(self.config)
        else:
            raise ValueError(f"Unknown transcription backend: {self.config.backend!r}")
        self._backend.load(device)

    def load_signature(self) -> tuple:
        # Per-call options (language / initial_prompt / word_timestamps) are
        # not part of model identity. Backend, model id, and (for whisperx)
        # alignment model id are.
        if self.config.backend == "whisperx":
            return (self.config.backend, self.config.model_name,
                    self.config.align_model_name)
        return (self.config.backend, self.config.model_name)

    def unload(self) -> None:
        if self._backend is not None:
            self._backend.unload()
            self._backend = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def run(self, ctx: PipelineContext) -> None:
        if self._backend is None:
            raise RuntimeError("TranscriptionStage.run called before load().")

        min_samples = int(ctx.sample_rate * _MIN_TRANSCRIBE_DURATION_S)

        # Per-speaker assembled streams. Skip streams that are too short OR
        # silent — the latter is the assembler's no-event sentinel (all-zeros,
        # but longer than min_samples), which Whisper would otherwise turn into
        # hallucinated text scored against a speaker who said nothing.
        results: dict[str, dict] = {}
        for spk, audio in (ctx.assembled or {}).items():
            if self._skip_transcription(audio, min_samples):
                results[spk] = _empty_result(self.config.language)
                continue
            results[spk] = self._backend.transcribe(audio)
        ctx.transcripts = results

        # Mixture baseline (single-stream Whisper on the whole recording).
        # Used by the ablation table — same backend / prompt / args as the
        # per-speaker pass, so the comparison is fair.
        if self.config.transcribe_mixture and ctx.audio is not None:
            if self._skip_transcription(ctx.audio, min_samples):
                ctx.mixture_transcript = _empty_result(self.config.language)
            else:
                ctx.mixture_transcript = self._backend.transcribe(ctx.audio)

    @staticmethod
    def _skip_transcription(audio: np.ndarray, min_samples: int) -> bool:
        """True if `audio` is too short or silent to be worth transcribing."""
        if len(audio) < min_samples or len(audio) == 0:
            return True
        return float(np.max(np.abs(audio))) < _SILENCE_FLOOR

    # ------------------------------------------------------------------
    # Spill
    # ------------------------------------------------------------------
    def spill(self, ctx: PipelineContext, artifact_dir: Path) -> None:
        if not ctx.transcripts:
            return
        for spk, result in ctx.transcripts.items():
            label = ctx.spk_to_label.get(spk, spk)
            txt_path = artifact_dir / f"transcript_{label}.txt"
            json_path = artifact_dir / f"transcript_{label}.json"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(format_transcript(result))
                f.write("\n")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(to_jsonable(result), f, indent=2, ensure_ascii=False)
