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
import math
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


def _temperature_schedule(temperature) -> list:
    """Normalise the config temperature into the list form Whisper expects.

    Both backends ultimately want a sequence: openai-whisper's ``transcribe``
    accepts a scalar or tuple, while WhisperX's ``TranscriptionOptions`` field
    is ``temperatures`` (plural, always a list). A bare float means "no
    fallback" → a single-element schedule, matching openai-whisper /
    faster-whisper fallback semantics.
    """
    if isinstance(temperature, (list, tuple)):
        return list(temperature)
    return [temperature]


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
    """Ensure both backends emit ``text`` and ``language`` at the top level.

    Also sanitises per-segment ``start`` / ``end`` timestamps: WhisperX's
    ``align()`` runs ``interpolate_nans`` over them, and an all-unalignable
    segment ffill/bfills to all-NaN that survives to output. A NaN (or ``None``)
    timestamp later crashes the EAF writer and emits literal ``nan`` into the
    ``.txt`` — silent data loss for a recording that transcribed fine. Coerce
    any non-finite/missing boundary to ``0.0`` here, at the single boundary both
    backends pass through, so every downstream writer receives clean timestamps.
    """
    out = dict(result)
    segments = out.get("segments") or []
    out["segments"] = [_sanitise_segment_times(s) for s in segments]
    if "text" not in out:
        out["text"] = " ".join((s.get("text") or "").strip() for s in segments).strip()
    if "language" not in out:
        out["language"] = language
    return out


def _finite_or_zero(value) -> float:
    """Coerce ``None`` / NaN / inf to ``0.0`` and clamp negatives to ``0.0``.

    ``or 0.0`` does not work here: ``float('nan')`` is truthy, so it would pass
    a NaN straight through.

    The non-negative clamp is the time-sanitisation contract's single owner.
    WhisperX's ``align()`` routinely emits small negative starts; written as
    ``[ -0.30 → ...]`` they are silently dropped by the eval reader's
    non-negative-only seconds grammar (``eval/transcript_parser`` deliberately
    does **not** accept a leading ``-``), so the whole utterance vanishes from
    the hypothesis. Clamp here, at the one boundary both backends pass through.
    """
    if value is None:
        return 0.0
    f = float(value)
    if not math.isfinite(f):
        return 0.0
    return max(0.0, f)


def _sanitise_segment_times(seg: dict) -> dict:
    """Return a copy of ``seg`` with finite ``start`` / ``end`` boundaries."""
    out = dict(seg)
    out["start"] = _finite_or_zero(seg.get("start", 0.0))
    out["end"] = _finite_or_zero(seg.get("end", 0.0))
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
        self._retry_ignored_logged = False

    def load(self, device: torch.device) -> None:
        import whisper
        self._model = whisper.load_model(self.cfg.model_name, device=str(device))
        self._device = device

    def transcribe(self, audio: np.ndarray) -> dict:
        # Decode knobs route into openai-whisper's transcribe(): temperature /
        # no_speech_threshold / compression_ratio_threshold /
        # condition_on_previous_text are named params; beam_size / patience
        # fall through transcribe()'s **decode_options into DecodingOptions.
        #
        # The three anti-hallucination knobs are faster-whisper-only. openai-
        # whisper has no equivalent for no_repeat_ngram_size / repetition_penalty
        # (they'd crash DecodingOptions), and its hallucination_silence_threshold
        # is a DIFFERENT algorithm — forwarding it would silently substitute one
        # behaviour for another (SCOPE §4.1). So they are never passed here; a
        # non-default value with backend="whisper" is a loud configuration error,
        # not a quiet downgrade. Defaults (0 / 1.0 / None) are a no-op.
        self._reject_unsupported_knobs()
        self._warn_retry_ignored()
        result = self._model.transcribe(
            audio.astype(np.float32),
            language=self.cfg.language,
            initial_prompt=self.cfg.initial_prompt,
            word_timestamps=self.cfg.word_timestamps,
            beam_size=self.cfg.beam_size,
            patience=self.cfg.patience,
            temperature=tuple(_temperature_schedule(self.cfg.temperature)),
            condition_on_previous_text=self.cfg.condition_on_previous_text,
            no_speech_threshold=self.cfg.no_speech_threshold,
            compression_ratio_threshold=self.cfg.compression_ratio_threshold,
            verbose=False,
        )
        return _normalise_result(result, self.cfg.language)

    def _reject_unsupported_knobs(self) -> None:
        """Fail loud if a faster-whisper-only anti-hallucination knob is set
        while running the openai-whisper backend (SCOPE §4.1: no silent
        substitution). Defaults (0 / 1.0 / None) pass silently — they're a
        no-op and never reach openai-whisper."""
        unsupported = []
        if self.cfg.no_repeat_ngram_size != 0:
            unsupported.append(
                f"no_repeat_ngram_size={self.cfg.no_repeat_ngram_size}"
            )
        if self.cfg.repetition_penalty != 1.0:
            unsupported.append(
                f"repetition_penalty={self.cfg.repetition_penalty}"
            )
        if self.cfg.hallucination_silence_threshold is not None:
            unsupported.append(
                "hallucination_silence_threshold="
                f"{self.cfg.hallucination_silence_threshold}"
            )
        # chunk_size is a WhisperX VAD-pipeline knob; openai-whisper does its own
        # internal 30 s windowing and has no equivalent, so a non-default value
        # would silently no-op (SCOPE §4.1). 30 = WhisperX default = no-op here.
        if self.cfg.chunk_size != 30:
            unsupported.append(f"chunk_size={self.cfg.chunk_size}")
        if unsupported:
            raise ValueError(
                "transcription.backend='whisper' (openai-whisper) does not "
                "support the WhisperX-only knobs "
                f"{', '.join(unsupported)} — they only take effect with "
                "backend='whisperx'. Either switch to backend='whisperx' or "
                "leave these at their defaults (no_repeat_ngram_size=0, "
                "repetition_penalty=1.0, hallucination_silence_threshold=None, "
                "chunk_size=30)."
            )

    def _warn_retry_ignored(self) -> None:
        """Visibly note (once) that the collapsed-window detect-and-retry is
        WhisperX-only and is ignored on the openai-whisper backend.

        Unlike the hard-rejected WhisperX-only knobs above, retry defaults to ON
        (retry_collapsed_chunk_size=8), so a hard error would break this backend
        out of the box. openai-whisper does its own internal windowing and does
        not exhibit the WhisperX over-merge collapse, so retry simply does not
        apply — but per SCOPE §4.1 the no-op must be visible, not silent."""
        if self.cfg.retry_collapsed_chunk_size != 0 and not self._retry_ignored_logged:
            _log(
                "retry_collapsed_chunk_size="
                f"{self.cfg.retry_collapsed_chunk_size} is a WhisperX-only knob "
                "(collapsed-window detect-and-retry); the 'whisper' "
                "(openai-whisper) backend has its own internal windowing and "
                "does not collapse this way, so it is ignored here. Switch to "
                "backend='whisperx' to enable it, or set "
                "retry_collapsed_chunk_size=0 to silence this note."
            )
            self._retry_ignored_logged = True

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
        # Decode knobs go into WhisperX's asr_options, which it merges over its
        # own default_asr_options before building a faster-whisper
        # TranscriptionOptions. NB the schedule key there is `temperatures`
        # (plural list), not `temperature`. These defaults reproduce WhisperX's
        # own defaults exactly (see config.TranscriptionConfig docstrings).
        asr_options = {
            "initial_prompt": self.cfg.initial_prompt,
            "beam_size": self.cfg.beam_size,
            "patience": self.cfg.patience,
            "temperatures": _temperature_schedule(self.cfg.temperature),
            "condition_on_previous_text": self.cfg.condition_on_previous_text,
            "no_speech_threshold": self.cfg.no_speech_threshold,
            "compression_ratio_threshold": self.cfg.compression_ratio_threshold,
            # Anti-hallucination knobs. WhisperX carries each in its own
            # default_asr_options at the faster-whisper signature default
            # (no_repeat_ngram_size=0, repetition_penalty=1, hallucination_
            # silence_threshold=None), so passing the config defaults is a
            # no-op — the merge below changes nothing for a baseline run.
            "no_repeat_ngram_size": self.cfg.no_repeat_ngram_size,
            "repetition_penalty": self.cfg.repetition_penalty,
            "hallucination_silence_threshold": self.cfg.hallucination_silence_threshold,
        }
        self._asr = whisperx.load_model(
            model_path,
            device=device_str,
            compute_type=compute_type,
            language=self.cfg.language,
            asr_options=asr_options,
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
        # With align_model_name=None the config echo shows "None", which reads
        # like "no aligner" — surface what WhisperX actually auto-picked
        # (mirrors load_align_model's TORCH-then-HF resolution order).
        resolved = self.cfg.align_model_name
        if resolved is None:
            from whisperx.alignment import (
                DEFAULT_ALIGN_MODELS_HF,
                DEFAULT_ALIGN_MODELS_TORCH,
            )
            resolved = DEFAULT_ALIGN_MODELS_TORCH.get(
                self.cfg.language
            ) or DEFAULT_ALIGN_MODELS_HF.get(self.cfg.language, "<unknown>")
        _log(
            f"align model: {resolved} "
            f"(language={self.cfg.language}, auto-selected="
            f"{self.cfg.align_model_name is None})"
        )
        self._device_str = device_str

    # whisperx's fixed internal audio rate (it resamples to this); the pipeline
    # is already 16 kHz, so the retry-window slicing below indexes at this rate.
    _SR = 16_000

    def transcribe(self, audio: np.ndarray) -> dict:
        import whisperx
        audio = audio.astype(np.float32)
        # chunk_size bounds the max merged VAD segment length. WhisperX's default
        # is 30 s (= Whisper's window); a long unbroken segment at that ceiling
        # can make Whisper collapse and drop ~all of it (see TranscriptionConfig).
        result = self._asr.transcribe(
            audio, language=self.cfg.language, chunk_size=self.cfg.chunk_size
        )
        # Detect-and-retry collapsed windows on the RAW (pre-alignment) segments,
        # before wav2vec2 alignment re-segments them (see TranscriptionConfig).
        if self.cfg.retry_collapsed_chunk_size and result.get("segments"):
            result = {**result,
                      "segments": self._retry_collapsed(audio, result["segments"])}
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

    def _retry_collapsed(self, audio: np.ndarray, segments: list) -> list:
        """Re-transcribe collapsed merged windows at a smaller chunk and splice.

        A *collapse* is a long merged VAD window (``dur >= collapse_min_duration_s``)
        whose word density is below ``collapse_max_wps`` — Whisper emitted ~nothing
        for ~30 s of speech (see ``TranscriptionConfig``). For each one we re-run
        the SAME backend on just that audio span at ``retry_collapsed_chunk_size``,
        offset the recovered segments back onto the original timeline, and splice
        them in — but only when the retry recovers MORE words than the collapsed
        original (the guard), so a window can never end up emptier. Survivors and
        non-collapsed windows pass through untouched; the spliced list is re-sorted
        by start time. Returns a new list (never mutates the input segments).
        """
        cs = self.cfg.retry_collapsed_chunk_size
        out: list = []
        for seg in segments:
            dur = seg["end"] - seg["start"]
            nw = len(seg["text"].split())
            collapsed = (
                dur >= self.cfg.collapse_min_duration_s
                and nw / max(dur, 1e-9) < self.cfg.collapse_max_wps
            )
            if not collapsed:
                out.append(seg)
                continue
            s0, e0 = seg["start"], seg["end"]
            sub = audio[int(s0 * self._SR):int(e0 * self._SR)]
            retry = self._asr.transcribe(sub, language=self.cfg.language, chunk_size=cs)
            rsegs = retry.get("segments") or []
            retry_words = sum(len(rs["text"].split()) for rs in rsegs)
            # Guard: keep the original unless the retry recovered more words.
            if not rsegs or retry_words <= nw:
                out.append(seg)
                continue
            _log(
                f"retry: collapsed window [{s0:.1f}-{e0:.1f}] "
                f"({nw}w, {nw / max(dur, 1e-9):.2f} w/s) re-transcribed at "
                f"chunk_size={cs} → {retry_words}w"
            )
            for rs in rsegs:
                out.append({**rs, "start": rs["start"] + s0, "end": rs["end"] + s0})
        out.sort(key=lambda s: s["start"])
        return out

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
                _log(
                    f"run: {spk} stream short/silent "
                    f"({len(audio) / ctx.sample_rate:.2f}s) — empty transcript, "
                    f"Whisper not called"
                )
                results[spk] = _empty_result(self.config.language)
                continue
            results[spk] = self._backend.transcribe(audio)
        ctx.transcripts = results

        # Mixture baseline (single-stream Whisper on the whole recording).
        # Used by the ablation table — same backend / prompt / args as the
        # per-speaker pass, so the comparison is fair.
        if self.config.transcribe_mixture and ctx.audio is not None:
            if self._skip_transcription(ctx.audio, min_samples):
                _log(
                    "run: mixture short/silent — empty transcript, "
                    "Whisper not called"
                )
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
