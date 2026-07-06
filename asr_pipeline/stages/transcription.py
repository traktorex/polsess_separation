"""Stage 5 — ASR per assembled per-speaker stream.

Two backends, same per-speaker output shape::

    {"text": str, "segments": [{"start": float, "end": float,
                                "text": str, "words": [...optional...]}],
     "language": str}

- ``whisperx``: WhisperX = faster-whisper + wav2vec2 forced alignment.
  Word-level timestamps to ±50 ms. Supports arbitrary HF Whisper model ids.
  The default/shipped backend.
- ``coherex``: Cohere ASR (Diffio-AI/CohereX) via an isolated-venv subprocess.
"""

from __future__ import annotations

import gc
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from asr_pipeline.config import TranscriptionConfig
from asr_pipeline.context import PipelineContext
from asr_pipeline.debug_log import dlog
from asr_pipeline.stages.base import Stage
from asr_pipeline.text_metrics import (
    _WORD_RE,
    find_phrase_runs,
    repetition_loop_score,
)
from asr_pipeline.transcript_format import format_transcript, to_jsonable


# Minimum stream length worth sending to Whisper. SR-relative so it tracks
# ctx.sample_rate rather than baking in 16 kHz (POC's lower bound).
_MIN_TRANSCRIBE_DURATION_S = 0.5
# Phrase-loop join gap (seconds): two consecutive segments are joined into one
# token stream only if their inter-segment gap is at or below this. Hallucination
# loops are temporally contiguous; a genuine phrase legitimately re-said across a
# longer pause must not join into one run (a unique sentinel breaks the join).
_PHRASE_JOIN_MAX_GAP_S = 2.0
# The peak-amplitude floor below which a stream is treated as silent and skipped
# is `TranscriptionConfig.silence_floor` (default 1e-4). The assembler emits
# all-zeros sentinels for no-event speakers (assembly.py:
# _concat_shortened/_concat_full_length); those clear the duration gate above,
# and Whisper hallucinates phantom Polish on pure silence — which would then be
# spilled and scored as insertions in the L3 WER table. The default value
# mirrors the silence floor in eval/layer2.py (duplicated, not imported: stages
# must not depend on eval).


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
            "length_penalty": self.cfg.length_penalty,
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
            # suppress_numerals is popped by load_model before TranscriptionOptions
            # is built (it's not a faster-whisper field) and handed to the
            # pipeline — so it belongs in asr_options. False = WhisperX default.
            "suppress_numerals": self.cfg.suppress_numerals,
        }
        # WhisperX internal-VAD onset/offset. load_model merges this over its
        # own default_vad_options; the config defaults equal those defaults
        # (0.500 / 0.363), so a baseline run is byte-identical.
        vad_options = {
            "vad_onset": self.cfg.vad_onset,
            "vad_offset": self.cfg.vad_offset,
        }
        self._asr = whisperx.load_model(
            model_path,
            device=device_str,
            compute_type=compute_type,
            language=self.cfg.language,
            asr_options=asr_options,
            vad_options=vad_options,
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
        # Conditional repetition-loop retry — the OVER-production mirror of the
        # collapse retry above. Also on the RAW segments, before alignment.
        if self.cfg.loop_retry and result.get("segments"):
            result = {**result,
                      "segments": self._retry_loops(audio, result["segments"])}
        # Multi-token PHRASE-loop retry — the mirror of loop_retry that the
        # dominant-token detector cannot see. Same RAW, pre-alignment placement.
        if self.cfg.loop_retry_phrase and result.get("segments"):
            result = {**result,
                      "segments": self._retry_phrase_loops(audio, result["segments"])}
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

    def _retry_loops(self, audio: np.ndarray, segments: list) -> list:
        """Re-transcribe repetition-loop windows with no_repeat_ngram_size, splice.

        A *repetition loop* is a merged window whose text is one token repeated
        (`No tak, tak, tak, ...`) — a hallucination the collapse detector cannot
        catch (it is OVER-production; collapse is under-production). Detected with
        the shared ``repetition_loop_score`` metric (the offline scanner
        ``docs/sweep_plan/scan_repetition_loops.py`` uses the same function, so a
        spliced window is never re-flagged). For each window scoring
        ``>= loop_score_threshold`` we re-run the SAME backend on just that span,
        temporarily overriding faster-whisper's ``TranscriptionOptions.no_repeat_
        ngram_size`` to ``loop_retry_ngram`` (the global ``no_repeat_ngram_size``
        knob is untouched) via ``dataclasses.replace`` — the exact route WhisperX
        itself uses for suppress_numerals — restored in a ``finally``.

        Accept guard (the INVERTED mirror of the collapse guard: for a loop the
        goal is FEWER repeats, not more words): splice the retry in ONLY if it is
        non-empty AND every retry segment scores below the threshold AND it does
        not emit more tokens than the loop it replaces; otherwise keep the
        original window and log the rejection. A retry is never allowed to empty
        a window that had content — nor to GROW it: under the ngram constraint
        the decoder can evade both loop detectors by counting ("D1. D2. ...
        D78."), which only the length bound catches. Returns a new list (never
        mutates the input segments); the spliced list is re-sorted by start time.
        """
        thr = self.cfg.loop_score_threshold
        ngram = self.cfg.loop_retry_ngram
        out: list = []
        for seg in segments:
            score = repetition_loop_score(seg["text"]).score
            if score < thr:
                out.append(seg)
                continue
            s0, e0 = seg["start"], seg["end"]
            sub = audio[int(s0 * self._SR):int(e0 * self._SR)]
            saved_options = self._asr.options
            self._asr.options = replace(saved_options, no_repeat_ngram_size=ngram)
            try:
                retry = self._asr.transcribe(
                    sub, language=self.cfg.language, chunk_size=self.cfg.chunk_size
                )
            finally:
                # Restore the original options no matter what — the ngram override
                # must never leak into the next window / stream.
                self._asr.options = saved_options
            rsegs = retry.get("segments") or []
            retry_text = " ".join((rs.get("text") or "") for rs in rsegs).strip()
            retry_score = max(
                (repetition_loop_score(rs.get("text")).score for rs in rsegs),
                default=0.0,
            )
            # Guard: keep the original unless the retry broke the loop AND left
            # content behind AND did not GROW the window. `retry_score >= thr` =
            # a segment still loops; `not retry_text` = the retry emptied the
            # window; the length bound is the inverted collapse guard — a loop
            # is OVER-production, so its repair must never produce more tokens
            # than the loop it replaces. Real trigger: under the ngram
            # constraint the decoder can evade both loop detectors by counting
            # ("D1. D2. ... D78.", 152ed870 — every repeat textually distinct),
            # which only the length bound catches.
            orig_tokens = len(_WORD_RE.findall(seg["text"].lower()))
            retry_tokens = len(_WORD_RE.findall(retry_text.lower()))
            if not retry_text or retry_score >= thr or retry_tokens > orig_tokens:
                reason = ("grew the window "
                          f"({orig_tokens} -> {retry_tokens} tokens)"
                          if retry_text and retry_score < thr else
                          f"score {score:.2f} -> {retry_score:.2f}")
                _log(
                    f"loop-retry REJECTED window [{s0:.1f}-{e0:.1f}] "
                    f"{reason} (no_repeat_ngram_size={ngram}); keeping original"
                )
                out.append(seg)
                continue
            _log(
                f"loop-retry window [{s0:.1f}-{e0:.1f}] score {score:.2f} -> "
                f"{retry_score:.2f} (no_repeat_ngram_size={ngram}); accepted, "
                f"{len(rsegs)} segment(s)"
            )
            for rs in rsegs:
                out.append({**rs, "start": rs["start"] + s0, "end": rs["end"] + s0})
        out.sort(key=lambda s: s["start"])
        return out

    def _join_segment_tokens(self, segments: list) -> tuple[list, list]:
        """Flatten segments into one token stream + a parallel owner list.

        ``tokens[i]`` is a lowercased word token (``text_metrics._WORD_RE``);
        ``owner[i]`` is the index of the segment it came from. Between two
        consecutive segments whose inter-segment gap exceeds
        ``_PHRASE_JOIN_MAX_GAP_S`` a UNIQUE sentinel token (owner ``-1``) is
        inserted so no phrase run can span the pause (uniqueness alone breaks the
        match). Sentinels can never fall INSIDE a detected run — a run's blocks
        are exact repeats and a unique token repeats nowhere.
        """
        tokens: list = []
        owner: list = []
        for k, seg in enumerate(segments):
            for tok in _WORD_RE.findall((seg.get("text") or "").lower()):
                tokens.append(tok)
                owner.append(k)
            if k + 1 < len(segments):
                gap = segments[k + 1]["start"] - segments[k]["end"]
                if gap > _PHRASE_JOIN_MAX_GAP_S:
                    tokens.append(f"\x00{k}")
                    owner.append(-1)
        return tokens, owner

    def _retry_phrase_loops(self, audio: np.ndarray, segments: list) -> list:
        """Re-transcribe multi-token phrase-loop windows, splice, re-sort.

        The mirror of ``_retry_loops`` for loops the dominant-token detector is
        blind to: a repeated 3-token phrase caps every token's dominant fraction
        at ~1/3, but a joined token stream over the segments exposes the repeated
        n-gram directly (``text_metrics.find_phrase_runs``). Detecting over the
        JOINED stream catches BOTH observed shapes with one mechanism — a run
        living inside one raw segment (``152ed870``), and a run of identical
        consecutive segments (``5bab2c34``). A run is not allowed to cross a long
        pause: consecutive segments >`_PHRASE_JOIN_MAX_GAP_S` apart are separated
        by a unique sentinel that no n-gram can match across.

        Each run span maps to its first/last owning segment; that window is
        re-transcribed with ``no_repeat_ngram_size = loop_retry_ngram`` (the
        global knob untouched, restored in a ``finally`` — the same route as
        ``_retry_loops``). Accept guard (mirrored): splice ONLY if the retry is
        non-empty AND itself carries no phrase run AND every retry segment scores
        below ``loop_score_threshold`` (a retry must not trade a phrase loop for a
        token loop) AND it does not emit more tokens than the window it replaces
        (the counting-evasion bound — see ``_retry_loops``). Token runs are first
        merged into disjoint segment-index
        intervals (two token-disjoint runs can share a segment) and the intervals
        are spliced in REVERSE order so indices stay valid. Returns a new list
        (never mutates the input segments).
        """
        tokens, owner = self._join_segment_tokens(segments)
        runs = find_phrase_runs(tokens)
        if not runs:
            return list(segments)
        thr = self.cfg.loop_score_threshold
        ngram = self.cfg.loop_retry_ngram
        out = list(segments)
        # Map token runs -> segment-index intervals, then merge overlapping /
        # touching intervals: two token-disjoint runs can share a segment, and
        # splicing the same segment twice would corrupt indices and duplicate
        # retry content. Each merged interval keeps its strongest run for the log.
        intervals: list[list] = []          # [first_seg, last_seg, strongest_run]
        for run in sorted(runs, key=lambda r: r.start):
            first, last = owner[run.start], owner[run.end - 1]
            if intervals and first <= intervals[-1][1]:
                intervals[-1][1] = max(intervals[-1][1], last)
                if run.run > intervals[-1][2].run:
                    intervals[-1][2] = run
            else:
                intervals.append([first, last, run])
        for first, last, run in reversed(intervals):
            s0, e0 = segments[first]["start"], segments[last]["end"]
            sub = audio[int(s0 * self._SR):int(e0 * self._SR)]
            saved_options = self._asr.options
            self._asr.options = replace(saved_options, no_repeat_ngram_size=ngram)
            try:
                retry = self._asr.transcribe(
                    sub, language=self.cfg.language, chunk_size=self.cfg.chunk_size
                )
            finally:
                # Restore the original options no matter what — the ngram override
                # must never leak into the next window / stream.
                self._asr.options = saved_options
            rsegs = retry.get("segments") or []
            retry_text = " ".join((rs.get("text") or "") for rs in rsegs).strip()
            retry_tokens, _ = self._join_segment_tokens(rsegs)
            retry_still_loops = bool(find_phrase_runs(retry_tokens))
            retry_tok_score = max(
                (repetition_loop_score(rs.get("text")).score for rs in rsegs),
                default=0.0,
            )
            # Guard: keep the original unless the retry broke the phrase loop,
            # left content behind, did not fall into a token loop instead, AND
            # did not GROW the window (inverted collapse guard: a loop is
            # OVER-production, its repair must never emit more tokens than the
            # loop it replaces). Real trigger: under the ngram constraint the
            # decoder can evade both loop detectors by counting ("D1. D2. ...
            # D78.", 152ed870 — every repeat textually distinct); only the
            # length bound catches it.
            orig_tokens = sum(
                len(_WORD_RE.findall((segments[k].get("text") or "").lower()))
                for k in range(first, last + 1)
            )
            n_retry_tokens = sum(
                len(_WORD_RE.findall((rs.get("text") or "").lower()))
                for rs in rsegs
            )
            grew = n_retry_tokens > orig_tokens
            if not retry_text or retry_still_loops or retry_tok_score >= thr or grew:
                reason = (f"grew the window ({orig_tokens} -> "
                          f"{n_retry_tokens} tokens)"
                          if grew and retry_text and not retry_still_loops
                          and retry_tok_score < thr else
                          f"run={run.run} phrase=«{run.phrase}»")
                _log(
                    f"phrase-loop-retry REJECTED window [{s0:.1f}-{e0:.1f}] "
                    f"{reason}; keeping original"
                )
                continue
            _log(
                f"phrase-loop-retry window [{s0:.1f}-{e0:.1f}] run={run.run} "
                f"phrase=«{run.phrase}»; accepted, {len(rsegs)} segment(s)"
            )
            out[first:last + 1] = [
                {**rs, "start": rs["start"] + s0, "end": rs["end"] + s0}
                for rs in rsegs
            ]
        out.sort(key=lambda s: s["start"])
        return out

    def unload(self) -> None:
        self._asr = None
        self._align_model = None
        self._align_metadata = None
        self._device_str = None


# Isolated-venv CohereX worker (Diffio-AI/CohereX). Invoked under $COHEREX_VENV_PY,
# NOT the main venv — see scripts/coherex_worker.py for the isolation rationale.
_COHEREX_WORKER = Path(__file__).resolve().parents[2] / "scripts" / "coherex_worker.py"


class _CohereXBackend:
    """Cohere ASR (Diffio-AI/CohereX) via an isolated-venv subprocess.

    CohereX's ``coherex`` package + transformers pins + the 2B Cohere model
    conflict with the main venv, so this backend shells out to the interpreter in
    ``$COHEREX_VENV_PY`` running ``scripts/coherex_worker.py`` (same isolation as
    the Brouhaha scorer; its pinned deps mean it cannot use the main venv via
    ``sys.executable``). The worker loads the model per ``transcribe``
    call (per-speaker stream) — a few extra minutes on a single recording; for
    batch eval use the standalone sweep driver instead. SCOPE §4: a missing
    venv/worker is a loud crash at ``load`` — never a silent fall-back to WhisperX.

    Maps existing ``TranscriptionConfig`` fields to the worker: ``model_name`` is
    the Cohere model id, ``align_model_name`` the wav2vec2 aligner (None → the
    worker's Polish default), plus ``language`` / ``chunk_size`` / ``vad_onset`` /
    ``vad_offset`` / ``repetition_penalty`` / ``no_repeat_ngram_size``.
    """

    def __init__(self, cfg: TranscriptionConfig) -> None:
        self.cfg = cfg
        self._venv_py: Optional[str] = None

    def load(self, device: torch.device) -> None:
        import os
        venv_py = os.environ.get("COHEREX_VENV_PY")
        if not venv_py:
            raise RuntimeError(
                "transcription.backend='coherex' requires $COHEREX_VENV_PY — the "
                "path to the isolated CohereX venv's python (e.g. "
                "~/asr_model_compare/coherex_venv/bin/python). Set it, or use "
                "backend='whisperx'. (No silent fall-back — SCOPE §4.)"
            )
        if not Path(venv_py).exists():
            raise FileNotFoundError(f"$COHEREX_VENV_PY not found: {venv_py}")
        if not _COHEREX_WORKER.exists():
            raise FileNotFoundError(f"CohereX worker missing: {_COHEREX_WORKER}")
        self._venv_py = venv_py
        _log(f"load: CohereX backend ready (venv={venv_py}, "
             f"worker={_COHEREX_WORKER.name}); model loads per-call in subprocess")

    def transcribe(self, audio: np.ndarray) -> dict:
        if self._venv_py is None:
            raise RuntimeError("_CohereXBackend.transcribe called before load().")
        import subprocess
        import tempfile

        import soundfile as sf
        with tempfile.TemporaryDirectory() as td:
            tin = str(Path(td) / "in.wav")
            tout = str(Path(td) / "out.json")
            sf.write(tin, np.asarray(audio, dtype=np.float32), 16_000)
            cmd = [
                self._venv_py, str(_COHEREX_WORKER),
                "--in", tin, "--out", tout,
                "--asr-model", self.cfg.model_name,
                "--language", self.cfg.language,
                "--chunk-size", str(self.cfg.chunk_size),
                "--vad-onset", str(self.cfg.vad_onset),
                "--vad-offset", str(self.cfg.vad_offset),
            ]
            if self.cfg.align_model_name:
                cmd += ["--align-model", self.cfg.align_model_name]
            if self.cfg.no_repeat_ngram_size:
                cmd += ["--no-repeat-ngram", str(self.cfg.no_repeat_ngram_size)]
            if self.cfg.repetition_penalty and self.cfg.repetition_penalty != 1.0:
                cmd += ["--rep-penalty", str(self.cfg.repetition_penalty)]
            # Neutral cwd (scripts/) so the worker never resolves the repo's local
            # `datasets/` package; inherit env for HF_TOKEN / CUDA.
            proc = subprocess.run(
                cmd, cwd=str(_COHEREX_WORKER.parent),
                capture_output=True, text=True,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"CohereX worker failed (exit {proc.returncode}). "
                    f"stderr tail:\n{proc.stderr[-2000:]}"
                )
            with open(tout, encoding="utf-8") as f:
                result = json.load(f)
        return _normalise_result(result, self.cfg.language)

    def unload(self) -> None:
        self._venv_py = None


# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------


class TranscriptionStage(Stage):
    name = "transcription"

    def __init__(self, config: TranscriptionConfig) -> None:
        super().__init__(enabled=config.enabled)
        self.config = config
        self._backend: Optional[
            _WhisperXBackend | _CohereXBackend] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def load(self, device: torch.device) -> None:
        # loop_retry / loop_retry_phrase re-decode looped windows by overriding
        # faster-whisper's TranscriptionOptions.no_repeat_ngram_size — machinery
        # only the whisperx backend carries. Fail loud here (at pipeline init,
        # before any audio) rather than silently no-op on whisper/coherex (SCOPE
        # §4.1). Same backend-level deferral as the other WhisperX-only knobs.
        if self.config.backend != "whisperx" and (
            self.config.loop_retry or self.config.loop_retry_phrase
        ):
            set_flags = [
                name for name in ("loop_retry", "loop_retry_phrase")
                if getattr(self.config, name)
            ]
            raise ValueError(
                f"transcription.{'/'.join(set_flags)}=True is only supported by "
                "the 'whisperx' backend (it re-decodes looped windows with "
                "no_repeat_ngram_size via faster-whisper's TranscriptionOptions); "
                f"backend={self.config.backend!r} has no equivalent. Use "
                f"backend='whisperx' or set {'/'.join(set_flags)}=False. (No "
                "silent no-op — SCOPE §4.1.)"
            )
        if self.config.backend == "whisperx":
            self._backend = _WhisperXBackend(self.config)
        elif self.config.backend == "coherex":
            self._backend = _CohereXBackend(self.config)
        else:
            raise ValueError(f"Unknown transcription backend: {self.config.backend!r}")
        self._backend.load(device)

    def load_signature(self) -> tuple:
        # Per-call options (language / initial_prompt / word_timestamps) are
        # not part of model identity. Backend, model id, and (for whisperx)
        # alignment model id are.
        if self.config.backend in ("whisperx", "coherex"):
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
            if self._skip_transcription(audio, min_samples, self.config.silence_floor):
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
            if self._skip_transcription(ctx.audio, min_samples, self.config.silence_floor):
                _log(
                    "run: mixture short/silent — empty transcript, "
                    "Whisper not called"
                )
                ctx.mixture_transcript = _empty_result(self.config.language)
            else:
                ctx.mixture_transcript = self._backend.transcribe(ctx.audio)

    @staticmethod
    def _skip_transcription(
        audio: np.ndarray, min_samples: int, silence_floor: float
    ) -> bool:
        """True if `audio` is too short or silent to be worth transcribing.

        `silence_floor` is the peak-amplitude gate
        (`TranscriptionConfig.silence_floor`); passed in because this is a
        staticmethod with no `self`/config access.
        """
        if len(audio) < min_samples or len(audio) == 0:
            return True
        return float(np.max(np.abs(audio))) < silence_floor

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
