"""Configuration for the ASR pipeline.

Pattern mirrors `polsess_separation/config.py`: nested dataclasses,
`__post_init__` validation, YAML round-trip via `asdict() + yaml.dump()`.

The configuration is intentionally self-contained — no dependency on the
parent project's `Config` — so the package can be lifted into CLARIN with
only the one separator-loading seam in `stages/separation.py` to edit.
"""

import json
import math
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Union

import yaml


# Placeholder written in place of a live HF token in any serialised snapshot.
# Shared by the redactor (write side) and the loader (read side) so they can't
# drift — see `redact_config_snapshot` and `load_pipeline_config_from_dict`.
_REDACTED = "REDACTED"


def redact_config_snapshot(config_snapshot: dict) -> dict:
    """Deep-copy a config snapshot with ``diarization.hf_token`` masked.

    The single source of truth for which config fields must never appear in
    a serialised snapshot (``metadata.json`` via ``io.write_pipeline_outputs``,
    saved YAML via ``save_pipeline_config_to_yaml``). The token is only needed
    at model-load time, never for reproducibility, so it's safe to mask
    unconditionally. Defined here in the lightweight config leaf so both
    writers can import it without pulling in ``io``'s heavy deps.
    """
    snap = json.loads(json.dumps(config_snapshot))  # cheap deep copy (JSON-safe input)
    diar = snap.get("diarization")
    if isinstance(diar, dict) and diar.get("hf_token"):
        diar["hf_token"] = _REDACTED
    return snap


def _one_of(value, name: str, allowed: tuple) -> None:
    """Raise ValueError unless `value` is one of `allowed`. Keeps the
    enum-string checks in `PipelineConfig.__post_init__` uniform."""
    if value not in allowed:
        raise ValueError(
            f"Invalid {name}: {value!r} (allowed: {', '.join(map(repr, allowed))})"
        )


# ---------------------------------------------------------------------------
# Per-stage configs
# ---------------------------------------------------------------------------


@dataclass
class DiarizationConfig:
    """Stage 1: pyannote speaker diarization."""

    enabled: bool = True
    model_id: str = "pyannote/speaker-diarization-3.1"
    hf_token: Optional[str] = field(
        default_factory=lambda: os.getenv("HF_TOKEN", None)
    )
    num_speakers: int = 2


@dataclass
class RoutingConfig:
    """Stage 2: decide what gets sent to SepFormer.

    The job is just to select overlap regions from pyannote's diarization
    output: drop ones that are too short to be worth separating, and merge
    ones that sit close enough together that SepFormer should see them as
    a single contiguous region rather than two back-to-back calls.

    Per-speaker solo regions are NOT computed here — the assembler in
    Stage 4 derives them on the fly from `ctx.diarization.segments_df`,
    subtracting each overlap's seam-adjusted emit region (see `assembly.py`).

    All thresholds in seconds.
    """

    enabled: bool = True
    min_overlap_dur: float = 0.20    # drop overlaps shorter than this
    merge_gap: float = 0.50          # merge overlap regions closer than this


@dataclass
class EnhancementConfig:
    """Stage 3a: full-recording speech enhancement (single pass; sliced
    per-speaker at assembly).

    Backends are ClearerVoice-Studio single-output SE models, trained on
    broad DNS-Challenge data. They self-download by name to a HuggingFace
    cache on first use (no checkpoint path to configure).
    """

    enabled: bool = True
    # Backend selector:
    #   - "frcrn_se_16k": FRCRN, DNS-2020 winner, native 16k (ClearerVoice)
    #   - "mossformer_gan_se_16k": MossFormer + GAN losses, 16k (ClearerVoice)
    #   - "zipenhancer_16k": ZipEnhancer, native 16k (ModelScope
    #     iic/speech_zipenhancer_ans_multiloss_16k_base; needs `modelscope`).
    #     DNS-2020 PESQ leader; run via ModelScope ANS pipeline.
    # Interim default per SCOPE §10 q7 (mpsenet removed 2026-06-11; FRCRN is
    # the evidence leader). The *final* default ruling is deferred until there
    # is substantive testing data.
    backend: str = "frcrn_se_16k"
    # Long recordings are processed via Hann overlap-add — chunks of this
    # size with a 50% hop (canonical COLA: window sum is 1.0 in the interior,
    # head/tail divided by actual weights). 8 s chunks were verified by ear on
    # long Polish recordings.
    max_segment_length_s: float = 8.0
    # Observation Adding (OA) / dry-wet mix per Iwamoto et al. 2022
    # (arXiv:2201.06685) and Wang et al. 2024 (arXiv:2406.12699): convexly
    # blend the original observed (dry) signal back into the enhanced (wet)
    # output —  `out = (1 - r)*enhanced + r*observed`. 0.0 = pure enhanced
    # (current behaviour); higher r dilutes SE artifacts that hurt ASR at the
    # cost of output audio quality (moves the WER↔SQUIM frontier). Solo-scoped
    # automatically: `enhanced_full` feeds only the per-speaker solo regions
    # (overlaps separate from the original audio), so OA never touches the
    # overlap path — exactly where the papers warn it harms.
    observation_mix_ratio: float = 0.0


@dataclass
class SeparationConfig:
    """Stage 3b: source separation on overlap regions + VAD gating.

    Knobs are exposed as enum-string fields rather than separate strategy
    classes (one function with branches per knob). Defaults match
    ``configs/default.yaml`` (the config the eval harness runs) so
    programmatic users and YAML users get the same pipeline; a pin test
    in ``tests/test_pipeline_config.py`` enforces the agreement.
    """

    enabled: bool = True
    checkpoint_path: str = (
        "checkpoints/mossformer2/SB/mossformer2_matched_128k_final_42_e31/mossformer2_SB_best_e31.pt"
    )
    separator_sample_rate: int = 8_000   # SR the separator was trained at
    # Audio duration (seconds) the separator was trained on. Used as the
    # target padded-window size for `context_window_mode != "none"` and as
    # the chunk size for long-overlap overlap-add.
    training_chunk_length_s: float = 4.0

    # Context-window strategy: how much audio around the overlap to feed the
    # separator. POC behaviour is `none` (no extra context).
    #   - "expand_to_chunk": expand the window asymmetrically until it matches
    #     the separator's training chunk length (`training_chunk_length_s`).
    #     The separator was trained on mid-utterance crops, so cutting an
    #     utterance at the pad boundary is fine here — utterance-aware
    #     boundary handling belongs in `seam_mode` (emit region).
    #   - "fixed_pad": symmetric ±`context_pad_seconds` window with the same
    #     `min_fragment_length_s` floor.
    #   - "none": no extra context (POC behaviour).
    context_window_mode: str = "expand_to_chunk"   # "expand_to_chunk" | "fixed_pad" | "none"
    context_pad_seconds: float = 1.0            # used by `fixed_pad`
    # Minimum total padded-window length sent to the separator (seconds).
    # Short overlaps (e.g. 0.4 s) padded by `context_pad_seconds` may still be
    # well under the separator's training chunk length, leaving it with little
    # context. When the natural pad fails to reach this floor, the helpers
    # extend the window further until it does. Asymmetric: if one side hits
    # the recording boundary, the leftover budget is redistributed to the
    # other side so the available context is maximised. Ignored when
    # `context_window_mode == "none"` (the user explicitly opted out of pad).
    min_fragment_length_s: float = 4.0

    # Seam strategy: where the boundary between separated 8k output and the
    # surrounding 16k solo audio is placed when the padded window exceeds the
    # original overlap region.
    #   - "overlap_boundary":  cut exactly at pyannote's overlap boundary
    #   - "zero_crossing":     boundary + small nudge to a nearby zero crossing
    #                          (avoids clicks at the splice point)
    #   - "snap_to_silence":   first compute the zero_crossing boundary, then
    #                          extend *outward* via VAD silence on the
    #                          separator output (captures vowel tails that
    #                          pyannote cut). Always emits a region at least
    #                          as wide as zero_crossing.
    seam_mode: str = "snap_to_silence"  # "zero_crossing" | "overlap_boundary" | "snap_to_silence"
    # Maximum distance (seconds) from the original overlap boundary to scan
    # for a zero crossing.
    seam_search_radius_s: float = 0.05
    # When `seam_mode == "snap_to_silence"`: maximum distance (seconds) to
    # extend each boundary outward looking for VAD silence in the separator
    # output. If no silence found in this window, falls back to the
    # zero_crossing boundary (never contracts).
    snap_silence_max_extend_s: float = 0.3

    # Long-overlap chunking: overlaps longer than this trigger overlap-add.
    overlap_add_threshold_s: float = 12.0

    # Volume normalisation applied to each separated stream before emission.
    # Per-speaker overlap-to-solo RMS matching lives in AssemblyConfig
    # (`overlap_rms_match_solo`), not here.
    volume_normalization: str = "sum_equals_mix"  # "sum_equals_mix" | "none"

    # VAD applied to each separator output to suppress residuals.
    # `vad_threshold` is the "definitely speech" upper bound. Frames above it
    # always become part of the speech mask.
    vad_threshold: float = 0.25
    # Schmitt-trigger style lower threshold. Frames between
    # `vad_soft_threshold` and `vad_threshold` count as speech *only* if they
    # extend a chain that touches a frame above `vad_threshold` (propagated
    # both forward and backward). This captures tails/onsets where silero is
    # less confident — common on neural-separator outputs. Set to a value
    # >= `vad_threshold` to disable (mask becomes a strict threshold).
    vad_soft_threshold: float = 0.10
    # Fixed positional dilation in 32 ms frames applied on top of the
    # threshold mask. Extends each speech run by `vad_attack_frames` before
    # its onset and `vad_release_frames` after its offset, regardless of
    # frame probability. Set both to 0 to disable.
    vad_attack_frames: int = 1
    vad_release_frames: int = 1


@dataclass
class PostSeparationProcessingConfig:
    """Stage 3c: VAD mask application + optional bandwidth extension.

    Reads ``s{1,2}_raw`` (unmasked separator output) and ``mask{1,2}``
    (VAD mask computed by 3b) from ``ctx.overlap_separated`` entries,
    runs the selected backend on the raw streams, multiplies by the
    mask, writes the result to ``s{1,2}_gated``. Stage 4 consumes
    ``_gated`` arrays only.

    Always-on: this stage has no ``enabled`` knob because Stage 4
    depends on ``s_gated`` being populated. To run "VAD mask only with
    no BWE", set ``backend: naive`` — the stage still runs (the mask
    multiplication is its core job) but no neural model touches the
    audio. The 8 kHz spectral content is left as-is.
    """

    # Backend selector:
    #   - "naive": no neural model. Input is already 16 kHz numerically
    #     (separator output upsampled by polyphase resampling); this
    #     backend is identity. Use it as the A/B baseline against the
    #     neural backends.
    #   - "ap_bwe": vendored AP-BWE (Lu et al. 2024). Discriminative
    #     ConvNeXt + STFT dual-stream model, ~22 M params, fully
    #     convolutional, ~18× real-time on CPU, native 8→16 kHz.
    #     Requires the user to download the pretrained checkpoint from
    #     Google Drive (see asr_pipeline/vendor/ap_bwe/README.md).
    #   - "flowhigh": FlowHigh (Yun et al., ICASSP 2025, 2501.04926).
    #     Single-step flow-matching SR model from Resemble AI's pip
    #     fork. Native output 48 kHz → downsampled internally to the
    #     pipeline rate. Reports beating AP-BWE on VCTK LSD/ViSQOL.
    #     Install: pip install git+https://github.com/resemble-ai/flowhigh.git@dev
    #     Checkpoint auto-downloads on first FlowHighSR.from_pretrained().
    backend: str = "naive"
    # Path to the AP-BWE generator checkpoint (PyTorch state dict
    # containing the 'generator' key). Ignored by non-AP-BWE backends.
    checkpoint_path: str = field(
        default_factory=lambda: os.getenv(
            "AP_BWE_CHECKPOINT",
            "/home/user/AP-BWE/checkpoints/8kto16k/g_8kto16k",
        )
    )
    # FlowHigh's input sample rate. The README explicitly lists 12 kHz
    # and 16 kHz examples and states "any rate < 48 kHz". 8 kHz isn't
    # confirmed in the docs but isn't excluded either — set this knob to
    # 8000 to A/B-test the narrower input (which matches the separator's
    # 0-4 kHz spectral content more honestly). Default 16000 matches the
    # pipeline rate so the in-path has no resample (only the 48→16
    # downsample on the way out). Ignored by non-FlowHigh backends.
    flowhigh_input_sr: int = 16_000


@dataclass
class AssemblyConfig:
    """Stage 4: per-speaker stream assembly + timestamp map."""

    enabled: bool = True
    min_solo_for_anchor_s: float = 3.0
    # ECAPA only needs a few seconds of audio for a stable speaker embedding,
    # but a richer anchor sharpens overlap speaker-assignment, so we feed as
    # much solo as is safe. On a long recording (e.g. 15 min) the per-speaker
    # solo concat can grow to hundreds of seconds; feeding 400 s+ to ECAPA in a
    # single forward OOMs the GPU or stalls the kernel, so we still cap (taking
    # a uniformly-strided sample so we don't bias toward the start). 240 s fully
    # covers our ≤90 s eval fragments (no clipping) and stays under the ~400 s
    # OOM zone. Set to None to disable the cap.
    anchor_max_duration_s: Optional[float] = 240.0
    # Output mode for the assembled per-speaker streams:
    #   "shortened"    -> speech-only concat with `silence_separator_s` between pieces
    #   "full_length"  -> total stream length = input length; gaps filled with silence
    output_mode: str = "shortened"              # "shortened" | "full_length"
    silence_separator_s: float = 0.3
    # Half-Hann fade applied at internal piece-to-piece seams (one piece's
    # fade-out meets the next piece's fade-in). Set to 0 to disable.
    crossfade_ms: float = 5.0
    # Half-Hann fade applied at the very start of a speaker's first piece
    # and the very end of their last piece — where the only neighbour is
    # silence outside the stream. Can be shorter than `crossfade_ms` because
    # only one side of the seam needs to ramp.
    edge_fade_ms: float = 2.0
    # Per-speaker RMS match: scale each overlap event so its RMS matches the
    # median RMS of that speaker's solo events. Fixes the common case where
    # SepFormer outputs (with sum_equals_mix normalisation) end up noticeably
    # louder than the enhancer's solo audio. Solo events are not touched, so the
    # speaker's natural dynamics are preserved.
    overlap_rms_match_solo: bool = True
    # Optional aggressive per-piece RMS normalisation before concat. When
    # enabled and `target_rms` is None, the median RMS across pieces is used.
    # Applied after `overlap_rms_match_solo`; if both are on, this dominates.
    per_piece_rms_norm: bool = False
    target_rms: Optional[float] = None


@dataclass
class TranscriptionConfig:
    """Stage 5: Whisper ASR per assembled stream.

    Two backends, same surface contract::

      - ``whisper`` (default): the original OpenAI Whisper package
        (``whisper.load_model``). Supports the canonical OpenAI checkpoints
        (``large-v3``, ``large-v2``, ``medium``, …). Fast to set up but no
        wav2vec2 word-level alignment — Whisper's own word timestamps are
        token-aligned and drift by 200–500 ms.
      - ``whisperx``: WhisperX = faster-whisper + wav2vec2 forced alignment.
        Word-level timestamps to ±50 ms. Also the only backend that supports
        non-OpenAI Whisper checkpoints via HF model id (e.g. a
        language-specific Whisper finetune).

    Both backends emit the same output shape per speaker::

        {"text": str, "segments": [{"start": float, "end": float,
                                    "text": str, "words": [...]?}],
         "language": str}
    """

    enabled: bool = True
    # Selector. See class docstring for trade-offs.
    backend: str = "whisper"           # whisper | whisperx
    # OpenAI short names (``large-v3``, ``large-v2``) or any HF Whisper model id
    # parseable by faster-whisper — the latter only when backend == ``whisperx``.
    # The ``whisper`` backend accepts canonical OpenAI names only.
    model_name: str = "large-v3"
    language: str = "pl"
    initial_prompt: str = "Rozmowa po polsku."
    word_timestamps: bool = True

    # --- Whisper decode knobs (both backends) -------------------------------
    # Exposed so a config sweep can vary decoding. Defaults reproduce the
    # CURRENT pipeline behaviour exactly — i.e. WhisperX's own
    # ``default_asr_options`` (whisperx/asr.py ``load_model``), which is the
    # default/eval backend. Note WhisperX overrides several faster-whisper
    # signature defaults (most notably ``condition_on_previous_text``), so
    # these defaults are taken from WhisperX, not from faster-whisper.
    #
    # Beam width for beam search. WhisperX default = 5 (asr.py
    # ``default_asr_options["beam_size"]``). Must be >= 1.
    beam_size: int = 5
    # Temperature fallback schedule. A single float means "no fallback"; a
    # list is the descending schedule Whisper retries when a decode trips the
    # compression-ratio / logprob gate (openai-whisper / faster-whisper
    # fallback semantics). WhisperX default = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    # (asr.py ``temperatures``). Each value must lie in [0, 1]. A list default
    # (via default_factory) so it stays YAML/JSON round-trippable as a plain
    # sequence; YAML may supply a scalar or a list.
    temperature: Union[float, List[float]] = field(
        default_factory=lambda: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    )
    # Feed the previous window's text as the next window's prompt. WhisperX
    # default = False (asr.py ``condition_on_previous_text``) — this DIFFERS
    # from faster-whisper's signature default of True, so the WhisperX value
    # is the one that reproduces current behaviour.
    condition_on_previous_text: bool = False
    # No-speech probability above which a segment is treated as silence.
    # WhisperX default = 0.6 (asr.py ``no_speech_threshold``). Finite.
    no_speech_threshold: float = 0.6
    # gzip compression ratio above which a decode is treated as a
    # hallucination and the temperature fallback fires. WhisperX default =
    # 2.4 (asr.py ``compression_ratio_threshold``). Finite.
    compression_ratio_threshold: float = 2.4
    # Beam-search patience (Kasai et al. 2021). WhisperX default = 1
    # (asr.py ``patience``); stored as float here, numerically identical.
    # Must be > 0.
    patience: float = 1.0

    # --- Anti-hallucination decode knobs (faster-whisper / WhisperX only) ----
    # These three live in faster-whisper's ``transcribe`` signature, and
    # WhisperX's ``default_asr_options`` carries each at the SAME value as the
    # faster-whisper signature default — so the defaults below reproduce current
    # behaviour exactly (a baseline run stays byte-identical). Evidence
    # (verified 2026-06-14 against the pinned venv): faster-whisper signature
    # defaults / WhisperX default_asr_options — ``no_repeat_ngram_size`` 0 / 0,
    # ``repetition_penalty`` 1.0 / 1, ``hallucination_silence_threshold``
    # None / None. All three agree, so unlike the six knobs above there is no
    # WhisperX-vs-faster-whisper override to reconcile.
    #
    # openai-whisper (the ``whisper`` backend) does NOT support these as
    # faster-whisper does: ``no_repeat_ngram_size`` / ``repetition_penalty``
    # are absent from its decode surface entirely, and while its ``transcribe``
    # has a param literally named ``hallucination_silence_threshold`` it is a
    # different feature (a different silence-skip algorithm). Forwarding any of
    # them to openai-whisper would either crash or silently substitute a
    # different behaviour — SCOPE §4.1 forbids the latter. So the defaults below
    # (0 / 1.0 / None) are a no-op for the ``whisper`` backend (never passed),
    # and a non-default value with ``backend == "whisper"`` is a loud error
    # (see ``_WhisperBackend.transcribe``).
    #
    # Block any N-gram of this size from repeating in the decode. WhisperX /
    # faster-whisper default = 0 (disabled). Must be an int >= 0.
    no_repeat_ngram_size: int = 0
    # Penalty applied to already-emitted tokens (> 1 discourages repeats).
    # WhisperX / faster-whisper default = 1.0 (no penalty). Must be > 0 and
    # finite.
    repetition_penalty: float = 1.0
    # When set, faster-whisper skips silent gaps longer than this many seconds
    # where hallucinations cluster (requires ``word_timestamps=True``, which the
    # pipeline already sets). WhisperX / faster-whisper default = None (off).
    # None = off, or a positive finite number of seconds.
    hallucination_silence_threshold: Optional[float] = None

    # WhisperX-only knobs (ignored when ``backend != whisperx``):
    # the wav2vec2 model used for forced alignment.
    #   None = WhisperX picks its per-language default
    #          (pl → jonatasgrosman/wav2vec2-large-xlsr-53-polish, the previous
    #          pinned value; en → torchaudio WAV2VEC2_ASR_BASE_960H).
    #   Set explicitly to override (e.g. the English XLSR-53 aligner — see
    #   configs/english.yaml).
    align_model_name: Optional[str] = None
    # WhisperX VAD chunk size (seconds): the max length of a merged VAD
    # speech segment fed to Whisper in one window. WhisperX default = 30
    # (= Whisper's receptive field). At 30 a long unbroken VAD segment can hit
    # the window ceiling and make Whisper collapse — emit ~nothing for ~30 s of
    # clear speech (observed on db15fc57: 39-68 s dropped). A smaller value
    # forces WhisperX to split such segments, recovering the dropped speech, at
    # the cost of slightly less decode context. Must be an int >= 1. 30 = current
    # behaviour (byte-identical baseline). Ignored by the ``whisper`` backend
    # (openai-whisper has no VAD chunking) — a non-default value there is a loud
    # error, like the anti-hallucination knobs above.
    chunk_size: int = 30

    # --- Detect-and-retry for collapsed WhisperX windows (WhisperX only) ------
    # The same over-merge failure that ``chunk_size`` addresses, but fixed
    # surgically instead of globally. WhisperX merges VAD speech into windows up
    # to ``chunk_size`` (=30 s); a long (~18-30 s) merged window can make Whisper
    # *collapse* — emit ~nothing for the whole window, dropping ~all of that
    # speech (observed on db15fc57, fe65d170, 72ca135e). Lowering ``chunk_size``
    # globally fixes the collapses but re-splits clean windows too, adding broad
    # collateral on configs that never collapse. Detect-and-retry instead runs
    # the normal ``chunk_size`` pass, detects only the collapsed windows, and
    # re-transcribes *just those* at a small chunk — near-zero collateral.
    #
    # These three knobs are WhisperX-only (the ``whisper`` backend does its own
    # internal windowing and does not exhibit this exact failure). Unlike
    # ``chunk_size``, the retry default is ON (8), so a non-default value with
    # backend="whisper" is NOT a hard error — it is simply not applicable there
    # and is ignored with a visible one-time log (SCOPE §4.1), never silently.
    #
    # Re-transcribe each detected collapsed window at this chunk size (seconds).
    # 0 = disabled (no retry pass; pure ``chunk_size`` behaviour). 8 = shipped
    # behaviour (a small chunk reliably breaks the over-merge). Must be an int
    # >= 0.
    retry_collapsed_chunk_size: int = 8
    # A window is collapse-eligible only if it is at least this long (seconds).
    # Collapsed windows always sit near the ``chunk_size`` ceiling; this floor
    # keeps the detector off normal short windows. Must be positive and finite.
    collapse_min_duration_s: float = 18.0
    # A long window counts as collapsed if its word density (words / second) is
    # below this. Normal Polish speech is ~2-4 w/s, so a long window emitting
    # < 0.7 w/s has effectively dropped its speech. Must be positive and finite.
    collapse_max_wps: float = 0.7
    # When True, additionally run the same backend on the whole mixture
    # (``ctx.audio``) as a single stream, writing the result to
    # ``ctx.mixture_transcript``. Used for the thesis ablation table
    # (mixture baseline vs. pipeline-with-separation). Same backend,
    # prompt, and args as the per-speaker transcription — otherwise the
    # ablation comparison isn't fair.
    transcribe_mixture: bool = False


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""

    # Working sample rate for the pipeline (diarization / SE / VAD / ASR all
    # operate here). The separator runs at its own `separator_sample_rate`.
    sample_rate: int = 16_000
    device: str = "cuda"

    # Force deterministic cuDNN algorithms so runs are reproducible. The
    # enhancement conv stack (e.g. FRCRN) is otherwise the pipeline's
    # sole source of run-to-run nondeterminism — it picks nondeterministic cuDNN
    # algorithms that inject ~1e-7 float noise into `enhanced_full`, which
    # WhisperX occasionally amplifies into a flipped token (per-speaker WER
    # swings; the mixture floor is unaffected). Every other stage is
    # deterministic given fixed input, so this one flag makes the whole pipeline
    # reproducible. Costs a modest enhancement-stage slowdown (no conv
    # autotuning); set False to restore cuDNN's default autotuning.
    deterministic: bool = True

    # If True, after each stage runs, its outputs are spilled to
    # `artifact_dir`. Models are still freed at unload time regardless.
    spill_intermediate: bool = False
    # Output directory for spilled artefacts. Required when `spill_intermediate`
    # is True; ignored otherwise.
    artifact_dir: Optional[str] = None

    diarization: DiarizationConfig = field(default_factory=DiarizationConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    enhancement: EnhancementConfig = field(default_factory=EnhancementConfig)
    separation: SeparationConfig = field(default_factory=SeparationConfig)
    post_separation_processing: PostSeparationProcessingConfig = field(
        default_factory=PostSeparationProcessingConfig
    )
    assembly: AssemblyConfig = field(default_factory=AssemblyConfig)
    transcription: TranscriptionConfig = field(default_factory=TranscriptionConfig)

    def __post_init__(self):
        # Validate enum-string knobs early so misconfiguration is loud.
        _one_of(self.separation.context_window_mode, "context_window_mode",
                ("expand_to_chunk", "fixed_pad", "none"))
        _one_of(self.separation.seam_mode, "seam_mode",
                ("zero_crossing", "overlap_boundary", "snap_to_silence"))
        _one_of(self.separation.volume_normalization, "volume_normalization",
                ("sum_equals_mix", "none"))
        _one_of(self.assembly.output_mode, "output_mode",
                ("shortened", "full_length"))
        _one_of(self.enhancement.backend, "enhancement.backend",
                ("frcrn_se_16k", "mossformer_gan_se_16k", "zipenhancer_16k"))
        _one_of(self.post_separation_processing.backend,
                "post_separation_processing.backend",
                ("naive", "ap_bwe", "flowhigh"))
        _one_of(self.transcription.backend, "transcription.backend",
                ("whisper", "whisperx"))

        if self.separation.training_chunk_length_s <= 0:
            raise ValueError(
                f"separation.training_chunk_length_s must be positive, got "
                f"{self.separation.training_chunk_length_s} (a non-positive "
                f"chunk length would make overlap-add hop 0 → infinite loop)."
            )
        if self.separation.overlap_add_threshold_s <= 0:
            raise ValueError(
                f"separation.overlap_add_threshold_s must be positive, got "
                f"{self.separation.overlap_add_threshold_s}"
            )
        if self.separation.vad_soft_threshold < 0:
            raise ValueError(
                f"separation.vad_soft_threshold must be >= 0, got "
                f"{self.separation.vad_soft_threshold} (a negative value "
                f"makes every frame 'weak' and floods the Schmitt mask)."
            )

        if self.post_separation_processing.flowhigh_input_sr <= 0:
            raise ValueError(
                f"flowhigh_input_sr must be positive, got "
                f"{self.post_separation_processing.flowhigh_input_sr}"
            )

        omr = self.enhancement.observation_mix_ratio
        if not math.isfinite(omr) or not (0.0 <= omr <= 1.0):
            raise ValueError(
                f"enhancement.observation_mix_ratio must be a finite value in "
                f"[0, 1] (convex dry-wet weight; 0 = pure enhanced), got {omr}"
            )

        # --- Transcription decode knobs ---
        tcfg = self.transcription
        if tcfg.beam_size < 1:
            raise ValueError(
                f"transcription.beam_size must be >= 1, got {tcfg.beam_size}"
            )
        if tcfg.patience <= 0 or not math.isfinite(tcfg.patience):
            raise ValueError(
                f"transcription.patience must be a positive finite number, got "
                f"{tcfg.patience}"
            )
        if not math.isfinite(tcfg.no_speech_threshold):
            raise ValueError(
                f"transcription.no_speech_threshold must be finite, got "
                f"{tcfg.no_speech_threshold}"
            )
        if not math.isfinite(tcfg.compression_ratio_threshold):
            raise ValueError(
                f"transcription.compression_ratio_threshold must be finite, got "
                f"{tcfg.compression_ratio_threshold}"
            )
        # temperature: scalar or schedule, each entry a finite value in [0, 1].
        temps = (
            tcfg.temperature
            if isinstance(tcfg.temperature, (list, tuple))
            else [tcfg.temperature]
        )
        if len(temps) == 0:
            raise ValueError(
                "transcription.temperature must be a float or a non-empty "
                "list of floats, got an empty sequence."
            )
        for t in temps:
            if not math.isfinite(t) or not (0.0 <= t <= 1.0):
                raise ValueError(
                    f"transcription.temperature values must each be in [0, 1], "
                    f"got {tcfg.temperature!r}"
                )
        # Anti-hallucination knobs (faster-whisper / WhisperX backend).
        if tcfg.no_repeat_ngram_size < 0:
            raise ValueError(
                f"transcription.no_repeat_ngram_size must be >= 0 (0 = disabled), "
                f"got {tcfg.no_repeat_ngram_size}"
            )
        if tcfg.repetition_penalty <= 0 or not math.isfinite(tcfg.repetition_penalty):
            raise ValueError(
                f"transcription.repetition_penalty must be a positive finite "
                f"number (1.0 = no penalty), got {tcfg.repetition_penalty}"
            )
        if tcfg.hallucination_silence_threshold is not None and (
            tcfg.hallucination_silence_threshold <= 0
            or not math.isfinite(tcfg.hallucination_silence_threshold)
        ):
            raise ValueError(
                f"transcription.hallucination_silence_threshold must be None "
                f"(off) or a positive finite number of seconds, got "
                f"{tcfg.hallucination_silence_threshold}"
            )
        if tcfg.chunk_size < 1:
            raise ValueError(
                f"transcription.chunk_size must be an int >= 1 (seconds; "
                f"30 = WhisperX default), got {tcfg.chunk_size}"
            )
        # Detect-and-retry knobs (WhisperX-only collapse recovery).
        if tcfg.retry_collapsed_chunk_size < 0:
            raise ValueError(
                f"transcription.retry_collapsed_chunk_size must be an int >= 0 "
                f"(0 = disabled; 8 = default), got "
                f"{tcfg.retry_collapsed_chunk_size}"
            )
        if (tcfg.collapse_min_duration_s <= 0
                or not math.isfinite(tcfg.collapse_min_duration_s)):
            raise ValueError(
                f"transcription.collapse_min_duration_s must be a positive "
                f"finite number of seconds, got {tcfg.collapse_min_duration_s}"
            )
        if (tcfg.collapse_max_wps <= 0
                or not math.isfinite(tcfg.collapse_max_wps)):
            raise ValueError(
                f"transcription.collapse_max_wps must be a positive finite "
                f"words/second threshold, got {tcfg.collapse_max_wps}"
            )

        if self.spill_intermediate and self.artifact_dir is None:
            raise ValueError(
                "spill_intermediate is True but artifact_dir is None — "
                "set artifact_dir (e.g. via --output) or disable spilling."
            )

        if self.diarization.enabled and not self.diarization.hf_token:
            raise ValueError(
                "diarization.enabled is True but hf_token is unset — "
                "export HF_TOKEN or set diarization.hf_token in YAML."
            )


# ---------------------------------------------------------------------------
# YAML loading / saving
# ---------------------------------------------------------------------------


def load_pipeline_config_from_dict(config_dict: dict) -> PipelineConfig:
    """Build a `PipelineConfig` from a nested dict (e.g. parsed YAML).

    Unknown top-level or stage-level keys raise; missing keys fall back to
    the dataclass defaults.
    """
    config_dict = dict(config_dict or {})

    sub_configs = {}
    for key, cls in (
        ("diarization", DiarizationConfig),
        ("routing", RoutingConfig),
        ("enhancement", EnhancementConfig),
        ("separation", SeparationConfig),
        ("post_separation_processing", PostSeparationProcessingConfig),
        ("assembly", AssemblyConfig),
        ("transcription", TranscriptionConfig),
    ):
        sub_dict = config_dict.pop(key, None)
        if key == "diarization" and sub_dict and sub_dict.get("hf_token") == _REDACTED:
            # A saved config redacts the token to _REDACTED; drop the key (a new
            # dict, never mutating the caller's) so DiarizationConfig's
            # default_factory re-resolves $HF_TOKEN instead of handing pyannote
            # the literal string "REDACTED".
            sub_dict = {k: v for k, v in sub_dict.items() if k != "hf_token"}
        sub_configs[key] = cls(**sub_dict) if sub_dict else cls()

    return PipelineConfig(**config_dict, **sub_configs)


def load_pipeline_config_from_yaml(yaml_path: str) -> PipelineConfig:
    """Load a `PipelineConfig` from a YAML file."""
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Pipeline config file not found: {yaml_path}")
    with open(yaml_path, "r") as f:
        config_dict = yaml.safe_load(f) or {}
    return load_pipeline_config_from_dict(config_dict)


def save_pipeline_config_to_yaml(config: PipelineConfig, yaml_path: str) -> None:
    """Save a `PipelineConfig` to YAML, preserving the nested structure.

    ``diarization.hf_token`` is masked as ``_REDACTED`` so a live token never
    lands in a saved config file. On reload the loader drops the placeholder
    and re-resolves the token from ``$HF_TOKEN`` (see
    ``load_pipeline_config_from_dict``); a redacted config therefore round-trips
    to the env token, never to the literal string.
    """
    yaml_path = Path(yaml_path)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    data = redact_config_snapshot(asdict(config))
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
