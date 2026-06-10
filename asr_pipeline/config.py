"""Configuration for the ASR pipeline.

Pattern mirrors `polsess_separation/config.py`: nested dataclasses,
`__post_init__` validation, YAML round-trip via `asdict() + yaml.dump()`.

The configuration is intentionally self-contained — no dependency on the
parent project's `Config` — so the package can be lifted into CLARIN with
only the one separator-loading seam in `stages/separation.py` to edit.
"""

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

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

    Multiple backends are supported. The default is the vendored MP-SENet
    (VoiceBank+DEMAND training distribution, narrow). ClearerVoice-Studio
    alternatives are trained on broader DNS-Challenge data and handle
    reverberant / out-of-distribution input more robustly.
    """

    enabled: bool = True
    # Backend selector:
    #   - "mpsenet": vendored MP-SENet checkpoint (16k, narrow training dist)
    #   - "frcrn_se_16k": ClearerVoice FRCRN, DNS-2020 winner, native 16k
    #   - "mossformer_gan_se_16k": ClearerVoice MossFormer + GAN losses, 16k
    #   - "mossformer2_se_48k": ClearerVoice MossFormer2, native 48k
    #     (pipeline at 16k → upsample/downsample handled internally by the
    #     backend; expect modest extra compute)
    # The non-MPSENet backends ignore the `checkpoint_path` / `config_path`
    # fields below — they self-download to their respective caches on
    # first use.
    backend: str = "mpsenet"
    # Path to the MP-SENet generator checkpoint (PyTorch state dict containing
    # the 'generator' key). The vendored model code reads its hyperparameters
    # from a `config.json` placed next to the checkpoint.
    checkpoint_path: str = field(
        default_factory=lambda: os.getenv(
            "MPSENET_CHECKPOINT",
            "/home/user/MP-SENet/best_ckpt/g_best_vb",
        )
    )
    # Path to MP-SENet's `config.json` (architecture hyperparameters).
    config_path: str = field(
        default_factory=lambda: os.getenv(
            "MPSENET_CONFIG",
            "/home/user/MP-SENet/config.json",
        )
    )
    # MP-SENet's time-axis attention is O(T^2), so feeding a long
    # recording to it in one shot blows up GPU memory. Recordings longer
    # than this duration are processed via Hann overlap-add — chunks of
    # this size with a 50% hop (canonical COLA: window sum is 1.0 in the
    # interior, head/tail divided by actual weights). MP-SENet was
    # trained on 2 s crops; 8 s chunks were verified by ear on long
    # Polish recordings.
    max_segment_length_s: float = 8.0


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
        "checkpoints/sepformer/SB/128_run/sepformer_SB_best_128k_e41.pt"
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
    # ECAPA only needs a few seconds of audio for a stable speaker embedding.
    # On a long recording (e.g. 15 min), the per-speaker solo concat can grow
    # to hundreds of seconds; feeding that to ECAPA in a single forward either
    # OOMs the GPU or stalls the kernel. We cap the concat at this length
    # (taking a uniformly-strided sample so we don't bias toward the start).
    # Set to None to disable the cap.
    anchor_max_duration_s: Optional[float] = 30.0
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
    # louder than MP-SENet's solo audio. Solo events are not touched, so the
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
    # OpenAI short names (``large-v3``, ``large-v2``) or any HF model id
    # parseable by faster-whisper when backend == ``whisperx``. Ignored
    # checkpoints for the ``whisper`` backend must be canonical OpenAI names.
    model_name: str = "large-v3"
    language: str = "pl"
    initial_prompt: str = "Rozmowa po polsku."
    word_timestamps: bool = True
    # WhisperX-only knobs (ignored when ``backend != whisperx``):
    # the wav2vec2 model used for forced alignment.
    align_model_name: str = "jonatasgrosman/wav2vec2-large-xlsr-53-polish"
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
    # enhancement conv stack (e.g. FRCRN / MP-SENet) is otherwise the pipeline's
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
                ("mpsenet", "frcrn_se_16k", "mossformer_gan_se_16k", "mossformer2_se_48k"))
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
