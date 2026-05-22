"""Configuration for the ASR pipeline.

Pattern mirrors `polsess_separation/config.py`: nested dataclasses,
`__post_init__` validation, YAML round-trip via `asdict() + yaml.dump()`.

The configuration is intentionally self-contained — no dependency on the
parent project's `Config` — so the package can be lifted into CLARIN with
only the one separator-loading seam in `stages/separation.py` to edit.
"""

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import yaml


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
    Stage 4 derives them on the fly by subtracting `ctx.overlap_regions`
    from `ctx.diarization.segments_df` per speaker.

    All thresholds in seconds.
    """

    enabled: bool = True
    min_overlap_dur: float = 0.20    # drop overlaps shorter than this
    merge_gap: float = 0.50          # merge overlap regions closer than this


@dataclass
class EnhancementConfig:
    """Stage 3a: speech enhancement on solo regions (MP-SENet, vendored)."""

    enabled: bool = True
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
    classes (one function with branches per knob). Defaults reproduce the
    POC notebook's behaviour.
    """

    enabled: bool = True
    checkpoint_path: str = (
        "checkpoints/sepformer/SB/sepformer_SB_best-64k_baseline_posenc_v25/"
        "sepformer_SB_best.pt"
    )
    separator_sample_rate: int = 8_000   # SR the separator was trained at
    # Audio duration (seconds) the separator was trained on. Used as the
    # target padded-window size for `context_window_mode != "none"` and as
    # the chunk size for long-overlap overlap-add.
    training_chunk_length_s: float = 4.0

    # Context-window strategy: how much audio around the overlap to feed the
    # separator. POC behaviour is `none` (no extra context). The default below
    # picks the more-considered `snap_to_vad` choice from the design discussion.
    context_window_mode: str = "snap_to_vad"   # "snap_to_vad" | "fixed_pad" | "none"
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
    seam_mode: str = "zero_crossing"  # "zero_crossing" | "overlap_boundary" | "snap_to_silence"
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
    volume_normalization: str = "sum_equals_mix"  # "sum_equals_mix" | "match_solo" | "none"

    # VAD applied to each separator output to suppress residuals.
    # `vad_threshold` is the "definitely speech" upper bound. Frames above it
    # always become part of the speech mask.
    vad_threshold: float = 0.50
    # Schmitt-trigger style lower threshold. Frames between
    # `vad_soft_threshold` and `vad_threshold` count as speech *only* if they
    # extend a chain that touches a frame above `vad_threshold` (propagated
    # both forward and backward). This captures tails/onsets where silero is
    # less confident — common on neural-separator outputs. Set to a value
    # >= `vad_threshold` to disable (mask becomes a strict threshold).
    vad_soft_threshold: float = 0.20
    # Fixed positional dilation in 32 ms frames applied on top of the
    # threshold mask. Extends each speech run by `vad_attack_frames` before
    # its onset and `vad_release_frames` after its offset, regardless of
    # frame probability. Set both to 0 to disable.
    vad_attack_frames: int = 1
    vad_release_frames: int = 1
    vad_mode: str = "silero"                    # "silero" | "energy" (energy = NotImplemented)


@dataclass
class AssemblyConfig:
    """Stage 4: per-speaker stream assembly + timestamp map."""

    enabled: bool = True
    min_solo_for_anchor_s: float = 3.0
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
    """Stage 5: Whisper ASR per assembled stream."""

    enabled: bool = True
    model_name: str = "large-v3"
    language: str = "pl"
    initial_prompt: str = "Rozmowa po polsku."
    word_timestamps: bool = True


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
    assembly: AssemblyConfig = field(default_factory=AssemblyConfig)
    transcription: TranscriptionConfig = field(default_factory=TranscriptionConfig)

    def __post_init__(self):
        # Validate enum-string knobs early so misconfiguration is loud.
        if self.separation.context_window_mode not in (
            "snap_to_vad", "fixed_pad", "none",
        ):
            raise ValueError(
                f"Invalid context_window_mode: {self.separation.context_window_mode!r}"
            )
        if self.separation.seam_mode not in (
            "zero_crossing", "overlap_boundary", "snap_to_silence",
        ):
            raise ValueError(f"Invalid seam_mode: {self.separation.seam_mode!r}")
        if self.separation.volume_normalization not in (
            "sum_equals_mix", "match_solo", "none",
        ):
            raise ValueError(
                f"Invalid volume_normalization: {self.separation.volume_normalization!r}"
            )
        if self.separation.vad_mode not in ("silero", "energy"):
            raise ValueError(f"Invalid vad_mode: {self.separation.vad_mode!r}")
        if self.assembly.output_mode not in ("shortened", "full_length"):
            raise ValueError(f"Invalid output_mode: {self.assembly.output_mode!r}")

        if self.spill_intermediate and self.artifact_dir is None:
            raise ValueError(
                "spill_intermediate is True but artifact_dir is None — "
                "set artifact_dir (e.g. via --output) or disable spilling."
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
        ("assembly", AssemblyConfig),
        ("transcription", TranscriptionConfig),
    ):
        sub_dict = config_dict.pop(key, None)
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
    """Save a `PipelineConfig` to YAML, preserving the nested structure."""
    yaml_path = Path(yaml_path)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w") as f:
        yaml.dump(asdict(config), f, default_flow_style=False, sort_keys=False)
