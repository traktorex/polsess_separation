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
    # RelabelStage (B/B+) carries its own hf_token field (for a pyannote-format
    # embedder id; unused for the custom "ecapa2" name). Mask it unconditionally
    # — cheap insurance so a live token never lands in a saved snapshot even when
    # relabel is enabled with a pyannote embedder.
    relabel = snap.get("relabel")
    if isinstance(relabel, dict) and relabel.get("hf_token"):
        relabel["hf_token"] = _REDACTED
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
class FusionConfig:
    """Disagreement-aware diarization fusion (SECOND_PASS_PLAN.md option 3).

    OFF (default) = byte-identical no-op; the second pass never runs.

    When enabled, a SECOND pyannote diarization runs on the ENHANCED audio
    (`ctx.enhanced_full`) and is FUSED with the raw pass-1 result by the
    asymmetry rule (plan §0): enhancement HELPS identity but HURTS presence (a
    single-output SE model suppresses the quieter speaker → an enhanced
    re-diarization under-detects overlaps / drops quiet turns). So PRESENCE
    (segment boundaries, overlap timeline, speaker count) stays with the RAW
    pass-1 result, and pass 2 may override only the IDENTITY label of a region
    pass 1 already calls single-speaker — and only on confident disagreement.

    Implemented as a separate post-enhancement stage (`FusionDiarizationStage`)
    rather than a flag inside DiarizationStage, because pass 2 needs the enhanced
    audio that does not exist at stage 1 (plan §7.9). The stage REPLACES the
    `speaker` column of `ctx.diarization.segments_df` in place; everything else
    (overlaps_df, boundaries, the label set) is untouched.
    """

    enabled: bool = False
    # Embedder for the pass-2 (identity) diarization. Same custom-name /
    # pyannote-id contract as DiarizationConfig.embedding (None = stock 3.1).
    embedding: Optional[str] = "ecapa2"
    # Override a solo region's label with pass 2's only when pass 2 assigns at
    # least this fraction of the region's duration to a SINGLE speaker (its
    # confidence). In (0, 1]. Higher = stricter (fewer overrides).
    confidence_min: float = 0.75
    # Minimum solo-region duration (s) to even consider overriding. Below this a
    # region's pass-2 label is too noisy to trust; keep pass 1. >= 0.
    min_region_s: float = 0.5


@dataclass
class DiarizationConfig:
    """Stage 1: pyannote speaker diarization."""

    enabled: bool = True
    model_id: str = "pyannote/speaker-diarization-3.1"
    hf_token: Optional[str] = field(
        default_factory=lambda: os.getenv("HF_TOKEN", None)
    )
    num_speakers: int = 2
    # pyannote front-end hyperparameters, applied via Pipeline.instantiate() at
    # load (DiarizationStage.load). Defaults = the shipped speaker-diarization-3.1
    # config.yaml values, so leaving them untouched is byte-identical to stock
    # pyannote; they exist to be SWEPT. Only segmentation.min_duration_off and
    # clustering.min_cluster_size are effective here: num_speakers=2 fixes the
    # agglomerative cluster COUNT, so clustering.threshold is INERT (kept for
    # completeness / if num_speakers is ever relaxed).
    segmentation_min_duration_off: float = 0.0    # fill intra-turn pauses <= this (s)
    clustering_threshold: float = 0.7045654963945799   # INERT under num_speakers=2
    clustering_min_cluster_size: int = 12
    # Agglomerative linkage method, applied in the embedding-swap /
    # reconstructed-3.1 path (pyannote AgglomerativeClustering). Only effective
    # when the 3.1 schema is instantiated (embedding != None or a 3.x model_id).
    # Linkage decides which short segments cluster together = exactly where
    # short-segment mislabels are decided. Default "centroid" = stock 3.1
    # (byte-identical). One of pyannote's accepted scipy linkages:
    # {"average", "centroid", "complete", "median", "single", "ward", "weighted"}.
    clustering_method: str = "centroid"
    # Speaker-embedding model swap. None = the stock model_id pipeline via
    # `from_pretrained` (byte-identical to shipped). Any other value triggers
    # RECONSTRUCTION of a 3.1-equivalent SpeakerDiarization (segmentation-3.0 +
    # AgglomerativeClustering + exclude_overlap) with the chosen embedder — the
    # only difference from stock 3.1. Two embedder families are accepted:
    #   - a pyannote-format model id/path (e.g.
    #     "eek/wespeaker-voxceleb-resnet293-LM"), handled by pyannote's own
    #     embedding factory; OR
    #   - a CUSTOM name — "ecapa2" (Jenthe/ECAPA2 TorchScript) or "eres2netv2"
    #     (3D-Speaker ERes2NetV2 via ModelScope) — which pyannote's factory does
    #     NOT accept, so DiarizationStage.load injects a wrapper from
    #     stages/custom_embeddings.py in place of pipeline._embedding (see
    #     CUSTOM_EMBEDDING_NAMES there).
    # The embedding decides clustering quality (see EMBEDDING_RESEARCH.md); this is
    # the lever for the db15fc57-style short-segment mislabels. Default stays None
    # (stock baseline) so the experimental baseline / f_oa03 (which omit
    # `embedding`) remain stock 3.1 and stay comparable to on-disk outputs; ECAPA2
    # is adopted ONLY in the shipped configs/sweep_best_e31.yaml.
    embedding: Optional[str] = None
    # Disagreement-aware fusion (option 3). Default OFF (FusionConfig.enabled =
    # False) → no second pass. See FusionConfig. Lives under diarization because
    # it is a property of the diarization output, but is RUN by the separate
    # post-enhancement FusionDiarizationStage (which needs enhanced audio).
    fusion: "FusionConfig" = field(default_factory=FusionConfig)


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
    # librosa/soxr resampling filter applied to every solo region on the
    # 16k->native->16k round-trip around the enhancer (the `res_type` arg in
    # stages/enhancement.py). Sets *what* spectrum reaches the enhancer and, via
    # the OA blend, WhisperX -- directly upstream of `observation_mix_ratio`.
    # Only the enhancement stage uses this; the separator's torchaudio resample
    # is unaffected (and deliberately not unified -- the two are not
    # bit-identical). Default "soxr_hq" = current behaviour (byte-identical),
    # inherited from the batch-script lineage the 48 kHz checkpoint was
    # characterised against. One of {"soxr_hq", "soxr_vhq", "kaiser_best"}.
    resample_quality: str = "soxr_hq"


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
        "checkpoints/mossformer2/SB/mossformer2_matched_128k_final_42_e46/mossformer2_SB_best_e46.pt"
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
    # When `seam_mode == "snap_to_silence"`: the VAD-mask cutoff below which a
    # frame counts as silence while extending the emit boundary outward
    # (`_extend_{start,end}_to_silence`). This is the companion to the swept VAD
    # thresholds: those gate the audio mask; this defines what "silence" means
    # when growing the seam, i.e. how far an emit region grows into adjacent solo
    # audio = how much speech is double-counted vs dropped at the seam. NOTE this
    # is a separate mask from `vad_threshold`/`vad_soft_threshold`. Default 0.5 =
    # current behaviour (byte-identical). Strictly in (0, 1).
    seam_silence_threshold: float = 0.5

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
    # Embedder used for the per-speaker anchor (`_compute_anchors`) AND the
    # per-overlap stream embedding (`_assign_overlaps`):
    #   - "ecapa1" (default): SpeechBrain `spkrec-ecapa-voxceleb`, the current
    #     POC embedder. Byte-identical to the pre-knob behaviour.
    #   - "ecapa2": the Jenthe/ECAPA2 TorchScript embedder via
    #     `build_custom_embedding` (the same custom wrapper the diarization stage
    #     uses). Option 4 of the 2nd-pass plan: does a stronger anchor embedder
    #     ALONE (no 2nd pass) sharpen overlap attribution? `_ecapa_embed`
    #     dispatches on this so both encoders share one call site. The assembly
    #     0.25 s pad floor is >= the wrapper's ~25 ms floor, so the custom path
    #     never under-feeds. Independent of the relabel stage's own ECAPA2.
    anchor_embedding: str = "ecapa1"           # "ecapa1" (SpeechBrain) | "ecapa2"
    min_solo_for_anchor_s: float = 3.0
    # Minimum solo duration (s) a speaker needs for an ECAPA *anchor*. Below this
    # `_compute_anchors` leaves the anchor None and ALL that speaker's overlaps
    # fall to fixed positional assignment — the real attribution fallback
    # trigger (distinct from `min_solo_for_anchor_s`, which only sets the
    # diagnostic `weak_anchor` flag). Also the zero-pad floor inside
    # `_ecapa_embed` for every anchor / overlap embedding. Default 0.25 =
    # current behaviour (byte-identical). >= 0.
    anchor_min_duration_s: float = 0.25
    # Minimum length (s) of a separated overlap stream for the per-overlap ECAPA
    # decision to run at all (`_assign_overlaps`). Shorter overlaps fall to fixed
    # (positional) assignment rather than trusting a noisy cosine on a fraction
    # of a syllable. Gates how many overlaps reach ANY assignment strategy.
    # Default 0.1 = current behaviour (byte-identical). >= 0.
    overlap_min_duration_s: float = 0.1
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
    # Margin-gated carry-forward prior for per-overlap ECAPA pairing
    # (`_assign_overlaps`). The per-overlap pairing picks argmax(straight,
    # swapped) summed cosine independently per overlap, but ~a third of overlaps
    # are sub-0.5 s where ECAPA is unreliable and the two cosines sit near a tie.
    # When this knob is > 0 and `abs(straight - swapped) < overlap_assign_min_margin`,
    # the overlap is treated as ambiguous: instead of the noisy argmax it inherits
    # the *carry-forward prior* — the pairing of the last confident (above-margin)
    # ECAPA decision. 0.0 = off → every ECAPA decision clears the (zero) margin,
    # so the prior never fires and behaviour is the pure-argmax current pipeline
    # (byte-identical baseline). A hypothesis to sweep, not a behaviour change.
    # Must lie in [0, 1) — the summed-cosine gap spans [0, 2], but a margin >= 1
    # would gate even decisive decisions, so the useful range is small.
    overlap_assign_min_margin: float = 0.0
    # Per-overlap speaker-assignment strategy (`_assign_overlaps`):
    #   "ecapa_argmax"        -> the POC default: argmax(straight, swapped) summed
    #                            cosine to the *global* solo anchors, with the
    #                            optional `overlap_assign_min_margin` carry-forward.
    #   "continuity_tiebreak" -> argmax as above, but a near-tie (gap <
    #                            `continuity_tiebreak_margin`) is broken by
    #                            *local* anchors built from each speaker's solo
    #                            audio within `continuity_window_s` of the overlap
    #                            — the temporally-adjacent ("speech continuity")
    #                            voice, more reliable than the global anchor on a
    #                            short ambiguous overlap. Stateless (no
    #                            carry-forward chain). Ignores
    #                            `overlap_assign_min_margin`.
    #   "consensus_2means"    -> a global pre-pass decides every overlap jointly
    #                            via constrained 2-means over all overlap
    #                            embeddings + the solo anchors, so the consensus
    #                            of the majority can overrule a lone confident-
    #                            but-wrong per-overlap decision (the lever the two
    #                            per-overlap modes above cannot reach).
    overlap_assignment: str = "ecapa_argmax"   # ecapa_argmax | continuity_tiebreak | consensus_2means
    # Near-tie threshold (τ) for "continuity_tiebreak": when the global-anchor
    # straight/swapped summed-cosine gap is < this, re-decide with local anchors.
    # 0.0 = off (no overlap is ever a near-tie) → byte-identical to ecapa_argmax,
    # so this is the swept hypothesis knob. Same [0, 1) range rationale as
    # `overlap_assign_min_margin`.
    continuity_tiebreak_margin: float = 0.0
    # Half-window (seconds) each side of an overlap from which the local
    # continuity anchor is built. A speaker with no solo audio in the window
    # yields no local anchor → that overlap falls back to the global argmax.
    continuity_window_s: float = 10.0
    # Overlap-stream routing MODE — orthogonal to `overlap_assignment` above,
    # which selects the per-overlap strategy:
    #   "anchor_argmax" (default) -> current behaviour: each overlap's
    #     straight/swapped is decided independently, governed by
    #     `overlap_assignment`. Byte-identical to the pre-knob pipeline.
    #   "cluster2" -> a global pre-pass (`_cluster2_pairings`) collects EVERY
    #     separated overlap-stream embedding in the fragment, runs cosine 2-means
    #     seeded from the two solo anchors (deterministic, no RNG), maps the two
    #     clusters back onto stream A/B by centroid-to-anchor similarity, and
    #     emits a per-overlap pairing. All streams are clustered jointly, so a run
    #     can no longer defect one overlap at a time to a weak anchor; and an
    #     overlap whose two streams land in the SAME cluster (a separation
    #     failure) is left undecided and falls through to the `overlap_assignment`
    #     ladder rather than being forced into a guessed pairing. NOTE (measured
    #     in-code): with the anchors used as BOTH the 2-means seeds and the
    #     cluster→stream mapping, cluster2 is algebraically ~= per-overlap argmax
    #     on symmetric (A,B) overlap pairs — its only distinct behaviour is the
    #     same-cluster fall-through — so like `consensus_2means` it is expected to
    #     be a near-no-op; it exists as a documentable sweep lever. Overlaps the
    #     B+ relabel handoff (`ctx.overlap_speaker_assignment`) already covers
    #     still win (a strictly stronger global decision); cluster2 fills the rest.
    assignment_mode: str = "anchor_argmax"   # "anchor_argmax" | "cluster2"
    # Onset boundary pad for SOLO pieces (seconds). pyannote turn starts lag true
    # speech onsets slightly, and assembly slices exactly at the diarization
    # boundary — so first phonemes get shaved ("szefie" audible as "efie") or
    # onset slivers split into / duplicated in the other stream (ear pass,
    # 2026-07). When > 0, each solo piece's START is extended earlier by up to
    # this much at audio-extraction time (`_pad_solo_onsets`), hard-clamped so
    # the pad can never enter an overlap region (that audio contains BOTH
    # speakers), never reach into ANY adjacent piece (either stream — the other
    # speaker's solo span right before this piece would inject their voice), and
    # never go below 0. Onset side only — piece ENDS are never padded. The
    # anchors / continuity intervals keep the unpadded boundaries (the pad is an
    # extraction detail, not a diarization change). 0.0 = off (byte-identical
    # baseline). Finite, >= 0.
    solo_onset_pad_s: float = 0.0


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
    # Peak-amplitude floor below which an assembled stream is treated as silent
    # and skipped (empty transcript, Whisper never called). The assembler emits
    # all-zeros sentinels for no-event speakers; without this gate Whisper
    # hallucinates phantom Polish on pure silence (scored as L3 insertions).
    # Default 1e-4 mirrors the silence floor in eval/layer2.py (duplicated, not
    # imported: stages must not depend on eval) = current behaviour
    # (byte-identical). 0.0 disables the gate (only literal all-zero is skipped,
    # via the length check); higher values zero out quieter real speakers
    # (deletions). A deletion↔insertion trade — the swept hypothesis knob. >= 0.
    silence_floor: float = 1e-4

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
    # Exponential length penalty (Google NMT, alpha). Both backends accept it,
    # so unlike the WhisperX-only knobs below it routes to ``_WhisperBackend``
    # too — but the no-op value differs per backend, so it is forwarded only
    # when non-default (see below). faster-whisper / WhisperX default = 1
    # (asr.py ``default_asr_options["length_penalty"]`` = the
    # ``WhisperModel.transcribe`` signature default), threaded into the WhisperX
    # ``asr_options`` at 1.0 = no-op. openai-whisper's default is ``None`` (plain
    # length normalisation), and a value of 1.0 there is NOT identical to None
    # (``((5+len)/6)`` vs ``len`` in MaximumLikelihoodRanker), so the
    # ``whisper`` backend forwards this only when it differs from 1.0 — keeping
    # the baseline byte-identical for both backends. openai-whisper additionally
    # requires the value in [0, 1]; faster-whisper has no such cap. Must be > 0.
    length_penalty: float = 1.0

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

    # Suppress numeric-symbol tokens so numbers come out spelled in words.
    # WhisperX-only: ``load_model`` pops ``suppress_numerals`` out of
    # ``default_asr_options`` (asr.py ~L404) and hands it to
    # ``FasterWhisperPipeline`` — it is NOT a faster-whisper
    # ``TranscriptionOptions`` field, so it is wired into the ``asr_options``
    # dict like the other decode knobs. WhisperX default = False (asr.py
    # ``default_asr_options["suppress_numerals"]``) = current behaviour.
    # openai-whisper has no equivalent on its decode surface, so a non-default
    # value with ``backend == "whisper"`` is a loud error (see
    # ``_WhisperBackend._reject_unsupported_knobs``).
    suppress_numerals: bool = False
    # WhisperX internal-VAD onset / offset probabilities (Schmitt-trigger style:
    # a frame enters speech above ``vad_onset`` and leaves below ``vad_offset``).
    # WhisperX-only: passed in the ``vad_options`` dict to ``load_model``, which
    # merges them over its own ``default_vad_options`` (asr.py ~L409-412). The
    # pipeline currently passes no ``vad_options``, so the defaults below
    # reproduce WhisperX's exactly — ``vad_onset=0.500``, ``vad_offset=0.363``
    # = byte-identical baseline. Lowering ``vad_offset`` keeps trailing speech
    # the VAD would otherwise clip; both must lie in (0, 1). openai-whisper does
    # its own internal windowing with no such VAD, so a non-default value with
    # ``backend == "whisper"`` is a loud error.
    vad_onset: float = 0.500
    vad_offset: float = 0.363

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


@dataclass
class RelabelConfig:
    """2nd-pass identity re-clustering on clean audio (opt-in). OFF = no-op.

    Runs as a model-bearing stage between post_separation_processing (3c) and
    assembly. Re-embeds the pass-1 single-speaker spans with a stronger embedder
    (ECAPA2 by default), splits them into exactly 2 by cosine 2-means seeded from
    the pass-1 speaker centroids, aligns the two clusters back onto the pass-1
    labels by max-duration overlap, and
    overwrites `segments_df["speaker"]`. Identity-only: presence
    (when/who-is-active) stays with the raw pass-1 diarization (see
    SECOND_PASS_PLAN.md §0). Two modes:

      - "solos"  (B): cluster pass-1 SOLO segments only (overlap-excluded).
      - "global" (B+): cluster the solos PLUS the VAD-gated separated overlap
        streams (`s1_gated`/`s2_gated`) jointly, and additionally emit
        `ctx.overlap_speaker_assignment` (the per-overlap straight/swapped
        decision) which assembly consumes via its consensus-injection seam.

    Default `enabled=False` → the orchestrator skips the stage entirely
    (byte-identical no-op; ECAPA2 is never loaded).
    """

    enabled: bool = False
    # "solos" (B) | "global" (B+: solos + separated overlap streams).
    source: str = "solos"
    # Custom embedder name (build_custom_embedding) or pyannote-format model id.
    embedding: str = "ecapa2"
    # Identity-audio source for the SOLO spans: "enhanced" (ctx.enhanced_full,
    # the cleaner identity signal — needs enhancement.enabled) or "raw"
    # (ctx.audio, the enhancement-isolating control).
    audio_source: str = "enhanced"
    # Subtract ctx.overlap_regions from each solo span before embedding (matches
    # assembly's solo derivation + pyannote's embedding_exclude_overlap intent).
    exclude_overlap: bool = True
    # Weight each point's contribution to its SEED centroid (the 2-means start)
    # by segment duration. Default OFF: the db15fc57 diagnosis is that long turns
    # DOMINATED the pass-1 centroid and buried a 2.5 s segment, so weighting the
    # seed centroids by duration re-creates the very bias this pass exists to fix
    # (SECOND_PASS_PLAN.md §3.2). Kept as an A/B knob only; duration weighting is
    # used in ALIGNMENT regardless (where long anchors SHOULD pin identity).
    duration_weighted: bool = False
    # Run-level (contiguous-run) relabel pass, applied AFTER the global solo
    # relabel above. OFF by default → byte-identical no-op. When True, the usable
    # solo segments are ordered by time and grouped into maximal contiguous
    # SAME-stream runs; a whole run flips to the other stream when its mean
    # embedding is closer to the other stream's centroid than to its own by more
    # than `run_margin` (cosine). This targets the class-A chunk swap the GLOBAL
    # relabel provably cannot repair — a single global A<->B flip is cpWER-free,
    # so it never fixes a stream that is only PARTIALLY mixed. Deterministic:
    # centroids are computed ONCE from the post-global-relabel assignment and runs
    # are processed in a fixed time order (no centroid drift, no RNG). Never flips
    # a run that IS its entire stream (a whole-stream flip is just a free global
    # swap, and this guards the degenerate flip-everything case). Solo-only: like
    # the global relabel it never touches the overlap streams / the B+ overlap
    # handoff (consistent with `exclude_overlap`).
    run_level: bool = False
    # Cosine margin the other-stream centroid must beat the own-stream centroid by
    # before a whole run is flipped (only consulted when `run_level=True`). Cosine
    # on unit ECAPA2 embeddings lies in [-1, 1], so the gap is in [-2, 2]; 0.05
    # requires a clear (not marginal) pull to the other speaker. >= 0.
    run_margin: float = 0.05
    # hf_token for a pyannote-format embedder id (unused for the custom "ecapa2"
    # name, whose loader hits the HF hub directly). Masked in saved snapshots.
    hf_token: Optional[str] = field(
        default_factory=lambda: os.getenv("HF_TOKEN", None)
    )


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
    # 2nd-pass identity re-clustering (B / B+), placed between 3c and assembly.
    # Default OFF → orchestrator skips it (byte-identical no-op).
    relabel: RelabelConfig = field(default_factory=RelabelConfig)
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
        _one_of(self.enhancement.resample_quality, "enhancement.resample_quality",
                ("soxr_hq", "soxr_vhq", "kaiser_best"))
        _one_of(self.post_separation_processing.backend,
                "post_separation_processing.backend",
                ("naive", "ap_bwe", "flowhigh"))
        _one_of(self.transcription.backend, "transcription.backend",
                ("whisper", "whisperx", "coherex"))
        _one_of(self.assembly.anchor_embedding, "assembly.anchor_embedding",
                ("ecapa1", "ecapa2"))
        _one_of(self.diarization.clustering_method, "diarization.clustering_method",
                ("average", "centroid", "complete", "median", "single",
                 "ward", "weighted"))
        _one_of(self.assembly.assignment_mode, "assembly.assignment_mode",
                ("anchor_argmax", "cluster2"))
        _one_of(self.relabel.source, "relabel.source", ("solos", "global"))
        _one_of(self.relabel.audio_source, "relabel.audio_source",
                ("enhanced", "raw"))
        # run_margin is a cosine gap floor (only used when run_level=True); a
        # negative value would flip every run. 0 = flip on any improvement.
        if not math.isfinite(self.relabel.run_margin) or self.relabel.run_margin < 0:
            raise ValueError(
                f"relabel.run_margin must be a finite value >= 0 (cosine margin; "
                f"only used when run_level=True), got {self.relabel.run_margin}"
            )
        # Relabel cross-checks (fail loud at config time, SCOPE §4.1 — never a
        # silent raw fallback / quiet downgrade at runtime):
        if (self.relabel.enabled and self.relabel.audio_source == "enhanced"
                and not self.enhancement.enabled):
            raise ValueError(
                "relabel.audio_source='enhanced' requires enhancement.enabled "
                "(no enhanced_full to re-cluster); use audio_source='raw' or "
                "disable relabel."
            )
        if (self.relabel.enabled and self.relabel.source == "global"
                and not self.separation.enabled):
            raise ValueError(
                "relabel.source='global' (B+) requires separation.enabled "
                "(no overlap_separated streams to cluster)."
            )
        # Fusion cross-checks (option 3). Pass 2 re-diarizes the ENHANCED audio
        # for identity → fail loud at config time if there is no enhanced audio
        # to re-diarize (SCOPE §4.1 — never a silent fallback to a single pass).
        fcfg = self.diarization.fusion
        if fcfg.enabled and not self.enhancement.enabled:
            raise ValueError(
                "diarization.fusion.enabled requires enhancement.enabled "
                "(pass 2 re-diarizes the enhanced audio for identity)."
            )
        if not math.isfinite(fcfg.confidence_min) or not (
            0.0 < fcfg.confidence_min <= 1.0
        ):
            raise ValueError(
                f"diarization.fusion.confidence_min must be in (0, 1], "
                f"got {fcfg.confidence_min}"
            )
        if not math.isfinite(fcfg.min_region_s) or fcfg.min_region_s < 0:
            raise ValueError(
                f"diarization.fusion.min_region_s must be >= 0, "
                f"got {fcfg.min_region_s}"
            )

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
        sst = self.separation.seam_silence_threshold
        if not math.isfinite(sst) or not (0.0 < sst < 1.0):
            raise ValueError(
                f"separation.seam_silence_threshold must be in (0, 1) "
                f"(VAD silence cutoff for snap_to_silence; 0.5 = default), "
                f"got {sst}"
            )

        if self.post_separation_processing.flowhigh_input_sr <= 0:
            raise ValueError(
                f"flowhigh_input_sr must be positive, got "
                f"{self.post_separation_processing.flowhigh_input_sr}"
            )

        # --- Assembly numeric knobs ---
        # A negative value here crashes deep in Stage 6 (`np.zeros(negative)` /
        # broadcast errors), not at config load — so fail loud and early,
        # naming the offending knob. All are seconds/ms durations: zero is a
        # valid disable (no gap / no fade / no cap), so the floor is >= 0.
        acfg = self.assembly
        for knob in (
            "silence_separator_s",
            "crossfade_ms",
            "edge_fade_ms",
            "min_solo_for_anchor_s",
            "anchor_min_duration_s",
            "overlap_min_duration_s",
        ):
            value = getattr(acfg, knob)
            if value < 0:
                raise ValueError(
                    f"assembly.{knob} must be >= 0, got {value}"
                )
        # anchor_max_duration_s is Optional (None = no cap); a non-None value
        # must be positive (a zero/negative cap would empty the anchor audio).
        if acfg.anchor_max_duration_s is not None and acfg.anchor_max_duration_s <= 0:
            raise ValueError(
                f"assembly.anchor_max_duration_s must be None (no cap) or "
                f"positive, got {acfg.anchor_max_duration_s}"
            )
        # overlap_assign_min_margin gates the carry-forward prior. The summed
        # cosine gap spans [0, 2]; 0 = off (current behaviour), and a margin
        # >= 1 would gate decisive decisions, so the valid range is [0, 1).
        if not math.isfinite(acfg.overlap_assign_min_margin) or not (
            0.0 <= acfg.overlap_assign_min_margin < 1.0
        ):
            raise ValueError(
                f"assembly.overlap_assign_min_margin must be in [0, 1) "
                f"(0 = off), got {acfg.overlap_assign_min_margin}"
            )
        # Per-overlap assignment strategy + the continuity tie-break knobs.
        valid_assign = {"ecapa_argmax", "continuity_tiebreak", "consensus_2means"}
        if acfg.overlap_assignment not in valid_assign:
            raise ValueError(
                f"assembly.overlap_assignment must be one of {sorted(valid_assign)}, "
                f"got {acfg.overlap_assignment!r}"
            )
        if not math.isfinite(acfg.continuity_tiebreak_margin) or not (
            0.0 <= acfg.continuity_tiebreak_margin < 1.0
        ):
            raise ValueError(
                f"assembly.continuity_tiebreak_margin must be in [0, 1) "
                f"(0 = off), got {acfg.continuity_tiebreak_margin}"
            )
        if not math.isfinite(acfg.continuity_window_s) or acfg.continuity_window_s <= 0:
            raise ValueError(
                f"assembly.continuity_window_s must be a positive finite number, "
                f"got {acfg.continuity_window_s}"
            )
        # solo_onset_pad_s needs its own guard (not the >= 0 loop above): NaN
        # passes a bare `< 0` check and would silently disable every clamp
        # comparison downstream.
        if not math.isfinite(acfg.solo_onset_pad_s) or acfg.solo_onset_pad_s < 0:
            raise ValueError(
                f"assembly.solo_onset_pad_s must be a finite value >= 0 "
                f"(0 = off), got {acfg.solo_onset_pad_s}"
            )

        # --- Diarization front-end knobs (pyannote instantiate params) ---
        dcfg = self.diarization
        if not math.isfinite(dcfg.segmentation_min_duration_off) or \
                dcfg.segmentation_min_duration_off < 0:
            raise ValueError(
                f"diarization.segmentation_min_duration_off must be >= 0, "
                f"got {dcfg.segmentation_min_duration_off}"
            )
        if not math.isfinite(dcfg.clustering_threshold) or not (
            0.0 <= dcfg.clustering_threshold <= 2.0
        ):
            raise ValueError(
                f"diarization.clustering_threshold must be in [0, 2] (cosine; "
                f"INERT under num_speakers=2), got {dcfg.clustering_threshold}"
            )
        if dcfg.clustering_min_cluster_size < 1:
            raise ValueError(
                f"diarization.clustering_min_cluster_size must be >= 1, "
                f"got {dcfg.clustering_min_cluster_size}"
            )

        omr = self.enhancement.observation_mix_ratio
        if not math.isfinite(omr) or not (0.0 <= omr <= 1.0):
            raise ValueError(
                f"enhancement.observation_mix_ratio must be a finite value in "
                f"[0, 1] (convex dry-wet weight; 0 = pure enhanced), got {omr}"
            )

        # --- Transcription decode knobs ---
        tcfg = self.transcription
        if not math.isfinite(tcfg.silence_floor) or tcfg.silence_floor < 0:
            raise ValueError(
                f"transcription.silence_floor must be a finite value >= 0 "
                f"(0 = gate off; 1e-4 = default), got {tcfg.silence_floor}"
            )
        if tcfg.beam_size < 1:
            raise ValueError(
                f"transcription.beam_size must be >= 1, got {tcfg.beam_size}"
            )
        if tcfg.patience <= 0 or not math.isfinite(tcfg.patience):
            raise ValueError(
                f"transcription.patience must be a positive finite number, got "
                f"{tcfg.patience}"
            )
        if tcfg.length_penalty <= 0 or not math.isfinite(tcfg.length_penalty):
            raise ValueError(
                f"transcription.length_penalty must be a positive finite number "
                f"(1.0 = no-op), got {tcfg.length_penalty}"
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
        # WhisperX internal-VAD onset/offset are probabilities → strictly in (0, 1).
        for knob in ("vad_onset", "vad_offset"):
            value = getattr(tcfg, knob)
            if not math.isfinite(value) or not (0.0 < value < 1.0):
                raise ValueError(
                    f"transcription.{knob} must be a probability in (0, 1), "
                    f"got {value}"
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
        ("relabel", RelabelConfig),
        ("transcription", TranscriptionConfig),
    ):
        sub_dict = config_dict.pop(key, None)
        if key in ("diarization", "relabel") and sub_dict and \
                sub_dict.get("hf_token") == _REDACTED:
            # A saved config redacts the token to _REDACTED; drop the key (a new
            # dict, never mutating the caller's) so the config's default_factory
            # re-resolves $HF_TOKEN instead of handing the literal "REDACTED".
            sub_dict = {k: v for k, v in sub_dict.items() if k != "hf_token"}
        if key == "diarization" and sub_dict and isinstance(sub_dict.get("fusion"), dict):
            # `diarization.fusion` is the one nested dataclass inside a stage
            # config (option 3). YAML / asdict() leaves it a plain dict, so
            # rebuild it into a FusionConfig before constructing DiarizationConfig
            # (a new dict, never mutating the caller's).
            sub_dict = dict(sub_dict)
            sub_dict["fusion"] = FusionConfig(**sub_dict["fusion"])
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
