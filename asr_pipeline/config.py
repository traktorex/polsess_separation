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

from asr_pipeline.text_metrics import LOOP_SCORE_THRESHOLD


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


def _range_phrase(lo, hi, lo_open, hi_open, allow_none, finite) -> str:
    """Human description of the accepted range for a `_require_range` message."""
    if lo is not None and hi is not None:
        left = "(" if lo_open else "["
        right = ")" if hi_open else "]"
        body = f"in {left}{lo}, {hi}{right}"
    elif lo is not None:
        body = f"{'>' if lo_open else '>='} {lo}"
    elif hi is not None:
        body = f"{'<' if hi_open else '<='} {hi}"
    else:
        body = ""
    if finite:
        phrase = f"a finite number {body}".rstrip()
    else:
        phrase = body or "a number"
    if allow_none:
        phrase = f"None or {phrase}"
    return phrase


def _require_range(value, name: str, *, lo=None, hi=None, lo_open=False,
                   hi_open=False, allow_none=False, finite=True, note="") -> None:
    """Raise ValueError unless `value` sits within the given numeric bound(s).

    Sibling of `_one_of` for the numeric-range checks in
    `PipelineConfig.__post_init__`. Bounds are optional; `lo_open`/`hi_open`
    make the respective side strict (`>`/`<` instead of `>=`/`<=`). `allow_none`
    lets `None` pass (an "off"/"no cap" sentinel). `finite=True` additionally
    rejects NaN/Inf; pass `finite=False` to reproduce a legacy check that had no
    finiteness guard (there, a NaN slips through the bare bound comparison
    exactly as it did before — behaviour-preserving). `note` appends a
    parenthetical to the message. Every message names the knob, the bound, and
    the offending value.
    """
    if value is None:
        valid = allow_none
    else:
        # Bound tests are written as "below lo" / "above hi" (never their
        # negation) so a NaN — for which every comparison is False — slips
        # through when `finite=False`, exactly as the pre-helper checks did.
        below_lo = lo is not None and (value <= lo if lo_open else value < lo)
        above_hi = hi is not None and (value >= hi if hi_open else value > hi)
        valid = (
            (not finite or math.isfinite(value))
            and not below_lo
            and not above_hi
        )
    if valid:
        return
    suffix = f" ({note})" if note else ""
    phrase = _range_phrase(lo, hi, lo_open, hi_open, allow_none, finite)
    raise ValueError(f"{name} must be {phrase}{suffix}, got {value!r}")


# ---------------------------------------------------------------------------
# Per-stage configs
# ---------------------------------------------------------------------------


@dataclass
class DiarizationConfig:
    """Stage 1: speaker diarization. Two backends (see ``backend``):

      - ``"pyannote"`` (default): pyannote ``speaker-diarization-3.1`` plus the
        embedding / clustering / segmentation front-end knobs below.
      - ``"sortformer"``: NVIDIA Sortformer-v1 offline EEND
        (``nvidia/diar_sortformer_4spk-v1``) — a clustering-free end-to-end
        diarizer run via an isolated-venv subprocess (``$SORTFORMER_VENV_PY``;
        see ``stages/diarization.py`` + ``scripts/sortformer_worker.py``). Probe
        evidence: ``docs/sweep_plan/eend_probe.py`` + ``_forensics/EEND_PROBE.md``.
    """

    enabled: bool = True
    # Diarizer backend. "pyannote" = the shipped clustering pipeline (all the
    # front-end knobs below apply). "sortformer" = the clustering-free EEND
    # alternative; on that path the pyannote-only knobs (model_id, hf_token,
    # embedding, clustering_*, segmentation_*) are IGNORED — only num_speakers
    # (as a top-2-head POST-filter) and the sortformer_* fields matter.
    backend: str = "pyannote"
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
    # --- Sortformer (EEND) backend knobs (IGNORED when backend == "pyannote") ---
    # Model id for the NeMo Sortformer offline EEND diarizer. Loaded in the
    # isolated NeMo venv by scripts/sortformer_worker.py (fixed 4-speaker head;
    # the 2 most-active heads are post-selected as our 2 speakers).
    sortformer_model_id: str = "nvidia/diar_sortformer_4spk-v1"
    # Sigmoid activity threshold turning Sortformer's per-frame speaker-activity
    # into speaker turns (the probe's DEFAULT_THR; the fused-trio / dev purity is
    # robust across {0.4, 0.5, 0.6} — EEND_PROBE.md threshold-robustness table).
    # Strictly in (0, 1). The turn-building gap-fill / min-duration constants are
    # fixed module-level in stages/diarization.py (ported from eend_probe.py).
    sortformer_threshold: float = 0.5
    # --- Sortformer v4.1 rehabilitation levers (docs/sweep_plan/V41_PREREG.md) ---
    # All FOUR default to the current v4 behaviour (top-2 discard, flat threshold,
    # no fallback), so an untouched sortformer config is byte-identical to `v4_eend`.
    #
    # L1 — surplus-head policy. "top2" (default) = the v4 rule: keep only the two
    # most-active heads, DISCARD the rest. "merge" = assign each surplus-head run
    # (>= the module floor `_SF_MERGE_MIN_DUR_S` in diarization.py) to the top-2
    # speaker whose SOLO speech it embeds closest to, when the cosine margin clears
    # `sortformer_merge_margin`; below-margin / too-short runs stay discarded. Local
    # ECAPA2 embedding only — NO global clustering (V41_PREREG.md L1).
    sortformer_head_policy: str = "top2"          # "top2" | "merge"
    # Cosine margin (best - other) a surplus run's ECAPA2 match must clear to be
    # merged into a top-2 speaker (only used when head_policy == "merge"). Anatomy
    # (SORTFORMER_FAILURE_ANATOMY.md): true miscount-head margins were 0.15-0.57,
    # so 0.10 splits cleanly. Finite, >= 0.
    sortformer_merge_margin: float = 0.10
    # --- Long-recording model routing (deployment/robustness, not a quality knob) ---
    # The offline v1 model runs the WHOLE file through two 18-layer fully-global-
    # attention encoders in one pass — O(T^2) activation memory (~5-6 min ceiling on
    # a 12 GB GPU; observed OOM on e14aa22f.wav, 32:47, 2026-07-06). Recordings
    # strictly LONGER than this threshold (s) are routed to the streaming model
    # below, which processes bounded windows with an Arrival-Order Speaker Cache
    # (memory flat in duration; identical (T, 4) @ 0.08 s output contract). 0
    # disables routing (always use sortformer_model_id). 240 s keeps eval fragments
    # (<= ~95 s) far below the threshold — byte-identical to v1 — while covering
    # long recordings well before v1's OOM ceiling. No-op when sortformer_model_id
    # already names a streaming model. Deliberate, documented substitution (a knob +
    # a loud warning, not a silent swap — SCOPE §4). >= 0.
    sortformer_long_audio_threshold_s: float = 240.0
    sortformer_long_audio_model_id: str = "nvidia/diar_streaming_sortformer_4spk-v2.1"


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
    backend: str = "naive"
    # Path to the AP-BWE generator checkpoint (PyTorch state dict
    # containing the 'generator' key). Ignored by non-AP-BWE backends.
    checkpoint_path: str = field(
        default_factory=lambda: os.getenv(
            "AP_BWE_CHECKPOINT",
            "/home/user/AP-BWE/checkpoints/8kto16k/g_8kto16k",
        )
    )


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
    # Solo-duration threshold (s) below which the DIAGNOSTIC `weak_anchor` flag is
    # raised (a speaker with less solo than this has a shaky anchor). Purely
    # informational: it does NOT gate the anchor fallback — that trigger is
    # `anchor_min_duration_s` below. Renamed from `min_solo_for_anchor_s` to make
    # the diagnostic-only role explicit. >= 0.
    weak_anchor_warn_below_s: float = 3.0
    # Minimum solo duration (s) a speaker needs for an ECAPA *anchor*. Below this
    # `_compute_anchors` leaves the anchor None and ALL that speaker's overlaps
    # fall to fixed positional assignment — the real attribution fallback
    # trigger (distinct from `weak_anchor_warn_below_s`, which only sets the
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
    """Stage 5: ASR per assembled stream.

    Two backends, same surface contract::

      - ``whisperx`` (default): WhisperX = faster-whisper + wav2vec2 forced
        alignment. Word-level timestamps to ±50 ms. Accepts OpenAI short names
        (``large-v3``, ``large-v2``, …) and non-OpenAI Whisper checkpoints via
        HF model id (e.g. a language-specific Whisper finetune).
      - ``coherex``: Cohere ASR (Diffio-AI/CohereX), run in an isolated-venv
        subprocess (``$COHEREX_VENV_PY``). Kept for choosability / re-testing.

    Both backends emit the same output shape per speaker::

        {"text": str, "segments": [{"start": float, "end": float,
                                    "text": str, "words": [...]?}],
         "language": str}
    """

    enabled: bool = True
    # Selector. See class docstring for trade-offs.
    backend: str = "whisperx"          # whisperx | coherex
    # OpenAI short names (``large-v3``, ``large-v2``) or any HF Whisper model id
    # parseable by faster-whisper (the whisperx backend).
    model_name: str = "large-v2"
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
    # Exponential length penalty (Google NMT, alpha). faster-whisper / WhisperX
    # default = 1 (asr.py ``default_asr_options["length_penalty"]`` = the
    # ``WhisperModel.transcribe`` signature default), threaded into the WhisperX
    # ``asr_options`` at 1.0 = no-op (byte-identical baseline). Must be > 0.
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
    suppress_numerals: bool = False
    # WhisperX internal-VAD onset / offset probabilities (Schmitt-trigger style:
    # a frame enters speech above ``vad_onset`` and leaves below ``vad_offset``).
    # WhisperX-only: passed in the ``vad_options`` dict to ``load_model``, which
    # merges them over its own ``default_vad_options`` (asr.py ~L409-412). The
    # pipeline currently passes no ``vad_options``, so the defaults below
    # reproduce WhisperX's exactly — ``vad_onset=0.500``, ``vad_offset=0.363``
    # = byte-identical baseline. Lowering ``vad_offset`` keeps trailing speech
    # the VAD would otherwise clip; both must lie in (0, 1).
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
    # behaviour (byte-identical baseline).
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

    # --- Conditional repetition-loop retry (WhisperX only) --------------------
    # The OVER-production mirror of the collapse retry above. WhisperX's batched
    # decode sometimes emits a repetition-loop hallucination — one merged VAD
    # window transcribed as a single token repeated dozens of times ("No tak,
    # tak, tak, ..." ×112). The collapse detector (an UNDER-production, low-wps
    # detector) cannot catch it: a loop's word density is 4-6 w/s, far ABOVE
    # collapse_max_wps, and its accept-if-more-words guard is inverted for a
    # hallucination where the goal is FEWER words. Setting no_repeat_ngram_size
    # GLOBALLY kills the loops but costs ~1 cpWER of cosmetic collateral in the
    # non-looping strata, so the fix is CONDITIONAL: score each window with the
    # dominant-token-fraction metric (asr_pipeline/text_metrics.repetition_loop_
    # score — shared with docs/sweep_plan/scan_repetition_loops.py so detector
    # and monitor agree), and re-transcribe ONLY the looped windows with
    # no_repeat_ngram_size set. Same detect-and-retry structure as the collapse
    # path (loop_retry runs after retry_collapsed, before wav2vec2 alignment).
    #
    # OFF by default → byte-identical to the shipped pipeline (nothing new runs).
    # WhisperX-only: coherex has no faster-whisper TranscriptionOptions to
    # override, so loop_retry=True with backend != whisperx is a loud config
    # error (SCOPE §4.1: no silent no-op).
    loop_retry: bool = False
    # no_repeat_ngram_size applied ONLY on the loop-retry pass (the global
    # `no_repeat_ngram_size` knob above stays independent and untouched). Must be
    # an int >= 1: 0 would leave the retry decode identical to the original and it
    # could never break the loop. 3 = the value shown to suppress the loops.
    loop_retry_ngram: int = 3
    # Dominant-token-fraction at/above which a window is a repetition loop and is
    # retried; a retry is accepted only if its own score falls back below it. In
    # (0, 1]. Default = the scanner's validated operating point
    # (text_metrics.LOOP_SCORE_THRESHOLD = 0.4).
    loop_score_threshold: float = LOOP_SCORE_THRESHOLD
    # The MULTI-TOKEN mirror of loop_retry. loop_retry's dominant-token detector
    # is structurally blind to a repeated PHRASE — a repeated 3-token phrase caps
    # every token's dominant fraction at ~1/3 < loop_score_threshold — yet WhisperX
    # emits phrase loops too: "Tak, to jest..." ×6 inside one segment
    # (152ed870__seg00) and "Jak pojedziemy do Dekathlonu... O!" ×14 across 14
    # consecutive segments (5bab2c34__seg00). Detection = the max consecutive
    # repeated 3–8-gram run (>= 2 distinct tokens) over the joined per-segment
    # token stream, gated at text_metrics.PHRASE_RUN_MIN = 4 (GT-calibrated: the
    # longest genuine run across 236 GT texts is 2, the hallucinations run 6 and
    # 14). Retry mechanics are identical to loop_retry — the same loop_retry_ngram
    # override on the retry decode, the same mirrored accept guard (splice only if
    # the retry neither phrase-loops nor token-loops) — so loop_retry_ngram /
    # loop_score_threshold are shared and need no extra validation. OFF by default
    # → byte-identical pipeline; whisperx-only (same loud config error as
    # loop_retry, enforced at TranscriptionStage.load).
    loop_retry_phrase: bool = False
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
    # Solo 2-means initialisation strategy (the "degeneracy rescue"). The
    # pass-1-seeded Lloyd's in `_cluster_two` cannot escape a corrupt pass-1
    # partition: on a handful of fragments pyannote pass-1 labels are near-random,
    # so the seeded loop converges to a duration-degenerate "outlier peel" (one
    # pseudo-speaker holding a vanishing share of solo duration) that a genuine
    # 2-speaker conversation never produces. Validated offline
    # (CLUSTERING_DIAGNOSIS.md §Re-seed validation): the ONLY do-no-harm fix is a
    # *conditional* rescue — unconditional seed swaps HARM clean fragments (a
    # corrupt peel scores BETTER than the genuine split on every compactness
    # objective, so the two are separable only by a balance constraint).
    #   "pass1"  (default): shipped behaviour — pass-1-seeded 2-means. Byte-identical.
    #   "rescue": after the pass-1-seeded 2-means, if the solo partition's
    #             min-cluster duration share is below `rescue_trigger_bal`,
    #             exhaustively pair-seed the SAME Lloyd's loop from every solo
    #             embedding pair, keep the fixed points whose share is at least
    #             `rescue_candidate_bal`, and adopt the one with the highest
    #             duration-weighted mean cosine of each piece to its own cluster
    #             centroid (no balanced fixed point → keep the pass-1 partition,
    #             logged). Deterministic; N<=~45 solos → <=~1000 millisecond Lloyd's
    #             runs on CPU.
    solo_clustering_init: str = "pass1"
    # Rescue TRIGGER: fire only when the pass-1-seeded solo partition's min-cluster
    # duration share is below this. 0.10 sits in the ~3.5x gap between the last
    # validated win (0.039) and the first harm (0.135). Share, in [0, 1].
    rescue_trigger_bal: float = 0.10
    # Rescue CANDIDATE filter: a pair-seeded fixed point is an eligible replacement
    # only if its min-cluster duration share is at least this (a genuine 2-speaker
    # split is balanced; the corrupt peels this rescue escapes are not). Share, in
    # [0, 1]; should be >= rescue_trigger_bal.
    rescue_candidate_bal: float = 0.20
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
                ("frcrn_se_16k", "mossformer_gan_se_16k"))
        _one_of(self.enhancement.resample_quality, "enhancement.resample_quality",
                ("soxr_hq", "soxr_vhq", "kaiser_best"))
        _one_of(self.post_separation_processing.backend,
                "post_separation_processing.backend",
                ("naive", "ap_bwe"))
        _one_of(self.transcription.backend, "transcription.backend",
                ("whisperx", "coherex"))
        _one_of(self.assembly.anchor_embedding, "assembly.anchor_embedding",
                ("ecapa1", "ecapa2"))
        _one_of(self.diarization.clustering_method, "diarization.clustering_method",
                ("average", "centroid", "complete", "median", "single",
                 "ward", "weighted"))
        _one_of(self.diarization.backend, "diarization.backend",
                ("pyannote", "sortformer"))
        _one_of(self.relabel.source, "relabel.source", ("solos", "global"))
        _one_of(self.relabel.audio_source, "relabel.audio_source",
                ("enhanced", "raw"))
        _one_of(self.relabel.solo_clustering_init, "relabel.solo_clustering_init",
                ("pass1", "rescue"))
        # Degeneracy-rescue balance thresholds are duration shares → finite, [0, 1]
        # (only consulted when solo_clustering_init="rescue").
        for _knob in ("rescue_trigger_bal", "rescue_candidate_bal"):
            _require_range(getattr(self.relabel, _knob), f"relabel.{_knob}",
                           lo=0.0, hi=1.0, note="duration share")
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
        # `finite=False` on the two positivity checks below preserves the legacy
        # bare-comparison behaviour (a NaN slipped through unflagged).
        _require_range(
            self.separation.training_chunk_length_s,
            "separation.training_chunk_length_s", lo=0.0, lo_open=True, finite=False,
            note="a non-positive chunk length would make overlap-add hop 0 → infinite loop",
        )
        _require_range(
            self.separation.overlap_add_threshold_s,
            "separation.overlap_add_threshold_s", lo=0.0, lo_open=True, finite=False,
        )
        _require_range(
            self.separation.vad_soft_threshold, "separation.vad_soft_threshold",
            lo=0.0, finite=False,
            note="a negative value makes every frame 'weak' and floods the Schmitt mask",
        )
        _require_range(
            self.separation.seam_silence_threshold,
            "separation.seam_silence_threshold", lo=0.0, hi=1.0,
            lo_open=True, hi_open=True,
            note="VAD silence cutoff for snap_to_silence; 0.5 = default",
        )

        # --- Assembly numeric knobs ---
        # A negative value here crashes deep in Stage 6 (`np.zeros(negative)` /
        # broadcast errors), not at config load — so fail loud and early,
        # naming the offending knob. All are seconds/ms durations: zero is a
        # valid disable (no gap / no fade / no cap), so the floor is >= 0.
        acfg = self.assembly
        # `finite=False` reproduces the legacy bare `< 0` check (NaN slips
        # through unflagged); solo_onset_pad_s keeps its own finite guard below.
        for knob in (
            "silence_separator_s",
            "crossfade_ms",
            "edge_fade_ms",
            "weak_anchor_warn_below_s",
            "anchor_min_duration_s",
            "overlap_min_duration_s",
        ):
            _require_range(getattr(acfg, knob), f"assembly.{knob}",
                           lo=0.0, finite=False)
        # anchor_max_duration_s is Optional (None = no cap); a non-None value
        # must be positive (a zero/negative cap would empty the anchor audio).
        _require_range(acfg.anchor_max_duration_s, "assembly.anchor_max_duration_s",
                       lo=0.0, lo_open=True, allow_none=True, finite=False,
                       note="None = no cap")
        # solo_onset_pad_s needs the finite guard (unlike the >= 0 loop above): a
        # NaN passes a bare `< 0` check and would silently disable every clamp
        # comparison downstream.
        _require_range(acfg.solo_onset_pad_s, "assembly.solo_onset_pad_s",
                       lo=0.0, note="0 = off")

        # --- Diarization front-end knobs (pyannote instantiate params) ---
        dcfg = self.diarization
        _require_range(dcfg.segmentation_min_duration_off,
                       "diarization.segmentation_min_duration_off", lo=0.0)
        _require_range(dcfg.clustering_threshold, "diarization.clustering_threshold",
                       lo=0.0, hi=2.0, note="cosine; INERT under num_speakers=2")
        _require_range(dcfg.clustering_min_cluster_size,
                       "diarization.clustering_min_cluster_size", lo=1, finite=False)
        # Sortformer (EEND) backend cross-checks. Its 4-speaker head is
        # post-filtered to the 2 most-active heads, so the pipeline's 2-speaker
        # assumption is a HARD requirement on this path — fail loud, never a
        # silent miscount (SCOPE §4). The turn threshold is a probability.
        if dcfg.backend == "sortformer" and dcfg.num_speakers != 2:
            raise ValueError(
                "diarization.backend='sortformer' requires num_speakers == 2 "
                "(the pipeline post-selects the 2 most-active of Sortformer's 4 "
                f"output heads); got num_speakers={dcfg.num_speakers}."
            )
        _require_range(dcfg.sortformer_threshold, "diarization.sortformer_threshold",
                       lo=0.0, hi=1.0, lo_open=True, hi_open=True,
                       note="probability; 0.5 = probe default")
        # Sortformer levers. Validated unconditionally (like sortformer_threshold)
        # so a YAML typo fails loud on any backend; the defaults preserve the
        # stock top-2 behaviour.
        _one_of(dcfg.sortformer_head_policy, "diarization.sortformer_head_policy",
                ("top2", "merge"))
        _require_range(dcfg.sortformer_merge_margin,
                       "diarization.sortformer_merge_margin", lo=0.0,
                       note="cosine margin; only used when sortformer_head_policy='merge'")
        _require_range(dcfg.sortformer_long_audio_threshold_s,
                       "diarization.sortformer_long_audio_threshold_s", lo=0.0,
                       note="seconds; 0 = disable long-audio model routing")

        _require_range(self.enhancement.observation_mix_ratio,
                       "enhancement.observation_mix_ratio", lo=0.0, hi=1.0,
                       note="convex dry-wet weight; 0 = pure enhanced")

        # --- Transcription decode knobs ---
        tcfg = self.transcription
        _require_range(tcfg.silence_floor, "transcription.silence_floor", lo=0.0,
                       note="0 = gate off; 1e-4 = default")
        _require_range(tcfg.beam_size, "transcription.beam_size", lo=1, finite=False)
        _require_range(tcfg.patience, "transcription.patience", lo=0.0, lo_open=True,
                       note="1.0 = default")
        _require_range(tcfg.length_penalty, "transcription.length_penalty",
                       lo=0.0, lo_open=True, note="1.0 = no-op")
        _require_range(tcfg.no_speech_threshold, "transcription.no_speech_threshold")
        _require_range(tcfg.compression_ratio_threshold,
                       "transcription.compression_ratio_threshold")
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
        _require_range(tcfg.no_repeat_ngram_size,
                       "transcription.no_repeat_ngram_size", lo=0, finite=False,
                       note="0 = disabled")
        _require_range(tcfg.repetition_penalty, "transcription.repetition_penalty",
                       lo=0.0, lo_open=True, note="1.0 = no penalty")
        _require_range(tcfg.hallucination_silence_threshold,
                       "transcription.hallucination_silence_threshold",
                       lo=0.0, lo_open=True, allow_none=True,
                       note="None = off; seconds")
        _require_range(tcfg.chunk_size, "transcription.chunk_size", lo=1,
                       finite=False, note="seconds; 30 = WhisperX default")
        # Detect-and-retry knobs (WhisperX-only collapse recovery).
        _require_range(tcfg.retry_collapsed_chunk_size,
                       "transcription.retry_collapsed_chunk_size", lo=0,
                       finite=False, note="0 = disabled; 8 = default")
        _require_range(tcfg.collapse_min_duration_s,
                       "transcription.collapse_min_duration_s", lo=0.0,
                       lo_open=True, note="seconds")
        _require_range(tcfg.collapse_max_wps, "transcription.collapse_max_wps",
                       lo=0.0, lo_open=True, note="words/second threshold")
        # Conditional repetition-loop retry (WhisperX-only).
        _require_range(tcfg.loop_retry_ngram, "transcription.loop_retry_ngram",
                       lo=1, finite=False,
                       note="no_repeat_ngram_size for the loop-retry pass; "
                            "0 would leave the retry unable to break the loop")
        _require_range(tcfg.loop_score_threshold,
                       "transcription.loop_score_threshold", lo=0.0, hi=1.0,
                       lo_open=True, note="dominant-token fraction; 0.4 = default")
        # NB the loop_retry-requires-whisperx cross-check is enforced at
        # TranscriptionStage.load() (backend level), not here — matching the
        # convention for the other WhisperX-only knobs (a config may override
        # loop_retry onto a default.yaml base whose backend is already whisperx
        # without re-declaring the backend). Still fails loud, before any audio.
        # WhisperX internal-VAD onset/offset are probabilities → strictly in (0, 1).
        for knob in ("vad_onset", "vad_offset"):
            _require_range(getattr(tcfg, knob), f"transcription.{knob}",
                           lo=0.0, hi=1.0, lo_open=True, hi_open=True,
                           note="probability")

        if self.spill_intermediate and self.artifact_dir is None:
            raise ValueError(
                "spill_intermediate is True but artifact_dir is None — "
                "set artifact_dir (e.g. via --output) or disable spilling."
            )

        # hf_token is a pyannote-backend requirement only: the sortformer worker
        # reads $HF_TOKEN from the process env for its (public) NeMo model
        # download, not from config.hf_token, so the sortformer path does not
        # need it set here.
        if (self.diarization.enabled
                and self.diarization.backend == "pyannote"
                and not self.diarization.hf_token):
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
        if key == "diarization" and sub_dict and "fusion" in sub_dict:
            # Back-compat: the disagreement-aware fusion stage was removed
            # (swept-and-rejected). Drop a stray `fusion` block from an old
            # saved config so it still loads, instead of crashing on an
            # unexpected keyword.
            sub_dict = {k: v for k, v in sub_dict.items() if k != "fusion"}
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


# ---------------------------------------------------------------------------
# Dotted-path overrides (shared by the CLI `--set` flag and the sweep registry)
# ---------------------------------------------------------------------------


def apply_overrides(config: PipelineConfig, overrides: dict) -> PipelineConfig:
    """Apply ``{"stage.knob": value}`` dotted-path overrides in place, re-validate.

    One override mechanism for the whole package: the CLI ``--set`` flag and
    ``scripts/sweep_pipeline.py``'s config registry both route through here, so
    there is a single fail-loud policy. A typo'd path must fail loud
    (SCOPE §4.1): a bare ``setattr`` would create a junk attribute, leave the
    intended knob at its default, and silently run the baseline under the typo'd
    name — a fabricated run with no signal. An unknown *leaf* raises
    ``AttributeError`` explicitly; an unknown *parent* stage raises
    ``AttributeError`` from ``getattr`` while walking the path.

    Values are used as given (already typed) — the CLI turns its ``--set``
    strings into typed values via `parse_cli_overrides` first. Re-runs
    ``__post_init__`` so every override is validated exactly as a YAML or
    programmatic config would be (an out-of-range value raises ``ValueError``
    here, not silently downstream). Returns the same (mutated) ``config``.
    """
    for path, val in overrides.items():
        obj = config
        *parents, leaf = path.split(".")
        for p in parents:
            obj = getattr(obj, p)
        if not hasattr(obj, leaf):
            raise AttributeError(f"unknown override path: {path!r}")
        setattr(obj, leaf, val)
    config.__post_init__()
    return config


def parse_cli_overrides(items: list) -> dict:
    """Parse ``["stage.knob=value", ...]`` CLI tokens into a typed override dict.

    Each token's value string is YAML-parsed for typing, so ``true``/``false``
    become bools, ``0.5`` a float, ``5`` an int, ``[0.0, 0.2]`` a list, and bare
    or quoted text a str (``large-v2`` -> ``"large-v2"``,
    ``nvidia/model-v1`` -> the string). The resulting dict feeds
    `apply_overrides`, which fails loud on any unknown path. A token without an
    ``=`` (or with an empty key) is a hard error — never silently ignored.
    """
    overrides: dict = {}
    for item in items:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(
                f"--set expects 'stage.knob=value', got {item!r} (no '=')."
            )
        key = key.strip()
        if not key:
            raise ValueError(f"--set has an empty knob path: {item!r}.")
        overrides[key] = yaml.safe_load(raw)
    return overrides
