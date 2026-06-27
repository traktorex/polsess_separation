"""2nd-pass identity re-clustering on clean audio (B / B+) — SECOND_PASS_PLAN.md.

This stage runs between post_separation_processing (3c) and assembly. It
re-decides speaker IDENTITY on cleaner audio than pass-1 diarization saw, while
leaving PRESENCE (segment boundaries, overlap timeline, speaker count) entirely
to the raw pass-1 result. The design driver (plan §0): enhanced audio HELPS
identity (denoise → cleaner embedding) but HURTS presence (a single-output SE
model suppresses the quieter speaker in an overlap), so enhanced/separated audio
may decide identity, never presence.

Two modes, selected by `relabel.source`:

- ``"solos"`` (B): re-embed the pass-1 SOLO segments (overlap-excluded) with a
  stronger embedder (ECAPA2), split into exactly 2 by cosine 2-means seeded from
  the pass-1 speaker centroids, align the two clusters back onto the pass-1
  labels by max-duration overlap, and overwrite `segments_df["speaker"]`. This
  targets the db15fc57-class mislabel: a ~2.5 s solo turn of GT-A that pass-1
  clustered into B.

- ``"global"`` (B+): cluster the solos PLUS the VAD-gated separated overlap
  streams (`s1_gated`/`s2_gated`, recovered from the un-enhanced mixture by the
  separator, so the quiet speaker survives) JOINTLY. Solos are relabeled exactly
  as in B; additionally each overlap's stream→speaker pairing is emitted to
  `ctx.overlap_speaker_assignment` (keyed by the dense list index `i_ovl`),
  which assembly consumes via its consensus-injection seam in place of its own
  per-overlap anchor argmax. Overlaps B+ cannot decide (a stream below the
  embedder's min_num_samples, or both streams landing in one cluster) are
  omitted → assembly falls back to its own ladder, logged.

Model-bearing (loads its own ECAPA2 via `build_custom_embedding`, the same
custom-embedder wrapper the diarization stage uses), so it overrides
load/unload/load_signature mirroring DiarizationStage. Phase-major: ECAPA2 is
loaded here between 3c (silero VAD, already unloaded) and assembly (its own
ECAPA1/2) and unloaded before assembly — never two big models co-resident.

Default `relabel.enabled=False` → the orchestrator skips the stage entirely:
ECAPA2 is never loaded, `ctx` is untouched, `overlap_speaker_assignment` stays
None — a byte-identical no-op.

SCOPE §4 (no silent substitution): a configured enhanced audio_source with no
`enhanced_full` (enhancement disabled) is a config-time crash (config.py
cross-check) backed by a runtime assert here; too-few usable segments, a single
pass-1 speaker, and per-overlap drop sites are visible no-ops / fall-softs that
say so via `dlog`, never quiet downgrades.
"""

from __future__ import annotations

import gc
from typing import Optional

import numpy as np
import torch

from asr_pipeline.config import RelabelConfig
from asr_pipeline.context import Interval, PipelineContext
from asr_pipeline.debug_log import dlog
from asr_pipeline.stages.assembly import _coalesce, _subtract
from asr_pipeline.stages.base import Stage
from asr_pipeline.stages.custom_embeddings import build_custom_embedding


def _log(msg: str) -> None:
    """Progress message — stdout + the durable debug log (the embedder build is
    multi-second and the WSL stdout bridge can drop; every model-bearing stage
    logs the same way)."""
    dlog("relabel", msg)


# ---------------------------------------------------------------------------
# Pure helpers (clustering / alignment / span derivation)
# ---------------------------------------------------------------------------


def _solo_embedding_spans(
    seg_df,
    overlap_regions: list[Interval],
    exclude_overlap: bool,
) -> list[tuple[int, list[Interval], str]]:
    """One (row_index, embedding_spans, pass1_label) per `seg_df` row.

    The embedding span for a segment is `[(start, end)] − overlap_regions` (when
    `exclude_overlap`), matching assembly's solo derivation and pyannote's
    `embedding_exclude_overlap=True` intent at the segment level. A segment fully
    covered by overlaps yields an empty span list (the caller leaves its pass-1
    label untouched). Row index is the positional index into `seg_df` so the
    caller can write `segments_df.iloc` rows back without label-string ambiguity.
    """
    spans: list[tuple[int, list[Interval], str]] = []
    blocked = _coalesce(overlap_regions) if exclude_overlap else []
    for pos, row in enumerate(seg_df.itertuples()):
        seg = [(float(row.start), float(row.end))]
        kept = _subtract(seg, blocked) if blocked else seg
        spans.append((pos, kept, str(row.speaker)))
    return spans


def _embed_spans(
    spans: list[list[Interval]],
    audio: np.ndarray,
    embedder,
    sr: int,
) -> np.ndarray:
    """Embed one concatenated waveform per span list → `(N, dim)` array.

    Slices each kept sub-span out of `audio`, concatenates per segment, and feeds
    `(1, 1, T)` to the custom embedder (which returns `(1, dim)` with a NaN row
    when the usable signal is below `min_num_samples`). A segment with no kept
    span (empty list) gets a NaN row directly — the caller drops NaN rows before
    clustering and keeps their pass-1 label. Done one-at-a-time (concat lengths
    differ per segment; the embedder loops per item internally anyway).
    """
    dim = embedder.dimension
    out = np.full((len(spans), dim), np.nan, dtype=np.float32)
    for i, span_list in enumerate(spans):
        slices: list[np.ndarray] = []
        for s, e in span_list:
            lo = int(s * sr)
            hi = int(e * sr)
            if hi > lo:
                slices.append(audio[lo:hi].astype(np.float32))
        if not slices:
            continue  # leave NaN — nothing to embed
        concat = np.concatenate(slices)
        wav = torch.from_numpy(concat).reshape(1, 1, -1)
        out[i] = embedder(wav)[0]
    return out


def _embed_overlap_streams(
    overlap_separated: list,
    embedder,
    sr: int,
) -> tuple[np.ndarray, list[tuple[int, str]]]:
    """Embed every VAD-gated separated overlap stream (B+ point set).

    Returns `(embeddings, meta)` where `embeddings[k]` corresponds to
    `meta[k] = (i_ovl, stream)` with `stream in {"s1", "s2"}` and `i_ovl` the
    DENSE list index into `overlap_separated` (the key assembly's consensus seam
    uses). Streams missing a `_gated` array or with too few finite (non-zero)
    samples are skipped here and so never enter the point set — their overlap is
    later found "incomplete" and omitted from the assignment (assembly anchor
    decides it, plan §4.3). NaN rows the embedder returns for sub-min input are
    likewise filtered by the caller.
    """
    min_n = embedder.min_num_samples
    embs: list[np.ndarray] = []
    meta: list[tuple[int, str]] = []
    for i_ovl, ovl in enumerate(overlap_separated):
        for stream in ("s1", "s2"):
            key = f"{stream}_gated"
            arr = ovl.get(key)
            if arr is None:
                _log(
                    f"  overlap {i_ovl} stream {stream}: no {key!r} — skipped "
                    f"(post_separation_processing did not populate it)"
                )
                continue
            # Count the non-zero (VAD-gated-in) samples: the gate zeros silence,
            # and a stream that is all/mostly zeros has no usable speech to embed.
            nonzero = int(np.count_nonzero(arr))
            if len(arr) < min_n or nonzero < min_n:
                _log(
                    f"  overlap {i_ovl} stream {stream}: too short for ECAPA2 "
                    f"({nonzero} usable samples < {min_n}) — omitted (assembly "
                    f"anchor will decide this overlap)"
                )
                continue
            wav = torch.from_numpy(np.asarray(arr, dtype=np.float32)).reshape(1, 1, -1)
            embs.append(embedder(wav)[0])
            meta.append((i_ovl, stream))
    if not embs:
        return np.zeros((0, embedder.dimension), dtype=np.float32), []
    return np.stack(embs).astype(np.float32), meta


def _cluster_two(
    emb: np.ndarray,
    seed_labels: np.ndarray,
    duration_weighted: bool,
    weights: np.ndarray,
) -> np.ndarray:
    """Split unit-normalised embeddings into exactly 2 by seeded cosine 2-means.

    The pass-1 diarization already partitions the SOLO points into two speakers;
    this pass only *refines* that identity on cleaner audio. So we seed Lloyd's
    algorithm with the two pass-1 speaker centroids (cosine = dot on unit vectors,
    matching the embedders' `metric="cosine"`) and iterate to convergence rather
    than rediscovering structure from scratch. `seed_labels` is a `(N,)` int array
    in {0, 1} for the SOLO rows carrying a pass-1 label and -1 for rows with none
    (B+'s overlap streams) — the latter join whichever centroid they are nearest.
    Returns a `(N,)` int array of cluster ids in {0, 1}.

    Why not agglomerative linkage. The previous implementation used scipy
    `centroid`/`average` linkage + `fcluster(maxclust=2)`. On real ECAPA2
    embeddings (192-D, two genuine speakers but only moderately separated:
    pairwise cosine median ~0.13–0.29) that path chains/peels a single outlier
    and returns a degenerate 1-vs-(N−1) split — observed 26-vs-1 on 065a9896 and
    16-vs-1 on db15fc57 — which then collapses almost every solo onto one speaker.
    Seeding from the pass-1 partition cannot produce that degeneracy: both
    centroids start populated and Lloyd's only moves a point when it is genuinely
    closer to the other speaker. Deterministic (fixed seeds, no random init), so
    it holds under `deterministic=True`. Caller guarantees N >= 2, all rows
    finite, and both pass-1 labels present in `seed_labels`.

    `duration_weighted` (default OFF) weights each seed point's contribution to
    its starting centroid by its duration. The db15fc57 diagnosis argues against
    it — long turns dominating the centroid is the very bias this pass exists to
    fix (SECOND_PASS_PLAN.md §3.2) — so it is kept only as an A/B knob; duration
    weighting is used in ALIGNMENT regardless, where long anchors SHOULD pin
    identity.
    """
    unit = emb / (np.linalg.norm(emb, axis=-1, keepdims=True) + 1e-12)
    seed_labels = np.asarray(seed_labels)

    # --- Seed: the two pass-1 speaker centroids (optionally duration-weighted). ---
    centroids = np.empty((2, unit.shape[1]), dtype=np.float64)
    for k in (0, 1):
        members = seed_labels == k
        if duration_weighted:
            w = weights[members].astype(np.float64)
            centroids[k] = np.average(unit[members], axis=0, weights=w)
        else:
            centroids[k] = unit[members].mean(axis=0)
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12

    # --- Lloyd's iterations under cosine similarity (dot on unit vectors). ---
    labels = seed_labels.copy()
    for _ in range(50):  # converges in a handful; 50 is a safe ceiling
        new = (unit @ centroids.T).argmax(axis=1)
        if np.array_equal(new, labels):
            break
        labels = new
        for k in (0, 1):
            members = labels == k
            if members.any():
                centroids[k] = unit[members].mean(axis=0)
        centroids /= np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12
    return labels.astype(int)


def _align_to_old(
    clusters: np.ndarray,
    pass1_labels: list[str],
    durations: np.ndarray,
    speakers: list[str],
) -> dict[int, str]:
    """Map the 2 cluster ids onto the 2 pass-1 labels by max-duration overlap.

    For each of the two permutations `{0,1} → {speakers[0], speakers[1]}`, sum
    the durations of points whose NEW cluster matches their OLD label under that
    permutation; pick the permutation with the larger agreement. Durations (not
    counts) so a few long anchor turns pin the global A↔B identity (keeping an
    almost-clean global swap free in cpWER) while mis-clustered short segments
    move. Returns `{cluster_id -> speaker_label}`.

    Only SOLO points (those carrying a pass-1 label) participate; B+'s overlap
    points have no single pass-1 label and are excluded by the caller.
    """
    a, b = speakers[0], speakers[1]

    def agreement(c_to_spk: dict[int, str]) -> float:
        total = 0.0
        for c, lbl, dur in zip(clusters, pass1_labels, durations):
            if c_to_spk[int(c)] == lbl:
                total += float(dur)
        return total

    straight = {0: a, 1: b}
    swapped = {0: b, 1: a}
    return straight if agreement(straight) >= agreement(swapped) else swapped


def _overlap_pairings(
    overlap_meta: list[tuple[int, str]],
    overlap_clusters: np.ndarray,
    cluster_to_spk: dict[int, str],
    speakers: list[str],
) -> dict[int, str]:
    """Per-overlap straight/swapped from the global clustering (B+).

    `cluster_to_spk` maps a cluster id to a pass-1 label (from `_align_to_old`).
    `speakers[0]` is the "straight" anchor: an overlap is "straight" when its s1
    stream clustered to `speakers[0]` (so s2 → speakers[1]), else "swapped".
    Only overlaps whose BOTH streams are present in the point set AND landed in
    DIFFERENT clusters are emitted; the rest are omitted (assembly anchor
    decides them, logged at the omission site). Keyed by the dense `i_ovl`.
    """
    a = speakers[0]
    # Gather per-overlap: which cluster did each present stream land in.
    by_ovl: dict[int, dict[str, int]] = {}
    for (i_ovl, stream), cid in zip(overlap_meta, overlap_clusters):
        by_ovl.setdefault(i_ovl, {})[stream] = int(cid)
    pairings: dict[int, str] = {}
    for i_ovl, streams in by_ovl.items():
        if "s1" not in streams or "s2" not in streams:
            _log(
                f"  overlap {i_ovl}: only one stream survived clustering "
                f"({sorted(streams)}) — omitted, assembly anchor decides"
            )
            continue
        if streams["s1"] == streams["s2"]:
            _log(
                f"  overlap {i_ovl}: both streams clustered to the same speaker "
                f"({cluster_to_spk[streams['s1']]}) — degenerate, omitted, "
                f"assembly anchor decides"
            )
            continue
        s1_spk = cluster_to_spk[streams["s1"]]
        pairings[i_ovl] = "straight" if s1_spk == a else "swapped"
    return pairings


# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------


class RelabelStage(Stage):
    name = "relabel"

    def __init__(self, config: RelabelConfig) -> None:
        super().__init__(enabled=config.enabled)
        self.config = config
        self._embedder = None
        self._device: Optional[torch.device] = None

    # ------------------------------------------------------------------
    # Lifecycle (model-bearing; mirror DiarizationStage)
    # ------------------------------------------------------------------
    def load(self, device: torch.device) -> None:
        _log(f"load: building embedder {self.config.embedding!r} on {device}...")
        embedder = build_custom_embedding(self.config.embedding, device)
        if embedder is None:
            # A non-custom name would be a pyannote-format id; the relabel stage
            # only supports the custom wrappers (it has no pyannote pipeline to
            # host an arbitrary embedding factory). Fail loud (SCOPE §4) rather
            # than silently doing nothing.
            raise RuntimeError(
                f"relabel.embedding={self.config.embedding!r} is not a custom "
                f"embedder name; RelabelStage supports the custom wrappers only "
                f"(e.g. 'ecapa2'). Pyannote-format ids are not supported here."
            )
        self._embedder = embedder
        self._device = device
        _log(
            f"load: embedder ready (dimension={embedder.dimension}, "
            f"min_num_samples={embedder.min_num_samples})"
        )

    def unload(self) -> None:
        self._embedder = None
        self._device = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load_signature(self) -> tuple:
        # Only the embedding name picks the model; source/audio_source/
        # exclude_overlap/duration_weighted are runtime knobs (no reload when
        # flipped interactively).
        return (self.config.embedding,)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def run(self, ctx: PipelineContext) -> None:
        _log(f"run: entered (source={self.config.source!r})")
        if self._embedder is None or self._device is None:
            raise RuntimeError("RelabelStage.run called before load().")
        if ctx.diarization is None:
            raise RuntimeError(
                "RelabelStage.run requires ctx.diarization (DiarizationStage "
                "must run first)."
            )
        if ctx.overlap_regions is None:
            raise RuntimeError(
                "RelabelStage.run requires ctx.overlap_regions (RoutingStage "
                "must run first)."
            )

        # Identity-audio source. The config cross-check already forbids
        # enhanced-without-enhancement at config time; this assert is
        # defence-in-depth (SCOPE §4: never a silent raw fallback).
        if self.config.audio_source == "enhanced":
            assert ctx.enhanced_full is not None, (
                "relabel.audio_source='enhanced' but ctx.enhanced_full is None "
                "(enhancement did not run)."
            )
            audio = ctx.enhanced_full
        else:
            if ctx.audio is None:
                raise RuntimeError(
                    "RelabelStage.run requires ctx.audio for audio_source='raw'."
                )
            audio = ctx.audio
        sr = ctx.sample_rate

        speakers = ctx.speakers
        if len(speakers) < 2:
            _log(
                f"no-op: {len(speakers)} pass-1 speaker(s) (< 2) — keeping "
                f"pass-1 labels (won't invent a 2nd speaker; SCOPE §3 'count "
                f"comes from diarization')."
            )
            return

        seg_df = ctx.diarization.segments_df
        overlap_regions = ctx.overlap_regions if self.config.exclude_overlap else []

        # --- Solo point set (both modes) ---
        solo_spans = _solo_embedding_spans(seg_df, overlap_regions, self.config.exclude_overlap)
        solo_emb = _embed_spans([s[1] for s in solo_spans], audio, self._embedder, sr)
        solo_finite = np.isfinite(solo_emb).all(axis=1)
        n_usable = int(solo_finite.sum())
        _log(
            f"solo points: {len(solo_spans)} segments, {n_usable} usable "
            f"(>= min_num_samples), {len(solo_spans) - n_usable} too short "
            f"(keep pass-1 label)."
        )
        if n_usable < 2:
            _log(
                f"no-op: only {n_usable} usable solo segment(s) — linkage is "
                f"undefined below 2; keeping pass-1 labels."
            )
            return

        # --- B+ overlap point set ---
        overlap_emb = np.zeros((0, self._embedder.dimension), dtype=np.float32)
        overlap_meta: list[tuple[int, str]] = []
        if self.config.source == "global":
            if ctx.overlap_separated:
                overlap_emb, overlap_meta = _embed_overlap_streams(
                    ctx.overlap_separated, self._embedder, sr
                )
            overlap_finite = (
                np.isfinite(overlap_emb).all(axis=1)
                if len(overlap_emb) else np.zeros(0, dtype=bool)
            )
            # Drop non-finite overlap rows (sub-min slipped through as NaN).
            if len(overlap_emb) and not overlap_finite.all():
                overlap_emb = overlap_emb[overlap_finite]
                overlap_meta = [m for m, ok in zip(overlap_meta, overlap_finite) if ok]
            _log(f"overlap points (B+): {len(overlap_meta)} usable separated streams.")

        # --- Cluster the joint point set into exactly 2 ---
        # Solo rows first (so their positions line up with `solo_finite`), then
        # the overlap rows. Only finite rows go into linkage.
        solo_idx = np.where(solo_finite)[0]
        solo_pts = solo_emb[solo_idx]
        solo_durs = np.array(
            [sum(e - s for s, e in solo_spans[i][1]) for i in solo_idx],
            dtype=np.float64,
        )
        solo_pass1 = [solo_spans[i][2] for i in solo_idx]

        # The seeded 2-means needs BOTH pass-1 speakers represented among the
        # usable solos to seed two distinct centroids. If every usable solo
        # carries the same pass-1 label (the other speaker's solos were all too
        # short to embed), there is nothing to re-cluster — keep pass-1 labels
        # (visible no-op, SCOPE §4: never an empty-centroid NaN or a 1-vs-rest
        # collapse).
        if len({s for s in solo_pass1 if s in (speakers[0], speakers[1])}) < 2:
            _log(
                f"no-op: usable solos carry a single pass-1 speaker "
                f"({sorted(set(solo_pass1))}) — cannot seed a 2-way split; "
                f"keeping pass-1 labels."
            )
            return

        all_emb = np.concatenate([solo_pts, overlap_emb], axis=0) if len(overlap_emb) \
            else solo_pts
        # Weights for the (optional) duration-weighted seeding: solo points by
        # their duration; overlap points by 1.0 (no natural single duration).
        all_w = np.concatenate(
            [solo_durs, np.ones(len(overlap_emb), dtype=np.float64)]
        ) if len(overlap_emb) else solo_durs
        # Seed each solo point with its pass-1 speaker (0/1 by `speakers` order);
        # overlap points have no pass-1 label → -1 (assigned to the nearest
        # centroid, never seeding one). This is the pass-1 partition the 2-means
        # refines (see _cluster_two).
        spk_to_seed = {speakers[0]: 0, speakers[1]: 1}
        solo_seed = np.array([spk_to_seed[s] for s in solo_pass1], dtype=int)
        all_seed = np.concatenate(
            [solo_seed, np.full(len(overlap_emb), -1, dtype=int)]
        ) if len(overlap_emb) else solo_seed
        clusters = _cluster_two(
            all_emb, all_seed, self.config.duration_weighted, all_w
        )
        solo_clusters = clusters[: len(solo_pts)]
        overlap_clusters = clusters[len(solo_pts):]

        # --- Align the 2 clusters to the pass-1 labels (solo points only) ---
        cluster_to_spk = _align_to_old(
            solo_clusters, solo_pass1, solo_durs, speakers
        )
        _log(
            f"cluster→speaker alignment: "
            f"{{0: {cluster_to_spk[0]!r}, 1: {cluster_to_spk[1]!r}}}"
        )

        # --- Emit: overwrite solo labels (finite points only) ---
        new_labels = seg_df["speaker"].tolist()
        n_changed = 0
        for pos, cid in zip(solo_idx, solo_clusters):
            seg_pos = solo_spans[pos][0]
            new = cluster_to_spk[int(cid)]
            if new_labels[seg_pos] != new:
                n_changed += 1
            new_labels[seg_pos] = new
        seg_df["speaker"] = new_labels
        _log(
            f"relabel: overwrote {len(solo_idx)} solo label(s), "
            f"{n_changed} changed from pass-1."
        )

        # --- B+ overlap handoff ---
        if self.config.source == "global":
            pairings = _overlap_pairings(
                overlap_meta, overlap_clusters, cluster_to_spk, speakers
            )
            ctx.overlap_speaker_assignment = pairings
            n_total = len(ctx.overlap_separated) if ctx.overlap_separated else 0
            _log(
                f"overlap assignment (B+): decided {len(pairings)}/{n_total} "
                f"overlap(s); {n_total - len(pairings)} fall back to assembly "
                f"anchor."
            )
        _log("done.")
