"""Unit tests for the 2nd-pass identity re-clustering stage
(asr_pipeline/stages/relabel.py) — options B (solos) and B+ (global).

A fake embedder (no model download) maps each waveform to a 2-D point so a test
can rig the geometry of the cluster split and check the relabel decision. All
CPU-only; the real ECAPA2 build is a third-party boundary exercised by the GPU
integration test, not here.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from asr_pipeline.config import RelabelConfig
from asr_pipeline.context import DiarizationResult, PipelineContext
from asr_pipeline.stages.relabel import (
    RelabelStage,
    _align_to_old,
    _cluster_two,
    _overlap_pairings,
    _run_level_flips,
    _solo_embedding_spans,
)

SR = 16_000
CPU = torch.device("cpu")


class _FakeEmbedder:
    """Maps a (1,1,T) waveform to a 2-D point by the SIGN of two halves of its
    mean — but really just keyed by the waveform's first sample so a test can
    pick exact embeddings via a table. Returns NaN for too-short signals.

    `dimension`=2, `min_num_samples`=`min_n` (default 0.1 s of samples) so a
    short overlap stream trips the NaN/skip path deterministically.
    """

    sample_rate = SR

    def __init__(self, table: dict[float, list[float]], min_n: int | None = None):
        self.table = table
        self._min_n = min_n if min_n is not None else int(0.1 * SR)

    @property
    def dimension(self) -> int:
        return 2

    @property
    def min_num_samples(self) -> int:
        return self._min_n

    def __call__(self, wav: torch.Tensor) -> np.ndarray:
        flat = wav.reshape(-1).numpy()
        # Count non-zero usable samples like the real wrapper's mask path.
        if int(np.count_nonzero(flat)) < self._min_n or len(flat) < self._min_n:
            return np.full((1, 2), np.nan, dtype=np.float32)
        key = round(float(flat[np.nonzero(flat)[0][0]]), 4)
        return np.asarray([self.table[key]], dtype=np.float32)


def _seg_df(rows):
    """rows = list of (speaker, start, end)."""
    return pd.DataFrame(
        [{"start": s, "end": e, "duration": e - s, "speaker": spk}
         for spk, s, e in rows],
        columns=["start", "end", "duration", "speaker"],
    )


def _stage(config: RelabelConfig, embedder) -> RelabelStage:
    stage = RelabelStage(config)
    stage._embedder = embedder
    stage._device = CPU
    return stage


def _ctx(seg_rows, audio, overlap_regions=None, overlap_separated=None, speakers=None):
    ctx = PipelineContext(sample_rate=SR)
    ctx.audio = audio
    ctx.enhanced_full = audio
    ctx.diarization = DiarizationResult(
        segments_df=_seg_df(seg_rows),
        overlaps_df=pd.DataFrame(columns=["start", "end", "duration"]),
        total_duration_s=len(audio) / SR,
    )
    ctx.overlap_regions = overlap_regions if overlap_regions is not None else []
    ctx.overlap_separated = overlap_separated or []
    ctx.speakers = (
        speakers if speakers is not None
        else sorted(ctx.diarization.segments_df["speaker"].unique())
    )
    return ctx


def _audio_with(marks: list[tuple[float, float, float]], total_s: float) -> np.ndarray:
    """Build audio where each (start, end, value) region is filled with `value`."""
    a = np.zeros(int(total_s * SR), dtype=np.float32)
    for s, e, v in marks:
        a[int(s * SR): int(e * SR)] = v
    return a


# ---------------------------------------------------------------------------
# Disabled = no-op
# ---------------------------------------------------------------------------


def test_disabled_is_noop():
    """enabled=False: the orchestrator skips the stage. We assert the stage
    object reports disabled and (if run anyway) would still leave the handoff
    None — but the contract is the orchestrator never calls it. Here we verify
    enabled wiring + that a disabled config produces a stage with enabled=False."""
    stage = RelabelStage(RelabelConfig(enabled=False))
    assert stage.enabled is False


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


def test_run_before_load_raises():
    stage = RelabelStage(RelabelConfig(enabled=True))
    ctx = _ctx([("A", 0.0, 1.0)], _audio_with([], 1.0))
    with pytest.raises(RuntimeError, match="before load"):
        stage.run(ctx)


def test_enhanced_source_none_asserts():
    """audio_source='enhanced' but enhanced_full is None → AssertionError
    (defence-in-depth; config cross-check normally catches this earlier)."""
    stage = _stage(RelabelConfig(enabled=True, audio_source="enhanced"),
                   _FakeEmbedder({}))
    ctx = _ctx([("A", 0.0, 1.0), ("B", 1.0, 2.0)], _audio_with([], 2.0))
    ctx.enhanced_full = None
    with pytest.raises(AssertionError, match="enhanced_full is None"):
        stage.run(ctx)


def test_one_speaker_is_noop():
    """< 2 pass-1 speakers → no-op (never invents a 2nd speaker)."""
    stage = _stage(RelabelConfig(enabled=True, audio_source="raw"),
                   _FakeEmbedder({0.5: [1.0, 0.0]}))
    audio = _audio_with([(0.0, 1.0, 0.5)], 1.0)
    ctx = _ctx([("A", 0.0, 1.0)], audio, speakers=["A"])
    before = ctx.diarization.segments_df["speaker"].tolist()
    stage.run(ctx)
    assert ctx.diarization.segments_df["speaker"].tolist() == before
    assert ctx.overlap_speaker_assignment is None


def test_single_pass1_speaker_among_usable_is_noop():
    """Both pass-1 speakers exist, but only ONE has usable (long-enough) solos —
    the other's solos are all too short to embed. Seeding a 2-means needs both
    centroids populated, so this is a visible no-op (never an empty-centroid NaN
    or a 1-vs-rest collapse), labels kept."""
    short = 0.05
    table = {0.5: [1.0, 0.0], 0.6: [0.9, 0.1], -0.5: [0.0, 1.0]}
    stage = _stage(RelabelConfig(enabled=True, audio_source="raw"),
                   _FakeEmbedder(table))
    audio = _audio_with([
        (0.0, 2.0, 0.5),                   # A usable
        (2.0, 4.0, 0.6),                   # A usable
        (4.0, 4.0 + short, -0.5),          # B but too short → no usable B solo
    ], 5.0)
    ctx = _ctx([("A", 0.0, 2.0), ("A", 2.0, 4.0), ("B", 4.0, 4.0 + short)], audio)
    before = ctx.diarization.segments_df["speaker"].tolist()
    stage.run(ctx)
    assert ctx.diarization.segments_df["speaker"].tolist() == before


def test_too_few_usable_segments_is_noop():
    """All solo spans below min_num_samples → < 2 usable → no-op, labels kept."""
    short = 0.05  # below the 0.1 s floor
    stage = _stage(RelabelConfig(enabled=True, audio_source="raw"),
                   _FakeEmbedder({0.5: [1.0, 0.0], -0.5: [0.0, 1.0]}))
    audio = _audio_with([(0.0, short, 0.5), (1.0, 1.0 + short, -0.5)], 2.0)
    ctx = _ctx([("A", 0.0, short), ("B", 1.0, 1.0 + short)], audio)
    before = ctx.diarization.segments_df["speaker"].tolist()
    stage.run(ctx)
    assert ctx.diarization.segments_df["speaker"].tolist() == before


# ---------------------------------------------------------------------------
# B (solos) — the db15fc57 miniature
# ---------------------------------------------------------------------------


def test_solos_relabel_moves_mislabeled_segment():
    """4 segments: two long A-turns and one long B-turn cluster correctly, but a
    short segment pass-1-labelled B is geometrically in A. Relabel must flip the
    short one to A and leave the long turns unchanged (db15fc57 in miniature)."""
    # Embeddings: A-voice = [1,0], B-voice = [0,1]. The mislabeled short segment
    # has an A-voice embedding but a pass-1 'B' label.
    table = {0.5: [1.0, 0.0], -0.5: [0.0, 1.0], 0.25: [0.95, 0.05]}
    stage = _stage(RelabelConfig(enabled=True, audio_source="raw"),
                   _FakeEmbedder(table))
    audio = _audio_with([
        (0.0, 2.0, 0.5),     # A long
        (2.0, 4.0, -0.5),    # B long
        (4.0, 6.0, 0.5),     # A long
        (6.0, 8.5, 0.25),    # mislabeled: A-voice, pass-1 'B'
    ], 9.0)
    ctx = _ctx([
        ("A", 0.0, 2.0), ("B", 2.0, 4.0), ("A", 4.0, 6.0), ("B", 6.0, 8.5),
    ], audio)
    stage.run(ctx)
    labels = ctx.diarization.segments_df["speaker"].tolist()
    # Long turns keep identity; the short mislabeled one flips to A.
    assert labels[0] == "A" and labels[2] == "A"
    assert labels[1] == "B"
    assert labels[3] == "A"


def test_clean_two_speaker_relabel_does_not_collapse():
    """REGRESSION (end-to-end). A clean, BALANCED 2-speaker recording (many A and
    B solos, each correctly pass-1-labelled) must come out of the relabel still
    balanced — not collapsed onto one speaker. This is the stage-level shape of
    the dr_refine catastrophe: the buggy 1-vs-rest clustering overwrote almost
    every solo with a single label, leaving one stream nearly empty."""
    # Two well-separated voices; alternating A/B turns, all pass-1-correct.
    table = {0.5: [1.0, 0.0], -0.5: [0.0, 1.0]}
    stage = _stage(RelabelConfig(enabled=True, audio_source="raw"),
                   _FakeEmbedder(table))
    marks, rows = [], []
    for i in range(10):
        s = float(2 * i)
        if i % 2 == 0:
            marks.append((s, s + 2.0, 0.5)); rows.append(("A", s, s + 2.0))
        else:
            marks.append((s, s + 2.0, -0.5)); rows.append(("B", s, s + 2.0))
    ctx = _ctx(rows, _audio_with(marks, 21.0))
    stage.run(ctx)
    labels = ctx.diarization.segments_df["speaker"].tolist()
    counts = {spk: labels.count(spk) for spk in ("A", "B")}
    # Both speakers keep roughly half the segments — never a 9-vs-1 collapse.
    assert min(counts.values()) >= 4, f"relabel collapsed to {counts}"
    # And it stayed correct: even-index turns A, odd-index turns B.
    assert all(labels[i] == "A" for i in range(0, 10, 2))
    assert all(labels[i] == "B" for i in range(1, 10, 2))


def test_label_alignment_preserves_identity_set():
    """set(after) == set(before): alignment maps clusters back onto the same two
    pass-1 label strings (never fresh ids), so ctx.speakers stays valid."""
    table = {0.5: [1.0, 0.0], -0.5: [0.0, 1.0]}
    stage = _stage(RelabelConfig(enabled=True, audio_source="raw"),
                   _FakeEmbedder(table))
    audio = _audio_with([(0.0, 2.0, 0.5), (2.0, 4.0, -0.5)], 4.0)
    ctx = _ctx([("A", 0.0, 2.0), ("B", 2.0, 4.0)], audio)
    before = set(ctx.diarization.segments_df["speaker"])
    stage.run(ctx)
    after = set(ctx.diarization.segments_df["speaker"])
    assert after == before


def test_short_segments_keep_pass1_label():
    """A sub-min solo segment (NaN embedding) keeps its pass-1 label while the
    usable segments are reclustered around it."""
    short = 0.05
    table = {0.5: [1.0, 0.0], -0.5: [0.0, 1.0], 0.9: [1.0, 0.0]}
    stage = _stage(RelabelConfig(enabled=True, audio_source="raw"),
                   _FakeEmbedder(table))
    audio = _audio_with([
        (0.0, 2.0, 0.5),                    # A usable
        (2.0, 4.0, -0.5),                   # B usable
        (4.0, 4.0 + short, 0.9),            # too short → keep pass-1 label
    ], 5.0)
    ctx = _ctx([("A", 0.0, 2.0), ("B", 2.0, 4.0), ("B", 4.0, 4.0 + short)], audio)
    stage.run(ctx)
    labels = ctx.diarization.segments_df["speaker"].tolist()
    assert labels[2] == "B"   # untouched (kept pass-1) — never NaN-relabelled


def test_solos_does_not_set_overlap_assignment():
    """source='solos' (B) never writes ctx.overlap_speaker_assignment."""
    table = {0.5: [1.0, 0.0], -0.5: [0.0, 1.0]}
    stage = _stage(RelabelConfig(enabled=True, source="solos", audio_source="raw"),
                   _FakeEmbedder(table))
    audio = _audio_with([(0.0, 2.0, 0.5), (2.0, 4.0, -0.5)], 4.0)
    ctx = _ctx([("A", 0.0, 2.0), ("B", 2.0, 4.0)], audio)
    stage.run(ctx)
    assert ctx.overlap_speaker_assignment is None


# ---------------------------------------------------------------------------
# Overlap-exclusion span derivation
# ---------------------------------------------------------------------------


def test_overlap_exclusion_span():
    """A solo segment overlapping a routing region yields the difference spans
    (the overlap frames are excluded from the embedding span)."""
    seg_df = _seg_df([("A", 0.0, 5.0)])
    spans = _solo_embedding_spans(seg_df, [(2.0, 3.0)], exclude_overlap=True)
    assert len(spans) == 1
    pos, kept, label = spans[0]
    assert pos == 0 and label == "A"
    assert kept == [(0.0, 2.0), (3.0, 5.0)]


def test_overlap_exclusion_disabled_keeps_full_span():
    seg_df = _seg_df([("A", 0.0, 5.0)])
    spans = _solo_embedding_spans(seg_df, [(2.0, 3.0)], exclude_overlap=False)
    assert spans[0][1] == [(0.0, 5.0)]


# ---------------------------------------------------------------------------
# B+ (global) — overlap stream assignment
# ---------------------------------------------------------------------------


def _gated_ovl(s1_val, s2_val, n_s=0.5):
    """An overlap_separated entry with VAD-gated streams of `n_s` seconds."""
    n = int(n_s * SR)
    return {
        "s1_gated": np.full(n, s1_val, dtype=np.float32),
        "s2_gated": np.full(n, s2_val, dtype=np.float32),
    }


def test_global_emits_overlap_assignment():
    """B+: solos cluster into A/B, and the overlap streams (s1=A-voice,
    s2=B-voice) emit 'straight' for that overlap, keyed by i_ovl."""
    table = {0.5: [1.0, 0.0], -0.5: [0.0, 1.0]}
    stage = _stage(
        RelabelConfig(enabled=True, source="global", audio_source="raw"),
        _FakeEmbedder(table),
    )
    audio = _audio_with([(0.0, 2.0, 0.5), (2.0, 4.0, -0.5)], 5.0)
    ctx = _ctx(
        [("A", 0.0, 2.0), ("B", 2.0, 4.0)], audio,
        overlap_separated=[_gated_ovl(0.5, -0.5)],   # s1→A, s2→B
    )
    stage.run(ctx)
    assert ctx.overlap_speaker_assignment == {0: "straight"}


def test_global_emits_swapped_when_streams_flipped():
    table = {0.5: [1.0, 0.0], -0.5: [0.0, 1.0]}
    stage = _stage(
        RelabelConfig(enabled=True, source="global", audio_source="raw"),
        _FakeEmbedder(table),
    )
    audio = _audio_with([(0.0, 2.0, 0.5), (2.0, 4.0, -0.5)], 5.0)
    ctx = _ctx(
        [("A", 0.0, 2.0), ("B", 2.0, 4.0)], audio,
        overlap_separated=[_gated_ovl(-0.5, 0.5)],   # s1→B, s2→A
    )
    stage.run(ctx)
    assert ctx.overlap_speaker_assignment == {0: "swapped"}


def test_global_substream_too_short_omitted():
    """A separated stream below min_num_samples → that overlap omitted from the
    assignment dict (assembly anchor decides it)."""
    table = {0.5: [1.0, 0.0], -0.5: [0.0, 1.0]}
    stage = _stage(
        RelabelConfig(enabled=True, source="global", audio_source="raw"),
        _FakeEmbedder(table),
    )
    audio = _audio_with([(0.0, 2.0, 0.5), (2.0, 4.0, -0.5)], 5.0)
    # 0.05 s streams < the 0.1 s floor → both NaN → overlap incomplete → omitted.
    ctx = _ctx(
        [("A", 0.0, 2.0), ("B", 2.0, 4.0)], audio,
        overlap_separated=[_gated_ovl(0.5, -0.5, n_s=0.05)],
    )
    stage.run(ctx)
    assert ctx.overlap_speaker_assignment == {}


def test_global_both_streams_same_cluster_omitted():
    """Degenerate: both overlap streams cluster to the same speaker → omitted."""
    table = {0.5: [1.0, 0.0], -0.5: [0.0, 1.0], 0.6: [1.0, 0.0]}
    stage = _stage(
        RelabelConfig(enabled=True, source="global", audio_source="raw"),
        _FakeEmbedder(table),
    )
    audio = _audio_with([(0.0, 2.0, 0.5), (2.0, 4.0, -0.5)], 5.0)
    # Both streams are A-voice → land in the same cluster → degenerate.
    ctx = _ctx(
        [("A", 0.0, 2.0), ("B", 2.0, 4.0)], audio,
        overlap_separated=[_gated_ovl(0.5, 0.6)],
    )
    stage.run(ctx)
    assert ctx.overlap_speaker_assignment == {}


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_cluster_two_splits():
    emb = np.array([[1.0, 0.0], [0.95, 0.05], [0.0, 1.0], [0.05, 0.95]],
                   dtype=np.float32)
    w = np.ones(4)
    seed = np.array([0, 0, 1, 1])
    labels = _cluster_two(emb, seed, duration_weighted=False, weights=w)
    assert set(labels) == {0, 1}
    # The two A-ish points share a cluster, distinct from the two B-ish points.
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]


def test_cluster_two_balanced_split_not_outlier_peel():
    """REGRESSION (the dr_refine catastrophe). On a clearly-bimodal but only
    *moderately* separated point set — two tight Gaussian blobs whose inter-blob
    cosine sim is well above 0, like real ECAPA2 solos — the previous centroid-/
    average-linkage path peeled a single outlier and returned a 1-vs-(N-1) split,
    collapsing almost every solo onto one speaker (observed 26-vs-1 / 16-vs-1 on
    real fragments). Seeded 2-means must instead return a BALANCED split that
    respects the two blobs."""
    rng = np.random.default_rng(0)
    dim = 192
    # Two distinct unit-norm anchor directions, blobs tight around each. The
    # blobs overlap enough that a chaining linkage would peel an outlier.
    a = rng.standard_normal(dim); a /= np.linalg.norm(a)
    b = rng.standard_normal(dim); b /= np.linalg.norm(b)
    n_a, n_b = 14, 13
    pts_a = a + 0.6 * rng.standard_normal((n_a, dim))
    pts_b = b + 0.6 * rng.standard_normal((n_b, dim))
    emb = np.concatenate([pts_a, pts_b]).astype(np.float32)
    seed = np.array([0] * n_a + [1] * n_b)
    w = np.ones(len(emb))
    labels = _cluster_two(emb, seed, duration_weighted=False, weights=w)
    sizes = np.bincount(labels, minlength=2)
    # Both clusters must be substantial — never the 1-vs-rest degeneracy.
    assert sizes.min() >= 5, f"degenerate split {sizes.tolist()} (1-vs-rest bug)"
    # And the split must track the two blobs (most a-points together, etc.).
    assert (labels[:n_a] == labels[0]).mean() > 0.8
    assert (labels[n_a:] == labels[n_a]).mean() > 0.8
    assert labels[0] != labels[n_a]


def test_cluster_two_overlap_points_assigned_not_seeding():
    """B+ overlap rows carry seed -1: they must be assigned to the nearest of the
    two pass-1 centroids, never left as a third (-1) label."""
    emb = np.array(
        [[1.0, 0.0], [0.9, 0.1],        # pass-1 A solos
         [0.0, 1.0], [0.1, 0.9],        # pass-1 B solos
         [0.95, 0.05], [0.05, 0.95]],   # overlap streams (no seed) near A / B
        dtype=np.float32,
    )
    seed = np.array([0, 0, 1, 1, -1, -1])
    w = np.ones(len(emb))
    labels = _cluster_two(emb, seed, duration_weighted=False, weights=w)
    assert set(labels) == {0, 1}                 # only two clusters, no -1
    assert labels[4] == labels[0]                # A-ish overlap → A cluster
    assert labels[5] == labels[2]                # B-ish overlap → B cluster


def test_align_to_old_uses_duration():
    """Alignment picks the permutation maximising duration-weighted agreement —
    one long correct anchor pins identity against a short wrong one."""
    clusters = np.array([0, 0, 1])
    pass1 = ["A", "B", "B"]               # cluster 0 has A(long) + B(short)
    durations = np.array([10.0, 0.5, 5.0])  # the long point is the A one
    mapping = _align_to_old(clusters, pass1, durations, ["A", "B"])
    # Straight (0→A,1→B): agreement = 10(A✓) + 5(B✓) = 15. Swapped = 0.5. Straight wins.
    assert mapping == {0: "A", 1: "B"}


def test_overlap_pairings_keys_and_values():
    meta = [(0, "s1"), (0, "s2"), (1, "s1"), (1, "s2")]
    clusters = np.array([0, 1, 1, 0])     # ovl0: s1→0,s2→1 ; ovl1: s1→1,s2→0
    mapping = {0: "A", 1: "B"}
    pairings = _overlap_pairings(meta, clusters, mapping, ["A", "B"])
    assert pairings == {0: "straight", 1: "swapped"}


def test_overlap_pairings_same_cluster_omitted():
    meta = [(0, "s1"), (0, "s2")]
    clusters = np.array([0, 0])           # both to cluster 0 → degenerate
    pairings = _overlap_pairings(meta, clusters, {0: "A", 1: "B"}, ["A", "B"])
    assert pairings == {}


# ---------------------------------------------------------------------------
# Run-level (contiguous-run) relabel pass — _run_level_flips
# ---------------------------------------------------------------------------


def test_run_level_flips_contiguous_misrouted_run():
    """A contiguous run of A-voice solos filed to stream B (the class-A chunk
    swap): its mean [1,0] is closer to the clean A centroid than to the
    contaminated B centroid → the whole run flips. Real B / A runs stay put."""
    # time order streams: B, A, A, [B B B]=misrouted A-voice, A, A, B.
    emb = np.array([
        [0.0, 1.0],   # 0 real B
        [1.0, 0.0],   # 1 real A
        [1.0, 0.0],   # 2 real A
        [1.0, 0.0],   # 3 misrouted (A-voice, labelled B)
        [1.0, 0.0],   # 4 misrouted
        [1.0, 0.0],   # 5 misrouted
        [1.0, 0.0],   # 6 real A
        [1.0, 0.0],   # 7 real A
        [0.0, 1.0],   # 8 real B
    ], dtype=np.float32)
    stream = ["B", "A", "A", "B", "B", "B", "A", "A", "B"]
    flips = _run_level_flips(np.arange(9), emb, stream, ["A", "B"], run_margin=0.05)
    assert flips == {3, 4, 5}


def test_run_level_below_margin_run_untouched():
    """Same geometry, but a margin (0.5) wider than the run's ~0.168 centroid gap
    leaves the run untouched — the flip only fires on a clear pull."""
    emb = np.array([
        [0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0],
        [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0],
    ], dtype=np.float32)
    stream = ["B", "A", "A", "B", "B", "B", "A", "A", "B"]
    flips = _run_level_flips(np.arange(9), emb, stream, ["A", "B"], run_margin=0.5)
    assert flips == set()


def test_run_level_whole_stream_run_not_flipped():
    """A run that IS its entire stream is never flipped (the flip-everything
    guard). Here stream B = a single contiguous A-voice run; a whole-stream flip
    would just be a free global swap and empty B — so it is skipped."""
    emb = np.array([
        [0.0, 1.0], [0.0, 1.0],   # A stream (B-voice embeddings, but that is fine)
        [1.0, 0.0], [1.0, 0.0],   # B stream = entire, A-voice
        [0.0, 1.0], [0.0, 1.0],   # A stream
    ], dtype=np.float32)
    stream = ["A", "A", "B", "B", "A", "A"]
    flips = _run_level_flips(np.arange(6), emb, stream, ["A", "B"], run_margin=0.05)
    assert flips == set()


def test_run_level_empty_stream_no_flips():
    """One stream has no members → nothing to compare against → no flips."""
    emb = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    stream = ["A", "A"]
    flips = _run_level_flips(np.arange(2), emb, stream, ["A", "B"], run_margin=0.05)
    assert flips == set()


def test_run_level_deterministic():
    """Same input twice → identical flip set (fixed centroids, fixed order)."""
    emb = np.array([
        [0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0],
        [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0],
    ], dtype=np.float32)
    stream = ["B", "A", "A", "B", "B", "B", "A", "A", "B"]
    f1 = _run_level_flips(np.arange(9), emb, stream, ["A", "B"], 0.05)
    f2 = _run_level_flips(np.arange(9), emb, stream, ["A", "B"], 0.05)
    assert f1 == f2 == {3, 4, 5}


def test_run_level_stage_threaded_and_safe(monkeypatch):
    """End-to-end wiring: run_level=True is threaded through RelabelStage.run and
    is SAFE — it never corrupts a case the global relabel already resolves. On the
    db15fc57 miniature the run-level pass runs but changes nothing (the global
    2-means already produced a self-consistent labelling), so labels match the
    run_level=False result exactly."""
    table = {0.5: [1.0, 0.0], -0.5: [0.0, 1.0], 0.25: [0.95, 0.05]}
    marks = [(0.0, 2.0, 0.5), (2.0, 4.0, -0.5), (4.0, 6.0, 0.5), (6.0, 8.5, 0.25)]
    rows = [("A", 0.0, 2.0), ("B", 2.0, 4.0), ("A", 4.0, 6.0), ("B", 6.0, 8.5)]

    def _labels(run_level):
        stage = _stage(RelabelConfig(enabled=True, audio_source="raw",
                                     run_level=run_level), _FakeEmbedder(table))
        ctx = _ctx(rows, _audio_with(marks, 9.0))
        stage.run(ctx)
        return ctx.diarization.segments_df["speaker"].tolist()

    assert _labels(True) == _labels(False) == ["A", "B", "A", "A"]


# ---------------------------------------------------------------------------
# Degeneracy rescue (solo_clustering_init="rescue")
# ---------------------------------------------------------------------------


def _degenerate_scenario():
    """Two genuine, duration-balanced voice groups (A-voice + B-voice) plus one
    short far-outlier piece, with ADVERSARIAL pass-1 labels: every genuine piece is
    lumped into one pass-1 speaker and only the outlier carries the other. The
    pass-1-seeded 2-means then freezes into the outlier-peel — {10 genuine} vs
    {outlier} — a fixed point whose min-cluster duration share is ~0.015. Returns
    (table, marks, rows)."""
    # A-voice = [1,0], B-voice = [0,1], outlier = [-1,-1] (cos -0.71 to both).
    table = {0.5: [1.0, 0.0], -0.5: [0.0, 1.0], 0.9: [-1.0, -1.0]}
    marks, rows = [], []
    for i in range(5):                       # 5 A-voice pieces, pass-1 = SPK0
        s = 2.0 * i
        marks.append((s, s + 2.0, 0.5)); rows.append(("SPK0", s, s + 2.0))
    for i in range(5, 10):                   # 5 B-voice pieces, ALSO pass-1 = SPK0
        s = 2.0 * i
        marks.append((s, s + 2.0, -0.5)); rows.append(("SPK0", s, s + 2.0))
    marks.append((20.0, 20.3, 0.9)); rows.append(("SPK1", 20.0, 20.3))  # outlier
    return table, marks, rows


def _rescue_labels(init, table, marks, rows, total_s=20.5):
    stage = _stage(
        RelabelConfig(enabled=True, audio_source="raw", solo_clustering_init=init),
        _FakeEmbedder(table),
    )
    ctx = _ctx(rows, _audio_with(marks, total_s))
    stage.run(ctx)
    return ctx.diarization.segments_df["speaker"].tolist()


def test_rescue_adopts_balanced_partition_on_degenerate_pass1():
    """The core case. pass-1 seeding converges to the outlier-peel (one pseudo-
    speaker holds a single piece); the rescue escapes it to a balanced 2-speaker
    split. Contrast the two inits on the SAME geometry."""
    table, marks, rows = _degenerate_scenario()

    pass1 = _rescue_labels("pass1", table, marks, rows)
    rescue = _rescue_labels("rescue", table, marks, rows)

    p1_counts = {spk: pass1.count(spk) for spk in ("SPK0", "SPK1")}
    rc_counts = {spk: rescue.count(spk) for spk in ("SPK0", "SPK1")}
    # pass-1 seed = the corrupt peel: 10 genuine vs the lone outlier.
    assert min(p1_counts.values()) == 1, f"pass1 not degenerate: {p1_counts}"
    # rescue = a balanced split (each genuine voice group becomes a speaker).
    assert min(rc_counts.values()) >= 4, f"rescue not balanced: {rc_counts}"


def test_rescue_no_candidate_keeps_original():
    """Degenerate trigger but NO balanced alternative: a single genuine voice that
    pass-1 wrongly split as 2 speakers, plus a short outlier. The trigger fires, but
    every pair-seeded fixed point is ALSO unbalanced (there is no real 2nd cluster),
    so the original (degenerate) partition is kept — identical to the pass-1 run."""
    # Only one real voice ([1,0], 10 pieces, pass-1 = SPK0) + a short outlier.
    table = {0.5: [1.0, 0.0], 0.9: [-1.0, -1.0]}
    marks, rows = [], []
    for i in range(10):
        s = 2.0 * i
        marks.append((s, s + 2.0, 0.5)); rows.append(("SPK0", s, s + 2.0))
    marks.append((20.0, 20.3, 0.9)); rows.append(("SPK1", 20.0, 20.3))

    rescue = _rescue_labels("rescue", table, marks, rows)
    pass1 = _rescue_labels("pass1", table, marks, rows)
    assert rescue == pass1                                   # no-op: original kept
    assert min(rescue.count(s) for s in ("SPK0", "SPK1")) == 1


def test_rescue_deterministic():
    """Two runs of the rescue on the same degenerate geometry produce the identical
    labelling (fixed pair-enumeration order, first-found J_d tie-break, no RNG)."""
    table, marks, rows = _degenerate_scenario()
    assert _rescue_labels("rescue", table, marks, rows) == \
        _rescue_labels("rescue", table, marks, rows)


def test_pass1_default_is_byte_identical():
    """The default init is 'pass1' and is inert: the db15fc57 miniature produces
    the identical labelling whether the knob is left at its default or set
    explicitly to 'pass1' — the shipped behaviour is untouched."""
    table = {0.5: [1.0, 0.0], -0.5: [0.0, 1.0], 0.25: [0.95, 0.05]}
    marks = [(0.0, 2.0, 0.5), (2.0, 4.0, -0.5), (4.0, 6.0, 0.5), (6.0, 8.5, 0.25)]
    rows = [("A", 0.0, 2.0), ("B", 2.0, 4.0), ("A", 4.0, 6.0), ("B", 6.0, 8.5)]

    def _labels(**kw):
        stage = _stage(RelabelConfig(enabled=True, audio_source="raw", **kw),
                       _FakeEmbedder(table))
        ctx = _ctx(rows, _audio_with(marks, 9.0))
        stage.run(ctx)
        return ctx.diarization.segments_df["speaker"].tolist()

    assert _labels() == _labels(solo_clustering_init="pass1") == ["A", "B", "A", "A"]


def test_rescue_leaves_clean_balanced_recording_untouched():
    """do-no-harm: a clean, balanced 2-speaker recording never trips the trigger,
    so solo_clustering_init='rescue' produces the identical labelling as 'pass1'."""
    table = {0.5: [1.0, 0.0], -0.5: [0.0, 1.0]}
    marks, rows = [], []
    for i in range(10):
        s = float(2 * i)
        if i % 2 == 0:
            marks.append((s, s + 2.0, 0.5)); rows.append(("A", s, s + 2.0))
        else:
            marks.append((s, s + 2.0, -0.5)); rows.append(("B", s, s + 2.0))
    assert _rescue_labels("rescue", table, marks, rows, total_s=21.0) == \
        _rescue_labels("pass1", table, marks, rows, total_s=21.0)


def test_unknown_solo_clustering_init_raises():
    """A typo in solo_clustering_init fails loud at stage init (ValueError)."""
    with pytest.raises(ValueError, match="solo_clustering_init"):
        RelabelStage(RelabelConfig(solo_clustering_init="kmeanspp"))


# ---------------------------------------------------------------------------
# load_signature
# ---------------------------------------------------------------------------


def test_load_signature_tracks_embedding_only():
    s1 = RelabelStage(RelabelConfig(embedding="ecapa2", source="solos"))
    s2 = RelabelStage(RelabelConfig(embedding="ecapa2", source="global",
                                    audio_source="raw", duration_weighted=True))
    # source / audio_source / duration_weighted are runtime knobs → same sig.
    assert s1.load_signature() == s2.load_signature() == ("ecapa2",)
    s3 = RelabelStage(RelabelConfig(embedding="eres2netv2"))
    assert s3.load_signature() != s1.load_signature()
