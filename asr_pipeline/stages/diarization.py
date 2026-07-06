"""Stage 1 — speaker diarization (pyannote or Sortformer EEND).

Ported from `asr/archive/asr_pipeline.ipynb` cell 10. The default `pyannote`
backend runs `speaker-diarization-3.1` on mono 16 kHz audio, constrained to two
speakers, and emits per-speaker segments + an overlap timeline. The alternative
`sortformer` backend (config `backend: sortformer`) runs NVIDIA Sortformer-v1
offline EEND — a clustering-free diarizer — via an isolated-venv subprocess and
adapts its per-frame activity into the SAME `DiarizationResult` (segments +
overlap timeline), so everything downstream is unchanged. See
`build_sortformer_annotation` / `sortformer_turns_from_probs` below and
`scripts/sortformer_worker.py`.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from asr_pipeline.config import DiarizationConfig
from asr_pipeline.context import DiarizationResult, PipelineContext
from asr_pipeline.debug_log import dlog
from asr_pipeline.stages.assembly import _coalesce, _subtract
from asr_pipeline.stages.base import Stage
from asr_pipeline.stages.custom_embeddings import (
    CUSTOM_EMBEDDING_NAMES,
    build_custom_embedding,
    embed_intervals,
    with_ecapa2,
)


def _log(msg: str) -> None:
    """Progress message — stdout + the durable debug log. Matters here because
    pyannote's load and forward are multi-minute and the WSL stdout bridge can
    drop; every other model-bearing stage logs the same way."""
    dlog("diarization", msg)


# ---------------------------------------------------------------------------
# Sortformer (EEND) backend — isolated-venv worker + turn adapter
# ---------------------------------------------------------------------------

# Isolated-venv Sortformer worker (NVIDIA NeMo offline EEND). Invoked under
# $SORTFORMER_VENV_PY, NOT the main venv — see scripts/sortformer_worker.py for
# the isolation rationale + venv recipe (same precedent as the CohereX worker).
_SORTFORMER_WORKER = (
    Path(__file__).resolve().parents[2] / "scripts" / "sortformer_worker.py"
)

# Turn-building constants, ported VERBATIM from docs/sweep_plan/eend_probe.py
# (MAX_GAP / MIN_DUR / SPK_ACTIVE_FRAC) — the exact params that produced the
# probe's scored purity (EEND_PROBE.md). Fixed module-level, not config knobs:
# only the activity threshold is exposed (diarization.sortformer_threshold).
_SF_MAX_GAP_S = 0.10        # merge active runs separated by <= this (fill jitter)
_SF_MIN_DUR_S = 0.10        # drop active runs shorter than this
_SF_SPK_ACTIVE_FRAC = 0.05  # a head counts as a "speaker" if active > 5% of speech
_SF_LEAK_WARN_FRAC = 0.05   # loud miscount alarm: heads 3/4 carry > 5% of speech

# v4.1 L1 merge-not-discard (V41_PREREG.md). Surplus-head runs shorter than the
# floor stay discarded — embeddings on a fraction of a syllable are unreliable
# (anatomy MERGE verdicts were all on multi-burst multi-second heads). The solo
# reference for each top-2 speaker is capped so a long recording's solo concat
# stays a stable, bounded ECAPA2 forward.
_SF_MERGE_MIN_DUR_S = 0.4     # min surplus-run duration (s) eligible for L1 merge
_SF_SOLO_EMBED_CAP_S = 20.0   # cap on top-2 solo speech concatenated for the ref embedding


def _runs_from_mask(mask, fs: float, max_gap: float, min_dur: float):
    """bool 1-D activity mask -> list of (start_s, end_s) after gap-fill +
    min-duration drop. Ported VERBATIM from
    docs/sweep_plan/eend_probe.py::_runs_from_mask."""
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return []
    runs = []
    s = idx[0]
    p = idx[0]
    for i in idx[1:]:
        if (i - p) * fs <= max_gap + 1e-9:
            p = i
            continue
        runs.append((s * fs, (p + 1) * fs))
        s = i
        p = i
    runs.append((s * fs, (p + 1) * fs))
    return [(a, b) for a, b in runs if (b - a) >= min_dur - 1e-9]


def sortformer_turns_from_probs(probs, frame_rate_s: float, threshold: float):
    """(T, S) sigmoid speaker-activity -> (turns, diag).

    Ported from docs/sweep_plan/eend_probe.py::candidate_metrics (the turn +
    head-selection half; the probe's diar_purity scoring stays in the probe).
    Binarize the activity at ``threshold``, gap-fill + min-duration each head's
    runs, pick the 2 most-active heads as our 2 speakers, and label their runs.
    Returns:

      turns : list of (label, start_s, end_s) for the 2 selected heads. Labels
              are "SPEAKER_00"/"SPEAKER_01" (most-active head first) so downstream
              — which treats speaker ids as opaque, sorted strings for the A/B
              letter — sees a clean 2-speaker set exactly like the pyannote path.
      diag  : {"top2", "n_spk", "leak", "head_dur_s", "speech_s", "head_runs",
              "frame_rate_s"} where `leak` is the activity duration OUTSIDE the
              top-2 heads as a fraction of total speech (the head-3/4 Polish-OOD
              miscount signal), `n_spk` the # heads active > 5% of speech, and
              `head_runs` the per-head (gap-filled, min-dur'd) run lists the L1
              merge lever consumes.
    """
    probs = np.asarray(probs, dtype=np.float32)
    fs = float(frame_rate_s)
    S = probs.shape[1]
    mask = probs >= threshold                       # (T, S)
    # per-head runs + active duration (after smoothing)
    head_runs = [
        _runs_from_mask(mask[:, k], fs, _SF_MAX_GAP_S, _SF_MIN_DUR_S)
        for k in range(S)
    ]
    head_dur = np.array([sum(b - a for a, b in r) for r in head_runs])
    # total speech = union of any-head-active frames (smoothed)
    any_runs = _runs_from_mask(mask.any(axis=1), fs, _SF_MAX_GAP_S, _SF_MIN_DUR_S)
    speech_tot = sum(b - a for a, b in any_runs) or 1e-9
    # speaker count: heads active > 5% of speech
    n_spk = int((head_dur > _SF_SPK_ACTIVE_FRAC * speech_tot).sum())
    # top-2 heads by active duration = our 2 speakers (num_speakers==2 prior)
    order = np.argsort(-head_dur)
    top2 = [int(order[0]), int(order[1])] if S >= 2 else [int(order[0])]
    # head3/4 leak = active duration outside the top-2 as a fraction of speech
    leak = float(
        head_dur[[k for k in range(S) if k not in top2]].sum() / speech_tot
    )
    labels = {top2[0]: "SPEAKER_00"}
    if len(top2) >= 2:
        labels[top2[1]] = "SPEAKER_01"
    turns = []
    for k in top2:
        for a, b in head_runs[k]:
            turns.append((labels[k], a, b))
    diag = {
        "top2": top2,
        "n_spk": n_spk,
        "leak": leak,
        "head_dur_s": head_dur.tolist(),
        "speech_s": float(speech_tot),
        # v4.1 L1 needs the per-head runs (surplus runs to reassign) + fs; the
        # binarized mask is reconstructable from these. Kept out of the serialized
        # census diag (the stage copies only scalar fields into metadata).
        "head_runs": head_runs,
        "frame_rate_s": fs,
    }
    return turns, diag


def build_sortformer_annotation(turns):
    """turns [(label, start_s, end_s)] -> a pyannote.core.Annotation.

    Overlapping intervals between the two speaker labels encode overlap NATURALLY,
    so the sortformer path reuses the SAME conversion the pyannote path uses —
    `diar_to_segments_df(annotation)` + `annotation.get_overlap()` — giving an
    identical DiarizationResult shape and diarization.json format (verified against
    io.write_pipeline_outputs, which reads segments_df; routing reads get_overlap
    via overlaps_df). Each turn gets a distinct track key so two heads that happen
    to share an exact (start, end) never collide. (The probe derived overlap from
    the raw per-frame AND of the two heads; deriving it from the built Annotation
    instead keeps the downstream code path byte-identical to pyannote and differs
    only by second-order gap-fill/min-dur effects.)"""
    from pyannote.core import Annotation, Segment

    ann = Annotation()
    for i, (label, start, end) in enumerate(turns):
        if end > start:
            ann[Segment(start, end), i] = label
    return ann


# ---------------------------------------------------------------------------
# Surplus-head merge (V41_PREREG.md L1)
# ---------------------------------------------------------------------------


def _cosine(a, b) -> float:
    """Cosine similarity of two 1-D vectors (0.0 for a degenerate zero vector)."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _merge_surplus_heads(
    head_runs,
    top2,
    labels,
    audio,
    sr: int,
    embedder,
    *,
    merge_margin: float,
    speech_tot: float,
    min_dur_s: float = _SF_MERGE_MIN_DUR_S,
    solo_cap_s: float = _SF_SOLO_EMBED_CAP_S,
):
    """L1 merge-not-discard (V41_PREREG.md). Reassign discarded surplus-head speech.

    For each surplus head (not in `top2`) and each of its runs of duration
    >= `min_dur_s`, embed that audio span and assign it to the top-2 speaker whose
    SOLO speech (frames where ONLY that head is active) it embeds closest to — but
    only when the cosine margin (best - other) >= `merge_margin`. Below-margin runs
    stay discarded and are counted as UNRESOLVED leak. Runs shorter than
    `min_dur_s` stay discarded (embeddings unreliable) and are counted separately.
    Local embedding only — NO global clustering.

    `head_runs` is the per-head run list from `sortformer_turns_from_probs`; `top2`
    the two selected head indices; `labels` maps each top-2 head → its speaker
    label. Returns `(merged_turns, stats)` where `merged_turns` is a list of
    `(label, start_s, end_s)` for the reassigned runs and `stats` carries the
    merged / unresolved / short accounting + the post-merge `unresolved_leak`
    fraction. Pure given `embedder` (deterministic).
    """
    S = len(head_runs)
    surplus = [k for k in range(S) if k not in top2]
    stats = {
        "n_merged": 0, "merged_s": 0.0,
        "n_unresolved": 0, "unresolved_s": 0.0,
        "n_short": 0, "short_s": 0.0,
        "unresolved_leak": 0.0,
        "assignments": [],
    }
    merged_turns: list = []

    # Partition surplus runs into "too short" (discard) and "eligible" (>= floor).
    eligible = []
    for k in surplus:
        for a, b in head_runs[k]:
            dur = float(b - a)
            if dur < min_dur_s - 1e-9:
                stats["n_short"] += 1
                stats["short_s"] += dur
            else:
                eligible.append((k, float(a), float(b), dur))

    if not eligible or len(top2) < 2:
        return merged_turns, stats

    # Solo reference per top-2 speaker: frames where ONLY that head is active
    # (its runs minus the union of every other head's runs), capped + embedded.
    solo_refs = {}
    for k in top2:
        others_union = _coalesce(
            [iv for j in range(S) if j != k for iv in head_runs[j]]
        )
        solo = _subtract(head_runs[k], others_union)
        solo_refs[k] = embed_intervals(solo, audio, sr, embedder, cap_s=solo_cap_s)
    have_refs = all(solo_refs[k] is not None for k in top2)

    for (k, a, b, dur) in eligible:
        run_emb = embed_intervals([(a, b)], audio, sr, embedder) if have_refs else None
        if run_emb is None:
            # No usable reference for one speaker, or an unembeddable run → cannot
            # decide → leave discarded, count as unresolved leak.
            stats["n_unresolved"] += 1
            stats["unresolved_s"] += dur
            continue
        sims = {kk: _cosine(run_emb, solo_refs[kk]) for kk in top2}
        best_k = max(top2, key=lambda kk: sims[kk])
        other_k = top2[1] if best_k == top2[0] else top2[0]
        margin = sims[best_k] - sims[other_k]
        if margin >= merge_margin:
            merged_turns.append((labels[best_k], a, b))
            stats["n_merged"] += 1
            stats["merged_s"] += dur
            stats["assignments"].append({
                "head": int(k), "start": round(a, 3), "end": round(b, 3),
                "assigned_to": labels[best_k], "margin": round(float(margin), 4),
            })
        else:
            stats["n_unresolved"] += 1
            stats["unresolved_s"] += dur

    stats["unresolved_leak"] = (
        stats["unresolved_s"] / speech_tot if speech_tot > 0 else 0.0
    )
    return merged_turns, stats


def run_sortformer_worker(venv_py: str, model_id: str, audio, sample_rate: int):
    """Shell out to scripts/sortformer_worker.py under $SORTFORMER_VENV_PY and
    return (probs (T, S) float32, frame_rate_s).

    Mirrors the CohereX backend's subprocess call: write the mono waveform to a
    temp wav, run the worker from a NEUTRAL cwd (the temp dir, OUTSIDE the repo —
    the repo's local `datasets/` package shadows HF `datasets` and breaks NeMo's
    imports), inherit env for $HF_TOKEN / CUDA, and fail loud on a non-zero exit
    (SCOPE §4 — never a silent fall-back to pyannote)."""
    import subprocess
    import tempfile

    import soundfile as sf

    with tempfile.TemporaryDirectory() as td:
        tin = str(Path(td) / "in.wav")
        tout = str(Path(td) / "out.json")
        sf.write(tin, np.asarray(audio, dtype=np.float32), sample_rate)
        cmd = [
            venv_py, str(_SORTFORMER_WORKER),
            "--in", tin, "--out", tout, "--model", model_id,
        ]
        _log(f"run: invoking sortformer worker (model={model_id})...")
        # cwd=td (a /tmp dir, OUTSIDE the repo) so the worker never resolves the
        # repo's local `datasets/` package; inherit env for HF_TOKEN / CUDA.
        proc = subprocess.run(cmd, cwd=td, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"Sortformer worker failed (exit {proc.returncode}). "
                f"stderr tail:\n{proc.stderr[-2000:]}"
            )
        with open(tout, encoding="utf-8") as f:
            payload = json.load(f)
    probs = np.asarray(payload["frame_probs"], dtype=np.float32)
    return probs, float(payload["frame_rate_s"])


def build_pyannote_pipeline(config: DiarizationConfig, device: torch.device):
    """Build a loaded pyannote diarization pipeline from a DiarizationConfig.

    Honours `config.embedding` (None = stock 3.1 via `from_pretrained`; a
    pyannote id or a custom-name wrapper otherwise) and the front-end
    instantiate hyperparameters. Fails loud (SCOPE §4) — no silent fall-back to
    pyannote when a custom embedder cannot load.
    """
    token = config.hf_token
    if not token:
        raise RuntimeError(
            "DiarizationConfig.hf_token is empty — set HF_TOKEN in the "
            "environment or put a literal value in the YAML config."
        )
    is_custom = config.embedding in CUSTOM_EMBEDDING_NAMES
    if config.embedding is None:
        from pyannote.audio import Pipeline

        _log(f"load: instantiating {config.model_id} on {device}...")
        pipeline = Pipeline.from_pretrained(config.model_id, token=token)
    else:
        # Embedding swap: reconstruct a 3.1-equivalent pipeline (its
        # segmentation-3.0 + AgglomerativeClustering + exclude_overlap
        # components) with the chosen embedder. Verified to reproduce stock
        # 3.1 when embedding == the default resnet34-LM, so the swapped
        # embedder is the only difference. NB: this hard-codes the 3.1
        # architecture — it is an embedding swap *within* the 3.1 design,
        # not a generic model_id loader.
        #
        # Two embedder families share this path:
        #   - a pyannote-format model id (e.g. wespeaker/speechbrain/...) is
        #     accepted directly by pyannote's embedding factory.
        #   - a CUSTOM name ("ecapa2"/"eres2netv2") is NOT — pyannote's
        #     factory only dispatches known prefixes. So we construct with a
        #     cheap pyannote placeholder embedder (so the pipeline builds and
        #     `instantiate` succeeds), then REPLACE `pipeline._embedding`
        #     below. The placeholder embedder is never used.
        from pyannote.audio.pipelines import SpeakerDiarization

        # Placeholder embedder for the custom path (replaced below), the real
        # model id otherwise. The placeholder must be a non-gated/already-
        # accepted, 16 kHz cosine model so construction never stalls on HF
        # auth — resnet34-LM is the stock 3.1 embedder (proven loadable with
        # our token), unlike "pyannote/embedding" which is gated (403).
        embedding_arg = (
            "pyannote/wespeaker-voxceleb-resnet34-LM"
            if is_custom else config.embedding
        )
        _log(
            f"load: reconstructing 3.1-equivalent pipeline with "
            f"embedding={config.embedding!r} on {device}..."
        )
        pipeline = SpeakerDiarization(
            segmentation="pyannote/segmentation-3.0",
            embedding=embedding_arg,
            embedding_exclude_overlap=True,
            clustering="AgglomerativeClustering",
            embedding_batch_size=32,
            segmentation_batch_size=32,
            token=token,
        )
    # Apply our exposed front-end hyperparameters. Defaults match the shipped
    # speaker-diarization-3.1 config.yaml, so at defaults this re-instantiates
    # to the identical pipeline (no behaviour change) — the point is to make
    # them sweepable. The full param dict is required by instantiate(); the
    # structure mirrors the model's config.yaml. (clustering.threshold is
    # inert under num_speakers=2 but must still be supplied here.)
    #
    # This param dict is the 3.x SCHEMA. Other diarization models (e.g.
    # `speaker-diarization-community-1`) expose a DIFFERENT instantiate schema —
    # forcing the 3.1 dict on them crashes ("parameter 'method' does not exist").
    # So apply it only to the 3.x family: a stock 3.0/3.1 `from_pretrained`, or
    # the reconstructed-3.1 embedding-swap path (which always builds
    # segmentation-3.0 + AgglomerativeClustering, i.e. the 3.x schema). A
    # non-3.x `model_id` runs with its native config — our 3.1 front-end knobs
    # (min_duration_off / min_cluster_size / threshold) simply don't apply to it.
    apply_31_schema = (
        config.embedding is not None
        or "speaker-diarization-3" in config.model_id
    )
    if apply_31_schema:
        pipeline.instantiate({
            "clustering": {
                "method": config.clustering_method,
                "min_cluster_size": config.clustering_min_cluster_size,
                "threshold": config.clustering_threshold,
            },
            "segmentation": {
                "min_duration_off": config.segmentation_min_duration_off,
            },
        })
    else:
        _log(
            f"load: {config.model_id} is not speaker-diarization-3.x — running "
            f"its native config (the 3.1 front-end knobs are not applied)."
        )
    pipeline = pipeline.to(device)
    if is_custom:
        # Inject the non-pyannote embedder, replacing the placeholder built
        # above. `Pipeline.to` does NOT touch `_embedding` (it is a plain
        # attribute, not in the moved pipeline/model/inference registries),
        # so we move the custom embedder to `device` ourselves. The custom
        # wrapper matches `PretrainedSpeakerEmbedding`'s interface
        # (__call__/dimension/sample_rate/metric/min_num_samples/to), all of
        # which `get_embeddings` relies on. Both supported custom models are
        # 16 kHz cosine, matching the placeholder's `_audio` resampler and
        # the clustering metric captured at construction. Fails loud if the
        # model can't load (SCOPE §4 — never a silent fall-back to pyannote).
        _log(f"load: injecting custom embedder {config.embedding!r}...")
        embedder = build_custom_embedding(config.embedding, device)
        pipeline._embedding = embedder
        _log(
            f"load: custom embedder ready "
            f"(dimension={embedder.dimension}, "
            f"min_num_samples={embedder.min_num_samples})"
        )
    return pipeline


def run_pyannote(pipeline, audio, sample_rate: int, num_speakers: int):
    """Run a loaded pyannote pipeline on a mono waveform → its Annotation.

    Handles pyannote 3.x (returns the Annotation directly) vs 4.x (returns a
    DiarizeOutput wrapper with `.speaker_diarization`). Shared by both passes.
    """
    waveform = torch.from_numpy(audio).unsqueeze(0)
    result = pipeline(
        {"waveform": waveform, "sample_rate": sample_rate},
        num_speakers=num_speakers,
    )
    return (
        result.speaker_diarization
        if hasattr(result, "speaker_diarization")
        else result
    )


def diar_to_segments_df(diar) -> pd.DataFrame:
    """A pyannote Annotation → the `start/end/duration/speaker` segments frame.

    Explicit columns so downstream `df["speaker"]` access works even when the
    Annotation has no tracks (e.g. silent input). Shared by both passes.
    """
    records = [
        {
            "start": round(t.start, 3),
            "end": round(t.end, 3),
            "duration": round(t.duration, 3),
            "speaker": spk,
        }
        for t, _, spk in diar.itertracks(yield_label=True)
    ]
    return pd.DataFrame(records, columns=["start", "end", "duration", "speaker"])


def _overlaps_df_from_annotation(diar) -> pd.DataFrame:
    """A pyannote Annotation → the `start/end/duration` overlaps frame.

    Explicit columns (same idiom as `diar_to_segments_df`) so an empty overlap
    timeline still yields the 3 expected columns. Shared by the pyannote `run()`
    path, the sortformer path, and the v4.1 pyannote fallback.
    """
    records = [
        {
            "start": round(s.start, 3),
            "end": round(s.end, 3),
            "duration": round(s.duration, 3),
        }
        for s in diar.get_overlap()
    ]
    return pd.DataFrame(records, columns=["start", "end", "duration"])


class DiarizationStage(Stage):
    name = "diarization"

    def __init__(self, config: DiarizationConfig) -> None:
        super().__init__(enabled=config.enabled)
        self.config = config
        self._pipeline = None  # pyannote backend: populated by load()
        self._venv_py = None   # sortformer backend: populated by load()
        self._device = None    # torch.device recorded at load (v4.1 needs it)

    def load(self, device: torch.device) -> None:
        # Record the device: the sortformer L1 merge lever loads its own model
        # (ECAPA2) inside run(), which only receives ctx.
        self._device = device
        if self.config.backend == "sortformer":
            self._load_sortformer()
            return
        self._pipeline = build_pyannote_pipeline(self.config, device)
        _log(
            f"load: ready (num_speakers={self.config.num_speakers}, "
            f"min_duration_off={self.config.segmentation_min_duration_off}, "
            f"min_cluster_size={self.config.clustering_min_cluster_size})"
        )

    def _load_sortformer(self) -> None:
        """Validate the isolated NeMo venv for the sortformer backend. No model is
        loaded in-process — the worker loads NeMo in its own venv subprocess (one
        model on GPU at a time, the pipeline norm). Fails loud on a missing
        venv/worker (SCOPE §4 — never a silent fall-back to pyannote). Mirrors
        `_CohereXBackend.load`."""
        import os

        venv_py = os.environ.get("SORTFORMER_VENV_PY")
        if not venv_py:
            raise RuntimeError(
                "diarization.backend='sortformer' requires $SORTFORMER_VENV_PY — "
                "the path to the isolated NeMo venv's python (e.g. "
                "~/sortformer_venv/bin/python; see scripts/sortformer_worker.py "
                "for the venv recipe). Set it, or use backend='pyannote'. (No "
                "silent fall-back — SCOPE §4.)"
            )
        if not Path(venv_py).exists():
            raise FileNotFoundError(f"$SORTFORMER_VENV_PY not found: {venv_py}")
        if not _SORTFORMER_WORKER.exists():
            raise FileNotFoundError(f"Sortformer worker missing: {_SORTFORMER_WORKER}")
        self._venv_py = venv_py
        _log(
            f"load: sortformer backend ready (venv={venv_py}, "
            f"worker={_SORTFORMER_WORKER.name}, "
            f"model={self.config.sortformer_model_id}); NeMo loads per-run "
            f"in subprocess"
        )

    def load_signature(self) -> tuple:
        # `num_speakers` is a runtime knob passed at call time, not a model
        # identity. The front-end hyperparameters ARE applied at load (via
        # instantiate), so they belong in the signature: changing one in the
        # interactive API must trigger a reload. The sortformer path has no
        # in-process model; it returns a distinctly-shaped (backend, model id)
        # tuple, so a backend swap can never collide with the pyannote tuple and
        # always triggers a reload. `sortformer_threshold` is a post-process
        # (run-time) knob, so it's excluded — like pyannote's num_speakers.
        if self.config.backend == "sortformer":
            return (self.config.backend, self.config.sortformer_model_id)
        return (
            self.config.model_id,
            self.config.embedding,
            self.config.segmentation_min_duration_off,
            self.config.clustering_threshold,
            self.config.clustering_min_cluster_size,
        )

    def run(self, ctx: PipelineContext) -> None:
        if self.config.backend == "sortformer":
            self._run_sortformer(ctx)
            return
        if self._pipeline is None:
            raise RuntimeError("DiarizationStage.run called before load().")
        if ctx.audio is None:
            raise RuntimeError("PipelineContext.audio is None — no input loaded.")

        _log(
            f"run: diarizing {len(ctx.audio)/ctx.sample_rate:.1f}s "
            f"(num_speakers={self.config.num_speakers})..."
        )
        diar = run_pyannote(
            self._pipeline, ctx.audio, ctx.sample_rate, self.config.num_speakers
        )
        seg_df = diar_to_segments_df(diar)
        ovl_df = _overlaps_df_from_annotation(diar)

        total_dur = len(ctx.audio) / ctx.sample_rate
        ctx.diarization = DiarizationResult(
            segments_df=seg_df,
            overlaps_df=ovl_df,
            total_duration_s=total_dur,
        )
        _log(
            f"run: {len(seg_df)} segment(s), {len(ovl_df)} overlap region(s) "
            f"over {total_dur:.1f}s"
        )

    def _run_sortformer(self, ctx: PipelineContext) -> None:
        """Sortformer (EEND) branch of run(). Shell out to the isolated-venv
        worker for the raw per-frame activity, then follow the fixed decision
        order: binarize (L2) → top-2 selection → L1 merge (if head_policy=merge)
        → build the SAME DiarizationResult as the pyannote path. Lever activations
        are logged loudly and recorded in `ctx.diarization_diag` (→ metadata.json).
        """
        if self._venv_py is None:
            raise RuntimeError("DiarizationStage.run called before load().")
        if ctx.audio is None:
            raise RuntimeError("PipelineContext.audio is None — no input loaded.")
        dcfg = self.config

        _log(
            f"run: diarizing {len(ctx.audio)/ctx.sample_rate:.1f}s via sortformer "
            f"(num_speakers={dcfg.num_speakers}, "
            f"head_policy={dcfg.sortformer_head_policy})..."
        )
        probs, frame_rate_s = run_sortformer_worker(
            self._venv_py, dcfg.sortformer_model_id, ctx.audio, ctx.sample_rate,
        )
        # L2 — binarize (flat threshold) + top-2 selection.
        turns, diag = sortformer_turns_from_probs(
            probs, frame_rate_s, dcfg.sortformer_threshold,
        )
        top2 = diag["top2"]
        head_runs = diag["head_runs"]

        # Loud miscount alarm (SCOPE §3 warnings must be visible), unchanged and
        # fired REGARDLESS of the levers: Sortformer's fixed 4-speaker head can
        # leak Polish-OOD activity into heads 3/4 — the probe saw this on dev
        # fragment 82627719 (22% leak → a phantom 3rd speaker). Warn visibly when
        # the non-selected heads carry > 5% of speech, or when != 2 heads clear the
        # 5% activity bar. We still proceed with the 2 most-active heads.
        miscount = diag["leak"] > _SF_LEAK_WARN_FRAC or diag["n_spk"] != 2
        if miscount:
            _log(
                f"run: WARNING sortformer head-miscount signal — "
                f"n_spk={diag['n_spk']} (expected 2), head3/4 leak="
                f"{diag['leak']:.1%} of speech (> {_SF_LEAK_WARN_FRAC:.0%} = "
                f"Polish-OOD alarm; top2 heads={top2}, per-head active s="
                f"{[round(d, 1) for d in diag['head_dur_s']]}). Proceeding with "
                f"the 2 most-active heads."
            )

        # Census diag written to metadata.json (scalar fields only — head_runs
        # stays internal). Grows as the levers run below.
        sf_diag = {
            "backend": "sortformer",
            "n_spk": diag["n_spk"],
            "leak": round(float(diag["leak"]), 4),
            "top2": top2,
            "speech_s": round(float(diag["speech_s"]), 3),
            "head_dur_s": [round(float(d), 3) for d in diag["head_dur_s"]],
            "head_policy": dcfg.sortformer_head_policy,
            "miscount_warning": bool(miscount),
        }

        labels = {top2[0]: "SPEAKER_00"}
        if len(top2) >= 2:
            labels[top2[1]] = "SPEAKER_01"

        # --- L1 merge-not-discard (surplus-head reassignment) ---
        merged_turns: list = []
        merge_stats = None
        if dcfg.sortformer_head_policy == "merge" and len(top2) >= 2:
            # Load ECAPA2 here — AFTER the worker subprocess has exited (GPU free)
            # and free it before this stage returns (phase-major: never two big
            # models co-resident). `with_ecapa2` builds the same embedder the
            # relabel stage uses and does the drop + GPU-memory release on exit,
            # even if the merge raises.
            _log("run: L1 merge — loading ECAPA2 embedder for surplus-head "
                 "reassignment...")
            with with_ecapa2(self._device) as embedder:
                merged_turns, merge_stats = _merge_surplus_heads(
                    head_runs, top2, labels, ctx.audio, ctx.sample_rate, embedder,
                    merge_margin=dcfg.sortformer_merge_margin,
                    speech_tot=diag["speech_s"],
                )
            if merged_turns:
                turns = turns + merged_turns
            _log(
                f"run: L1 merge — merged {merge_stats['n_merged']} run(s) "
                f"({merge_stats['merged_s']:.1f}s) into the top-2 speakers, "
                f"{merge_stats['n_unresolved']} unresolved below margin "
                f"({merge_stats['unresolved_s']:.1f}s), {merge_stats['n_short']} "
                f"too short ({merge_stats['short_s']:.1f}s); unresolved leak "
                f"{merge_stats['unresolved_leak']:.1%} of speech "
                f"(margin >= {dcfg.sortformer_merge_margin})."
            )
            sf_diag["merge"] = merge_stats

        ctx.diarization_diag = sf_diag

        # Build the sortformer annotation (+ any L1-merged turns).
        diar = build_sortformer_annotation(turns)
        seg_df = diar_to_segments_df(diar)
        ovl_df = _overlaps_df_from_annotation(diar)

        total_dur = len(ctx.audio) / ctx.sample_rate
        ctx.diarization = DiarizationResult(
            segments_df=seg_df,
            overlaps_df=ovl_df,
            total_duration_s=total_dur,
        )
        _log(
            f"run: {len(seg_df)} segment(s), {len(ovl_df)} overlap region(s) "
            f"over {total_dur:.1f}s (sortformer, top2 heads={top2}, "
            f"policy={dcfg.sortformer_head_policy})"
        )

    def unload(self) -> None:
        self._pipeline = None
        self._venv_py = None
        self._device = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def spill(self, ctx: PipelineContext, artifact_dir: Path) -> None:
        # Diagnostic spill (only when spill_intermediate=True). This is the
        # {segments, overlaps} schema shared with scripts/diarize_clarin_2speakers
        # and read by scripts/clarin_fragment_finder — DELIBERATELY distinct from
        # the eval-facing {turns} schema io.write_pipeline_outputs writes to
        # pipeline/diarization.json (see io.py module docstring). Don't unify:
        # the fragment finder needs the `overlaps` array that {turns} omits.
        if ctx.diarization is None:
            return
        payload = {
            "total_duration_s": ctx.diarization.total_duration_s,
            "segments": ctx.diarization.segments_df.to_dict(orient="records"),
            "overlaps": ctx.diarization.overlaps_df.to_dict(orient="records"),
        }
        with open(artifact_dir / "diarization.json", "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
