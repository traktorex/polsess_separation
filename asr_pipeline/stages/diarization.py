"""Stage 1 — pyannote speaker diarization.

Ported from `asr/archive/asr_pipeline.ipynb` cell 10. Runs pyannote
`speaker-diarization-3.1` on mono 16 kHz audio, constrained to two
speakers, and emits per-speaker segments + an overlap timeline.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import pandas as pd
import torch

from asr_pipeline.config import DiarizationConfig
from asr_pipeline.context import DiarizationResult, PipelineContext
from asr_pipeline.debug_log import dlog
from asr_pipeline.stages.base import Stage
from asr_pipeline.stages.custom_embeddings import (
    CUSTOM_EMBEDDING_NAMES,
    build_custom_embedding,
)


def _log(msg: str) -> None:
    """Progress message — stdout + the durable debug log. Matters here because
    pyannote's load and forward are multi-minute and the WSL stdout bridge can
    drop; every other model-bearing stage logs the same way."""
    dlog("diarization", msg)


def build_pyannote_pipeline(config: DiarizationConfig, device: torch.device):
    """Build a loaded pyannote diarization pipeline from a DiarizationConfig.

    Shared by DiarizationStage (pass 1, raw) and FusionDiarizationStage (pass 2,
    enhanced) so the second pass re-uses the *identical* construction — only the
    waveform fed to it differs. Honours `config.embedding` (None = stock 3.1 via
    `from_pretrained`; a pyannote id or a custom-name wrapper otherwise) and the
    front-end instantiate hyperparameters. Fails loud (SCOPE §4) — no silent
    fall-back to pyannote when a custom embedder cannot load.
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


class DiarizationStage(Stage):
    name = "diarization"

    def __init__(self, config: DiarizationConfig) -> None:
        super().__init__(enabled=config.enabled)
        self.config = config
        self._pipeline = None  # populated by load()

    def load(self, device: torch.device) -> None:
        self._pipeline = build_pyannote_pipeline(self.config, device)
        _log(
            f"load: ready (num_speakers={self.config.num_speakers}, "
            f"min_duration_off={self.config.segmentation_min_duration_off}, "
            f"min_cluster_size={self.config.clustering_min_cluster_size})"
        )

    def load_signature(self) -> tuple:
        # `num_speakers` is a runtime knob passed at call time, not a model
        # identity. The front-end hyperparameters ARE applied at load (via
        # instantiate), so they belong in the signature: changing one in the
        # interactive API must trigger a reload.
        return (
            self.config.model_id,
            self.config.embedding,
            self.config.segmentation_min_duration_off,
            self.config.clustering_threshold,
            self.config.clustering_min_cluster_size,
        )

    def run(self, ctx: PipelineContext) -> None:
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

        ovl_records = [
            {
                "start": round(s.start, 3),
                "end": round(s.end, 3),
                "duration": round(s.duration, 3),
            }
            for s in diar.get_overlap()
        ]
        # Explicit columns (same idiom as seg_df) so an empty overlap timeline
        # still yields the 3 expected columns rather than a column-less frame.
        ovl_df = pd.DataFrame(ovl_records, columns=["start", "end", "duration"])

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

    def unload(self) -> None:
        self._pipeline = None
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
