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


def _log(msg: str) -> None:
    """Progress message — stdout + the durable debug log. Matters here because
    pyannote's load and forward are multi-minute and the WSL stdout bridge can
    drop; every other model-bearing stage logs the same way."""
    dlog("diarization", msg)


class DiarizationStage(Stage):
    name = "diarization"

    def __init__(self, config: DiarizationConfig) -> None:
        super().__init__(enabled=config.enabled)
        self.config = config
        self._pipeline = None  # populated by load()

    def load(self, device: torch.device) -> None:
        from pyannote.audio import Pipeline

        token = self.config.hf_token
        if not token:
            raise RuntimeError(
                "DiarizationConfig.hf_token is empty — set HF_TOKEN in the "
                "environment or put a literal value in the YAML config."
            )
        _log(f"load: instantiating {self.config.model_id} on {device}...")
        self._pipeline = Pipeline.from_pretrained(
            self.config.model_id, token=token
        ).to(device)
        _log(f"load: ready (num_speakers={self.config.num_speakers})")

    def load_signature(self) -> tuple:
        # `num_speakers` is a runtime knob passed at call time, not a model
        # identity — so only model_id triggers a reload.
        return (self.config.model_id,)

    def run(self, ctx: PipelineContext) -> None:
        if self._pipeline is None:
            raise RuntimeError("DiarizationStage.run called before load().")
        if ctx.audio is None:
            raise RuntimeError("PipelineContext.audio is None — no input loaded.")

        _log(
            f"run: diarizing {len(ctx.audio)/ctx.sample_rate:.1f}s "
            f"(num_speakers={self.config.num_speakers})..."
        )
        waveform = torch.from_numpy(ctx.audio).unsqueeze(0)
        result = self._pipeline(
            {"waveform": waveform, "sample_rate": ctx.sample_rate},
            num_speakers=self.config.num_speakers,
        )
        # pyannote 4.x returns a DiarizeOutput wrapper (.speaker_diarization);
        # 3.x returns the Annotation directly. Support both.
        diar = (
            result.speaker_diarization
            if hasattr(result, "speaker_diarization")
            else result
        )

        seg_records = [
            {
                "start": round(t.start, 3),
                "end": round(t.end, 3),
                "duration": round(t.duration, 3),
                "speaker": spk,
            }
            for t, _, spk in diar.itertracks(yield_label=True)
        ]
        # Explicit columns so downstream `df["speaker"]` access works even
        # when pyannote returns no segments at all (e.g. silent input) —
        # `pd.DataFrame([])` would otherwise have no columns.
        seg_df = pd.DataFrame(
            seg_records, columns=["start", "end", "duration", "speaker"]
        )

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
