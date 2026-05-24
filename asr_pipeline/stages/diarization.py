"""Stage 1 — pyannote speaker diarization.

Ported from `asr/asr_pipeline.ipynb` cell 10. Runs pyannote
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
from asr_pipeline.stages.base import Stage


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
        self._pipeline = Pipeline.from_pretrained(
            self.config.model_id, token=token
        ).to(device)

    def load_signature(self) -> tuple:
        # `num_speakers` is a runtime knob passed at call time, not a model
        # identity — so only model_id triggers a reload.
        return (self.config.model_id,)

    def run(self, ctx: PipelineContext) -> None:
        if self._pipeline is None:
            raise RuntimeError("DiarizationStage.run called before load().")
        if ctx.audio is None:
            raise RuntimeError("PipelineContext.audio is None — no input loaded.")

        waveform = torch.from_numpy(ctx.audio).unsqueeze(0)
        result = self._pipeline(
            {"waveform": waveform, "sample_rate": ctx.sample_rate},
            num_speakers=self.config.num_speakers,
        )
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
        seg_df = pd.DataFrame(seg_records)

        ovl_records = [
            {
                "start": round(s.start, 3),
                "end": round(s.end, 3),
                "duration": round(s.duration, 3),
            }
            for s in diar.get_overlap()
        ]
        ovl_df = (
            pd.DataFrame(ovl_records)
            if ovl_records
            else pd.DataFrame(columns=["start", "end", "duration"])
        )

        total_dur = len(ctx.audio) / ctx.sample_rate
        ctx.diarization = DiarizationResult(
            segments_df=seg_df,
            overlaps_df=ovl_df,
            total_duration_s=total_dur,
        )

    def unload(self) -> None:
        self._pipeline = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def spill(self, ctx: PipelineContext, artifact_dir: Path) -> None:
        if ctx.diarization is None:
            return
        payload = {
            "total_duration_s": ctx.diarization.total_duration_s,
            "segments": ctx.diarization.segments_df.to_dict(orient="records"),
            "overlaps": ctx.diarization.overlaps_df.to_dict(orient="records"),
        }
        with open(artifact_dir / "diarization.json", "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
