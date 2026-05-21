"""Stage 3a — solo enhancement with MP-SENet.

Uses the vendored MP-SENet code at `asr_pipeline.vendor.mpsenet`. STFT
preprocessing (compressed-magnitude + unwrapped-phase) is ported inline
from POC cell 14; it matches MP-SENet's `mag_pha_stft` / `mag_pha_istft`
in upstream `utils.py`.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from asr_pipeline.config import EnhancementConfig
from asr_pipeline.context import PipelineContext
from asr_pipeline.stages.base import Stage
from asr_pipeline.vendor.mpsenet import MPNet


class _AttrDict(dict):
    """Dict that supports attribute access (needed by MPNet.__init__)."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e


# ---------------------------------------------------------------------------
# STFT helpers — exact port of upstream MP-SENet `mag_pha_stft` / `mag_pha_istft`
# (compressed-magnitude + raw phase representation).
# ---------------------------------------------------------------------------


def _mpsenet_stft(audio_t: torch.Tensor, h: dict) -> tuple[torch.Tensor, torch.Tensor]:
    window = torch.hann_window(h["win_size"]).to(audio_t.device)
    spec = torch.stft(
        audio_t,
        h["n_fft"],
        hop_length=h["hop_size"],
        win_length=h["win_size"],
        window=window,
        center=True,
        pad_mode="reflect",
        normalized=False,
        return_complex=True,
    )
    mag = spec.abs().pow(h["compress_factor"])
    pha = spec.angle()
    return mag, pha


def _mpsenet_istft(mag: torch.Tensor, pha: torch.Tensor, h: dict) -> torch.Tensor:
    mag = mag.pow(1.0 / h["compress_factor"])
    spec = torch.complex(mag * torch.cos(pha), mag * torch.sin(pha))
    window = torch.hann_window(h["win_size"]).to(spec.device)
    return torch.istft(
        spec,
        h["n_fft"],
        hop_length=h["hop_size"],
        win_length=h["win_size"],
        window=window,
        center=True,
        normalized=False,
        onesided=True,
    )


class EnhancementStage(Stage):
    name = "enhancement"

    def __init__(self, config: EnhancementConfig) -> None:
        super().__init__(enabled=config.enabled)
        self.config = config
        self._model: MPNet | None = None
        self._h: _AttrDict | None = None
        self._device: torch.device | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def load(self, device: torch.device) -> None:
        config_path = Path(self.config.config_path)
        ckpt_path = Path(self.config.checkpoint_path)
        if not config_path.exists():
            raise FileNotFoundError(f"MP-SENet config.json not found: {config_path}")
        if not ckpt_path.exists():
            raise FileNotFoundError(f"MP-SENet checkpoint not found: {ckpt_path}")

        with open(config_path) as f:
            h = _AttrDict(json.load(f))

        model = MPNet(h).to(device)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["generator"])
        model.eval()

        self._model = model
        self._h = h
        self._device = device

    def unload(self) -> None:
        self._model = None
        self._h = None
        self._device = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _enhance_one_shot(self, audio_np: np.ndarray) -> np.ndarray:
        """Enhance one mono region in a single MP-SENet forward pass.

        Caller is responsible for keeping the region under
        `max_segment_length_s` — longer regions blow up MP-SENet's O(T^2)
        time-axis attention.
        """
        assert self._model is not None and self._h is not None
        if len(audio_np) < self._h["win_size"]:
            return audio_np.copy()
        audio = torch.from_numpy(audio_np).to(self._device)
        norm = torch.sqrt(audio.numel() / (audio.pow(2).sum() + 1e-12))
        audio_n = (audio * norm).unsqueeze(0)
        mag, pha = _mpsenet_stft(audio_n, self._h)
        out = self._model(mag, pha)
        if isinstance(out, (tuple, list)):
            amp_out, pha_out = out[0], out[1]
        else:
            amp_out, pha_out = out, pha
        audio_out = _mpsenet_istft(amp_out, pha_out, self._h).squeeze(0)
        audio_out = (audio_out / (norm + 1e-12)).cpu().numpy().astype(np.float32)
        if len(audio_out) < len(audio_np):
            audio_out = np.pad(audio_out, (0, len(audio_np) - len(audio_out)))
        else:
            audio_out = audio_out[: len(audio_np)]
        return audio_out

    @torch.no_grad()
    def _enhance_chunked(self, audio_np: np.ndarray, sample_rate: int) -> np.ndarray:
        """Hann overlap-add MP-SENet over a long region.

        Chunk size = `max_segment_length_s`; hop = chunk / 2 (canonical
        50% COLA — the Hann window sum is exactly 1.0 in the interior, so
        we get a click-free reconstruction; head/tail are divided by their
        actual accumulated weights).
        """
        chunk_n = int(self.config.max_segment_length_s * sample_rate)
        hop_n = chunk_n // 2
        n = len(audio_np)
        if n <= chunk_n:
            return self._enhance_one_shot(audio_np)

        out = np.zeros(n, dtype=np.float32)
        weights = np.zeros(n, dtype=np.float32)
        window = np.hanning(chunk_n).astype(np.float32)

        start = 0
        while start < n:
            end = min(start + chunk_n, n)
            seg = audio_np[start:end]
            if len(seg) < self._h["win_size"]:
                # Too short for MP-SENet's STFT; pass through.
                chunk_out = seg.astype(np.float32)
            else:
                seg_padded = (
                    np.pad(seg, (0, chunk_n - len(seg)))
                    if len(seg) < chunk_n
                    else seg
                )
                chunk_out = self._enhance_one_shot(seg_padded)[: len(seg)]
            w = window[: len(seg)]
            out[start:end] += chunk_out * w
            weights[start:end] += w
            if end == n:
                break
            start += hop_n

        weights = np.maximum(weights, 1e-6)
        return (out / weights).astype(np.float32)

    def _enhance_region(self, audio_np: np.ndarray, sample_rate: int) -> np.ndarray:
        """Dispatch one solo region to one-shot or chunked enhancement."""
        if len(audio_np) < 256:
            # Too short for MP-SENet to do anything useful; pass through.
            return audio_np
        threshold_n = int(self.config.max_segment_length_s * sample_rate)
        if len(audio_np) <= threshold_n:
            return self._enhance_one_shot(audio_np)
        return self._enhance_chunked(audio_np, sample_rate)

    def run(self, ctx: PipelineContext) -> None:
        if self._model is None:
            raise RuntimeError("EnhancementStage.run called before load().")
        if ctx.partition_df is None:
            raise RuntimeError(
                "EnhancementStage.run requires ctx.partition_df "
                "(RoutingStage must run first)."
            )
        if ctx.audio is None:
            raise RuntimeError("PipelineContext.audio is None.")

        solo_rows = ctx.partition_df[
            ctx.partition_df["kind"].str.startswith("solo")
        ]
        results: dict[str, dict] = {}
        for idx, row in solo_rows.iterrows():
            s_idx = int(row["start"] * ctx.sample_rate)
            e_idx = int(row["end"] * ctx.sample_rate)
            raw = ctx.audio[s_idx:e_idx].astype(np.float32)
            enhanced = self._enhance_region(raw, ctx.sample_rate)
            key = f"{row['kind']}_#{idx}"
            results[key] = {
                "key": key,
                "start": float(row["start"]),
                "end": float(row["end"]),
                "speaker": row["speaker"],
                "raw": raw,
                "enhanced": enhanced,
            }
        ctx.solo_enhanced = results

    # ------------------------------------------------------------------
    # Spill
    # ------------------------------------------------------------------
    def spill(self, ctx: PipelineContext, artifact_dir: Path) -> None:
        if not ctx.solo_enhanced:
            return
        for key, seg in ctx.solo_enhanced.items():
            # `solo-A_#3` -> `solo_solo-A_3.wav` is noisy; flatten the key.
            safe_key = key.replace("#", "").replace("/", "-")
            fname = f"{safe_key}.wav"
            sf.write(
                artifact_dir / fname,
                seg["enhanced"].astype(np.float32),
                ctx.sample_rate,
            )
