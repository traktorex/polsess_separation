"""Custom speaker-embedding wrappers for the diarization stage.

pyannote's `SpeakerDiarization` clusters speech segments by speaker-embedding
similarity. Its built-in embedding factory
(`pyannote/audio/pipelines/speaker_verification.py:PretrainedSpeakerEmbedding`)
only dispatches model-id strings containing ``pyannote`` / ``speechbrain`` /
``nvidia`` / ``wespeaker`` — arbitrary embedders are NOT reachable through the
``embedding=`` string. To run a non-pyannote embedder we therefore build the
pipeline with a pyannote placeholder, then **replace** ``pipeline._embedding``
with one of the wrappers below (see ``DiarizationStage.load``).

Each wrapper conforms to the same duck-typed interface pyannote relies on
(``pyannote.audio.core.inference.BaseInference``), exactly as the stock
``PyannoteAudioPretrainedSpeakerEmbedding`` / ``SpeechBrainPretrainedSpeakerEmbedding``
classes do:

  - ``__call__(waveforms, masks=None) -> (batch, dimension) np.ndarray``
    where ``waveforms`` is ``(batch, channel=1, num_samples)`` at
    ``self.sample_rate`` and ``masks`` is an optional ``(batch, num_frames)``
    frame-level weight tensor (supplied when ``embedding_exclude_overlap=True``).
  - properties ``.dimension`` (int), ``.sample_rate`` (int), ``.metric`` (str),
    ``.min_num_samples`` (int).
  - ``.to(device)`` returning ``self``.

The supported embedder (verified empirically on CPU, 2026-06-18):

  - ``"ecapa2"``  — Jenthe/ECAPA2, a single TorchScript blob ``ecapa2.pt``
    pulled from the HF hub. Input ``(batch, num_samples)`` @ 16 kHz, output
    ``(batch, 192)``. CC-BY-NC (research use). min_num_samples ≈ 400.

SCOPE §4 (no silent substitution): if a configured custom embedder fails to
load, the factory / wrapper raises loudly — there is no quiet fall-back to the
pyannote default.
"""

from __future__ import annotations

import gc
from contextlib import contextmanager
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from pyannote.audio.core.inference import BaseInference


# Names accepted by the `diarization.embedding` config knob (besides None and a
# pyannote-format model id). Single source of truth for the selector.
CUSTOM_EMBEDDING_NAMES = ("ecapa2",)


def _apply_masks_to_signals(
    waveforms: torch.Tensor, masks: torch.Tensor
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Turn a frame-level weight mask into per-item masked-in waveforms.

    Mirrors the stock pyannote `SpeechBrainPretrainedSpeakerEmbedding.__call__`
    masking: the ``(batch, num_frames)`` mask is nearest-interpolated up to
    ``num_samples``, thresholded at 0.5, and the kept samples are gathered into
    a per-item contiguous signal (overlap frames having been zeroed upstream by
    ``embedding_exclude_overlap``). Returns the list of per-item signals (kept
    samples only) plus a ``(batch,)`` tensor of kept-sample counts.
    """
    batch_size, _, num_samples = waveforms.shape
    wav = waveforms.squeeze(1)  # (batch, num_samples)
    imasks = F.interpolate(
        masks.unsqueeze(1), size=num_samples, mode="nearest"
    ).squeeze(1) > 0.5
    signals = [w[m].contiguous() for w, m in zip(wav, imasks)]
    lengths = imasks.sum(dim=1)
    return signals, lengths


class BaseCustomSpeakerEmbedding(BaseInference):
    """Shared scaffolding for the custom embedders.

    Subclasses implement ``_embed_one(signal_1d) -> (dimension,) np.ndarray``
    for a single mono 16 kHz waveform on ``self.device``; this base supplies the
    pyannote interface (``__call__`` with the mask/short-signal/NaN bookkeeping,
    the four properties, ``to``). Embedding is done one item at a time — a single
    uniform per-item path keeps the mask/short-signal/NaN bookkeeping simple.
    """

    metric = "cosine"
    sample_rate = 16_000

    def __init__(self, device: Optional[torch.device] = None) -> None:
        super().__init__()
        self.device = device or torch.device("cpu")
        self._dimension: Optional[int] = None
        self._min_num_samples: Optional[int] = None

    # --- subclass hooks ---------------------------------------------------
    def _embed_one(self, signal: torch.Tensor) -> np.ndarray:
        """Embed a single 1-D waveform (num_samples,) → (dimension,) ndarray."""
        raise NotImplementedError

    # --- pyannote interface ----------------------------------------------
    def to(self, device: torch.device) -> "BaseCustomSpeakerEmbedding":
        if not isinstance(device, torch.device):
            raise TypeError(
                f"`device` must be a torch.device, got {type(device).__name__}"
            )
        self._move_to(device)
        self.device = device
        return self

    def _move_to(self, device: torch.device) -> None:
        """Move the underlying model to `device`. Override per backend."""
        raise NotImplementedError

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            # One probe forward on a 1 s dummy signal fixes the output width.
            emb = self._embed_one(torch.zeros(self.sample_rate, device=self.device))
            self._dimension = int(np.asarray(emb).reshape(-1).shape[0])
        return self._dimension

    @property
    def min_num_samples(self) -> int:
        """Smallest sample count that yields a finite embedding.

        Found by the same bisection the stock pyannote classes use (search in
        [2, 0.5 s]), but on a tone rather than noise — these models' pooling can
        emit non-finite values on too-short *random* input even when a real
        short signal would be fine, and we want the floor for genuine speech.
        """
        if self._min_num_samples is not None:
            return self._min_num_samples

        def _ok(n: int) -> bool:
            t = torch.arange(n, device=self.device) / self.sample_rate
            sig = 0.1 * torch.sin(2 * np.pi * 220.0 * t).float()
            try:
                emb = self._embed_one(sig)
            except Exception:
                return False
            return bool(np.all(np.isfinite(np.asarray(emb))))

        lower, upper = 2, round(0.5 * self.sample_rate)
        while lower + 1 < upper:
            middle = (lower + upper) // 2
            if _ok(middle):
                upper = middle
            else:
                lower = middle
        self._min_num_samples = upper
        return upper

    def __call__(
        self, waveforms: torch.Tensor, masks: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """Embed a batch of (masked) waveforms.

        Parameters
        ----------
        waveforms : (batch, 1, num_samples) torch.Tensor at self.sample_rate.
        masks : (batch, num_frames) torch.Tensor, optional. Frame-level weights;
            interpolated to num_samples and thresholded at 0.5 (the pyannote
            convention). When present, only mask-in samples feed the embedder —
            this is how ``embedding_exclude_overlap=True`` keeps overlap frames
            out of the speaker centroid.

        Returns
        -------
        (batch, dimension) np.ndarray. Rows whose usable signal is shorter than
        ``min_num_samples`` are filled with NaN, exactly as the stock pyannote
        embedders do (the clusterer tolerates NaN rows).
        """
        batch_size, num_channels, num_samples = waveforms.shape
        assert num_channels == 1, "only mono (num_channels == 1) is supported"

        if masks is None:
            signals = [w.squeeze(0) for w in waveforms]  # each (num_samples,)
            lengths = torch.full((batch_size,), num_samples)
        else:
            assert masks.shape[0] == batch_size
            signals, lengths = _apply_masks_to_signals(waveforms, masks)

        min_n = self.min_num_samples
        dim = self.dimension
        out = np.full((batch_size, dim), np.nan, dtype=np.float32)
        for i, (sig, n) in enumerate(zip(signals, lengths)):
            if int(n) < min_n:
                continue  # leave NaN — too short for a stable embedding
            emb = np.asarray(self._embed_one(sig.to(self.device))).reshape(-1)
            out[i] = emb
        return out


class ECAPA2Embedding(BaseCustomSpeakerEmbedding):
    """Jenthe/ECAPA2 TorchScript embedder.

    Loads the single ``ecapa2.pt`` blob from the HF hub and runs it via
    ``torch.jit``. The scripted model accepts ``(batch, num_samples)`` (and
    ``(batch, 1, num_samples)``); we feed one ``(1, num_samples)`` per item.
    """

    REPO_ID = "Jenthe/ECAPA2"
    FILENAME = "ecapa2.pt"

    def __init__(self, device: Optional[torch.device] = None) -> None:
        super().__init__(device=device)
        from huggingface_hub import hf_hub_download  # local: heavy import

        # Fail loud (SCOPE §4): a download / load error must surface, never a
        # silent downgrade to the pyannote default.
        path = hf_hub_download(repo_id=self.REPO_ID, filename=self.FILENAME)
        self.model_ = torch.jit.load(path, map_location=self.device)
        self.model_.eval()

    def _move_to(self, device: torch.device) -> None:
        self.model_ = self.model_.to(device)

    @torch.inference_mode()
    def _embed_one(self, signal: torch.Tensor) -> np.ndarray:
        # ECAPA2 wants (batch, num_samples); give it a 1-item batch.
        #
        # `optimized_execution(False)` disables the TorchScript optimizing
        # (profiling) executor for this JIT forward, which keeps the CUDA tensor-
        # expression fuser (NNC/NVRTC) off the graph. ECAPA2's front-end computes
        # a complex STFT, and once the profiling executor warms up on CUDA the
        # fuser tries to codegen a `c10::complex<float>` elementwise kernel
        # (`fabs` over the complex spectrogram) — emitting invalid CUDA that NVRTC
        # rejects ("name followed by '::' must be a class or namespace name").
        # CPU has no such fuser, so this only bites on GPU (the integration path).
        # Scoped to this call so the rest of the process keeps its fusion.
        with torch.jit.optimized_execution(False):
            out = self.model_(signal.unsqueeze(0))
        return out.squeeze(0).float().cpu().numpy()


def build_custom_embedding(
    name: str, device: torch.device
) -> Optional[BaseCustomSpeakerEmbedding]:
    """Return a custom embedder wrapper for a known custom name, else None.

    ``name`` is the value of ``DiarizationConfig.embedding``. Returns the
    wrapper (already on ``device``) for ``"ecapa2"``, and ``None`` for anything
    else — letting the caller treat a non-custom string as a pyannote-format
    model id. Construction failures propagate (SCOPE §4).
    """
    if name == "ecapa2":
        return ECAPA2Embedding(device=device)
    return None


# ---------------------------------------------------------------------------
# Shared embedding plumbing (used by diarization + relabel + the L1 merge)
# ---------------------------------------------------------------------------


def embed_intervals(intervals, audio, sr, embedder, *, cap_s=None):
    """Concatenate the audio under `intervals` (time order, optionally capped to
    `cap_s` seconds) and embed it with a custom ECAPA2-style wrapper.

    Single source of truth for the "slice intervals → concat → embed → drop if
    unreliable" pattern shared by ``diarization._merge_surplus_heads`` and
    ``relabel._embed_spans``. Returns a ``(dim,)`` float32 embedding, or ``None``
    when there is nothing to embed, the usable audio is below the embedder's
    ``min_num_samples``, or the embedding is non-finite — all three mean "no
    reliable embedding", and callers treat them the same (drop the row / keep the
    pass-1 label).
    """
    slices = []
    total = 0.0
    for s, e in intervals:
        if cap_s is not None and total >= cap_s:
            break
        lo = max(0, int(s * sr))
        hi = int(e * sr)
        if cap_s is not None:
            hi = min(hi, lo + int(max(0.0, cap_s - total) * sr))
        if hi > lo:
            slices.append(np.asarray(audio[lo:hi], dtype=np.float32))
            total += (hi - lo) / sr
    if not slices:
        return None
    concat = np.concatenate(slices)
    if len(concat) < embedder.min_num_samples:
        return None
    wav = torch.from_numpy(concat).reshape(1, 1, -1)
    emb = np.asarray(embedder(wav)[0], dtype=np.float32)
    if not np.all(np.isfinite(emb)):
        return None
    return emb


def release_gpu_memory() -> None:
    """Drop-a-model teardown: run GC and free CUDA's cached blocks.

    The phase-major discipline (one big model on the GPU at a time) requires the
    ECAPA2 embedder's memory be released the moment a stage is done with it.
    Shared by every ECAPA2 lifecycle site — ``RelabelStage.unload``,
    ``AssemblyStage.unload``, and the scoped ``with_ecapa2`` teardown below — so
    the same three lines aren't copied per site.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@contextmanager
def with_ecapa2(device):
    """Build an ECAPA2 embedder on `device`, yield it, and free it on exit.

    For SCOPED ECAPA2 usage — the diarization L1 surplus-head merge, which loads
    the embedder AFTER the Sortformer worker subprocess has exited (GPU free) and
    must free it before the stage returns so it never co-resides with the next
    stage's model (phase-major). Teardown (drop + ``release_gpu_memory``) runs
    even if the body raises, exactly like the previous try/finally. Stages whose
    embedder lives across the orchestrator's load→run→unload lifecycle
    (Relabel/Assembly) keep their split load/unload and call ``release_gpu_memory``
    in ``unload`` instead.
    """
    embedder = build_custom_embedding("ecapa2", device)
    try:
        yield embedder
    finally:
        del embedder
        release_gpu_memory()
