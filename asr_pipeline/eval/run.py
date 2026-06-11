"""Orchestrator: ``evaluate_recording(rec)`` → ``ScoreCard``.

One call computes both eval layers (L2 + L3) for one recording. The caller (the
evaluate notebook or a batch script) iterates a list of recordings from
``walk_eval_tree`` and accumulates score cards.

When you're running over many recordings, pass an already-loaded SQUIM
model in (``squim_model`` / ``squim_device``) to avoid reloading it per
recording — see :func:`evaluate_many`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from asr_pipeline.eval.layer2 import (
    compute_layer2,
    load_squim_model,
    unload_squim_model,
)
from asr_pipeline.eval.layer3 import HypFilter, compute_layer3
from asr_pipeline.eval.recordings import Recording


@dataclass
class ScoreCard:
    """Aggregated eval result for one recording."""

    recording: Recording
    layer2: Optional[dict]       # SI-SDR/PESQ/STOI/SQUIM (or None when no pipeline)
    layer3: Optional[dict]       # cpWER ablation (or None when no GT transcripts)

    @property
    def id(self) -> str:
        return self.recording.id

    @property
    def dataset(self) -> str:
        return self.recording.dataset


def evaluate_recording(
    rec: Recording,
    sr: int = 16_000,
    squim_model=None,
    squim_device=None,
    tcp_collar_s: float = 5.0,
    hyp_filter: Optional[HypFilter] = None,
) -> ScoreCard:
    """Compute both eval layers (L2 audio quality, L3 ASR) for one recording.

    ``squim_model`` lets the caller share a loaded SQUIM across many
    recordings. None means "load + unload inside the L2 call" (fine for
    one-shot, wasteful for sweeps).

    ``hyp_filter`` (default None) is threaded to L3 to rewrite hypothesis
    streams before scoring — EdAcc passes ``eval.edacc.excise_stella_passage``
    here; every other dataset leaves it None. It is dataset-specific and
    opt-in by design: the caller decides per dataset (see the notebook).
    """
    return ScoreCard(
        recording=rec,
        layer2=compute_layer2(rec, sr=sr, squim_model=squim_model, squim_device=squim_device),
        layer3=compute_layer3(rec, tcp_collar_s=tcp_collar_s, hyp_filter=hyp_filter),
    )


def evaluate_many(
    recordings,
    sr: int = 16_000,
    tcp_collar_s: float = 5.0,
    hyp_filter: Optional[HypFilter] = None,
) -> list[ScoreCard]:
    """Evaluate a sequence of recordings, loading SQUIM once.

    ``hyp_filter`` is forwarded to every recording — use it for a single-dataset
    batch (e.g. all of EdAcc). For a mixed batch, score each dataset separately
    so the filter only ever sees the dataset it's built for.
    """
    squim_model, squim_device = load_squim_model()
    try:
        cards = [
            evaluate_recording(
                rec,
                sr=sr,
                squim_model=squim_model,
                squim_device=squim_device,
                tcp_collar_s=tcp_collar_s,
                hyp_filter=hyp_filter,
            )
            for rec in recordings
        ]
    finally:
        unload_squim_model(squim_model)
    return cards
