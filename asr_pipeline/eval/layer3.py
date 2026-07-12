"""Layer 3 — ASR error rates + WER ablation table.

For each pipeline mode that's been run (full / no-sep / no-enh / minimal),
compute cpWER + tcpWER against the per-speaker GT transcripts. The
differences between modes give the ablation: how much does separation buy,
how much does enhancement buy?

Plus ORC-WER between the mixture-baseline transcript (single-stream
Whisper on the raw mixture, written by Stage 5 when
``transcription.transcribe_mixture: true``) and the GT — the *no-pipeline*
baseline. The full chain reads::

    mixture (no pipeline)  ─→  ORC-WER vs GT
    pipeline_minimal       ─→  cpWER vs GT   (diarize+slice only, both off)
    pipeline_noenh         ─→  cpWER vs GT
    pipeline_nosep         ─→  cpWER vs GT
    pipeline (full)        ─→  cpWER vs GT

Lower is better; the spread tells you which stages earn their compute.

Caveat: the WER comparison requires both sides to use the same
transcription backend with the same prompt / args, otherwise the diff
between numbers reflects model choice, not pipeline contribution. The
pipeline writer enforces this — the mixture transcript is produced by
the same Stage 5 backend as the per-speaker transcripts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from asr_pipeline.debug_log import dlog
from asr_pipeline.eval.metrics import (
    mimo_wer_meeteval,
    orc_wer_meeteval,
    per_fragment_metrics,
)
from asr_pipeline.eval.recordings import Recording, load_reference_utterances
from asr_pipeline.eval.transcript_parser import (
    Utterance,
    is_untimed,
    parse_gt_txt,
)


def _resolve_lang(rec: Recording) -> str:
    """Transcription language for this recording, for language-aware WER
    normalization (E13).

    Read from the first available pipeline mode's ``metadata.json`` config
    snapshot (``config.transcription.language``, written by
    ``io.write_pipeline_outputs``). Falls back to ``"pl"`` when no metadata or
    config snapshot is present — the project default and the pre-E13 behaviour.
    The number speller for English hypotheses would otherwise
    spell digits in Polish, fabricating substitutions.
    """
    for d in (rec.pipeline_dir, rec.pipeline_nosep_dir, rec.pipeline_noenh_dir,
              rec.pipeline_minimal_dir):
        if d is None:
            continue
        meta_path = d / "metadata.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            lang = meta.get("config", {}).get("transcription", {}).get("language")
        except (json.JSONDecodeError, AttributeError, OSError):
            continue
        if isinstance(lang, str) and lang:
            return lang
    return "pl"


def read_per_speaker(pipeline_dir: Path) -> Optional[dict]:
    """Read whichever of ``transcript_{A,B}.txt`` exist; None when neither does.

    A one-speaker pipeline collapse (assembly yielding a single stream, or
    ECAPA merging both pieces to one identity) writes only ``transcript_A.txt``.
    Reading just the present transcript lets ``cpwer_meeteval`` charge the
    absent speaker as deletions — the right penalty for a separation failure
    L3 exists to quantify. Requiring both files instead would silently drop the
    mode from the table and bias it optimistically.

    Public: shared with the script drivers (``scripts/sweep_pipeline.py``) so
    this "read whichever speaker exists" policy has exactly one implementation.
    Script→package imports are the allowed direction (SCOPE §7).
    """
    paths = {"A": pipeline_dir / "transcript_A.txt",
             "B": pipeline_dir / "transcript_B.txt"}
    hyp = {label: parse_gt_txt(p) for label, p in paths.items() if p.exists()}
    return hyp or None


def read_mixture(pipeline_dir: Path) -> Optional[list]:
    """Read ``transcript_mixture.txt`` if present. Public (see ``read_per_speaker``)."""
    path = pipeline_dir / "transcript_mixture.txt"
    if not path.exists():
        return None
    return parse_gt_txt(path)


def compute_layer3(
    rec: Recording,
    tcp_collar_s: float = 5.0,
) -> Optional[dict]:
    """ASR error rates per ablation mode + ORC-WER baseline.

    Returns None when the GT transcripts are missing — without them L3 is
    meaningless. Requires reference tiers labelled exactly ``A`` and ``B``
    (enforced corpus-wide by ``scripts/check_gt_eaf.py``); any other label set
    returns None. With at least one pipeline mode populated, returns::

        {
            "ref_lengths": {"A": int, "B": int},
            "ref_untimed": bool,              # True → tcpWER skipped (no GT times)
            "modes": {
                "full":     {"cpwer": …, "cpcer": …, "tcpwer": … or None,
                             "tcp_skipped": …},
                "no_sep":   {…} or None,
                "no_enh":   {…} or None,
                "minimal":  {…} or None,   # both stages off
            },
            "mixture_orc":  {"orc_wer": …, …} or None,
            "mixture_mimo": {"mimo_wer": …, …} or None,
            "tcp_collar_s": float,
        }

    Each populated mode carries the full ``cpwer_meeteval`` result (``cpwer``,
    ``tcpwer``, assignments, error/length counts — unchanged keys) plus the
    character-error-rate under the same cpWER routing: ``cpcer`` (0..1 fraction),
    ``cpcer_errors``, ``cpcer_length``. cpCER is the campaign's co-headline metric
    (Polish morphology inflates WER); surfacing it here means
    ``evaluate_recording`` / ``summarize_layer3`` produce it without the sweep
    script's private scorer. The per-mode dict is built from the shared
    :func:`~asr_pipeline.eval.metrics.per_fragment_metrics` scoring core, so the
    numbers match the sweep/rescore tables exactly.

    When the reference is **untimed** (``start=end=None``), tcpWER is
    skipped — scoring it on placeholder times would fabricate a number
    (SCOPE §4.1). ``ref_untimed=True`` and each mode's ``tcpwer`` is ``None``
    with ``tcp_skipped=True``; cpWER/cpCER and ORC/MIMO (time-agnostic) are
    unaffected.
    """
    ref_utts = {k: v for k, v in load_reference_utterances(rec).items() if v}
    if "A" not in ref_utts or "B" not in ref_utts:
        return None
    ref_lengths = {label: len(utts) for label, utts in ref_utts.items()}
    lang = _resolve_lang(rec)

    # Untimed GT → no per-word time gating is possible; skip tcpWER
    # rather than score it on placeholder times. Visible via ref_untimed +
    # each mode's tcp_skipped flag (surfaced by summarize_layer3).
    ref_untimed = any(is_untimed(utts) for utts in ref_utts.values())
    if ref_untimed:
        dlog("layer3",
             f"compute_layer3[{rec.id}]: untimed reference → tcpWER skipped")

    modes_out: dict[str, Optional[dict]] = {}
    for mode, dir_ in (
        ("full", rec.pipeline_dir),
        ("no_sep", rec.pipeline_nosep_dir),
        ("no_enh", rec.pipeline_noenh_dir),
        ("minimal", rec.pipeline_minimal_dir),
    ):
        if dir_ is None:
            modes_out[mode] = None
            continue
        hyp = read_per_speaker(dir_)
        if hyp is None:
            modes_out[mode] = None
            continue
        # Route through the shared scoring core (the sweep/rescore path) so the
        # cpWER and cpCER numbers here match those tables exactly. ``mix=None``:
        # the single-stream mixture floor is scored once, recording-level, below —
        # not per mode. ``per_fragment_metrics`` also computes the content-floor
        # metrics (orc/mimo/orccer); L3 does not surface them per mode, so they
        # are ignored here.
        pf = per_fragment_metrics(
            ref_utts, hyp, session_id=rec.id, lang=lang,
            tcp_collar_s=tcp_collar_s, skip_tcp=ref_untimed,
        )
        # ``pf["cp"]`` IS the former direct ``cpwer_meeteval(...)`` result —
        # every pre-cpCER key/value is byte-identical; cpcer* are the additions.
        mode_out = dict(pf["cp"])
        cpcer = pf["cpcer"]
        mode_out["cpcer"] = cpcer["cer"]
        mode_out["cpcer_errors"] = cpcer["errors"]
        mode_out["cpcer_length"] = cpcer["length"]
        modes_out[mode] = mode_out

    # Mixture baseline (single-stream) — take whichever mode dir has a
    # transcript_mixture.txt. The mixture transcript is Whisper on the *raw*
    # loaded mixture (`ctx.audio`), which is the same audio in every ablation
    # arm — enhancement writes to `ctx.enhanced_full`, separation never
    # touches `ctx.audio` — so the transcript is identical across modes given
    # deterministic transcription. (The earlier comment justified this by "same
    # backend"; the real reason is "same input audio". The per-mode *directory*
    # differs, the transcribed signal does not.) Any one mode's copy is fine.
    mixture_utts = None
    for d in (rec.pipeline_dir, rec.pipeline_nosep_dir, rec.pipeline_noenh_dir,
              rec.pipeline_minimal_dir):
        if d is None:
            continue
        mixture_utts = read_mixture(d)
        if mixture_utts is not None:
            break
    # Mixture floor scored two ways: ORC (time-fixed reference merge) and
    # MIMO (optimised interleaving). MIMO <= ORC always; it doesn't penalise
    # the unpredictable order Whisper interleaves the speakers in overlaps,
    # and is more robust to faulty reference timestamps (MeetEval paper,
    # Fig 1c). MIMO is the more principled single-stream floor; we keep ORC
    # too for continuity.
    mixture_orc = (
        orc_wer_meeteval(ref_utts, mixture_utts, session_id=rec.id, lang=lang)
        if mixture_utts is not None
        else None
    )
    mixture_mimo = (
        mimo_wer_meeteval(ref_utts, mixture_utts, session_id=rec.id, lang=lang)
        if mixture_utts is not None
        else None
    )

    return {
        "ref_lengths": ref_lengths,
        "ref_untimed": ref_untimed,
        "modes": modes_out,
        "mixture_orc": mixture_orc,
        "mixture_mimo": mixture_mimo,
        "tcp_collar_s": tcp_collar_s,
    }
