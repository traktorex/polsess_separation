"""Whisper-result formatting helpers (writer side).

Used by both Stage 5's per-stage spill (``stages/transcription.py``) and the
per-recording writer in ``io.write_pipeline_outputs()``.

Three output flavours:

- :func:`format_transcript` — ``[start → end] text`` decimal-seconds plain
  text. The eval module's ``parse_gt_txt`` reads this directly.
- :func:`to_jsonable` — recursively converts a Whisper result dict into
  JSON-safe Python (handles numpy / torch scalars). Used for the
  ``transcript_*.json`` outputs (full Whisper structure, word-level
  alignment, etc.).
- :func:`write_eaf` — ELAN annotation format (XML). Multi-tier, one tier per
  speaker, references the source audio. This is what ELAN opens for
  human GT correction.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable
from xml.dom import minidom

import numpy as np
import torch


def format_transcript(whisper_result: dict) -> str:
    """Render a Whisper result dict to the per-segment text format.

    Each segment becomes one line: ``[start → end]  text`` with decimal-second
    timestamps. No speaker header (the file's name is `transcript_<label>.txt`,
    which encodes the speaker).

    Returns an empty string when ``whisper_result`` is None or empty. When the
    result has no segments, falls back to the top-level ``text`` field.
    """
    if not isinstance(whisper_result, dict):
        return ""
    segments = whisper_result.get("segments") or []
    if not segments:
        return (whisper_result.get("text") or "").strip()
    lines = []
    for seg in segments:
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        text = (seg.get("text") or "").strip()
        lines.append(f"[{start:6.2f} → {end:6.2f}]  {text}")
    return "\n".join(lines)


def to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy / torch scalars to plain Python for json.dump.

    Handles ``np.floating``, ``np.integer``, ``np.ndarray``, and ``torch.Tensor``
    at any nesting depth within dicts and lists / tuples. Anything else passes
    through unchanged.
    """
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return obj


# ---------------------------------------------------------------------------
# EAF (ELAN annotation format)
# ---------------------------------------------------------------------------


def _whisper_segments_to_tuples(result: dict) -> list[tuple[float, float, str]]:
    """Pull ``(start, end, text)`` tuples out of a Whisper-style result dict.

    Skips segments with empty text. Returns a list in input order (Whisper
    already emits segments in time order; we don't re-sort).
    """
    if not isinstance(result, dict):
        return []
    out: list[tuple[float, float, str]] = []
    for seg in result.get("segments") or []:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        out.append((
            float(seg.get("start", 0.0)),
            float(seg.get("end", 0.0)),
            text,
        ))
    return out


def write_eaf(
    utts_by_speaker: dict[str, list[tuple[float, float, str]]],
    media_path: Path,
    eaf_path: Path,
    locale: str = "pl",
) -> None:
    """Write an ELAN EAF file with one parallel tier per speaker.

    Parameters:
        utts_by_speaker: mapping ``{speaker_label: [(start_s, end_s, text), ...]}``.
            Each speaker becomes a tier named ``Speaker_<label>``. Empty-text
            entries are dropped by the caller (use :func:`_whisper_segments_to_tuples`).
        media_path: path to the audio file. The MEDIA_DESCRIPTOR carries both
            an absolute ``file://`` URL and a relative URL from the EAF's
            directory so ELAN finds the audio whether the dir was moved or not.
        eaf_path: where to write the EAF.
        locale: default tier locale (POSIX language code, e.g. ``"pl"``,
            ``"en"``). Affects ELAN's spell-check + IME defaults; doesn't
            affect parsing.

    Implementation notes:

    - Time units are milliseconds (the ELAN convention). We deduplicate time
      values across speakers so the TIME_ORDER block stays compact even when
      both speakers share boundaries.
    - One LINGUISTIC_TYPE (``"default-lt"``) is declared; all tiers reference it.
    - The bare-minimum CONSTRAINT block is included — ELAN tolerates its
      absence but writes it back in when you save, so we emit it pre-emptively
      to keep round-trips clean.
    """
    media_path = Path(media_path)
    eaf_path = Path(eaf_path)

    # --- Unique time slots --------------------------------------------------
    times_ms: set[int] = set()
    for utts in utts_by_speaker.values():
        for start, end, _ in utts:
            times_ms.add(int(round(start * 1000)))
            times_ms.add(int(round(end * 1000)))
    sorted_times = sorted(times_ms)
    ts_id_by_ms = {ms: f"ts{i+1}" for i, ms in enumerate(sorted_times)}

    # --- Root --------------------------------------------------------------
    root = ET.Element("ANNOTATION_DOCUMENT", {
        "AUTHOR": "asr_pipeline",
        "DATE": "2025-01-01T00:00:00+00:00",
        "FORMAT": "3.0",
        "VERSION": "3.0",
    })

    # --- Header + MEDIA_DESCRIPTOR -----------------------------------------
    header = ET.SubElement(root, "HEADER", {
        "MEDIA_FILE": "",
        "TIME_UNITS": "milliseconds",
    })
    try:
        rel_path = os.path.relpath(media_path, start=eaf_path.parent)
    except ValueError:
        rel_path = media_path.name
    rel_url = rel_path.replace(os.sep, "/")
    if not rel_url.startswith(("./", "../")):
        rel_url = "./" + rel_url
    ET.SubElement(header, "MEDIA_DESCRIPTOR", {
        "MEDIA_URL": "file://" + str(media_path.resolve()),
        "MIME_TYPE": "audio/x-wav",
        "RELATIVE_MEDIA_URL": rel_url,
    })

    # --- TIME_ORDER --------------------------------------------------------
    time_order = ET.SubElement(root, "TIME_ORDER")
    for ms in sorted_times:
        ET.SubElement(time_order, "TIME_SLOT", {
            "TIME_SLOT_ID": ts_id_by_ms[ms],
            "TIME_VALUE": str(ms),
        })

    # --- Tiers (one per speaker) -------------------------------------------
    ann_counter = 0
    for spk_label, utts in utts_by_speaker.items():
        tier = ET.SubElement(root, "TIER", {
            "DEFAULT_LOCALE": locale,
            "LINGUISTIC_TYPE_REF": "default-lt",
            "TIER_ID": f"Speaker_{spk_label}",
        })
        for start, end, text in utts:
            if not text.strip():
                continue
            ann_counter += 1
            ann = ET.SubElement(tier, "ANNOTATION")
            aa = ET.SubElement(ann, "ALIGNABLE_ANNOTATION", {
                "ANNOTATION_ID": f"a{ann_counter}",
                "TIME_SLOT_REF1": ts_id_by_ms[int(round(start * 1000))],
                "TIME_SLOT_REF2": ts_id_by_ms[int(round(end * 1000))],
            })
            v = ET.SubElement(aa, "ANNOTATION_VALUE")
            v.text = text.strip()

    # --- LINGUISTIC_TYPE + standard CONSTRAINTs -----------------------------
    ET.SubElement(root, "LINGUISTIC_TYPE", {
        "GRAPHIC_REFERENCES": "false",
        "LINGUISTIC_TYPE_ID": "default-lt",
        "TIME_ALIGNABLE": "true",
    })
    for stereotype, desc in [
        ("Time_Subdivision", "Time subdivision of parent annotation's time interval, no time gaps allowed within this interval"),
        ("Symbolic_Subdivision", "Symbolic subdivision of a parent annotation. Annotations refering to the same parent are ordered"),
        ("Symbolic_Association", "1-1 association with a parent annotation"),
        ("Included_In", "Time alignable annotations within the parent annotation's time interval, gaps are allowed"),
    ]:
        ET.SubElement(root, "CONSTRAINT", {"DESCRIPTION": desc, "STEREOTYPE": stereotype})

    rough = ET.tostring(root, encoding="utf-8")
    pretty = minidom.parseString(rough).toprettyxml(indent="  ", encoding="UTF-8")
    eaf_path.write_bytes(pretty)


def write_eaf_from_whisper_results(
    results_by_speaker: dict[str, dict],
    media_path: Path,
    eaf_path: Path,
    locale: str = "pl",
) -> int:
    """Convenience wrapper: takes raw Whisper result dicts (the shape stored
    in ``ctx.transcripts``) and produces an EAF.

    Returns the total number of annotations written (sum across tiers).
    Returns 0 (and does not write) when no speaker has any non-empty text.
    """
    utts_by_speaker = {
        spk: _whisper_segments_to_tuples(res)
        for spk, res in results_by_speaker.items()
    }
    utts_by_speaker = {k: v for k, v in utts_by_speaker.items() if v}
    if not utts_by_speaker:
        return 0
    write_eaf(utts_by_speaker, media_path, eaf_path, locale=locale)
    return sum(len(v) for v in utts_by_speaker.values())
