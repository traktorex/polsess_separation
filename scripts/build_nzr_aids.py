"""Build per-``<nzr>`` listening-aid bundles next to the hand-corrected EAFs.

The author hand-corrects ground-truth transcripts in ELAN on the Windows F:
drive (``/mnt/f/clarin_fragments/<frag_id>/annotation.eaf``) and marks
unintelligible words inline as ``<nzr>`` (stored XML-escaped as ``&lt;nzr&gt;``
inside ``ANNOTATION_VALUE``; sometimes a bare ``<nzr>`` is the whole utterance).
This script produces, for every ``<nzr>`` occurrence, a bundle of listening aids
written *beside the EAF on F:* so resolving an ``<nzr>`` is one click away in the
Windows file explorer.

For each occurrence (= one annotation whose unescaped text contains ``<nzr>``)
it writes ``<drive>/<frag_id>/nzr_aids/<NN>_<tier>_<start>s/`` containing:

1. ``context.txt``    — the occurrence's tier / time / full text, plus the 2
                        preceding and 2 following utterances from BOTH tiers,
                        interleaved chronologically and tier/time-labelled.
2. ``mix.wav``        — the utterance slice (16 kHz) with +-2.5 s padding,
                        clamped to the fragment bounds.
3. ``mix_slow080.wav``,
   ``mix_slow060.wav`` — pitch-preserving tempo-stretched copies of mix.wav
                        (librosa phase-vocoder ``time_stretch``; rate < 1 = slower).
4. ``sep_A.wav``,
   ``sep_B.wav``      — SepFormer separation of the slice.
5. ``enh.wav``        — MossFormerGAN-enhanced slice (16 kHz).
6. ``guesses.txt``    — Whisper-large-v2 (language=pl) candidate readings of
                        mix / sep_A / sep_B / enh, primed with the preceding
                        utterance as ``initial_prompt``, at temperatures 0.0 and
                        0.6. A guess menu, not truth.

Plus a top-level index ``<drive>/NZR_AIDS_INDEX.md`` — one row per occurrence,
regenerated fully on every run (derived state).

Design / judgement calls (documented per the SCOPE spirit — nothing fails
silently, every skipped item is reported):

* **Source audio.** Slices are cut from ``--audio-root`` (the authoritative
  ``~/datasets/eval/clarin_fragments/<id>/<id>.wav``, always present), never from
  the F: copy, which the spec flags as possibly missing. Only the *outputs* land
  on F:.
* **Separator output rate.** The SepFormer checkpoint runs at 8 kHz; its two
  streams are upsampled back to 16 kHz before writing so all wavs in a bundle
  share one sample rate (the author can drag any of them onto the same player).
* **Occurrence granularity.** One occurrence == one *annotation* containing at
  least one ``<nzr>`` (even if an annotation carries two markers). The bundle is
  named by the annotation's tier + start time; collisions are disambiguated by
  the running NN index.
* **Phase-major execution.** All slices are cut first (CPU), then each GPU model
  is loaded once, run over *every* slice, and unloaded before the next loads, so
  the GPU never holds two models. A per-occurrence model failure is caught,
  reported to stdout and into the index row, and never aborts the batch.

Usage::

    source venv/bin/activate
    python scripts/build_nzr_aids.py                 # all `done` fragments
    python scripts/build_nzr_aids.py --skip-existing # cheap re-run
    python scripts/build_nzr_aids.py --fragments 595aa511__seg00 8993ff2e__seg00
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

# Reuse the pipeline's parser + enhancement backend wrapper. The separator is
# loaded via the same parent-project seam the pipeline uses.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from asr_pipeline.eval.transcript_parser import Utterance, parse_eaf  # noqa: E402
from asr_pipeline.stages.enhancement import (  # noqa: E402
    _CLEARVOICE_BACKENDS,
    _ClearVoiceBackend,
)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

NZR = "<nzr>"
PAD_S = 2.5                      # +-padding around each utterance slice
SLOW_RATES = (0.80, 0.60)       # librosa time_stretch rates (< 1 = slower)
CONTEXT_NEIGHBOURS = 2          # utterances of context on each side, both tiers
SAMPLE_RATE = 16_000            # pipeline / output sample rate
WHISPER_MODEL = "large-v2"
WHISPER_TEMPS = (0.0, 0.6)
SEP_CHECKPOINT = "checkpoints/sepformer/SB/128_run/sepformer_SB_best_128k_e41.pt"
SEP_SAMPLE_RATE = 8_000         # rate the SepFormer checkpoint operates at
ENH_BACKEND_KEY = "mossformer_gan_se_16k"
INDEX_NAME = "NZR_AIDS_INDEX.md"

# A fragment id looks like `595aa511__seg00`. Used both to validate robione.txt
# lines and to keep junk lines out.
_FRAG_ID_RE = re.compile(r"^[0-9a-z]+__seg\d+$")


# --------------------------------------------------------------------------- #
# Pure helpers (covered by tests/test_nzr_aids.py)
# --------------------------------------------------------------------------- #


def parse_robione(text: str) -> list[str]:
    """Return the fragment ids marked ``done`` in a ``robione.txt`` body.

    The file is hand-maintained and contains junk: blank lines, free-text
    notes, and even fragment-id-shaped lines that are *not* done
    (``f7ed6ed8__seg00 - remove, ...``). A line counts as done iff its first
    whitespace token is a valid fragment id AND a later token is exactly
    ``done`` (case-insensitive). CRLF endings (drvfs) are tolerated.

    Order of first appearance is preserved; duplicates are dropped.
    """
    out: list[str] = []
    seen: set[str] = set()
    for raw in text.splitlines():
        tokens = raw.replace("\r", "").split()
        if len(tokens) < 2:
            continue
        frag = tokens[0]
        if not _FRAG_ID_RE.match(frag):
            continue
        if not any(t.lower() == "done" for t in tokens[1:]):
            continue
        if frag not in seen:
            seen.add(frag)
            out.append(frag)
    return out


@dataclass
class NzrOccurrence:
    """One ``<nzr>``-bearing annotation in a fragment."""

    frag_id: str
    tier: str
    start: float
    end: float
    text: str
    # 0-based running index across a fragment's occurrences, set by the scanner.
    index: int = 0
    # Chronologically-sorted context utterances (both tiers) carried alongside
    # so the bundle writer has everything in one object.
    preceding: list[tuple[str, Utterance]] = field(default_factory=list)
    following: list[tuple[str, Utterance]] = field(default_factory=list)


def scan_eaf_for_nzr(eaf_path: str | Path, frag_id: str) -> list[NzrOccurrence]:
    """Parse an EAF and return its ``<nzr>`` occurrences with surrounding context.

    Uses :func:`parse_eaf` (ELAN reader; unescapes ``&lt;nzr&gt;`` -> ``<nzr>``).
    An occurrence is any utterance whose text contains ``<nzr>`` — including a
    bare ``<nzr>`` utterance. Context is the ``CONTEXT_NEIGHBOURS`` utterances
    before and after the occurrence in the *global chronological* ordering of
    both tiers (so the author sees what the other speaker was saying too).
    """
    by_tier = parse_eaf(eaf_path)
    # Flatten to a single chronologically-sorted list, carrying the tier label.
    flat: list[tuple[str, Utterance]] = [
        (tier, u) for tier, utts in by_tier.items() for u in utts
    ]
    flat.sort(key=lambda tu: (tu[1].start, tu[1].end))

    occurrences: list[NzrOccurrence] = []
    running = 0
    for i, (tier, u) in enumerate(flat):
        if NZR not in u.text:
            continue
        occ = NzrOccurrence(
            frag_id=frag_id,
            tier=tier,
            start=u.start,
            end=u.end,
            text=u.text,
            index=running,
            preceding=flat[max(0, i - CONTEXT_NEIGHBOURS): i],
            following=flat[i + 1: i + 1 + CONTEXT_NEIGHBOURS],
        )
        occurrences.append(occ)
        running += 1
    return occurrences


def bundle_dir_name(occ: NzrOccurrence) -> str:
    """Windows/drvfs-safe directory name for one occurrence.

    Form: ``<NN>_<tier>_<start>s`` — e.g. ``00_B_6_08s``. The start time is
    formatted with an underscore instead of a decimal point and carries no
    colons or ``<>`` (forbidden on Windows / mangled on drvfs). The tier label
    is sanitised the same way in case ELAN ever yields an exotic tier id.
    """
    start_str = f"{occ.start:.2f}".replace(".", "_")
    tier = re.sub(r"[^0-9A-Za-z_-]", "_", occ.tier)
    return f"{occ.index:02d}_{tier}_{start_str}s"


def _fmt_time(t: float) -> str:
    """``mm:ss.cc`` for human-readable context lines."""
    m = int(t // 60)
    s = t - 60 * m
    return f"{m:02d}:{s:05.2f}"


def render_context(occ: NzrOccurrence) -> str:
    """Render ``context.txt`` for one occurrence."""
    lines: list[str] = []
    lines.append(f"Fragment : {occ.frag_id}")
    lines.append(f"Tier     : {occ.tier}")
    lines.append(
        f"Interval : {_fmt_time(occ.start)} -> {_fmt_time(occ.end)} "
        f"({occ.start:.2f}s -> {occ.end:.2f}s, dur {occ.end - occ.start:.2f}s)"
    )
    lines.append("")
    lines.append("=== <nzr> utterance ===")
    lines.append(f"[{occ.tier} {_fmt_time(occ.start)}] {occ.text}")
    lines.append("")
    lines.append("=== context (both tiers, chronological) ===")
    for tier, u in occ.preceding:
        lines.append(f"  before [{tier} {_fmt_time(u.start)}] {u.text}")
    lines.append(f"> NZR    [{occ.tier} {_fmt_time(occ.start)}] {occ.text}")
    for tier, u in occ.following:
        lines.append(f"  after  [{tier} {_fmt_time(u.start)}] {u.text}")
    return "\n".join(lines) + "\n"


def preceding_prompt(occ: NzrOccurrence) -> str:
    """The nearest preceding utterance text (any tier), for Whisper priming.

    Empty string if there is no preceding context (Whisper accepts an empty
    ``initial_prompt``).
    """
    if not occ.preceding:
        return ""
    return occ.preceding[-1][1].text.strip()


# --------------------------------------------------------------------------- #
# Audio slicing
# --------------------------------------------------------------------------- #


def slice_with_pad(
    audio: np.ndarray, start_s: float, end_s: float, sr: int, pad_s: float
) -> np.ndarray:
    """Cut ``[start-pad, end+pad]`` from ``audio``, clamped to the array bounds."""
    n = len(audio)
    lo = max(0, int(round((start_s - pad_s) * sr)))
    hi = min(n, int(round((end_s + pad_s) * sr)))
    if hi <= lo:  # degenerate (interval outside the array) — clamp to >=1 sample
        lo = max(0, min(lo, n - 1))
        hi = min(n, lo + 1)
    return audio[lo:hi].astype(np.float32)


# --------------------------------------------------------------------------- #
# Per-occurrence work item + index entry
# --------------------------------------------------------------------------- #


@dataclass
class WorkItem:
    """One occurrence's bundle directory + cut slice + accumulating errors."""

    occ: NzrOccurrence
    bundle: Path
    mix: np.ndarray
    prompt: str
    errors: list[str] = field(default_factory=list)
    # Best mix.wav guess (temp 0.0), filled by the transcription phase; surfaced
    # in the index.
    best_guess: str = ""
    # ASR readings: {(variant, temp): text}, filled by the transcription phase.
    guesses: dict[tuple[str, float], str] = field(default_factory=dict)

    def note_error(self, stage: str, exc: Exception) -> None:
        msg = f"{stage}: {type(exc).__name__}: {exc}"
        self.errors.append(msg)
        print(f"    [FAIL] {self.occ.frag_id} {self.bundle.name} {msg}")


@dataclass
class IndexEntry:
    """One row of the top-level index — fresh builds and kept bundles alike.

    The index is regenerated fully on every run, so a ``--skip-existing`` run
    must still produce a row for every occurrence: skipped bundles get an
    entry with ``item=None`` and their best guess recovered from the existing
    ``guesses.txt`` on disk.
    """

    occ: NzrOccurrence
    bundle: Path
    item: Optional[WorkItem] = None  # None == kept (skipped-existing) bundle
    best_guess: str = ""
    status: str = "kept"

    def finalise(self) -> None:
        """Pull post-phase results from the WorkItem (fresh entries only)."""
        if self.item is not None:
            self.best_guess = self.item.best_guess
            self.status = (
                "OK" if not self.item.errors else f"FAIL ({len(self.item.errors)})"
            )


_BEST_GUESS_RE = re.compile(r"^mix\s+@\s+0\.0\s+->\s+(.*)$")


def recover_best_guess(guesses_text: str) -> str:
    """Pull the ``mix @ 0.0`` reading back out of an existing guesses.txt body."""
    for line in guesses_text.splitlines():
        m = _BEST_GUESS_RE.match(line)
        if m:
            return m.group(1).strip()
    return ""


def _md_cell(text: str) -> str:
    """Escape a string for a Markdown table cell, keeping ``<nzr>`` visible.

    Raw ``<nzr>`` would be swallowed as an unknown HTML tag by most renderers
    (Obsidian, VS Code preview, GitHub); backslash-escaping the angle brackets
    makes it render literally while staying readable in plain text.
    """
    return (
        text.replace("|", "\\|")
        .replace("<", "\\<")
        .replace(">", "\\>")
        .replace("\n", " ")
    )


# --------------------------------------------------------------------------- #
# GPU model phases (each loads once, runs over all items, unloads)
# --------------------------------------------------------------------------- #


@torch.no_grad()
def _separate_slice(
    mix_16k: np.ndarray,
    separator: torch.nn.Module,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample 16k->8k, separate, upsample both streams back to 16k.

    Mirrors ``asr_pipeline.stages.separation._separate_single`` (same seam,
    same resampler) but standalone so this helper doesn't depend on the stage's
    config object.
    """
    orig_len = len(mix_16k)
    audio_t = torch.from_numpy(mix_16k).unsqueeze(0)
    audio_lo = AF.resample(audio_t, SAMPLE_RATE, SEP_SAMPLE_RATE).to(device)
    est = separator(audio_lo)  # [1, 2, T_lo]
    s1_lo = est[:, 0, :].cpu()
    s2_lo = est[:, 1, :].cpu()
    s1 = AF.resample(s1_lo, SEP_SAMPLE_RATE, SAMPLE_RATE).squeeze(0).numpy()
    s2 = AF.resample(s2_lo, SEP_SAMPLE_RATE, SAMPLE_RATE).squeeze(0).numpy()

    def _fit(arr: np.ndarray) -> np.ndarray:
        arr = arr.astype(np.float32)
        if len(arr) < orig_len:
            return np.pad(arr, (0, orig_len - len(arr)))
        return arr[:orig_len]

    return _fit(s1), _fit(s2)


def phase_separation(items: list[WorkItem], device: torch.device) -> None:
    """Load SepFormer once, write sep_A/sep_B for every item, unload."""
    from utils.model_utils import load_model_for_inference

    print(f"[sep] loading SepFormer {SEP_CHECKPOINT} on {device}")
    t0 = time.perf_counter()
    separator, _ = load_model_for_inference(SEP_CHECKPOINT, device=str(device))
    separator.eval()
    print(f"[sep] loaded in {time.perf_counter() - t0:.1f}s; "
          f"separating {len(items)} slice(s)")
    try:
        for it in items:
            try:
                s1, s2 = _separate_slice(it.mix, separator, device)
                sf.write(it.bundle / "sep_A.wav", s1, SAMPLE_RATE)
                sf.write(it.bundle / "sep_B.wav", s2, SAMPLE_RATE)
            except Exception as exc:  # noqa: BLE001 — per-item isolation
                it.note_error("separation", exc)
    finally:
        del separator
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()


def phase_enhancement(items: list[WorkItem], device: torch.device) -> None:
    """Load MossFormerGAN once, write enh.wav for every item, unload."""
    model_name, native_sr = _CLEARVOICE_BACKENDS[ENH_BACKEND_KEY]
    print(f"[enh] loading {model_name} on {device}")
    t0 = time.perf_counter()
    backend = _ClearVoiceBackend(model_name, native_sr)
    backend.load(device)
    print(f"[enh] loaded in {time.perf_counter() - t0:.1f}s; "
          f"enhancing {len(items)} slice(s)")
    try:
        for it in items:
            try:
                enh = backend.enhance(it.mix, SAMPLE_RATE)
                sf.write(it.bundle / "enh.wav", enh.astype(np.float32), SAMPLE_RATE)
            except Exception as exc:  # noqa: BLE001
                it.note_error("enhancement", exc)
    finally:
        backend.unload()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _whisper_text(model, audio: np.ndarray, prompt: str, temp: float) -> str:
    """One Whisper decode at a fixed temperature -> stripped text.

    A single-float ``temperature`` disables Whisper's fallback ladder, so the
    decode is the requested temperature exactly (one reading per request).
    """
    result = model.transcribe(
        audio.astype(np.float32),
        language="pl",
        initial_prompt=prompt or None,
        temperature=temp,
        word_timestamps=False,
        verbose=False,
    )
    return (result.get("text") or "").strip()


def phase_transcription(items: list[WorkItem], device: torch.device) -> None:
    """Load Whisper large-v2 once, transcribe every variant of every item, unload.

    Variants: mix / sep_A / sep_B / enh, at each of ``WHISPER_TEMPS``. The
    separated/enhanced wavs are read back from disk (written in the GPU phases),
    so a per-item separation/enhancement failure simply drops that variant from
    the guess menu without losing the others.
    """
    import whisper

    print(f"[asr] loading Whisper {WHISPER_MODEL} on {device}")
    t0 = time.perf_counter()
    model = whisper.load_model(WHISPER_MODEL, device=str(device))
    print(f"[asr] loaded in {time.perf_counter() - t0:.1f}s; "
          f"transcribing {len(items)} item(s)")
    try:
        for it in items:
            # (label, on-disk path or None for the in-memory mix)
            variants: list[tuple[str, Optional[Path]]] = [
                ("mix", None),
                ("sep_A", it.bundle / "sep_A.wav"),
                ("sep_B", it.bundle / "sep_B.wav"),
                ("enh", it.bundle / "enh.wav"),
            ]
            for label, path in variants:
                if path is not None and not path.exists():
                    continue  # its GPU phase failed; skip this variant
                try:
                    if path is None:
                        audio = it.mix
                    else:
                        audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
                    for temp in WHISPER_TEMPS:
                        it.guesses[(label, temp)] = _whisper_text(
                            model, audio, it.prompt, temp
                        )
                except Exception as exc:  # noqa: BLE001
                    it.note_error(f"transcription[{label}]", exc)
            it.best_guess = it.guesses.get(("mix", 0.0), "")
            _write_guesses(it)
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _write_guesses(it: WorkItem) -> None:
    """Write ``guesses.txt`` for one item (called after its ASR variants run)."""
    lines = [
        "ASR candidate readings — a GUESS MENU, not ground truth.",
        "Each line: <variant> @ temp <t> -> decoded text.",
        f"Priming prompt (preceding utterance): {it.prompt!r}",
        "",
    ]
    order = ["mix", "sep_A", "sep_B", "enh"]
    for label in order:
        for temp in WHISPER_TEMPS:
            if (label, temp) in it.guesses:
                lines.append(f"{label:6s} @ {temp:.1f}  ->  {it.guesses[(label, temp)]}")
    if not it.guesses:
        lines.append("(no readings — all ASR variants failed; see errors below)")
    if it.errors:
        lines.append("")
        lines.append("errors:")
        lines.extend(f"  - {e}" for e in it.errors)
    (it.bundle / "guesses.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Slow-audio phase (CPU, no model)
# --------------------------------------------------------------------------- #


def write_slow_versions(it: WorkItem) -> None:
    """Write pitch-preserving tempo-stretched copies of mix.wav."""
    names = {0.80: "mix_slow080.wav", 0.60: "mix_slow060.wav"}
    for rate in SLOW_RATES:
        try:
            slow = librosa.effects.time_stretch(it.mix, rate=rate)
            sf.write(it.bundle / names[rate], slow.astype(np.float32), SAMPLE_RATE)
        except Exception as exc:  # noqa: BLE001
            it.note_error(f"slow[{rate}]", exc)


# --------------------------------------------------------------------------- #
# Index
# --------------------------------------------------------------------------- #


def write_index(index_path: Path, entries: list[IndexEntry], drive_root: Path) -> None:
    """Regenerate the top-level index table from scratch (all occurrences —
    fresh builds and kept bundles alike)."""
    rows = [
        "# `\\<nzr\\>` listening-aid index",
        "",
        f"Regenerated {time.strftime('%Y-%m-%d %H:%M:%S')} — "
        f"{len(entries)} occurrence(s) across "
        f"{len({e.occ.frag_id for e in entries})} fragment(s).",
        "",
        "| Fragment | Tier | Time | Utterance | Bundle | Best mix guess | Status |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for e in entries:
        e.finalise()
        rel = e.bundle.relative_to(drive_root).as_posix()
        rows.append(
            f"| {e.occ.frag_id} | {e.occ.tier} | {_fmt_time(e.occ.start)} "
            f"| {_md_cell(e.occ.text)} | `{rel}` "
            f"| {_md_cell(e.best_guess or '')} | {e.status} |"
        )
    index_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def _load_audio(audio_root: Path, frag_id: str) -> np.ndarray:
    """Load a fragment's authoritative 16 kHz mono audio."""
    wav = audio_root / frag_id / f"{frag_id}.wav"
    if not wav.exists():
        raise FileNotFoundError(f"fragment audio not found: {wav}")
    audio, _ = librosa.load(wav, sr=SAMPLE_RATE, mono=True)
    return audio.astype(np.float32)


def build_items(
    fragments: list[str],
    drive_root: Path,
    audio_root: Path,
    skip_existing: bool,
) -> tuple[list[WorkItem], list[IndexEntry], list[str]]:
    """Scan EAFs, cut slices, create bundle dirs, write the CPU-only artefacts.

    Returns ``(work_items, index_entries, skip_messages)``. ``index_entries``
    covers *every* occurrence — including ``--skip-existing`` keeps, whose best
    guess is recovered from the on-disk ``guesses.txt`` — so the regenerated
    index never loses rows. Fragments with no EAF or no audio are reported
    (not silently dropped) and contribute a skip message.
    """
    items: list[WorkItem] = []
    entries: list[IndexEntry] = []
    skips: list[str] = []
    for frag in fragments:
        eaf = drive_root / frag / "annotation.eaf"
        if not eaf.exists():
            skips.append(f"{frag}: no annotation.eaf (not yet annotated?)")
            continue
        occurrences = scan_eaf_for_nzr(eaf, frag)
        if not occurrences:
            continue
        try:
            audio = _load_audio(audio_root, frag)
        except FileNotFoundError as exc:
            skips.append(str(exc))
            continue

        for occ in occurrences:
            bundle = drive_root / frag / "nzr_aids" / bundle_dir_name(occ)
            if skip_existing and bundle.is_dir() and any(bundle.iterdir()):
                skips.append(f"{frag}/{bundle.name}: exists, skipped")
                guesses_file = bundle / "guesses.txt"
                best = ""
                if guesses_file.exists():
                    best = recover_best_guess(
                        guesses_file.read_text(encoding="utf-8")
                    )
                entries.append(IndexEntry(
                    occ=occ, bundle=bundle, item=None,
                    best_guess=best, status="kept",
                ))
                continue
            bundle.mkdir(parents=True, exist_ok=True)
            mix = slice_with_pad(audio, occ.start, occ.end, SAMPLE_RATE, PAD_S)
            it = WorkItem(
                occ=occ, bundle=bundle, mix=mix, prompt=preceding_prompt(occ)
            )
            # CPU-only artefacts, written immediately so a later GPU crash still
            # leaves a usable (if incomplete) bundle on disk.
            (bundle / "context.txt").write_text(render_context(occ), encoding="utf-8")
            sf.write(bundle / "mix.wav", mix, SAMPLE_RATE)
            write_slow_versions(it)
            items.append(it)
            entries.append(IndexEntry(occ=occ, bundle=bundle, item=it))
    return items, entries, skips


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--drive-root", default="/mnt/f/clarin_fragments",
                   help="Windows F: clarin_fragments root (outputs land here).")
    p.add_argument("--audio-root",
                   default=str(Path.home() / "datasets/eval/clarin_fragments"),
                   help="Authoritative fragment audio root (slices cut from here).")
    p.add_argument("--fragments", nargs="*", default=None,
                   help="Explicit fragment ids (default: all `done` from robione.txt).")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip occurrences whose bundle dir already exists non-empty.")
    return p.parse_args(argv)


def resolve_fragments(args: argparse.Namespace, drive_root: Path) -> list[str]:
    """Explicit ``--fragments`` if given, else the `done` set from robione.txt."""
    if args.fragments:
        return list(args.fragments)
    robione = drive_root / "robione.txt"
    if not robione.exists():
        raise SystemExit(f"robione.txt not found at {robione} and no --fragments given.")
    return parse_robione(robione.read_text(encoding="utf-8", errors="replace"))


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    drive_root = Path(args.drive_root)
    audio_root = Path(args.audio_root).expanduser()

    # Fail loud if the drive isn't mounted (a missing F: silently produces an
    # empty run otherwise).
    if not drive_root.is_dir():
        raise SystemExit(
            f"drive root not mounted / not a directory: {drive_root} "
            f"(is the Windows F: drive mounted under /mnt/f?)"
        )

    fragments = resolve_fragments(args, drive_root)
    print(f"[scan] {len(fragments)} fragment(s) to scan")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    items, entries, skips = build_items(
        fragments, drive_root, audio_root, args.skip_existing
    )
    print(f"[scan] {len(entries)} <nzr> occurrence(s) total; "
          f"{len(items)} fresh bundle(s) to build")
    for s in skips:
        print(f"  [skip] {s}")

    if items:
        # Phase-major: one GPU model at a time over all slices.
        phase_separation(items, device)
        phase_enhancement(items, device)
        phase_transcription(items, device)

    # Index covers every occurrence (fresh + kept), regenerated from scratch.
    index_path = drive_root / INDEX_NAME
    write_index(index_path, entries, drive_root)

    failed = [it for it in items if it.errors]
    print("\n=== summary ===")
    print(f"  fragments scanned : {len(fragments)}")
    print(f"  bundles built     : {len(items)}")
    print(f"  with failures     : {len(failed)}")
    print(f"  skipped           : {len(skips)}")
    print(f"  index             : {index_path}")
    for it in failed:
        print(f"  [FAIL] {it.occ.frag_id}/{it.bundle.name}: {'; '.join(it.errors)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
