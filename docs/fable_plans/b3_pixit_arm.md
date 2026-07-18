# B3 — PixIT joint diarization+separation arm: harness VERIFIED end-to-end

*2026-07-18 (overnight session + same-day follow-up). Status: **both gates
accepted by the author; worker + glue verified end-to-end on one fragment.**
Ready for the dev run on the author's go. Backlog: `thesis/x_notes/backlog.md`
§B3; chapter slot: `[PENDING B3]` in `thesis/claude-writing/ch07/07_arc.md`
§7.8.*

## Verification log (2026-07-18)

- Both HF gates accepted (`speech-separation-ami-1.0` AND the dependent
  `separation-ami-1.0` — the second 403'd separately, as predicted).
- Venv needed two more era-pins beyond the initial set: `matplotlib`
  (unconditional pyannote 3.3.2 import), `speechbrain==1.0.2` (1.1.0 dropped
  the `use_auth_token` kwarg pyannote passes). Recipe in the worker docstring
  is the source of truth.
- **Worker bug found + fixed at smoke:** ToTaToNet sources are unnormalised
  and HOT (peak ≈ 86 raw); soundfile's WAV default subtype PCM_16 flat-clipped
  **40% of samples** → garbage ASR (92.7 cpWER). Fix: one common peak-scale
  across sources (inter-speaker balance preserved, scale recorded in
  `meta.json`) + FLOAT subtype.
- Post-fix smoke (`005cba37__seg00`): 2 clean sources (cross-corr 0.12,
  sensible levels), coherent per-speaker Polish transcripts, pipeline's
  loop-retry machinery active inside the reused WhisperX backend (ASR parity
  confirmed). Score: **76.7 cpWER / 59.4 cpCER** — dominated by a coverage
  gap (95 hyp words vs 155 GT words ≈ 40% deletions: speech PixIT's own
  diarization missed is silence in both sources). Single fragment — NOT a
  result; interpretation waits for the dev run.
- **⚠ Test-exposure disclosure:** the smoke fragment turned out to be in the
  FROZEN TEST split (picked alphabetically, split not checked first — process
  error). One fragment, harness-debug purpose, no decision taken from its
  number; must be disclosed in B3's eventual test prereg. A hard guard now
  exists: `pixit_baseline.py` refuses to score test-split fragments without
  `--allow-test`.

## DEV RESULT (2026-07-18, 23 frags, stock PixIT + num_speakers=2 hint)

**Micro-avg (ref-weighted): 36.6 cpWER / 29.3 cpCER** over 22 scored
fragments — vs v41_merge dev 19.7 / 12.8. Verdict: **real, not a harness
bug**, on three grounds:
1. **Wide, structured distribution** (min 14.8 / median 34.0 / max 82.4): the
   best fragments land in the pipeline's own territory — impossible under a
   uniform glue/scoring bug. Word coverage 0.89 (the test-smoke's 0.61 mass-
   deletion pattern was an outlier).
2. **Failure-mode signature identified**: the tail (5 frags > 50 cpWER) shows
   per-speaker *activity-attribution collapse* — e.g. worst fragment
   33a47eae: 42 utterances in one stream vs 4 in the other while the streams
   themselves stay acoustically distinct (cross-corr 0.04). PixIT's own
   diarization funnels most speech into one source; cpWER punishes the
   miscount maximally. Plus one loud hard failure (9a651086__seg01: PixIT
   finds 1 speaker, worker dies per SCOPE §4 — recorded in
   separate_failures.csv).
3. **Harness is the pipeline's own** (WhisperX config verbatim — its
   loop-retry visibly fired; same meeteval scorer; clipping bug already fixed
   with FLOAT+common-scale output).

Honest caveats for the chapter row: stock AMI-tuned hyperparameters
(as-published external system — B3's design), AMI/English acoustic domain,
`num_speakers=2` hint.

**Sensitivity probes (2026-07-18, author-sanctioned, dev-only — reported as
sensitivity, not selection):**
- Probe A, no `num_speakers` hint: **38.6 / 31.5** — worse. Unconstrained,
  PixIT emits 3–4 speakers on 7/23 fragments, 1 speaker on 2, hard-fails 2.
  The hint helps it.
- Probe B, `clustering.min_cluster_size` 15→2 (the absorption-hypothesis
  knob): **81.9 / 74.7** — catastrophic; surviving micro-clusters shatter the
  local→global speaker assignment (constant stream swapping). Stock 15 wins.

Conclusion: the published config + hint (the main run's setting) is the best
of the three configurations tried — the 36.6/29.3 number stands, and the
probes double as evidence the comparison isn't handicapping the rival.
Artifacts: `_b3_pixit_nohint/`, `_b3_pixit_mcs2/` beside the main run.

Artifacts: `<eval_root>/_b3_pixit/` (per-frag sources, transcripts,
`pixit_scores.csv`, `separate_failures.csv`).

## The question (unchanged from backlog)

Which ASR-pipeline *architecture* transcribes clarin_fragments better — joint
diarization+separation (PixIT: `pyannote/speech-separation-ami-1.0`, one model
emits per-speaker sources + diarization in a single pass) or our modular
region-routed cascade — with ASR (WhisperX-large-v2-pl) and scoring (meeteval
cpWER/cpCER) held fixed. Not a training-data comparison (that's B1).

## Blocker: gated access (author, ~1 minute)

`$HF_TOKEN` gets **403** on `pyannote/speech-separation-ami-1.0` (checked
2026-07-18; the token itself is fine — pyannote diarization loads with it).
Accept the user conditions at
<https://huggingface.co/pyannote/speech-separation-ami-1.0>; if the first run
then 403s on a *dependent* repo (the pipeline config may pull its ToTaToNet /
embedding checkpoints from separate gated repos), accept those too — the
worker fails loud with the blocked repo id.

## What exists

- **`~/pixit_venv`** — isolated venv, `pyannote.audio[separation]==3.3.2`
  (the model card's pin; main venv carries pyannote 4.x for the diarization
  stage, so no sharing — Sortformer/CohereX/Brouhaha precedent). Import- and
  CUDA-verified 2026-07-18; the full pin set that made it work (setuptools<81,
  torch/torchaudio 2.4.1, transformers 4.44.2, lightning 2.4.0) is recorded in
  the worker's docstring recipe. Export `PIXIT_VENV_PY=~/pixit_venv/bin/python`.
- **`scripts/pixit_worker.py`** — subprocess worker on the
  `sortformer_worker.py` template (no repo imports, neutral-cwd safe, JSON+wav
  protocol): one wav in → `source_<label>.wav` per speaker + diarization
  segments JSON + meta. **UNTESTED until the gate opens**; self-test command
  in its docstring.

## Glue design (write + test once the gate opens — not before)

Shape it like `scripts/compare_asr.py` (the proven fixed-audio ASR-swap
harness), not like a pipeline batch tree:

1. For each fragment in the split (`_load_split`-style, `dev` first):
   `$PIXIT_VENV_PY scripts/pixit_worker.py --in frag.wav --out-dir
   <eval>/_b3_pixit/<frag>/` (cwd outside the repo).
2. Transcribe each source with the pipeline's own WhisperX backend (reuse the
   wired backend class the way `compare_asr.py` reuses `_CohereXBackend`) —
   same model/params as the shipped config, so ASR is held fixed.
3. Speaker mapping + scoring: cpWER/cpCER via the exported meeteval helpers
   (`cpwer_meeteval` handles the transcript↔GT speaker permutation, so the
   worker's arbitrary label order is fine). Per-fragment CSV + micro-avg under
   `<eval>/_b3_pixit/`, mirroring `_forensics/asr_compare/` outputs.
4. Expected wrinkles to handle *visibly* (log, don't paper over): PixIT may
   emit ≠2 sources on some fragments (report count; score top-2 by speech
   duration with a note); leakage/duplicated speech across sources double-
   counts insertions — that's part of the measured architecture difference,
   not a bug to fix.

## Disclosures pre-committed (backlog + 07_arc)

- Acoustic-domain asymmetry: AMI-SDM/English training vs Polish conversational
  eval; separation is largely language-agnostic, so the gap read is acoustic,
  not lexical. State it, don't apologize for it.
- Our cascade's numbers come from the frozen v41_merge instrument; PixIT gets
  the same ASR + scorer but none of our attribution machinery (relabel/rescue
  operate on *our* streams) — that asymmetry is the architecture comparison.

## Run plan + prereg

- Dev (23 frags) first — no prereg needed, hours of compute.
- If a test read is wanted for the chapter row: **fresh prereg** (budget spent
  2026-07-04), one-shot, cpCER primary + cpWER confirm, recording-clustered
  bootstrap vs the v41_merge test numbers. The chapter slot works with a
  dev-only number + disclosure if the author prefers to keep test closed.
- **Fallback external** (if PixIT access/quality dead-ends): NOTSOFAR-1 CSS
  baseline (`microsoft/NOTSOFAR1-Challenge`, MIT, CSS weights included) —
  moderate effort, adds a true sliding-window CSS representative; needs its
  ASR stage rewired to WhisperX. Dead ends already checked 2026-07-07:
  DCF-DS (no license/checkpoints), TS-SEP (AGPL, no weights) — cite-only.

## Author decision points

- [ ] Accept the HF gate conditions (unblocks everything).
- [ ] Dev-only chapter row vs one-shot test exposure (fresh prereg).
- [ ] `num_speakers=2` hint vs letting PixIT decide (worker supports both;
      decide before dev runs so the arm is one configuration).
