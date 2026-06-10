# ASR Pipeline — Scope Contract

**What this is.** The author's intent model for `asr_pipeline/`, written down so it
stops living only in his head. Agents (review loops included) read this before
changing code under `asr_pipeline/`. When a reviewer instinct ("add a fallback",
"handle this edge case") conflicts with this file, this file wins. When this file
conflicts with the author, the author wins — then this file gets updated.

Status: v1.1, drafted 2026-06-10 from a structured interview, approved by the
author the same day. `UNDECIDED` markers are real open questions reserved for
the author — not placeholders to be quietly resolved by an agent.
v1.2 (2026-06-10, author-approved): ledger sites renamed line numbers → symbols;
four fallback sites touched by the deep-review pass added as rows; the
BWE-ImportError REMOVE verdict verified already absent (see ledger footnote).

## 1. Purpose and lifecycle

The pipeline has **two lives**:

- **Life 1 — thesis instrument (now).** Produce the L1/L2/L3 evaluation tables
  that show the effect of speech-separation preprocessing on Polish 2-speaker
  ASR. The thesis discusses the pipeline's *design*, not its implementation;
  the code is a tool that produces results, not an exhibit that gets read
  line-by-line.
- **Life 2 — CLARIN platform service (after the thesis).** Deployed on the
  CLARIN platform (PJATK server, has job scheduling) as a transcript
  pre-generation tool: annotators creating speech corpora get a machine draft
  to correct instead of transcribing from scratch. Academic and possibly
  commercial use.

**Operating rule: build for Life 1, don't block Life 2.** Platform-grade
engineering (job queues, APIs, service hardening) is out of scope until
deployment work explicitly starts. But "it's just thesis code, nobody will run
it after August" is never a valid justification — this code is not abandoned on
defense day.

## 2. Users

- Now: the author. Supervisors should be able to **run** it, not just read it —
  keep setup friction low: documented env vars, no machine-specific paths in
  code (paths belong in configs).
- Later: CLARIN platform operators and, indirectly, annotators.
- Maintainers after the thesis: the author and (probably) the supervisor.
  Documentation targets "supervisor can run and maintain it" — not onboarding
  a team of strangers.

## 3. Input domain

Current universe (all Polish, exactly 2 speakers, 16 kHz a non-issue):

- `~/datasets/clarin_all_2speaker_fragments/` — 128-recording eval set,
  ~90 s overlap-rich fragments cut from the full 2-speaker CLARIN download
  (`~/datasets/clarin_all_2speakers/clarin_download/`, full recordings
  10–90 min).
- `~/datasets/clarin_gotowy/gotowy/` — 6 full-length recordings, unique in
  having 2 separate channels; split L/R and debleeded → oracle channels for
  intrusive metrics.

At hand but undecided: LibriCSS; possibly other English datasets for
tuning/testing. The code must not *assume* the input universe is closed
(Life 2 forbids that), but it also must not grow handling for datasets nobody
has decided to use.

**Speaker count.** Today: 2, everywhere. Direction of travel: the speaker count
should eventually come from diarization, and the pipeline should adapt
(1 speaker → skip separation entirely; 3+ → like today except concurrent
3-speaker overlap, which may need a future 3-speaker separation model). Do
**not** build N-speaker support now; do avoid hard-coding `2` in *new* code
where a parameter is free.

Known anomaly: voice-like background sources (e.g. GPS navigation) can diarize
as a phantom 3rd speaker. Handling `UNDECIDED` — the current warn-and-continue
in assembly stays until decided.

## 4. Error philosophy

1. **No silent substitution.** The pipeline must never report success while
   having done something other than what was configured. A missing dependency
   for a configured backend is a startup crash, not a quiet downgrade — the
   user must never believe BWE (or anything else) ran when it didn't.
2. **Crash semantics.** Single-recording run: fail loud, die. Batch/eval run:
   a failure kills *that recording*, is recorded as a failure, and the batch
   continues. Degrade-gracefully-and-warn: approximately never.
3. **Warnings must be visible** (stdout, not only the debug log). There is
   currently no sanctioned warn-and-continue tier for events that affect
   output quality — if quality is compromised, it should probably be an error.
   What stays visible in deployment gets decided at deployment time.
4. **Input-layout tolerance is not substitution.** Reading several *known*
   on-disk layouts (eval tree variants) is fine; it changes how inputs are
   found, not what the pipeline does to them.

## 5. Fallback ledger

Verdicts from the 2026-06-10 interview; rows marked *(review 2026-06-10)* were
added after the deep-review pass, author-approved. New fallbacks join this
table or they don't get merged. Sites are named by symbol, not line number —
review passes shift lines too fast for refs to stay honest.

| Site | What it does | Verdict |
|---|---|---|
| `transcription.py` `_normalise_result`; `io.py` `write_pipeline_outputs` (EAF locale → literal `"pl"`) | missing `language` in WhisperX result → config value | **KEEP** — the operator declared the language; trusting config is correct |
| `eval/recordings.py` `Recording` / `load_recording` | eval-tree layout fallbacks (.txt+.rttm; old `mixture.wav` symlink) | **KEEP** (rule 4); not set in stone, prune layouts that die |
| `eval/metrics.py` `_digits_to_words_pl` (module-level `_num2words` import) | number-to-words dep missing → digits stay digits | **UNDECIDED** — silently changes scores with environment; candidate: make it a hard dep |
| `stages/assembly.py` `_assign_overlaps` | straight-through fallback in overlap assignment | **UNDECIDED** — needs a dedicated look |
| `stages/assembly.py` `_assign_overlaps` (>2 speakers warn) | 3rd speaker → warn and continue | **KEEP for now** — see §3 phantom-speaker anomaly |
| `stages/transcription.py` `_skip_transcription` *(review 2026-06-10)* | short (<0.5 s) or silent stream → empty transcript, Whisper not called | **KEEP** — silence in, silence out; Whisper hallucinates Polish on the assembler's all-zeros no-event sentinel, which L3 then scores as insertions. Logged visibly via dlog |
| `stages/enhancement.py` `_MIN_ENHANCE_SAMPLES` *(review 2026-06-10)* | input <256 samples (16 ms) → passed through unenhanced | **KEEP** — below one STFT frame; nothing to enhance |
| `stages/post_separation_processing.py` `_APBWEBackend.extend` (`< n_fft` gate) *(review 2026-06-10)* | input shorter than one STFT frame → passed through without BWE | **KEEP** — reflect-pad STFT crashes below this; reachable only via `context_window_mode: none`, which no shipped config sets |
| `stages/post_separation_processing.py` `_FLOWHIGH_MIN_SAMPLES` *(review 2026-06-10)* | input <512 samples (32 ms) → passed through without BWE | **KEEP** — same as the AP-BWE row (FlowHigh frames at 48 kHz) |

Resolved rows:

- BWE backend ImportError → continues without BWE (**REMOVE**, 2026-06-10
  interview): verified already absent on audit the same day — both BWE
  backends fail loud (FlowHigh re-raises `ImportError` with an install hint;
  AP-BWE raises `FileNotFoundError` on a missing checkpoint). Nothing to remove.

## 6. Configuration: live variables vs frozen decisions

**Live** (still being varied for the thesis):

- Enhancement backend — sweep ongoing; FRCRN currently leading.
- VAD thresholds.
- BWE backend: `ap_bwe` (16 kHz-native, slightly worse) vs `flowhigh` (better,
  48 kHz → downsample) — decision blocked on a compute-cost comparison.
  `naive` = VAD mask only, no BWE.
- Transcription backend: `whisperx` is the workhorse; vanilla `whisper` is
  `UNDECIDED` — possibly kept to demonstrate WhisperX's advantage, possibly
  dropped.
- Padding ablation knobs (`p4_fixed_pad` / `p5_full_length`).

**Frozen / removable:**

- `mpsenet` — consistently the worst enhancement backend; candidate for
  removal once the config sweep concludes. Note: it is still the dataclass
  *default* today — that default should not outlive the sweep conclusion.

**Ablation surface (fixed for the thesis):** full pipeline vs (no-sep, yes-enh)
vs (yes-sep, no-enh) vs (no-sep, no-enh) vs whisper-on-raw-mixture (no
splitting at all). The (no-sep, no-enh) arm is `pipeline_minimal` in
`run_pipeline_on_recording.py` and is scored by L3 as mode `minimal`
(gap closed 2026-06-10, author-approved — see §10 q6).

**Lifetime rule:** a knob lives as long as it serves a planned comparison.
Once the thesis decision is made and written down, losing branches may be
deleted — git remembers them.

## 7. Rules for agents

- New dependencies / config knobs / fallback branches: *free within reason* —
  "reason" means it serves Life 1 or visibly unblocks Life 2. "Might be useful
  someday" fails review.
- A new defensive branch requires a **real observed trigger** (name the
  recording or the reproducible case). Hypothetical robustness is sediment.
- Tests are a regression net, not a thesis deliverable. Don't write tests for
  inputs outside §3's universe.
- Performance is secondary to correctness and reproducibility, but real:
  runtime may influence decisions (see the flowhigh question). Optimization
  work needs a measured pain point, not a hunch.
- **Coupling:** `asr_pipeline/` is expected to be extracted into its own
  project eventually. Do not deepen imports from the parent
  `polsess_separation` codebase; keep the boundary clean.

## 8. Done criteria

- **Life 1 done:** the pipeline processes 2-speaker recordings into
  per-speaker transcripts, and the evaluation shows the outputs beat running
  Whisper on the raw mixture. Past that point, only bugfixes that block the
  thesis tables.
- **Life 2** starts as its own explicitly-scoped effort (extraction +
  deployment). It does not creep into Life 1 work item by item.

## 9. Life 2 design directions (recorded, not scheduled)

Facts and directions that constrain code *now* but whose build-out waits until
deployment work starts:

- **Output format.** The platform has no defined transcript format yet
  (annotation currently happens outside it). Presumptive target: `.eaf`
  (ELAN's XML — an open format) with per-speaker tiers and timestamps, so an
  annotator can continue directly from the machine draft. Don't build the
  exporter until it's needed or it serves Life 1.
- **Not Polish-only.** The platform will serve other languages. The wav2vec2
  alignment model must stay swappable like every other model; new code must
  not pin Polish any deeper than it already is.
- **Long recordings.** End-to-end verified on one ~15-minute recording
  (442dd69e from `clarin_gotowy`, good output). 60–90-minute recordings are
  untested; a smoke test on one is a known TODO before deployment work starts.

## 10. Open questions

These are the **author's** decisions. Agents do not resolve them on their own
initiative — `UNDECIDED` means frozen until the author rules.

1. Phantom 3rd speaker (GPS-navigation case): error, ignore-the-extra-cluster,
   or merge-into-nearest? Needs a decision informed by real recordings.
2. `eval/metrics.py:62` digits fallback — hard dep or documented behavior?
3. Assembly straight-through fallback — keep, narrow, or remove?
4. flowhigh vs ap_bwe — measure compute cost; if marginal, the quality winner
   takes the default.
   *Data gathered 2026-06-10 (5 pilot fragments, deterministic,
   frcrn_vad_strict base; `scripts/sweep_pipeline.py` configs
   `frcrn_vad_strict_flowhigh{,16}`): compute — flowhigh ~10.8× realtime
   (+7–9 s per 90 s fragment ≈ +15% pipeline wall time, 697 MB peak) vs
   ap_bwe 76–599× realtime (sub-second, 244 MB). Quality (cpWER / CER) —
   ap_bwe 16.92 / 11.65; flowhigh@16k-in 17.42 / 11.48; flowhigh@8k-in
   18.81 / 12.70. Notes: default.yaml ships `flowhigh_input_sr: 8000`,
   clearly the worse setting; at 16k-in flowhigh is ~tied (n=5, mixed
   per-fragment direction) while costing more compute + an extra git dep.
   The decision remains the author's.*
5. Vanilla Whisper backend — keep as a thesis comparison or delete?
6. ~~Missing (no-sep, no-enh) ablation arm — add to
   `run_pipeline_on_recording.py`?~~ **RESOLVED 2026-06-10 (author): yes.**
   The producer already existed (`pipeline_minimal`, both stages off); the
   eval layer now discovers it (`Recording.pipeline_minimal_dir`), scores it
   (L3 mode `minimal`), and surfaces it (`summarize_layer3.minimal_cpwer`).
7. mpsenet removal + default-backend change after the sweep concludes.
