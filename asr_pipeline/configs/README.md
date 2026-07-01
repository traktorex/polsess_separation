# `asr_pipeline/configs/` — config rationale

This directory holds YAML configs for the pipeline. The dataclass definitions are
in `asr_pipeline/config.py`; YAMLs override defaults via
`load_pipeline_config_from_yaml()`. Per-knob rationale lives in dataclass field
comments — read those first.

What lives here:

- `default.yaml` — POC-equivalent values; loaded by default.
- `english.yaml` — English preset (per-language WhisperX alignment).
- `frcrn_vadstrict.yaml` — knob variant: FRCRN enhancement + strict VAD.
- `p4_fixed_pad.yaml` — knob-smoke variant: `context_window_mode: fixed_pad`.
- `p5_full_length.yaml` — knob-smoke variant: `output_mode: full_length`.
- `sweep_best_e31.yaml`, `sweep_best_e31_refineplus.yaml` — dev-era finalist
  snapshots (see "Which 'best' is authoritative?" below).

## Which "best" is authoritative? (read before trusting any `sweep_best_*`)

The held-out **TEST finalist is the sweep arm `dr_oa050`** — the `dr_refineplus`
recipe (e31 Observation-Adding + ECAPA2 diarization + 2nd-pass B+ relabel) with
`enhancement.observation_mix_ratio` raised **0.3 → 0.5**. It exists only as an arm
in `scripts/sweep_pipeline.py`, **not** as a standalone YAML here. The authoritative
analysis is `docs/sweep_plan/TEST_ANALYSIS.md`: on the test set only the **separator**
(pipeline vs `nosep`) clears multiple-comparison correction, so `dr_oa050` is kept as
a *justified default*, not a proven-optimal knob setting.

The `sweep_best_e31*.yaml` files are **dev-selected snapshots**, kept for provenance:

- `sweep_best_e31.yaml` — dev OA-0.3 + ECAPA2-diar winner (2026-06-15).
- `sweep_best_e31_refineplus.yaml` — the above + 2nd-pass B+ relabel ("dr_refineplus",
  2026-06-19); `dr_oa050` is this recipe with OA 0.3 → 0.5.

(The older pre-e31 `sweep_best.yaml` and `sweep_best_excl_db15fc57.yaml` snapshots,
and the dev-era `asr_pipeline/SWEEP_FINAL.md` / `SWEEP_RUNLOG.md` writeups, were
removed 2026-07-01 as superseded by `docs/sweep_plan/`.)

What lives below (notes that don't fit in a YAML comment):

## Transcription backend selection (2026-05-25 / -26)

We compared 5 `(backend, model_name)` combinations on a 10-min Polish
conversational recording (CLARIN `442dd69e` debleed channels) with
diarization masking + hand-corrected GT. Each config was evaluated by two
angles — Polish linguistic quality and catastrophic failure modes (loops,
content drops, subtitle hallucinations, mega-segments).

| `backend`  | `model_name`                                | Outcome / why rejected |
|------------|---------------------------------------------|------------------------|
| `whisperx` | `large-v2`                                  | **CHOSEN** — best balance of Polish quality + safety; no catastrophic failures on either channel; word timestamps to ±50 ms. |
| `whisperx` | `large-v3`                                  | Close second. Unique wins on R channel (`przypiąć`, `oblali`) but L-channel regressions (`kontakt sobie jechał`, `stópku` for `słupek`, `awans` for `awarię`). |
| `whisperx` | `bardsai/whisper-large-v2-pl-v2`            | **Rejected** — Polish finetune narrowed robustness on conversation. Invents non-Polish gibberish (`opildować`, `przymieniać`, `kotyk`, `Aniącie`), destroys proper nouns (`Bemowie` → `wymowie`), emits English tokens (`"Low Low Low"`) in silence. Likely overfit to Common Voice 11. |
| `whisper`  | `large-v2`                                  | **Rejected** — L channel fine but R hallucinates in long silences: invented opener, `"Nie ma"` ×6 loop (zero in GT), 40 zero-duration empty segments. |
| `whisper`  | `large-v3`                                  | **Rejected** — catastrophic truncation on long audio: stops transcribing after ~11:25 on L and ~12:24 on R, losing entire final third of the recording. Plus `kukiełki` ×4 loop on L and `"..."` ×14 loop on R. |

Full 5-variant transcripts at `~/datasets/clarin_gotowy/gotowy/whisper_test_debleed/`.

Full thesis writeup: `thesis/`.
