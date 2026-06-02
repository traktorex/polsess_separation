# `asr_pipeline/configs/` — config rationale

This directory holds YAML configs for the pipeline. The dataclass definitions are
in `asr_pipeline/config.py`; YAMLs override defaults via
`load_pipeline_config_from_yaml()`. Per-knob rationale lives in dataclass field
comments — read those first.

What lives here:

- `default.yaml` — POC-equivalent values; loaded by default.
- `p4_fixed_pad.yaml` — knob-smoke variant: `context_window_mode: fixed_pad`.
- `p5_full_length.yaml` — knob-smoke variant: `output_mode: full_length`.

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
