"""Standalone CohereX (Diffio-AI/CohereX) transcription worker.

Run as a SUBPROCESS by ``asr_pipeline.stages.transcription._CohereXBackend``.
It runs in the ISOLATED CohereX venv — the ``coherex`` package + its transformers
pins + the 2B Cohere model conflict with the main venv, so they cannot share it.
The backend invokes this script with the ``$COHEREX_VENV_PY`` interpreter, NOT the
main-venv python. (Same isolation rationale as the Brouhaha scorer; contrast
``zipenhancer_worker.py``, which only escapes an import shadow and still uses the
main venv via ``sys.executable``.)

Keep this script free of any ``import`` from the repo (no ``asr_pipeline`` etc.) —
it must import cleanly under the CohereX venv, which does not have the repo on path,
and runs from a neutral cwd so it never resolves the repo's local ``datasets/`` pkg.

Transcribes ONE mono-16 kHz wav with Cohere ASR + pyannote VAD + a wav2vec2 forced
aligner, and writes a JSON result the backend maps into the pipeline's
``{text, language, segments:[{start,end,text,words}]}`` shape. Parity with the
standalone ``run_coherex.py`` used for the thesis CohereX comparison.

Usage:
    python coherex_worker.py --in IN.wav --out OUT.json \
        --asr-model CohereLabs/cohere-transcribe-03-2026 \
        --align-model jonatasgrosman/wav2vec2-large-xlsr-53-polish \
        --language pl [--vad pyannote] [--chunk-size 30] [--batch-size 4] \
        [--vad-onset 0.5] [--vad-offset 0.363] \
        [--rep-penalty 1.2] [--no-repeat-ngram 3]
"""
import argparse
import json
import os
import sys


def _unstub_tf():
    """coherex.asr / coherex.alignment set sys.modules["tensorflow"]=None to keep
    transformers from importing TF. einops' backend probe (pyannote VAD) then does
    `import tensorflow`, sees the None entry as "present but broken", and crashes.
    Pop the None stubs so einops cleanly skips the TF backend. Call after any
    coherex lazy import that re-inserts the stubs."""
    for _m in ("tensorflow", "tensorflow_text", "keras"):
        if sys.modules.get(_m) is None and _m in sys.modules:
            del sys.modules[_m]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--asr-model", default="CohereLabs/cohere-transcribe-03-2026")
    ap.add_argument("--align-model",
                    default="jonatasgrosman/wav2vec2-large-xlsr-53-polish")
    ap.add_argument("--language", default="pl")
    ap.add_argument("--vad", default="pyannote",
                    choices=["pyannote", "firered", "none"])
    ap.add_argument("--chunk-size", type=float, default=30.0)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--vad-onset", type=float, default=0.500)
    ap.add_argument("--vad-offset", type=float, default=0.363)
    ap.add_argument("--max-new-tokens", type=int, default=448)
    ap.add_argument("--rep-penalty", type=float, default=None)
    ap.add_argument("--no-repeat-ngram", type=int, default=None)
    a = ap.parse_args()

    import coherex
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    token = os.environ.get("HF_TOKEN")

    model = coherex.load_model(
        a.asr_model, device=device,
        compute_type="float16" if device == "cuda" else "float32",
        language=a.language, vad_method=a.vad,
        vad_options={"vad_onset": a.vad_onset, "vad_offset": a.vad_offset},
        asr_options={"punctuation": True, "suppress_numerals": False,
                     "max_new_tokens": a.max_new_tokens},
        use_auth_token=token,
    )
    _unstub_tf()

    # Decode tuning: base Cohere ships no generation_config.json, so these default
    # OFF (1.0 / 0). CohereX's chunk decode is greedy (do_sample=False, num_beams=1);
    # repetition_penalty / no_repeat_ngram_size are only honored via generation_config.
    if a.rep_penalty is not None or a.no_repeat_ngram is not None:
        gconf = model.model.generation_config
        if a.rep_penalty is not None:
            gconf.repetition_penalty = float(a.rep_penalty)
        if a.no_repeat_ngram is not None:
            gconf.no_repeat_ngram_size = int(a.no_repeat_ngram)

    align_model, align_meta = coherex.load_align_model(
        a.language, device, model_name=a.align_model)
    _unstub_tf()

    audio = coherex.load_audio(a.inp)
    result = model.transcribe(audio, batch_size=a.batch_size,
                              chunk_size=a.chunk_size, print_progress=False)
    if result.get("segments"):
        aligned = coherex.align(result["segments"], align_model, align_meta,
                                audio, device, return_char_alignments=False)
        segments = aligned["segments"]
    else:
        segments = []

    def seg_json(seg):
        out = {"start": round(float(seg["start"]), 3),
               "end": round(float(seg["end"]), 3),
               "text": (seg.get("text") or "").strip(),
               "words": []}
        for w in seg.get("words", []) or []:
            wd = {"word": w.get("word", "")}
            for k in ("start", "end", "score"):
                if w.get(k) is not None:
                    wd[k] = round(float(w[k]), 3)
            out["words"].append(wd)
        return out

    payload = {
        "language": a.language,
        "text": " ".join((s.get("text") or "").strip() for s in segments).strip(),
        "segments": [seg_json(s) for s in segments],
    }
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
