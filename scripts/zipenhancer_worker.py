"""Standalone ModelScope ZipEnhancer (acoustic-noise-suppression) worker.

Run as a SUBPROCESS by `asr_pipeline.stages.enhancement._ZipEnhancerBackend`.
The reason it's a separate process: the repo has a local top-level `datasets/`
package (the source-separation dataset registry) that shadows the pip-installed
HuggingFace `datasets` package that modelscope's pipeline framework imports
(`from datasets import Dataset, ...`). In-process, running from the repo root,
that import resolves to the repo's `datasets/` and crashes. A subprocess whose
`sys.path[0]` is this script's dir (scripts/, no `datasets/` there) and whose
cwd is neutral resolves `import datasets` to the installed HF package.

Keep this script free of any `import` from the repo (no asr_pipeline, etc.).

Usage:
    python zipenhancer_worker.py --in IN.wav --out OUT.wav [--model ID] [--sr 16000]
"""

import argparse

import numpy as np
import soundfile as sf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--model", default="iic/speech_zipenhancer_ans_multiloss_16k_base")
    ap.add_argument("--sr", type=int, default=16_000)
    a = ap.parse_args()

    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks

    ans = pipeline(Tasks.acoustic_noise_suppression, model=a.model)
    res = ans(a.inp)
    pcm = res["output_pcm"] if isinstance(res, dict) else res
    if isinstance(pcm, (bytes, bytearray)):
        out = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    else:
        arr = np.asarray(pcm)
        out = (arr.astype(np.float32) / 32768.0) if arr.dtype == np.int16 \
            else arr.astype(np.float32).squeeze()
    sf.write(a.out, out, a.sr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
