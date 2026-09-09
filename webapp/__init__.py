"""Showcase web front-end for `asr_pipeline`.

A thin, **read-only consumer** of the pipeline package (SCOPE §1/§4/§7): the
webapp imports `asr_pipeline` and never modifies it. Upload a recording ->
watch the eight stages execute live -> listen to the separated speaker streams
and read the speaker-attributed transcripts.

Modules:

- `app`             FastAPI application factory + every route in `webapp/API.md`.
- `queue`           job registry, the single worker thread, and the `Runner` seam.
- `render`          peak envelopes + result-payload assembly from an output dir.
- `eta`             duration-weighted per-stage ETA estimator (design §5.2).
- `examples_build`  offline CLI that builds `examples_manifest.json` from the
                    frozen v41_merge eval tree.

The binding request/response contract is `webapp/API.md`.
"""
