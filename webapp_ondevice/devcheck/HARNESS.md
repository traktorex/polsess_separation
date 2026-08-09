# devcheck — how to see the Road 2 app without a browser window

You are building `webapp_ondevice/site/` and you cannot open a browser. This
directory is how you look at it anyway: `serve.py` puts the app on
`http://127.0.0.1`, `check_page.py` loads it in headless Chromium, screenshots
it, and prints every console message, uncaught exception, and failed request.

**Read a screenshot after every visible change, and never call UI work done on
a run that did not exit 0.**

## Prerequisites

Both scripts run under the shared Playwright venv built for the Road 1 webapp —
`~/playwright_venv/bin/python`, Chromium already installed. Do **not** create a
new venv and do **not** pip-install into the repo venv.

`check_page.py` has that interpreter in its shebang, so `./check_page.py` works.
`serve.py` is stdlib only, so any `python3` runs it.

Write screenshots to your scratchpad, never into the repo.

## (a) Serve the site

`file://` is fine for a single self-contained HTML page, but the real app
`fetch()`es its ONNX Runtime `.wasm` and its `.onnx` models, and `file://`
blocks that. Serve it:

```bash
# start in the background, then keep working
python3 webapp_ondevice/devcheck/serve.py webapp_ondevice/site --port 8123
#   -> [serve] http://127.0.0.1:8123/
```

It sends `Content-Type: application/wasm` for `.wasm`, `text/javascript` for
`.mjs`, `Cache-Control: no-store` (so your edits take effect on reload), and
binds 127.0.0.1 only. Requests are logged to stderr — that log is where you see
a 404 on a model file. Stop it by killing the background job.

Add `--coi` if the app needs `SharedArrayBuffer` (multi-threaded ORT-Web wasm);
it sends `Cross-Origin-Opener-Policy: same-origin` +
`Cross-Origin-Embedder-Policy: require-corp`. Leave it off otherwise — it also
blocks cross-origin assets that do not send CORP headers.

## (b) Screenshot a state

```bash
~/playwright_venv/bin/python webapp_ondevice/devcheck/check_page.py \
    http://127.0.0.1:8123/index.html \
    --hash "state=S6&clip=A" \
    --shot "$SCRATCH/s6_light.png"
```

Flags:

| flag | meaning |
| --- | --- |
| `--hash "state=S6&clip=A"` | append a fragment (the app deep-links its state there) |
| `--width N` / `--height N` | viewport, default **390×844** (phone) — use `--width 1280 --height 900` for desktop |
| `--dark` | emulate `prefers-color-scheme: dark` |
| `--full-page` | capture the whole scrollable page instead of the viewport |
| `--wait-selector CSS` | wait for an element before capture (use it for "app ready") |
| `--click CSS` | click an element before capture; repeatable, applied in order |
| `--wait-ms N` | settle time before capture |
| `--js "EXPR"` | evaluate a JS expression and print the JSON result |
| `--lenient` | do not fail the run on failed requests (see exit codes) |

Order after load: `--wait-selector`, then the `--click`s in order, then
`--wait-ms`, then `--js`, then the screenshot. A local path is accepted instead
of a URL and is converted to `file://`.

Then **Read the PNG** — that is the point of the exercise.

## (c) Probe the app with `--js` and `--click`

`--js` evaluates one expression in the page and prints it as JSON. Use it to
assert state rather than squinting at pixels:

```bash
--js "({state: S.state, clip: S.clip, regions: document.querySelectorAll('.region').length})"
```

`--click` drives the UI, so you can reach states that have no deep-link:

```bash
--wait-selector "#devbar" --click '#devbar [data-clip="B"]' --wait-ms 300
```

Multi-statement probes need arrow-function form: `--js "() => { ...; return x; }"`.

## (d) Exit codes

| code | meaning |
| --- | --- |
| **0** | clean — zero error-level console messages, zero uncaught page errors, zero failed requests |
| **1** | the page reported problems; the `FAIL:` line has the counts and the sections above it have the detail |
| **2** | the harness could not run the check — bad path, navigation failure, `--wait-selector` timeout, `--js` threw |

A *failed request* is a network-level failure (connection reset, aborted fetch)
**or** an HTTP response with status ≥ 400 — a 404 on a model file is a response,
not a network failure, so both are counted.

`--lenient` excuses failed requests *and* the browser's own
`Failed to load resource` console error for them — use it for layout checks made
before the `.onnx`/`.wasm` files exist. Errors thrown by the app's own code still
fail a lenient run.

## Worked example (the accepted design mockup)

The mockup that the design was accepted from lives in the session scratchpad and
deep-links its state through the same `#state=…&clip=…` fragment the app uses:

```bash
MOCK=/tmp/claude-1000/-home-user-polsess-separation/7141ae4e-414d-43bd-b989-aa62765019b2/scratchpad/mockup/road2_mockup.html
SHOTS=/tmp/claude-1000/-home-user-polsess-separation/7141ae4e-414d-43bd-b989-aa62765019b2/scratchpad/devcheck_shots

~/playwright_venv/bin/python webapp_ondevice/devcheck/check_page.py \
    "$MOCK" --hash "state=S6&clip=A" --dark \
    --shot "$SHOTS/s6_dark.png" \
    --js "({state: S.state, clip: S.clip, dark: matchMedia('(prefers-color-scheme: dark)').matches})"
```

```
=== target ===
file:///…/road2_mockup.html#state=S6&clip=A
viewport 390x844  scheme=dark

=== console (0) ===

=== page errors (0) ===

=== failed requests (0) ===

=== js ===
{
  "state": "S6",
  "clip": "A",
  "dark": true
}

=== shot ===
/…/devcheck_shots/s6_dark.png  (viewport)

=== result ===
OK: 0 console error(s), 0 page error(s), 0 failed request(s)
```

Exit 0, screenshot on disk, state confirmed by the probe. That is what a passing
check looks like; anything else, read the sections above the `result` line.
