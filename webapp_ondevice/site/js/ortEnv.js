/**
 * ONNX Runtime Web environment + session factory.
 *
 * One place decides how the runtime is configured, because every one of these
 * settings has a documented reason to be exactly what it is
 * (`webapp_ondevice/build/NOTES.md` §3):
 *
 * * **Vendored, never a CDN.** `ort.env.wasm.wasmPaths` points at
 *   `site/vendor/ort/` (onnxruntime-web 1.23.2, the version that validated
 *   every shipping model), so the page makes zero third-party requests.
 * * **`numThreads = 1`.** GitHub Pages cannot send COOP/COEP, so
 *   `SharedArrayBuffer` — and therefore multi-threaded WASM — is unavailable
 *   there. Pinning 1 makes the dev box behave like the deploy target instead of
 *   flattering it. (There is only one CPU `.wasm` in this version line;
 *   `ort-wasm-simd-threaded.wasm` serves both cases. SIMD is assumed, as it has
 *   been in every browser since ~2021, and 1.23 has no `env.wasm.simd` knob.)
 * * **`graphOptimizationLevel: 'extended'` for the pyannote fp16 model.** ORT's
 *   default (`all`) *segfaults the process* on that graph on the Python CPU EP
 *   (NOTES.md §2). The WASM build most likely does not contain the offending
 *   x86 layout transformer — `devcheck` proves the page loads either way — but
 *   the flag costs nothing and the failure mode is a dead tab.
 *
 * Everything runs on the main thread (`env.wasm.proxy` is left off). The UI
 * wave may want `proxy = true` to keep the page responsive during a separator
 * call; that is a UI decision, so it is not made here.
 */
import * as ort from '../vendor/ort/ort.wasm.bundle.min.mjs';

export { ort };

/** Absolute URL of the vendored runtime directory. */
export const ORT_VENDOR_URL = new URL('../vendor/ort/', import.meta.url).href;
/** Absolute URL of the model directory. */
export const MODELS_URL = new URL('../models/', import.meta.url).href;

let configured = false;

/**
 * Apply the global ORT-Web settings. Idempotent; called by `createSession`.
 * @returns {typeof ort} the configured `ort` namespace
 */
export function configureOrt() {
  if (!configured) {
    ort.env.wasm.wasmPaths = ORT_VENDOR_URL;
    ort.env.wasm.numThreads = 1;
    // ORT-Web pipes its native log to `console.error` REGARDLESS of severity,
    // so a benign W-level line makes the page look broken and fails every
    // `devcheck/check_page.py` run on its console-error count. The one this
    // machine emits at WASM init is
    //   [W:onnxruntime:Default, cpuid_info.cc:91] Unknown CPU vendor.
    // (a WSL2 cpuinfo quirk, no effect on execution). Dropping to 'error' hides
    // ORT's warnings and nothing else — ORT errors, and every console message
    // the app itself writes, still come through.
    ort.env.logLevel = 'error';
    configured = true;
  }
  return ort;
}

/**
 * Create an inference session on the WASM execution provider.
 *
 * @param {string|Uint8Array|ArrayBuffer} url absolute or page-relative URL of
 *        the `.onnx` file, or its bytes — the app fetches models itself to show
 *        download progress, and ORT accepts either form here.
 * @param {{graphOptimizationLevel?: 'disabled'|'basic'|'extended'|'all'}} [opts]
 *        `graphOptimizationLevel` defaults to `'extended'` — the level the
 *        pyannote fp16 model requires and the separators are indifferent to.
 * @returns {Promise<ort.InferenceSession>}
 */
export async function createSession(url, opts = {}) {
  configureOrt();
  return ort.InferenceSession.create(url, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: opts.graphOptimizationLevel ?? 'extended',
  });
}
