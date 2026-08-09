/**
 * Audio decode + rate conversion. No dependencies: Web Audio for decoding, a
 * hand-rolled FIR for decimation.
 *
 * Two jobs, matching the two rates the demo runs at:
 *
 * * **16 kHz mono** — what pyannote-segmentation-3.0 needs. A file is decoded
 *   inside an `OfflineAudioContext` created at 16 kHz, so the browser's own
 *   decoder does any rate conversion during decode.
 * * **8 kHz mono** — what the separators need. Routed regions are decimated
 *   2:1 by an explicit windowed-sinc FIR, NOT by the Web Audio graph. Why:
 *
 *   > Measured in headless Chromium 149 (`devcheck/check_page.py --js`):
 *   > rendering a 16 kHz `AudioBuffer` through an 8 kHz `OfflineAudioContext`
 *   > returns **exactly** `x[2k]` — max |out[k] - x[2k]| = 0.0 over a 2 s tone.
 *   > `AudioBufferSourceNode` rate conversion is bare interpolation with **no
 *   > anti-alias filter**: a 6 kHz tone comes back at full amplitude, folded
 *   > down to 2 kHz. Feeding that to the separator would put mirrored fricative
 *   > energy inside its band. (The *decode* path is fine — Chromium's file
 *   > decoder does filter, measured -57 dB on the same tone — which is why
 *   > `decodeToMono16k` may keep using it.)
 *
 *   So the decimator here is 127 explicit taps: auditable, deterministic,
 *   identical in every browser, and about 4 MMAC per 4 s region.
 *
 * **Parity note.** No browser resampler is torchaudio or soxr. The FIR below is
 * a Hamming-windowed sinc (passband to ~3.6 kHz, stopband from ~4 kHz at about
 * -53 dB) where the reference used soxr VHQ, so the two are close but not
 * identical. The parity fixtures avoid the question entirely rather than
 * tolerate it: the OSD vectors ship as 16 kHz WAVs (decoded, never resampled)
 * and the separator vector ships already decimated by
 * `reference/make_sep_vector.py`.
 */

/** Rate the OSD model runs at. (The separators' 8 kHz lives in `separate.js`,
 *  which owns that model's contract — one name, one home.) */
export const OSD_SAMPLE_RATE = 16000;

/**
 * Decode an audio file to mono 16 kHz.
 *
 * Multi-channel input is averaged down (not just channel 0) so a stereo
 * recording with one speaker per channel still reads as a mixture.
 *
 * NB `decodeAudioData` **detaches** the ArrayBuffer it is given; pass a copy if
 * you still need it.
 *
 * @param {ArrayBuffer|Blob|File} source encoded audio (anything the browser
 *        can decode: WAV, MP3, M4A, OGG, FLAC...)
 * @returns {Promise<Float32Array>} mono samples at 16 kHz
 */
export async function decodeToMono16k(source) {
  const bytes = source instanceof ArrayBuffer ? source : await source.arrayBuffer();
  // Length 1 — this context only ever decodes; nothing is rendered through it.
  const ctx = new OfflineAudioContext(1, 1, OSD_SAMPLE_RATE);
  const decoded = await ctx.decodeAudioData(bytes);
  return downmixToMono(decoded);
}

/**
 * Average an `AudioBuffer`'s channels into one Float32Array.
 * @param {AudioBuffer} buffer
 * @returns {Float32Array}
 */
export function downmixToMono(buffer) {
  const n = buffer.length;
  const mono = new Float32Array(n);
  for (let c = 0; c < buffer.numberOfChannels; c++) {
    const ch = buffer.getChannelData(c);
    for (let i = 0; i < n; i++) mono[i] += ch[i];
  }
  if (buffer.numberOfChannels > 1) {
    const scale = 1 / buffer.numberOfChannels;
    for (let i = 0; i < n; i++) mono[i] *= scale;
  }
  return mono;
}

/** Anti-alias FIR length. Odd, so the filter has an integer group delay. */
const DECIMATE_TAPS = 127;
/** Cutoff, below the 4 kHz Nyquist of the target rate by half a transition band. */
const DECIMATE_CUTOFF_HZ = 3800;

/**
 * Hamming-windowed sinc low-pass, normalised to unity DC gain.
 * @param {number} taps odd
 * @param {number} cutoffHz
 * @param {number} sampleRate
 * @returns {Float32Array}
 */
function lowpassKernel(taps, cutoffHz, sampleRate) {
  const h = new Float32Array(taps);
  const mid = (taps - 1) / 2;
  const wc = (2 * Math.PI * cutoffHz) / sampleRate;
  let sum = 0;
  for (let i = 0; i < taps; i++) {
    const k = i - mid;
    const sinc = k === 0 ? wc / Math.PI : Math.sin(wc * k) / (Math.PI * k);
    const window = 0.54 - 0.46 * Math.cos((2 * Math.PI * i) / (taps - 1));
    h[i] = sinc * window;
    sum += h[i];
  }
  for (let i = 0; i < taps; i++) h[i] /= sum;
  return h;
}

const DECIMATE_KERNEL = lowpassKernel(DECIMATE_TAPS, DECIMATE_CUTOFF_HZ, OSD_SAMPLE_RATE);

/**
 * Decimate a 16 kHz region to the separators' 8 kHz: low-pass, then keep every
 * second sample.
 *
 * The filter is linear phase and its group delay is compensated exactly, so
 * output sample k lines up with input sample 2k — region timings stay valid
 * without an offset correction. Samples off either end are treated as zero;
 * routed regions are context-padded, so the edges are not where the speech is.
 *
 * @param {Float32Array} audio16k
 * @returns {Float32Array} `round(n / 2)` samples at 8 kHz (synchronous — no
 *          Web Audio graph is involved)
 */
export function decimateTo8k(audio16k) {
  const h = DECIMATE_KERNEL;
  const mid = (h.length - 1) / 2;
  const outLength = Math.round(audio16k.length / 2);
  const out = new Float32Array(outLength);
  for (let k = 0; k < outLength; k++) {
    const centre = 2 * k;
    let acc = 0;
    for (let j = 0; j < h.length; j++) {
      const idx = centre + mid - j;
      if (idx >= 0 && idx < audio16k.length) acc += h[j] * audio16k[idx];
    }
    out[k] = acc;
  }
  return out;
}
