/**
 * Float32 audio -> 16-bit PCM WAV blob.
 *
 * Textbook 44-byte RIFF header, the same encoder the January PoC used
 * (`mobile/webapp/app.js:810-874`), rewritten to take raw Float32Arrays instead
 * of an `AudioBuffer` so it can encode separator output directly.
 *
 * Samples outside [-1, 1] are **clipped**, not rescaled: the separators are
 * scale-free and can hand back peaks well above 1, but silently attenuating a
 * whole stream would misrepresent the separation the demo is showing. Callers
 * that want headroom should apply a gain of their own choosing and say so in
 * the UI.
 */

/**
 * @param {Float32Array|Float32Array[]} channels one Float32Array (mono) or an
 *        array of equal-length Float32Arrays (interleaved on the way out)
 * @param {number} sampleRate e.g. 8000 for separator output, 16000 for input
 * @returns {Blob} `audio/wav`
 */
export function encodeWav(channels, sampleRate) {
  const chans = channels instanceof Float32Array ? [channels] : channels;
  if (chans.length === 0) throw new Error('encodeWav needs at least one channel');
  const frames = chans[0].length;
  for (const ch of chans) {
    if (ch.length !== frames) throw new Error('encodeWav channels must be the same length');
  }

  const bytesPerSample = 2;
  const dataBytes = frames * chans.length * bytesPerSample;
  const view = new DataView(new ArrayBuffer(44 + dataBytes));

  writeAscii(view, 0, 'RIFF');
  view.setUint32(4, 36 + dataBytes, true);
  writeAscii(view, 8, 'WAVE');
  writeAscii(view, 12, 'fmt ');
  view.setUint32(16, 16, true); // PCM fmt chunk size
  view.setUint16(20, 1, true); // format = PCM integer
  view.setUint16(22, chans.length, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * chans.length * bytesPerSample, true); // byte rate
  view.setUint16(32, chans.length * bytesPerSample, true); // block align
  view.setUint16(34, 8 * bytesPerSample, true);
  writeAscii(view, 36, 'data');
  view.setUint32(40, dataBytes, true);

  let offset = 44;
  for (let i = 0; i < frames; i++) {
    for (let c = 0; c < chans.length; c++) {
      const s = Math.max(-1, Math.min(1, chans[c][i]));
      // Asymmetric scaling keeps -1.0 and +1.0 both representable in int16.
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
      offset += 2;
    }
  }
  return new Blob([view.buffer], { type: 'audio/wav' });
}

function writeAscii(view, offset, text) {
  for (let i = 0; i < text.length; i++) view.setUint8(offset + i, text.charCodeAt(i));
}
