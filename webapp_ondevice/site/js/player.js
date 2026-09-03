/**
 * Playback: one AudioContext, and "channel groups" whose members are mutually
 * exclusive but share a clock.
 *
 * The core interaction of the demo is switching Mix / Mówca 1 / Mówca 2 **while
 * playing** without losing your place. So a group starts every one of its
 * buffers at the same offset and routes each through its own GainNode; picking
 * a channel flips gains (0/1). No node is stopped and restarted, so the
 * switch is sample-accurate and free.
 *
 * Only one group plays at a time — starting one stops the other, which is what
 * you want when the page has a clip player, a region player and two full-stream
 * players on screen simultaneously.
 */

let ctx = null;
let current = null;

/** The page's single AudioContext, created on first use (a user gesture). */
export function audioContext() {
  if (!ctx) ctx = new (window.AudioContext || window.webkitAudioContext)();
  if (ctx.state === 'suspended') ctx.resume().catch(() => { /* no gesture yet */ });
  return ctx;
}

/**
 * Wrap raw samples in an AudioBuffer at their own rate; the graph resamples on
 * the way out, so 8 kHz separator output and 16 kHz mix play back side by side.
 * @param {Float32Array} samples
 * @param {number} sampleRate
 */
export function toAudioBuffer(samples, sampleRate) {
  const buffer = audioContext().createBuffer(1, Math.max(1, samples.length), sampleRate);
  buffer.copyToChannel(samples instanceof Float32Array ? samples : Float32Array.from(samples), 0);
  return buffer;
}

export class Group {
  /**
   * @param {Record<string, AudioBuffer>} buffers keyed by channel name
   * @param {{loop?: boolean, onEnd?: () => void, gain?: number}} [opts]
   *        `gain` applies to every channel equally — the separators are
   *        scale-free (`separate.js`), so a group needs a listening level, but
   *        it must be ONE level or the comparison the demo is making would be
   *        a comparison of gains.
   */
  constructor(buffers, opts = {}) {
    this.buffers = buffers;
    this.keys = Object.keys(buffers);
    this.duration = buffers[this.keys[0]].duration;
    this.channel = this.keys[0];
    this.loop = !!opts.loop;
    this.gain = opts.gain === undefined ? 1 : opts.gain;
    this.onEnd = opts.onEnd || null;
    this.playing = false;
    this.offset = 0;
    this.startedAt = 0;
    this.nodes = [];
    this.gains = {};
  }

  /** Playhead in seconds. */
  get time() {
    if (!this.playing) return this.offset;
    const t = this.offset + (audioContext().currentTime - this.startedAt);
    if (this.loop) return this.duration ? t % this.duration : 0;
    return Math.min(t, this.duration);
  }

  setChannel(key) {
    if (!this.buffers[key]) return;
    this.channel = key;
    for (const [name, node] of Object.entries(this.gains)) {
      node.gain.value = name === key ? this.gain : 0;
    }
  }

  setLoop(loop) {
    this.loop = loop;
    for (const node of this.nodes) node.loop = loop;
  }

  play(offsetS = this.offset) {
    const context = audioContext();
    this.stop(true);
    if (current && current !== this) current.pause();
    current = this;
    this.offset = Math.max(0, Math.min(offsetS, Math.max(0, this.duration - 0.001)));
    for (const key of this.keys) {
      const source = context.createBufferSource();
      source.buffer = this.buffers[key];
      source.loop = this.loop;
      const gain = context.createGain();
      gain.gain.value = key === this.channel ? this.gain : 0;
      source.connect(gain).connect(context.destination);
      source.start(0, this.offset);
      this.nodes.push(source);
      this.gains[key] = gain;
    }
    // One `ended` event is enough — every member has the same length.
    this.nodes[0].onended = () => {
      if (!this.playing) return; // stopped by us, not by reaching the end
      this.playing = false;
      this.offset = 0;
      if (this.onEnd) this.onEnd();
    };
    this.startedAt = context.currentTime;
    this.playing = true;
  }

  pause() {
    if (!this.playing) return;
    const at = this.time;
    this.stop(true);
    this.offset = at >= this.duration - 0.005 ? 0 : at;
  }

  /** @param {boolean} [silent] suppress the `onEnd` callback */
  stop(silent) {
    for (const node of this.nodes) {
      node.onended = null;
      try { node.stop(); } catch (err) { /* never started */ }
      node.disconnect();
    }
    this.nodes = [];
    this.gains = {};
    const wasPlaying = this.playing;
    this.playing = false;
    if (wasPlaying && !silent && this.onEnd) this.onEnd();
  }

  seek(seconds) {
    if (this.playing) this.play(seconds);
    else this.offset = Math.max(0, Math.min(seconds, this.duration));
  }

  toggle(offsetS) {
    if (this.playing) this.pause();
    else this.play(offsetS === undefined ? this.offset : offsetS);
  }
}

/** Stop whatever is playing (input change, new run, cancel). */
export function stopAll() {
  if (current) current.stop(true);
  current = null;
}

/** The group that owns playback right now, if any. */
export function currentGroup() {
  return current;
}
