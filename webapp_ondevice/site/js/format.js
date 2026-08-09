/**
 * Polish number/time formatting. Port of the mockup's helpers (which are the
 * Road 1 `webapp/static/format.js` conventions): decimal comma everywhere,
 * `m:ss,t` clocks, correct plural forms.
 */

/** `12.34 -> "12,3"` — decimal comma, fixed decimals (default 1). */
export function num(value, decimals = 1) {
  return Number(value).toFixed(decimals).replace('.', ',');
}

/** Seconds -> `m:ss,t` (tenths), e.g. `0:38,4`. */
export function clock(seconds) {
  const x = Math.max(0, seconds);
  const m = Math.floor(x / 60);
  const rest = x - m * 60;
  const whole = Math.floor(rest);
  const tenth = Math.floor((rest - whole) * 10 + 1e-6);
  return `${m}:${String(whole).padStart(2, '0')},${tenth}`;
}

/** Seconds -> `m:ss`, rounded — the timeline ruler's format. */
export function clockShort(seconds) {
  const x = Math.max(0, Math.round(seconds));
  return `${Math.floor(x / 60)}:${String(x % 60).padStart(2, '0')}`;
}

/**
 * Polish plural: 1 region / 2-4 regiony / 5+ regionów.
 * @param {number} n
 * @param {string} one singular
 * @param {string} few 2-4 form
 * @param {string} many 5+ / genitive form
 */
export function plural(n, one, few, many) {
  const a = Math.abs(n);
  const m10 = a % 10;
  const m100 = a % 100;
  if (a === 1) return one;
  if (m10 >= 2 && m10 <= 4 && (m100 < 12 || m100 > 14)) return few;
  return many;
}

/** Bytes -> `12,3 MB` (MiB, the unit `build/NOTES.md` quotes model sizes in). */
export function mb(bytes) {
  return `${num(bytes / 1048576, 1)} MB`;
}
