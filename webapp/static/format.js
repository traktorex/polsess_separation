/* Polish-locale formatting. Decimal comma throughout; clock as m:ss,d. */

/** A number with `dec` decimals and a Polish decimal comma. */
export function num(value, dec = 1) {
  if (value === null || value === undefined || !isFinite(value)) return "—";
  return Number(value).toFixed(dec).replace(".", ",");
}

/** Clock with tenths: 92.34 -> "1:32,3". */
export function clock(seconds) {
  if (seconds === null || seconds === undefined || !isFinite(seconds)) return "—";
  const s = Math.max(0, seconds);
  const m = Math.floor(s / 60);
  const rest = s - m * 60;
  const whole = Math.floor(rest);
  const tenth = Math.floor((rest - whole) * 10);
  return `${m}:${String(whole).padStart(2, "0")},${tenth}`;
}

/** Clock without tenths: 92.34 -> "1:32". */
export function clockShort(seconds) {
  if (seconds === null || seconds === undefined || !isFinite(seconds)) return "—";
  const s = Math.max(0, Math.round(seconds));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

/** Elapsed time for the progress card: "28 s" under a minute, else "1:05". */
export function elapsed(seconds) {
  if (seconds === null || seconds === undefined || !isFinite(seconds)) return "";
  return seconds < 60 ? `${Math.round(seconds)} s` : clockShort(seconds);
}

/** Seconds with a unit: 4.83 -> "4,8 s". */
export function secs(value, dec = 1) {
  return value === null || value === undefined || !isFinite(value)
    ? "—"
    : `${num(value, dec)} s`;
}

/** Coarse, bucketed ETA text (design §5.2: never a countdown). */
export function eta(seconds) {
  if (seconds === null || seconds === undefined || !isFinite(seconds) || seconds <= 0) {
    return "";
  }
  if (seconds < 12) return "≈ jeszcze chwila";
  if (seconds < 25) return "≈ jeszcze około 20 sekund";
  if (seconds < 45) return "≈ jeszcze około pół minuty";
  if (seconds < 95) return "≈ jeszcze około minuty";
  if (seconds < 150) return "≈ jeszcze około 2 minut";
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `≈ jeszcze około ${minutes} minut`;
  const hours = Math.round(seconds / 360) / 10;
  return `≈ jeszcze około ${num(hours, 1)} godziny`;
}

/** Polish plural for a count: (2, "zadanie", "zadania", "zadań"). */
export function plural(n, one, few, many) {
  const abs = Math.abs(n);
  if (abs === 1) return one;
  const mod10 = abs % 10;
  const mod100 = abs % 100;
  if (mod10 >= 2 && mod10 <= 4 && (mod100 < 12 || mod100 > 14)) return few;
  return many;
}

/** ISO 8601 -> local "28.07, 14:03"; the raw string if it will not parse. */
export function when(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  if (isNaN(d.getTime())) return String(iso);
  const pad = (x) => String(x).padStart(2, "0");
  return `${pad(d.getDate())}.${pad(d.getMonth() + 1)}, ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

/** Polish status label for a job status. */
export function statusLabel(status) {
  return {
    queued: "w kolejce",
    running: "przetwarzanie",
    done: "gotowe",
    failed: "błąd",
  }[status] || String(status || "");
}
