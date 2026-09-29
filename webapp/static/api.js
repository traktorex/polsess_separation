/* Fetch helpers for webapp/API.md. Errors carry the server's own message —
   the pipeline is loud on failure and so is this UI (SCOPE §4.1). */

export class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function readError(response) {
  try {
    const body = await response.json();
    if (body && typeof body.detail === "string") return body.detail;
    if (body && body.detail) return JSON.stringify(body.detail);
  } catch (err) {
    /* not JSON — fall through to the status line */
  }
  return `${response.status} ${response.statusText}`;
}

export async function getJSON(url, options) {
  const request = Object.assign({ headers: { Accept: "application/json" } }, options || {});
  const response = await fetch(url, request);
  if (!response.ok) throw new ApiError(await readError(response), response.status);
  return response.json();
}

/** POST /api/jobs — either a File or {source_example}. Returns the job id. */
export async function createJob({ file = null, sourceExample = null }) {
  const form = new FormData();
  if (file) form.append("file", file, file.name);
  if (sourceExample) form.append("source_example", sourceExample);
  const response = await fetch("/api/jobs", { method: "POST", body: form });
  if (!response.ok) throw new ApiError(await readError(response), response.status);
  const body = await response.json();
  if (!body || !body.job_id) throw new ApiError("Serwer nie zwrócił identyfikatora zadania.", 500);
  return body.job_id;
}

/** DELETE /api/jobs — drop finished jobs and their files.
 *  Returns {removed, skipped}; skipped counts jobs still queued or running. */
export async function deleteJobs(url = "/api/jobs") {
  const body = await getJSON(url, { method: "DELETE" });
  return {
    removed: Number((body && body.removed) || 0),
    skipped: Number((body && body.skipped) || 0),
  };
}

/** Does this file URL exist? One-byte ranged GET — the file routes are
 *  GET-only (HEAD answers 405), so a Range request is the cheap probe. */
export async function fileExists(url) {
  try {
    const response = await fetch(url, { headers: { Range: "bytes=0-0" } });
    return response.ok || response.status === 206;
  } catch (err) {
    return false;
  }
}
