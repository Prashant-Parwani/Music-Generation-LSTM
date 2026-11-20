import { API_BASE } from "./config.js";

async function request(path, options) {
  let res;
  try {
    res = await fetch(`${API_BASE}${path}`, options);
  } catch {
    throw new Error("Cannot reach the music server. Is it running (or still waking up)?");
  }
  if (!res.ok) {
    let detail = `Server error (${res.status})`;
    try {
      const body = await res.json();
      if (typeof body.detail === "string") detail = body.detail;
      else if (Array.isArray(body.detail)) detail = body.detail.map((d) => d.msg).join("; ");
    } catch { /* ignore */ }
    throw new Error(detail);
  }
  return res.json();
}

export const getHealth = () => request("/api/health");
export const getConfig = () => request("/api/config");

/** Poll /api/health until the server answers (free hosts go to sleep and need a minute to wake up). */
export async function waitForServer(onTick, signal, maxSeconds = 180) {
  const start = Date.now();
  for (;;) {
    try {
      return await getHealth();
    } catch (e) {
      const waited = Math.round((Date.now() - start) / 1000);
      if (signal?.aborted || waited > maxSeconds) throw e;
      onTick?.(waited);
      await new Promise((r) => setTimeout(r, 2500));
    }
  }
}

/**
 * Start a generation job and poll until it finishes.
 * onUpdate({status, progress, position}) is called on every poll.
 */
export async function generate(settings, onUpdate, signal) {
  const { job_id } = await request("/api/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(settings),
    signal,
  });
  const started = Date.now();
  for (;;) {
    if (signal?.aborted) throw new DOMException("Cancelled", "AbortError");
    if (Date.now() - started > 15 * 60 * 1000) throw new Error("Generation took too long. Try a shorter piece.");
    const job = await request(`/api/jobs/${job_id}`, { signal });
    onUpdate?.(job);
    if (job.status === "done") return job.result;
    if (job.status === "error") throw new Error(job.error || "Generation failed.");
    await new Promise((r) => setTimeout(r, 600));
  }
}
