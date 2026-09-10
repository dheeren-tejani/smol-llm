/* SSE transport layer + backend health/wake helpers.
   POSTs to the inference endpoint, reads the chunk stream with
   response.body.getReader() + TextDecoder, parses SSE frames with line
   buffering across chunk boundaries, and is abortable via the passed
   AbortSignal.

   Wire contract (matches GenerateRequest in the FastAPI backend):
     request  — flat body: { messages, max_tokens, temperature, top_p,
                             top_k, repetition_penalty }
     response — text/event-stream frames:
                 data: {"token": "..."}                        per token
                 data: {"done": true, "tokens_generated": N}   terminator
                 data: {"error": "..."}                        mid-stream failure
   Non-2xx responses carry JSON bodies with a "detail" field (rate limits,
   capacity, validation) which is surfaced in the client error banner.

   Cold-start handling: 502/504 (Netlify proxy died while Modal was
   booting), 503 (model loading / at capacity), and the proxy's own 500
   are classified as BackendWakingError — the hook then polls /api/health
   (free: no auth, no rate limit) until the model is loaded and re-sends. */

import type { StreamPayload } from '../types/chat';

const API_ENDPOINT: string = import.meta.env.VITE_API_ENDPOINT ?? '/api/chat';
const HEALTH_ENDPOINT: string = import.meta.env.VITE_HEALTH_ENDPOINT ?? '/api/health';
const CHUNK_TIMEOUT_MS = 25_000;

/* Backend validation bounds (GenerateRequest in main.py). Keep in sync
   with the backend — the wire layer clamps to these as a final safety
   net on top of the slider ranges, so an out-of-bounds value can never
   trigger a 422 even if it somehow reaches this layer. */
const BACKEND_BOUNDS = {
  max_tokens: { min: 1, max: 1024 },
  temperature: { min: 0.01, max: 5 },
  top_p: { min: 0, max: 1 },
  top_k: { min: 1, max: 200 },
  repetition_penalty: { min: 1, max: 3 },
} as const;

export class HttpError extends Error {
  status: number;
  detail: string;
  constructor(status: number, detail = '') {
    super('HTTP ' + status);
    this.name = 'HttpError';
    this.status = status;
    this.detail = detail;
  }
}

export class StreamTimeoutError extends Error {
  constructor() {
    super('stream stalled');
    this.name = 'StreamTimeoutError';
  }
}

/** Mid-stream failure reported by the backend as data: {"error": "…"}. */
export class StreamError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'StreamError';
  }
}

/** Backend is cold-booting / loading / at capacity — worth polling health
    and re-sending. Carries the upstream detail for the give-up message. */
export class BackendWakingError extends Error {
  detail: string;
  constructor(reason: string, detail = '') {
    super(reason);
    this.name = 'BackendWakingError';
    this.detail = detail;
  }
}

class UnexpectedResponseError extends Error {
  constructor() {
    super('unexpected response');
    this.name = 'UnexpectedResponseError';
  }
}

export function isWakeRetryable(err: unknown): boolean {
  return err instanceof BackendWakingError;
}

interface ResponseLike {
  ok: boolean;
  status: number;
  body: ReadableStream<Uint8Array>;
}

/** Map the app-internal StreamPayload onto the backend's flat schema.
    This is the ONLY place where the internal `max_output_tokens` name is
    translated to the backend's `max_tokens`. */
function toWireBody(payload: StreamPayload): Record<string, unknown> {
  const p = payload.parameters;
  const clamp = (v: number, b: { min: number; max: number }): number =>
    parseFloat(Math.min(b.max, Math.max(b.min, v)).toFixed(4));
  return {
    messages: payload.messages,
    max_tokens: Math.round(clamp(p.max_output_tokens, BACKEND_BOUNDS.max_tokens)),
    temperature: clamp(p.temperature, BACKEND_BOUNDS.temperature),
    top_p: clamp(p.top_p, BACKEND_BOUNDS.top_p),
    top_k: Math.round(clamp(p.top_k, BACKEND_BOUNDS.top_k)),
    repetition_penalty: clamp(p.repetition_penalty, BACKEND_BOUNDS.repetition_penalty),
    // `system` and `range_epsilon` are omitted — the backend applies its
    // own defaults for both.
  };
}

/** Pull a human-readable message out of a FastAPI error body. */
function extractDetail(text: string): string {
  try {
    const parsed = JSON.parse(text) as { detail?: unknown; error?: unknown };
    const d = parsed.detail;
    if (typeof d === 'string' && d.trim()) return d;
    if (Array.isArray(d) && d.length > 0) {
      // FastAPI 422 validation errors: [{ loc, msg, type }, …]
      const first = d[0] as { msg?: unknown } | null;
      if (first && typeof first.msg === 'string' && first.msg) return first.msg;
    }
    if (typeof parsed.error === 'string' && parsed.error.trim()) return parsed.error;
  } catch { /* body wasn't JSON */ }
  return text.trim().slice(0, 200);
}

async function streamChat(payload: StreamPayload, signal: AbortSignal): Promise<ResponseLike> {
  const res = await fetch(API_ENDPOINT, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
    body: JSON.stringify(toWireBody(payload)),
    signal,
  });
  if (res.ok && res.body) {
    const type = res.headers.get('content-type') ?? '';
    if (type.includes('text/html')) {
      // Dev-server SPA fallback: the endpoint doesn't exist on this origin.
      throw new UnexpectedResponseError();
    }
    return res as unknown as ResponseLike;
  }
  let detail = '';
  try { detail = extractDetail(await res.text()); } catch { /* no body */ }

  // Wake classification:
  //  502/504 — the Netlify function died while Modal was cold-booting
  //            (sync functions have a short execution budget).
  //  503     — "Model is still loading." or "Server is at capacity".
  //  500     — the proxy's own catch ("Internal Server Proxy Error"),
  //            typically its upstream fetch failing during a cold boot.
  if (res.status === 502 || res.status === 504 || res.status === 503 ||
      (res.status === 500 && /proxy error/i.test(detail))) {
    throw new BackendWakingError('status ' + res.status, detail);
  }
  throw new HttpError(res.status, detail);
}

export function describeError(err: unknown): string {
  if (err instanceof BackendWakingError) {
    return err.detail || 'The backend is waking up from idle — this can take up to a minute.';
  }
  if (err instanceof HttpError) return err.detail || ('The server responded with ' + err.status + '.');
  if (err instanceof StreamError) return err.message;
  if (err instanceof StreamTimeoutError) return 'No tokens received for 25s — the stream stalled.';
  if (err instanceof UnexpectedResponseError) return 'The endpoint did not return an event stream.';
  if ((err as { name?: string } | null)?.name === 'AbortError') return 'Request cancelled.';
  if (err instanceof TypeError) return 'Network unreachable while contacting the endpoint.';
  return 'Unexpected error while streaming the response.';
}

/* ── health / wake helpers ──────────────────────────────────────────── */

export interface BackendHealth {
  reachable: boolean;
  modelLoaded: boolean;
}

/** One probe of /api/health. Never throws — an unreachable backend is a
    valid observation (cold boot), not an exceptional condition. */
export async function checkBackendHealth(signal?: AbortSignal): Promise<BackendHealth> {
  try {
    const res = await fetch(HEALTH_ENDPOINT, { cache: 'no-store', signal });
    const text = await res.text();
    let parsed: unknown = null;
    try { parsed = JSON.parse(text); } catch { /* not JSON */ }
    if (parsed && typeof parsed === 'object' &&
        typeof (parsed as { model_loaded?: unknown }).model_loaded === 'boolean') {
      // FastAPI returns 200 when ready, 503 while loading — the body is a
      // HealthResponse either way, so parse regardless of status.
      return { reachable: true, modelLoaded: (parsed as { model_loaded: boolean }).model_loaded };
    }
    return { reachable: false, modelLoaded: false };
  } catch {
    return { reachable: false, modelLoaded: false };
  }
}

/** Poll /api/health until the model reports loaded, the deadline passes,
    or the signal aborts. Health polls are free on the backend (no auth,
    no rate limiter), so this is the cheap way to wait out a cold boot. */
export async function waitForBackendReady(
  signal: AbortSignal,
  opts: { pollMs?: number; maxWaitMs?: number } = {},
): Promise<boolean> {
  const pollMs = opts.pollMs ?? 3000;
  const maxWaitMs = opts.maxWaitMs ?? 60_000;
  const deadline = Date.now() + maxWaitMs;
  await new Promise((r) => setTimeout(r, 500)); // brief pause before the first poll
  for (;;) {
    if (signal.aborted) return false;
    const h = await checkBackendHealth(signal);
    if (h.modelLoaded) return true;
    if (Date.now() >= deadline) return false;
    await new Promise((r) => setTimeout(r, pollMs));
  }
}

/* ── stream consumer ────────────────────────────────────────────────── */

interface StreamHandlers { onToken(token: string): void }

export async function consumeStream(
  payload: StreamPayload,
  signal: AbortSignal,
  handlers: StreamHandlers,
): Promise<void> {
  const res = await streamChat(payload, signal);
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let timer: ReturnType<typeof setTimeout> | undefined;

  try {
    for (;;) {
      let read: ReadableStreamReadResult<Uint8Array>;
      try {
        // Watchdog: a silent stream is a dead stream.
        const timeout = new Promise<never>((_, reject) => {
          timer = setTimeout(() => reject(new StreamTimeoutError()), CHUNK_TIMEOUT_MS);
        });
        read = await Promise.race([reader.read(), timeout]);
      } finally {
        if (timer !== undefined) clearTimeout(timer);
      }
      if (read.done) return;

      buffer += decoder.decode(read.value, { stream: true });
      // SSE frames are newline-delimited; buffer partial lines across chunks.
      let nl: number;
      while ((nl = buffer.indexOf('\n')) >= 0) {
        const line = buffer.slice(0, nl).replace(/\r$/, '');
        buffer = buffer.slice(nl + 1);
        if (!line.startsWith('data:')) continue;
        const data = line.slice(5).trim();
        if (!data || data === '[DONE]') continue;
        try {
          const evt = JSON.parse(data) as { token?: unknown; error?: unknown };
          if (typeof evt.token === 'string') {
            handlers.onToken(evt.token);
          }
          // data: {"done": true, …} — terminator; the stream ends right
          // after it, so nothing to do here.
          else if (typeof evt.error === 'string' && evt.error) {
            throw new StreamError(evt.error);
          }
        } catch (e) {
          if (e instanceof StreamError) throw e; // propagate backend errors
          // otherwise: tolerate a malformed frame
        }
      }
    }
  } finally {
    try { reader.cancel(); } catch { /* already released */ }
  }
}