// Lightweight proxy to the Modal backend's /health endpoint.
// Wakes a cold/sleeping backend and reports whether the model is loaded.
// The backend's /health route is public (no auth dependency, no rate
// limiter in main.py), so no token is minted here — and the Modal URL
// itself never reaches the browser.

const FETCH_TIMEOUT_MS = 8000; // stay well under the function execution budget

function json(obj: unknown, status: number): Response {
  return new Response(JSON.stringify(obj), {
    status,
    headers: { "Content-Type": "application/json", "Cache-Control": "no-store" },
  });
}

export default async (req: Request): Promise<Response> => {
  if (req.method !== "GET") {
    return new Response("Method Not Allowed", { status: 405 });
  }

  const MODAL_URL = process.env.MODAL_BACKEND_URL;
  if (!MODAL_URL) {
    console.error("Missing MODAL_BACKEND_URL in Netlify environment.");
    return json({ detail: "Server configuration error" }, 500);
  }

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);
  try {
    // While the container cold-boots this fetch simply waits — aborting
    // after 8s is fine: the boot was already triggered, and the client
    // keeps polling.
    const upstream = await fetch(`${MODAL_URL}/health`, {
      method: "GET",
      headers: { Accept: "application/json" },
      signal: controller.signal,
    });
    const body = await upstream.text();
    return new Response(body, {
      status: upstream.status, // 200 when ready, 503 while loading
      headers: { "Content-Type": "application/json", "Cache-Control": "no-store" },
    });
  } catch {
    // Most common cause: backend cold-booting and didn't answer within the
    // window. The client interprets this as "waking", not "down".
    return json({ detail: "Backend unreachable — likely cold-starting." }, 502);
  } finally {
    clearTimeout(timer);
  }
};