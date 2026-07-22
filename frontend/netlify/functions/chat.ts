import { Context } from "@netlify/functions";
import * as crypto from "crypto";

// Helper to generate the exact token format expected by auth.py
function generateAuthToken(secretBase64: string, keyValue: string): string {
  // Decode the 32-byte AES key from base64
  const key = Buffer.from(secretBase64, "base64");
  if (key.length !== 32) {
    throw new Error("Invalid AUTH_SECRET_KEY: must decode to 32 bytes.");
  }

  // 12-byte IV for AES-GCM
  const iv = crypto.randomBytes(12);
  
  // The payload expected by your backend
  const payload = JSON.stringify({ key: keyValue, ts: Date.now() });

  // Encrypt the payload
  const cipher = crypto.createCipheriv("aes-256-gcm", key, iv);
  const ciphertext = Buffer.concat([cipher.update(payload, "utf8"), cipher.final()]);
  
  // 16-byte auth tag
  const authTag = cipher.getAuthTag();

  // Python's cryptography library decrypts (iv, ciphertext + tag)
  const combinedCt = Buffer.concat([ciphertext, authTag]);

  return `${iv.toString("base64")}.${combinedCt.toString("base64")}`;
}

export default async (req: Request, context: Context) => {
  // Only accept POST requests
  if (req.method !== "POST") {
    return new Response("Method Not Allowed", { status: 405 });
  }

  try {
    // Access environment variables stored securely in Netlify
    const MODAL_URL = process.env.MODAL_BACKEND_URL; // e.g., https://your-workspace--smol-lm-backend-fastapi-app.modal.run
    const AUTH_SECRET = process.env.AUTH_SECRET_KEY;
    const AUTH_VALUE = process.env.AUTH_KEY_VALUE;

    if (!MODAL_URL || !AUTH_SECRET || !AUTH_VALUE) {
      console.error("Missing required environment variables in Netlify.");
      return new Response("Server configuration error", { status: 500 });
    }

    // Generate the time-stamped authentication token
    const token = generateAuthToken(AUTH_SECRET, AUTH_VALUE);

    // Extract the client's real IP address
    const clientIp = context.ip || req.headers.get("x-nf-client-connection-ip") || "unknown";

    // Forward the JSON body exactly as received from the React frontend
    const requestBody = await req.text();

    // Call your Modal backend's streaming endpoint
    const modalResponse = await fetch(`${MODAL_URL}/generate/stream`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Auth-Token": token,
        "X-Forwarded-For": clientIp, // Feeds your SlidingWindowRateLimiter
      },
      body: requestBody,
    });

    // If the backend rejects it (e.g., rate limit hit, model loading), pass the status forward
    if (!modalResponse.ok) {
      const errorText = await modalResponse.text();
      return new Response(errorText, { 
        status: modalResponse.status,
        headers: { "Content-Type": "application/json" }
      });
    }

    // Stream the response directly back to the React client
    return new Response(modalResponse.body, {
      status: 200,
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
      },
    });

  } catch (error) {
    console.error("Proxy error:", error);
    return new Response(JSON.stringify({ error: "Internal Server Proxy Error" }), { 
      status: 500,
      headers: { "Content-Type": "application/json" }
    });
  }
};