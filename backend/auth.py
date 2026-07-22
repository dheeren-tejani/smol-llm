"""
auth.py — Shared-secret AES-256-GCM request authentication.

Protocol: frontend encrypts {"key": AUTH_KEY_VALUE, "ts": Date.now()} with a
shared AES-256-GCM key via WebCrypto, sends it as:
    X-Auth-Token: <iv_b64>.<ciphertext_b64>
Backend decrypts with the same key, checks the embedded key matches, and
checks the timestamp is recent (blocks replay of a captured token past its
TTL).

Ceiling on what this protects: the key ships inside the frontend's JS bundle
by necessity (browsers can't keep secrets from their own user), so this
stops blind/automated hits, not a targeted person reading your bundle.
"""

import base64
import json
import time
from typing import Optional

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from fastapi import Header, HTTPException


def _load_key(b64_key: str) -> bytes:
    key = base64.b64decode(b64_key)
    if len(key) != 32:
        raise ValueError("AUTH_SECRET_KEY must decode to exactly 32 bytes (AES-256-GCM).")
    return key


def decrypt_token(token: str, key: bytes) -> dict:
    """Token format: base64(iv) + '.' + base64(ciphertext+tag) — matches
    WebCrypto AES-GCM's output (12-byte IV, 16-byte tag appended to ciphertext)."""
    try:
        iv_b64, ct_b64 = token.split(".", 1)
        iv = base64.b64decode(iv_b64)
        ct = base64.b64decode(ct_b64)
        plaintext = AESGCM(key).decrypt(iv, ct, None)
        return json.loads(plaintext.decode("utf-8"))
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or malformed auth token.")


def make_verify_auth_token(
    secret_key_b64: str,
    expected_auth_key: str,
    max_age_seconds: float = 30.0,
    enabled: bool = True,
):
    key = _load_key(secret_key_b64) if enabled else None

    async def verify(x_auth_token: Optional[str] = Header(default=None)):
        if not enabled:
            return

        if not x_auth_token:
            raise HTTPException(status_code=401, detail="Missing X-Auth-Token header.")

        payload = decrypt_token(x_auth_token, key)
        auth_key = payload.get("key")
        ts = payload.get("ts")

        if auth_key != expected_auth_key:
            raise HTTPException(status_code=401, detail="Invalid auth token.")

        if not isinstance(ts, (int, float)):
            raise HTTPException(status_code=401, detail="Invalid auth token.")

        age = time.time() - (ts / 1000.0)   # ts is JS Date.now() — milliseconds
        if age > max_age_seconds or age < -5:   # small negative allowance for clock skew
            raise HTTPException(status_code=401, detail="Auth token expired — resync clock and retry.")

    return verify